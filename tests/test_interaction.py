"""Regression tests for multi-click interaction. Run: python tests/test_interaction.py

Covers the pieces that scripts/evaluate.py and iterative training in
scripts/train_full.py depend on, none of which can be checked by looking at a
loss curve: where the simulated next click lands, that it is drawn into the
right channel, that NoC is counted the way the literature counts it, that a
five-channel checkpoint widens to six and still loads, and that the predictor
replays clicks in order for a previous-mask model.

No trained checkpoint, no dataset, no network: every model here is a throwaway
from-scratch UNet, so this runs anywhere in a few seconds.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_model import build_payload
from src.data.clicks import Click
from src.inference.predictor import ClickPredictor
from src.model.build import build_model, detect_arch, expand_input_channels
from src.training.interaction import (
    NEGATIVE_CHANNEL,
    POSITIVE_CHANNEL,
    PREVIOUS_MASK_CHANNEL,
    add_training_clicks,
    clicks_to_reach,
    next_click,
    next_clicks,
    run_interaction,
    select_consistent,
    stamp_clicks,
    with_previous_mask_channel,
)


def _square(height: int, width: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_next_click() -> None:
    target = _square(64, 64, 16, 48, 16, 48)

    # Nothing predicted: the click is positive, at the centre of the object.
    click = next_click(np.zeros_like(target), target)
    assert click is not None and click[2] is True
    assert abs(click[0] - 31.5) <= 1 and abs(click[1] - 31.5) <= 1, click

    # Perfect prediction: no click.
    assert next_click(target.copy(), target) is None

    # Over-segmentation: the mask spills to the right, so the click is negative
    # and lands inside the spill, outside the object.
    spill = target | _square(64, 64, 16, 48, 48, 64)
    click = next_click(spill, target)
    assert click is not None and click[2] is False
    assert not target[click[0], click[1]] and spill[click[0], click[1]], click

    # Batched with a stride: same polarity, coordinates back at full res.
    pred = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
    tgt = torch.from_numpy(target)[None, None].repeat(2, 1, 1, 1)
    pred[1] = tgt[1]
    batch = next_clicks(pred, tgt, stride=4)
    assert batch[1] is None
    assert batch[0] is not None and batch[0][2] is True
    assert 24 <= batch[0][0] <= 40 and 24 <= batch[0][1] <= 40, batch[0]
    print("next click lands on the largest error region with the right sign  ok")


def test_stamp_and_previous_mask() -> None:
    inputs = torch.zeros(2, 5, 32, 32)
    stamp_clicks(inputs, [(10, 10, True), (20, 20, False)], radius=3)
    assert inputs[0, POSITIVE_CHANNEL, 10, 10] == 1 and inputs[0, NEGATIVE_CHANNEL].sum() == 0
    assert inputs[1, NEGATIVE_CHANNEL, 20, 20] == 1 and inputs[1, POSITIVE_CHANNEL].sum() == 0
    assert 25 <= inputs[0, POSITIVE_CHANNEL].sum() <= 30, inputs[0, POSITIVE_CHANNEL].sum()  # disk r=3

    # Clamped, not crashed, at the border.
    stamp_clicks(inputs, [None, (-5, 40, True)], radius=3)
    assert inputs[1, POSITIVE_CHANNEL, 0, 31] == 1

    six = with_previous_mask_channel(inputs, 6)
    assert six.shape[1] == 6 and six[:, PREVIOUS_MASK_CHANNEL].sum() == 0
    assert with_previous_mask_channel(inputs, 5) is inputs
    print("clicks are drawn into the right channel, previous mask appended     ok")


def test_clicks_to_reach() -> None:
    iou = torch.tensor([[0.5, 0.9, 0.95], [0.2, 0.3, 0.4], [0.86, 0.1, 0.1]])
    noc = clicks_to_reach(iou, 0.85)
    assert noc.tolist() == [2, 3, 1], noc.tolist()
    assert clicks_to_reach(iou, 0.85, max_clicks=20).tolist() == [2, 20, 1]
    print("NoC counts the first click that reaches the target, cap otherwise   ok")


def test_select_consistent() -> None:
    # Two candidates: 0 is a box that misses the click, 1 contains it. The
    # score head prefers 0; consistency must overrule it.
    logits = torch.full((1, 2, 32, 32), -5.0)
    logits[0, 0, 0:8, 0:8] = 5.0
    logits[0, 1, 12:20, 12:20] = 5.0
    scores = torch.tensor([[0.9, 0.1]])
    inputs = torch.zeros(1, 5, 32, 32)
    stamp_clicks(inputs, [(16, 16, True)], radius=2)
    chosen = select_consistent(logits, scores, inputs)
    assert chosen[0, 0, 16, 16] > 0, "the candidate containing the click was not chosen"

    # No candidate contains it: fall back to the score head.
    inputs = torch.zeros(1, 5, 32, 32)
    stamp_clicks(inputs, [(28, 28, True)], radius=2)
    chosen = select_consistent(logits, scores, inputs)
    assert chosen[0, 0, 4, 4] > 0, "fallback to score selection failed"
    print("consistent selection drops candidates that contradict the clicks    ok")


def test_widen_checkpoint_and_detect() -> None:
    five = build_model({"arch": "unet", "base_channels": 8, "depth": 2, "num_masks": 3})
    state = five.state_dict()
    assert detect_arch(state)["in_channels"] == 5

    widened = expand_input_channels(state, 6)
    arch = detect_arch(widened)
    assert arch["in_channels"] == 6 and arch["base_channels"] == 8 and arch["depth"] == 2
    six = build_model(arch)
    six.load_state_dict(widened)

    # Zero filters for the new channel: with it empty, both models agree exactly.
    x = torch.rand(2, 5, 32, 32)
    five.eval(), six.eval()
    with torch.no_grad():
        a, _ = five(x)
        b, _ = six(with_previous_mask_channel(x, 6))
    assert torch.allclose(a, b, atol=1e-6), "widened model does not reproduce the original"
    assert expand_input_channels(state, 5) is state
    print("five-channel weights widen to six and reproduce the original         ok")


def test_run_interaction_and_training_clicks() -> None:
    model = build_model({"arch": "unet", "base_channels": 8, "depth": 2, "num_masks": 3, "in_channels": 6})
    inputs = torch.rand(3, 5, 32, 32)
    inputs[:, 3:] = 0
    stamp_clicks(inputs, [(16, 16, True)] * 3, radius=2)
    targets = torch.zeros(3, 1, 32, 32)
    targets[:, :, 8:24, 8:24] = 1

    result = run_interaction(model, inputs, targets, 4, in_channels=6, radius=2)
    assert result["iou"].shape == (3, 4) and result["oracle"].shape == (3, 4)
    assert (result["oracle"] >= result["iou"] - 1e-6).all(), "oracle below the selected IoU"
    assert result["final_inputs"].shape[1] == 6
    # Three corrective clicks were added on top of the initial one.
    assert result["final_inputs"][:, 3:5].sum() > inputs[:, 3:5].sum()
    # The original batch was not modified.
    assert inputs.shape[1] == 5

    model.train()
    trained_on = add_training_clicks(model, inputs, targets, 2, in_channels=6, radius=2, click_stride=2, prev_mask_drop=1.0)
    assert model.training, "add_training_clicks must restore train mode"
    assert trained_on.shape == (3, 6, 32, 32)
    assert trained_on[:, PREVIOUS_MASK_CHANNEL].sum() == 0, "prev_mask_drop=1.0 must zero the channel"
    trained_on = add_training_clicks(model, inputs, targets, 2, in_channels=6, radius=2, click_stride=2, prev_mask_drop=0.0)
    assert trained_on[:, PREVIOUS_MASK_CHANNEL].sum() > 0, "previous mask was not filled in"
    print("interaction loop and iterative training clicks run end to end       ok")


def test_predictor_replays_previous_mask() -> None:
    model = build_model({"arch": "unet", "base_channels": 8, "depth": 2, "num_masks": 3, "in_channels": 6})
    checkpoint = {"epoch": 1, "model_state_dict": model.state_dict(), "best_val_iou": 0.5}
    train_config = {"model": {}, "data": {"image_size": [64, 64]}}
    click_config = {"encoding": "disk", "radius": 5, "max_distance": 64}
    payload = build_payload(checkpoint, train_config, click_config)
    assert payload["inference_config"]["input_channels"][-1] == "previous_mask"

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "six.pt"
        torch.save(payload, path)
        predictor = ClickPredictor(path, device="cpu")

    assert predictor.uses_previous_mask and predictor.in_channels == 6

    forwards = {"n": 0}
    original = predictor._forward

    def counting(*args, **kwargs):
        forwards["n"] += 1
        return original(*args, **kwargs)

    predictor._forward = counting

    image = (np.random.default_rng(0).random((90, 120, 3)) * 255).astype(np.uint8)
    clicks = [Click(y=40, x=60, positive=True), Click(y=10, x=10, positive=False)]
    mask, probs = predictor.predict(image, clicks)
    assert mask.shape == (90, 120) and probs.shape == (90, 120)
    assert forwards["n"] == 2, f"two clicks should be two passes, got {forwards['n']}"

    # Adding a third click costs one more pass; undoing it costs none.
    predictor.predict(image, clicks + [Click(y=70, x=100, positive=True)])
    assert forwards["n"] == 3, forwards["n"]
    predictor.predict(image, clicks)
    assert forwards["n"] == 3, "cached prefixes were recomputed"
    print("predictor replays clicks in order and caches each prefix             ok")


def main() -> None:
    torch.manual_seed(0)
    test_next_click()
    test_stamp_and_previous_mask()
    test_clicks_to_reach()
    test_select_consistent()
    test_widen_checkpoint_and_detect()
    test_run_interaction_and_training_clicks()
    test_predictor_replays_previous_mask()
    print("\nall interaction tests passed")


if __name__ == "__main__":
    main()
