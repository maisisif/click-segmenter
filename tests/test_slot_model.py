"""Checks for the slot architecture. Run: python tests/test_slot_model.py

The overfit check here is the one that matters, and it is deliberately not a
loss check. A set-prediction model with broken matching still drives its loss
down -- it just converges on every slot predicting the same blurry average of
every object. Loss alone cannot tell the two apart.

So the test measures per-object IoU *after* assignment. That only gets high if
distinct slots have specialised on distinct objects, which is precisely what
Hungarian matching is supposed to produce and what a bug in it would destroy.

Needs no checkpoint and no network: the backbone is built with pretrained=False.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clicks import Click
from src.inference.slot_predictor import SlotPrediction
from src.model.slot_unet import SlotUNet
from src.training.losses import HungarianMaskLoss

SIZE = 128
NUM_SLOTS = 8


def _scene() -> list[torch.Tensor]:
    """Three disjoint rectangles -- objects a slot can plausibly specialise on."""
    masks = torch.zeros(3, SIZE, SIZE)
    masks[0, 10:50, 10:50] = 1
    masks[1, 60:110, 20:60] = 1
    masks[2, 20:60, 70:120] = 1
    return masks


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = (a * b).sum()
    union = ((a + b) > 0).float().sum().clamp(min=1)
    return float(intersection / union)


def test_shapes() -> None:
    for stride in (2, 4):
        model = SlotUNet(num_slots=64, pretrained=False, mask_stride=stride)
        with torch.no_grad():
            masks, objectness = model(torch.randn(2, 3, 384, 512))
        assert masks.shape == (2, 64, 384 // stride, 512 // stride), masks.shape
        assert objectness.shape == (2, 64), objectness.shape
    print("shapes correct at both mask strides                          ok")


def test_empty_image_is_survivable() -> None:
    """An image with no annotated objects must not crash or produce NaN.

    ADE20K has images where every instance is dropped as empty at the training
    resolution, so this happens in a real epoch, not just in theory.
    """
    model = SlotUNet(num_slots=NUM_SLOTS, pretrained=False, decoder_channels=(64, 32, 32, 32))
    logits, objectness = model(torch.rand(2, 3, SIZE, SIZE))
    loss = HungarianMaskLoss()(logits, objectness, [torch.zeros(0, SIZE, SIZE)] * 2)
    loss.backward()
    assert torch.isfinite(loss), loss
    print("an image with zero objects yields a finite loss              ok")


def test_overfits_distinct_objects() -> None:
    torch.manual_seed(0)
    model = SlotUNet(num_slots=NUM_SLOTS, pretrained=False, decoder_channels=(64, 32, 32, 32))
    loss_fn = HungarianMaskLoss()

    images = torch.rand(2, 3, SIZE, SIZE)
    targets = [_scene(), _scene()]

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        logits, objectness = model(images)
        loss = loss_fn(logits, objectness, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, objectness = model(images)
        upsampled = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        predicted = (torch.sigmoid(upsampled) > 0.5).float()

    for index, target in enumerate(targets):
        cost = torch.tensor(
            [[-_iou(predicted[index, n], target[k]) for k in range(len(target))]
             for n in range(NUM_SLOTS)]
        )
        rows, cols = linear_sum_assignment(cost.numpy())
        ious = [-cost[r, c].item() for r, c in zip(rows, cols)]
        assert min(ious) > 0.9, f"image {index} matched IoUs {ious}"

        # Each object must land in its own slot, or the model has collapsed
        # onto one average mask -- the exact failure a loss check would miss.
        assert len(set(rows.tolist())) == len(target)

        active = int((torch.sigmoid(objectness[index]) > 0.5).sum())
        assert active == len(target), f"objectness fired on {active} slots, expected {len(target)}"

    print("three objects memorised into three separate slots, IoU > 0.9  ok")


def _nested_scene() -> SlotPrediction:
    """Five slots with known nesting, so selection can be checked exactly."""
    size = 64
    probs = torch.zeros(5, size, size)
    probs[0, 10:50, 10:50] = 0.9   # person
    probs[1, 15:35, 15:35] = 0.9   # torso, inside the person
    probs[2, 18:28, 18:28] = 0.9   # shirt, inside the torso
    probs[3, 55:60, 55:60] = 0.9   # an unrelated object elsewhere
    probs[4, 0:64, 0:64] = 0.9     # a slot objectness rejects
    objectness = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.1])
    return SlotPrediction(probs, objectness, original_size=(size, size))


def test_click_selection() -> None:
    state = _nested_scene()
    assert state.num_objects == 4, "objectness should reject the fifth slot"

    inside = Click(y=22, x=22, positive=True)

    # Nested objects all contain the click; smallest is shown, rest are offered.
    _, ranked = state.select([inside])
    assert ranked == [2, 1, 0], ranked

    # A negative inside the torso but outside the shirt leaves only the shirt.
    _, ranked = state.select([inside, Click(y=32, x=32, positive=False)])
    assert ranked == [2], ranked

    # A negative inside the person but outside the torso leaves both smaller ones.
    _, ranked = state.select([inside, Click(y=45, x=45, positive=False)])
    assert ranked == [2, 1], ranked

    # Positive and negative on the same object is contradictory: return nothing
    # rather than guessing which of the two the user meant.
    mask, ranked = state.select([inside, Click(y=20, x=20, positive=False)])
    assert ranked == [] and not mask.any()

    # Only the person contains both clicks.
    _, ranked = state.select([Click(y=12, x=12, positive=True), inside])
    assert ranked == [0], ranked
    print("nested objects, negatives and contradictions select correctly  ok")


def test_clicks_are_in_original_image_coordinates() -> None:
    """The mask grid is not the photo. A click at (88, 88) on a 256x256 image is
    the same point as (22, 22) on the 64x64 grid, and must select the same slot."""
    state = _nested_scene()
    state.original_size = (256, 256)
    mask, ranked = state.select([Click(y=88, x=88, positive=True)])
    assert ranked == [2, 1, 0], ranked
    assert mask.shape == (256, 256), mask.shape
    print("clicks map from original image pixels onto the mask grid      ok")


if __name__ == "__main__":
    test_shapes()
    test_empty_image_is_survivable()
    test_overfits_distinct_objects()
    test_click_selection()
    test_clicks_are_in_original_image_coordinates()
    print("\nall slot model tests passed")
