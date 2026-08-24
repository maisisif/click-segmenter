"""Regression tests for the interface's event wiring. Run: python tests/test_app_wiring.py

Two things here are worth a test rather than a manual look.

The first is a bug this file exists because of. The interface paints the mask
and the click markers onto the image the user is looking at. That same component
was previously also the model's image input, so from the second click onward the
model was segmenting a green-tinted photo with rings drawn on it -- silently,
with no error, just quietly worse results the more a user tried to refine. The
fix is a State holding the untouched upload; the test asserts that State still
holds the original photo after a click has annotated the display.

The second is Gradio's calling convention: a listener's `inputs=[...]` list is
positional, so adding a parameter to a handler without adding the component to
its inputs list fails at click time, in the browser, not at import. Calling the
handlers through the listeners Blocks actually registered catches that here.

No trained checkpoint is needed -- a randomly initialised model predicts nonsense
but exercises every code path the real one does.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.ui import EXCLUDE, INCLUDE, build_ui
from src.inference.predictor import ClickPredictor
from src.model.build import build_model
from scripts.export_model import build_payload


def _tiny_checkpoint(directory: Path) -> Path:
    """An exported checkpoint of a small from-scratch UNet.

    From-scratch on purpose: the ResNet variant would download ImageNet weights,
    and this test must run on a machine with no network.
    """
    model = build_model({"arch": "unet", "base_channels": 8, "depth": 2, "num_masks": 3})
    training_checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "best_val_iou": 0.5,
    }
    train_config = {"model": {}, "data": {"image_size": [64, 64]}}
    click_config = {"encoding": "disk", "radius": 5, "max_distance": 64}

    path = directory / "tiny.pt"
    torch.save(build_payload(training_checkpoint, train_config, click_config), path)
    return path


def _photo() -> np.ndarray:
    """A synthetic scene with an actual boundary in it."""
    image = Image.new("RGB", (480, 360), (135, 170, 210))
    ImageDraw.Draw(image).rectangle([60, 140, 220, 320], fill=(200, 90, 70))
    return np.array(image)


def _listeners(demo: gr.Blocks) -> dict[tuple[str, str], object]:
    """Index the registered listeners by (component label or button text, event)."""
    found = {}
    for fn in demo.fns.values():
        targets = getattr(fn, "targets", [])
        if not targets:
            continue
        block_id, event = targets[0]
        block = demo.blocks.get(block_id)
        name = getattr(block, "label", None) or getattr(block, "value", None)
        found.setdefault((name, event), fn)
    return found


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = _tiny_checkpoint(Path(directory))
        predictor = ClickPredictor(checkpoint, device="cpu")

    # The exported checkpoint must have carried its own settings: this
    # constructor never opened configs/train.yaml.
    assert predictor.image_size == (64, 64), predictor.image_size
    assert predictor.arch["base_channels"] == 8, predictor.arch
    print("exported checkpoint loads without configs/                     ok")

    photo = _photo()
    demo = build_ui(predictor)
    listeners = _listeners(demo)

    select = listeners[("Click on an object", "select")]
    assert len(select.inputs) == 5, f"select declares {len(select.inputs)} inputs"

    def click(displayed, original, clicks, x, y, mode=INCLUDE, threshold=0.5):
        event = gr.SelectData(target=None, data={"index": [x, y], "value": None, "selected": True})
        return select.fn(displayed, original, clicks, mode, threshold, event)

    view, original, clicks, _, _ = click(photo, None, [], x=140, y=230)
    assert np.array_equal(original, photo), "pristine copy was not adopted from the display"
    assert len(clicks) == 1 and clicks[0].positive
    assert view.shape == photo.shape
    assert not np.array_equal(view, photo), "nothing was drawn, so the next assertion is vacuous"
    print("first click adopts the displayed image as the original           ok")

    # The crucial one: pass the *annotated* view back in, as the component does.
    _, original, clicks, _, _ = click(view, original, clicks, x=350, y=150, mode=EXCLUDE)
    assert np.array_equal(original, photo), "the annotated view overwrote the pristine copy"
    assert len(clicks) == 2 and not clicks[1].positive
    print("second click still predicts from the untouched photo             ok")

    buttons = {name: fn for (name, event), fn in listeners.items() if event == "click"}
    undo = buttons["Undo last click"]
    _, remaining, _, _ = undo.fn(photo, clicks, 0.5)
    assert len(remaining) == 1, f"undo left {len(remaining)} clicks"
    restored, cleared, _, _ = buttons["Clear clicks"].fn(photo)
    assert cleared == [] and np.array_equal(restored, photo)
    print("undo removes one click, clear restores the original photo        ok")

    threshold = listeners[("Mask threshold", "release")]
    view_low, _, _ = threshold.fn(photo, clicks, 0.05)
    view_high, _, _ = threshold.fn(photo, clicks, 0.95)
    assert view_low.shape == photo.shape and view_high.shape == photo.shape
    print("threshold re-renders without needing another click               ok")

    print("\nall wiring tests passed")


if __name__ == "__main__":
    main()
