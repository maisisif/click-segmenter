"""The Gradio interface: Home, Segment and Help pages over one ClickPredictor.

This module builds the UI and nothing else. The two entry points that launch it
-- scripts/app.py locally and deploy/huggingface/app.py on Hugging Face Spaces
-- differ only in where they get the checkpoint from, so the interface a user
sees is the same in both places by construction rather than by discipline.

One structural detail worth stating, because getting it wrong is invisible: the
image the user clicks on is *not* the image the model receives. The displayed
image is progressively painted with the mask tint and the click markers, so
feeding it back in would have the model segmenting its own annotations from
click two onward. The untouched upload is held in a State and is what every
prediction actually runs on.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.app.pages import HELP, HOME
from src.data.clicks import Click
from src.inference.predictor import ClickPredictor

POSITIVE_COLOUR = np.array([0, 220, 120])
NEGATIVE_COLOUR = np.array([230, 60, 60])

INCLUDE = "Object (include)"
EXCLUDE = "Exclude"

START_MESSAGE = "Upload an image, then click on an object."

# Passed to Blocks.launch() by the entry points rather than set here: Gradio 6
# moved theme off the Blocks constructor, and keeping it in one place means the
# local app and the hosted Space cannot drift apart visually.
THEME = gr.themes.Soft()


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tint the masked region so the boundary is visible against the photo."""
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * POSITIVE_COLOUR
    return out.astype(np.uint8)


def _draw_markers(image: np.ndarray, clicks: list[Click], radius: int = 6) -> np.ndarray:
    """Draw a ring at each click. Rings, not filled dots, so the pixel the user
    actually clicked stays visible underneath."""
    out = image.copy()
    height, width = out.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    for click in clicks:
        ring = np.abs(np.sqrt((yy - click.y) ** 2 + (xx - click.x) ** 2) - radius) < 1.8
        out[ring] = POSITIVE_COLOUR if click.positive else NEGATIVE_COLOUR
    return out


def _describe(clicks: list[Click], mask: np.ndarray, probs: np.ndarray) -> str:
    n_positive = sum(click.positive for click in clicks)
    n_negative = len(clicks) - n_positive
    counts = f"{n_positive} object click(s), {n_negative} exclude click(s)."

    if not mask.any():
        return (
            f"{counts} Nothing passed the threshold. Lower the threshold, or click "
            "closer to the middle of the object."
        )
    return (
        f"{counts} The mask covers {100 * mask.mean():.1f}% of the image, "
        f"average confidence {probs[mask].mean():.2f}."
    )


def _mask_png(mask: np.ndarray) -> str:
    """Write the mask as a black-and-white PNG and return the path to download.

    A separate file per call: Gradio serves it from a temporary directory and
    reusing one name would hand a stale mask to a second user of a shared demo.
    """
    handle = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
    Image.fromarray((mask * 255).astype(np.uint8)).save(handle.name)
    return handle.name


def build_ui(predictor: ClickPredictor, title: str = "Click to segment") -> gr.Blocks:
    """Assemble the interface around an already-loaded predictor."""

    def run(original: np.ndarray | None, clicks: list[Click], threshold: float):
        """Predict from the pristine image and render the annotated view."""
        if original is None or not clicks:
            return original, None, START_MESSAGE

        mask, probs = predictor.predict(original, clicks, threshold=threshold)
        view = _draw_markers(_overlay(original, mask), clicks)
        download = _mask_png(mask) if mask.any() else None
        return view, download, _describe(clicks, mask, probs)

    def on_upload(image: np.ndarray | None):
        # Keep the untouched upload; every later prediction runs on this copy.
        return image, [], None, "Click on an object to segment it."

    def on_click(
        displayed: np.ndarray | None,
        original: np.ndarray | None,
        clicks: list,
        mode: str,
        threshold: float,
        event: gr.SelectData,
    ):
        if original is None:
            # No pristine copy yet, because the image arrived by some route
            # other than the upload button -- a paste, an example, a value set
            # in code. What is on screen has not been tinted yet (nothing has
            # been predicted), so adopting it now is safe and keeps every later
            # click running against the real photo.
            original, clicks = displayed, []
        if original is None:
            return None, None, clicks, None, "Upload an image first."

        x, y = event.index  # Gradio reports (x, y); Click stores (y, x).
        clicks = clicks + [Click(y=int(y), x=int(x), positive=(mode == INCLUDE))]
        view, download, status = run(original, clicks, threshold)
        return view, original, clicks, download, status

    def on_threshold(original, clicks, threshold):
        """Re-render at the new cut-off without re-clicking. The model runs
        again, but only the threshold applied to its output has changed."""
        view, download, status = run(original, clicks, threshold)
        return view, download, status

    def on_undo(original, clicks, threshold):
        if not clicks:
            return original, [], None, "No clicks to undo."
        clicks = clicks[:-1]
        if not clicks:
            return original, [], None, "All clicks removed. Click an object to start again."
        view, download, status = run(original, clicks, threshold)
        return view, clicks, download, status

    def on_reset(original):
        return original, [], None, "Cleared. Click an object to segment it."

    trained = f"epoch {predictor.trained_epoch}" if predictor.trained_epoch is not None else "n/a"
    if predictor.trained_val_iou is not None:
        trained += f", validation IoU {predictor.trained_val_iou:.4f}"
    footer = (
        f"Checkpoint: {trained}. Architecture `{predictor.arch['arch']}`, "
        f"{predictor.num_masks} candidate mask(s), input {predictor.image_size[0]}x"
        f"{predictor.image_size[1]}. Running on `{predictor.device}`."
    )

    with gr.Blocks(title=title) as demo:
        original_state = gr.State(None)
        clicks_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Home"):
                gr.Markdown(HOME)

            with gr.Tab("Segment"):
                with gr.Row():
                    with gr.Column(scale=3):
                        image_in = gr.Image(
                            label="Click on an object",
                            type="numpy",
                            height=520,
                            sources=["upload", "clipboard"],
                        )
                    with gr.Column(scale=1):
                        mode = gr.Radio(
                            [INCLUDE, EXCLUDE],
                            value=INCLUDE,
                            label="Click type",
                            info="Include pulls the mask towards a region, Exclude pushes it away.",
                        )
                        threshold = gr.Slider(
                            0.05, 0.95, value=0.5, step=0.05,
                            label="Mask threshold",
                            info="Lower includes more pixels, higher is stricter.",
                        )
                        with gr.Row():
                            undo = gr.Button("Undo last click")
                            reset = gr.Button("Clear clicks")
                        status = gr.Textbox(
                            label="Status", value=START_MESSAGE, interactive=False, lines=3
                        )
                        download = gr.File(label="Download mask (PNG)", interactive=False)
                gr.Markdown(footer)

            with gr.Tab("Help"):
                gr.Markdown(HELP)

        # `upload` rather than `change`: `change` also fires when a handler
        # writes the annotated view back into the component, which would
        # overwrite the pristine copy with an already-tinted image.
        image_in.upload(
            on_upload,
            inputs=[image_in],
            outputs=[original_state, clicks_state, download, status],
        )
        image_in.clear(
            lambda: (None, [], None, START_MESSAGE),
            outputs=[original_state, clicks_state, download, status],
        )
        image_in.select(
            on_click,
            inputs=[image_in, original_state, clicks_state, mode, threshold],
            outputs=[image_in, original_state, clicks_state, download, status],
        )
        threshold.release(
            on_threshold,
            inputs=[original_state, clicks_state, threshold],
            outputs=[image_in, download, status],
        )
        undo.click(
            on_undo,
            inputs=[original_state, clicks_state, threshold],
            outputs=[image_in, clicks_state, download, status],
        )
        reset.click(
            on_reset,
            inputs=[original_state],
            outputs=[image_in, clicks_state, download, status],
        )

    return demo
