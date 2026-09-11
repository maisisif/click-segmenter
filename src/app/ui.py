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

from src.app.pages import classes_help, help_page, home
from src.data.clicks import Click
from src.inference.class_predictor import ClassPredictor
from src.inference.predictor import ClickPredictor

POSITIVE_COLOUR = np.array([0, 220, 120])
NEGATIVE_COLOUR = np.array([230, 60, 60])

# One colour per class on the Classes tab, in class order. Chosen to stay
# apart from each other and from the green/red used for clicks.
CLASS_COLOURS = [
    np.array([66, 135, 245]),   # blue
    np.array([255, 170, 0]),    # orange
    np.array([200, 60, 220]),   # purple
    np.array([0, 200, 200]),    # cyan
    np.array([255, 90, 90]),    # salmon
    np.array([160, 220, 60]),   # lime
    np.array([255, 230, 60]),   # yellow
    np.array([120, 120, 255]),  # periwinkle
]

INCLUDE = "Object (include)"
EXCLUDE = "Exclude"

START_MESSAGE = "Upload an image, then click on an object."
CLASSES_START = "Upload an image to see every known class, then click a pixel to ask what is there."
CLASSES_LABEL = "Click a pixel to ask which class is there"

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


def _overlay_classes(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tint each class in its own colour. Where classes overlap (a pillow on a
    bed) the later class paints over the earlier one, which is visible but
    honest: the model really does say both."""
    out = image.astype(np.float32).copy()
    for k, mask in enumerate(masks):
        colour = CLASS_COLOURS[k % len(CLASS_COLOURS)]
        out[mask] = (1 - alpha) * out[mask] + alpha * colour
    return out.astype(np.uint8)


def _describe_classes(class_names: list[str], masks: np.ndarray, probs: np.ndarray) -> str:
    lines = []
    for k, name in enumerate(class_names):
        cover = 100 * masks[k].mean()
        if masks[k].any():
            lines.append(f"{name}: {cover:.1f}% of the image, average confidence {probs[k][masks[k]].mean():.2f}")
        else:
            lines.append(f"{name}: not found (max confidence {probs[k].max():.2f})")
    return "\n".join(lines)


def _mask_png(mask: np.ndarray) -> str:
    """Write the mask as a black-and-white PNG and return the path to download.

    A separate file per call: Gradio serves it from a temporary directory and
    reusing one name would hand a stale mask to a second user of a shared demo.
    """
    handle = tempfile.NamedTemporaryFile(suffix="_mask.png", delete=False)
    Image.fromarray((mask * 255).astype(np.uint8)).save(handle.name)
    return handle.name


def build_ui(
    predictor: ClickPredictor,
    title: str = "Click to segment",
    class_predictor: ClassPredictor | None = None,
) -> gr.Blocks:
    """Assemble the interface around an already-loaded predictor.

    `class_predictor` is optional: when a class-segmentation checkpoint is
    loaded as well, a Classes tab appears (upload -> every class's mask; click
    -> the mask of the class under the click). Without it the interface is the
    click tool alone, exactly as before.
    """

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

    # ------------------------------------------------------------ Classes tab

    def classes_all(original: np.ndarray | None, threshold: float):
        """Every class at once: the view, no download, a per-class summary."""
        if original is None or class_predictor is None:
            return original, None, CLASSES_START
        probs = class_predictor.predict(original)
        masks = probs >= threshold
        return _overlay_classes(original, masks), None, _describe_classes(class_predictor.class_names, masks, probs)

    def classes_on_upload(image: np.ndarray | None, threshold: float):
        view, download, status = classes_all(image, threshold)
        return image, view, download, status

    def classes_on_click(displayed, original, threshold: float, event: gr.SelectData):
        """A click is a segmentation request: which class is here, and where
        else is it. Same pristine-copy rule as the Segment tab."""
        if original is None:
            original = displayed
        if original is None or class_predictor is None:
            return None, None, None, "Upload an image first."
        x, y = event.index
        probs = class_predictor.predict(original)
        k = class_predictor.class_at(original, int(y), int(x), threshold=threshold)
        click = [Click(y=int(y), x=int(x), positive=True)]
        if k is None:
            best = int(probs[:, int(y), int(x)].argmax())
            status = (
                f"No known class at ({x}, {y}). The closest is {class_predictor.class_names[best]} "
                f"at confidence {probs[best, int(y), int(x)]:.2f}, below the threshold {threshold:.2f}."
            )
            return _draw_markers(original, click), original, None, status
        mask = probs[k] >= threshold
        name = class_predictor.class_names[k]
        view = _draw_markers(_overlay_classes(original, np.where(np.arange(len(probs))[:, None, None] == k, mask, False)), click)
        status = (
            f"({x}, {y}) is {name}, confidence {probs[k, int(y), int(x)]:.2f}. "
            f"Its mask covers {100 * mask.mean():.1f}% of the image."
        )
        return view, original, _mask_png(mask), status

    def classes_on_threshold(original, threshold: float):
        return classes_all(original, threshold)

    trained = f"epoch {predictor.trained_epoch}" if predictor.trained_epoch is not None else "n/a"
    if predictor.trained_val_iou is not None:
        trained += f", validation IoU {predictor.trained_val_iou:.4f}"
    footer = (
        f"Checkpoint: {trained}. Architecture `{predictor.arch['arch']}`, "
        f"{predictor.num_masks} candidate mask(s), input {predictor.image_size[0]}x"
        f"{predictor.image_size[1]}. Running on `{predictor.device}`."
    )

    class_names = class_predictor.class_names if class_predictor is not None else None
    if class_predictor is not None:
        parts = [f"Classes: {', '.join(class_names)}."]
        if class_predictor.trained_val_iou is not None:
            parts.append(f"Best validation IoU {class_predictor.trained_val_iou:.4f} (epoch {class_predictor.trained_epoch}).")
        if class_predictor.test:
            per_class = class_predictor.test.get("per_class") or {}
            if per_class:
                parts.append("Test IoU " + ", ".join(f"{n} {c['iou']:.3f}" for n, c in per_class.items()) + ".")
            elif class_predictor.test.get("iou") is not None:
                parts.append(f"Test IoU {class_predictor.test['iou']:.4f}.")
        parts.append(f"Input {class_predictor.image_size[0]}x{class_predictor.image_size[1]}. Running on `{class_predictor.device}`.")
        classes_footer = " ".join(parts)

    with gr.Blocks(title=title) as demo:
        original_state = gr.State(None)
        clicks_state = gr.State([])
        classes_original = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Home"):
                gr.Markdown(home(predictor.num_masks, class_names))

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

            if class_predictor is not None:
                with gr.Tab("Classes"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            classes_image = gr.Image(
                                label=CLASSES_LABEL,
                                type="numpy",
                                height=520,
                                sources=["upload", "clipboard"],
                            )
                        with gr.Column(scale=1):
                            gr.Markdown(
                                "**Legend**  \n"
                                + "  \n".join(
                                    f"<span style='color: rgb({c[0]},{c[1]},{c[2]})'>&#9632;</span> {n}"
                                    for n, c in zip(class_names, CLASS_COLOURS)
                                )
                            )
                            classes_threshold = gr.Slider(
                                0.05, 0.95, value=0.5, step=0.05,
                                label="Class threshold",
                                info="Per-pixel confidence needed to count as the class.",
                            )
                            show_all = gr.Button("Show all classes")
                            classes_status = gr.Textbox(
                                label="Status", value=CLASSES_START, interactive=False, lines=4
                            )
                            classes_download = gr.File(label="Download clicked mask (PNG)", interactive=False)
                    gr.Markdown(classes_footer)

            with gr.Tab("Help"):
                gr.Markdown(help_page(class_names))

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

        if class_predictor is not None:
            classes_image.upload(
                classes_on_upload,
                inputs=[classes_image, classes_threshold],
                outputs=[classes_original, classes_image, classes_download, classes_status],
            )
            classes_image.clear(
                lambda: (None, None, CLASSES_START),
                outputs=[classes_original, classes_download, classes_status],
            )
            classes_image.select(
                classes_on_click,
                inputs=[classes_image, classes_original, classes_threshold],
                outputs=[classes_image, classes_original, classes_download, classes_status],
            )
            classes_threshold.release(
                classes_on_threshold,
                inputs=[classes_original, classes_threshold],
                outputs=[classes_image, classes_download, classes_status],
            )
            show_all.click(
                classes_all,
                inputs=[classes_original, classes_threshold],
                outputs=[classes_image, classes_download, classes_status],
            )

    return demo
