"""M6: interactive click-to-segment interface.

Upload an image, click an object, get its mask. Left click marks the object;
right-click-equivalent (the "negative" mode) marks something to exclude, which
is how you correct a mask that has bled into a neighbouring object.

Run from the repo root:
    python scripts/app.py                       # local only
    python scripts/app.py --share               # temporary public link
    python scripts/app.py --checkpoint results/run-10000-images/best.pt

On MetaCentrum, run it inside an interactive GPU job (or a Jupyter session's
terminal) and use --share, since compute nodes are not directly reachable.

Deliberately thin: all the real work is in src/inference/predictor.py, so the
interface can be swapped without touching the model logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clicks import Click
from src.inference.predictor import ClickPredictor

PREDICTOR: ClickPredictor | None = None

POSITIVE_COLOUR = np.array([0, 220, 120])
NEGATIVE_COLOUR = np.array([230, 60, 60])


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Tint the masked region so the boundary is visible against the photo."""
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * POSITIVE_COLOUR
    return out.astype(np.uint8)


def _draw_markers(image: np.ndarray, clicks: list[Click], radius: int = 6) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for click in clicks:
        ring = np.abs(np.sqrt((yy - click.y) ** 2 + (xx - click.x) ** 2) - radius) < 1.8
        out[ring] = POSITIVE_COLOUR if click.positive else NEGATIVE_COLOUR
    return out


def on_click(
    image: np.ndarray | None,
    clicks: list,
    mode: str,
    threshold: float,
    event: gr.SelectData,
):
    """Handle a click on the image: add it, re-predict, redraw."""
    if image is None:
        return None, clicks, "Upload an image first."

    x, y = event.index  # gradio gives (x, y)
    clicks = clicks + [Click(y=int(y), x=int(x), positive=(mode == "Object (include)"))]

    assert PREDICTOR is not None
    mask, probs = PREDICTOR.predict(image, clicks, threshold=threshold)

    view = _draw_markers(_overlay(image, mask), clicks)

    coverage = 100 * mask.mean()
    n_pos = sum(c.positive for c in clicks)
    n_neg = len(clicks) - n_pos
    status = (
        f"{n_pos} object click(s), {n_neg} exclude click(s). "
        f"Mask covers {coverage:.1f}% of the image. "
        f"Mean confidence inside mask: {probs[mask].mean():.2f}"
        if mask.any()
        else "No region passed the threshold. Try lowering it, or click nearer the object's centre."
    )
    return view, clicks, status


def on_reset(image: np.ndarray | None):
    return image, [], "Cleared. Click an object to segment it."


def build_ui() -> gr.Blocks:
    assert PREDICTOR is not None
    trained = (
        f"checkpoint from epoch {PREDICTOR.trained_epoch}"
        + (
            f", validation IoU {PREDICTOR.trained_val_iou:.4f}"
            if PREDICTOR.trained_val_iou is not None
            else ""
        )
    )

    with gr.Blocks(title="Click to segment") as demo:
        gr.Markdown(
            "# Click to segment\n"
            "Upload an image and click an object to get its mask. Add more clicks to refine: "
            "**Object** clicks pull the mask towards a region, **Exclude** clicks push it away.\n\n"
            f"UNet trained from scratch on ADE20K, {trained}. Running on `{PREDICTOR.device}`."
        )

        clicks_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                image_in = gr.Image(label="Click on an object", type="numpy", height=480)
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["Object (include)", "Exclude"],
                    value="Object (include)",
                    label="Click type",
                )
                threshold = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.05,
                    label="Mask threshold",
                    info="Lower includes more pixels, higher is stricter.",
                )
                reset = gr.Button("Clear clicks")
                status = gr.Textbox(label="Status", interactive=False, lines=3)

        image_in.select(
            on_click,
            inputs=[image_in, clicks_state, mode, threshold],
            outputs=[image_in, clicks_state, status],
        )
        reset.click(on_reset, inputs=[image_in], outputs=[image_in, clicks_state, status])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    global PREDICTOR
    PREDICTOR = ClickPredictor(
        args.checkpoint,
        train_config_path=args.train_config,
        clicks_config_path=args.clicks_config,
        device=args.device,
    )
    print(f"Loaded {args.checkpoint} on {PREDICTOR.device}")

    build_ui().launch(share=args.share, server_port=args.port, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
