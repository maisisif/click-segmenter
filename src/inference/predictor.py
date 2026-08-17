"""Run a trained checkpoint on a real user click.

Kept separate from the GUI so the interface stays a thin shell: the same
predictor can back a Gradio app, a CLI, or the evaluation notebook. It is also
the piece a future scene-graph stage would call.

The important detail is resolution. The model is trained at a fixed size
(configs/train.yaml `data.image_size`), but a user uploads an image of any size
and clicks in that image's coordinates. So the flow is: remember the original
size, downscale the image, convert the click into model coordinates, predict,
then upscale the mask back so it can be overlaid on what the user actually sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.clicks import Click
from src.data.dataset import _normalize_size
from src.data.encoding import encode_clicks
from src.model.unet import UNet, migrate_legacy_state_dict
from src.training.device import get_device


class ClickPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        train_config_path: str | Path = "configs/train.yaml",
        clicks_config_path: str | Path = "configs/clicks.yaml",
        device: str | None = None,
    ) -> None:
        with open(train_config_path) as f:
            train_config = yaml.safe_load(f)
        with open(clicks_config_path) as f:
            self.click_config = yaml.safe_load(f)["clicks"]

        self.image_size = _normalize_size(train_config["data"]["image_size"])  # (H, W)
        self.device = get_device(device or "auto")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])

        # Old checkpoints predate the depth option and are always depth 3;
        # detect it from the weights rather than trusting the current config,
        # so a config bumped to depth 4 can still load the released depth-3 model.
        checkpoint_depth = 1 + max(
            int(k.split(".")[1]) for k in state_dict if k.startswith("downs.")
        )

        self.model = UNet(
            in_channels=5,
            out_channels=1,
            base_channels=train_config["model"]["base_channels"],
            depth=checkpoint_depth,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # A depth-d model needs input divisible by 2**d; pad the configured
        # size up if needed rather than failing.
        divisor = 2**checkpoint_depth
        self.image_size = tuple(-(-s // divisor) * divisor for s in self.image_size)

        self.trained_epoch = checkpoint.get("epoch")
        self.trained_val_iou = checkpoint.get("best_val_iou")

    def predict(
        self,
        image: np.ndarray,
        clicks: list[Click],
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment the object indicated by `clicks` in a full-resolution image.

        `clicks` are in the *original* image's pixel coordinates. Returns
        (binary mask, probability map), both at the original image size.
        """
        if not clicks:
            raise ValueError("at least one click is required")

        original_h, original_w = image.shape[:2]
        model_h, model_w = self.image_size

        resized = np.array(Image.fromarray(image).resize((model_w, model_h), Image.BILINEAR))

        # Scale clicks into model coordinates. Clamp because a click exactly on
        # the right or bottom edge would otherwise round to an out-of-range index.
        scaled = [
            Click(
                y=min(int(c.y * model_h / original_h), model_h - 1),
                x=min(int(c.x * model_w / original_w), model_w - 1),
                positive=c.positive,
            )
            for c in clicks
        ]

        encoded = encode_clicks(
            scaled,
            shape=(model_h, model_w),
            radius=self.click_config.get("radius", 5),
            encoding=self.click_config.get("encoding", "disk"),
            max_distance=self.click_config.get("max_distance", 64.0),
        )

        image_chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        model_input = torch.from_numpy(np.concatenate([image_chw, encoded], axis=0)).float()

        with torch.no_grad():
            logits = self.model(model_input.unsqueeze(0).to(self.device))
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Back to the user's resolution. The probability map is resized rather
        # than the thresholded mask, so the boundary is interpolated smoothly
        # instead of showing model-resolution staircase edges.
        probs_full = np.array(
            Image.fromarray(probs).resize((original_w, original_h), Image.BILINEAR)
        )
        return probs_full > threshold, probs_full
