"""Run a trained checkpoint on a real user click.

Kept separate from the GUI so the interface stays a thin shell: the same
predictor can back a Gradio app, a CLI, or the evaluation notebook. It is also
the piece a future scene-graph stage would call.

The important detail is resolution. The model is trained at a fixed size
(128px by default), but a user uploads an image of any size and clicks in that
image's coordinates. So the flow is: remember the original size, downscale the
image, convert the click into model coordinates, predict, then upscale the mask
back so it can be overlaid on what the user actually sees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.clicks import Click
from src.data.encoding import encode_clicks
from src.model.unet import UNet
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

        self.image_size = int(train_config["data"]["image_size"])
        self.device = get_device(device or "auto")

        self.model = UNet(
            in_channels=5,
            out_channels=1,
            base_channels=train_config["model"]["base_channels"],
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

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
        size = (self.image_size, self.image_size)

        resized = np.array(Image.fromarray(image).resize(size, Image.BILINEAR))

        # Scale clicks into model coordinates. Clamp because a click exactly on
        # the right or bottom edge would otherwise round to an out-of-range index.
        scaled = [
            Click(
                y=min(int(c.y * self.image_size / original_h), self.image_size - 1),
                x=min(int(c.x * self.image_size / original_w), self.image_size - 1),
                positive=c.positive,
            )
            for c in clicks
        ]

        encoded = encode_clicks(
            scaled,
            shape=size,
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
        # instead of showing 128px staircase edges.
        probs_full = np.array(
            Image.fromarray(probs).resize((original_w, original_h), Image.BILINEAR)
        )
        return probs_full > threshold, probs_full
