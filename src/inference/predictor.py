"""Run a trained checkpoint on a real user click.

Kept separate from the GUI so the interface stays a thin shell: the same
predictor can back a Gradio app, a CLI, or the evaluation notebook. It is also
the piece a future scene-graph stage would call.

The important detail is resolution. The model is trained at a fixed size
(configs/train.yaml `data.image_size`), but a user uploads an image of any size
and clicks in that image's coordinates. So the flow is: remember the original
size, downscale the image, convert the click into model coordinates, predict,
then upscale the mask back so it can be overlaid on what the user actually sees.

Two kinds of checkpoint load here. A **training** checkpoint holds only weights,
so the settings that describe how to feed it -- resolution, click encoding --
come from configs/train.yaml and configs/clicks.yaml alongside it. A
**deployment** checkpoint written by scripts/export_model.py carries those
settings inside itself, so it loads with no repo checkout at all; that is what
the hosted app pulls from the Hugging Face model repo. Everything about the
architecture is read from the weight shapes in both cases.

Models with a **previous-mask channel** (six input channels) are stateful by
design: the mask after click k depends on the mask after click k-1. The
interface is stateless -- it hands over the full click list every time -- so the
predictor replays the clicks in order, feeding each step the probability map
from the one before. Intermediate maps are cached per (image, click prefix), so
adding one click costs one forward pass, and undo costs none.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.clicks import Click
from src.data.dataset import _normalize_size
from src.data.encoding import encode_clicks
from src.model.build import build_model, detect_arch
from src.model.unet import migrate_legacy_state_dict
from src.training.device import get_device
from src.training.interaction import PREVIOUS_MASK_CHANNEL, predict_probs


class ClickPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        train_config_path: str | Path = "configs/train.yaml",
        clicks_config_path: str | Path = "configs/clicks.yaml",
        device: str | None = None,
        selection: str = "score",
        flip_tta: bool = False,
        cache_size: int = 256,
    ) -> None:
        self.device = get_device(device or "auto")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # An exported checkpoint describes itself; a training checkpoint needs
        # the configs it was trained with. Reading the checkpoint first means
        # the config files are opened only when they are actually needed, so a
        # deployment never has to ship them.
        inference_config = checkpoint.get("inference_config")
        if inference_config is None:
            recorded = checkpoint.get("train_settings")
            if recorded is not None:
                inference_config = {"image_size": recorded["image_size"], "clicks": recorded["clicks"]}
            else:
                with open(train_config_path) as f:
                    train_config = yaml.safe_load(f)
                with open(clicks_config_path) as f:
                    click_config = yaml.safe_load(f)["clicks"]
                inference_config = {
                    "image_size": train_config["data"]["image_size"],
                    "clicks": click_config,
                }

        self.click_config = inference_config["clicks"]
        self.image_size = _normalize_size(inference_config["image_size"])  # (H, W)

        state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])

        # Build whatever architecture the checkpoint was actually trained with,
        # detected from its weights rather than trusted from the current config
        # -- so the app loads any era of checkpoint without config surgery.
        arch_config = detect_arch(state_dict)
        self.model = build_model(arch_config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Pad the working size up to what the architecture divides by.
        divisor = 32 if arch_config["arch"] == "resnet34_unet" else 2 ** arch_config["depth"]
        self.image_size = tuple(-(-s // divisor) * divisor for s in self.image_size)

        self.arch = arch_config
        self.num_masks = arch_config["num_masks"]
        self.in_channels = arch_config.get("in_channels", 5)
        self.uses_previous_mask = self.in_channels == PREVIOUS_MASK_CHANNEL + 1
        self.selection = selection
        self.flip_tta = flip_tta
        provenance = checkpoint.get("provenance", checkpoint)
        self.trained_epoch = provenance.get("epoch")
        self.trained_val_iou = provenance.get("best_val_iou")

        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ core

    def _forward(
        self,
        image_chw: np.ndarray,
        clicks: list[Click],
        previous: np.ndarray | None,
        threshold: float,
    ) -> np.ndarray:
        """One model pass at working resolution -> probability map (H, W)."""
        model_h, model_w = self.image_size
        encoded = encode_clicks(
            clicks,
            shape=(model_h, model_w),
            radius=self.click_config.get("radius", 5),
            encoding=self.click_config.get("encoding", "disk"),
            max_distance=self.click_config.get("max_distance", 64.0),
        )
        channels = [image_chw, encoded]
        if self.uses_previous_mask:
            if previous is None:
                previous = np.zeros((model_h, model_w), dtype=np.float32)
            channels.append(previous[None].astype(np.float32))

        model_input = torch.from_numpy(np.concatenate(channels, axis=0)).float().unsqueeze(0)
        probs, _, _ = predict_probs(
            self.model,
            model_input.to(self.device),
            selection=self.selection,
            flip_tta=self.flip_tta,
            threshold=threshold,
        )
        return probs[0, 0].cpu().numpy()

    def _remember(self, key: tuple, probs: np.ndarray) -> None:
        self._cache[key] = probs
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _replay(
        self, image_key: str, image_chw: np.ndarray, clicks: list[Click], threshold: float
    ) -> np.ndarray:
        """Feed the clicks one at a time, each step seeing the previous mask."""
        probs = None
        for k in range(1, len(clicks) + 1):
            prefix = clicks[:k]
            key = (image_key, tuple((c.y, c.x, c.positive) for c in prefix), self.selection, threshold)
            cached = self._cache.get(key)
            if cached is None:
                cached = self._forward(image_chw, prefix, probs, threshold)
                self._remember(key, cached)
            probs = cached
        return probs

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

        image_chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0

        if self.uses_previous_mask:
            image_key = hashlib.sha1(resized.tobytes()).hexdigest()
            probs = self._replay(image_key, image_chw, scaled, threshold)
        else:
            # Stateless model: every click is in the channels already, one pass.
            probs = self._forward(image_chw, scaled, None, threshold)

        # Back to the user's resolution. The probability map is resized rather
        # than the thresholded mask, so the boundary is interpolated smoothly
        # instead of showing model-resolution staircase edges.
        probs_full = np.array(
            Image.fromarray(probs).resize((original_w, original_h), Image.BILINEAR)
        )
        return probs_full > threshold, probs_full
