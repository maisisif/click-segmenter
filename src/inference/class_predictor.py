"""Run a class-segmentation checkpoint (scripts/train_class.py) on a user image.

The click model answers "what is at this click?" with one mask. The class model
answers "where is every bed / floor / ... in this photo?" with one probability
map per class, from the RGB image alone. It is the plain semantic-segmentation
baseline asked for on 2026-09-08, and in the interface it backs the Classes
tab: upload an image, get all masks; click a pixel, get the mask of the class
under it ("each click is a segmentation request").

Two kinds of checkpoint load here, as for ClickPredictor. A **training**
checkpoint from train_class.py carries `train_settings` (class names, image
size), so it is already self-describing. A **deployment** checkpoint written by
scripts/export_class_model.py drops the optimizer state and records the same
settings under `inference_config`, plus provenance (epoch, validation and test
IoU) for the interface footer. The architecture is read from the weight shapes
either way, so no configs/ checkout is needed on the serving side.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import _normalize_size
from src.model.build import build_model, detect_arch
from src.model.unet import migrate_legacy_state_dict
from src.training.device import get_device


class ClassPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.device = get_device(device or "auto")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        settings = checkpoint.get("inference_config") or checkpoint.get("train_settings")
        if settings is None or "class_names" not in settings:
            raise ValueError(
                f"{checkpoint_path} is not a class-segmentation checkpoint "
                "(no class_names in train_settings / inference_config)"
            )
        self.class_names: list[str] = list(settings["class_names"])
        self.image_size = _normalize_size(settings["image_size"])  # (H, W)

        state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])
        arch_config = detect_arch(state_dict)
        if arch_config["num_masks"] != len(self.class_names):
            raise ValueError(
                f"checkpoint has {arch_config['num_masks']} output channels "
                f"but names {len(self.class_names)} classes: {self.class_names}"
            )
        self.model = build_model(arch_config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        divisor = 32 if arch_config["arch"] == "resnet34_unet" else 2 ** arch_config.get("depth", 3)
        self.image_size = tuple(-(-s // divisor) * divisor for s in self.image_size)
        self.arch = arch_config

        provenance = checkpoint.get("provenance", checkpoint)
        self.trained_epoch = provenance.get("best_epoch") or provenance.get("epoch")
        self.trained_val_iou = provenance.get("best_val_iou")
        self.test = provenance.get("test")  # dict from counts.json, or None

        self._cache_key: str | None = None
        self._cache_probs: np.ndarray | None = None

    # ------------------------------------------------------------------ core

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Per-class probabilities, shape (C, H, W), at the image's own size.

        Cached for the most recent image, because the Classes tab calls this
        once per click on the same photo.
        """
        key = hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()
        if key == self._cache_key and self._cache_probs is not None:
            return self._cache_probs

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image[..., :3]
        height, width = image.shape[:2]
        model_h, model_w = self.image_size
        small = np.array(Image.fromarray(image).resize((model_w, model_h), Image.BILINEAR))
        x = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float().div(255.0)
        x = x.unsqueeze(0).to(self.device)

        out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits)
        probs = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
        result = probs[0].cpu().numpy().astype(np.float32)

        self._cache_key, self._cache_probs = key, result
        return result

    def masks(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary masks, shape (C, H, W)."""
        return self.predict(image) >= threshold

    def class_at(self, image: np.ndarray, y: int, x: int, threshold: float = 0.5) -> int | None:
        """Index of the class the model puts under pixel (y, x), or None.

        The most probable class wins; None when even that one is below the
        threshold, i.e. the model does not think any known class is there.
        """
        probs = self.predict(image)
        y = int(np.clip(y, 0, probs.shape[1] - 1))
        x = int(np.clip(x, 0, probs.shape[2] - 1))
        column = probs[:, y, x]
        best = int(column.argmax())
        return best if column[best] >= threshold else None
