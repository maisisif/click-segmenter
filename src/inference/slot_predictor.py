"""Run a slot checkpoint: encode an image once, then answer clicks from cache.

The whole point of the slot architecture is that the expensive part does not
depend on the click, so this interface splits in two where `ClickPredictor`'s
does not:

    state = predictor.encode(image)      # one forward pass, the expensive bit
    mask, alternatives = state.select(clicks)   # pure tensor indexing, instant
    mask, alternatives = state.select(clicks + [another])   # still instant

`predict()` does both in one call and matches `ClickPredictor.predict`'s
signature, so evaluation code can treat the two architectures interchangeably.
Interactive callers should hold the state instead: that is where the speed is.

Masks are cached at the model's own stride (half resolution by default) rather
than upscaled eagerly. 64 slots at 384x512 is 50 MB per image in float32; at
half resolution it is 12 MB, and only the chosen mask is ever upsampled.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from src.data.clicks import Click
from src.data.dataset import _normalize_size
from src.model.build import build_model, detect_arch
from src.training.device import get_device


class SlotPrediction:
    """One image's slot masks, ready to answer any number of clicks.

    Holds probabilities at the model's mask resolution plus the original image
    size, so `select` can map clicks in, and upsample only what it returns.
    """

    def __init__(
        self,
        probs: torch.Tensor,          # (N, h, w) on cpu
        objectness: torch.Tensor,     # (N,)
        original_size: tuple[int, int],
        threshold: float = 0.5,
        objectness_threshold: float = 0.5,
    ) -> None:
        self.probs = probs
        self.objectness = objectness
        self.original_size = original_size
        self.threshold = threshold
        self.objectness_threshold = objectness_threshold

    @property
    def num_objects(self) -> int:
        """How many slots the model believes hold a real object."""
        return int((self.objectness > self.objectness_threshold).sum())

    def _grid_click(self, click: Click) -> tuple[int, int]:
        """Map a click in original image pixels onto the mask grid."""
        height, width = self.probs.shape[-2:]
        original_h, original_w = self.original_size
        y = min(int(click.y * height / original_h), height - 1)
        x = min(int(click.x * width / original_w), width - 1)
        return y, x

    def rank(self, clicks: list[Click]) -> list[int]:
        """Slots consistent with the clicks, smallest first.

        Smallest-first is the selection rule: a click inside nested objects
        (shirt, torso, person) is consistent with all three, and returning the
        tightest one matches what people usually mean when they point at
        something specific. The rest come back as alternatives to cycle through,
        which is the ambiguity handling the multi-mask model needed a dedicated
        head to produce.

        Negative clicks are pure exclusion here -- no forward pass, no extra
        machinery, just dropping every slot that contains the point.
        """
        binary = self.probs > self.threshold
        candidates = (self.objectness > self.objectness_threshold).nonzero().flatten().tolist()

        positives = [c for c in clicks if c.positive]
        negatives = [c for c in clicks if not c.positive]

        keep = []
        for slot in candidates:
            mask = binary[slot]
            if not all(mask[self._grid_click(c)] for c in positives):
                continue
            if any(mask[self._grid_click(c)] for c in negatives):
                continue
            keep.append(slot)

        # Nothing contains every positive click -- the model has no object there,
        # or the clicks straddle two. Fall back to whichever slot is most
        # confident at the first positive click, so a user always gets an answer
        # rather than a blank screen.
        if not keep and positives:
            y, x = self._grid_click(positives[0])
            column = self.probs[:, y, x].clone()
            for negative in negatives:
                ny, nx = self._grid_click(negative)
                column[binary[:, ny, nx]] = -1.0
            best = int(column.argmax())
            if column[best] > 0:
                keep = [best]

        keep.sort(key=lambda s: int(binary[s].sum()))
        return keep

    def select(self, clicks: list[Click]) -> tuple[np.ndarray, list[int]]:
        """Return (mask at the original image size, ranked slot indices)."""
        if not clicks:
            raise ValueError("at least one click is required")

        ranked = self.rank(clicks)
        if not ranked:
            return np.zeros(self.original_size, dtype=bool), []
        return self.mask_for(ranked[0]), ranked

    def mask_for(self, slot: int) -> np.ndarray:
        """Upsample one slot's mask to the original image size.

        Probabilities are resized rather than the thresholded mask, so the
        boundary interpolates smoothly instead of showing the mask grid.
        """
        probs = self.probs[slot][None, None]
        full = F.interpolate(
            probs, size=self.original_size, mode="bilinear", align_corners=False
        )
        return (full[0, 0] > self.threshold).numpy()


class SlotPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        train_config_path: str | Path = "configs/train.yaml",
        device: str | None = None,
    ) -> None:
        self.device = get_device(device or "auto")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        inference_config = checkpoint.get("inference_config")
        if inference_config is None:
            with open(train_config_path) as f:
                inference_config = {
                    "image_size": yaml.safe_load(f)["data"]["image_size"],
                }
        self.image_size = _normalize_size(inference_config["image_size"])

        state_dict = checkpoint["model_state_dict"]
        arch_config = detect_arch(state_dict)
        if arch_config["arch"] != "slot_unet":
            raise ValueError(
                f"{checkpoint_path} holds a {arch_config['arch']} checkpoint; "
                "use ClickPredictor for that architecture"
            )

        self.model = build_model(arch_config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # The encoder divides by 32, so round the working size up to match.
        self.image_size = tuple(-(-s // 32) * 32 for s in self.image_size)

        self.arch = arch_config
        self.num_slots = arch_config["num_slots"]
        provenance = checkpoint.get("provenance", checkpoint)
        self.trained_epoch = provenance.get("epoch")
        self.trained_val_iou = provenance.get("best_val_iou")

    def encode(self, image: np.ndarray, threshold: float = 0.5) -> SlotPrediction:
        """One forward pass. Every subsequent click is answered from the result."""
        original_h, original_w = image.shape[:2]
        model_h, model_w = self.image_size

        resized = np.array(Image.fromarray(image).resize((model_w, model_h), Image.BILINEAR))
        tensor = torch.from_numpy(resized.astype(np.float32).transpose(2, 0, 1) / 255.0)

        with torch.no_grad():
            logits, objectness = self.model(tensor.unsqueeze(0).to(self.device))

        return SlotPrediction(
            probs=torch.sigmoid(logits)[0].cpu(),
            objectness=torch.sigmoid(objectness)[0].cpu(),
            original_size=(original_h, original_w),
            threshold=threshold,
        )

    def predict(
        self,
        image: np.ndarray,
        clicks: list[Click],
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode and select in one call, matching ClickPredictor.predict.

        Convenient, and correct, but it throws away the cached encoding -- so an
        interactive caller clicking repeatedly on one image should use `encode`
        and hold the state instead.
        """
        state = self.encode(image, threshold=threshold)
        mask, ranked = state.select(clicks)
        probs = (
            state.mask_for(ranked[0]).astype(np.float32)
            if ranked
            else np.zeros(state.original_size, dtype=np.float32)
        )
        return mask, probs
