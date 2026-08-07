"""PyTorch dataset that turns ADE20K instances into (image+clicks, mask) pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ade20k import Instance, Sample, load_sample
from src.data.clicks import simulate_clicks
from src.data.encoding import encode_clicks


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.array(resized) > 127


def _discover_instance_ids(image_path: Path) -> list[int]:
    """Read just the (small) sidecar JSON to list top-level instance ids for an
    image, without decoding the image or any mask PNGs. Mirrors the part_level
    filtering in `load_sample`, so lazy mode enumerates the same items eager
    mode would.
    """
    json_path = image_path.parent / f"{image_path.stem}.json"
    if not json_path.exists():
        return []
    with open(json_path) as f:
        annotated_objects = json.load(f)["annotation"]["object"]
    return [int(obj["id"]) for obj in annotated_objects if int(obj["parts"]["part_level"]) == 0]


class ClickSegmentationDataset(Dataset):
    """One item = one (image, instance) pair, with a freshly simulated click.

    Two loading modes:
      - `lazy=False` (default): loads every sample's image + masks up front.
        Fine for the handful of images we have locally, and needed for the
        overfit sanity check (`scripts/train.py`), which ranks instances by
        mask area to pick a cherry-picked subset.
      - `lazy=True`: only reads the small sidecar JSON files up front to
        enumerate (image_path, instance_id) pairs; each `__getitem__` call
        loads and decodes just that one image + its instance masks, then lets
        it be garbage-collected. Use this once the full ~27k-image dataset is
        in play (M4+) to avoid holding everything in memory at once.
        `mask_areas` isn't available in this mode (it would require decoding
        every mask up front, defeating the purpose).
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        click_config: dict,
        deterministic: bool = False,
        lazy: bool = False,
    ) -> None:
        self.image_size = (image_size, image_size)
        self.click_config = click_config
        self.deterministic = deterministic
        self.lazy = lazy

        if lazy:
            self.mask_areas = None
            self._lazy_items: list[tuple[Path, int]] = [
                (path, instance_id) for path in image_paths for instance_id in _discover_instance_ids(path)
            ]
        else:
            self.items: list[tuple[Sample, Instance]] = []
            self.mask_areas: list[int] = []
            for path in image_paths:
                sample = load_sample(path)
                for instance in sample.instances:
                    resized_mask = _resize_mask(instance.mask_visible, self.image_size)
                    # Drop instances that vanish entirely once downsized to image_size —
                    # a click can't be simulated on an empty mask.
                    if resized_mask.any():
                        self.items.append((sample, instance))
                        self.mask_areas.append(int(resized_mask.sum()))

    def __len__(self) -> int:
        return len(self._lazy_items) if self.lazy else len(self.items)

    def _load_image_and_mask(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.lazy:
            path, instance_id = self._lazy_items[idx]
            sample = load_sample(path)  # loads only this one image + its masks
            instance = next(i for i in sample.instances if i.id == instance_id)
        else:
            sample, instance = self.items[idx]

        image = _resize_image(sample.image, self.image_size)
        mask = _resize_mask(instance.mask_visible, self.image_size)
        return image, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # In lazy mode, downsizing to image_size can occasionally empty out a
        # thin/tiny instance (the eager path filters these out up front — see
        # class docstring). Retry a handful of nearby indices rather than
        # crashing mid-epoch on a rare edge case.
        max_attempts = 5
        for attempt in range(max_attempts):
            candidate_idx = (idx + attempt) % len(self)
            image, mask = self._load_image_and_mask(candidate_idx)
            if mask.any():
                idx = candidate_idx
                break
        else:
            raise RuntimeError(
                f"Could not find a non-empty mask near index {idx} after {max_attempts} attempts"
            )

        # Deterministic mode fixes the click (via the item index as seed) so the
        # overfit sanity check tests pure memorization of a fixed input->output
        # mapping. Real training should leave this off, so a fresh click is
        # sampled each epoch — acting as a natural form of augmentation.
        rng = np.random.default_rng(idx if self.deterministic else None)

        # Split the click config: sampling parameters go to simulate_clicks,
        # rendering parameters go to encode_clicks.
        encoding_keys = {"encoding", "radius", "max_distance"}
        simulate_kwargs = {k: v for k, v in self.click_config.items() if k not in encoding_keys}
        clicks = simulate_clicks(mask, rng, **simulate_kwargs)
        encoded = encode_clicks(
            clicks,
            shape=mask.shape,
            radius=self.click_config.get("radius", 5),
            encoding=self.click_config.get("encoding", "disk"),
            max_distance=self.click_config.get("max_distance", 64.0),
        )

        image_chw = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_array = np.concatenate([image_chw, encoded], axis=0)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return input_tensor, target_tensor
