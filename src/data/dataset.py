"""PyTorch dataset that turns ADE20K instances into (image+clicks, mask) pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ade20k import Instance, Sample, load_instance, load_sample
from src.data.clicks import simulate_clicks
from src.data.encoding import encode_clicks


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.array(resized) > 127


def _index_cache_key(image_paths: list[Path], image_size: int) -> str:
    """Fingerprint of the exact inputs a cached index is valid for."""
    digest = hashlib.sha256()
    digest.update(str(image_size).encode())
    for path in sorted(image_paths):
        digest.update(str(path).encode())
    return digest.hexdigest()


def _build_valid_index(
    image_paths: list[Path], image_size: tuple[int, int], cache_path: Path | None
) -> list[tuple[Path, int]]:
    """List every (image, instance) pair whose mask survives downsizing.

    An instance whose mask is empty at `image_size` can't have a click
    simulated on it, so it must be excluded rather than skipped at access
    time. Determining this requires actually decoding the masks, which is
    slow enough (tens of thousands of small PNGs on shared storage) that the
    result is cached and keyed by the input paths and image size.
    """
    key = _index_cache_key(image_paths, image_size[0])

    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                print(f"Using cached instance index ({len(cached['items'])} instances) from {cache_path}")
                return [(Path(p), int(i)) for p, i in cached["items"]]
            print("Instance index cache is stale (inputs changed), rebuilding")
        except (json.JSONDecodeError, KeyError, TypeError):
            print("Instance index cache is unreadable, rebuilding")

    print(f"Building instance index over {len(image_paths)} images (one-off, then cached)...")
    items: list[tuple[Path, int]] = []
    dropped = 0
    for n, path in enumerate(image_paths, 1):
        sample = load_sample(path)
        for instance in sample.instances:
            if _resize_mask(instance.mask_visible, image_size).any():
                items.append((path, instance.id))
            else:
                dropped += 1
        if n % 500 == 0:
            print(f"  indexed {n}/{len(image_paths)} images, {len(items)} instances kept")

    print(f"Instance index built: {len(items)} usable, {dropped} dropped as empty at {image_size[0]}px")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"key": key, "items": [[str(p), i] for p, i in items]}, f)
        print(f"Cached instance index to {cache_path}")

    return items


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
        index_cache: Path | None = None,
    ) -> None:
        self.image_size = (image_size, image_size)
        self.click_config = click_config
        self.deterministic = deterministic
        self.lazy = lazy

        if lazy:
            self.mask_areas = None
            self._lazy_items = _build_valid_index(image_paths, self.image_size, index_cache)
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
            # Decode only the one mask we need, not all ~20 for this image.
            raw_image, raw_mask = load_instance(path, instance_id)
        else:
            sample, instance = self.items[idx]
            raw_image, raw_mask = sample.image, instance.mask_visible

        return _resize_image(raw_image, self.image_size), _resize_mask(raw_mask, self.image_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Both modes guarantee every indexed instance has a non-empty mask at
        # image_size, so no retry logic is needed here.
        image, mask = self._load_image_and_mask(idx)

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
