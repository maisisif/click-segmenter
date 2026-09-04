"""PyTorch dataset that turns ADE20K instances into (image+clicks, mask) pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ade20k import Instance, Sample, load_instance, load_instance_mask, load_sample
from src.data.clicks import Click, sample_positive_click, simulate_clicks
from src.data.encoding import encode_clicks

# Keys in clicks.yaml that configure rendering or the dataset itself, not the
# click sampler. Everything else is forwarded to simulate_clicks.
_NON_SAMPLER_KEYS = {"encoding", "radius", "max_distance", "neighbor_negative_prob"}


def _normalize_size(size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Accept a single int (square) or an (H, W) pair, return (H, W)."""
    if isinstance(size, int):
        return (size, size)
    height, width = size
    return (int(height), int(width))


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # `size` is (H, W); PIL wants (W, H).
    return np.array(Image.fromarray(image).resize((size[1], size[0]), Image.BILINEAR))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((size[1], size[0]), Image.NEAREST)
    return np.array(resized) > 127


def _index_cache_key(image_paths: list[Path], image_size: tuple[int, int]) -> str:
    """Fingerprint of the exact inputs a cached index is valid for."""
    digest = hashlib.sha256()
    digest.update(str(tuple(image_size)).encode())
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
    key = _index_cache_key(image_paths, image_size)

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

    print(f"Instance index built: {len(items)} usable, {dropped} dropped as empty at {image_size}")

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
      - `lazy=True`: builds (or loads) an index of usable (image, instance)
        pairs, then decodes exactly one image + one mask per __getitem__.
        Use this for real training so the full dataset never sits in memory.
        `mask_areas` isn't available in this mode.

    Neighbour negatives (Xu et al. 2016, strategy 2): with probability
    `clicks.neighbor_negative_prob`, one extra negative click is placed on a
    *different instance in the same image*. Without this, negative clicks only
    ever land on nearby background, and the model is never explicitly taught
    that the adjacent object is not the target — which shows up as masks
    bleeding across instance boundaries.
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int | tuple[int, int] | list[int],
        click_config: dict,
        deterministic: bool = False,
        lazy: bool = False,
        index_cache: Path | None = None,
    ) -> None:
        self.image_size = _normalize_size(image_size)
        self.click_config = click_config
        self.deterministic = deterministic
        self.lazy = lazy

        if lazy:
            self.mask_areas = None
            self._lazy_items = _build_valid_index(image_paths, self.image_size, index_cache)
            # Per-image list of instance ids, for neighbour-negative sampling.
            self._ids_by_path: dict[Path, list[int]] = {}
            for path, instance_id in self._lazy_items:
                self._ids_by_path.setdefault(path, []).append(instance_id)
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

    def _neighbor_mask(self, idx: int, rng: np.random.Generator) -> np.ndarray | None:
        """The resized mask of a randomly chosen *other* instance in this image."""
        if self.lazy:
            path, instance_id = self._lazy_items[idx]
            others = [i for i in self._ids_by_path.get(path, []) if i != instance_id]
            if not others:
                return None
            raw = load_instance_mask(path, others[rng.integers(len(others))])
        else:
            sample, instance = self.items[idx]
            others = [i for i in sample.instances if i.id != instance.id]
            if not others:
                return None
            raw = others[rng.integers(len(others))].mask_visible
        return _resize_mask(raw, self.image_size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Both modes guarantee every indexed instance has a non-empty mask at
        # image_size, so no retry logic is needed here.
        image, mask = self._load_image_and_mask(idx)

        # Deterministic mode fixes the click (via the item index as seed) so the
        # overfit sanity check tests pure memorization of a fixed input->output
        # mapping. Real training should leave this off, so a fresh click is
        # sampled each epoch — acting as a natural form of augmentation.
        rng = np.random.default_rng(idx if self.deterministic else None)

        simulate_kwargs = {k: v for k, v in self.click_config.items() if k not in _NON_SAMPLER_KEYS}
        clicks = simulate_clicks(mask, rng, **simulate_kwargs)

        neighbor_prob = self.click_config.get("neighbor_negative_prob", 0.0)
        if neighbor_prob > 0 and rng.random() < neighbor_prob:
            neighbor = self._neighbor_mask(idx, rng)
            if neighbor is not None:
                # Never place the "not the target" click inside the target.
                neighbor_only = neighbor & ~mask
                if neighbor_only.any():
                    click = sample_positive_click(neighbor_only, rng)
                    clicks.append(Click(y=click.y, x=click.x, positive=False))

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


class SlotSegmentationDataset(Dataset):
    """One item = one whole image with *all* its instance masks.

    The counterpart to ClickSegmentationDataset, for the slot architecture. The
    click model needs one object per example because the click selects it before
    the forward pass; the slot model predicts every object at once, so an example
    has to carry every object.

    That inverts the loading trade-off. `load_instance` exists because decoding
    ~20 masks to use 1 made the dataloader the bottleneck -- but here all ~20 are
    used, and an epoch is one item per image rather than one per instance. On the
    12k set that is 8,402 items instead of 167,210: each is ~20x more work, but
    there are ~20x fewer, and far fewer forward passes.

    The instance index is the same cache the click dataset builds, keyed by the
    same paths and image size, so switching architectures does not force a
    rebuild.

    Returns (image, masks):
        image  (3, H, W) float in [0, 1]
        masks  (K, H, W) float, K variable and possibly 0

    K varies per image, so batches need `slot_collate` rather than the default.
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int | tuple[int, int] = 128,
        max_objects: int = 64,
        index_cache: Path | None = None,
    ) -> None:
        self.image_size = _normalize_size(image_size)
        self.max_objects = max_objects

        items = _build_valid_index(image_paths, self.image_size, index_cache)
        ids_by_path: dict[Path, list[int]] = {}
        for path, instance_id in items:
            ids_by_path.setdefault(path, []).append(instance_id)

        # Sorted so the dataset order is deterministic across runs and machines.
        self.images = sorted(ids_by_path)
        self.ids_by_path = ids_by_path

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.images[idx]

        image = _resize_image(np.array(Image.open(path).convert("RGB")), self.image_size)
        image_tensor = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0)

        masks = []
        for instance_id in self.ids_by_path[path]:
            mask = _resize_mask(load_instance_mask(path, instance_id), self.image_size)
            if mask.any():  # the index promises this, but a mask file can go missing
                masks.append(mask)

        # Roughly 5% of ADE20K images carry more objects than there are slots.
        # Keep the largest: small objects are both less likely to be clicked and
        # the ones the model segments worst, so they are the cheapest to drop.
        if len(masks) > self.max_objects:
            masks.sort(key=lambda m: -int(m.sum()))
            masks = masks[: self.max_objects]

        if not masks:
            return image_tensor, torch.zeros(0, *self.image_size)
        return image_tensor, torch.from_numpy(np.stack(masks).astype(np.float32))


def slot_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Stack images, leave masks as a list -- object count varies per image.

    This is the shape HungarianMaskLoss expects, and padding to a fixed K would
    only mean masking the padding out again during matching.
    """
    images = torch.stack([image for image, _ in batch])
    targets = [masks for _, masks in batch]
    return images, targets
