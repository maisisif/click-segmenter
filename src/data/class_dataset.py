"""One-class semantic segmentation data: RGB image in, one binary mask out.

This is the simplified task Kassem asked for on 2026-09-08: pick a single
ADE20K class (chair), keep only the images that contain it, and give each of
those images exactly one mask -- the union of every instance of that class.
No clicks, no candidates. IoU is then computed between the predicted chair
pixels and the ground-truth chair pixels only; background is never scored.

Building the index reads each image's JSON once to find which top-level
objects carry the class name. Only those few instance PNGs are decoded per
item (typically 1-3 for chair), so nothing is precomputed and the dataloader
stays cheap. The index is cached as JSON because scanning 12k JSON files off
shared storage takes a minute or two.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ade20k import load_instance_mask
from src.data.dataset import _resize_image, _resize_mask


def _matches(name: str, class_name: str, match: str) -> bool:
    name = name.strip().lower()
    if match == "exact":
        # ADE20K names are comma-separated synonym lists ("chair", "armchair",
        # "chair, seat"); exact means the first term is the class.
        first = name.split(",")[0].strip()
        return first == class_name
    if match == "contains":
        return class_name in name
    raise ValueError(f"unknown match mode {match!r}, expected 'exact' or 'contains'")


def _scan_image(image_path: Path, class_name: str, match: str) -> tuple[str, list[int]]:
    stem = image_path.stem
    json_path = image_path.parent / f"{stem}.json"
    instances_dir = image_path.parent / stem
    ids: list[int] = []
    if not json_path.exists() or not instances_dir.is_dir():
        return str(image_path), ids
    with open(json_path) as f:
        objects = json.load(f)["annotation"]["object"]
    for obj in objects:
        parts = obj.get("parts") or {}
        if int(parts.get("part_level", 0)) != 0:
            continue  # a chair leg is a part, not a chair
        if not _matches(str(obj.get("name", "")), class_name, match):
            continue
        instance_id = int(obj["id"])
        if (instances_dir / f"instance_{instance_id:03d}_{stem}.png").exists():
            ids.append(instance_id)
    return str(image_path), ids


def build_class_index(
    image_paths: list[Path],
    class_name: str,
    match: str = "exact",
    cache_path: Path | None = None,
    workers: int = 16,
) -> dict[str, list[int]]:
    """Map image path -> instance ids of `class_name`, for every image given.

    Images without the class map to an empty list, so the same index answers
    both "which images have a chair" and "how many chairs are in each".
    """
    class_name = class_name.strip().lower()
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("class_name") == class_name and cached.get("match") == match:
            index = cached["index"]
            wanted = {str(p) for p in image_paths}
            if wanted <= set(index):
                return {k: index[k] for k in wanted}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda p: _scan_image(p, class_name, match), image_paths))
    index = dict(results)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"class_name": class_name, "match": match, "index": index}, f)
    return index


def summarize_index(index: dict[str, list[int]]) -> dict[str, int]:
    """The counts Kassem asked for: images scanned, images with the class, instances."""
    with_class = [ids for ids in index.values() if ids]
    return {
        "images_scanned": len(index),
        "images_with_class": len(with_class),
        "images_without_class": len(index) - len(with_class),
        "instances": sum(len(ids) for ids in with_class),
    }


class ClassSegmentationDataset(Dataset):
    """(image, union mask) pairs for one class.

    `entries` is a list of (image_path, instance_ids). Only pass images that
    contain the class unless negatives are wanted on purpose; an image with no
    instances yields an all-zero mask, which the loss handles but the per-image
    IoU does not (0/0), so `run_epoch` in the training script skips those in
    the IoU average and counts them separately.
    """

    def __init__(
        self,
        entries: list[tuple[Path, list[int]]],
        image_size: tuple[int, int],
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.entries = entries
        self.image_size = tuple(image_size)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, ids = self.entries[idx]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=bool)
        for instance_id in ids:
            mask |= load_instance_mask(image_path, instance_id)

        image = _resize_image(image, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        if self.augment and self.rng.random() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        return image_t, mask_t
