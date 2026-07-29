"""Loading for ADE20K instance-level annotations.

Each image `X.jpg` has a `X.json` listing annotated objects (and their
sub-parts) with a per-image `id`, and a sibling folder `X/` containing one
binary amodal mask per top-level object: `instance_{id:03d}_X.png`. Pixel
values in those masks are 0 (background), 128 (occluded but part of the
object), or 255 (visible).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Instance:
    id: int
    name: str
    mask_visible: np.ndarray  # bool HxW, True where the object is visible
    mask_amodal: np.ndarray  # bool HxW, True where the object is visible or occluded


@dataclass
class Sample:
    stem: str
    image_path: Path
    image: np.ndarray  # HxWx3 uint8 RGB
    instances: list[Instance]


def discover_samples(root: Path) -> list[Path]:
    """Find every image under `root` that has a matching _seg.png sibling."""
    return sorted(p for p in root.rglob("*.jpg") if (p.parent / f"{p.stem}_seg.png").exists())


def _load_instance_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(Image.open(path))
    return arr == 255, arr >= 128


def load_sample(image_path: Path) -> Sample:
    stem = image_path.stem
    parent = image_path.parent
    json_path = parent / f"{stem}.json"
    instances_dir = parent / stem

    image = np.array(Image.open(image_path).convert("RGB"))

    instances: list[Instance] = []
    if json_path.exists() and instances_dir.is_dir():
        with open(json_path) as f:
            annotated_objects = json.load(f)["annotation"]["object"]

        for obj in annotated_objects:
            if int(obj["parts"]["part_level"]) != 0:
                continue  # sub-parts of an object aren't independently clickable instances

            mask_path = instances_dir / f"instance_{int(obj['id']):03d}_{stem}.png"
            if not mask_path.exists():
                continue  # a handful of annotated objects have no exported mask

            mask_visible, mask_amodal = _load_instance_mask(mask_path)
            instances.append(
                Instance(id=int(obj["id"]), name=obj["name"], mask_visible=mask_visible, mask_amodal=mask_amodal)
            )

    return Sample(stem=stem, image_path=image_path, image=image, instances=instances)
