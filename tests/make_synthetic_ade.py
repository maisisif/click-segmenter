"""Write a tiny synthetic dataset in the on-disk ADE20K layout, for CPU tests.

    python tests/make_synthetic_ade.py --root /tmp/fake-ade --images 24

Each image is a flat "room" with a random subset of coloured rectangles, one
per class name, and gets the three things the loaders look for: `X.jpg`,
`X_seg.png` (existence only), `X.json` with the object list, and a folder `X/`
of `instance_{id:03d}_X.png` masks (255 where visible). The class names cover
both scene-like and object-like ADE names so `--select any` can be exercised:
every image has a sky and a floor, only some have a bed, a car or a person.

Nothing here resembles real photographs; the point is that the whole pipeline
(index -> split -> train -> export -> predict) runs end to end in a minute on a
laptop with no dataset, which is where most wiring mistakes are caught.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {
    # name: (always present?, colour)
    "sky": (True, (150, 190, 240)),
    "floor": (True, (170, 130, 90)),
    "wall": (True, (220, 215, 200)),
    "bed": (False, (200, 60, 60)),
    "car": (False, (40, 40, 60)),
    "person": (False, (240, 190, 150)),
}


def make_image(rng: np.random.Generator, size: tuple[int, int]) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)
    objects, masks = [], []
    horizon = h // 2
    image[:horizon] = CLASSES["sky"][1]
    image[horizon:] = CLASSES["floor"][1]
    sky = np.zeros((h, w), bool); sky[:horizon] = True
    floor = np.zeros((h, w), bool); floor[horizon:] = True
    layers = [("sky", sky), ("floor", floor)]

    # a wall strip just above the horizon
    wall = np.zeros((h, w), bool); wall[horizon - h // 6 : horizon] = True
    image[wall] = CLASSES["wall"][1]
    layers.append(("wall", wall))

    for name in ("bed", "car", "person"):
        if rng.random() < 0.5:
            bh, bw = int(h * rng.uniform(0.15, 0.35)), int(w * rng.uniform(0.15, 0.35))
            y0 = int(rng.uniform(horizon - bh // 2, h - bh))
            x0 = int(rng.uniform(0, w - bw))
            m = np.zeros((h, w), bool); m[y0 : y0 + bh, x0 : x0 + bw] = True
            image[m] = CLASSES[name][1]
            layers.append((name, m))

    # later layers occlude earlier ones; masks store the visible part only
    covered = np.zeros((h, w), bool)
    for name, m in reversed(layers):
        visible = m & ~covered
        covered |= m
        objects.append(name); masks.append(visible)
    objects.reverse(); masks.reverse()

    noise = rng.integers(-12, 12, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image, [{"name": n} for n in objects], masks


def write_dataset(root: Path, images: int, size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(images):
        stem = f"ADE_fake_{i:08d}"
        image, objects, masks = make_image(rng, size)
        Image.fromarray(image).save(root / f"{stem}.jpg", quality=90)
        Image.fromarray(np.zeros(size, np.uint8)).save(root / f"{stem}_seg.png")
        inst_dir = root / stem
        inst_dir.mkdir(exist_ok=True)
        entries = []
        for k, (obj, mask) in enumerate(zip(objects, masks), start=1):
            Image.fromarray((mask * 255).astype(np.uint8)).save(inst_dir / f"instance_{k:03d}_{stem}.png")
            entries.append({"id": k, "name": obj["name"], "parts": {"part_level": 0}})
        (root / f"{stem}.json").write_text(json.dumps({"annotation": {"object": entries}}))
    print(f"wrote {images} synthetic images to {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--images", type=int, default=24)
    parser.add_argument("--size", type=int, nargs=2, default=[96, 128], metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    write_dataset(Path(args.root), args.images, tuple(args.size), args.seed)


if __name__ == "__main__":
    main()
