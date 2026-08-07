"""Deterministic train/validation/test splits.

Splitting happens at the **image** level, never the instance level. A single
ADE20K image contributes many instances, and if instances from the same image
landed in both train and validation, the model would be validated on scenes it
had already memorized — inflating the numbers without anyone noticing. Image
level splitting keeps the sets genuinely disjoint.

The split is a deterministic function of the (sorted) input paths and the seed,
so the same data always produces the same split, across machines and runs, with
no split file to keep in sync.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def split_image_paths(
    image_paths: list[Path],
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 0,
) -> dict[str, list[Path]]:
    """Split image paths into train/val/test by the given ratios.

    Returns a dict with keys "train", "val" and "test".
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")
    if len(image_paths) < 3:
        raise ValueError(f"need at least 3 images to make three non-empty splits, got {len(image_paths)}")

    ordered = sorted(image_paths)  # never trust incoming order
    indices = np.random.default_rng(seed).permutation(len(ordered))

    train_ratio, val_ratio, _ = ratios
    n = len(ordered)
    train_end = int(round(n * train_ratio))
    val_end = train_end + int(round(n * val_ratio))

    # Guarantee every split gets at least one image, which matters on the tiny
    # 3-image sample used for dry runs.
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    return {
        "train": [ordered[i] for i in indices[:train_end]],
        "val": [ordered[i] for i in indices[train_end:val_end]],
        "test": [ordered[i] for i in indices[val_end:]],
    }
