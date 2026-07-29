"""Rendering helpers for inspecting ADE20K samples."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps

from src.data.ade20k import Sample


def overlay_instances(sample: Sample, *, use_amodal: bool = False, alpha: float = 0.5) -> np.ndarray:
    """Blend each instance's mask over the image in a distinct color."""
    canvas = sample.image.astype(np.float32).copy()
    cmap = colormaps["tab20"]

    for i, instance in enumerate(sample.instances):
        mask = instance.mask_amodal if use_amodal else instance.mask_visible
        color = np.array(cmap(i % cmap.N)[:3]) * 255
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color

    return canvas.astype(np.uint8)
