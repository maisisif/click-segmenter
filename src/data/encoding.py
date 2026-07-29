"""Encoding simulated clicks as extra input channels for the model."""

from __future__ import annotations

import numpy as np

from src.data.clicks import Click


def encode_clicks(clicks: list[Click], shape: tuple[int, int], radius: int = 5) -> np.ndarray:
    """Render clicks as a (2, H, W) float32 array: channel 0 is positive clicks,
    channel 1 is negative clicks, each a disk of `radius` pixels at 1.0.
    """
    # Recomputes the full coordinate grid per call; fine for prototyping, but this
    # runs in the training dataloader's hot path later — revisit with a local
    # bounding-box stamp instead of a whole-image grid if it becomes a bottleneck.
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]

    positive = np.zeros(shape, dtype=np.float32)
    negative = np.zeros(shape, dtype=np.float32)

    for click in clicks:
        disk = (yy - click.y) ** 2 + (xx - click.x) ** 2 <= radius**2
        target = positive if click.positive else negative
        target[disk] = 1.0

    return np.stack([positive, negative], axis=0)
