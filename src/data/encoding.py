"""Encoding simulated clicks as extra input channels for the model.

Two encodings are available, selected by `clicks.encoding` in configs/clicks.yaml:

**disk** — each click is a filled disk of `radius` pixels set to 1.0, everything
else 0. Simple and local. The catch is that a radius-5 disk covers ~0.5% of a
128x128 image, and our depth-3 UNet has a receptive field of roughly 68 pixels
at the bottleneck. For a large object, the click simply cannot influence
predictions at the far side of the mask, so the model can't reliably tell two
instances in the same image apart.

**distance** — each pixel stores its (truncated, normalized) proximity to the
nearest click: 1.0 at a click, falling linearly to 0.0 at `max_distance` pixels
away. Every pixel therefore carries information about where the click was,
regardless of receptive field. This is the encoding used by the original
interactive segmentation work (Xu et al. 2016, "Deep Interactive Object
Selection") and it is the safer default for large objects.

Both produce a (2, H, W) float32 array: channel 0 positive clicks, channel 1
negative clicks.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.data.clicks import Click


def _encode_disk(clicks: list[Click], shape: tuple[int, int], radius: int) -> np.ndarray:
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


def _encode_distance(clicks: list[Click], shape: tuple[int, int], max_distance: float) -> np.ndarray:
    """Truncated, normalized proximity to the nearest click of each polarity.

    A channel with no clicks stays all zeros, which is meaningful: it says
    "no negative clicks were given" rather than "a negative click is infinitely
    far away".
    """
    channels = []
    for positive in (True, False):
        points = [c for c in clicks if c.positive == positive]
        if not points:
            channels.append(np.zeros(shape, dtype=np.float32))
            continue

        # distance_transform_edt measures distance to the nearest ZERO, so seed
        # the click locations as 0 and everything else as 1.
        seeds = np.ones(shape, dtype=np.uint8)
        for click in points:
            seeds[click.y, click.x] = 0

        distance = distance_transform_edt(seeds)
        proximity = 1.0 - np.clip(distance, 0, max_distance) / max_distance
        channels.append(proximity.astype(np.float32))

    return np.stack(channels, axis=0)


def encode_clicks(
    clicks: list[Click],
    shape: tuple[int, int],
    radius: int = 5,
    encoding: str = "disk",
    max_distance: float = 64.0,
) -> np.ndarray:
    """Render clicks as a (2, H, W) float32 array. See the module docstring."""
    if encoding == "disk":
        return _encode_disk(clicks, shape, radius)
    if encoding == "distance":
        return _encode_distance(clicks, shape, max_distance)
    raise ValueError(f"unknown click encoding {encoding!r}, expected 'disk' or 'distance'")
