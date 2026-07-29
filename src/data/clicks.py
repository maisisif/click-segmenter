"""Simulating user clicks from ground-truth masks.

Real click data doesn't exist in ADE20K, so during training we synthesize
clicks that mimic how a person would actually click: a positive click near
the "middle" of the target object, and (sometimes) a negative click just
outside its boundary — the kind of correction click a user makes on a
nearby confusable region rather than somewhere irrelevant in the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class Click:
    y: int
    x: int
    positive: bool


def sample_positive_click(
    mask_visible: np.ndarray, rng: np.random.Generator, interior_frac: float = 0.5
) -> Click:
    """Sample a point biased toward the interior of the mask, away from its boundary."""
    if not mask_visible.any():
        raise ValueError("mask has no foreground pixels to click on")

    distance_to_boundary = distance_transform_edt(mask_visible)
    threshold = distance_to_boundary.max() * interior_frac
    candidates = np.argwhere(distance_to_boundary >= threshold)

    y, x = candidates[rng.integers(len(candidates))]
    return Click(y=int(y), x=int(x), positive=True)


def sample_negative_click(
    mask_visible: np.ndarray,
    rng: np.random.Generator,
    min_distance: float = 5,
    max_distance: float = 40,
) -> Click | None:
    """Sample a background point within [min_distance, max_distance] of the mask boundary."""
    distance_to_mask = distance_transform_edt(~mask_visible)
    band = (distance_to_mask >= min_distance) & (distance_to_mask <= max_distance)
    candidates = np.argwhere(band)

    if len(candidates) == 0:
        return None  # e.g. the mask fills (almost) the entire image

    y, x = candidates[rng.integers(len(candidates))]
    return Click(y=int(y), x=int(x), positive=False)


def simulate_clicks(
    mask_visible: np.ndarray,
    rng: np.random.Generator,
    *,
    positive_interior_frac: float = 0.5,
    negative_min_distance: float = 5,
    negative_max_distance: float = 40,
    negative_prob: float = 0.5,
) -> list[Click]:
    """Simulate the clicks a user might make to select this object: always one
    positive click, plus a negative click with probability `negative_prob`.
    """
    clicks = [sample_positive_click(mask_visible, rng, positive_interior_frac)]

    if rng.random() < negative_prob:
        negative = sample_negative_click(mask_visible, rng, negative_min_distance, negative_max_distance)
        if negative is not None:
            clicks.append(negative)

    return clicks
