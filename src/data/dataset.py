"""PyTorch dataset that turns ADE20K instances into (image+clicks, mask) pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ade20k import Instance, Sample, load_sample
from src.data.clicks import simulate_clicks
from src.data.encoding import encode_clicks


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(image).resize(size, Image.BILINEAR))


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    return np.array(resized) > 127


class ClickSegmentationDataset(Dataset):
    """One item = one (image, instance) pair, with a freshly simulated click.

    Loads every sample eagerly, which is fine for the handful of images we
    have locally — this needs to become lazy, per-item loading once the full
    ~27k-image dataset is in play (M4), to avoid holding everything in memory.
    """

    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        click_config: dict,
        deterministic: bool = False,
    ) -> None:
        self.image_size = (image_size, image_size)
        self.click_config = click_config
        self.deterministic = deterministic

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
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample, instance = self.items[idx]

        image = _resize_image(sample.image, self.image_size)
        mask = _resize_mask(instance.mask_visible, self.image_size)

        # Deterministic mode fixes the click (via the item index as seed) so the
        # overfit sanity check tests pure memorization of a fixed input->output
        # mapping. Real training should leave this off, so a fresh click is
        # sampled each epoch — acting as a natural form of augmentation.
        rng = np.random.default_rng(idx if self.deterministic else None)

        simulate_kwargs = {k: v for k, v in self.click_config.items() if k != "radius"}
        clicks = simulate_clicks(mask, rng, **simulate_kwargs)
        encoded = encode_clicks(clicks, shape=mask.shape, radius=self.click_config["radius"])

        image_chw = image.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_array = np.concatenate([image_chw, encoded], axis=0)

        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return input_tensor, target_tensor
