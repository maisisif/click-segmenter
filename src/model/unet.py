"""A UNet-style encoder-decoder, trained from scratch (no pretrained weights).

Input: the image stacked with the click-encoding channels from `src.data.encoding`.
Output: a single-channel logit map — the predicted foreground (clicked object) mask.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """UNet with a configurable number of downsampling stages.

    `depth` counts the downsampling steps (the last one is the bottleneck), so
    input height and width must both be divisible by 2**depth. Channel widths
    double each stage from `base_channels`: depth 3 at base 32 gives
    32-64-128-256, depth 4 adds a 512 bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.depth = depth
        self.stem = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList(Down(channels[i], channels[i + 1]) for i in range(depth))
        self.ups = nn.ModuleList(
            Up(channels[i + 1], channels[i], channels[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.head(x)


# Checkpoints saved before depth was configurable used fixed layer names
# (down1/down2/bottleneck/up2/up1/up0). This maps them onto the ModuleList
# naming so old checkpoints (e.g. the released best.pt) keep loading.
_LEGACY_KEY_MAP = {
    "down1.": "downs.0.",
    "down2.": "downs.1.",
    "bottleneck.": "downs.2.",
    "up2.": "ups.0.",
    "up1.": "ups.1.",
    "up0.": "ups.2.",
}


def migrate_legacy_state_dict(state_dict: dict) -> dict:
    """Rename pre-`depth` checkpoint keys to the current layout. No-op for new ones."""
    if not any(key.startswith(("down1.", "bottleneck.")) for key in state_dict):
        return state_dict
    migrated = {}
    for key, value in state_dict.items():
        for old, new in _LEGACY_KEY_MAP.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        migrated[key] = value
    return migrated
