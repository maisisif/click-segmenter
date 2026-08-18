"""UNet with an ImageNet-pretrained ResNet-34 encoder.

Why this exists: three controlled experiments on the from-scratch UNet moved
test IoU by at most ~0.01 each (data x3.3, more epochs, 12x resolution +
depth + neighbour clicks). Every published method that reaches 0.8+ IoU
(RITM, SimpleClick, FocalClick) starts from a pretrained encoder, and
SimpleClick explicitly credits pretraining as the main factor. This is that
change: the encoder already knows what objects look like; we teach it what
clicks mean.

Two details that make it work:

- The first conv is rebuilt for 5 input channels. The pretrained RGB weights
  are copied in; the two click channels are ZERO-initialised (Xu et al. 2016),
  so at step 0 the network behaves exactly like the pretrained model and the
  click signal grows in during training instead of destroying the features.
- ImageNet models expect normalised RGB. Normalisation happens inside the
  model (registered buffers), so the dataset, the app and the notebook keep
  feeding [0,1] images unchanged.

The decoder reuses the same Up/DoubleConv blocks as the from-scratch UNet.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from src.model.unet import DoubleConv, Up

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetUNet(nn.Module):
    """Encoder: pretrained ResNet-34 (stages /2 to /32). Decoder: plain UNet ups.

    Input height and width must be divisible by 32.
    """

    def __init__(self, in_channels: int = 5, out_channels: int = 1, pretrained: bool = True) -> None:
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        # Rebuild the stem conv for 5 channels; keep pretrained RGB filters,
        # zero the click channels so pretrained behaviour is undisturbed at init.
        old_conv = backbone.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.zero_()
            self.conv1.weight[:, :3] = old_conv.weight

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4, 64
        self.layer2 = backbone.layer2  # /8, 128
        self.layer3 = backbone.layer3  # /16, 256
        self.layer4 = backbone.layer4  # /32, 512

        self.up1 = Up(512, 256, 256)  # -> /16
        self.up2 = Up(256, 128, 128)  # -> /8
        self.up3 = Up(128, 64, 64)    # -> /4
        self.up4 = Up(64, 64, 32)     # -> /2
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = DoubleConv(32, 16)
        self.head = nn.Conv2d(16, out_channels, kernel_size=1)

        self.register_buffer("rgb_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = (x[:, :3] - self.rgb_mean) / self.rgb_std
        x = torch.cat([rgb, x[:, 3:]], dim=1)

        s0 = self.relu(self.bn1(self.conv1(x)))   # /2, 64
        s1 = self.layer1(self.maxpool(s0))        # /4, 64
        s2 = self.layer2(s1)                      # /8, 128
        s3 = self.layer3(s2)                      # /16, 256
        bottom = self.layer4(s3)                  # /32, 512

        d = self.up1(bottom, s3)
        d = self.up2(d, s2)
        d = self.up3(d, s1)
        d = self.up4(d, s0)
        return self.head(self.final_conv(self.final_up(d)))
