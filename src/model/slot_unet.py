"""Class-agnostic slot segmentation: predict every object once, select by click.

A different answer to the same question as `resnet_unet.py`, and worth stating
the contrast plainly because it inverts the design.

In the click-conditioned model, the click is an *input*: it is stamped into two
extra image channels, the network computes one object's mask conditioned on it,
and every new click costs a full forward pass. Here the click is a *selector*:
the network sees only RGB, emits a fixed tensor of `num_slots` class-agnostic
object masks in one pass, and a click just picks a channel out of that tensor.

What this buys:

- **One forward pass per image, not per click.** Clicks become a lookup into a
  cached tensor, which on CPU is the difference between seconds and instant.
  The click-conditioned model cannot be cached this way: its click channels
  enter at the first convolution, so every layer depends on the click.
- **Ambiguity falls out for free.** Shirt, torso and person are three separate
  ground-truth instances, so they occupy three slots. A click inside all three
  returns all three as candidates, rather than needing a special multi-mask head
  to manufacture alternatives.
- **Negative clicks cost nothing**: drop any slot containing the negative click.

What it costs: the network no longer knows what you are pointing at while it
computes, so it must segment everything well, unconditioned -- a strictly harder
task. And `num_slots` caps how many objects in an image can ever be selected.

Slots are interchangeable and unordered. Nothing says which object belongs in
slot 7, so training pairs predictions to ground truth with Hungarian matching
(see `HungarianMaskLoss`), which is what makes the ordering problem disappear
rather than being learned around.

Both halves are ResNet-50: a pretrained encoder, and a decoder built from the
same bottleneck residual blocks.

Masks are produced at reduced resolution and upsampled by the caller, because
64 slots at full resolution is 1.6 GB of logits for a batch of 32 before
gradients. `mask_stride` sets that trade, and it was chosen by measuring the
IoU ceiling it imposes -- downsample a real ADE20K mask to the grid, upsample
it back, and see what overlap is even reachable:

    stride 4 (96x128)   ceiling 0.92 overall, 0.89 for objects under 1% of frame
    stride 2 (192x256)  ceiling 0.97 overall, 0.96 for the same small objects
                        201 MB and 805 MB of logits respectively, batch 32

Stride 2 is the default. Stride 4 would not have been fatal, but ADE20K's median
object covers 0.45% of the frame, so the small-object column is the relevant
one, and 600 MB is a cheap way to buy back 0.07 of ceiling.

forward returns (mask_logits, objectness):
    mask_logits  (B, num_slots, H/mask_stride, W/mask_stride)
    objectness   (B, num_slots)  -- logit for "this slot holds a real object"
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class Bottleneck(nn.Module):
    """ResNet-50's residual bottleneck: 1x1 squeeze, 3x3 spatial, 1x1 expand.

    The 3x3 -- the expensive part -- runs at a quarter of the block's width, so
    a bottleneck block is far cheaper than two plain 3x3 convs at full width
    while being deeper. The residual add is what lets the decoder be this deep
    without the gradient degrading on the way back up.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = max(out_channels // 4, 8)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Projection shortcut only when the shapes actually differ, matching
        # torchvision's ResNet: an identity shortcut carries gradient better,
        # so it is worth keeping wherever it is legal.
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.shortcut(x))


class SlotUp(nn.Module):
    """One decoder stage: upsample, merge a projected skip, two bottleneck blocks.

    The projection matters. ResNet-50's skips are 256 / 512 / 1024 / 2048
    channels, four times ResNet-34's, and concatenating one of those directly
    would put most of the model's parameters in the decoder's first layer for no
    benefit. A 1x1 conv brings each skip down to a working width first.
    """

    def __init__(self, in_channels: int, skip_channels: int, proj_channels: int,
                 out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.project = nn.Sequential(
            nn.Conv2d(skip_channels, proj_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            Bottleneck(in_channels + proj_channels, out_channels),
            Bottleneck(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.blocks(torch.cat([self.project(skip), x], dim=1))


class ObjectnessHead(nn.Module):
    """Predicts, per slot, whether it holds a real object or is empty.

    With 64 slots and typically ~20 objects in an image, most slots are empty on
    any given example. This head is what lets inference ignore them -- without
    it, every slot would produce some mask and the click selector would have to
    choose between real objects and noise.
    """

    def __init__(self, in_channels: int, num_slots: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_slots),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SlotUNet(nn.Module):
    """ResNet-50 encoder, ResNet-50-style decoder, `num_slots` mask channels.

    The mask head is a 1x1 convolution, which is exactly N learned query vectors
    dot-producted with the per-pixel embedding -- the static form of
    Mask2Former's mask head, without the transformer that refines queries per
    image. Simpler, and enough to test whether the architecture works here.

    Input is RGB in [0, 1]; ImageNet normalisation happens inside the model, so
    callers feed images unchanged exactly as they do for `ResNetUNet`.
    """

    def __init__(
        self,
        num_slots: int = 64,
        pretrained: bool = True,
        mask_stride: int = 2,
        decoder_channels: tuple[int, int, int, int] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        if mask_stride not in (2, 4):
            raise ValueError(f"mask_stride must be 2 or 4, got {mask_stride}")
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # RGB only, so the pretrained stem is used exactly as trained -- none of
        # the 5-channel surgery the click-conditioned model needs.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4,   256
        self.layer2 = backbone.layer2  # /8,   512
        self.layer3 = backbone.layer3  # /16, 1024
        self.layer4 = backbone.layer4  # /32, 2048

        d1, d2, d3, d4 = decoder_channels
        self.up1 = SlotUp(2048, 1024, 256, d1)  # -> /16
        self.up2 = SlotUp(d1, 512, 128, d2)     # -> /8
        self.up3 = SlotUp(d2, 256, 64, d3)      # -> /4
        # The last stage merges the /2 stem features, and only exists at
        # stride 2. Stopping the decoder early is what makes stride 4 cheap.
        self.up4 = SlotUp(d3, 64, 32, d4) if mask_stride == 2 else None

        self.head = nn.Conv2d(d4 if mask_stride == 2 else d3, num_slots, kernel_size=1)
        self.objectness = ObjectnessHead(2048, num_slots)
        self.num_slots = num_slots
        self.mask_stride = mask_stride

        self.register_buffer("rgb_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - self.rgb_mean) / self.rgb_std

        s0 = self.relu(self.bn1(self.conv1(x)))  # /2,   64
        s1 = self.layer1(self.maxpool(s0))       # /4,  256
        s2 = self.layer2(s1)                     # /8,  512
        s3 = self.layer3(s2)                     # /16, 1024
        bottom = self.layer4(s3)                 # /32, 2048

        d = self.up1(bottom, s3)
        d = self.up2(d, s2)
        d = self.up3(d, s1)
        if self.up4 is not None:
            d = self.up4(d, s0)

        return self.head(d), self.objectness(bottom)
