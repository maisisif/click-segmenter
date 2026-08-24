"""Model construction from config, shared by training scripts and the predictor."""

from __future__ import annotations

import torch

from src.model.unet import UNet, migrate_legacy_state_dict


def build_model(model_config: dict) -> torch.nn.Module:
    """Build the architecture named by `model.arch` in configs/train.yaml."""
    arch = model_config.get("arch", "unet")
    num_masks = model_config.get("num_masks", 1)

    if arch == "unet":
        return UNet(
            in_channels=5,
            num_masks=num_masks,
            base_channels=model_config["base_channels"],
            depth=model_config.get("depth", 3),
        )
    if arch == "resnet34_unet":
        # Imported lazily: torchvision is only needed for this arch, and the
        # pretrained weights download on first use (cache them on a node with
        # internet -- see scripts/metacentrum/train.pbs).
        from src.model.resnet_unet import ResNetUNet

        return ResNetUNet(
            in_channels=5,
            num_masks=num_masks,
            pretrained=model_config.get("pretrained", True),
        )
    raise ValueError(f"unknown model.arch {arch!r}, expected 'unet' or 'resnet34_unet'")


def detect_arch(state_dict: dict) -> dict:
    """Infer the full architecture from a checkpoint's weight shapes.

    Lets the app and notebook load any checkpoint without the config having to
    match the era it was saved in. The returned dict is complete enough to pass
    straight to `build_model` -- which is what makes an exported deployment
    checkpoint self-contained, with no configs/ checkout on the serving side.
    """
    state_dict = migrate_legacy_state_dict(state_dict)

    # head.weight is (num_masks, C, 1, 1) for both architectures.
    num_masks = int(state_dict["head.weight"].shape[0])

    if any(key.startswith("layer1.") for key in state_dict):
        # weights come from the checkpoint, so don't re-download ImageNet ones
        return {"arch": "resnet34_unet", "pretrained": False, "num_masks": num_masks}

    depth = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("downs."))
    # The stem's first conv is (base_channels, in_channels, 3, 3).
    base_channels = int(state_dict["stem.block.0.weight"].shape[0])
    return {
        "arch": "unet",
        "depth": depth,
        "base_channels": base_channels,
        "num_masks": num_masks,
    }
