"""Model construction from config, shared by training scripts and the predictor."""

from __future__ import annotations

import torch

from src.model.unet import UNet, migrate_legacy_state_dict


def build_model(model_config: dict) -> torch.nn.Module:
    """Build the architecture named by `model.arch` in configs/train.yaml."""
    arch = model_config.get("arch", "unet")
    if arch == "unet":
        return UNet(
            in_channels=5,
            out_channels=1,
            base_channels=model_config["base_channels"],
            depth=model_config.get("depth", 3),
        )
    if arch == "resnet34_unet":
        # Imported lazily: torchvision is only needed for this arch, and the
        # pretrained weights download on first use (cache them on a node with
        # internet -- see scripts/metacentrum/train.pbs).
        from src.model.resnet_unet import ResNetUNet

        return ResNetUNet(in_channels=5, out_channels=1, pretrained=model_config.get("pretrained", True))
    raise ValueError(f"unknown model.arch {arch!r}, expected 'unet' or 'resnet34_unet'")


def detect_arch(state_dict: dict) -> dict:
    """Infer which architecture (and depth) a checkpoint was trained with.

    Lets the app and notebook load any checkpoint without the config having to
    match the era it was saved in.
    """
    state_dict = migrate_legacy_state_dict(state_dict)
    if any(key.startswith("layer1.") for key in state_dict):
        return {"arch": "resnet34_unet", "pretrained": False}  # weights come from the checkpoint
    depth = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("downs."))
    return {"arch": "unet", "depth": depth}
