"""Model construction from config, shared by training scripts and the predictor."""

from __future__ import annotations

import torch

from src.model.unet import UNet, migrate_legacy_state_dict


def build_model(model_config: dict) -> torch.nn.Module:
    """Build the architecture named by `model.arch` in configs/train.yaml."""
    arch = model_config.get("arch", "unet")
    num_masks = model_config.get("num_masks", 1)
    # 5 = RGB + positive clicks + negative clicks. 6 adds the model's own
    # previous prediction as an input (RITM's mask guidance), which is what
    # lets a second click *correct* the first mask rather than start over.
    in_channels = model_config.get("in_channels", 5)

    if arch == "unet":
        return UNet(
            in_channels=in_channels,
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
            in_channels=in_channels,
            num_masks=num_masks,
            pretrained=model_config.get("pretrained", True),
        )
    if arch == "slot_unet":
        # Class-agnostic slot segmentation: RGB in, every object out, the click
        # selects afterwards. Takes num_slots rather than num_masks -- the two
        # mean different things (objects in the image vs readings of one click),
        # so they are deliberately not the same key.
        from src.model.slot_unet import SlotUNet

        return SlotUNet(
            num_slots=model_config.get("num_slots", 64),
            pretrained=model_config.get("pretrained", True),
            mask_stride=model_config.get("mask_stride", 2),
        )
    raise ValueError(
        f"unknown model.arch {arch!r}, expected 'unet', 'resnet34_unet' or 'slot_unet'"
    )


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

    if any(key.startswith("objectness.") for key in state_dict):
        # Only the slot model has an objectness head. Its decoder runs one stage
        # further at stride 2, so up4's presence gives the mask resolution back.
        return {
            "arch": "slot_unet",
            "pretrained": False,
            "num_slots": num_masks,
            "mask_stride": 2 if any(key.startswith("up4.") for key in state_dict) else 4,
        }

    if any(key.startswith("layer1.") for key in state_dict):
        # weights come from the checkpoint, so don't re-download ImageNet ones
        return {
            "arch": "resnet34_unet",
            "pretrained": False,
            "num_masks": num_masks,
            "in_channels": int(state_dict["conv1.weight"].shape[1]),
        }

    depth = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("downs."))
    # The stem's first conv is (base_channels, in_channels, 3, 3).
    stem_weight = state_dict["stem.block.0.weight"]
    return {
        "arch": "unet",
        "depth": depth,
        "base_channels": int(stem_weight.shape[0]),
        "num_masks": num_masks,
        "in_channels": int(stem_weight.shape[1]),
    }


def input_conv_key(state_dict: dict) -> str:
    """Name of the first convolution's weight, whichever architecture this is."""
    return "conv1.weight" if "conv1.weight" in state_dict else "stem.block.0.weight"


def expand_input_channels(state_dict: dict, in_channels: int) -> dict:
    """Give a checkpoint's first convolution more input channels, zero-filled.

    This is how a model trained on five channels is fine-tuned with a sixth
    (previous-mask) one: the new channel's filters start at zero, so on the
    first step the network computes exactly what it did before and the new
    input grows in during training -- the same trick the ResNet stem uses for
    the click channels over the pretrained RGB filters. Returns a new dict;
    the input is not modified. A no-op when the count already matches.
    """
    key = input_conv_key(state_dict)
    weight = state_dict[key]
    current = int(weight.shape[1])
    if current == in_channels:
        return state_dict
    if current > in_channels:
        raise ValueError(
            f"checkpoint has {current} input channels, cannot shrink to {in_channels}"
        )
    expanded = weight.new_zeros((weight.shape[0], in_channels, *weight.shape[2:]))
    expanded[:, :current] = weight
    out = dict(state_dict)
    out[key] = expanded
    return out
