"""Turn a training checkpoint into a self-contained deployment checkpoint.

A training checkpoint carries everything needed to *resume* a run: optimizer
moments (roughly two extra copies of every parameter), scheduler state and the
full epoch history. None of that is needed to predict, and it makes the file
about 3x larger than it has to be -- which matters when the weights are pushed
to a Hugging Face model repo and pulled down by a free-tier Space on every cold
start.

It also carries the opposite problem: a training checkpoint holds only weights,
so loading it requires configs/train.yaml and configs/clicks.yaml to still
describe the run that produced it. The deployment target has no repo checkout
and no guarantee the configs still match. So this script strips what inference
does not need and embeds what it does.

What gets embedded is only the part `detect_arch()` cannot recover from the
weight shapes themselves:

  - data.image_size      -- the resolution the model was trained at
  - the clicks block     -- encoding, radius, max_distance (how a click becomes
                            input channels; a disk of the wrong radius is a
                            different input than the model was trained on)
  - normalization        -- mean/std, and whether the model applies it itself
  - click channel order  -- which of the two trailing channels is positive

Architecture, depth and mask count stay inferred from the weights, so this file
does not go stale when the config changes.

Usage:
    python scripts/export_model.py \
        --checkpoint results/run-multimask/best.pt \
        --output outputs/export/click-segmenter.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.build import build_model, detect_arch
from src.model.unet import migrate_legacy_state_dict

# Bumped when the payload layout changes in a way an older loader cannot read.
FORMAT_VERSION = 1


def resolve_settings(
    checkpoint: dict,
    train_config: dict,
    click_config: dict,
    image_size_override: list[int] | None = None,
) -> tuple[dict, str]:
    """Decide the image size and click settings to embed, and say where from.

    Order of trust, most to least:

    1. `--image-size`, when the operator states it outright.
    2. `train_settings` inside the checkpoint, written by train_full.py. This
       is the only source that is guaranteed to describe *this* checkpoint.
    3. configs/, which describes whatever the repo is set up to train now.

    Case 3 is a real trap and the reason this function reports its source. The
    network is fully convolutional, so it runs at any resolution without
    complaint -- exporting a checkpoint trained at 128x128 while the config
    says 384x512 produces a file that loads, predicts, and is quietly worse,
    with nothing anywhere to indicate why.
    """
    settings = {"image_size": train_config["data"]["image_size"], "clicks": click_config}
    source = "configs/ (NOT recorded in the checkpoint -- verify it is right)"

    recorded = checkpoint.get("train_settings")
    if recorded:
        settings = {"image_size": recorded["image_size"], "clicks": recorded["clicks"]}
        source = "the checkpoint's own train_settings"

    if image_size_override:
        settings["image_size"] = list(image_size_override)
        source = "--image-size (overriding " + source + ")"

    return settings, source


def build_payload(
    checkpoint: dict,
    train_config: dict,
    click_config: dict,
    image_size_override: list[int] | None = None,
) -> dict:
    """Assemble the deployment payload from a loaded training checkpoint."""
    settings, settings_source = resolve_settings(
        checkpoint, train_config, click_config, image_size_override
    )
    state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])
    # From the weights alone, not from train_config["model"] -- the config
    # describes whatever the repo is set up to train *now*, which need not be
    # what produced this checkpoint. (`pretrained: false` in the result means
    # "the weights are already here, do not fetch ImageNet ones", not that the
    # model was trained from scratch.)
    arch_config = detect_arch(state_dict)

    # Normalization is a property of the architecture, not of the config: the
    # ResNet encoder holds mean/std as buffers and normalizes inside forward(),
    # so callers feed [0, 1] either way. Recording it makes that contract
    # explicit for anyone writing a different client against this file.
    if "rgb_mean" in state_dict:
        normalization = {
            "applied": "in_model",
            "mean": state_dict["rgb_mean"].flatten().tolist(),
            "std": state_dict["rgb_std"].flatten().tolist(),
        }
    else:
        normalization = {"applied": "none"}

    return {
        "format_version": FORMAT_VERSION,
        "model_state_dict": state_dict,
        "inference_config": {
            "image_size": list(settings["image_size"]),
            "clicks": dict(settings["clicks"]),
            # Channel order is fixed by src/data/encoding.py; naming it here
            # means a client does not have to read that module to get it right.
            "input_channels": ["red", "green", "blue", "positive_clicks", "negative_clicks"],
            "input_range": [0.0, 1.0],
            "normalization": normalization,
        },
        "provenance": {
            "arch": arch_config,
            "epoch": checkpoint.get("epoch"),
            "best_val_iou": checkpoint.get("best_val_iou"),
            "loss": checkpoint.get("loss"),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "settings_source": settings_source,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/export/click-segmenter.pt")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Resolution the checkpoint was trained at, if configs/ no longer says so",
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="Do not rebuild the model from the exported payload to verify it loads",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]

    payload = build_payload(checkpoint, train_config, click_config, args.image_size)

    source = payload["provenance"]["settings_source"]
    size = payload["inference_config"]["image_size"]
    print(f"input resolution {size[0]}x{size[1]}, from {source}")
    if source.startswith("configs/"):
        print(
            "  WARNING: this checkpoint predates train_settings, so the resolution\n"
            "  above is whatever configs/train.yaml says today. If this checkpoint\n"
            "  was trained at a different size, the export will load and predict\n"
            "  and be silently worse. Pass --image-size H W if you know better."
        )

    # Verify before writing, not after: an export that cannot be loaded is
    # worth catching here rather than on a Space that has already been pushed.
    if not args.skip_load_check:
        model = build_model(payload["provenance"]["arch"])
        model.load_state_dict(payload["model_state_dict"])
        print("load check: state dict matches the architecture detected from it")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    source_mb = Path(args.checkpoint).stat().st_size / 1e6
    output_mb = output.stat().st_size / 1e6
    print(f"{args.checkpoint}  {source_mb:7.1f} MB")
    print(f"{output}  {output_mb:7.1f} MB  ({source_mb / output_mb:.1f}x smaller)")
    print(json.dumps(payload["provenance"], indent=2, default=str))


if __name__ == "__main__":
    main()
