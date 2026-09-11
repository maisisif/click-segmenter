"""Turn a train_class.py checkpoint into a self-contained deployment checkpoint.

Same idea as scripts/export_model.py for the click model: the training file
carries optimizer moments and the epoch history that resuming needs and
inference does not, roughly tripling its size. This strips them, records the
class names and input size under `inference_config`, and copies the run's
numbers (best epoch, validation IoU, test IoU from counts.json when it sits
next to the checkpoint) under `provenance` so the interface can show them.

    python scripts/export_class_model.py \\
        --checkpoint results/class/bed/checkpoints/best.pt \\
        --output weights/class-bed.pt --half

`--half` stores the weights in float16 (about half the size of float32). They
are cast back to float32 on load; the effect on the masks is negligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.build import detect_arch
from src.model.unet import migrate_legacy_state_dict

FORMAT_VERSION = 1


def build_payload(checkpoint: dict, counts: dict | None = None, half: bool = False, source: str = "") -> dict:
    settings = checkpoint.get("inference_config") or checkpoint["train_settings"]
    state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])
    if half:
        state_dict = {
            k: (v.half() if torch.is_tensor(v) and v.is_floating_point() and k not in ("rgb_mean", "rgb_std") else v)
            for k, v in state_dict.items()
        }
    arch = detect_arch(state_dict)

    provenance = checkpoint.get("provenance") or {
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_iou": checkpoint.get("best_val_iou"),
    }
    if counts is not None:
        provenance["test"] = counts.get("test")
        provenance["images_with_class"] = counts.get("images_with_class")
        provenance["splits"] = counts.get("splits")

    return {
        "format_version": FORMAT_VERSION,
        "kind": "class_segmentation",
        "model_state_dict": state_dict,
        "architecture": arch,
        "inference_config": {
            "task": "class_segmentation",
            "class_names": list(settings["class_names"]),
            "match": settings.get("match", "exact"),
            "image_size": list(settings["image_size"]),
            "input": "RGB in [0, 1]; ImageNet normalisation is applied inside the model",
            "output": "one sigmoid logit map per class, in class_names order",
        },
        "provenance": provenance,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "weights_dtype": "float16" if half else "float32",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", default=None, help="counts.json of the run (default: ../counts.json next to the checkpoint)")
    parser.add_argument("--half", action="store_true", help="store float16 weights")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    counts_path = Path(args.counts) if args.counts else checkpoint_path.parent.parent / "counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else None
    if counts is None:
        print(f"no counts.json at {counts_path}; provenance will lack test numbers")

    payload = build_payload(checkpoint, counts, half=args.half, source=str(checkpoint_path))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    before = checkpoint_path.stat().st_size / 1e6
    after = output.stat().st_size / 1e6
    cfg = payload["inference_config"]
    print(f"classes {cfg['class_names']}  input {cfg['image_size']}  arch {payload['architecture']['arch']}")
    print(f"provenance {json.dumps(payload['provenance'], default=str)[:300]}")
    print(f"{checkpoint_path} ({before:.1f} MB) -> {output} ({after:.1f} MB)")


if __name__ == "__main__":
    main()
