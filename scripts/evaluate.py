"""Measure a checkpoint the way interactive segmenters are actually judged.

Everything reported so far is the IoU of one click. That is not the field's
metric, and it is not how the tool is used: a person clicks, looks, and clicks
again where the mask is wrong. This script runs that loop with a simulated user
(the RITM / SimpleClick protocol, see src/training/interaction.py) and reports:

    mIoU@k      mean IoU after k clicks, for k = 1 .. --max-clicks
    NoC@80/85/90  mean number of clicks to reach that IoU, capped at
                  --max-clicks (instances that never reach it count as the cap)
    >= cap        fraction of instances that never reached the target
    per size      the same numbers by object size, because the median ADE20K
                  object covers under half a percent of the frame and that is
                  where single-click IoU is lost

It also lets inference-time options be measured before they go near the app:
    --selection consistent   drop candidates that contradict the clicks
    --flip-tta               average with the horizontally mirrored prediction
    --threshold              the probability cut-off (0.5 in training)

Run from the repo root, on a machine that can see the dataset:

    python scripts/evaluate.py --checkpoint results/run-7/best.pt \
        --data-root /storage/brno2/home/$USER/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE \
        --split val --max-instances 5000 --max-clicks 20 \
        --output outputs/eval/run7-val.json

The split is rebuilt exactly as training built it: the same image list, the
same subsampling, the same seed, the same fixed validation clicks. Pass
--max-images if the checkpoint predates that being recorded; getting it wrong
silently evaluates on images the model trained on. Use --split test once, on
the final configuration, and never to choose between options.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.dataset import ClickSegmentationDataset
from src.data.splits import split_image_paths
from src.model.build import build_model, detect_arch
from src.model.unet import migrate_legacy_state_dict
from src.training.device import get_device
from src.training.interaction import clicks_to_reach, run_interaction

# Object size as a fraction of the frame. The boundaries follow the dataset
# measurement in docs/TECHNICAL-RECORD.md: the median object is ~0.45%.
SIZE_BUCKETS = [
    ("tiny  <0.5%", 0.0, 0.005),
    ("small 0.5-2%", 0.005, 0.02),
    ("medium 2-10%", 0.02, 0.10),
    ("large >10%", 0.10, 1.01),
]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    """Rebuild the model from its weights; return (model, arch, recorded train settings)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = migrate_legacy_state_dict(checkpoint["model_state_dict"])
    arch = detect_arch(state_dict)
    if arch["arch"] == "slot_unet":
        raise SystemExit("evaluate.py measures the click-conditioned model; use train_slots.py's validation for the slot model")
    model = build_model(arch).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    settings = checkpoint.get("train_settings") or {}
    inference = checkpoint.get("inference_config")
    if not settings and inference:
        settings = {"image_size": inference["image_size"], "clicks": inference["clicks"]}
    provenance = checkpoint.get("provenance", checkpoint)
    settings.setdefault("epoch", provenance.get("epoch"))
    settings.setdefault("best_val_iou", provenance.get("best_val_iou"))
    return model, arch, settings


def build_split_loader(
    args: argparse.Namespace, train_config: dict, click_config: dict, image_size, max_images: int
) -> tuple[DataLoader, int]:
    """The requested split, rebuilt exactly as train_full.py built it."""
    root = Path(args.data_root).expanduser()
    image_paths = discover_samples(root)
    if not image_paths:
        raise SystemExit(f"No images found under {root}")
    print(f"Found {len(image_paths)} images under {root}")

    training = train_config["training"]
    if max_images and len(image_paths) > max_images:
        rng = np.random.default_rng(training["split_seed"])
        keep = sorted(rng.choice(len(image_paths), size=max_images, replace=False))
        image_paths = [image_paths[i] for i in keep]
        print(f"Subsampled to {len(image_paths)} images (max_images={max_images})")

    splits = split_image_paths(image_paths, ratios=tuple(training["splits"]), seed=training["split_seed"])
    paths = splits[args.split]
    print(f"{args.split} split: {len(paths)} images")

    dataset = ClickSegmentationDataset(
        paths,
        image_size=image_size,
        click_config=click_config,
        deterministic=True,  # the same first click training validated with
        lazy=True,
        index_cache=Path("outputs") / f"instance_index_{args.split}.json",
    )
    total = len(dataset)
    if args.max_instances and total > args.max_instances:
        rng = np.random.default_rng(args.subsample_seed)
        chosen = np.sort(rng.choice(total, size=args.max_instances, replace=False))
        dataset = Subset(dataset, chosen.tolist())
        print(f"Evaluating {len(dataset)} of {total} instances (--max-instances, seed {args.subsample_seed})")
    else:
        print(f"Evaluating all {total} instances")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return loader, total


def summarise(iou: np.ndarray, oracle: np.ndarray, area: np.ndarray, max_clicks: int, targets: list[float]) -> dict:
    """Turn per-instance (N, K) IoU curves into the reported numbers."""
    iou_t = torch.from_numpy(iou)
    summary: dict = {
        "instances": int(iou.shape[0]),
        "max_clicks": max_clicks,
        "miou_at_k": [float(v) for v in iou.mean(axis=0)],
        "oracle_miou_at_k": [float(v) for v in oracle.mean(axis=0)],
        "noc": {},
        "by_size": [],
    }
    for target in targets:
        noc = clicks_to_reach(iou_t, target, max_clicks)
        summary["noc"][f"{int(round(target * 100))}"] = {
            "mean_clicks": float(noc.float().mean()),
            "never_reached_fraction": float((iou_t.max(dim=1).values < target).float().mean()),
        }
    for name, low, high in SIZE_BUCKETS:
        member = (area >= low) & (area < high)
        if not member.any():
            continue
        bucket_iou = iou[member]
        entry = {
            "bucket": name,
            "count": int(member.sum()),
            "miou_at_1": float(bucket_iou[:, 0].mean()),
            "miou_at_k": float(bucket_iou[:, -1].mean()),
        }
        for target in targets:
            noc = clicks_to_reach(torch.from_numpy(bucket_iou), target, max_clicks)
            entry[f"noc_{int(round(target * 100))}"] = float(noc.float().mean())
        summary["by_size"].append(entry)
    return summary


def print_report(summary: dict, label: str) -> None:
    k = summary["max_clicks"]
    miou = summary["miou_at_k"]
    oracle = summary["oracle_miou_at_k"]
    print(f"\n== {label}: {summary['instances']} instances, up to {k} clicks ==\n")

    shown = [c for c in (1, 2, 3, 5, 10, 15, 20) if c <= k]
    if k not in shown:
        shown.append(k)
    print("| clicks | " + " | ".join(str(c) for c in shown) + " |")
    print("| --- | " + " | ".join("---" for _ in shown) + " |")
    print("| mIoU | " + " | ".join(f"{miou[c - 1]:.4f}" for c in shown) + " |")
    print("| oracle | " + " | ".join(f"{oracle[c - 1]:.4f}" for c in shown) + " |")

    print(f"\n| target | NoC (cap {k}) | never reached |")
    print("| --- | --- | --- |")
    for target, entry in summary["noc"].items():
        print(f"| IoU {target}% | {entry['mean_clicks']:.2f} | {100 * entry['never_reached_fraction']:.1f}% |")

    if summary["by_size"]:
        print(f"\n| object size | count | mIoU@1 | mIoU@{k} | NoC@85 | NoC@90 |")
        print("| --- | --- | --- | --- | --- | --- |")
        for entry in summary["by_size"]:
            noc85 = entry.get("noc_85", float("nan"))
            noc90 = entry.get("noc_90", float("nan"))
            print(
                f"| {entry['bucket']} | {entry['count']} | {entry['miou_at_1']:.4f} | "
                f"{entry['miou_at_k']:.4f} | {noc85:.2f} | {noc90:.2f} |"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Absolute path to .../images/ADE")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-clicks", type=int, default=20)
    parser.add_argument("--noc-targets", type=float, nargs="+", default=[0.80, 0.85, 0.90])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--selection", default="score", choices=["score", "consistent"])
    parser.add_argument("--flip-tta", action="store_true")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="training.max_images the checkpoint was trained with (0 = all). Read from the "
        "checkpoint when recorded; REQUIRED to match the run, or the split leaks training images.",
    )
    parser.add_argument("--max-instances", type=int, default=5000, help="0 = every instance in the split")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default=None, help="Where to write the JSON report (+ per-instance curves)")
    parser.add_argument("--label", default=None, help="Name for the report; defaults to the checkpoint path")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]

    model, arch, settings = load_model(Path(args.checkpoint), device)
    image_size = settings.get("image_size", train_config["data"]["image_size"])
    clicks = settings.get("clicks", click_config)
    if clicks.get("encoding", "disk") != "disk":
        raise SystemExit("the interaction protocol draws disk clicks; this checkpoint uses another encoding")
    radius = int(clicks.get("radius", 5))
    in_channels = arch.get("in_channels", 5)
    print(
        f"Checkpoint: {arch['arch']}, {arch['num_masks']} candidate mask(s), {in_channels} input "
        f"channels, {image_size[0]}x{image_size[1]}, epoch {settings.get('epoch')}, "
        f"val IoU {settings.get('best_val_iou')}"
    )

    recorded_max_images = settings.get("max_images")
    if args.max_images is not None:
        max_images = args.max_images
    elif recorded_max_images is not None:
        max_images = recorded_max_images
        print(f"max_images={max_images} read from the checkpoint")
    else:
        max_images = 0
        print(
            "WARNING: this checkpoint does not record training.max_images and --max-images was not "
            "given; assuming 0 (all images). Runs 1-4 and 6 used 3000 -- pass --max-images 3000 for those."
        )

    loader, total = build_split_loader(args, train_config, clicks, image_size, max_images)

    all_iou, all_oracle, all_area = [], [], []
    started = time.time()
    for n, (inputs, targets) in enumerate(loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        result = run_interaction(
            model,
            inputs,
            targets,
            args.max_clicks,
            in_channels=in_channels,
            radius=radius,
            threshold=args.threshold,
            selection=args.selection,
            flip_tta=args.flip_tta,
            click_stride=1,
        )
        all_iou.append(result["iou"].cpu())
        all_oracle.append(result["oracle"].cpu())
        all_area.append(targets.mean(dim=(1, 2, 3)).cpu())
        if n % 10 == 0 or n == len(loader):
            so_far = torch.cat(all_iou)
            print(
                f"  batch {n}/{len(loader)}  mIoU@1 {so_far[:, 0].mean():.4f}  "
                f"mIoU@{args.max_clicks} {so_far[:, -1].mean():.4f}  ({time.time() - started:.0f}s)"
            )

    iou = torch.cat(all_iou).numpy()
    oracle = torch.cat(all_oracle).numpy()
    area = torch.cat(all_area).numpy()

    summary = summarise(iou, oracle, area, args.max_clicks, args.noc_targets)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "split_instances_total": total,
            "arch": arch,
            "image_size": list(image_size),
            "options": {
                "threshold": args.threshold,
                "selection": args.selection,
                "flip_tta": args.flip_tta,
                "max_images": max_images,
                "max_instances": args.max_instances,
                "subsample_seed": args.subsample_seed,
            },
            "seconds": time.time() - started,
        }
    )
    if args.label:
        label = args.label
    else:
        options = [args.split, args.selection] + (["flip-tta"] if args.flip_tta else [])
        label = f"{Path(args.checkpoint)} [{', '.join(options)}]"
    print_report(summary, label)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        # Per-instance curves alongside, for the notebook: iou (N, K), oracle
        # (N, K), area (N,). Not JSON, it would be tens of MB on the full split.
        curves = output.with_suffix(".npz")
        np.savez_compressed(curves, iou=iou, oracle=oracle, area=area)
        print(f"Wrote {output} and {curves}")


if __name__ == "__main__":
    main()
