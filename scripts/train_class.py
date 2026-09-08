"""One-class UNet: segment every instance of a single ADE20K class (default chair).

Kassem's request of 2026-09-08: "build a UNet (resnet or anything) that
segments one object of your choice, just one object, like a chair", with
"each image with a chair has the corresponding mask (each image has exactly
one mask)" and IoU "computed between the object of interest aka chair and the
mask". This script does exactly that and nothing more:

  - scans the dataset for images containing the class; the union of all its
    instances in an image is that image's one mask (src/data/class_dataset.py)
  - prints the image and instance counts he asked for, per split
  - trains the same ResNet-34 UNet as the click model, but with 3 input
    channels (RGB only) and 1 output mask
  - reports chair-only IoU: per-image IoU averaged over images, plus the
    pixel-pooled IoU over the whole split. Background is never scored.
  - keeps the best-validation checkpoint, evaluates it once on the test split

Run from the repo root:
    python scripts/train_class.py --device cpu --epochs 2 --data-root <tiny>  # dry run
    python scripts/train_class.py --class-name chair                           # real run

On the cluster, through the shared job script:
    qsub -v TRAIN_SCRIPT=scripts/train_class.py,EXTRA_ARGS="--class-name chair" \
         scripts/metacentrum/train.pbs

Everything lands under outputs/class/<class-name>/ so it can never clobber a
click-model run. --auto-resume continues after a walltime kill as usual.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ade20k import discover_samples
from src.data.class_dataset import ClassSegmentationDataset, build_class_index, summarize_index
from src.data.splits import split_image_paths
from src.model.build import build_model
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.device import get_device
from src.training.losses import BCEDiceLoss

EPS = 1e-6


def build_loader(
    entries: list[tuple[Path, list[int]]],
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    augment: bool,
    seed: int,
) -> DataLoader:
    dataset = ClassSegmentationDataset(entries, image_size=image_size, augment=augment, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One pass. Trains if an optimizer is given, else evaluates.

    Returns:
        loss        mean BCE+Dice per image
        iou         chair-only IoU per image, averaged over images that contain
                    the class (an image with no chair has no defined IoU)
        iou_pooled  chair-only IoU with intersection and union summed over all
                    pixels in the split (the "dataset IoU" some papers report)
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    scored = 0
    total_samples = 0
    inter_sum = 0.0
    union_sum = 0.0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            batch_size = inputs.shape[0]

            if is_train:
                optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits.detach()) > 0.5).float()
                intersection = (preds * targets).sum(dim=(1, 2, 3))
                union = ((preds + targets) > 0).float().sum(dim=(1, 2, 3))
                has_target = targets.sum(dim=(1, 2, 3)) > 0
                per_image = intersection / union.clamp_min(EPS)
                total_iou += per_image[has_target].sum().item()
                scored += int(has_target.sum().item())
                inter_sum += intersection.sum().item()
                union_sum += union.sum().item()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "iou": total_iou / max(scored, 1),
        "iou_pooled": inter_sum / max(union_sum, EPS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset.root (absolute path on the cluster)")
    parser.add_argument("--class-name", default="chair", help="ADE20K object name to segment")
    parser.add_argument(
        "--match",
        default="exact",
        choices=["exact", "contains"],
        help="exact: the object's first name is the class ('chair' but not 'armchair'); "
        "contains: any name containing it ('armchair', 'swivel chair', ...)",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.0,
        help="Add this many class-free images per class image to training (0 = only images with the class, "
        "which is what Kassem asked for). Validation and test always use class images only.",
    )
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=0, help="Cap on images scanned (0 = all)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None, help="Default outputs/class/<class-name>")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from <output-dir>/checkpoints/latest.pt if present")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    training = train_config["training"]

    epochs = args.epochs if args.epochs is not None else training["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training["batch_size"]
    lr = args.lr if args.lr is not None else training["lr"]
    num_workers = args.num_workers if args.num_workers is not None else training["num_workers"]
    image_size = tuple(train_config["data"]["image_size"])
    seed = train_config["seed"]
    class_name = args.class_name.strip().lower()

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "class" / class_name.replace(" ", "_")
    checkpoint_dir = output_dir / "checkpoints"
    history_path = output_dir / "history.json"
    counts_path = output_dir / "counts.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    device = get_device(args.device or train_config.get("device", "auto"))
    print(f"Using device: {device}")
    print(f"Class: {class_name!r} (match={args.match})  output: {output_dir}")

    root = Path(args.data_root or data_config["dataset"]["root"]).expanduser()
    image_paths = discover_samples(root)
    print(f"Found {len(image_paths)} images under {root}")
    if not image_paths:
        raise SystemExit(f"No images found under {root}")
    if args.max_images and len(image_paths) > args.max_images:
        rng = np.random.default_rng(training["split_seed"])
        keep = sorted(rng.choice(len(image_paths), size=args.max_images, replace=False))
        image_paths = [image_paths[i] for i in keep]
        print(f"Subsampled to {len(image_paths)} images (max_images={args.max_images})")

    # Which images contain the class, and which instances. Cached per class.
    started = time.time()
    index = build_class_index(
        image_paths,
        class_name,
        match=args.match,
        cache_path=output_dir / f"index_{args.match}.json",
        workers=max(num_workers, 4),
    )
    totals = summarize_index(index)
    print(
        f"Scanned {totals['images_scanned']} images in {time.time() - started:.0f}s: "
        f"{totals['images_with_class']} contain a {class_name} "
        f"({totals['instances']} instances), {totals['images_without_class']} do not"
    )
    if totals["images_with_class"] < 3:
        raise SystemExit(f"Only {totals['images_with_class']} images contain {class_name!r}; nothing to train on")

    # Split BY IMAGE over the images that contain the class, same seed and
    # ratios as every other run, so the numbers are comparable.
    positives = sorted(Path(p) for p, ids in index.items() if ids)
    splits = split_image_paths(positives, ratios=tuple(training["splits"]), seed=training["split_seed"])
    entries = {name: [(p, index[str(p)]) for p in paths] for name, paths in splits.items()}

    counts = {"class_name": class_name, "match": args.match, **totals, "splits": {}}
    for name, items in entries.items():
        counts["splits"][name] = {
            "images": len(items),
            "instances": sum(len(ids) for _, ids in items),
        }
        print(f"  {name}: {len(items)} images, {counts['splits'][name]['instances']} instances")

    if args.negative_ratio > 0:
        negatives = sorted(Path(p) for p, ids in index.items() if not ids)
        rng = np.random.default_rng(training["split_seed"])
        take = min(len(negatives), int(round(args.negative_ratio * len(entries["train"]))))
        chosen = rng.choice(len(negatives), size=take, replace=False)
        entries["train"] += [(negatives[i], []) for i in sorted(chosen)]
        counts["splits"]["train"]["negatives"] = take
        print(f"  + {take} negative images (no {class_name}) added to train only")

    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Counts written to {counts_path}")

    train_loader = build_loader(entries["train"], image_size, batch_size, num_workers, shuffle=True, augment=True, seed=seed)
    val_loader = build_loader(entries["val"], image_size, batch_size, num_workers, shuffle=False, augment=False, seed=seed)

    model_config = dict(train_config["model"])
    model_config.update({"arch": "resnet34_unet", "in_channels": 3, "num_masks": 1})
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=training["lr_decay_factor"], patience=training["lr_decay_patience"]
    )

    history: list[dict] = []
    best_val_iou = float("-inf")
    best_epoch = 0
    start_epoch = 1

    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        candidate = checkpoint_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate
            print(f"Auto-resuming from {candidate}")
        else:
            print("No checkpoint found, starting from scratch")
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer, device, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", float("-inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(f"Resumed at epoch {start_epoch} (best val IoU {best_val_iou:.4f} at epoch {best_epoch})")

    patience = training["early_stopping_patience"]
    print(f"Training {class_name}: {epochs} epochs max, batch {batch_size}, lr {lr}, image {image_size}, patience {patience}")

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, criterion, device, optimizer)
        val = run_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - started

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val["iou"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_iou": train["iou"],
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_iou_pooled": val["iou_pooled"],
                "lr": lr_now,
                "seconds": elapsed,
            }
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        marker = ""
        if val["iou"] > best_val_iou:
            best_val_iou, best_epoch = val["iou"], epoch
            marker = "  <- best"

        state = {
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "history": history,
            "scheduler_state_dict": scheduler.state_dict(),
            "train_settings": {
                "task": "class_segmentation",
                "class_name": class_name,
                "match": args.match,
                "image_size": list(image_size),
                "in_channels": 3,
                "max_images": int(args.max_images or 0),
                "negative_ratio": args.negative_ratio,
            },
        }
        if marker:
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, val["loss"], state)
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train loss={train['loss']:.4f} IoU={train['iou']:.4f}  |  "
            f"val loss={val['loss']:.4f} IoU={val['iou']:.4f} pooled={val['iou_pooled']:.4f}"
            f"  lr={lr_now:.2e}  ({elapsed:.0f}s){marker}"
        )
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, train["loss"], state)

        if epoch - best_epoch >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs (best {best_val_iou:.4f} at epoch {best_epoch})")
            break

    print(f"Best val IoU {best_val_iou:.4f} at epoch {best_epoch}; checkpoint {checkpoint_dir / 'best.pt'}")

    if args.skip_test:
        return
    best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    test_loader = build_loader(entries["test"], image_size, batch_size, num_workers, shuffle=False, augment=False, seed=seed)
    test = run_epoch(model, test_loader, criterion, device)
    print(
        f"TEST ({class_name}, {len(entries['test'])} images): "
        f"IoU={test['iou']:.4f} pooled={test['iou_pooled']:.4f} loss={test['loss']:.4f}"
    )
    counts["test"] = {"iou": test["iou"], "iou_pooled": test["iou_pooled"], "best_epoch": best_epoch, "best_val_iou": best_val_iou}
    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=2)


if __name__ == "__main__":
    main()
