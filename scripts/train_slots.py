"""Train the slot architecture: predict every object, select by click.

The counterpart to train_full.py, kept as a separate entry point on purpose.
The two differ in what an example *is* -- one (image, instance) pair with a
click, versus one image with all its objects -- and threading both through one
loop would tangle the pipeline that produced the current best model for no
benefit. Nothing here touches the click-conditioned path.

    python scripts/train_slots.py --data-root /path/to/ADE --auto-resume

Cluster usage is identical to train_full.py: a 12-hour job will not finish a
run, so long runs chain across jobs and `--auto-resume` restores optimizer,
scheduler, best score and history from `latest.pt`.

**Validation is click-based, and that is the point.** The loss here is a set
loss over all objects, which is not comparable to anything measured before. So
validation simulates a click on each held-out object, runs the same selection
rule the app uses, and reports single-click IoU -- directly comparable to the
click-conditioned model's 0.6194. Early stopping tracks that number, not the
loss, for the same reason the other script does.

Two IoUs are reported per epoch:
    selected  -- what the selection rule actually returns; the headline number
    oracle    -- the best of the candidates it offers, so the gap says how much
                 a "cycle through alternatives" control could recover
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.clicks import sample_positive_click
from src.data.dataset import SlotSegmentationDataset, _normalize_size, slot_collate
from src.data.splits import split_image_paths
from src.inference.slot_predictor import SlotPrediction
from src.model.build import build_model
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.device import get_device
from src.training.losses import HungarianMaskLoss


def build_loader(
    image_paths: list[Path],
    train_config: dict,
    num_slots: int,
    *,
    shuffle: bool,
    split_name: str,
    mask_size: tuple[int, int] | None,
    num_workers: int,
) -> DataLoader:
    """Loader for one split. `mask_size` governs the memory budget -- see below.

    This dataset moves far more data per item than the click one: up to
    `num_slots` masks per image rather than one. An early run was OOM-killed in
    a worker after 15 minutes because full-resolution float32 masks came to
    ~1.6 GB per batch, times the batches every worker keeps prefetched. Masks
    are now bool, training uses the loss's own resolution, and the queue depth
    is shallower.
    """
    dataset = SlotSegmentationDataset(
        image_paths,
        image_size=train_config["data"]["image_size"],
        max_objects=num_slots,
        mask_size=mask_size,
        # Shared with the click dataset, keyed by the same paths and size, so
        # an existing index is reused rather than rebuilt.
        index_cache=Path("outputs") / f"instance_index_{split_name}.json",
    )
    batch_size = train_config["training"]["batch_size"]

    height, width = dataset.mask_size
    peak_gb = batch_size * num_slots * height * width * (num_workers * 2 + 2) / 1e9
    print(
        f"  {split_name}: masks at {height}x{width}, {num_workers} workers, "
        f"worst-case mask memory in flight ~{peak_gb:.1f} GB"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # Object count varies per image, so masks stay a list rather than being
        # padded to a fixed K and masked out again during matching.
        collate_fn=slot_collate,
        # Pinning a batch this large costs more than the transfer it saves.
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_images = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        optimizer.zero_grad()
        mask_logits, objectness = model(images)
        loss = criterion(mask_logits, objectness, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]
        total_images += images.shape[0]

    return total_loss / max(total_images, 1)


@torch.no_grad()
def evaluate_by_click(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Single-click IoU over every held-out object, plus the oracle over candidates.

    One forward pass serves every object in an image, which is the architecture's
    whole advantage showing up in the evaluation as well as the app: the click
    model needs one pass per instance, this needs one per image.

    Clicks are drawn from a fixed seed so the curve reflects the model rather
    than the click sampler, matching how the click model validates.

    Returns (loss, selected IoU, oracle IoU, instances scored).
    """
    model.eval()
    rng = np.random.default_rng(seed)

    total_loss = 0.0
    total_images = 0
    selected_sum = 0.0
    oracle_sum = 0.0
    instances = 0

    for images, targets in loader:
        device_targets = [t.to(device, non_blocking=True) for t in targets]
        mask_logits, objectness = model(images.to(device, non_blocking=True))

        total_loss += criterion(mask_logits, objectness, device_targets).item() * images.shape[0]
        total_images += images.shape[0]

        probs = torch.sigmoid(mask_logits).cpu()
        confidence = torch.sigmoid(objectness).cpu()

        for i, masks in enumerate(targets):
            if masks.shape[0] == 0:
                continue

            height, width = masks.shape[-2:]
            state = SlotPrediction(probs[i], confidence[i], original_size=(height, width))

            for k in range(masks.shape[0]):
                truth = masks[k].numpy() > 0.5
                if not truth.any():
                    continue

                click = sample_positive_click(truth, rng)
                ranked = state.rank([click])
                instances += 1
                if not ranked:
                    continue

                ious = []
                for slot in ranked:
                    predicted = state.mask_for(slot)
                    union = (predicted | truth).sum()
                    ious.append(float((predicted & truth).sum()) / max(int(union), 1))

                selected_sum += ious[0]      # what the rule returns
                oracle_sum += max(ious)      # what cycling alternatives could reach

    return (
        total_loss / max(total_images, 1),
        selected_sum / max(instances, 1),
        oracle_sum / max(instances, 1),
        instances,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset.root")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="0 = all")
    parser.add_argument("--num-slots", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Dataloader workers. Lower than the click model's 16: each item "
        "carries up to num_slots masks, so workers dominate memory here.",
    )
    parser.add_argument("--mask-stride", type=int, default=2, choices=[2, 4])
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Continue from outputs/checkpoints/latest.pt if it exists",
    )
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    training = train_config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["lr"] = args.lr

    torch.manual_seed(train_config["seed"])
    device = get_device(args.device or train_config.get("device", "auto"))
    print(f"Using device: {device}")

    root = Path(args.data_root or data_config["dataset"]["root"]).expanduser()
    image_paths = discover_samples(root)
    print(f"Found {len(image_paths)} images under {root}")

    max_images = args.max_images if args.max_images is not None else training.get("max_images")
    if max_images and len(image_paths) > max_images:
        rng = np.random.default_rng(training["split_seed"])
        keep = sorted(rng.choice(len(image_paths), size=max_images, replace=False))
        image_paths = [image_paths[i] for i in keep]
        print(f"Subsampled to {len(image_paths)} images (max_images={max_images})")
    if not image_paths:
        raise SystemExit(
            f"No images found under {root}.\n"
            f"  HOME is currently {Path.home()}.\n"
            "  On MetaCentrum, $HOME differs between nodes, so a '~' in "
            "configs/data.yaml can resolve to the wrong storage. Pass an "
            "absolute /storage/... path with --data-root."
        )

    splits = split_image_paths(
        image_paths, ratios=tuple(training["splits"]), seed=training["split_seed"]
    )
    for name, paths in splits.items():
        print(f"  {name}: {len(paths)} images")

    # Training targets are only ever compared at the logits' resolution, so
    # carrying them at full size wastes memory the dataloader does not have.
    # Validation keeps full resolution: its IoU is the number reported against
    # the click model's 0.6194, and measuring it on a coarser grid would
    # flatter it.
    image_size = _normalize_size(train_config["data"]["image_size"])
    train_mask_size = (image_size[0] // args.mask_stride, image_size[1] // args.mask_stride)

    train_loader = build_loader(
        splits["train"], train_config, args.num_slots, shuffle=True, split_name="train",
        mask_size=train_mask_size, num_workers=args.num_workers,
    )
    val_loader = build_loader(
        splits["val"], train_config, args.num_slots, shuffle=False, split_name="val",
        mask_size=None, num_workers=max(args.num_workers // 2, 1),
    )
    print(
        f"Images -> train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  (test built later)"
    )

    model_config = {
        "arch": "slot_unet",
        "num_slots": args.num_slots,
        "mask_stride": args.mask_stride,
        "pretrained": True,
    }
    model = build_model(model_config).to(device)
    print(
        f"slot_unet: {args.num_slots} slots, mask stride {args.mask_stride}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    # A single learning rate for now. If the run proves unstable, the standard
    # fix for a set-prediction model on a pretrained backbone is a lower rate
    # for the encoder than the decoder (DETR uses 10x lower).
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    criterion = HungarianMaskLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training["lr_decay_factor"],
        patience=training["lr_decay_patience"],
    )

    checkpoint_dir = Path(train_config["checkpoint"]["dir"])
    history_path = Path(training["history_path"])
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
        checkpoint = load_checkpoint(resume_path, model, optimizer, device, scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_iou = float(checkpoint.get("best_val_iou", float("-inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        print(f"Resumed at epoch {start_epoch} (best {best_val_iou:.4f} at epoch {best_epoch})")

    epochs = training["epochs"]
    patience = training["early_stopping_patience"]

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_oracle, scored = evaluate_by_click(
            model, val_loader, criterion, device
        )
        scheduler.step(val_iou)
        elapsed = time.time() - started
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_oracle": val_oracle,
                "val_instances": scored,
                "lr": current_lr,
                "seconds": elapsed,
            }
        )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        marker = ""
        if val_iou > best_val_iou:
            best_val_iou, best_epoch = val_iou, epoch
            marker = "  <- best"

        state = {
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "history": history,
            "scheduler_state_dict": scheduler.state_dict(),
            "train_settings": {
                "image_size": list(train_config["data"]["image_size"]),
                "model": model_config,
            },
        }
        if marker:
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, val_loss, state)

        print(
            f"epoch {epoch:3d}/{epochs}  train loss={train_loss:.4f}  |  "
            f"val loss={val_loss:.4f} IoU={val_iou:.4f} (oracle {val_oracle:.4f})  "
            f"lr={current_lr:.2e}  ({elapsed:.0f}s){marker}",
            flush=True,
        )
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, train_loss, state)

        if epoch - best_epoch >= patience:
            print(f"Early stopping: no validation improvement in {patience} epochs "
                  f"(best was epoch {best_epoch})")
            break

    print(f"Best validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    if not args.skip_test:
        print("Evaluating best checkpoint on the held-out test split...")
        best = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        test_loader = build_loader(
            splits["test"], train_config, args.num_slots, shuffle=False, split_name="test",
            mask_size=None, num_workers=max(args.num_workers // 2, 1),
        )
        test_loss, test_iou, test_oracle, scored = evaluate_by_click(
            model, test_loader, criterion, device
        )
        print(
            f"Test loss={test_loss:.4f}  Test IoU={test_iou:.4f}  "
            f"(oracle {test_oracle:.4f})  ({scored} instances)"
        )
        with open(history_path, "w") as f:
            json.dump(
                {
                    "history": history,
                    "test": {
                        "loss": test_loss,
                        "iou": test_iou,
                        "oracle": test_oracle,
                        "instances": scored,
                    },
                },
                f,
                indent=2,
            )
    print(f"Wrote history and results to {history_path}")


if __name__ == "__main__":
    main()
