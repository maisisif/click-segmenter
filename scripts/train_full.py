"""M4: real training on ADE20K with train/validation/test splits.

Distinct from `scripts/train.py`, which only runs the M3 overfit sanity check on
a cherry-picked handful of instances. This is the actual training run:

  - splits images 70/20/10 (see `src/data/splits.py` for why it's by image)
  - trains on the train split with a freshly simulated click each epoch,
    which doubles as augmentation
  - validates every epoch with *fixed* clicks, so the val curve reflects model
    changes rather than click sampling noise
  - keeps the checkpoint from the best validation epoch, not the last one
  - evaluates that best checkpoint once on the held-out test split at the end
  - writes the full per-epoch history to JSON for plotting in the notebook

Run from the repo root:
    python scripts/train_full.py                      # uses configs/train.yaml
    python scripts/train_full.py --device cpu --epochs 2   # quick dry run

The test split is touched exactly once, at the very end. Peeking at it during
development is how you end up reporting a number that means nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.dataset import ClickSegmentationDataset
from src.data.splits import split_image_paths
from src.model.unet import UNet
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.device import get_device
from src.training.losses import BCEDiceLoss
from src.training.metrics import iou_score


def build_loader(
    image_paths: list[Path],
    train_config: dict,
    click_config: dict,
    *,
    shuffle: bool,
    deterministic: bool,
) -> DataLoader:
    dataset = ClickSegmentationDataset(
        image_paths,
        image_size=train_config["data"]["image_size"],
        click_config=click_config,
        deterministic=deterministic,
        lazy=True,  # the full dataset can't be held in memory
    )
    return DataLoader(
        dataset,
        batch_size=train_config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=train_config["training"]["num_workers"],
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains if an optimizer is given, else evaluates.

    Returns (mean loss, mean IoU), both averaged per sample rather than per
    batch so a smaller final batch doesn't skew the numbers.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]

            if is_train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size
            total_iou += iou_score(logits.detach(), targets) * batch_size
            total_samples += batch_size

    return total_loss / total_samples, total_iou / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset.root. Use an absolute /storage/... path in batch jobs, "
        "where a '~' in the config resolves against the wrong node's HOME.",
    )
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override training.lr")
    parser.add_argument("--base-channels", type=int, default=None, help="Override model.base_channels")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.pt) to resume from")
    parser.add_argument("--skip-test", action="store_true", help="Skip the final test-set evaluation")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    if args.base_channels is not None:
        train_config["model"]["base_channels"] = args.base_channels

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
    if not image_paths:
        raise SystemExit(
            f"No images found under {root}.\n"
            f"  HOME is currently {Path.home()}.\n"
            "  On MetaCentrum, $HOME differs between nodes, so a '~' in "
            "configs/data.yaml can resolve to the wrong storage. Batch jobs "
            "should set HOME explicitly (see scripts/metacentrum/train.pbs) or "
            "use an absolute /storage/... path."
        )

    splits = split_image_paths(
        image_paths,
        ratios=tuple(training["splits"]),
        seed=training["split_seed"],
    )
    for name, paths in splits.items():
        print(f"  {name}: {len(paths)} images")

    # Validation and test use fixed clicks so their curves reflect the model,
    # not the click sampler. Training resamples every epoch as augmentation.
    train_loader = build_loader(splits["train"], train_config, click_config, shuffle=True, deterministic=False)
    val_loader = build_loader(splits["val"], train_config, click_config, shuffle=False, deterministic=True)
    print(
        f"Instances -> train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  (test built later)"
    )

    model = UNet(in_channels=5, out_channels=1, base_channels=train_config["model"]["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    criterion = BCEDiceLoss()

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device) + 1
        print(f"Resumed from {args.resume!r} at epoch {start_epoch}")

    checkpoint_dir = Path(train_config["checkpoint"]["dir"])
    history_path = Path(training["history_path"])
    history: list[dict] = []
    best_val_iou = float("-inf")
    best_epoch = 0

    epochs = training["epochs"]
    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - started

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "seconds": elapsed,
            }
        )
        # Written every epoch so a job that dies or hits its walltime still
        # leaves usable curves behind.
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        marker = ""
        if val_iou > best_val_iou:
            best_val_iou, best_epoch = val_iou, epoch
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, val_loss)
            marker = "  <- best"

        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train loss={train_loss:.4f} IoU={train_iou:.4f}  |  "
            f"val loss={val_loss:.4f} IoU={val_iou:.4f}  ({elapsed:.0f}s){marker}"
        )
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, train_loss)

    print(f"\nBest validation IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    if args.skip_test:
        print("Skipping test evaluation (--skip-test).")
        return

    # Load the best-validation weights before touching the test set: reporting
    # the *last* epoch's weights would be reporting a model we never selected.
    print("Evaluating best checkpoint on the held-out test split...")
    load_checkpoint(str(checkpoint_dir / "best.pt"), model, optimizer, device)
    test_loader = build_loader(splits["test"], train_config, click_config, shuffle=False, deterministic=True)
    test_loss, test_iou = run_epoch(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test IoU={test_iou:.4f}  ({len(test_loader.dataset)} instances)")

    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "test_loss": test_loss,
        "test_iou": test_iou,
        "splits": {name: len(paths) for name, paths in splits.items()},
        "history": history,
    }
    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote history and results to {history_path}")


if __name__ == "__main__":
    main()
