"""M3: baseline UNet + overfit sanity check.

Before investing in full-scale training, we deliberately try to overfit a
tiny, fixed subset of examples (same simulated clicks every epoch) to
near-zero loss. If the model can't memorize a handful of examples, something
in the data pipeline, loss, or training loop is broken — better to find that
out locally in a minute than after a multi-hour MetaCentrum job.

Run from the repo root:
    python scripts/train.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.dataset import ClickSegmentationDataset
from src.model.build import build_model
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.device import get_device
from src.training.losses import MultiMaskLoss
from src.training.metrics import iou_score, select_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--data-root", default=None, help="Override dataset.root (see train_full.py)")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output", default="outputs/overfit_check.png")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.pt) to resume training from")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Override the device set in the train config (useful for a quick CPU check on an incompatible GPU node)",
    )
    # Overrides for the overfit settings, so diagnostic runs don't require
    # editing (and then remembering to revert) configs/train.yaml. The config
    # stays the canonical M3 setting; these are for one-off experiments.
    parser.add_argument("--subset-size", type=int, default=None, help="Override overfit.subset_size")
    parser.add_argument("--epochs", type=int, default=None, help="Override overfit.epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override overfit.lr")
    parser.add_argument("--base-channels", type=int, default=None, help="Override model.base_channels")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    if args.base_channels is not None:
        train_config["model"]["base_channels"] = args.base_channels

    overfit_config = train_config["overfit"]
    if args.subset_size is not None:
        overfit_config["subset_size"] = args.subset_size
    if args.epochs is not None:
        overfit_config["epochs"] = args.epochs
    if args.lr is not None:
        overfit_config["lr"] = args.lr

    torch.manual_seed(train_config["seed"])
    device = get_device(args.device or train_config.get("device", "auto"))
    print(f"Using device: {device}")
    print(
        f"Overfit settings: subset_size={overfit_config['subset_size']} "
        f"epochs={overfit_config['epochs']} lr={overfit_config['lr']} "
        f"base_channels={train_config['model']['base_channels']}"
    )

    root = Path(args.data_root or data_config["dataset"]["root"]).expanduser()
    image_paths = discover_samples(root)
    dataset = ClickSegmentationDataset(
        image_paths,
        image_size=train_config["data"]["image_size"],
        click_config=click_config,
        deterministic=True,  # fixed clicks per item: we're testing pure memorization
    )
    print(f"Dataset has {len(dataset)} (image, instance) pairs")

    # Pick the largest-area instances for the overfit subset. This check only
    # needs *some* examples the model can trivially memorize fast — tiny/thin
    # objects (a few pixels wide at 128x128) make the sanity check itself slow
    # and noisy without indicating anything wrong with the pipeline. Real
    # training (M4) must not cherry-pick like this.
    subset_size = overfit_config["subset_size"]
    largest_indices = sorted(range(len(dataset)), key=lambda i: dataset.mask_areas[i], reverse=True)
    subset_indices = largest_indices[:subset_size]
    for i in subset_indices:
        _, instance = dataset.items[i]
        print(f"  overfit subset: {instance.name!r} (area={dataset.mask_areas[i]} px)")

    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

    model = build_model(train_config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=overfit_config["lr"])
    criterion = MultiMaskLoss()

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)["epoch"] + 1
        print(f"Resumed from {args.resume!r} at epoch {start_epoch}")

    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        initial_logits, initial_scores = model(inputs)
        initial_pred = torch.sigmoid(select_masks(initial_logits, initial_scores))

    model.train()
    epochs = overfit_config["epochs"]
    log_every = overfit_config["log_every"]
    checkpoint_dir = Path(train_config["checkpoint"]["dir"])
    save_every = train_config["checkpoint"]["save_every"]

    # Track the best weights separately from the latest. Adam at this lr
    # occasionally destabilizes for an epoch or two, and BatchNorm makes the
    # damage worse than it looks: a spike pollutes the running statistics, so
    # eval-mode output degrades even further than the training loss suggests.
    # Without this, a spike on the final epoch throws away an otherwise good
    # run — cheap here, very expensive on a multi-hour M4 job.
    best_iou = float("-inf")
    best_epoch = 0

    for epoch in range(start_epoch, epochs + 1):
        optimizer.zero_grad()
        logits, scores = model(inputs)
        loss = criterion(logits, targets, scores)
        loss.backward()
        optimizer.step()

        epoch_iou = iou_score(logits.detach(), targets, scores=scores.detach() if scores is not None else None)
        if epoch_iou > best_iou:
            best_iou, best_epoch = epoch_iou, epoch
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, loss.item())

        if epoch % log_every == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  IoU={epoch_iou:.4f}")

        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, loss.item())

    model.eval()
    with torch.no_grad():
        final_logits, final_scores = model(inputs)
        final_pred = torch.sigmoid(select_masks(final_logits, final_scores))
    final_iou = iou_score(final_logits, targets, scores=final_scores)
    print(f"Final overfit IoU on the {len(subset)}-example subset: {final_iou:.4f}")
    print(f"Best training IoU: {best_iou:.4f} (epoch {best_epoch}, saved to {checkpoint_dir / 'best.pt'})")
    if final_iou < best_iou - 0.05:
        print(
            "NOTE: final IoU is well below the best seen — training likely spiked late. "
            "Use best.pt rather than latest.pt, and consider lowering the learning rate."
        )

    _save_comparison(inputs, targets, initial_pred, final_pred, args.output)


def _save_comparison(
    inputs: torch.Tensor, targets: torch.Tensor, initial_pred: torch.Tensor, final_pred: torch.Tensor, output: str
) -> None:
    n = inputs.shape[0]
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        image = inputs[i, :3].cpu().permute(1, 2, 0).numpy()
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("image")
        axes[i, 1].imshow(targets[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("ground truth")
        axes[i, 2].imshow(initial_pred[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("prediction (untrained)")
        axes[i, 3].imshow(final_pred[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title("prediction (overfit)")
        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    print(f"Saved before/after comparison to {output_path}")


if __name__ == "__main__":
    main()
