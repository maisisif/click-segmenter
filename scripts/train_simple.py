"""The whole training pipeline in one readable file: load, train, visualize.

This is the walkthrough version. It does exactly what scripts/train_full.py
does, minus everything that exists for cluster reality (checkpoint chaining,
early stopping, LR scheduling, lazy loading, index caching, worker tuning).
Read this file to understand the system; use train_full.py for real runs.

The flow, start to finish:

  1. Find images. Each ADE20K image comes with one binary mask per object.
  2. Build (input, target) pairs. The input is a 5-channel tensor: the RGB
     image, a map with a disk where a "this is the object" click landed, and a
     map for "this is not the object" clicks. Clicks are simulated from the
     ground-truth mask, since no real user clicked anything.
  3. Split by IMAGE into train/val. Splitting by object would leak: two
     objects from the same photo would let the model validate on scenes it
     memorized.
  4. Train: forward pass -> loss (BCE + Dice) -> backward -> optimizer step.
  5. Validate each epoch on clicks that never change, so the number reflects
     the model, not click randomness.
  6. Save a picture: image + click, ground truth, prediction. Numbers hide
     failure modes; pictures don't.

Run from the repo root (CPU is fine, it's deliberately small):
    python scripts/train_simple.py --data-root ~/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.dataset import ClickSegmentationDataset
from src.data.splits import split_image_paths
from src.model.unet import UNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import iou_score

IMAGE_SIZE = (192, 256)  # (height, width) — small enough to train on a laptop
CLICKS = {"radius": 5, "negative_prob": 0.5, "neighbor_negative_prob": 0.3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="outputs/simple_run.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 1-2. Find images, keep a small subset, split by image.
    image_paths = discover_samples(Path(args.data_root).expanduser())[: args.num_images]
    splits = split_image_paths(image_paths, ratios=(0.8, 0.1, 0.1), seed=0)
    print(f"{len(image_paths)} images -> {len(splits['train'])} train, {len(splits['val'])} val")

    # 3. Datasets. Training clicks are re-simulated every epoch (free
    # augmentation); validation clicks are fixed (comparable numbers).
    train_set = ClickSegmentationDataset(splits["train"], IMAGE_SIZE, CLICKS, deterministic=False)
    val_set = ClickSegmentationDataset(splits["val"], IMAGE_SIZE, CLICKS, deterministic=True)
    print(f"{len(train_set)} training objects, {len(val_set)} validation objects")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # 4. Model, loss, optimizer.
    model = UNet(in_channels=5, out_channels=1, base_channels=32, depth=3).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. The training loop.
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)

        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_iou += iou_score(model(inputs), targets) * len(inputs)

        print(
            f"epoch {epoch:2d}/{args.epochs}  "
            f"train loss {train_loss / len(train_set):.4f}  "
            f"val IoU {val_iou / len(val_set):.4f}"
        )

    # 6. Visualize a few validation predictions.
    n = min(4, len(val_set))
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n), squeeze=False)
    model.eval()
    for row in range(n):
        inputs, target = val_set[row]
        with torch.no_grad():
            pred = torch.sigmoid(model(inputs.unsqueeze(0).to(device)))[0, 0].cpu() > 0.5

        axes[row][0].imshow(inputs[:3].permute(1, 2, 0))
        axes[row][0].contour(inputs[3], colors="lime")   # positive click
        if inputs[4].any():
            axes[row][0].contour(inputs[4], colors="red")  # negative click(s)
        axes[row][0].set_title("image + clicks")
        axes[row][1].imshow(target[0], cmap="gray")
        axes[row][1].set_title("ground truth")
        axes[row][2].imshow(pred, cmap="gray")
        axes[row][2].set_title("prediction")
        for ax in axes[row]:
            ax.axis("off")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=100)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
