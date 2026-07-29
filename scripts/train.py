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
from src.model.unet import UNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import iou_score


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output", default="outputs/overfit_check.png")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    torch.manual_seed(train_config["seed"])
    device = get_device()
    print(f"Using device: {device}")

    root = Path(data_config["dataset"]["root"]).expanduser()
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
    subset_size = train_config["overfit"]["subset_size"]
    largest_indices = sorted(range(len(dataset)), key=lambda i: dataset.mask_areas[i], reverse=True)
    subset_indices = largest_indices[:subset_size]
    for i in subset_indices:
        _, instance = dataset.items[i]
        print(f"  overfit subset: {instance.name!r} (area={dataset.mask_areas[i]} px)")

    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

    model = UNet(in_channels=5, out_channels=1, base_channels=train_config["model"]["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["overfit"]["lr"])
    criterion = BCEDiceLoss()

    inputs, targets = next(iter(loader))
    inputs, targets = inputs.to(device), targets.to(device)

    model.eval()
    with torch.no_grad():
        initial_pred = torch.sigmoid(model(inputs))

    model.train()
    epochs = train_config["overfit"]["epochs"]
    log_every = train_config["overfit"]["log_every"]
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            iou = iou_score(logits.detach(), targets)
            print(f"epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  IoU={iou:.4f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(inputs)
        final_pred = torch.sigmoid(final_logits)
    final_iou = iou_score(final_logits, targets)
    print(f"Final overfit IoU on the {len(subset)}-example subset: {final_iou:.4f}")

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
