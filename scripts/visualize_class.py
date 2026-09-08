"""Qualitative figure for a one-class UNet (train_class.py): random test images
with the input, the model's mask, the ground-truth mask and an overlay.

Kassem, 2026-09-08: "Take a random image from the testset, pass it through the
model, plot the image, the output of the model, the ground truth and the
segmentation overlay."

The test split is rebuilt from the index that the training run cached in its
output directory (index_<match>.json), NOT by rescanning the dataset. The
export may have added images since the run started; rescanning would change
the split and show images the model was trained on. If the cached index is
missing, the script falls back to a rescan and says so.

Run from the repo root (CPU is fine, a handful of forward passes):
    python scripts/visualize_class.py --class-name bed
    python scripts/visualize_class.py --class-name bed --num 8 --seed 1 --output figure.png

On the cluster, inside the container (see CLAUDE.md for the one-liner).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ade20k import discover_samples  # noqa: E402
from src.data.class_dataset import ClassSegmentationDataset, build_class_index  # noqa: E402
from src.data.splits import split_image_paths  # noqa: E402
from src.model.build import build_model  # noqa: E402
from src.training.device import get_device  # noqa: E402

EPS = 1e-6


def mask_edge(mask: np.ndarray) -> np.ndarray:
    """One-pixel-wide boundary of a boolean mask (4-neighbourhood, no scipy)."""
    padded = np.pad(mask, 1, mode="edge")
    eroded = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~eroded


def overlay(image: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Image with the prediction tinted red and the ground-truth outline in green."""
    out = image.astype(np.float32) / 255.0
    red = np.array([1.0, 0.15, 0.15], dtype=np.float32)
    out[pred] = 0.55 * out[pred] + 0.45 * red
    edge = mask_edge(truth)
    # Thicken the outline so it survives downscaling in a chat window.
    edge = edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
    out[edge] = np.array([0.1, 1.0, 0.1], dtype=np.float32)
    return (out * 255).clip(0, 255).astype(np.uint8)


def test_entries(output_dir: Path, class_name: str, match: str, root: Path | None, training: dict) -> list[tuple[Path, list[int]]]:
    """Rebuild the exact test split of the training run."""
    cache_path = output_dir / f"index_{match}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        index = cached["index"]
        print(f"Using the training run's cached index ({len(index)} images) from {cache_path}")
    else:
        if root is None:
            raise SystemExit(f"No cached index at {cache_path}; pass --data-root to rescan")
        print(f"WARNING: no cached index at {cache_path}; rescanning {root} (split may differ from the run)")
        index = build_class_index(discover_samples(root), class_name, match=match, cache_path=None)

    positives = sorted(Path(p) for p, ids in index.items() if ids)
    splits = split_image_paths(positives, ratios=tuple(training["splits"]), seed=training["split_seed"])
    entries = [(p, index[str(p)]) for p in splits["test"]]

    counts_path = output_dir / "counts.json"
    if counts_path.exists():
        with open(counts_path) as f:
            recorded = json.load(f)["splits"]["test"]["images"]
        if recorded != len(entries):
            print(f"WARNING: rebuilt test split has {len(entries)} images, the run recorded {recorded}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", default="chair")
    parser.add_argument("--output-dir", default=None, help="Training output dir, default outputs/class/<class-name>")
    parser.add_argument("--checkpoint", default=None, help="Default <output-dir>/checkpoints/best.pt")
    parser.add_argument("--data-root", default=None, help="Only needed if the run's cached index is missing")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--num", type=int, default=6, help="How many random test images to show")
    parser.add_argument("--seed", type=int, default=0, help="Which random images; change it for a different set")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="PNG path, default <output-dir>/qualitative_seed<seed>.png")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    class_name = args.class_name.strip().lower()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "class" / class_name.replace(" ", "_")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best.pt"
    figure_path = Path(args.output) if args.output else output_dir / f"qualitative_seed{args.seed}.png"

    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    training = train_config["training"]

    device = get_device(args.device or "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    settings = ckpt.get("train_settings", {})
    match = settings.get("match", "exact")
    image_size = tuple(settings.get("image_size", train_config["data"]["image_size"]))
    if settings.get("class_name", class_name) != class_name:
        print(f"WARNING: checkpoint was trained on {settings['class_name']!r}, not {class_name!r}")
    print(f"Checkpoint {checkpoint_path}: epoch {ckpt.get('epoch')}, best val IoU {ckpt.get('best_val_iou', float('nan')):.4f}")

    model_config = dict(train_config["model"])
    model_config.update({"arch": "resnet34_unet", "in_channels": 3, "num_masks": 1, "pretrained": False})
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    entries = test_entries(output_dir, class_name, match, Path(args.data_root) if args.data_root else None, training)
    if not entries:
        raise SystemExit("Test split is empty")
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(entries), size=min(args.num, len(entries)), replace=False))
    picked = [entries[i] for i in chosen]
    dataset = ClassSegmentationDataset(picked, image_size=image_size, augment=False)

    rows = len(picked)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows), squeeze=False)
    titles = ["Image", "Model output", "Ground truth", "Overlay (red = prediction, green = GT outline)"]
    ious = []
    with torch.no_grad():
        for row, ((image_path, ids), (image_t, mask_t)) in enumerate(zip(picked, dataset)):
            logits, _ = model(image_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = prob > args.threshold
            truth = mask_t[0].numpy() > 0.5
            inter = np.logical_and(pred, truth).sum()
            union = np.logical_or(pred, truth).sum()
            iou = inter / max(union, EPS)
            ious.append(iou)

            image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            panels = [image, pred, truth, overlay(image, pred, truth)]
            for col, (ax, panel) in enumerate(zip(axes[row], panels)):
                ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0, vmax=1 if panel.ndim == 2 else None)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col], fontsize=11)
            axes[row][0].set_ylabel(f"{image_path.stem}\n{len(ids)} {class_name}(s)  IoU {iou:.3f}", fontsize=9)
            print(f"{image_path.stem}: {len(ids)} instance(s), IoU {iou:.4f}")

    fig.suptitle(
        f"{class_name}: {rows} random test images (seed {args.seed}), mean IoU on these {np.mean(ious):.3f}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=110)
    print(f"Figure written to {figure_path}")


if __name__ == "__main__":
    main()
