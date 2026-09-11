"""Qualitative figure for a one- or multi-class UNet (train_class.py): random
test images with the input, the model's masks, the ground-truth masks and an
overlay.

Kassem, 2026-09-08: "Take a random image from the testset, pass it through the
model, plot the image, the output of the model, the ground truth and the
segmentation overlay."

The test split is rebuilt from the index that the training run cached in its
output directory (index_<match>.json, or index_<match>_<class>.json per class
for a multi-class run), NOT by rescanning the dataset. The export may have added
images since the run started; rescanning would change the split and show images
the model was trained on. If the cached index is missing, the script falls back
to a rescan and says so.

Run from the repo root (CPU is fine, a handful of forward passes):
    python scripts/visualize_class.py --class-name bed
    python scripts/visualize_class.py --class-name bed floor --num 8 --seed 1

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
# One colour per output channel: red, blue, yellow, magenta, cyan, orange.
COLORS = np.array(
    [[1.0, 0.15, 0.15], [0.2, 0.4, 1.0], [1.0, 0.9, 0.1], [0.9, 0.2, 0.9], [0.1, 0.9, 0.9], [1.0, 0.55, 0.0]],
    dtype=np.float32,
)


def mask_edge(mask: np.ndarray) -> np.ndarray:
    """Boundary of a boolean mask, two pixels wide so it survives downscaling."""
    padded = np.pad(mask, 1, mode="edge")
    eroded = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    edge = mask & ~eroded
    return edge | np.roll(edge, 1, 0) | np.roll(edge, 1, 1)


def paint(masks: np.ndarray) -> np.ndarray:
    """(C, H, W) boolean masks -> RGB image, black background, one colour per class."""
    out = np.zeros((*masks.shape[1:], 3), dtype=np.float32)
    for k, mask in enumerate(masks):
        out[mask] = COLORS[k % len(COLORS)]
    return out


def overlay(image: np.ndarray, preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Image with each class's prediction tinted in its colour and the
    ground-truth outline of that class drawn solid in the same colour."""
    out = image.astype(np.float32) / 255.0
    for k, pred in enumerate(preds):
        out[pred] = 0.55 * out[pred] + 0.45 * COLORS[k % len(COLORS)]
    for k, truth in enumerate(truths):
        out[mask_edge(truth)] = COLORS[k % len(COLORS)]
    return (out * 255).clip(0, 255).astype(np.uint8)


def load_indices(output_dir: Path, class_names: list[str], match: str, root: Path | None) -> dict[str, dict[str, list[int]]]:
    indices = {}
    for name in class_names:
        cache_name = f"index_{match}.json" if len(class_names) == 1 else f"index_{match}_{name.replace(' ', '_')}.json"
        cache_path = output_dir / cache_name
        if cache_path.exists():
            with open(cache_path) as f:
                indices[name] = json.load(f)["index"]
            print(f"Using the training run's cached {name} index ({len(indices[name])} images) from {cache_path}")
        else:
            if root is None:
                raise SystemExit(f"No cached index at {cache_path}; pass --data-root to rescan")
            print(f"WARNING: no cached index at {cache_path}; rescanning {root} (split may differ from the run)")
            indices[name] = build_class_index(discover_samples(root), name, match=match, cache_path=None)
    return indices


def test_entries(output_dir: Path, class_names: list[str], match: str, root: Path | None, training: dict) -> list[tuple[Path, list]]:
    """Rebuild the exact test split of the training run (see train_class.py)."""
    indices = load_indices(output_dir, class_names, match, root)
    anchor = indices[class_names[0]]
    positives = sorted(Path(p) for p, ids in anchor.items() if ids)
    splits = split_image_paths(positives, ratios=tuple(training["splits"]), seed=training["split_seed"])
    if len(class_names) == 1:
        entries = [(p, anchor[str(p)]) for p in splits["test"]]
    else:
        entries = [(p, [indices[n][str(p)] for n in class_names]) for p in splits["test"]]

    counts_path = output_dir / "counts.json"
    if counts_path.exists():
        with open(counts_path) as f:
            recorded = json.load(f)["splits"]["test"]["images"]
        if recorded != len(entries):
            print(f"WARNING: rebuilt test split has {len(entries)} images, the run recorded {recorded}")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class-name", nargs="+", default=["chair"], help="Same name(s), same order, as the training run")
    parser.add_argument("--output-dir", default=None, help="Training output dir, default outputs/class/<names joined by _>")
    parser.add_argument("--checkpoint", default=None, help="Default <output-dir>/checkpoints/best.pt")
    parser.add_argument("--data-root", default=None, help="Only needed if the run's cached index is missing")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--num", type=int, default=6, help="How many random test images to show")
    parser.add_argument("--seed", type=int, default=0, help="Which random images; change it for a different set")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="PNG path, default <output-dir>/qualitative_seed<seed>.png")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    class_names = [c.strip().lower().replace("_", " ") for c in args.class_name]  # pool_table -> "pool table"
    label = "+".join(class_names)
    default_dir = Path("outputs") / "class" / "_".join(c.replace(" ", "_") for c in class_names)
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
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
    trained_on = settings.get("class_names", [settings.get("class_name", class_names[0])])
    if trained_on != class_names:
        print(f"WARNING: checkpoint was trained on {trained_on}, you asked for {class_names}")
    print(f"Checkpoint {checkpoint_path}: epoch {ckpt.get('epoch')}, best val IoU {ckpt.get('best_val_iou', float('nan')):.4f}")

    model_config = dict(train_config["model"])
    model_config.update({"arch": "resnet34_unet", "in_channels": 3, "num_masks": len(class_names), "pretrained": False})
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    entries = test_entries(output_dir, class_names, match, Path(args.data_root) if args.data_root else None, training)
    if not entries:
        raise SystemExit("Test split is empty")
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(len(entries), size=min(args.num, len(entries)), replace=False))
    picked = [entries[i] for i in chosen]
    dataset = ClassSegmentationDataset(picked, image_size=image_size, augment=False)

    rows = len(picked)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 3.2 * rows), squeeze=False)
    legend = ", ".join(f"{n} = {c}" for n, c in zip(class_names, ["red", "blue", "yellow", "magenta", "cyan", "orange"]))
    titles = ["Image", "Model output", "Ground truth", "Overlay (filled = prediction, outline = GT)"]
    all_ious: list[list[float]] = [[] for _ in class_names]
    with torch.no_grad():
        for row, ((image_path, ids), (image_t, mask_t)) in enumerate(zip(picked, dataset)):
            logits, _ = model(image_t.unsqueeze(0).to(device))
            preds = (torch.sigmoid(logits)[0].cpu().numpy() > args.threshold)  # (C, H, W)
            truths = mask_t.numpy() > 0.5

            parts = []
            for k, name in enumerate(class_names):
                if not truths[k].any():
                    parts.append(f"{name}: absent")
                    continue
                inter = np.logical_and(preds[k], truths[k]).sum()
                union = np.logical_or(preds[k], truths[k]).sum()
                iou = inter / max(union, EPS)
                all_ious[k].append(iou)
                parts.append(f"{name} {iou:.3f}")

            image = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            panels = [image, paint(preds), paint(truths), overlay(image, preds, truths)]
            for col, (ax, panel) in enumerate(zip(axes[row], panels)):
                ax.imshow(panel)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(titles[col], fontsize=11)
            axes[row][0].set_ylabel(f"{image_path.stem}\nIoU " + "  ".join(parts), fontsize=8)
            print(f"{image_path.stem}: " + ", ".join(parts))

    means = "  ".join(f"{n} {np.mean(v):.3f}" for n, v in zip(class_names, all_ious) if v)
    fig.suptitle(f"{label}: {rows} random test images (seed {args.seed}). {legend}. Mean IoU on these: {means}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=110)
    print(f"Figure written to {figure_path}")


if __name__ == "__main__":
    main()
