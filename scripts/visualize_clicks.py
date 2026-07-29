"""M2: simulate clicks on one instance and visualize the encoded channels.

Run from the repo root:
    python scripts/visualize_clicks.py --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples, load_sample
from src.data.clicks import simulate_clicks
from src.data.encoding import encode_clicks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--index", type=int, default=0, help="Which discovered sample to use")
    parser.add_argument(
        "--instance-id", type=int, default=None, help="Instance id to click on (defaults to the largest instance)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/click_visualization.png")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        clicks_config = yaml.safe_load(f)["clicks"]

    root = Path(data_config["dataset"]["root"]).expanduser()
    samples = discover_samples(root)
    if not samples:
        raise SystemExit(f"No samples found under {root}")

    sample = load_sample(samples[args.index])
    if not sample.instances:
        raise SystemExit(f"{sample.stem} has no instances to click on")

    if args.instance_id is not None:
        matches = [inst for inst in sample.instances if inst.id == args.instance_id]
        if not matches:
            raise SystemExit(f"No instance with id={args.instance_id} in {sample.stem}")
        instance = matches[0]
    else:
        instance = max(sample.instances, key=lambda inst: inst.mask_visible.sum())

    print(f"Sample: {sample.stem}  |  instance id={instance.id}  name={instance.name!r}")

    rng = np.random.default_rng(args.seed)
    clicks = simulate_clicks(
        instance.mask_visible,
        rng,
        positive_interior_frac=clicks_config["positive_interior_frac"],
        negative_min_distance=clicks_config["negative_min_distance"],
        negative_max_distance=clicks_config["negative_max_distance"],
        negative_prob=clicks_config["negative_prob"],
    )
    for click in clicks:
        print(f"  {'positive' if click.positive else 'negative'} click at (y={click.y}, x={click.x})")

    encoded = encode_clicks(clicks, shape=sample.image.shape[:2], radius=clicks_config["radius"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(sample.image)
    axes[0].imshow(instance.mask_visible, alpha=0.3, cmap="Greens")
    for click in clicks:
        marker, color = ("+", "lime") if click.positive else ("x", "red")
        axes[0].scatter(click.x, click.y, marker=marker, color=color, s=200, linewidths=3)
    axes[0].set_title(f"{instance.name} — mask + simulated clicks")
    axes[0].axis("off")

    axes[1].imshow(encoded[0], cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("Positive click channel")
    axes[1].axis("off")

    axes[2].imshow(encoded[1], cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Negative click channel")
    axes[2].axis("off")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
