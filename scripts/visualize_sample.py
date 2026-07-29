"""M1: load one ADE20K sample and save an image + instance-mask overlay.

Run from the repo root:
    python scripts/visualize_sample.py --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples, load_sample
from src.data.visualize import overlay_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--index", type=int, default=0, help="Which discovered sample to visualize")
    parser.add_argument(
        "--amodal", action="store_true", help="Show amodal (occlusion-inclusive) masks instead of visible-only"
    )
    parser.add_argument("--output", default="outputs/sample_overlay.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    root = Path(config["dataset"]["root"]).expanduser()
    samples = discover_samples(root)
    if not samples:
        raise SystemExit(f"No samples found under {root}")

    print(f"Found {len(samples)} sample(s) under {root}")
    image_path = samples[args.index]
    sample = load_sample(image_path)
    print(f"Loaded {sample.stem}: {len(sample.instances)} instance(s)")
    for inst in sample.instances:
        print(f"  id={inst.id:>3}  {inst.name}")

    overlay = overlay_instances(sample, use_amodal=args.amodal)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(f"Saved overlay to {output_path}")


if __name__ == "__main__":
    main()
