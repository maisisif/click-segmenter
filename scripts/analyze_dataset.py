"""Measure the exported dataset. Run this before making design decisions.

Reads only the sidecar JSON files (no image decoding), so it takes seconds.
Everything printed is counted from the files actually on disk, not quoted from
the ADE20K paper.

The question this is meant to settle: if the model outputs one channel per
object CLASS (channel 0 = chair, channel 1 = closet, ...), how often would a
single channel have to represent two or more separate objects? That is the
difference between "click selects one chair" and "click selects all chairs".

Run from the repo root:
    python scripts/analyze_dataset.py --data-root /storage/brno2/home/$USER/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only look at the first N images")
    args = parser.parse_args()

    image_paths = discover_samples(Path(args.data_root).expanduser())
    if args.limit:
        image_paths = image_paths[: args.limit]
    print(f"Analysing {len(image_paths)} images\n")

    per_image_counts: list[int] = []
    class_counter: Counter[str] = Counter()
    images_missing_json = 0

    # (image, class) pairs where that class occurs more than once in the image
    duplicated_pairs = 0
    total_pairs = 0
    instances_with_a_sibling = 0
    total_instances = 0

    for path in image_paths:
        json_path = path.parent / f"{path.stem}.json"
        if not json_path.exists():
            images_missing_json += 1
            continue
        with open(json_path) as f:
            objects = json.load(f)["annotation"]["object"]

        # part_level 0 == a top-level object, i.e. what the dataset treats as
        # independently clickable. Sub-parts are excluded, same as training.
        names = [o["name"] for o in objects if int(o["parts"]["part_level"]) == 0]

        per_image_counts.append(len(names))
        class_counter.update(names)
        total_instances += len(names)

        counts_here = Counter(names)
        total_pairs += len(counts_here)
        for name, n in counts_here.items():
            if n > 1:
                duplicated_pairs += 1
                instances_with_a_sibling += n

    print("=== Size ===")
    print(f"images with annotations : {len(per_image_counts)}")
    print(f"images missing JSON     : {images_missing_json}")
    print(f"total objects           : {total_instances}")
    print(f"distinct class names    : {len(class_counter)}")

    counts = sorted(per_image_counts)
    def pct(p: float) -> int:
        return counts[min(int(len(counts) * p), len(counts) - 1)]

    print("\n=== Objects per image ===")
    print(f"min {counts[0]}   median {statistics.median(counts):.0f}   "
          f"mean {statistics.mean(counts):.1f}   max {counts[-1]}")
    print(f"50th {pct(0.50)}   90th {pct(0.90)}   95th {pct(0.95)}   99th {pct(0.99)}")

    print("\n=== Repeated classes within one image ===")
    print("(this is what decides whether one channel per class merges objects)")
    print(f"(image, class) pairs total          : {total_pairs}")
    print(f"  ... where the class occurs 2+ times: {duplicated_pairs} "
          f"({100 * duplicated_pairs / total_pairs:.1f}%)")
    print(f"objects sharing their class with another object in the same image: "
          f"{instances_with_a_sibling} ({100 * instances_with_a_sibling / total_instances:.1f}%)")

    print("\n=== Most common classes ===")
    for name, n in class_counter.most_common(20):
        print(f"  {n:7d}  {name}")

    print("\n=== Coverage if we keep only the top K classes ===")
    ordered = [n for _, n in class_counter.most_common()]
    for k in (50, 100, 150, 300, 500, 1000):
        if k <= len(ordered):
            covered = sum(ordered[:k])
            print(f"  top {k:4d} classes cover {100 * covered / total_instances:5.1f}% of objects")

    print("\n=== What this means for a fixed-size output tensor ===")
    print(f"A tensor with one channel per class needs K channels and makes")
    print(f"{100 * instances_with_a_sibling / total_instances:.1f}% of objects "
          f"inseparable from a same-class neighbour.")
    print(f"A tensor with one channel per object slot needs {counts[-1]} channels "
          f"to fit the largest image ({pct(0.95)} would cover 95% of images).")


if __name__ == "__main__":
    main()
