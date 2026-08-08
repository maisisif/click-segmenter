"""Export the Hugging Face `1aurent/ADE20K` mirror into the on-disk layout
`src/data/ade20k.py` expects, so the data loader needs no changes.

For each row of the HF dataset we write, under `--output-root`:
    {split}/{scene_category}/{scene_subcategory}/{stem}.jpg
    {split}/{scene_category}/{scene_subcategory}/{stem}_seg.png
    {split}/{scene_category}/{scene_subcategory}/{stem}.json
    {split}/{scene_category}/{scene_subcategory}/{stem}/instance_{id:03d}_{stem}.png

This mirrors the original ADE20K toolkit convention (see CLAUDE.md). Schema
notes on the HF mirror (verified against `1aurent/ADE20K`, config "default"):
  - `folder` looks like "ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor"
    -> strip everything up to and including "images/ADE/" to get the relative
    output dir ("training/cultural/apse__indoor").
  - `filename` already includes the ".jpg" extension.
  - `instances[i]` (mode "L" image) aligns 1:1 with `objects[i]` by list order
    (verified: id 0..N-1 in objects matches instances index for two sampled
    rows). Pixel values are the ternary amodal encoding {0, 128, 255}. These
    MUST be saved as PNG (lossless) -- the HF datasets-server *preview* assets
    are JPEG-recompressed and smear those exact values, but the actual
    parquet-stored bytes accessed via `datasets.load_dataset` are lossless.
  - `segmentations[0]` is the packed class+instance RGB mask (-> `_seg.png`,
    also must stay PNG/lossless). `segmentations` can have more than one
    entry per row; we only need the first for our pipeline today.
  - `objects[i].parts.part_level` feeds directly into the JSON's
    `annotation.object[i].parts.part_level`, which `load_sample()` uses to
    skip sub-parts. We export every object (not just part_level == 0) since
    the loader already filters at read time.

Run from the repo root, e.g. a small dry run first:
    python scripts/export_ade20k.py --split train --limit 20 \
        --output-root ~/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE

Then the full split (large: ~25,574 train / 2,000 validation images, several
GB of downloads from the HF hub):
    python scripts/export_ade20k.py --split train \
        --output-root ~/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE

Requires: pip install datasets pyarrow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _relative_dir(folder: str) -> str:
    """"ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor" -> "training/cultural/apse__indoor"."""
    marker = "images/ADE/"
    idx = folder.find(marker)
    if idx == -1:
        raise ValueError(f"Unexpected folder format, missing {marker!r}: {folder!r}")
    return folder[idx + len(marker):]


def _export_row(row: dict, output_root: Path) -> None:
    stem = Path(row["filename"]).stem
    out_dir = output_root / _relative_dir(row["folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    row["image"].convert("RGB").save(out_dir / f"{stem}.jpg", "JPEG")
    row["segmentations"][0].convert("RGB").save(out_dir / f"{stem}_seg.png", "PNG")

    objects = row["objects"]
    # Several of these fields are nullable in the mirror: a top-level object
    # has no parent, so `is_part_of` is None, and objects without sub-parts
    # have `has_parts` as None. Only `part_level` is actually read back by
    # src/data/ade20k.py; the rest are kept for reference and must not crash
    # the export.
    annotation_objects = []
    for obj in objects:
        parts = obj.get("parts") or {}
        annotation_objects.append(
            {
                "id": int(obj["id"]),
                "name": obj["name"],
                "parts": {
                    "part_level": int(parts.get("part_level") or 0),
                    "is_part_of": int(parts["is_part_of"]) if parts.get("is_part_of") is not None else None,
                    "has_parts": list(parts.get("has_parts") or []),
                },
            }
        )
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"annotation": {"object": annotation_objects}}, f)

    if objects:
        instances_dir = out_dir / stem
        instances_dir.mkdir(exist_ok=True)
        instance_images = row["instances"]
        for obj, mask_img in zip(objects, instance_images):
            mask_path = instances_dir / f"instance_{int(obj['id']):03d}_{stem}.png"
            mask_img.convert("L").save(mask_path, "PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="1aurent/ADE20K")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--output-root", required=True, help="Directory matching configs/data.yaml's dataset.root")
    parser.add_argument("--limit", type=int, default=None, help="Export only the first N rows (for a dry run)")
    parser.add_argument("--start", type=int, default=0, help="Skip the first N rows before exporting")
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download the full split up front instead of streaming row-by-row (uses more disk/RAM)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=not args.no_streaming)

    exported = 0
    failed = 0
    for i, row in enumerate(dataset):
        if i < args.start:
            continue
        if args.limit is not None and exported >= args.limit:
            break

        # One malformed row out of 25k must not abort a multi-hour export.
        # Log it, skip it, keep going -- a handful of missing images is far
        # cheaper than losing the whole run.
        try:
            _export_row(row, output_root)
            exported += 1
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            failed += 1
            print(f"  SKIPPED {row.get('filename', f'row {i}')}: {type(exc).__name__}: {exc}")
            if failed > max(20, i // 10):
                raise RuntimeError(f"aborting: {failed} failures, something is systematically wrong") from exc
            continue

        if exported % 100 == 0 or exported == 1:
            print(f"exported {exported} rows (last: {row['filename']})")

    print(f"Done. Exported {exported} rows from split={args.split!r} to {output_root} ({failed} skipped)")


if __name__ == "__main__":
    main()
