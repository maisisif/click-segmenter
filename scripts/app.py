"""Run the click-to-segment interface on this machine.

Upload an image, click an object, get its mask. Object clicks select; exclude
clicks carve away a region the mask wrongly swallowed.

    python scripts/app.py                                   # local only
    python scripts/app.py --share                           # temporary public link
    python scripts/app.py --checkpoint results/run-multimask/best.pt

The checkpoint may be either a training checkpoint (in which case configs/
supplies the resolution and click settings) or one exported by
scripts/export_model.py, which carries those settings inside it.

On MetaCentrum, run this inside an interactive job on a node with internet and
use --share, since compute nodes are not directly reachable. If that node has no
internet the share tunnel cannot be created -- run it on the laptop instead.

Deliberately thin: the interface lives in src/app/ui.py and the model logic in
src/inference/predictor.py, so the same UI backs the hosted demo unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app.ui import THEME, build_ui
from src.inference.predictor import ClickPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    predictor = ClickPredictor(
        args.checkpoint,
        train_config_path=args.train_config,
        clicks_config_path=args.clicks_config,
        device=args.device,
    )
    print(f"Loaded {args.checkpoint} on {predictor.device}")

    build_ui(predictor).launch(
        theme=THEME, share=args.share, server_port=args.port, server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
