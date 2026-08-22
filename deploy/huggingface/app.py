"""Entry point for the hosted demo on Hugging Face Spaces.

Spaces runs `app.py` at the root of the Space repo, so this file is uploaded
there as `app.py` alongside a copy of `src/` (scripts/deploy_space.py does
that). It differs from scripts/app.py in exactly one respect: where the
checkpoint comes from.

**The weights live in a separate model repo, not in this Space.** A Space is a
git repo, and a 90 MB checkpoint tracked in it means every one-line edit to the
interface pushes through Git LFS and re-uploads on each redeploy. Keeping the
weights in their own model repo means UI changes are text-sized pushes, and the
model can be replaced -- when a better checkpoint finishes training -- without
touching the app at all. `hf_hub_download` caches the file on the Space's disk,
so this is a first-boot cost, not a per-request one.

Configure with Space variables (Settings -> Variables and secrets):

    MODEL_REPO   Hugging Face model repo holding the weights
    MODEL_FILE   filename within that repo (default click-segmenter.pt)
    MODEL_TOKEN  a read token, only if the model repo is private

The checkpoint must be one produced by scripts/export_model.py: it carries its
own resolution and click settings, which is why no configs/ directory is
uploaded to the Space.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from huggingface_hub import hf_hub_download

from src.app.ui import THEME, build_ui
from src.inference.predictor import ClickPredictor

DEFAULT_MODEL_REPO = "maisisif/click-segmenter"
DEFAULT_MODEL_FILE = "click-segmenter.pt"


def resolve_checkpoint() -> str:
    """Return a local path to the weights, downloading them on first boot.

    A local file named by MODEL_PATH wins if it exists, which is what makes this
    file runnable on a laptop for a rehearsal of the deploy before pushing it.
    """
    local = os.environ.get("MODEL_PATH")
    if local and Path(local).exists():
        print(f"using local checkpoint {local}")
        return local

    repo = os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO)
    filename = os.environ.get("MODEL_FILE", DEFAULT_MODEL_FILE)
    print(f"downloading {filename} from {repo}")
    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=os.environ.get("MODEL_TOKEN"),
    )


def main() -> None:
    # Always CPU: the free tier has no GPU, and asking for "auto" would only
    # make a failure here look like a device-detection problem.
    predictor = ClickPredictor(resolve_checkpoint(), device="cpu")
    print(f"loaded {predictor.arch['arch']} at {predictor.image_size}")

    # No share tunnel and no explicit port: Spaces provides the public URL and
    # sets the port itself.
    build_ui(predictor).launch(theme=THEME)


if __name__ == "__main__":
    main()
