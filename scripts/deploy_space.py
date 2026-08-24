"""Push the interface to a Hugging Face Space, and the weights to a model repo.

Two separate uploads, deliberately:

  --space  uploads the app (deploy/huggingface/* plus a copy of src/) to a
           Space. This is text-sized and happens every time the UI changes.
  --model  uploads one exported checkpoint to a model repo. This is ~90 MB and
           happens only when a better checkpoint finishes training.

Keeping the weights out of the Space repo is what stops every interface tweak
from re-pushing the model through Git LFS, and lets a new checkpoint replace the
old one without redeploying the app.

Typical first deploy:

    python scripts/export_model.py --checkpoint results/run-multimask/best.pt \
        --output outputs/export/click-segmenter.pt
    huggingface-cli login
    python scripts/deploy_space.py --model maisisif/click-segmenter \
        --checkpoint outputs/export/click-segmenter.pt
    python scripts/deploy_space.py --space maisisif/click-segmenter

Later, to update only the interface:

    python scripts/deploy_space.py --space maisisif/click-segmenter

Both flags can be given at once. Nothing is uploaded without --yes or an
interactive confirmation, because both targets are public by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_DIR = REPO_ROOT / "deploy" / "huggingface"

# Everything the app imports at runtime. configs/ is deliberately absent: an
# exported checkpoint carries its own settings, so shipping the configs would
# only create a second source of truth that can disagree with the weights.
SRC_PACKAGES = ["app", "data", "inference", "model", "training"]


def stage_space(directory: Path) -> Path:
    """Lay out exactly what the Space should contain, in a temporary directory.

    Assembled rather than uploaded in place so the Space gets `src/` at its root
    next to `app.py`, which is the layout the imports in app.py expect.
    """
    staged = directory / "space"
    staged.mkdir()

    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(SPACE_DIR / name, staged / name)

    shutil.copy2(REPO_ROOT / "src" / "__init__.py", _mkdir(staged / "src") / "__init__.py")
    for package in SRC_PACKAGES:
        shutil.copytree(
            REPO_ROOT / "src" / package,
            staged / "src" / package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return staged


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{message}\nRefusing to upload without --yes in a non-interactive shell.")
        return False
    return input(f"{message} [y/N] ").strip().lower() in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--space", help="Space repo id, e.g. maisisif/click-segmenter")
    parser.add_argument("--model", help="Model repo id to upload the checkpoint to")
    parser.add_argument(
        "--checkpoint",
        default="outputs/export/click-segmenter.pt",
        help="Exported checkpoint to upload with --model",
    )
    parser.add_argument(
        "--model-file",
        default="click-segmenter.pt",
        help="Filename to give the checkpoint inside the model repo",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the Space contents and list them without uploading anything",
    )
    args = parser.parse_args()

    if not args.space and not args.model:
        parser.error("give --space, --model, or both")

    from huggingface_hub import HfApi

    api = HfApi()

    if args.model:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            parser.error(f"{checkpoint} does not exist -- run scripts/export_model.py first")
        size_mb = checkpoint.stat().st_size / 1e6

        if args.dry_run:
            print(f"would upload {checkpoint} ({size_mb:.1f} MB) "
                  f"to {args.model}/{args.model_file}")
        elif confirm(
            f"Upload {checkpoint} ({size_mb:.1f} MB) to model repo {args.model} "
            f"as {args.model_file}?",
            args.yes,
        ):
            api.create_repo(args.model, repo_type="model", private=args.private, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(checkpoint),
                path_in_repo=args.model_file,
                repo_id=args.model,
                repo_type="model",
            )
            print(f"uploaded to https://huggingface.co/{args.model}")

    if args.space:
        with tempfile.TemporaryDirectory() as directory:
            staged = stage_space(Path(directory))
            files = sorted(p.relative_to(staged) for p in staged.rglob("*") if p.is_file())
            total_kb = sum((staged / f).stat().st_size for f in files) / 1024
            print(f"staged {len(files)} files ({total_kb:.0f} KB):")
            for f in files:
                print(f"  {f}")

            if args.dry_run:
                print(f"would upload the above to Space {args.space}")
            elif confirm(f"Upload the above to Space {args.space}?", args.yes):
                api.create_repo(
                    args.space,
                    repo_type="space",
                    space_sdk="gradio",
                    private=args.private,
                    exist_ok=True,
                )
                api.upload_folder(folder_path=str(staged), repo_id=args.space, repo_type="space")
                print(f"deployed to https://huggingface.co/spaces/{args.space}")
                print("The Space now builds; watch its Logs tab. First build takes a few minutes.")


if __name__ == "__main__":
    main()
