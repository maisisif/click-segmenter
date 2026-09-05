"""M4: real training on ADE20K with train/validation/test splits.

Distinct from `scripts/train.py`, which only runs the M3 overfit sanity check on
a cherry-picked handful of instances. This is the actual training run:

  - splits images 70/20/10 (see `src/data/splits.py` for why it's by image)
  - trains on the train split with a freshly simulated click each epoch,
    which doubles as augmentation
  - validates every epoch with *fixed* clicks, so the val curve reflects model
    changes rather than click sampling noise
  - keeps the checkpoint from the best validation epoch, not the last one
  - evaluates that best checkpoint once on the held-out test split at the end
  - writes the full per-epoch history to JSON for plotting in the notebook

Run from the repo root:
    python scripts/train_full.py                      # uses configs/train.yaml
    python scripts/train_full.py --device cpu --epochs 2   # quick dry run

Iterative (multi-click) training, RITM-style, fine-tuned from an existing run:

    python scripts/train_full.py --init-from results/run-7/best.pt \
        --prev-mask --iterative-clicks 3 --lr 0.0001 --max-images 0 \
        --checkpoint-dir outputs/iterative/checkpoints \
        --history-path outputs/iterative/history.json

  --iterative-clicks N  before each supervised step, run the model 0..N times
                        without gradient and add a corrective click where it was
                        wrong (src/training/interaction.py). Teaches the model
                        what clicks two and onward mean. Validation then reports
                        IoU after --val-clicks clicks (default N) and selects the
                        best epoch on that, plus single-click IoU for comparison.
  --prev-mask           give the model a sixth input channel holding its own
                        previous prediction, so a later click corrects the mask
                        rather than predicting afresh. Weights from --init-from
                        are widened with zero filters for the new channel.
  --init-from PATH      start from another run's weights with a fresh optimizer.
                        Ignored when --auto-resume finds a checkpoint to resume.

The test split is touched exactly once, at the very end. Peeking at it during
development is how you end up reporting a number that means nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ade20k import discover_samples
from src.data.dataset import ClickSegmentationDataset
from src.data.splits import split_image_paths
from src.model.build import build_model, expand_input_channels
from src.model.unet import migrate_legacy_state_dict
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.device import get_device
from src.training.interaction import add_training_clicks, run_interaction
from src.training.losses import MultiMaskLoss
from src.training.metrics import best_of_n_iou, iou_score


def build_loader(
    image_paths: list[Path],
    train_config: dict,
    click_config: dict,
    *,
    shuffle: bool,
    deterministic: bool,
    split_name: str,
) -> DataLoader:
    dataset = ClickSegmentationDataset(
        image_paths,
        image_size=train_config["data"]["image_size"],
        click_config=click_config,
        deterministic=deterministic,
        lazy=True,  # the full dataset can't be held in memory
        index_cache=Path("outputs") / f"instance_index_{split_name}.json",
    )
    # Training is I/O bound, not compute bound: each item decodes a JPEG and a
    # PNG off shared storage, so the GPU spends most of its time waiting. These
    # three settings target that directly.
    num_workers = train_config["training"]["num_workers"]
    return DataLoader(
        dataset,
        batch_size=train_config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        # Page-locked memory makes the host-to-device copy faster and lets it
        # overlap with compute.
        pin_memory=True,
        # Without this the workers are torn down and respawned every epoch,
        # and each respawn re-imports torch and reopens the dataset.
        persistent_workers=num_workers > 0,
        # Each worker keeps this many batches queued ahead of the GPU.
        prefetch_factor=4 if num_workers > 0 else None,
    )


class Interaction:
    """Settings for iterative training and multi-click validation.

    `max_iters` 0 with `in_channels` 5 is the original single-click behaviour,
    and `run_epoch` then takes exactly the code path it always did.
    """

    def __init__(
        self,
        max_iters: int,
        val_clicks: int,
        in_channels: int,
        radius: int,
        click_stride: int,
        prev_mask_drop: float,
        seed: int,
    ) -> None:
        self.max_iters = max_iters
        self.val_clicks = val_clicks
        self.in_channels = in_channels
        self.radius = radius
        self.click_stride = click_stride
        self.prev_mask_drop = prev_mask_drop
        self.rng = np.random.default_rng(seed)

    @property
    def active(self) -> bool:
        return self.max_iters > 0 or self.in_channels != 5 or self.val_clicks > 1


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    interaction: Interaction | None = None,
) -> dict[str, float]:
    """One pass over `loader`. Trains if an optimizer is given, else evaluates.

    Returns per-sample means (so a smaller final batch doesn't skew them):
        loss        the training loss
        iou         the headline IoU: single-click, or after `val_clicks`
                    clicks when interaction is active
        best_of_n   oracle IoU over candidates, at the same click count
        iou_at_1    single-click IoU (equal to `iou` when not interactive)
    For a single-mask model iou and best_of_n are identical.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    interactive = interaction is not None and interaction.active

    totals = {"loss": 0.0, "iou": 0.0, "best_of_n": 0.0, "iou_at_1": 0.0}
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.shape[0]

            if is_train:
                if interactive:
                    # 0..max_iters corrective clicks, sampled per batch as RITM
                    # does, so the model sees every click count during training.
                    num_iters = int(interaction.rng.integers(0, interaction.max_iters + 1))
                    inputs = add_training_clicks(
                        model,
                        inputs,
                        targets,
                        num_iters,
                        in_channels=interaction.in_channels,
                        radius=interaction.radius,
                        click_stride=interaction.click_stride,
                        prev_mask_drop=interaction.prev_mask_drop,
                    )
                optimizer.zero_grad()
                logits, scores = model(inputs)
                loss = criterion(logits, targets, scores)
                loss.backward()
                optimizer.step()

                detached = logits.detach()
                detached_scores = scores.detach() if scores is not None else None
                iou = iou_score(detached, targets, scores=detached_scores)
                totals["loss"] += loss.item() * batch_size
                totals["iou"] += iou * batch_size
                totals["best_of_n"] += best_of_n_iou(detached, targets) * batch_size
                totals["iou_at_1"] += iou * batch_size  # not separately measured in training

            elif interactive:
                # The evaluation protocol: click, predict, correct, repeat. The
                # stride keeps click placement cheap enough to run every epoch;
                # scripts/evaluate.py uses stride 1 for the reported numbers.
                result = run_interaction(
                    model,
                    inputs,
                    targets,
                    interaction.val_clicks,
                    in_channels=interaction.in_channels,
                    radius=interaction.radius,
                    click_stride=interaction.click_stride,
                )
                loss = criterion(result["final_logits"], targets, result["final_scores"])
                totals["loss"] += loss.item() * batch_size
                totals["iou"] += result["iou"][:, -1].sum().item()
                totals["best_of_n"] += result["oracle"][:, -1].sum().item()
                totals["iou_at_1"] += result["iou"][:, 0].sum().item()

            else:
                logits, scores = model(inputs)
                loss = criterion(logits, targets, scores)
                iou = iou_score(logits, targets, scores=scores)
                totals["loss"] += loss.item() * batch_size
                totals["iou"] += iou * batch_size
                totals["best_of_n"] += best_of_n_iou(logits, targets) * batch_size
                totals["iou_at_1"] += iou * batch_size

            total_samples += batch_size

    return {key: value / total_samples for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset.root. Use an absolute /storage/... path in batch jobs, "
        "where a '~' in the config resolves against the wrong node's HOME.",
    )
    parser.add_argument("--clicks-config", default="configs/clicks.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override training.lr")
    parser.add_argument("--base-channels", type=int, default=None, help="Override model.base_channels")
    parser.add_argument("--depth", type=int, default=None, help="Override model.depth")
    parser.add_argument("--max-images", type=int, default=None, help="Override training.max_images (0 = all)")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.pt) to resume from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from <checkpoint-dir>/latest.pt if it exists. Lets a long run "
        "continue across several jobs after a walltime kill.",
    )
    parser.add_argument("--skip-test", action="store_true", help="Skip the final test-set evaluation")
    parser.add_argument(
        "--init-from",
        default=None,
        help="Start from this checkpoint's weights with a fresh optimizer (fine-tuning). "
        "Ignored if --auto-resume finds a checkpoint.",
    )
    parser.add_argument(
        "--prev-mask",
        action="store_true",
        help="Add a sixth input channel carrying the model's previous prediction",
    )
    parser.add_argument(
        "--iterative-clicks",
        type=int,
        default=0,
        help="Max corrective clicks added per training step (0 = single-click training as before)",
    )
    parser.add_argument(
        "--val-clicks",
        type=int,
        default=None,
        help="Clicks used for validation/test IoU; default = --iterative-clicks, or 1",
    )
    parser.add_argument("--prev-mask-drop", type=float, default=0.2, help="Fraction of samples whose previous mask is zeroed")
    parser.add_argument("--click-stride", type=int, default=4, help="Grid stride for click placement during training")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint.dir")
    parser.add_argument("--history-path", default=None, help="Override training.history_path")
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)
    with open(args.clicks_config) as f:
        click_config = yaml.safe_load(f)["clicks"]
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    if args.base_channels is not None:
        train_config["model"]["base_channels"] = args.base_channels
    if args.depth is not None:
        train_config["model"]["depth"] = args.depth

    training = train_config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.lr is not None:
        training["lr"] = args.lr
    if args.checkpoint_dir is not None:
        train_config["checkpoint"]["dir"] = args.checkpoint_dir
    if args.history_path is not None:
        training["history_path"] = args.history_path

    in_channels = 6 if args.prev_mask else 5
    train_config["model"]["in_channels"] = in_channels
    val_clicks = args.val_clicks if args.val_clicks is not None else max(1, args.iterative_clicks)
    if args.iterative_clicks > 0 and click_config.get("encoding", "disk") != "disk":
        raise SystemExit("iterative training draws disk clicks; set clicks.encoding: disk")
    if args.prev_mask and args.iterative_clicks == 0:
        print("WARNING: --prev-mask without --iterative-clicks: the previous-mask channel will only ever be zero")
    interaction = Interaction(
        max_iters=args.iterative_clicks,
        val_clicks=val_clicks,
        in_channels=in_channels,
        radius=int(click_config.get("radius", 5)),
        click_stride=args.click_stride,
        prev_mask_drop=args.prev_mask_drop,
        seed=train_config["seed"],
    )
    if interaction.active:
        print(
            f"Interactive training: up to {args.iterative_clicks} corrective clicks per step, "
            f"{in_channels} input channels, validation IoU after {val_clicks} click(s)"
        )

    torch.manual_seed(train_config["seed"])
    device = get_device(args.device or train_config.get("device", "auto"))
    print(f"Using device: {device}")

    root = Path(args.data_root or data_config["dataset"]["root"]).expanduser()
    image_paths = discover_samples(root)
    print(f"Found {len(image_paths)} images under {root}")

    # Deterministic subsample. Our data ablation (3k vs 10k images: +0.0045
    # test IoU) showed volume is not the limit, so training on fewer images at
    # higher resolution is a measured trade for iteration speed.
    max_images = args.max_images if args.max_images is not None else training.get("max_images")
    if max_images and len(image_paths) > max_images:
        rng = np.random.default_rng(training["split_seed"])
        keep = sorted(rng.choice(len(image_paths), size=max_images, replace=False))
        image_paths = [image_paths[i] for i in keep]
        print(f"Subsampled to {len(image_paths)} images (max_images={max_images})")
    if not image_paths:
        raise SystemExit(
            f"No images found under {root}.\n"
            f"  HOME is currently {Path.home()}.\n"
            "  On MetaCentrum, $HOME differs between nodes, so a '~' in "
            "configs/data.yaml can resolve to the wrong storage. Batch jobs "
            "should set HOME explicitly (see scripts/metacentrum/train.pbs) or "
            "use an absolute /storage/... path."
        )

    splits = split_image_paths(
        image_paths,
        ratios=tuple(training["splits"]),
        seed=training["split_seed"],
    )
    for name, paths in splits.items():
        print(f"  {name}: {len(paths)} images")

    # Validation and test use fixed clicks so their curves reflect the model,
    # not the click sampler. Training resamples every epoch as augmentation.
    train_loader = build_loader(splits["train"], train_config, click_config, shuffle=True, deterministic=False, split_name="train")
    val_loader = build_loader(splits["val"], train_config, click_config, shuffle=False, deterministic=True, split_name="val")
    print(
        f"Instances -> train: {len(train_loader.dataset)}  "
        f"val: {len(val_loader.dataset)}  (test built later)"
    )

    model = build_model(train_config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["lr"])
    criterion = MultiMaskLoss()

    # Drop the learning rate when validation IoU stops improving. Adam alone
    # bounces around the minimum at a fixed rate; decaying on plateau lets it
    # settle into it. `mode="max"` because we track IoU, where higher is better.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training["lr_decay_factor"],
        patience=training["lr_decay_patience"],
    )

    checkpoint_dir = Path(train_config["checkpoint"]["dir"])
    history_path = Path(training["history_path"])
    history: list[dict] = []
    best_val_iou = float("-inf")
    best_epoch = 0
    start_epoch = 1

    # A 200-epoch run cannot fit in one 12-hour job, so a long run is a chain
    # of jobs that each resume from the last checkpoint. Everything needed to
    # continue seamlessly (scheduler state, best score, history) is restored
    # here, not just the weights.
    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        candidate = checkpoint_dir / "latest.pt"
        if candidate.exists():
            resume_path = candidate
            print(f"Auto-resuming from {candidate}")
        else:
            print("No checkpoint found, starting from scratch")

    if resume_path is not None:
        ckpt = load_checkpoint(resume_path, model, optimizer, device, scheduler)
        start_epoch = ckpt["epoch"] + 1
        best_val_iou = ckpt.get("best_val_iou", float("-inf"))
        best_epoch = ckpt.get("best_epoch", 0)
        history = ckpt.get("history", [])
        print(
            f"Resumed from {resume_path} at epoch {start_epoch} "
            f"(best val IoU {best_val_iou:.4f} at epoch {best_epoch}, "
            f"{len(history)} epochs of history)"
        )
        if args.init_from:
            print(f"--init-from {args.init_from} ignored: resuming an existing run")
    elif args.init_from:
        # Weights only. The optimizer starts fresh: Adam's moments from a run
        # with a different objective (single-click) would mis-scale the first
        # updates of this one. A five-channel checkpoint is widened for a
        # six-channel model with zero filters, so step one reproduces the old
        # model exactly and the new channel grows in from there.
        source = torch.load(args.init_from, map_location=device, weights_only=False)
        state_dict = migrate_legacy_state_dict(source["model_state_dict"])
        state_dict = expand_input_channels(state_dict, in_channels)
        model.load_state_dict(state_dict)
        source_iou = source.get("best_val_iou")
        print(
            f"Initialised from {args.init_from} (epoch {source.get('epoch')}, "
            f"val IoU {source_iou:.4f}), optimizer fresh"
            if source_iou is not None
            else f"Initialised from {args.init_from}, optimizer fresh"
        )

    epochs = training["epochs"]
    patience = training["early_stopping_patience"]
    clicks_label = f"@{val_clicks}" if interaction.active else ""

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, criterion, device, optimizer, interaction)
        val = run_epoch(model, val_loader, criterion, device, interaction=interaction)
        elapsed = time.time() - started

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val["iou"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_iou": train["iou"],
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_best_of_n_iou": val["best_of_n"],
                "val_iou_at_1": val["iou_at_1"],
                "val_clicks": val_clicks,
                "lr": lr_now,
                "seconds": elapsed,
            }
        )
        # Written every epoch so a job that dies or hits its walltime still
        # leaves usable curves behind.
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        marker = ""
        if val["iou"] > best_val_iou:
            best_val_iou, best_epoch = val["iou"], epoch
            marker = "  <- best"

        state = {
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "history": history,
            "scheduler_state_dict": scheduler.state_dict(),
            # The settings that produced these weights, so a checkpoint found
            # months later can be exported and served correctly without anyone
            # having to remember which configs/ the run used. Resolution in
            # particular cannot be recovered from the weights: the network is
            # fully convolutional and will happily run at a size it was never
            # trained at, quietly and with no error.
            "train_settings": {
                "image_size": list(train_config["data"]["image_size"]),
                "clicks": dict(click_config),
                "in_channels": in_channels,
                "iterative_clicks": args.iterative_clicks,
                "val_clicks": val_clicks,
                # Which images the split was built from. evaluate.py needs this
                # to rebuild the same validation split, not one that overlaps
                # the training images.
                "max_images": int(max_images or 0),
                "init_from": args.init_from,
            },
        }
        if marker:
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, val["loss"], state)

        extra = ""
        if interaction.active:
            extra = f" IoU@1={val['iou_at_1']:.4f}"
        if val["best_of_n"] > val["iou"] + 1e-6:
            extra += f" (best-of-N {val['best_of_n']:.4f})"
        print(
            f"epoch {epoch:3d}/{epochs}  "
            f"train loss={train['loss']:.4f} IoU={train['iou']:.4f}  |  "
            f"val loss={val['loss']:.4f} IoU{clicks_label}={val['iou']:.4f}{extra}"
            f"  lr={lr_now:.2e}  ({elapsed:.0f}s){marker}"
        )
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, model, optimizer, train["loss"], state)

        # Stop once validation has not improved for `patience` epochs. Training
        # past that point only widens the train/validation gap.
        if epoch - best_epoch >= patience:
            print(
                f"\nEarly stopping: no validation improvement in {patience} epochs "
                f"(best was epoch {best_epoch})"
            )
            break

    print(f"\nBest validation IoU{clicks_label}: {best_val_iou:.4f} at epoch {best_epoch}")

    if args.skip_test:
        print("Skipping test evaluation (--skip-test).")
        return

    # Load the best-validation weights before touching the test set: reporting
    # the *last* epoch's weights would be reporting a model we never selected.
    print("Evaluating best checkpoint on the held-out test split...")
    load_checkpoint(checkpoint_dir / "best.pt", model, optimizer, device)
    test_loader = build_loader(splits["test"], train_config, click_config, shuffle=False, deterministic=True, split_name="test")
    test = run_epoch(model, test_loader, criterion, device, interaction=interaction)
    extra = f"  IoU@1={test['iou_at_1']:.4f}" if interaction.active else ""
    print(
        f"Test loss={test['loss']:.4f}  Test IoU{clicks_label}={test['iou']:.4f}{extra}"
        + (f"  (best-of-N {test['best_of_n']:.4f})" if test["best_of_n"] > test["iou"] + 1e-6 else "")
        + f"  ({len(test_loader.dataset)} instances)"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "val_clicks": val_clicks,
        "test_loss": test["loss"],
        "test_iou": test["iou"],
        "test_iou_at_1": test["iou_at_1"],
        "test_best_of_n_iou": test["best_of_n"],
        "splits": {name: len(paths) for name, paths in splits.items()},
        "history": history,
    }
    with open(history_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote history and results to {history_path}")


if __name__ == "__main__":
    main()
