"""Checkpoint saving and loading, shared by the training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint atomically.

    `extra` carries anything else needed to resume exactly where the run left
    off (scheduler state, best score so far, the history list). MetaCentrum
    jobs get killed at their walltime, so a long run is a chain of jobs that
    each pick up from the last checkpoint -- anything not saved here is lost
    at every handover.

    Written to a temporary file and renamed, because a job killed midway
    through a plain torch.save leaves a truncated file that cannot be loaded,
    which would destroy the run it was meant to protect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    if extra:
        payload.update(extra)

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    """Load state in place and return the full checkpoint payload.

    The caller needs more than the epoch number to resume cleanly (best score
    so far, accumulated history), so the whole dict comes back.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
