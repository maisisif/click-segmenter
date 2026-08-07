"""Checkpoint saving and loading, shared by the training entry points."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> int:
    """Loads model/optimizer state in place, returns the epoch it was saved at."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
