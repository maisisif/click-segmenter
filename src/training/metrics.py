"""Segmentation evaluation metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def iou_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = ((preds + target) > 0).float().sum(dim=(1, 2, 3))
    return ((intersection + eps) / (union + eps)).mean().item()
