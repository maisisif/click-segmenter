"""Segmentation evaluation metrics."""

from __future__ import annotations

import torch

EPS = 1e-6


@torch.no_grad()
def iou_per_sample(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = EPS
) -> torch.Tensor:
    """IoU of every (sample, candidate) pair.

    logits (B, M, H, W), target (B, 1, H, W) -> (B, M).
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    expanded = target.expand(-1, logits.shape[1], -1, -1)
    intersection = (preds * expanded).sum(dim=(2, 3))
    union = ((preds + expanded) > 0).float().sum(dim=(2, 3))
    return (intersection + eps) / (union + eps)


@torch.no_grad()
def select_masks(logits: torch.Tensor, scores: torch.Tensor | None = None) -> torch.Tensor:
    """Choose one candidate per sample WITHOUT looking at ground truth.

    This is what inference does, so it is also what the headline metric must
    use. Selection is by predicted score; if the model has no score head, by
    highest mean predicted probability, which is a weak but usable proxy.

    Returns (B, 1, H, W).
    """
    if logits.shape[1] == 1:
        return logits

    if scores is None:
        chosen = torch.sigmoid(logits).mean(dim=(2, 3)).argmax(dim=1)
    else:
        chosen = scores.argmax(dim=1)

    index = chosen.view(-1, 1, 1, 1).expand(-1, 1, *logits.shape[2:])
    return logits.gather(1, index)


@torch.no_grad()
def iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = EPS,
    scores: torch.Tensor | None = None,
) -> float:
    """Mean IoU of the mask the system would actually return.

    For a single-mask model this is the plain IoU it always was, so numbers
    stay comparable with earlier runs.
    """
    selected = select_masks(logits, scores)
    return iou_per_sample(selected, target, threshold, eps).mean().item()


@torch.no_grad()
def best_of_n_iou(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = EPS
) -> float:
    """Mean IoU of the BEST candidate, chosen using ground truth.

    An oracle upper bound, not an achievable score: it says how good the
    candidates are, ignoring whether selection can find the good one. The gap
    between this and `iou_score` is exactly how much the score head is losing.
    """
    return iou_per_sample(logits, target, threshold, eps).max(dim=1).values.mean().item()
