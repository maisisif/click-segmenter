"""Segmentation losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6


class DiceLoss(nn.Module):
    def __init__(self, eps: float = EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE calibrates per-pixel probabilities; Dice directly optimizes mask
    overlap and is far less sensitive to the foreground/background pixel-count
    imbalance that plain BCE struggles with on small objects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logits, target) + self.dice(logits, target)


class MultiMaskLoss(nn.Module):
    """SAM-style ambiguity-aware loss over several candidate masks.

    A single click is often genuinely ambiguous: clicking a person's shirt
    could mean the shirt, the torso, or the whole person, and all three
    contain that pixel. A model forced to emit one mask hedges and blurs
    between them. Emitting M candidates and back-propagating only through
    the one that best matches the target lets each candidate specialise on a
    different interpretation instead of averaging.

    Shapes: logits (B, M, H, W), target (B, 1, H, W), scores (B, M) or None.

    The score head predicts each candidate's IoU. It is trained on ALL
    candidates (not just the winner) with the true IoU as target, detached so
    it never pushes gradient back into the masks. At inference there is no
    ground truth, so this head is what decides which candidate to show.

    With M == 1 this reduces to plain BCE+Dice, so the same loss covers both
    the single-mask and multi-mask configurations.
    """

    def __init__(self, score_weight: float = 1.0, eps: float = EPS) -> None:
        super().__init__()
        self.score_weight = score_weight
        self.eps = eps

    def _per_candidate(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE+Dice for every (sample, candidate) pair -> (B, M)."""
        num_masks = logits.shape[1]
        expanded = target.expand(-1, num_masks, -1, -1)

        bce = F.binary_cross_entropy_with_logits(logits, expanded, reduction="none").mean(dim=(2, 3))

        probs = torch.sigmoid(logits)
        intersection = (probs * expanded).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + expanded.sum(dim=(2, 3))
        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)

        return bce + dice

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        per_candidate = self._per_candidate(logits, target)
        best = per_candidate.min(dim=1).values.mean()

        if scores is None:
            return best

        with torch.no_grad():
            num_masks = logits.shape[1]
            expanded = target.expand(-1, num_masks, -1, -1)
            hard = (torch.sigmoid(logits) > 0.5).float()
            intersection = (hard * expanded).sum(dim=(2, 3))
            union = ((hard + expanded) > 0).float().sum(dim=(2, 3))
            true_iou = (intersection + self.eps) / (union + self.eps)

        return best + self.score_weight * F.mse_loss(scores, true_iou)
