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


class HungarianMaskLoss(nn.Module):
    """Set-prediction loss for the slot architecture: match, then score.

    The model emits `num_slots` class-agnostic masks in a fixed tensor, but the
    slots are unordered -- nothing says which object belongs in slot 7, and
    nothing should. Imposing an order (by area, say) would force the network to
    learn "am I the third-largest object here", which is an unstable target the
    moment two objects are similar in size.

    So each image's predictions are paired to its ground-truth objects by
    *optimal bipartite matching*: build the full cost matrix of every prediction
    against every object, solve it exactly (Hungarian algorithm), and supervise
    only the pairs that assignment chose. Slots stay interchangeable, and the
    network converges on using them consistently by itself. This is the DETR /
    Mask2Former formulation.

    Matching is done under no_grad on detached tensors -- it decides *which*
    pairs are compared, and should not itself be something the model can
    influence to make its loss smaller.

    Slots left unmatched are not free: their objectness is pushed towards "no
    object". That term is down-weighted (`no_object_weight`), because with 64
    slots and typically ~20 objects most slots are empty on any example, and an
    unweighted loss would be dominated by predicting emptiness.

    Shapes:
        mask_logits  (B, N, h, w)
        objectness   (B, N)
        targets      list of B tensors, each (K_i, H, W), K_i >= 0 and variable

    Targets are resized to the logits' resolution if they differ, so the caller
    can pass full-resolution masks without knowing the model emits quarter-size
    ones.
    """

    def __init__(
        self,
        objectness_weight: float = 1.0,
        no_object_weight: float = 0.1,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.objectness_weight = objectness_weight
        self.no_object_weight = no_object_weight
        self.eps = eps

    def _cost_matrix(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pairwise BCE + Dice between N predictions and K objects -> (N, K).

        Computed with matrix products rather than an N*K loop: for BCE, the
        per-pixel cost of predicting logit x against label 1 is softplus(-x) and
        against label 0 is softplus(x), so the pairwise total is two matmuls
        against the target and its complement. Same trick as DETR.
        """
        num_pixels = logits.shape[1]

        pos = F.softplus(-logits)  # cost of calling each pixel foreground
        neg = F.softplus(logits)   # cost of calling each pixel background
        bce = (pos @ target.t() + neg @ (1 - target).t()) / num_pixels

        probs = torch.sigmoid(logits)
        intersection = probs @ target.t()
        union = probs.sum(dim=1, keepdim=True) + target.sum(dim=1).unsqueeze(0)
        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)

        return bce + dice

    def _matched_mask_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE + Dice for the matched pairs only, both (M, P)."""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=1)

        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        dice = 1 - (2 * intersection + self.eps) / (union + self.eps)

        return (bce + dice).sum()

    def forward(
        self,
        mask_logits: torch.Tensor,
        objectness: torch.Tensor,
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        # Imported here rather than at module scope: the click-conditioned
        # losses above are used on machines where scipy may not be installed.
        from scipy.optimize import linear_sum_assignment

        batch, num_slots, height, width = mask_logits.shape
        objectness_target = torch.zeros_like(objectness)

        mask_loss = mask_logits.sum() * 0.0  # zero that carries the graph
        num_matched = 0

        for i, target in enumerate(targets):
            if target.numel() == 0 or target.shape[0] == 0:
                continue

            if target.shape[-2:] != (height, width):
                target = F.interpolate(
                    target.unsqueeze(1), size=(height, width), mode="nearest"
                ).squeeze(1)

            flat_logits = mask_logits[i].flatten(1)  # (N, P)
            flat_target = target.flatten(1).float()  # (K, P)

            with torch.no_grad():
                cost = self._cost_matrix(flat_logits.detach(), flat_target)
                rows, cols = linear_sum_assignment(cost.cpu().numpy())

            rows = torch.as_tensor(rows, device=mask_logits.device, dtype=torch.long)
            cols = torch.as_tensor(cols, device=mask_logits.device, dtype=torch.long)

            mask_loss = mask_loss + self._matched_mask_loss(flat_logits[rows], flat_target[cols])
            num_matched += len(rows)
            objectness_target[i, rows] = 1.0

        mask_loss = mask_loss / max(num_matched, 1)

        # Weight the empty slots down so the objectness term is not dominated by
        # the many slots that are correctly empty on any given image.
        weight = torch.where(
            objectness_target > 0,
            torch.ones_like(objectness_target),
            torch.full_like(objectness_target, self.no_object_weight),
        )
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness, objectness_target, weight=weight
        )

        return mask_loss + self.objectness_weight * objectness_loss
