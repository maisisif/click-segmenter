"""Simulated multi-click interaction, shared by training and evaluation.

Everything measured so far is single-click IoU. The field's metric for an
interactive tool is what happens over a *sequence* of clicks: mean IoU after k
clicks (mIoU@k) and the number of clicks needed to reach a target IoU (NoC@85,
NoC@90). Both need a rule for where the next click goes, and that rule is the
one RITM and SimpleClick use for evaluation:

    Take the error regions -- pixels of the object the mask missed (false
    negatives) and pixels outside it the mask included (false positives).
    Put the next click at the point deepest inside the larger of the two
    (largest distance to the region's boundary). Positive if it was a miss,
    negative if it was spurious.

The same rule drives *iterative training*: run the model, place a corrective
click where it went wrong, run again, and supervise the corrected prediction.
Without that, the model only ever sees one positive click during training and
a second click is out of distribution -- which is exactly why extra clicks
barely helped in the interface.

Coordinates are (y, x) at the model's working resolution. Clicks are drawn as
disks straight into the input tensor's click channels, so the encoding must be
`disk`; the `distance` encoding is not supported here.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from src.training.metrics import iou_per_sample, select_masks

# Input channel layout, fixed by src/data/encoding.py and the dataset.
POSITIVE_CHANNEL = 3
NEGATIVE_CHANNEL = 4
PREVIOUS_MASK_CHANNEL = 5

NextClick = tuple[int, int, bool] | None  # (y, x, positive), or None when there is no error


def _boundary_distance(region: np.ndarray) -> np.ndarray | None:
    """Distance of every pixel in `region` from the region's edge.

    Padded by one pixel first so the image border counts as an edge; otherwise
    a region touching the border would put its "deepest" point on the border.
    """
    if not region.any():
        return None
    padded = np.pad(region, 1, mode="constant", constant_values=False)
    return distance_transform_edt(padded)[1:-1, 1:-1]


def next_click(pred: np.ndarray, target: np.ndarray) -> NextClick:
    """Where a simulated user clicks next, given the current mask and the truth.

    Both arrays are bool (H, W). Returns None when the prediction is exact.
    Deterministic, so evaluation is reproducible.
    """
    fn_dt = _boundary_distance(target & ~pred)
    fp_dt = _boundary_distance(pred & ~target)
    fn_max = float(fn_dt.max()) if fn_dt is not None else 0.0
    fp_max = float(fp_dt.max()) if fp_dt is not None else 0.0
    if fn_max == 0.0 and fp_max == 0.0:
        return None

    positive = fn_max >= fp_max
    dt = fn_dt if positive else fp_dt
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return int(y), int(x), bool(positive)


def next_clicks(pred: torch.Tensor, target: torch.Tensor, stride: int = 1) -> list[NextClick]:
    """`next_click` for a batch. pred, target: (B, 1, H, W) bool tensors.

    `stride` > 1 places clicks on a subsampled grid. The distance transform of
    a 384x512 map costs ~5-10 ms on one CPU core, twice per sample per click;
    at stride 4 it is well under a millisecond, and a click that lands within
    two pixels of the ideal point is indistinguishable to a model that sees a
    radius-5 disk. Training uses the stride; evaluation should not, so that the
    protocol matches the published one exactly.
    """
    if stride > 1:
        pred = pred[..., ::stride, ::stride]
        target = target[..., ::stride, ::stride]
    pred_np = pred[:, 0].cpu().numpy().astype(bool)
    target_np = target[:, 0].cpu().numpy().astype(bool)

    clicks: list[NextClick] = []
    for p, t in zip(pred_np, target_np):
        click = next_click(p, t)
        if click is not None and stride > 1:
            y, x, positive = click
            click = (y * stride + stride // 2, x * stride + stride // 2, positive)
        clicks.append(click)
    return clicks


def stamp_clicks(inputs: torch.Tensor, clicks: list[NextClick], radius: int) -> None:
    """Draw each click as a filled disk into the input's click channels, in place.

    Matches `_encode_disk` in src/data/encoding.py, which is what the model was
    trained on. Clicks are clamped into the image.
    """
    _, _, height, width = inputs.shape
    yy = torch.arange(height, device=inputs.device).view(height, 1)
    xx = torch.arange(width, device=inputs.device).view(1, width)
    for b, click in enumerate(clicks):
        if click is None:
            continue
        y, x, positive = click
        y = min(max(y, 0), height - 1)
        x = min(max(x, 0), width - 1)
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        channel = POSITIVE_CHANNEL if positive else NEGATIVE_CHANNEL
        inputs[b, channel][disk] = 1.0


def with_previous_mask_channel(inputs: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Append an all-zero previous-mask channel when the model expects one.

    A dataset item has five channels. A model trained with previous-mask input
    has six, the last holding its own prediction from the step before -- zero
    before the first click, which is also what it was trained to see there.
    """
    if in_channels == inputs.shape[1]:
        return inputs
    if in_channels == inputs.shape[1] + 1:
        return torch.cat([inputs, torch.zeros_like(inputs[:, :1])], dim=1)
    raise ValueError(f"model takes {in_channels} channels but the batch has {inputs.shape[1]}")


def binary_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """IoU of bool masks, per sample. pred, target: (B, 1, H, W) -> (B,)."""
    intersection = (pred & target).sum(dim=(1, 2, 3)).float()
    union = (pred | target).sum(dim=(1, 2, 3)).float()
    return (intersection + eps) / (union + eps)


def select_consistent(
    logits: torch.Tensor,
    scores: torch.Tensor | None,
    inputs: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Score-head selection, restricted to candidates that agree with the clicks.

    A candidate that leaves a positive click outside the mask, or swallows a
    negative one, contradicts what the user just said. Dropping those before
    consulting the score head costs nothing and needs no retraining. Falls back
    to plain score selection when no candidate is consistent.

    Returns (B, 1, H, W), like `select_masks`.
    """
    if logits.shape[1] == 1:
        return logits

    preds = torch.sigmoid(logits) > threshold  # (B, M, H, W)
    positive = inputs[:, POSITIVE_CHANNEL : POSITIVE_CHANNEL + 1] > 0.5
    negative = inputs[:, NEGATIVE_CHANNEL : NEGATIVE_CHANNEL + 1] > 0.5

    eps = 1e-6
    pos_covered = (preds & positive).sum(dim=(2, 3)).float() / (positive.sum(dim=(2, 3)).float() + eps)
    neg_covered = (preds & negative).sum(dim=(2, 3)).float() / (negative.sum(dim=(2, 3)).float() + eps)
    consistent = (pos_covered >= 0.5) & (neg_covered <= 0.5)

    if scores is None:
        scores = torch.sigmoid(logits).mean(dim=(2, 3))
    masked = scores.masked_fill(~consistent, float("-inf"))
    any_consistent = consistent.any(dim=1, keepdim=True)
    chosen = torch.where(any_consistent, masked, scores).argmax(dim=1)

    index = chosen.view(-1, 1, 1, 1).expand(-1, 1, *logits.shape[2:])
    return logits.gather(1, index)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    selection: str = "score",
    flip_tta: bool = False,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One forward pass, returning (selected probs (B,1,H,W), all logits, scores).

    `selection` is "score" (the score head, what the app does today) or
    "consistent" (see `select_consistent`). `flip_tta` also runs the
    horizontally mirrored input and averages the two *selected* probability
    maps -- selected first, then averaged, because candidate i of the mirrored
    pass need not mean the same thing as candidate i of the original.
    """
    logits, scores = model(inputs)
    if selection == "consistent":
        selected = select_consistent(logits, scores, inputs, threshold)
    else:
        selected = select_masks(logits, scores)
    probs = torch.sigmoid(selected)

    if flip_tta:
        mirrored = inputs.flip(-1)
        logits_m, scores_m = model(mirrored)
        if selection == "consistent":
            selected_m = select_consistent(logits_m, scores_m, mirrored, threshold)
        else:
            selected_m = select_masks(logits_m, scores_m)
        probs = 0.5 * (probs + torch.sigmoid(selected_m).flip(-1))

    return probs, logits, scores


@torch.no_grad()
def add_training_clicks(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_iters: int,
    *,
    in_channels: int = 5,
    radius: int = 5,
    threshold: float = 0.5,
    click_stride: int = 4,
    prev_mask_drop: float = 0.0,
) -> torch.Tensor:
    """RITM-style iterative click sampling for one training batch.

    Runs the model `num_iters` times without gradient, each time adding the
    corrective click the evaluation protocol would add, and returns the input
    the supervised step should train on. With a previous-mask channel the
    model's own probabilities are written there, and `prev_mask_drop` zeroes
    that channel for a random fraction of samples so the model cannot learn to
    simply copy its previous answer.

    The model is switched to eval mode for the sampling passes (so BatchNorm
    statistics are not updated by them) and restored afterwards.
    """
    inputs = with_previous_mask_channel(inputs.clone(), in_channels)
    uses_prev = in_channels == PREVIOUS_MASK_CHANNEL + 1
    target_bool = targets > 0.5

    was_training = model.training
    model.eval()
    for _ in range(num_iters):
        probs, _, _ = predict_probs(model, inputs, threshold=threshold)
        stamp_clicks(inputs, next_clicks(probs > threshold, target_bool, stride=click_stride), radius)
        if uses_prev:
            inputs[:, PREVIOUS_MASK_CHANNEL : PREVIOUS_MASK_CHANNEL + 1] = probs

    if uses_prev and prev_mask_drop > 0 and num_iters > 0:
        drop = torch.rand(inputs.shape[0], device=inputs.device) < prev_mask_drop
        inputs[drop, PREVIOUS_MASK_CHANNEL] = 0.0

    if was_training:
        model.train()
    return inputs


@torch.no_grad()
def run_interaction(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_clicks: int,
    *,
    in_channels: int = 5,
    radius: int = 5,
    threshold: float = 0.5,
    selection: str = "score",
    flip_tta: bool = False,
    click_stride: int = 1,
) -> dict[str, torch.Tensor]:
    """The evaluation protocol: click, predict, correct, repeat.

    `inputs` come from the dataset with the first click already drawn in.
    Each round predicts, records the IoU, and adds one corrective click. A
    model with a previous-mask channel is fed its own last probability map.

    Returns per-sample tensors of shape (B, num_clicks):
        iou     -- IoU of the returned mask after k clicks
        oracle  -- IoU of the best candidate after k clicks (ground-truth
                   selection; an upper bound on what any selector could get)
    plus `final_logits`, `final_scores` and `final_inputs` from the last round.
    """
    model.eval()
    inputs = with_previous_mask_channel(inputs.clone(), in_channels)
    uses_prev = in_channels == PREVIOUS_MASK_CHANNEL + 1
    target_bool = targets > 0.5

    ious = []
    oracles = []
    logits = scores = None
    for k in range(num_clicks):
        probs, logits, scores = predict_probs(
            model, inputs, selection=selection, flip_tta=flip_tta, threshold=threshold
        )
        pred = probs > threshold
        ious.append(binary_iou(pred, target_bool))
        oracles.append(iou_per_sample(logits, targets, threshold).max(dim=1).values)

        if k + 1 < num_clicks:
            stamp_clicks(inputs, next_clicks(pred, target_bool, stride=click_stride), radius)
            if uses_prev:
                inputs[:, PREVIOUS_MASK_CHANNEL : PREVIOUS_MASK_CHANNEL + 1] = probs

    return {
        "iou": torch.stack(ious, dim=1),
        "oracle": torch.stack(oracles, dim=1),
        "final_logits": logits,
        "final_scores": scores,
        "final_inputs": inputs,
    }


def clicks_to_reach(iou: torch.Tensor, target_iou: float, max_clicks: int | None = None) -> torch.Tensor:
    """Number of clicks until IoU first reaches `target_iou`, per sample.

    `iou` is (B, K). Samples that never get there count as `max_clicks`
    (default K), which is how NoC is conventionally reported -- the cap is part
    of the metric's definition, so quote it alongside the number.
    """
    batch, num_clicks = iou.shape
    cap = max_clicks if max_clicks is not None else num_clicks
    reached = iou >= target_iou
    never = torch.full((batch,), cap, device=iou.device, dtype=torch.long)
    first = torch.where(reached.any(dim=1), reached.float().argmax(dim=1) + 1, never)
    return first.clamp(max=cap)
