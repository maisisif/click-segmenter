# PROGRESS

Living state of the project. `CLAUDE.md` holds the fixed plan; this file tracks
where things actually stand. Read both at the start of a session.

Last updated: 2026-08-07

## Project summary

An interactive click-to-segment tool: the user clicks an object in an image and
the system returns a segmentation mask for that specific instance. The model is
a UNet-style encoder-decoder trained from scratch (no SAM or other pretrained
interactive segmenter), taking a 5-channel input (RGB + positive/negative click
maps). Data is ADE20K at instance level, using its per-instance amodal masks.
Training runs on MetaCentrum; development is local. The codebase is kept modular
so a scene-object-graph stage can later consume the segmentation outputs.

## Milestones

- **M1 — Data loading + visualization** ✅ Loads ADE20K images with per-instance
  amodal masks (`src/data/ade20k.py`), overlays saved to `outputs/`.
- **M2 — Click simulation + input encoding** ✅ Positive clicks sampled toward
  mask interior, optional negative clicks in a band outside the boundary
  (`src/data/clicks.py`), encoded as 2 extra channels (`src/data/encoding.py`).
- **M3 — Baseline UNet + overfit sanity check** ✅ Depth-3 UNet memorizes all 4
  overfit instances to **IoU 1.0000** at `base_channels: 32`, confirming the
  data pipeline, loss and training loop are correct.
- **M4 — Full training on MetaCentrum** 🔄 `scripts/train_full.py` is written
  (70/20/10 splits, per-epoch validation, best-epoch selection, held-out test
  evaluation, JSON history). Not yet run: needs the full dataset and a working
  GPU node.
- **M5 — Evaluation (IoU/Dice/NoC) + SAM baseline comparison** ⬜ Basic IoU is
  in place; Dice and NoC are not.
- **M6 — Interactive UI** ⬜
- **M7 — Polish (README, results, demo GIF)** ⬜

## Current status

The pipeline runs end to end on MetaCentrum from a clean clone, on CPU, inside
the OnDemand JupyterLab container.

**M3 is resolved.** It took two causes, found by bisecting the overfit subset:

1. *Learning rate.* At `lr: 0.003` training destabilized deterministically
   around epoch 587. At `lr: 0.001` that disappears entirely.
2. *Model capacity.* At `base_channels: 16` the model memorizes 1, 2 or 3
   instances to IoU 1.0000 but stalls at ~0.91 on 4. At `base_channels: 32` all
   4 reach 1.0000 by epoch 500. A model that cannot memorize 4 examples is far
   too small for ~700k instances, so this directly informs M4.

Two hypotheses were tested and falsified along the way: that the click signal
couldn't propagate far enough (disproved — 3 instances including two from the
same image converge perfectly), and that mask fragmentation was to blame
(disproved — the most fragmented instance, `building` with 10 components,
converges fine). The distance-transform click encoding added while chasing the
first hypothesis remains available via `clicks.encoding: distance` but is
non-default and untested.

The loss and metric code were reviewed and found correct (loss 0.118 decomposed
exactly to IoU 0.90). Best-checkpoint tracking was added along the way, so a
late spike can no longer discard an otherwise good run.

No GPU run has succeeded yet, and no training on the full dataset has been
attempted.

## Supervisor requirements (Kassem, 2026-08-07)

Agreed direction for the deliverable, which shapes M4/M5:

- Split 70/20/10 train/validation/test. Train on 70, validate on 20, pick the
  best-performing epoch, then evaluate that epoch once on the 10% test set.
- Report back with results and training curves.
- Deliver as a Jupyter notebook with all plots rendered inline.
- After the model works: save weights, build a GUI that runs inference from
  them, then the scene-graph stage.
- Spend time on the underlying theory, not just the code.

## Blocked on

1. **GPU node incompatibility.** The OnDemand `NGC/PyTorch:25.02-py3.SIF`
   container ships CUDA kernels for compute capability sm_75 and newer. The
   `konos` cluster is GTX 1080 Ti (sm_61), so `torch.cuda.is_available()` returns
   True but any kernel launch fails with "no kernel image is available for
   execution on the device". Needs a session on sm_75+ hardware (`galdor`,
   `fobos`, `zia`, `bee`, `fer`, `glados`). The OnDemand launch form exposes no
   `gpu_cap` field, only a PBS Queue dropdown, so how to pin the hardware is
   still open.

2. **Full dataset not yet materialized.** ADE20K official registration
   (ade20k.csail.mit.edu) has been broken for weeks. Decision: use the Hugging
   Face mirror `1aurent/ADE20K` (full 2021 release). Its schema and ternary
   {0, 128, 255} amodal mask encoding were verified, and
   `scripts/export_ade20k.py` is written, but has not been run. COCO and PSG
   were both ruled out.

## Next steps

1. Dry-run the M4 script on the 3-image sample to shake out bugs before
   committing to a real run:
   `python3 scripts/train_full.py --device cpu --epochs 2`
2. Submit `scripts/metacentrum/export_data.pbs` from a frontend over SSH
   (`qsub` is unavailable inside the OnDemand Jupyter container). The job is
   self-contained: it builds its own small venv, caches Hugging Face downloads
   on node-local scratch rather than home, and defaults to 3000 train / 600 val
   images rather than the full 25,574. Exporting everything means ~700k small
   PNGs, which risks hitting an inode quota; a few thousand images is enough to
   train a genuinely useful model and is much faster to get moving.
3. Get an OnDemand session on a GPU with compute capability >= 7.5 and confirm
   CUDA works.
4. Run full training, then build the notebook with curves and results.

## Environment notes

**Local dev:** macOS, Apple MPS. Repo at `~/projects/click-segmenter`, ADE20K
toolkit (with 3-image sample) at `~/projects/ade20k-reference`.

**MetaCentrum:** home directory is shared across nodes, so files persist between
sessions and clusters. Same layout as local.

Access is via OnDemand → Interactive Apps → Jupyter Notebook/Lab, image
`NGC/PyTorch:25.02-py3.SIF`. That container already provides torch 2.7.0a0 with
CUDA, numpy, pillow, matplotlib, pyyaml and scipy, so **no pip installs or venv
are needed** for training. `git` is available inside it; `qsub` and `module` are
not.

`scripts/metacentrum/setup_env.sh` builds a module-based venv instead, and is
only relevant for plain SSH sessions on a frontend, not the container workflow.

**Configs** (`configs/`): `data.yaml` (dataset root), `clicks.yaml` (click
simulation), `train.yaml` (seed, device, model width, image size, `overfit`
settings for M3, `training` settings for M4, checkpoint dir/frequency).

**Entry points:**

- `scripts/train.py` — M3 overfit sanity check only. Accepts `--device`,
  `--subset-size`, `--epochs`, `--lr`, `--resume`.
- `scripts/train_full.py` — M4 real training. Accepts `--device`, `--epochs`,
  `--batch-size`, `--lr`, `--resume`, `--skip-test`.
- `scripts/export_ade20k.py` — HF mirror to on-disk layout.

`device: auto` picks cuda > mps > cpu; naming a device explicitly makes it fail
loudly if unavailable, which is what a cluster job wants.

**Splits:** `src/data/splits.py` splits by **image**, never by instance, since
instances from one image appearing in both train and val would leak memorized
scenes into validation. Deterministic from the sorted paths plus `split_seed`,
so no split file needs to be stored.

**Checkpoints:** `outputs/checkpoints/best.pt` (best validation IoU in M4, best
training IoU in M3) and `latest.pt`. Prefer `best.pt`.

**History:** `scripts/train_full.py` writes per-epoch metrics to
`outputs/history.json` every epoch, so a job killed by walltime still leaves
plottable curves.
