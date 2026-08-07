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
- **M3 — Baseline UNet + overfit sanity check** 🔄 Depth-3 UNet (16/32/64/128)
  trains and learns, but the overfit check plateaus at IoU ~0.90 rather than the
  near-perfect memorization this test is supposed to demonstrate. Under
  investigation — see "Blocked on".
- **M4 — Full training on MetaCentrum** ⬜ Cluster-readiness work is done (lazy
  loading, checkpoint/resume, config-driven device), but no full run yet.
- **M5 — Evaluation (IoU/Dice/NoC) + SAM baseline comparison** ⬜
- **M6 — Interactive UI** ⬜
- **M7 — Polish (README, results, demo GIF)** ⬜

## Current status

The full pipeline runs end to end on MetaCentrum from a clean clone, on CPU,
inside the OnDemand JupyterLab container. Latest verified result: on the bundled
3-image sample (109 instances), the 4-example overfit check reaches a best
training IoU of **0.9064 at epoch 587**, then destabilizes on the final epochs
(loss 0.118 → 0.552) and ends at eval IoU 0.2939.

Runs are bit-identical across repeats, so that late blow-up is deterministic
under `seed: 0` and `lr: 0.003`, not random. Best-checkpoint tracking now saves
`best.pt` at the peak, so a late spike no longer discards a good run.

No GPU run has succeeded yet. No training on the full dataset has been attempted.

## Blocked on

1. **M3 is not conclusively passing.** An overfit check on a handful of fixed
   examples should reach near-zero loss and IoU ~0.97+; ours flatlines at ~0.90
   from epoch 500 on. The loss and metric code were reviewed and are correct
   (loss 0.118 decomposes exactly to IoU 0.90), and the overfit subset contains
   no duplicate targets. The leading hypothesis is that `lr: 0.003` is too high
   to settle into the minimum, supported by the deterministic divergence at
   epoch 587. Not yet confirmed.

2. **GPU node incompatibility.** The OnDemand `NGC/PyTorch:25.02-py3.SIF`
   container ships CUDA kernels for compute capability sm_75 and newer. The
   `konos` cluster is GTX 1080 Ti (sm_61), so `torch.cuda.is_available()` returns
   True but any kernel launch fails with "no kernel image is available for
   execution on the device". Needs a session on sm_75+ hardware (`galdor`,
   `fobos`, `zia`, `bee`, `fer`, `glados`). The OnDemand launch form exposes no
   `gpu_cap` field, only a PBS Queue dropdown, so how to pin the hardware is
   still open.

3. **Full dataset not yet materialized.** ADE20K official registration
   (ade20k.csail.mit.edu) has been broken for weeks. Decision: use the Hugging
   Face mirror `1aurent/ADE20K` (full 2021 release). Its schema and ternary
   {0, 128, 255} amodal mask encoding were verified, and `scripts/export_ade20k.py`
   is written, but has not been run. COCO and PSG were both ruled out.

## Next steps

1. Run the M3 diagnostic to test the learning-rate hypothesis:
   `python3 scripts/train.py --subset-size 1 --epochs 2000 --lr 0.001`.
   Expect IoU > 0.97. If a single example still caps at ~0.90, investigate the
   click encoding and mask resizing instead.
2. Get an OnDemand session on a GPU with compute capability >= 7.5 and rerun
   `python3 scripts/train.py` to confirm CUDA works.
3. Submit `scripts/metacentrum/export_data.pbs` to materialize the full dataset.
   Note `qsub` is unavailable inside the Jupyter container, so submit via the
   OnDemand Job Composer or an SSH session to a frontend.

## Environment notes

**Local dev:** macOS, Apple MPS. Repo at `~/projects/click-segmenter`, ADE20K
toolkit (with 3-image sample) at `~/projects/ade20k-reference`.

**MetaCentrum:** home directory is shared across nodes, so files persist between
sessions and clusters. Same layout as local: `~/projects/click-segmenter` and
`~/projects/ade20k-reference`.

Access is via OnDemand → Interactive Apps → Jupyter Notebook/Lab, image
`NGC/PyTorch:25.02-py3.SIF`. That container already provides torch 2.7.0a0 with
CUDA, numpy, pillow, matplotlib, pyyaml and scipy, so **no pip installs or venv
are needed**. `git` is available inside it; `qsub` and `module` are not.

`scripts/metacentrum/setup_env.sh` builds a module-based venv instead, and is
only relevant for plain SSH sessions on a frontend, not the container workflow.

**Configs** (`configs/`): `data.yaml` (dataset root path), `clicks.yaml` (click
simulation parameters), `train.yaml` (seed, device, model width, image size,
overfit settings, checkpoint dir/frequency).

`train.yaml` is the canonical setting; `scripts/train.py` accepts `--device`,
`--subset-size`, `--epochs` and `--lr` overrides for one-off experiments, plus
`--resume <checkpoint>`. `device: auto` picks cuda > mps > cpu; naming a device
explicitly makes it fail loudly if unavailable, which is what a cluster job
wants.

**Checkpoints:** `outputs/checkpoints/best.pt` (highest training IoU) and
`latest.pt` (periodic, by `checkpoint.save_every`). Prefer `best.pt`.

**Data export:** `scripts/export_ade20k.py` pulls from the HF mirror and writes
the original toolkit folder layout, so `src/data/ade20k.py` needs no changes.
Requires `datasets` and `pyarrow`, which the NGC container does not include.
