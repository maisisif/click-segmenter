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
- **M4 — Full training on MetaCentrum** ✅ First real training run complete.
  40 epochs on 3,000 exported images, best validation IoU **0.4928** at epoch
  32, held-out test IoU **0.4990**.
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

## M4 results (2026-08-08, job 22750050)

Data: 3,000 train + 600 validation images exported from the HF mirror, split
70/20/10 by image into 2,522 / 721 / 360 images = 50,451 / 14,599 / 7,163
instances. Model: depth-3 UNet, `base_channels: 32`, 128px, batch 16, Adam at
lr 0.001, 40 epochs, ~390s/epoch on an A40 (~4.4 hours total).

| Split | Loss | IoU |
| --- | --- | --- |
| train (epoch 40) | 0.4584 | 0.5530 |
| validation (best, epoch 32) | 0.5345 | 0.4928 |
| **test (best checkpoint)** | 0.5257 | **0.4990** |

Test tracking validation this closely is the key sanity check: the splits are
sound and epoch selection did not overfit to validation.

Validation plateaued around epoch 29 (0.483-0.493) while training IoU kept
rising to 0.553, so the model had begun to overfit and more epochs would not
help. The largest available lever is more data: this used 3,000 of the 25,574
images in the mirror, roughly 12%. Other levers, roughly in order of expected
value: pretrained encoder, higher input resolution, stronger augmentation,
iterative click refinement at training time.

## Blocked on

Nothing. Both earlier blockers are resolved: GPU jobs work via
`gpu_cap=compute_75` in a batch job, and 3,600 images are exported to brno2.

## Lessons that cost time (worth not repeating)

- **`$HOME` is not stable across MetaCentrum nodes.** You have a home on every
  storage, and `$HOME` points at whichever belongs to the node you landed on.
  Three jobs failed this way. Batch jobs must use absolute `/storage/brno2/...`
  paths. Overriding `HOME` for a Singularity container does not work either --
  it explicitly refuses.
- **`qsub` only exists on frontends**, not inside the OnDemand Jupyter
  container. Use Clusters -> Shell Access.
- **Interactive jobs die when the browser tab reloads.** Anything longer than a
  few minutes belongs in a batch job.
- **`load_sample` decodes every mask for an image** (~20, sometimes 70). Using
  it in the training loop made the dataloader ~10x slower than necessary;
  `load_instance` decodes only what is needed.
- **Old GPUs pass `torch.cuda.is_available()` and then fail at kernel launch.**
  The NGC container needs sm_75+; always request `gpu_cap=compute_75`.

## Next steps

1. **Build the results notebook** Kassem asked for: training curves from
   `outputs/history.json`, best epoch marked, the test number, and example
   predictions with the click overlaid. Keep logic in `src/`; the notebook
   imports and plots. This is the deliverable.
2. **M5 metrics**: add Dice, and NoC (number of clicks to reach a target IoU),
   the standard interactive-segmentation metric. Optionally compare against a
   pretrained SAM baseline.
3. **Scale up the data.** Rerun `export_data.pbs` with `LIMIT_TRAIN=` empty for
   all 25,574 images (~4GB, ~630k files -- quota is fine, brno2 has no file
   count limit). Retrain. This is the highest-value single change.
4. **M6**: inference GUI that loads `best.pt` and segments from a real click.

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
