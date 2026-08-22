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
- **M4 — Full training on MetaCentrum** ✅ Two runs complete. Best result:
  test IoU **0.5035** on 10,000 images. Tripling the data from the first run
  moved test IoU by only +0.0045, which rules out data volume as the limit.
- **M5 — Evaluation (IoU/Dice/NoC) + SAM baseline comparison** ⬜ Basic IoU is
  in place; Dice and NoC are not.
- **M6 — Interactive UI** 🔄 `scripts/app.py` (Gradio) built: upload an image,
  click to segment, add include/exclude clicks to refine, adjustable threshold.
  Inference logic lives in `src/inference/predictor.py` so the interface is
  swappable. Not yet run against a real checkpoint.
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

## M4 second run (2026-08-11, 10,000 images)

Same architecture, 3.3x the data, plus ReduceLROnPlateau, early stopping and
batch 128. Stopped early at epoch 65 (best was epoch 35).

| Run | Train instances | Best val IoU | Test IoU |
| --- | --- | --- | --- |
| 3,000 images | 50,451 | 0.4928 | 0.4990 |
| 10,000 images | 168,000 | 0.5072 | **0.5035** |

**Tripling the data gained +0.0045 test IoU.** The "model is data-starved"
hypothesis is falsified. Training IoU rose further than before (0.614 vs 0.553)
while validation stayed at ~0.50, so the extra data was absorbed into fitting
the training set, not into generalisation.

A ceiling that does not move when the dataset triples is not a data ceiling.
Remaining suspects, in order:

1. **Input resolution.** At 128px a two-pixel boundary error is expensive in
   IoU terms, and that cost is independent of dataset size. Leading suspect.
2. **Architecture.** Depth-3 UNet, 16x16 bottleneck, limited global context.
   Attention at the bottleneck is cheap (256 tokens) and is what the supervisor
   suggested.
3. **Click encoding.** `clicks.encoding: distance` is implemented but has never
   been run. Still on `disk`.
4. **Single-click ambiguity.** One click on a building is genuinely ambiguous
   between wall, facade and whole structure. Some of the gap may be an
   irreducible ceiling given the task definition.

Diagnostic: the notebook's example predictions distinguish these. Blurry but
roughly correct masks point at resolution; wrong object entirely points at the
click encoding or ambiguity.

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

## Literature scan (2026-08-16: RITM, SimpleClick, FocalClick, Xu 2016)

What published interactive segmenters that reach 0.8+ actually do, and what it
means for us:

- **Disk click encoding is correct.** RITM's ablation found disks (r=2-5) beat
  Euclidean distance maps. Our `clicks.encoding: distance` option is therefore
  ruled out by published evidence and stays non-default.
- **Every method uses an ImageNet-pretrained encoder** and SimpleClick credits
  pretraining as the main factor. Asked supervisor whether an ImageNet
  encoder inside our own UNet is within project rules.
- **Target-centered crops beat raw resolution** (FocalClick runs at 128-256px
  competitively). "What's in the pixels" matters more than "how many".
- **Xu-style neighbour negatives** train the model that adjacent objects are
  not the target — directly aimed at our mask-bleeding failure. Implemented
  (`clicks.neighbor_negative_prob`).
- Previous-mask-as-input + iterative training is what makes clicks 2+
  converge (RITM). Candidate for later, compounds with crops.
- Caveat: these train on COCO+LVIS (~1.5M instances); ADE20K is smaller and
  stuff-heavy. Part of any gap to 0.9 is the dataset, not the code.

## Current run config (third training run, per supervisor feedback)

384x512 aspect-preserving input (was 128 square), depth-4 UNet (was 3),
neighbour negatives at p=0.3, batch 32, lr 0.001, 3000 images (justified by
the data ablation). Old checkpoints still load via a legacy key remap in
`src/model/unet.py`; the predictor auto-detects checkpoint depth.

## Deployment work (2026-08-22)

Planning settled that the grade is on the system, not on IoU ("a fully working
software that serves some purpose", Kassem, 2026-08-12), so the shippable
product comes before further model work. Real deadline 2026-09-10; 2026-09-15 is
the Computer Vision exam, not a defense.

Built this session:

- **`scripts/export_model.py`** turns a training checkpoint into a deployment
  one. Measured on a real ResNet-34 checkpoint: 294.4 MB to 98.2 MB, exactly 3x,
  which is the optimizer state. It also embeds what `detect_arch()` cannot read
  off the weight shapes (image size, the clicks block, normalization), so the
  serving side needs no `configs/` at all.
- **`detect_arch()` now returns `base_channels`** too, read from
  `stem.block.0.weight`. It was the last piece of architecture still coming from
  the config, and without it a from-scratch checkpoint could not be rebuilt
  without the repo.
- **The interface has three pages** (Home / Segment / Help) and moved to
  `src/app/`, so `scripts/app.py` and the Space entry point are both thin
  shells over the same UI. Added undo and a mask PNG download.
- **`deploy/huggingface/` + `scripts/deploy_space.py`**, splitting the app (69 KB,
  redeployed freely) from the weights (~98 MB, in a separate model repo). The
  requirements pin `+cpu` wheels; the default PyPI Linux torch is the ~2.5 GB
  CUDA build, which a free CPU Space cannot use and may not even build.
- **`docs/DEPLOY.md`**: install, run, publish, and two troubleshooting tables.
- **`tests/test_app_wiring.py`**: builds a throwaway model, so it runs anywhere
  in seconds with no checkpoint and no network.

**Bug found and fixed: refinement clicks were being thrown away.** The old
`scripts/app.py` wrote the annotated view (mask tint plus click rings) back into
the same `gr.Image` that was the model's image input. So from the second click
onward the model was segmenting a green-tinted photo with rings painted on it,
not the photograph. No error, no warning -- just quietly worse results the more
a user tried to correct the mask. The pristine upload now lives in a `gr.State`,
and `tests/test_app_wiring.py` asserts it survives an annotated view being
passed back in. This is separate from the *design* gap that click two is a fresh
prediction rather than a correction; that one needs previous-mask input and
iterative training, and is Week 2 work.

Verified end to end: a real ResNet-34 export predicts a 1536x2048 photo through
the Space entry point, from the staged Space layout, with the project not
importable and no configs present.

## Next steps

1. **Deploy.** Export the run-6 checkpoint, push weights and Space, send Kassem
   the URL (target Fri 2026-08-28). Warn him about the free-tier cold start.
2. **GitLab migration**, excluding `.claude/`, `CLAUDE.md` and
   `segmentation-project-prompt.md`.
3. **Previous-mask input + iterative click training.** This is the product
   defect, not just a metric: extra clicks barely help because the model never
   sees what it just predicted.
4. **M5 metrics**: NoC@85 and NoC@90, the field's standard measure and the
   honest way to describe an interactive tool. Frame results to Kassem as NoC,
   not as single-click IoU.
5. One multi-head attention run, if time allows. Model freeze Sat 2026-09-05.

Cut deliberately: target-centered crops, a bigger backbone, augmentation.

Do not spend more effort on dataset size. That was tested and rejected.

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
