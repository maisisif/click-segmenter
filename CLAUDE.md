# CLAUDE.md — project context for continuing this work

Read this file plus PROGRESS.md at the start of every session. This file holds
the stable facts: what the project is, what has been built and measured, how
the infrastructure works, and what remains. PROGRESS.md tracks the live state.
Last full update: 2026-08-16, while the fifth training run (pretrained
encoder) was in progress.

## People

Mais (github `maisisif`, MetaCentrum user `mais999`) builds the project;
Kassem Anis Bouali reviews results and sets direction over Discord. The
project target, in his words: a "fully working software that serves some
purpose, that is documented" — and he considers ~0.5 IoU not good. His
technical requests from the most recent review (2026-08-15/16): website with
pages (home, how to navigate) and deploy instructions a random user can
follow; inputs at 512x384; a network deeper than 3 layers; a simple readable
implementation (data loading, training, visualisation without the tricks);
multi-head attention layers were suggested earlier.

## The project

Interactive object segmentation: user loads an image, clicks an object, the
system returns a mask for that instance. Positive clicks select, negative
clicks exclude. Model is trained by ourselves — the original rule was no
pretrained SAM or off-the-shelf interactive segmenter as the core.
**Note:** the fifth run uses an ImageNet-pretrained ResNet-34 *encoder* inside
our own UNet. Whether that is inside or outside the "train it ourselves" rule
has NOT been explicitly confirmed with Kassem; the question was drafted but
Mais chose to run the experiment first. Raise it with him.

Original milestone plan: M1 data loading, M2 click simulation, M3 overfit
sanity check, M4 full training, M5 evaluation (IoU/Dice/NoC + SAM baseline),
M6 interactive UI, M7 polish/README. M1-M4 done, M6 mostly done, M5 barely
started (IoU only), M7 partially (README exists; website pages and deploy
docs do not). Future extension kept in mind: scene object graphs consuming
the segmentation output (`src/inference/predictor.py` is the intended
interface for that).

## Repository state (github.com/maisisif/click-segmenter, branch main)

```
configs/
  data.yaml         dataset root (uses ~; see $HOME warning below)
  clicks.yaml       click simulation + encoding params
  train.yaml        model arch/size, training, checkpoint settings
src/
  data/ade20k.py    discover_samples, load_sample (all masks), load_instance
                    (one image+mask), load_instance_mask (one mask only)
  data/dataset.py   ClickSegmentationDataset: eager or lazy, cached instance
                    index, neighbour-negative clicks, (H,W) sizes
  data/clicks.py    click simulation from GT masks (interior positives,
                    boundary-band negatives)
  data/encoding.py  clicks -> 2 extra channels; disk (default) or distance
  data/splits.py    70/20/10 split BY IMAGE, deterministic, order-independent
  model/unet.py     from-scratch UNet, configurable depth; legacy checkpoint
                    key migration (migrate_legacy_state_dict)
  model/resnet_unet.py  ResNet-34 ImageNet-pretrained encoder + UNet decoder;
                    click channels zero-initialised; ImageNet normalisation
                    inside the model
  model/build.py    build_model(config), detect_arch(state_dict) — the app and
                    notebook load any era of checkpoint via this
  training/         losses (BCE+Dice), metrics (IoU), checkpoints (atomic
                    save, full-state resume), device selection
  inference/predictor.py  ClickPredictor: full-res image + clicks in original
                    coordinates -> mask at original resolution
scripts/
  app.py            Gradio app (single page: click type, threshold, status)
  train.py          M3 overfit sanity check (CLI overrides for diagnostics)
  train_full.py     real training: splits, per-epoch val, best-checkpoint by
                    val IoU, ReduceLROnPlateau, early stopping (patience 30),
                    --auto-resume across walltime kills, history.json every
                    epoch, test evaluated once at the end
  train_simple.py   the whole pipeline in one readable file (Kassem's
                    "simple implementation" ask) — for understanding, not runs
  export_ade20k.py  HF mirror parquet -> on-disk toolkit layout
  metacentrum/      PBS jobs: train.pbs, export_data.pbs, app.pbs (CPU),
                    hello_gpu.pbs, setup_env.sh (legacy, container made it moot)
notebooks/results.ipynb  curves, baseline comparison, qualitative examples
results/          archived history.json + best.pt per finished run
requirements.txt  runtime deps; requirements-data.txt adds datasets/pyarrow
README.md         quick start, usage, results table (numbers predate run 5)
```

## Data

- Official ADE20K registration was broken; dataset built from the Hugging
  Face mirror `1aurent/ADE20K` (full 2021 release, parquet). COCO and PSG
  were considered and ruled out by Kassem.
- Exported to MetaCentrum at
  `/storage/brno2/home/mais999/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE`:
  **12,003 images** (10,000 train split + 2,000 val split + 3 original
  samples), ~340k instance masks. Verified: mask PNGs use exactly
  {0, 128, 255} (0 background, 128 occluded, 255 visible); `objects[i].id`
  aligns 1:1 with the instances sequence; some `parts` fields are null
  (export handles this; one malformed row must not abort an export — it
  skips and logs).
- Training uses the visible mask (`arr == 255`). Splitting is always by
  image, never by instance (instances from one image in different splits
  would leak memorised scenes into validation).
- `training.max_images: 3000` subsamples deterministically; the 12k are on
  disk. Instance indexes are cached in `outputs/instance_index_*.json`,
  keyed by image list + size; they rebuild automatically when inputs change.

## Experimental record (all numbers are held-out test IoU unless stated)

| # | Setup | Test IoU |
|---|-------|----------|
| 1 | from scratch, 128px sq, depth 3, base 32, 3k images | 0.4990 |
| 2 | same but 10k images | 0.5035 |
| 3 | from scratch, 384x512, depth 4, neighbour negatives, 3k | 0.5125 |
| 4 | (best val comparison for run 3) | val 0.5157 |
| 5 | ResNet-34 pretrained encoder, 384x512, 3k — IN PROGRESS | best val 0.5699 @ epoch 37 (test pending) |

Established by direct experiment (safe to rely on):

- **lr 0.003 was unstable** (deterministic collapse ~epoch 587 in the overfit
  test); 0.001 stable. Overfit test passes: 4 instances memorised to IoU
  1.0000 (requires base_channels >= 32; 16 saturates at ~0.91).
- **Data volume was not the from-scratch bottleneck**: 3.3x images gained
  +0.0045. (Caveat: measured on the from-scratch model only. With the
  pretrained encoder and its larger train/val gap, retesting with
  `--max-images 0` is a reasonable, not-yet-run experiment.)
- **More epochs do not help** past the plateau; validation flattens ~epoch
  30-37 in every run while train IoU keeps climbing (overfitting gap).
- **Resolution + depth + neighbour negatives combined**: +0.009 (run 3 vs 2;
  three changes at once, individually unattributed).
- **Pretrained encoder**: the largest single improvement so far (+~0.05 val
  over run 3's best; final test number pending run completion).
- Trivial baselines on test (128px era): random 0.041, all-foreground 0.048,
  disk-at-click 0.116. IoU is not accuracy; random is near 0, not 0.5.
- Single-click IoU is the only thing ever measured. Multi-click behaviour
  and NoC have never been evaluated (the app supports multiple clicks; the
  evaluation does not exist yet).

From the literature scan (RITM, SimpleClick, FocalClick, Xu et al. 2016 —
read 2026-08-16, summaries in PROGRESS.md):

- Disk click encoding beats distance maps (RITM ablation) — our default was
  already correct; the implemented `encoding: distance` option is
  literature-deprecated and untested.
- All 0.8+ methods use pretrained backbones; SimpleClick credits pretraining
  as the main factor.
- FocalClick shows target-centered *crops* matter more than raw resolution.
- RITM-style previous-mask input + iterative click training is what makes
  additional clicks converge. Neither is implemented here.
- These methods train on COCO+LVIS (~1.5M instances); ADE20K is smaller and
  stuff-heavy — part of any gap to their numbers is the dataset.

## MetaCentrum operations (hard-won; do not relearn these)

- Access: OnDemand (ondemand.metacentrum.cz) → Clusters → Skirit/Perian
  Shell Access for a frontend terminal (has qsub, internet). Jupyter
  sessions run inside an NGC container that has torch+CUDA but NO qsub.
  SSH by password from a terminal has failed repeatedly; use OnDemand.
- Everything long-running is a **batch job** (`qsub scripts/metacentrum/*.pbs`).
  Interactive jobs die when the browser tab drops. PBS writes its .o file
  only at job end; our jobs tee live logs to `outputs/train.log` / `app.log`.
- **$HOME is not stable across nodes** (multiple storage homes; a job can
  land with $HOME on praha5-elixir or praha2-natur while the data is on
  brno2). Three jobs failed this way. Batch scripts hardcode
  `/storage/brno2/home/$USER` and pass `--data-root` explicitly; Singularity
  refuses `--env HOME=...`.
- Container: `/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF`
  (torch 2.7 + CUDA + torchvision + numpy/scipy/PIL/yaml/matplotlib; no
  gradio, no datasets/pyarrow). Needs GPUs with compute capability >= 7.5:
  on older cards (konos 1080Ti) `torch.cuda.is_available()` is True but the
  first kernel launch fails. Jobs request `gpu_cap=compute_75`; train.pbs
  additionally `gpu_mem=30gb` (384x512 depth-4/ResNet batch 32 exceeds 16GB
  cards).
- Some compute nodes have **no internet** (pip and wget fail). ImageNet
  weights are cached once from a frontend to
  `/storage/brno2/home/mais999/.cache/torch/hub/checkpoints/resnet34-b627a593.pth`;
  train.pbs sets TORCH_HOME there and verifies. The Gradio share app runs as
  a CPU-only batch job (app.pbs) but still needs a node with internet for
  pip+tunnel; if it fails, run the app locally instead.
- Walltime 12h; `train_full.py --auto-resume` continues a killed run from
  `outputs/checkpoints/latest.pt` (atomic saves; optimizer, scheduler, best,
  history all restored). Resubmitting the same train.pbs is the whole
  procedure. Before a NEW experiment: archive
  `outputs/history.json` + `outputs/checkpoints/best.pt` into `results/<name>/`,
  then `rm -rf outputs/checkpoints outputs/history.json` so auto-resume
  cannot continue the wrong model.
- Git hygiene on the cluster: running the notebook or sed-editing configs
  dirties the checkout and blocks `git pull` (has bitten twice:
  `git checkout -- <file>` then pull). The cluster clone is read-only in
  spirit: all commits happen on the laptop and flow through GitHub.
- The epoch-time bottleneck is the dataloader (shared-storage PNG decode),
  not the GPU. Fixed by `load_instance` (decode 1 mask not ~20; 10x),
  16 workers, persistent_workers, pin_memory, prefetch. Current run:
  ~230s/epoch at 384x512 batch 32 on an A40.

## Local (laptop) setup

macOS, venv at `~/projects/click-segmenter/.venv` with requirements.txt
installed. The app runs locally on CPU:
`python scripts/app.py --checkpoint ~/Downloads/best.pt --device cpu`
(checkpoint downloaded via OnDemand Files from
`.../click-segmenter/outputs/checkpoints/`). `--share` gives a public link.
The predictor auto-detects checkpoint architecture, so old and new
checkpoints both load.

## Current run (5) — check on return

ResNet34-UNet, 3k images, lr 3e-4 with plateau decay, batch 32. At the time
of writing: epoch 64, best val 0.5699 at epoch 37, validation drifting
below best while train reaches 0.77 — early stopping (patience 30) should
end it around epoch 67, then test evaluation runs automatically and
`outputs/history.json` gets the final summary. On completion: archive to
`results/run-resnet34/`, update README results table and the notebook
(note: the notebook's model-loading cell was patched for arch detection via
`build.py`, but has only ever been executed against the 128px-era
checkpoint; re-run it fully on the cluster against the new checkpoint).

## Outstanding work, in Kassem's priority order as understood

1. **Website pages + deploy docs** (his oldest unmet ask): Home / Segment /
   Help tabs in the Gradio app; deployment instructions a stranger can
   follow (local install is proven; Hugging Face Spaces was identified as a
   candidate for a permanent hosted URL but nothing is built).
2. **Model quality**: he considers ~0.5 not good. Multi-click evaluation
   (IoU at k clicks, NoC@85/90 — the field's standard metrics, M5 in the
   original plan) has been proposed to Mais and accepted in principle but
   NOT built. Literature-supported next levers if quality work continues:
   target-centered crops, previous-mask + iterative training. Attention at
   the bottleneck was Kassem's suggestion and remains unimplemented.
3. **The 12k-image rerun** with the pretrained encoder (single flag) —
   worth one shot given the current train/val gap.
4. **Mais's understanding**: he must be able to explain the pipeline
   (5-channel input, click simulation, BCE+Dice, image-level splits, the
   overfit test, what each run proved) without assistance. train_simple.py
   exists for exactly this walkthrough.

## Verification habits this project relies on

Run the overfit test after any pipeline/model change
(`python scripts/train.py --device cpu --subset-size 4` — must reach IoU
~1.0). Compile-check and logic-test before pushing; the cluster round-trip
is expensive. Every claim in reports traces to a measured number; failed
hypotheses are recorded in PROGRESS.md rather than deleted.
