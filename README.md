# click-segmenter

Interactive object segmentation. Load an image, click an object, get a mask for
that object.

The model is a UNet-style encoder-decoder with an ImageNet-pretrained ResNet-34
encoder, reaching **0.6194 IoU from a single click** on held-out ADE20K. It
takes five input channels: the RGB image plus two channels encoding where the
user clicked (one for "this is the object", one for "this is not"), and returns
three candidate masks with a predicted quality score for each, so that an
ambiguous click (a person's shirt could mean shirt, torso, or whole person)
does not have to be averaged into one blurry answer.

Trained by us on ADE20K. The encoder starts from ImageNet weights, but no
pretrained SAM or other off-the-shelf interactive segmenter is used at any
point, so the architecture stays open to modification.

## Quick start

Python 3.10+. Runs on CPU; a GPU only matters for training.

```bash
git clone https://github.com/maisisif/click-segmenter.git
cd click-segmenter

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a trained checkpoint (`best.pt`) and run:

```bash
python scripts/app.py --checkpoint path/to/best.pt --device cpu
```

Open <http://127.0.0.1:7860>. Add `--share` for a temporary public link.

The interface has three pages: **Home** explains what the model does, **Segment**
is the tool itself, **Help** covers how to get better results and where it
fails.

The app auto-detects which architecture a checkpoint was trained with, so any
checkpoint from this repo's history loads without configuration changes.

For step-by-step installation, hosting the app on Hugging Face Spaces, and a
troubleshooting table, see **[docs/DEPLOY.md](docs/DEPLOY.md)**.

### Using it

- Click an object. A mask appears over it.
- Wrong region included? Switch **Click type** to **Exclude** and click there.
  The mask is recomputed from all clicks together.
- **Mask threshold** trades coverage against precision, and re-renders without
  needing another click.
- **Undo last click** removes the most recent click; **Clear clicks** starts
  over on the same image.
- **Download mask (PNG)** saves the mask alone, at the original resolution.

It expects scene photography — streets, rooms, landscapes — which is what
ADE20K contains. Product shots on white backgrounds and close-up portraits are
outside the training distribution and give poor results.

## Results

Trained on 12,003 ADE20K images split 70/20/10 by image (167k / 49k / 24k
clickable object instances).

Single-click IoU, each row a controlled change from the one above it. Test IoU
is the held-out split, touched once on the epoch chosen by validation.

| # | Change | Images | Test IoU |
| --- | --- | --- | --- |
| 1 | baseline: from scratch, 128x128, depth 3 | 3k | 0.4990 |
| 2 | + more data | 10k | 0.5035 |
| 3 | + 384x512 input, depth 4, neighbour negatives | 3k | 0.5125 |
| 4 | + **ImageNet-pretrained ResNet-34 encoder** | 3k | **0.5710** |
| 5 | + more data (killed at walltime on its best epoch) | 12k | val 0.6175 |
| 6 | + **3 candidate masks with a score head** | 3k | **0.5796** |
| 7 | run 6 architecture at full data, converged | 12k | **0.6194** |
| 8 | slot architecture, 64 masks + click selection | 12k | val 0.4294 |

**Run 7 is the shipped model**: test IoU 0.6194 over 24,271 held-out
instances, validation 0.6191. It converged on its own — early stopping at
epoch 76 with the best at 56.

Four findings, each from a controlled comparison:

- **Pretraining is the largest single factor (+0.0585).** Consistent with the
  literature, where every method reaching high scores starts pretrained.
- **Data volume only pays once the model can overfit.** Tripling the data from
  scratch bought +0.0045; the same increase with a pretrained encoder bought
  +0.045, ten times as much.
- **Feeding the click *into* the network is worth about 0.19.** Run 8 removes
  it — predicting every object at once and selecting afterwards — and drops to
  0.4294. That is three times the pretraining gain, and it is the strongest
  justification for the design.
- **The score head leaves ~0.09 unclaimed.** Run 7 scores 0.6194 selected but
  **0.7068 best-of-N**: the right mask is among the three candidates and the
  selector picks a worse one. Quadrupling the data moved that gap by 0.005, so
  it is not a data problem.

Measured against strategies that learn nothing, on the same test set:

| Strategy | IoU |
| --- | --- |
| random prediction | 0.041 |
| predict everything as foreground | 0.048 |
| fixed disk drawn at the click | 0.116 |
| **this model** | **0.6194** |

That comparison matters because IoU is not accuracy: a random prediction scores
near zero on it, not 0.5. The model is roughly five times better than the
strongest strategy that knows where you clicked but has learned nothing else.

It is still short of a polished segmenter. Masks are soft at boundaries and can
bleed into a neighbouring object, and **extra clicks help less than they
should** — the model never sees its own previous mask, so click two is a fresh
prediction rather than a correction. Multi-click accuracy (NoC), the field's
standard metric, is not measured here.

## Repository layout

```
configs/          YAML configuration, no hardcoded hyperparameters
  data.yaml         where the dataset lives
  clicks.yaml       click simulation and encoding
  train.yaml        model, training, checkpointing
src/
  data/             loading, click simulation, encoding, splits
  model/            UNet, ResNet-UNet, slot UNet, construction from config
  training/         losses, metrics, checkpoints, device selection
  inference/        running a checkpoint on a real click
  app/              the web interface: layout and page text
scripts/
  app.py            launch the interface locally
  export_model.py   training checkpoint -> self-contained deployment checkpoint
  deploy_space.py   push the app to a Space and the weights to a model repo
  train_simple.py   the whole pipeline in one readable file
  train.py          overfit sanity check
  train_full.py     real training with validation and test evaluation
  train_slots.py    training for the slot architecture (whole-image examples)
  export_ade20k.py  build the dataset from the Hugging Face mirror
  analyze_dataset.py  measure object and class statistics
  metacentrum/      PBS job scripts for the cluster
deploy/huggingface/ what gets uploaded to the hosted demo
docs/DEPLOY.md      installing, running and publishing it
tests/              regression tests that need no checkpoint and no network
notebooks/
  results.ipynb     training curves, baselines, example predictions
```

Run the tests with `python tests/test_app_wiring.py` and
`python tests/test_slot_model.py`. Both build a throwaway model, so they work on
any machine in a couple of seconds with no checkpoint and no network.

## Understanding the code

Start with **`scripts/train_simple.py`**. It is the entire pipeline — load,
train, visualise — in one commented file, small enough to run on a laptop:

```bash
python scripts/train_simple.py --data-root path/to/ADE20K/images/ADE
```

`scripts/train_full.py` does the same thing plus everything needed for cluster
reality: checkpoint resume across jobs, early stopping, learning-rate
scheduling, lazy loading and an instance index cache.

## Training your own

The dataset is not included. ADE20K's official registration has been
unavailable, so this builds it from the
[Hugging Face mirror](https://huggingface.co/datasets/1aurent/ADE20K). That step
needs two extra packages:

```bash
pip install -r requirements-data.txt

python scripts/export_ade20k.py --split train --limit 10000 \
    --output-root ~/ade20k/images/ADE
python scripts/export_ade20k.py --split validation --limit 2000 \
    --output-root ~/ade20k/images/ADE
```

Point `configs/data.yaml` at that directory, then:

```bash
# sanity check first: can the model memorise a handful of examples?
python scripts/train.py --subset-size 4 --epochs 2000 --lr 0.001

# real training
python scripts/train_full.py
```

The sanity check should reach IoU 1.0000. If it does not, something is wrong
with the data or the training loop and a full run would only waste time.

Training splits by **image**, never by instance: one image contains many
objects, and putting some of an image's objects in training and others in
validation would mean validating on scenes the model had already memorised. It
keeps the checkpoint from the best validation epoch, stops early when
validation stops improving, and evaluates the test split exactly once at the
end.

Reported metrics are the IoU of the mask the system would actually return
(chosen by the score head, with no access to ground truth) and, alongside it,
the best-of-N IoU as an oracle upper bound. The gap between them measures how
much the selection step is losing.

### On a cluster

`scripts/metacentrum/` holds PBS scripts for MetaCentrum: dataset export,
training and the app. Training supports `--auto-resume`, so a run longer than
one job's walltime continues where it left off on resubmission.

Two things that cost time and are worth knowing there: `$HOME` differs between
nodes, so batch jobs must use absolute `/storage/...` paths; and the PyTorch
container ships CUDA kernels for compute capability 7.5 and newer, so jobs need
`gpu_cap=compute_75` or they fail at the first kernel launch on older cards.

## Options

`scripts/app.py`

| Flag | Purpose |
| --- | --- |
| `--checkpoint` | which trained model to load |
| `--device` | `auto`, `cuda`, `mps`, `cpu` |
| `--share` | temporary public link |
| `--port` | default 7860 |

`scripts/train_full.py`

| Flag | Purpose |
| --- | --- |
| `--data-root` | override the dataset path from the config |
| `--epochs`, `--batch-size`, `--lr` | override config values |
| `--base-channels`, `--depth` | model size (from-scratch UNet only) |
| `--max-images` | cap how many images to use (0 = all) |
| `--auto-resume` | continue from the last checkpoint |
| `--skip-test` | leave the test split untouched |

Key settings in `configs/train.yaml`: `model.arch` (`unet` or
`resnet34_unet`), `model.num_masks` (3 for candidate masks, 1 for single-mask),
`data.image_size` as `[height, width]`.

## Limitations and next steps

- Only single-click IoU is measured. The standard metric for interactive
  segmentation is NoC (clicks needed to reach a target IoU), which is not
  implemented.
- The whole image is resized to 384x512 before the model sees it. Published
  work suggests cropping around the click instead is more effective than raw
  resolution.
- Each click is treated independently; feeding the previous mask back as an
  input, with iterative training, is what makes successive corrections converge.

## Acknowledgements

ADE20K: Zhou et al., *Scene Parsing through ADE20K Dataset*, CVPR 2017.
Computed on [MetaCentrum](https://www.metacentrum.cz/), supported by CESNET and
CERIT-SC.
