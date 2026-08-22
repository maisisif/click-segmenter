# click-segmenter

Interactive object segmentation. Load an image, click an object, get a mask for
that object.

The model is a UNet-style encoder-decoder with an ImageNet-pretrained ResNet-34
encoder. It takes five input channels: the RGB image plus two channels encoding
where the user clicked (one for "this is the object", one for "this is not"),
and returns three candidate masks with a predicted quality score for each, so
that an ambiguous click (a person's shirt could mean shirt, torso, or whole
person) does not have to be averaged into one blurry answer.

Trained from scratch on ADE20K. No pretrained SAM or other off-the-shelf
interactive segmenter is used, so the architecture stays open to modification.

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

The app auto-detects which architecture a checkpoint was trained with, so any
checkpoint from this repo's history loads without configuration changes.

### Using it

- Click an object. A mask appears over it.
- Wrong region included? Switch **Click type** to **Exclude** and click there.
  The mask is recomputed from all clicks together.
- **Mask threshold** trades coverage against precision.
- **Clear clicks** starts over on the same image.

It expects scene photography — streets, rooms, landscapes — which is what
ADE20K contains. Product shots on white backgrounds and close-up portraits are
outside the training distribution and give poor results.

## Results

Trained on 12,003 ADE20K images split 70/20/10 by image (167k / 49k / 24k
clickable object instances).

| Change | Test IoU |
| --- | --- |
| baseline: from scratch, 128x128, depth 3 | 0.4990 |
| + more data (3k to 10k images) | 0.5035 |
| + 384x512 input, depth 4, neighbour negatives | 0.5125 |
| + **ImageNet-pretrained encoder** | **0.5710** |
| + more data again (3k to 12k) | val 0.6175 |

Two findings worth noting. Pretraining was the single largest improvement,
consistent with the interactive-segmentation literature, where every method
reaching high scores starts from a pretrained backbone. And more data helped
ten times as much *after* pretraining than before it — the from-scratch model
could not fit the data well enough for extra examples to matter.

Measured against strategies that learn nothing, on the same test set:

| Strategy | IoU |
| --- | --- |
| random prediction | 0.041 |
| predict everything as foreground | 0.048 |
| fixed disk drawn at the click | 0.116 |
| **this model** | **0.4990** |

That comparison matters because IoU is not accuracy: a random prediction scores
near zero on it, not 0.5. The model is roughly four times better than the
strongest strategy that knows where you clicked but has learned nothing else.

It is still well short of a polished segmenter. Masks are soft at boundaries
and can bleed into a neighbouring object.

## Repository layout

```
configs/          YAML configuration, no hardcoded hyperparameters
  data.yaml         where the dataset lives
  clicks.yaml       click simulation and encoding
  train.yaml        model, training, checkpointing
src/
  data/             loading, click simulation, encoding, splits
  model/            UNet, ResNet-UNet, construction from config
  training/         losses, metrics, checkpoints, device selection
  inference/        running a checkpoint on a real click
scripts/
  app.py            the interactive interface
  train_simple.py   the whole pipeline in one readable file
  train.py          overfit sanity check
  train_full.py     real training with validation and test evaluation
  export_ade20k.py  build the dataset from the Hugging Face mirror
  analyze_dataset.py  measure object and class statistics
  metacentrum/      PBS job scripts for the cluster
notebooks/
  results.ipynb     training curves, baselines, example predictions
```

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
