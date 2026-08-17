# click-segmenter

Interactive object segmentation. Load an image, click an object, get a mask for
that specific object.

The model is a UNet-style encoder-decoder trained from scratch on ADE20K. It
takes five input channels: the RGB image plus two channels encoding where the
user clicked (one for "this is the object", one for "this is not"). No
pretrained backbone and no off-the-shelf segmenter such as SAM is used, so the
architecture stays open to modification.

## Quick start

Requires Python 3.10 or newer. Runs on CPU; a GPU is optional and only matters
for training.

```bash
git clone https://github.com/maisisif/click-segmenter.git
cd click-segmenter

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download a trained checkpoint (`best.pt`) and point the app at it:

```bash
python scripts/app.py --checkpoint path/to/best.pt --device cpu
```

Open <http://127.0.0.1:7860>. Add `--share` to get a temporary public link you
can send to someone else.

### Using it

- Click an object. A mask appears over it.
- Wrong region included? Switch **Click type** to **Exclude** and click there.
  The mask is recomputed from all clicks together.
- **Mask threshold** trades coverage against precision. Lower includes more
  pixels.
- **Clear clicks** starts over on the same image.

It expects scene photography (streets, rooms, landscapes), which is what
ADE20K contains. Product shots on white backgrounds and close-up portraits are
outside what it was trained on and give poor results.

## Results

Trained on 10,000 ADE20K images split 70/20/10 by image, giving 168k / 48k /
24k clickable object instances.

| Split | IoU |
| --- | --- |
| validation (best epoch) | 0.5072 |
| **test (held out)** | **0.5035** |

Measured against strategies that learn nothing, on the same test set:

| Strategy | IoU |
| --- | --- |
| random prediction | 0.041 |
| predict everything as foreground | 0.048 |
| fixed disk drawn at the click | 0.116 |
| **this model** | **0.4990** |

The last row is the point of comparison that matters. IoU is not accuracy: a
random prediction scores near zero on it, not 0.5. The model is roughly four
times better than the strongest strategy that knows where you clicked but has
learned nothing else.

It is still a long way from a polished segmenter. Masks are soft at boundaries
and sometimes bleed into a neighbouring object. See
[PROGRESS.md](PROGRESS.md) for what was measured about why.

## Repository layout

```
configs/          YAML configuration, no hardcoded hyperparameters
  data.yaml         where the dataset lives
  clicks.yaml       click simulation and encoding
  train.yaml        model, training, checkpointing
src/
  data/             dataset loading, click simulation, encoding, splits
  model/            the UNet
  training/         losses, metrics, checkpoints, device selection
  inference/        running a checkpoint on a real click
scripts/
  app.py            the interactive interface
  train.py          M3 overfit sanity check
  train_full.py     real training with validation and test evaluation
  export_ade20k.py  build the dataset from the Hugging Face mirror
  metacentrum/      PBS job scripts for the cluster
notebooks/
  results.ipynb     training curves, baselines, example predictions
```

## Training your own

The dataset is not included; it is roughly 4 GB as individual files. ADE20K's
official registration has been unavailable, so this builds it from the
[Hugging Face mirror](https://huggingface.co/datasets/1aurent/ADE20K). This
step needs two extra packages:

```bash
pip install -r requirements-data.txt
```

```bash
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

`train_full.py` splits by **image**, never by instance: one image contains many
objects, and putting some of an image's objects in training and others in
validation would mean validating on scenes the model had already memorised.

It keeps the checkpoint from the best validation epoch, stops early when
validation stops improving, and evaluates that checkpoint on the test split
exactly once at the end.

### On a cluster

`scripts/metacentrum/` holds PBS scripts for MetaCentrum: dataset export,
training, and the app. Training supports `--auto-resume`, so a run longer than
one job's walltime continues where it left off when resubmitted.

Two things that cost time and are worth knowing on that system: `$HOME` differs
between nodes, so batch jobs must use absolute `/storage/...` paths; and the
PyTorch container ships CUDA kernels for compute capability 7.5 and newer, so
jobs need `gpu_cap=compute_75` or they fail at the first kernel launch on older
cards.

## Useful options

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
| `--base-channels` | model width |
| `--auto-resume` | continue from the last checkpoint |
| `--skip-test` | leave the test split untouched |

## Acknowledgements

ADE20K: Zhou et al., *Scene Parsing through ADE20K Dataset*, CVPR 2017.
Computed on [MetaCentrum](https://www.metacentrum.cz/), supported by CESNET
and CERIT-SC.
