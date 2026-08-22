# Deploying and running the app

Two audiences, two halves. **Running it locally** needs nothing but Python and a
checkpoint. **Publishing it** needs a free Hugging Face account. Neither needs a
GPU.

Every command below is run from the root of the project checkout.

---

## 1. Run it on your own machine

### Requirements

- Python 3.10 or newer (`python3 --version` to check)
- About 2 GB of disk for the dependencies
- No GPU. It runs on CPU; a click takes a few seconds.

### Install

```bash
git clone <repository-url> click-segmenter
cd click-segmenter
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a machine with no NVIDIA GPU, install the CPU build of PyTorch instead — it
is a tenth of the size:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### Get the trained weights

The weights are not in this repository — they are hosted separately so that
cloning the code stays fast.

```bash
python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('maisisif/click-segmenter', 'click-segmenter.pt'))"
```

That prints the path it downloaded to. Or download the file by hand from the
model page and note where you put it.

### Start it

```bash
python scripts/app.py --checkpoint /path/to/click-segmenter.pt --device cpu
```

Open <http://localhost:7860>. Add `--share` for a temporary public link that
works from another machine.

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run from the project root, not from inside `scripts/`. |
| `ModuleNotFoundError: No module named 'gradio'` | The virtualenv is not active. Re-run the `source .venv/bin/activate` line. |
| `Address already in use` | Something else holds port 7860. Add `--port 7861`. |
| `Error(s) in loading state_dict` | The checkpoint does not match the code. Re-export it with `scripts/export_model.py`, or pull the latest code. |
| Clicking does nothing | Click *inside* the image area, not the upload border. The first click after startup is slowest. |

---

## 2. Publish it to Hugging Face Spaces

The hosted demo is two repositories, not one:

- a **Space** holding the interface — a few hundred KB of Python, redeployed
  whenever the UI changes;
- a **model repo** holding the weights — around 90 MB, replaced only when a
  better checkpoint finishes training.

They are kept apart on purpose. With the checkpoint inside the Space, every
one-line edit to the interface would re-push a 90 MB file through Git LFS, and
swapping in a new model would mean redeploying the app.

### One-time setup

1. Create a free account at <https://huggingface.co>.
2. Create an access token with **write** permission at
   <https://huggingface.co/settings/tokens>.
3. Log in locally:

   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

### Export the checkpoint

A training checkpoint carries optimizer state that inference never uses, and
lacks the settings needed to run it without the config files. This produces a
deployment checkpoint that is about three times smaller and self-describing:

```bash
python scripts/export_model.py \
  --checkpoint results/run-multimask/best.pt \
  --output outputs/export/click-segmenter.pt
```

It prints both file sizes and refuses to write a checkpoint it cannot load back.

It also prints the input resolution it embedded and where that came from. Read
that line. Checkpoints from before this was recorded in training take their
resolution from `configs/train.yaml`, which describes what the repo trains
*today* — and the network is fully convolutional, so a checkpoint trained at
128x128 exported as 384x512 loads, predicts, and is quietly worse with nothing
to indicate why. If the export warns, pass the size the run actually used:

```bash
python scripts/export_model.py --checkpoint results/run-3000-images/best.pt \
  --image-size 128 128 --output outputs/export/click-segmenter.pt
```

Checkpoints written from now on record their own settings and need no override.

### Upload the weights

```bash
python scripts/deploy_space.py \
  --model YOUR_NAME/click-segmenter \
  --checkpoint outputs/export/click-segmenter.pt
```

### Deploy the interface

```bash
python scripts/deploy_space.py --space YOUR_NAME/click-segmenter
```

The script prints exactly what it will upload and asks before doing it; add
`--dry-run` to see the file list without uploading, or `--yes` to skip the
prompt. Then, if your model repo is not the default `maisisif/click-segmenter`,
open the Space's **Settings → Variables and secrets** and add:

| Variable | Value |
|---|---|
| `MODEL_REPO` | `YOUR_NAME/click-segmenter` |
| `MODEL_FILE` | `click-segmenter.pt` (only if you renamed it) |
| `MODEL_TOKEN` | a **read** token — only if the model repo is private |

The Space builds itself after the upload. Watch the **Logs** tab; the first
build takes a few minutes because it installs PyTorch.

### Updating later

- Changed the interface: re-run the `--space` command.
- Trained a better model: re-run `export_model.py`, then the `--model` command.
  The Space picks it up on its next restart — use **Settings → Factory reboot**
  to clear the cached copy immediately.

### What to expect from the free tier

- **It sleeps.** After about 48 hours without a visitor the Space suspends, and
  the next visit takes a minute or so to wake it. This is normal and is worth
  warning anyone you send the link to.
- **CPU only**, so a click takes a few seconds rather than being instant.
- **Image size matters.** `deploy/huggingface/requirements.txt` pins the `+cpu`
  PyTorch wheels; without that pin pip installs the CUDA build, which is roughly
  2.5 GB of GPU runtime the Space can never use and can fail the build outright.

### If the Space fails to build

Read the **Logs** tab first — the error is almost always in the pip install.

| Log says | Fix |
|---|---|
| `No matching distribution found for torch==...+cpu` | That version left the CPU index. Pick a version listed at <https://download.pytorch.org/whl/cpu/torch/> and update `deploy/huggingface/requirements.txt`. |
| `Repository Not Found` / `401` at startup | `MODEL_REPO` is wrong, or the repo is private and `MODEL_TOKEN` is missing. |
| Build exceeds the size limit | The CUDA torch wheel was installed. Check the `--extra-index-url` line survived your edit to `requirements.txt`. |
| Gradio raises on a component argument | `sdk_version` in `deploy/huggingface/README.md` and the `gradio==` pin in `requirements.txt` disagree. Make them match. |

---

## 3. Rehearse the deploy without publishing

Worth doing before the first real push. This builds exactly what would be
uploaded and runs it as the Space would — from its own directory, with the
project not importable:

```bash
python scripts/deploy_space.py --space you/click-segmenter --dry-run
python tests/test_app_wiring.py
```

The first prints the file list; the second checks the interface's event wiring
against a throwaway model, so it needs no checkpoint and no network.
