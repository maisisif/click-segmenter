# CLAUDE.md — project context for continuing this work

Read this file plus PROGRESS.md at the start of every session. This file holds
the stable facts: what the project is, what has been built and measured, how to
operate the infrastructure, and what remains.

Last updated: 2026-08-22.

---

## THE COMMANDS (copy these, do not improvise)

Everything runs in one of two places. Check with `hostname`: `skirit...` is the
cluster, anything else is the laptop.

### Cluster (OnDemand -> Clusters -> Skirit Shell Access)

**Start or continue a training run.** The `&&` chain matters: a job snapshots
the code when it launches, so pulling *after* qsub silently runs stale code.
This has cost two full 12-hour runs.

```bash
cd ~/projects/click-segmenter && git pull && qsub scripts/metacentrum/train.pbs && qstat -u $USER
```

**Check on it.** `Q` queued, `R` running, absent means finished or died.

```bash
qstat -u $USER
tail -20 ~/projects/click-segmenter/outputs/train.log
```

**Full history across resumed jobs** (train.log only covers the current job):

```bash
cd ~/projects/click-segmenter && python3 -c "
import json
h = json.load(open('outputs/history.json'))
h = h['history'] if isinstance(h, dict) else h
for e in h[-15:]:
    print(f\"epoch {e['epoch']:3d} train {e['train_iou']:.4f} val {e['val_iou']:.4f} \"
          f\"train_loss {e['train_loss']:.4f} val_loss {e['val_loss']:.4f}\")
print('best val:', max(e['val_iou'] for e in h))
"
```

**Before starting a NEW experiment** (different architecture, data or size).
Skipping the rm makes --auto-resume load a checkpoint that no longer matches
the model, which either crashes or silently trains the wrong thing.

```bash
cd ~/projects/click-segmenter
mkdir -p results/run-NAME && cp outputs/checkpoints/best.pt outputs/history.json results/run-NAME/
rm -rf outputs/checkpoints outputs/history.json
# add this too if the image set or image_size changed:
rm -f outputs/instance_index_*.json
```

**After a walltime kill** (job gone, log has no "Early stopping" line): just
resubmit, `--auto-resume` continues from `latest.pt` with optimizer, scheduler,
best score and history intact.

**Other jobs.**

```bash
qsub scripts/metacentrum/export_data.pbs   # build dataset from HF mirror
qsub scripts/metacentrum/app.pbs           # Gradio demo, CPU only
grep gradio.live outputs/app.log           # read the public URL once it starts
qdel <jobid>                               # cancel
```

**Measure the dataset** (fast, JSON only):

```bash
cd ~/projects/click-segmenter && python3 scripts/analyze_dataset.py \
  --data-root /storage/brno2/home/$USER/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE
```

**Git hygiene on the cluster.** Running the notebook or editing configs dirties
the checkout and blocks `git pull`. The cluster clone is read-only in spirit;
all commits happen on the laptop.

```bash
git checkout -- configs/train.yaml notebooks/results.ipynb   # discard local edits
```

### Laptop

```bash
cd ~/projects/click-segmenter
git add -A && git commit -m "message" && git push origin main

source .venv/bin/activate
python scripts/train.py --device cpu --subset-size 4    # overfit check, must reach IoU ~1.0
python scripts/app.py --checkpoint ~/Downloads/best.pt --device cpu   # add --share for a public link
python tests/test_app_wiring.py                         # seconds, no checkpoint, no network
```

**Ship a checkpoint.** Export first (strips optimizer state, 3x smaller, and
embeds the settings so the serving side needs no `configs/`), then push the
weights and the app separately. Full procedure in `docs/DEPLOY.md`.

```bash
python scripts/export_model.py --checkpoint results/run-NAME/best.pt \
  --output outputs/export/click-segmenter.pt
python scripts/deploy_space.py --model maisisif/click-segmenter \
  --checkpoint outputs/export/click-segmenter.pt   # ~98 MB, only on a new model
python scripts/deploy_space.py --space maisisif/click-segmenter   # 69 KB, any UI change
```

`--dry-run` stages and lists without uploading. Both uploads prompt before
touching a public repo.

Checkpoints come down via OnDemand -> Files -> /storage/brno2 -> projects ->
click-segmenter -> outputs/checkpoints (or results/run-*/).

### Two failure modes that keep recurring

1. **Double submission.** After any `qsub`, run `qstat -u $USER` before
   pressing anything else. Two jobs writing the same checkpoints corrupt each
   other.
2. **Wrong terminal.** `qsub`/`qstat` exist only on the frontend, never inside
   the Jupyter container or on the laptop. `hostname` settles it.

---

## The project

Interactive object segmentation: the user loads an image, clicks an object, and
the system returns a mask for that instance. Positive clicks select, negative
clicks exclude. The model is trained by us; no pretrained SAM or off-the-shelf
interactive segmenter is used.

**Confirmed 2026-08-19:** an ImageNet-pretrained ResNet-34 *encoder* inside our
own UNet is approved by Kassem ("that works fine, actually its a good idea").

Milestones: M1 data loading, M2 click simulation, M3 overfit check, M4 full
training, M5 evaluation (Dice/NoC/SAM baseline), M6 interactive UI, M7 polish.
M1-M4 done, M6 mostly, M5 barely (IoU only), M7 partial. Future extension to
design for: scene object graphs consuming segmentation output —
`src/inference/predictor.py` is the intended interface.

## People

Mais (github `maisisif`, MetaCentrum `mais999`) builds it; Kassem Anis Bouali
reviews over Discord. Target in his words: "fully working software that serves
some purpose, that is documented"; he considers ~0.5 IoU not good. Outstanding
requests: website with pages (home, how to navigate) and deploy instructions a
random user can follow; multi-head attention layers; migrate the repo to a
GitLab he created, with no `.claude` or AI folders in it.

Asked on 2026-08-12 whether the grade is on the model or the system, he answered
the system. **So a working, documented, deployed tool outranks another point of
IoU**, and the plan is ordered that way. Report results to him as NoC@85 /
NoC@90 rather than as single-click IoU -- NoC is what an interactive tool is
actually judged on, and 0.85 single-click was never the target.

Calendar. Real deadline **2026-09-10**; 2026-09-15 is the Computer Vision exam,
not a project defense, and 09-11 to 09-15 is exam study.

- Week 1 (08-22 to 08-28): GitLab migration, export + inference wrapper, the
  three pages, deploy to Spaces. Send Kassem the URL by Fri 08-28.
- Week 2 (08-29 to 09-04): NoC evaluation, previous-mask + iterative training,
  Help page and README polish, one multi-head attention run.
- Week 3 (09-05 to 09-10): **model freeze Sat 09-05**, final checkpoint swapped
  in Sunday, deploy instructions tested on a clean machine. Done 09-10.
- Cut deliberately: target-centered crops, a bigger backbone, augmentation.

## How the model works (for explaining it)

The input is **5 channels**: RGB plus two click maps (a disk of radius 5 marks
positive clicks, another marks negatives). So an image clicked on the chair and
the same image clicked on the lamp are *different inputs* with one correct
answer each — the click is what identifies the object, which is why one image
yields ~20 training examples rather than one.

The output is **3 candidate masks** plus a predicted IoU for each. Training
back-propagates only through the best-matching candidate, so the three
specialise on different readings of an ambiguous click (shirt / torso / whole
person) instead of averaging them. At inference the score head picks one.

Loss is BCE + Dice. Splitting is by **image**, never by instance, because two
objects from one photo in different splits would leak memorised scenes into
validation.

## Repository layout

```
configs/            data.yaml (dataset root), clicks.yaml (click sim +
                    encoding), train.yaml (model, training, checkpoints)
src/data/           ade20k.py (discover_samples, load_sample = all masks,
                      load_instance = one image+mask, load_instance_mask)
                    dataset.py (eager/lazy, cached instance index, neighbour
                      negatives, (H,W) sizes)
                    clicks.py, encoding.py (disk default, distance available)
                    splits.py (70/20/10 by image, deterministic)
src/model/          unet.py (from-scratch, configurable depth, ScoreHead,
                      legacy checkpoint key migration)
                    resnet_unet.py (ImageNet ResNet-34 encoder, click channels
                      zero-initialised, ImageNet norm inside the model)
                    build.py (build_model from config, detect_arch from weights)
src/training/       losses.py (BCEDiceLoss, MultiMaskLoss), metrics.py
                    (iou_score, best_of_n_iou, select_masks), checkpoints.py
                    (atomic save, full-state resume), device.py
src/inference/      predictor.py (full-res image + clicks -> mask; reads its
                      settings from an exported checkpoint, falls back to
                      configs/ for a training one)
src/app/            ui.py (Blocks layout and event wiring), pages.py (Home and
                      Help copy). Both entry points are thin shells over this.
scripts/            app.py (launch locally), export_model.py (deployment
                    checkpoint), deploy_space.py (push app + weights),
                    train.py (M3 overfit check), train_full.py (real training),
                    train_simple.py (readable walkthrough version),
                    export_ade20k.py, analyze_dataset.py, metacentrum/*.pbs
deploy/huggingface/ what gets uploaded to the Space: app.py, requirements.txt
                    (pins +cpu wheels), README.md (Space frontmatter)
docs/DEPLOY.md      install / run / publish, with troubleshooting tables
tests/              test_app_wiring.py -- builds a throwaway model, so it needs
                    no checkpoint and no network
notebooks/          results.ipynb (curves, baselines, examples)
results/            archived history.json + best.pt per finished run
```

## Data

Official ADE20K registration was broken; the dataset is built from the Hugging
Face mirror `1aurent/ADE20K`. COCO and PSG were ruled out by Kassem.

On the cluster at
`/storage/brno2/home/mais999/projects/ade20k-reference/dataset/ADE20K_2021_17_01/images/ADE`.

Measured with `scripts/analyze_dataset.py` over all 12,003 images:

- 12,003 images (10,000 train split + 2,000 val split + 3 samples)
- **240,671 objects**, **2,041 distinct classes**
- objects per image: min 2, median 16, mean 20.1, max 275 (95th pct 49)
- **63.8% of objects share their class with another object in the same image**
- top 150 classes cover 91% of objects; top 300 cover 96.2%
- mask PNGs use exactly {0, 128, 255} = background / occluded / visible;
  `objects[i].id` aligns 1:1 with the instances sequence; some `parts` fields
  are null (the export handles this and skips malformed rows rather than
  aborting)

Training uses the visible mask (`arr == 255`).

## Experimental record (held-out test IoU unless noted)

| # | Setup | Result |
|---|-------|--------|
| 1 | from scratch, 128px sq, depth 3, base 32, 3k images | 0.4990 |
| 2 | same, 10k images | 0.5035 |
| 3 | from scratch, 384x512, depth 4, neighbour negatives, 3k | 0.5125 |
| 4 | ResNet-34 pretrained encoder, 384x512, 3k | **0.5710** |
| 5 | same, 12k images (walltime-killed before test eval) | val 0.6175 |
| 6 | + 3 candidate masks with score head | in progress |

Established by direct experiment:

- **lr 0.003 is unstable** (deterministic collapse ~epoch 587 in the overfit
  test); 0.001 stable. Overfit check passes at IoU 1.0000 with
  base_channels >= 32; at 16 it stalls at ~0.91.
- **Data volume was not the from-scratch bottleneck**: 3.3x images gained
  +0.0045. But with the pretrained encoder, 3k -> 12k gained **+0.045** — the
  difference is the train/val gap. A model that overfits benefits from more
  data; one that cannot fit the data at all does not.
- **More epochs never help** past the plateau (~epoch 30-40 in every run).
- **Resolution + depth + neighbour negatives** together: +0.009.
- **Pretrained encoder**: +0.0585, the largest single change measured.
- Trivial baselines on test (128px era): random 0.041, all-foreground 0.048,
  disk-at-click 0.116. IoU is not accuracy; random scores near 0, not 0.5.
- Single-click IoU is the only thing measured. Multi-click and NoC are not
  implemented.

## Design decisions and why

**Rejected: fixed per-class or per-slot output tensor** (Kassem's suggestion,
2026-08-19). Measurement decided it: one channel per class would make 63.8% of
objects individually unselectable, and top-150 classes only cover 91% of
objects. One channel per object slot would need 275 channels for the largest
image and reintroduces the ordering problem (nothing says which chair goes in
which slot). The click already identifies the object with no channel limit.

**Adopted instead: SAM-style 3 candidate masks** with a score head. Handles the
genuine ambiguity (nested objects containing one click) without losing instance
separation.

**Click encoding stays disks**, not distance maps: RITM's ablation found disks
better. The `distance` option exists in the code but is non-default and
untested.

## Literature (RITM, SimpleClick, FocalClick, Xu et al. 2016)

Read 2026-08-16. What methods reaching 0.8+ do that we do not:

- ImageNet-pretrained encoders — adopted, biggest single gain
- Xu-style negative clicks on neighbouring instances — adopted
  (`clicks.neighbor_negative_prob`)
- Target-centered crops (FocalClick): crop around the click instead of resizing
  the whole scene; a 128px network is competitive that way. Not implemented,
  and the strongest remaining candidate.
- Previous mask as an extra input channel + iterative click training (RITM):
  what makes clicks 2+ converge, and what NoC measures. Not implemented.
- Caveat: they train on COCO+LVIS (~1.5M instances). ADE20K is smaller and
  stuff-heavy, so part of any gap is the dataset.

## MetaCentrum specifics (hard-won)

- **`$HOME` differs per node.** You have a home on every storage; a job can land
  with `$HOME` on praha5-elixir or praha2-natur while the data is on brno2.
  Three jobs failed this way. Batch scripts hardcode `/storage/brno2/home/$USER`
  and pass `--data-root`. Overriding HOME for a Singularity container does not
  work — it refuses.
- **Container:** `/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF`
  has torch 2.7 + CUDA + torchvision + numpy/scipy/PIL/yaml/matplotlib. No
  gradio, no datasets/pyarrow.
- **GPU capability:** older cards (konos, 1080 Ti) pass
  `torch.cuda.is_available()` then fail at the first kernel launch. Jobs request
  `gpu_cap=compute_75`; train.pbs also `gpu_mem=30gb` for the larger
  activations. This narrows the pool, so queue waits can be hours.
- **Some compute nodes have no internet** (pip and wget fail). ImageNet weights
  are cached once from a frontend to
  `/storage/brno2/home/mais999/.cache/torch/hub/checkpoints/resnet34-b627a593.pth`;
  train.pbs sets TORCH_HOME there and verifies. The Gradio app needs a node
  with internet; if it fails, run the app on the laptop instead.
- **PBS writes its .o file only at job end**, so jobs tee live logs to
  `outputs/train.log` / `outputs/app.log`.
- **Walltime is 12h.** ~230s/epoch at 384x512 batch 32 on 3k images; ~1350s on
  12k images.
- **Dataloader is the bottleneck**, not the GPU (measured ~124ms/batch against
  ~15ms GPU work). Fixed by `load_instance` (decode 1 mask, not ~20), 16
  workers, persistent_workers, pin_memory, prefetch.

## Verification habits

After any pipeline or model change, run the overfit check
(`python scripts/train.py --device cpu --subset-size 4`) — it must reach IoU
~1.0. It writes to `outputs/checkpoints/`, so delete that afterwards or the next
`--auto-resume` picks up a 2-example toy model.

After any interface change, run `python tests/test_app_wiring.py`. It exists
because of a bug that was invisible for weeks: the app painted the mask and
click markers onto the same `gr.Image` it fed the model, so every click after
the first segmented an annotated image instead of the photo. Nothing errored;
refinement just quietly did not work. Gradio's `inputs=[...]` lists are
positional and unchecked, so this class of mistake shows up in the browser, not
at import — the test calls the handlers through the listeners Blocks actually
registered. Compile-check and logic-test before pushing; the cluster round-trip is
expensive. Every reported claim traces to a measured number, and falsified
hypotheses stay recorded in PROGRESS.md rather than being deleted.

## Outstanding

1. **Deploy the Space** (2026-08-22: pages and docs are built, nothing is
   published yet). Home / Segment / Help tabs exist, `docs/DEPLOY.md` is
   written, `scripts/deploy_space.py` is dry-run tested. What remains is
   exporting the run-6 checkpoint, running the two upload commands, and sending
   Kassem the URL -- with a warning that a free Space sleeps and the first
   visit takes a minute.
2. **GitLab migration**, excluding `.claude/`, `CLAUDE.md`,
   `segmentation-project-prompt.md`.
3. **Multi-click / NoC evaluation** — the field's standard metric, never
   measured, and the honest way to assess an interactive tool.
4. **Model quality**: target crops and previous-mask iterative training are the
   literature-supported next levers. Attention remains unimplemented.
5. **Update README results table and the notebook** with the latest checkpoint.
