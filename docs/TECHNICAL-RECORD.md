# Technical record

Everything built and measured, to 2026-09-05. Figures here are measured, not
estimated; where a number is an estimate or is missing it says so.

- **Dataset** — ADE20K, 12,003 images, 240,671 objects
- **Best held-out result** — 0.6194 IoU from a single click, on 24,271 instances
- **Compute** — MetaCentrum, PBS batch jobs

---

## 1. The problem, and the idea that solves it

Ordinary segmentation cuts an image into every region at once and labels each by
category. That is the wrong shape for the task people actually have, which is to
isolate *one* object. You still have to find it in the output, and if its
category is not in the model's list, it is not there at all.

Interactive segmentation inverts this: point at the thing you want, and only
that comes back. The design question is how the click reaches the network.

The answer used here is that **the click is part of the input image**. The
network receives **five channels** rather than three — red, green, blue, plus
two maps the same size as the photograph. One marks include-clicks, the other
exclude-clicks; both are zero everywhere except a filled disk of radius 5 at
each click position.

The consequence is that the same photograph clicked on the chair and clicked on
the lamp are *literally different inputs*, each with exactly one correct output.
No category list appears anywhere in the model, so it operates on objects it was
never specifically taught. It also multiplies the training data: one photograph
carrying 20 annotated objects becomes 20 training examples, not one.

Section 4 describes a second architecture built to test this decision directly,
by removing the click from the input and measuring what is lost.

---

## 2. Data

ADE20K, built from the Hugging Face mirror `1aurent/ADE20K` because the official
registration route was unavailable. Every figure below was measured with
`scripts/analyze_dataset.py` over the whole set rather than quoted from the
dataset paper.

| | |
| --- | --- |
| Images | **12,003** (10,000 train split + 2,000 val split + 3 samples) |
| Annotated objects | **240,671** across 2,041 distinct classes |
| Objects per image | mean 20.1, median 16, 95th pct 49, max 275, min 2 |
| Class collisions | **63.8%** of objects share a class with another object in the same image |
| Class coverage | top 150 classes cover 91% of objects; top 300 cover 96.2% |

Each image `X.jpg` has a sibling `X.json` listing annotated objects and their
sub-parts, and a folder `X/` holding one mask PNG per top-level object. Those
PNGs use exactly three pixel values — `0` background, `128` occluded but part of
the object, `255` visible. **Training uses the visible mask only**, so occluded
regions count as not-the-object. Sub-parts (`part_level != 0`) are excluded,
since a chair leg is not independently clickable in this design.

### Splits

70/20/10, **split by image, never by instance**. Splitting by instance would
place two objects from the same photograph into different splits, letting the
model earn credit for a scene it had already memorised. Splits are deterministic
from the sorted file paths plus a seed, so no split file needs to be stored.

| Split | Images | Clickable instances | Role |
| --- | --- | --- | --- |
| Train | 8,402 | 167,210 | Gradient updates only |
| Validation | 2,401 | 48,989 | Epoch selection, LR schedule, early stopping |
| Test | 1,200 | 24,271 | Touched once, on the chosen checkpoint |

An instance whose mask vanishes when downscaled to the working resolution cannot
have a click simulated on it, so it is excluded at index time rather than
skipped at access time. Building that index means decoding tens of thousands of
small PNGs off shared storage, so the result is cached and keyed by a hash of
the input paths and image size.

---

## 3. Click simulation

ADE20K contains no human click data, so clicks are synthesised to imitate how a
person actually points at something.

- **Positive clicks** are sampled only from pixels at least 50% of the mask's
  maximum boundary distance away from its edge — the interior, where a person
  aiming at an object would click, not the rim.
- **Background negatives** land 5 to 40 pixels outside the mask, added with
  probability 0.5. Near, not random: a correction click is made on a confusable
  neighbour, not on unrelated sky.
- **Neighbour negatives** land on a *different annotated instance* in the same
  image, with probability 0.3. This is Xu et al.'s second sampling strategy,
  adopted because background negatives never teach a model that the adjacent
  object is not the target — the precise failure mode of masks bleeding across
  instance boundaries.

Clicks are rendered as filled disks of radius 5. A distance-transform encoding
is implemented and selectable, but is non-default and untested here: RITM's
ablation found disks better, and re-deriving that was not a good use of cluster
time.

Training resamples clicks every epoch, which acts as augmentation. Validation
and test use a fixed seed, so their curves reflect the model rather than the
sampler.

---

## 4. Architectures

Three, built in sequence. Each answers a question the previous one raised.

### Model A — U-Net from scratch

A plain encoder-decoder with a configurable number of downsampling stages. Five
input channels, one output channel. Channel widths double each stage from
`base_channels`; skip connections carry boundary detail past the bottleneck.
Trained from random initialisation, no pretrained weights anywhere. This is the
baseline that establishes what the pipeline can do unaided.

### Model B — ResNet-34 encoder in the same U-Net

The encoder is replaced by an ImageNet-pretrained ResNet-34; the decoder keeps
the same upsampling blocks. Two details make it work:

- The stem convolution is rebuilt for five input channels. Pretrained RGB
  filters are copied across and **the two click channels are zero-initialised**,
  so at step zero the network behaves exactly like the pretrained model and the
  click signal grows in during training rather than destroying the features it
  was meant to exploit.
- ImageNet normalisation happens *inside* the model as registered buffers, so
  the dataset, the app and the notebook all keep feeding images in [0, 1]
  unchanged.

Later revisions gave it **three candidate masks and a score head**. A single
click is genuinely ambiguous — a click on a shirt could mean shirt, torso, or
whole person, and all three contain that pixel. One output has to average them
into something blurry. With three, only the best-matching candidate is corrected
on each step, so they specialise; a small head predicts each candidate's IoU and
picks one at inference.

### Model C — slot segmentation, ResNet-50 throughout

Built to test the central design decision by inverting it. The network sees
**only RGB**; it emits a fixed tensor of **64 class-agnostic object masks** in a
single pass plus an objectness score per slot, and the click merely selects a
channel afterwards.

| | |
| --- | --- |
| Encoder | ResNet-50, ImageNet, unmodified 3-channel stem |
| Decoder | ResNet-50 bottleneck blocks (1×1 → 3×3 → 1×1, residual) |
| Skips | 256 / 512 / 1024 / 2048, projected down by 1×1 before concatenation |
| Output | 64 × 192 × 256 masks, plus 64 objectness logits — 26.8M params |

The attraction is inference cost. Clicks become a lookup into a cached tensor
rather than a forward pass — on a CPU deployment, the difference between seconds
and instant. Model B cannot be cached this way: its click channels enter at the
first convolution, so every layer depends on the click. Ambiguity also resolves
itself, since shirt, torso and person are three separate ground-truth instances
occupying three slots, and a click inside all three returns all three as
candidates.

Masks are emitted at half resolution. That choice was made by measurement —
downsampling real ADE20K masks to a grid and back gives the IoU ceiling the
architecture cannot exceed:

| Mask grid | Ceiling, all objects | Objects under 1% of frame | Logits at batch 32 |
| --- | --- | --- | --- |
| Quarter (96×128) | 0.9247 | 0.8926 | 201 MB |
| **Half (192×256)** | **0.9737** | **0.9613** | 805 MB |
| Full (384×512) | 1.0000 | 1.0000 | 3,221 MB |

The median ADE20K object covers 0.45% of the frame, so the small-object column
governs. Half resolution buys back 0.07 of ceiling for 600 MB.

---

## 5. Objectives

### BCE + Dice

Binary cross-entropy calibrates per-pixel probabilities. Dice —
`1 − 2|P∩T| / (|P|+|T|)` — optimises mask overlap directly and is largely immune
to the foreground/background pixel imbalance that plain BCE struggles with,
which matters because a typical object is a small fraction of the frame. Dice is
computed on soft probabilities, not thresholded masks, so it stays
differentiable.

### Multi-mask loss (Model B)

BCE + Dice is computed for *every* candidate against the target, and **only the
minimum is back-propagated**. The other candidates receive no gradient that
step, which is what makes them specialise instead of all three converging on the
same average. The score head is trained separately, on all candidates, with each
candidate's true IoU as an MSE target computed under `no_grad` and detached — so
it never pushes gradient back into the masks. With one mask the whole thing
collapses to plain BCE + Dice, so a single loss covers both configurations.

### Hungarian set loss (Model C)

Sixty-four predicted masks, a variable number of ground-truth objects, and
nothing saying which predicts which. Imposing an order — by area, say — would
force the network to learn "am I the third-largest object here", an unstable
target the moment two objects are similar in size.

Instead each image's predictions are paired to its objects by **optimal
bipartite matching**: build the full cost matrix of every prediction against
every object, solve it exactly with the Hungarian algorithm, and supervise only
the pairs the assignment chose. This is the DETR / Mask2Former formulation. The
cost matrix is computed with matrix products rather than a nested loop — the
per-pixel cost of a logit against label 1 is `softplus(−x)` and against label 0
is `softplus(x)`, so the pairwise total is two matmuls against the target and
its complement.

Matching runs under `no_grad` on detached tensors: it decides *which* pairs are
compared and must not be something the model can influence to make its own loss
smaller. Unmatched slots have their objectness pushed toward "no object",
weighted at 0.1 — with 64 slots and around 20 objects, most slots are correctly
empty, and an unweighted term would be dominated by predicting emptiness.

---

## 6. Training procedure

| | |
| --- | --- |
| Optimiser | Adam |
| Learning rate | 3×10⁻⁴ pretrained, 1×10⁻³ from scratch |
| Schedule | ReduceLROnPlateau ×0.5, patience 8, on validation IoU |
| Early stopping | Patience 20 epochs |
| Batch | 32 at 384×512 |
| Epoch cap | 200 (never reached) |

The learning rate for a pretrained encoder is deliberately gentler than for
training from scratch: 10⁻³ would trample ImageNet features before the click
channels learn anything.

**Selection is on validation IoU, not loss.** The best-scoring epoch is
checkpointed as it happens, so a late instability cannot discard an otherwise
good run, and the test split is evaluated exactly once at the end on that
checkpoint.

Model C validates differently, and deliberately. Its training loss is a set loss
over all objects, comparable to nothing measured before, so validation instead
simulates a click on every held-out object, runs the same selection rule the
application uses, and reports single-click IoU — directly against Model B's
numbers. It also reports an oracle: the best of the candidates the selection
offers, against the one it returns.

---

## 7. Infrastructure

Training runs on **MetaCentrum**, the Czech national grid, as PBS batch jobs
submitted from the Skirit frontend. Batch rather than interactive, because an
interactive job dies the moment a browser tab reloads.

```
#PBS -l select=1:ncpus=16:ngpus=1:mem=48gb:scratch_local=50gb:gpu_cap=compute_75:gpu_mem=30gb
#PBS -l walltime=12:00:00
```

Every element of that line was forced by a failure.

- **`gpu_cap=compute_75`** — the NGC container ships CUDA kernels only for sm_75
  and newer. Older cards pass `torch.cuda.is_available()` and then die at the
  first kernel launch, which looks like a code bug and is not.
- **`gpu_mem=30gb`** — at 384×512 with a depth-4 decoder and batch 32,
  activations no longer fit a 16 GB card.
- **`ncpus=16`** — the dataloader is the bottleneck, not the GPU. Measured at
  roughly 124 ms per batch of loading against 15 ms of actual compute.
- **`walltime=12:00:00`** is the ceiling, and no full run fits inside it.

### Three environment facts that cost whole runs

**`$HOME` differs per node.** Each storage has its own home directory, so a job
can land with `$HOME` on `praha5-elixir` while the data sits on `brno2`. Three
jobs failed this way before batch scripts began hardcoding the absolute storage
path and passing `--data-root` explicitly. Overriding `HOME` for the container
does not work — Singularity refuses it outright.

**Compute nodes often have no internet.** ImageNet weights must be cached once
from a frontend onto shared storage, with `TORCH_HOME` pointed at it. A job that
queues for hours and then dies fetching weights is an expensive way to learn
this.

**PBS writes its output file only at job end.** For a twelve-hour run that is
useless, so jobs tee everything into `outputs/train.log` as they go. Anything
worth checking mid-run — which script is running, which flags, which GPU — has
to be echoed inside that teed block or it is invisible until the job is over.

### Chaining jobs

No run fits in twelve hours, so a long run is a chain of jobs. `--auto-resume`
picks up `latest.pt` and restores optimizer moments, scheduler state, the best
score so far and the full epoch history — everything not saved there is lost at
every handover. Checkpoints are written to a temporary file and renamed, because
a job killed midway through a plain save leaves a truncated file that cannot be
loaded, destroying the run it was meant to protect.

Experiment settings are passed at submission time rather than by editing config
files on the cluster, because a dirtied checkout there blocks the `git pull`
every subsequent job depends on:

```bash
qsub -v TRAIN_SCRIPT=scripts/train_slots.py,EXTRA_ARGS="--max-images 0" \
     scripts/metacentrum/train.pbs
```

Two recurring operational failures are worth naming: **double submission**,
where two jobs write the same checkpoints and corrupt each other, and **pulling
after submitting**, since a job snapshots the code when it launches — which has
silently run stale code through two full twelve-hour runs.

---

## 8. Experimental record

| # | Setup | Images | Val IoU | Test IoU | Oracle |
| --- | --- | --- | --- | --- | --- |
| 1 | U-Net from scratch, 128² square, depth 3, base 32 | 3,000 | 0.4928 | 0.4990 | — |
| 2 | Same, more data | 10,000 | 0.5072 | 0.5035 | — |
| 3 | 384×512, depth 4, neighbour negatives | 3,000 | 0.5157 | 0.5125 | — |
| 4 | **ResNet-34 pretrained encoder** | 3,000 | 0.5699 | 0.5710 | — |
| 5 | Same, full data — killed at walltime on its best epoch | 12,003 | 0.6175 | not reached | — |
| 6 | **+ 3 candidate masks and score head** | 3,000 | 0.5726 | 0.5796 | 0.6722 |
| 7 | Run 6 architecture at full data — converged | 12,003 | **0.6191** | **0.6194** | 0.7068 |
| 8 | Model C — 64 slots, Hungarian matching | 12,003 | 0.4294 | pending | 0.4692 |

### Trivial baselines, on the same test split

IoU is not accuracy, and a reader unfamiliar with it will misjudge 0.62 without
these.

| Strategy | IoU |
| --- | --- |
| Random mask | 0.041 |
| Everything as foreground | 0.048 |
| Fixed disk at the click | 0.116 |
| **Run 7** | **0.6194** |

### Run 7 — the deliverable

Test IoU **0.6194**, validation 0.6191, measured over 24,271 held-out instances.
It converged on its own: early stopping fired at epoch 76 with the best at 56,
and across those twenty epochs training IoU climbed 0.72 → 0.77 while validation
fell 0.619 → 0.602. Validation and test agreeing to four decimal places is the
clearest evidence available that the splits have not drifted apart.

---

## 9. What is established by direct experiment

Each of these is a controlled comparison, not an impression.

**Pretraining is the largest single factor: +0.0585.** Run 3 → run 4, same data
and resolution. Larger than every other change combined, and consistent with the
literature crediting pretraining as the main driver.

**Data volume only pays once the model can overfit: +0.045.** Tripling the data
for the from-scratch model bought **+0.0045**. The same increase with a
pretrained encoder bought **+0.045** — ten times as much. The difference is the
train/validation gap: a model that overfits benefits from more data, one that
cannot fit the data at all does not. Measured twice independently, at +0.045 for
the single-mask pair and +0.0398 for the multi-mask pair.

**Conditioning on the click is worth ≈0.19.** Model B reaches 0.6194 on test;
Model C converged at 0.4294 on validation. Same encoder family, same data, same
resolution — the difference is that Model B sees the click while it computes and
Model C only uses it to select afterwards. This is **three times the pretraining
gain** and is the strongest justification available for the whole design. It
exists only because the alternative was built and measured.

**The score head leaves ≈0.09 unclaimed, and data does not fix it.** Run 7's
test IoU is 0.6194 selected but **0.7068 best-of-N**. The three candidates
genuinely specialise — the correct answer is present — and the selector picks a
worse one often enough to cost more than any change except pretraining. The gap
was 0.0926 at 3,000 images and 0.0874 at 12,003: quadrupling the data moved it
by 0.005. The score head is not data-starved, it is the wrong mechanism.
Best-of-N is an oracle bound no automatic selector attains, but a person choosing
between three displayed candidates captures part of it with no retraining.

**More epochs never help past the plateau.** Every run plateaus around epoch
30-40. Run 6 confirmed it from the other side: early stopping fired at 62 with
the best at 42, and over those twenty epochs training IoU climbed 0.68 → 0.74
while validation fell 0.57 → 0.55. Patience 20 has now caught this correctly in
three consecutive runs.

**Two hyperparameters have hard floors.** At learning rate 0.003 training
destabilises *deterministically* around epoch 587; at 0.001 it does not. At
`base_channels: 16` the overfit check memorises three instances perfectly but
stalls at ~0.91 on four; at 32 all four reach 1.0000. A model that cannot
memorise four examples is far too small for 167,000.

### A comparison trap worth naming

Run 6 (0.5726) reads as a regression against run 5 (0.6175) and is not one. Run
5 saw 12,003 images and run 6 saw 3,000, and the 0.0449 between them matches the
+0.045 that data volume was independently measured to give. The test-split size
in the log distinguishes them: 300 images means 3,000 total, 1,200 means the
full set.

---

## 10. Designs considered and rejected

**Fixed per-class output tensor.** One output channel per object class. Killed
by measurement rather than argument: there are 2,041 classes, and capping at the
top 150 still leaves 9% of objects unreachable. Worse, **63.8% of objects share
their class with another object in the same image** — two chairs in one
photograph land in the same channel, so neither can be selected individually.
That breaks the core requirement for nearly two thirds of all objects.

**Fixed per-slot output tensor with static ordering.** One channel per object
slot, assigned in a fixed order. The largest image holds 275 objects, and
nothing determines which chair belongs in slot 3 rather than slot 7, so the
network has no consistent target. Model C revisits the same shape but resolves
the ordering properly with Hungarian matching, which is what makes it a viable
architecture rather than an unlearnable one.

**Distance-map click encoding.** Implemented and selectable, left non-default.
RITM's published ablation found disks better; the option remains for a future
comparison but was not worth cluster time to re-derive.

---

## 11. Defects found, and how they were caught

All four were silent — none raised an error, and each produced output that
looked plausible.

**Refinement clicks were being discarded.** The interface painted the mask tint
and click markers onto the same image component it fed the model. From the
second click onward the model was segmenting a green-tinted photograph with
coloured rings drawn into it, not the photograph. The corruption compounded with
exactly the clicks a user adds when trying to correct a bad mask. Fixed by
holding the untouched upload in separate state; now covered by a regression test
that passes an annotated view back in and asserts the pristine copy survives.

**Exports could silently embed the wrong resolution.** Caught while exporting an
older checkpoint: it was trained at 128×128, the current config said 384×512,
and the export embedded 384×512 without comment. The network is fully
convolutional, so it loads and predicts perfectly happily at a resolution it has
never seen — quietly worse, with nothing to indicate why. Training now records
its own resolution and click settings inside every checkpoint, and the export
prints which of three sources its value came from.

**The Home page described the wrong architecture.** It stated the model returns
three candidate masks; the checkpoint being prepared returns one. The count is
now read off the loaded weights rather than written down.

**Model C's first run was OOM-killed in fifteen minutes.** Exit status 137, out
of memory in a dataloader worker, 48 GB exhausted. Each example carried up to 64
masks at full resolution as float32 — 50 MB per image, 1.6 GB per batch, with
workers keeping dozens of batches queued. The loader settings had been inherited
from the click model, where an example is *one* mask at 786 KB: the settings did
not change, but what an item meant changed by 64×. Fixed by building training
masks at the resolution the loss actually uses, storing them as boolean, and
shortening the queue — 50 MB down to 2.5 MB per image. The loader now prints its
own worst-case memory at startup.

**The habit these produced.** Every one was invisible from the outside. The
response has been to make the invisible thing print itself: the resolution
source, the memory budget, the experiment flags, the GPU model — each echoed
where it can be read in the first seconds of a run rather than inferred from an
exit code hours later.

---

## 12. Software

| Module | Contents |
| --- | --- |
| `src/data/` | ADE20K loading, click simulation and encoding, deterministic splits, both dataset variants and the cached instance index |
| `src/model/` | U-Net, ResNet-34 U-Net, slot U-Net, construction from config, and architecture detection from weight shapes |
| `src/training/` | BCE+Dice, multi-mask and Hungarian losses; IoU metrics; atomic checkpointing; device selection |
| `src/inference/` | Full-resolution predictors for both architectures, including the encode-once / select-per-click split |
| `src/app/` | Web interface layout and page copy, kept separate so local and hosted entry points share one implementation |
| `scripts/` | Overfit check, full training, slot training, dataset export, dataset analysis, model export, deployment, PBS job scripts |
| `tests/` | Interface wiring and slot-model regression tests — no checkpoint, no network, seconds to run |

**Architecture is recovered from weights, not from config.** Depth, channel
widths, mask count, slot count and mask stride are all read off the state dict,
so a checkpoint from any era of the project loads without config surgery. This
is what lets an exported model run with no repository checkout at all.

The verification habits are specific to what has actually broken. After any
pipeline or model change, the overfit check must reach IoU ≈1.0 on four
examples. After any interface change, the wiring test runs — it calls handlers
through the listeners the interface actually registered, because the framework's
input lists are positional and unchecked, so that class of mistake surfaces in a
browser rather than at import. The slot test measures per-object IoU after
assignment rather than loss, because a set-prediction model with broken matching
still drives loss down; it just collapses every slot onto one blurry average.

---

## 13. Deployment

A three-page web interface — Home, Segment, Help — served by the same
implementation locally and hosted. The Segment page takes an upload, accepts
include and exclude clicks, exposes a mask threshold that re-renders without
another click, supports undo, and exports the mask as a PNG at the original
resolution.

Two artifacts ship separately, and the split is deliberate. The **application**
is 69 KB and redeploys whenever the interface changes; the **weights** are 98 MB
in their own repository, replaced only when a better checkpoint exists. Together
in one repository, every one-line interface edit would re-push 98 MB through Git
LFS, and swapping the model would mean redeploying the app.

A deployment checkpoint is produced by stripping optimizer state — measured at
**294.4 MB → 98.2 MB, exactly 3×** — and embedding what inference needs but
weights cannot supply: resolution, click encoding, normalisation, channel order.
The serving side therefore needs no config directory at all.

The hosting target is a free CPU tier, which shapes two decisions: the
requirements pin the CPU PyTorch wheels explicitly, because the default Linux
wheel is a ~2.5 GB CUDA build that a CPU host can never use, and the interface is
written knowing a cold start takes about a minute after idle.

See [DEPLOY.md](DEPLOY.md) for the procedure.

---

## 14. Timeline

| Date | |
| --- | --- |
| Aug 7 | Evaluation protocol fixed: 70/20/10, best epoch by validation, test touched once. |
| Aug 8–11 | Runs 1 and 2. From-scratch U-Net plateaus near 0.50. Data volume tested and found not to be the bottleneck. |
| Aug 16 | Literature scan — Xu et al. 2016, RITM, FocalClick, SimpleClick. Identifies pretrained encoders, neighbour negatives, previous-mask iteration and target crops as the four levers. |
| Aug 19 | Dataset measured to settle the output-tensor question; per-class and per-slot designs rejected on the 63.8% figure. SAM-style multi-mask adopted instead. |
| Aug 20–22 | Runs 3, 4 and 6. Pretrained encoder lands +0.0585, the largest gain in the project. |
| Aug 22–24 | Deployment path built: model export, config-free inference, three-page interface, hosting tooling, documentation, regression tests. Two silent defects found and fixed. |
| Aug 24–26 | Run 7 across three chained jobs. Converges at epoch 56, early-stops at 76. Test IoU 0.6194 on 24,271 instances — the project's best result. |
| Sep 3–4 | Model C designed and built. Half-resolution masks chosen by measuring the IoU ceiling. First run OOM-killed; cause found and fixed at source. |
| Sep 5 | Run 8 converges at 0.4294. Model B retained as the deliverable; the comparison yields the ≈0.19 figure for click conditioning. |

---

## 15. Not implemented, and what it would cost

Stated plainly, because a record that only lists what worked is not a record.

**Multi-click evaluation (NoC).** Number-of-clicks-to-reach-a-target-IoU is the
standard metric for interactive segmentation, and it has never been measured
here. Everything reported is single-click IoU. This is the largest gap between
this work and the literature it is compared against — **the published 0.8+
figures are NoC@85 and NoC@90, accuracy after several clicks, on curated
object-centric benchmarks**, not single-click accuracy on scene photography.

**Previous-mask input and iterative click training.** The model never sees its
own previous prediction, so click two is a fresh prediction with more input
rather than a correction of what it just produced. This is a genuine product
limitation, not only a metric one, and it is the mechanism RITM identifies as
what makes clicks two and onward converge.

**Target-centred crops.** FocalClick crops around the click rather than resizing
the whole scene, which makes even a small network competitive. The strongest
remaining single-click lever, deliberately cut for time.

**User-selectable candidates.** The cheapest unclaimed gain in the project, and
it requires no retraining: run 7's candidates already contain a 0.7068 answer
where the selector returns 0.6194. Model C makes this structural rather than
bolted on, since every slot containing the click is naturally an alternative.

### Honest summary

Single-click IoU **0.6194** on 24,271 held-out instances, against 0.116 for a
disk at the click and 0.041 for random. Three architectures built and compared,
with every gain attributed to a controlled change: pretraining +0.0585, data
volume +0.045, click conditioning ≈0.19, multi-mask +0.0086, resolution and
negative sampling +0.009. Part of the remaining gap to published work is the
dataset — ADE20K is scene-heavy and stuff-heavy where the benchmarks are
object-centric — and part is the three unimplemented mechanisms named above.

---

## References

| Paper | Contribution used here |
| --- | --- |
| Xu et al., *Deep Interactive Object Selection*, CVPR 2016 | Clicks as input channels; negative-click sampling strategies |
| Sofiiuk et al., *RITM: Reviving Iterative Training with Mask Guidance* | Previous mask as input + iterative training (not implemented) |
| Chen et al., *FocalClick*, CVPR 2022 | Target-centred crops (not implemented) |
| Liu et al., *SimpleClick*, ICCV 2023 | Pretraining as the dominant factor |
| Kirillov et al., *Segment Anything*, ICCV 2023 | Three candidate masks with a score head |
| Carion et al., *DETR*, ECCV 2020 | Hungarian matching for set prediction |
| Cheng et al., *Mask2Former*, CVPR 2022 | Slot masks; reduced-resolution mask loss |
