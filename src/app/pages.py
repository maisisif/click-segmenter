"""Prose for the Home and Help pages.

Kept out of ui.py so the layout code stays readable, and so the text can be
reviewed as text. Everything here is written for someone who has never seen the
project: no repo paths in the Home page, no jargon without a gloss.

Numbers in this file are measured, not aspirational -- see the experimental
record in CLAUDE.md / README.md. If a new checkpoint changes them, change them
here too; a demo that overstates itself is worse than one that admits a limit.
"""

from __future__ import annotations

_HOME_TEMPLATE = """
# Click to segment

**Click an object in a photo. Get a mask of that object.**

Segmentation normally means cutting an image into every region at once, and then
you still have to find the one you wanted. This does the opposite: you point at
the thing you care about, and only that thing comes back. One click is the whole
interface.

### Try it

Open the **Segment** tab, upload a photo, and click something in it. The mask
appears immediately. If it grabs too much -- the chair *and* the table it stands
on -- switch to **Exclude** and click the part you did not mean. If it grabs too
little, add another **Object** click.

### How it works, briefly

The model is a U-Net with an ImageNet-pretrained ResNet-34 encoder, trained by us
on [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/). It is not SAM
and does not use any off-the-shelf interactive segmenter.

The interesting part is the input. The network sees **five channels**, not three:
red, green, blue, and two more that mark where you clicked -- one for "include",
one for "exclude". The same photo clicked on the lamp and clicked on the chair
are therefore *different inputs*, each with one right answer. That is what makes
a single image worth around twenty training examples, and it is why the click can
select any object without the model needing a fixed list of object types.

{output_paragraph}

### What it does well, and where it struggles

It is most reliable on medium-sized, clearly bounded objects with visible edges.
It is weakest on very small objects, on "stuff" regions with no real boundary
(sky, road, wall), and on an object that is mostly hidden behind another.

Accuracy from a single click is **around 0.6 IoU** — 0.62 on the validation set
for the checkpoint served here, and 0.57 on a held-out test set for the closest
checkpoint that was measured on one. IoU is overlap between the predicted mask
and the true one, and it is a harsh scale: returning a plain disk at the click
scores 0.12, and a random mask scores 0.04. So 0.6 is a real result, not a coin
flip — but it is not a solved problem either, and you will see it fail. The Help
tab is honest about which failures are the model and which are known gaps.
""".strip()


_MULTI_MASK = """
The output is **three candidate masks** plus a confidence score for each. A click
in the middle of a person is genuinely ambiguous — shirt, torso, or whole person
are all defensible — so instead of averaging those into one blurry answer, the
three candidates specialise and the score picks between them.
""".strip()

_SINGLE_MASK = """
The output is **one mask**. A click in the middle of a person is genuinely
ambiguous — shirt, torso, or whole person are all defensible — and a model with
a single output has to commit to one reading, which is a real source of the
errors you will see. A version returning three scored candidates is built and
training; it is not the checkpoint served here.
""".strip()


def home(num_masks: int) -> str:
    """The Home page, describing the checkpoint that is actually loaded.

    How many masks the model outputs is a property of the served weights, not of
    the repository, so it is read from the model instead of written down. This
    is not hypothetical: the first checkpoint prepared for deployment returned
    one mask while this page promised three, which would have been the first
    thing a visitor read.
    """
    return _HOME_TEMPLATE.format(
        output_paragraph=_MULTI_MASK if num_masks > 1 else _SINGLE_MASK
    )


HELP = """
# How to use this

### The basics

1. Go to the **Segment** tab and upload an image, or drag one onto the box.
2. **Click on the object you want.** The mask appears as a green tint.
3. Not right? Refine it:
   - **Object (include)** -- click a part of the object the mask missed.
   - **Exclude** -- click a region the mask wrongly swallowed. Marked in red.
   - **Undo last click** removes the most recent click and re-predicts.
   - **Clear clicks** starts the image over.
4. **Mask threshold** trades completeness against precision. The model outputs a
   confidence per pixel; the threshold is the cut-off for "in the mask". Lower it
   when the mask is patchy, raise it when it bleeds into the background. 0.5 is a
   sensible default, and moving it is often faster than adding a click.
5. **Download mask (PNG)** gives you the mask on its own -- white object, black
   background, at the original image's resolution.

### Getting better results

- **Click the middle, not the edge.** The model was trained on clicks sampled
  from object interiors, so a click near a boundary is out of distribution and
  genuinely less reliable.
- **Exclude clicks work best right next to the mistake.** A click on the
  neighbouring object that the mask leaked into does more than a click on empty
  background far away.
- **Large, close-up objects are easier than small distant ones.** Everything is
  resized to 384x512 before the model sees it, so an object that is 20 pixels
  across in a phone photo has very little left to work with.

### Known limitations (stated plainly)

- **Extra clicks help less than you would expect.** The model is trained on
  single clicks and does not see its own previous mask, so click two is a fresh
  prediction with more input rather than a correction of what it just did. The
  standard fix -- feeding the previous mask back in and training on click
  sequences -- is the next thing being built.
- **Single-click accuracy is what has been measured.** The field's usual metric,
  NoC (number of clicks to reach a given accuracy), is not measured yet.
- **ADE20K is scene-heavy.** Methods that reach 0.8+ IoU train on far more
  object-centric data (COCO+LVIS, over a million instances). Part of the gap is
  the dataset, not only the model.
- **The hosted demo runs on a free CPU tier.** The first request after a period
  of inactivity has to wake the server and can take a minute. After that, each
  click takes a few seconds. It is not slow because of your image.

### Running it yourself

Everything needed is in the project's README: installing the dependencies,
downloading the trained weights, and starting this same interface locally. On a
laptop with no GPU it works exactly as it does here, only without the cold start.
""".strip()
