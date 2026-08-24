---
title: Click to Segment
emoji: 🖱️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: mit
short_description: Click an object in a photo and get a mask of just that object.
---

# Click to segment

Click an object in a photo, get a mask of that object. Positive clicks select,
negative clicks exclude.

A U-Net with an ImageNet-pretrained ResNet-34 encoder, trained on ADE20K. Not
SAM and not built on any off-the-shelf interactive segmenter — the click is fed
to the network as two extra input channels alongside RGB, and the network is
trained to return the one object the click points at.

**This Space runs on the free CPU tier**, so the first request after a quiet
period has to wake the container and can take a minute. Afterwards each click
takes a few seconds.

The Home and Help tabs inside the app explain how it works and where it
struggles. Source, training code and measurements: see the project repository.

## Deploying your own copy

This directory is generated, not hand-edited. It is pushed from the project
repository with:

```
python scripts/deploy_space.py --space YOUR_NAME/click-segmenter
```

The trained weights are not stored here — they live in a separate Hugging Face
model repo and are downloaded at startup. See `docs/DEPLOY.md` in the project
repository for the full procedure.
