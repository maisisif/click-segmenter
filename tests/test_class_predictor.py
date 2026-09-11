"""Smoke test for the class model's export and inference path.
Run: python tests/test_class_predictor.py

A tiny from-scratch UNet with three input channels and two output channels
stands in for the trained ResNet checkpoint (no network, no weights download).
The test checks that a training-style checkpoint exports, that the export
loads without configs/, that predictions come back at the photo's own size,
and that a click resolves to a class index or to None.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.export_class_model import build_payload
from src.inference.class_predictor import ClassPredictor
from src.model.build import build_model


def _training_checkpoint() -> dict:
    model = build_model({"arch": "unet", "base_channels": 8, "depth": 2, "in_channels": 3, "num_masks": 2})
    return {
        "epoch": 7,
        "best_epoch": 5,
        "best_val_iou": 0.61,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {"junk": torch.zeros(10)},
        "train_settings": {
            "task": "class_segmentation",
            "class_names": ["bed", "floor"],
            "match": "exact",
            "image_size": [64, 96],
            "in_channels": 3,
        },
    }


def main() -> None:
    photo = (np.random.default_rng(0).random((180, 240, 3)) * 255).astype(np.uint8)
    counts = {"test": {"iou": 0.7, "iou_pooled": 0.72}, "images_with_class": 10, "splits": {}}

    with tempfile.TemporaryDirectory() as d:
        training = _training_checkpoint()
        torch.save(training, Path(d) / "best.pt")
        direct = ClassPredictor(Path(d) / "best.pt", device="cpu")
        assert direct.class_names == ["bed", "floor"]
        print("training checkpoint loads and names its classes                  ok")

        payload = build_payload(training, counts, half=True, source="test")
        assert "optimizer_state_dict" not in payload
        torch.save(payload, Path(d) / "export.pt")
        predictor = ClassPredictor(Path(d) / "export.pt", device="cpu")

    assert predictor.class_names == ["bed", "floor"]
    assert predictor.test["iou"] == 0.7 and predictor.trained_epoch == 5
    print("half-precision export loads without configs/ and keeps provenance   ok")

    probs = predictor.predict(photo)
    assert probs.shape == (2, 180, 240), probs.shape
    assert 0.0 <= probs.min() and probs.max() <= 1.0
    same = direct.predict(photo)
    assert np.abs(probs - same).max() < 1e-2, "float16 export drifted from the training weights"
    print("probabilities come back per class at the photo's own size         ok")

    assert predictor.predict(photo) is probs, "second call on the same photo should hit the cache"
    masks = predictor.masks(photo, threshold=0.5)
    assert masks.dtype == bool and masks.shape == probs.shape
    hit = predictor.class_at(photo, 90, 120, threshold=0.0)
    assert hit in (0, 1)
    assert predictor.class_at(photo, 90, 120, threshold=1.01) is None
    assert predictor.class_at(photo, 10_000, -5, threshold=0.0) in (0, 1), "out-of-range clicks must clamp"
    print("click resolves to a class index, or None below threshold          ok")

    print("\nall class predictor tests passed")


if __name__ == "__main__":
    main()
