"""Device selection, shared by the training entry points."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device from config.

    "auto" picks the best available backend. An explicit choice
    ("cuda"/"mps"/"cpu") fails loudly if unavailable, which is what we want on
    MetaCentrum: a GPU job silently falling back to cpu would waste the whole
    allocation without anyone noticing.

    Note that `torch.cuda.is_available()` returning True is not a guarantee the
    GPU actually works — a build without kernels for the card's compute
    capability passes this check and then fails at the first kernel launch.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device: cuda requested in config but torch.cuda.is_available() is False")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device: mps requested in config but torch.backends.mps.is_available() is False")
    return torch.device(preference)
