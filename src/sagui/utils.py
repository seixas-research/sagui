"""Small cross-cutting helpers: seeding, devices, dtypes, logging."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch

__all__ = ["set_seed", "resolve_device", "resolve_dtype", "setup_logging", "count_parameters"]

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch from a single integer."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``"auto"`` into the best available device.

    MPS is deliberately *not* auto-selected: force training needs
    double-backward through the autograd graph, which the Metal backend does
    not support for every op used here.  Ask for it explicitly if you want it.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype '{name}'; choose from {sorted(_DTYPES)}") from exc


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure a single stream (plus optional file) handler on the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
