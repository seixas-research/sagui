"""Small cross-cutting helpers: seeding, devices, dtypes, logging."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch

__all__ = [
    "set_seed",
    "resolve_device",
    "resolve_dtype",
    "resolve_device_and_dtype",
    "setup_logging",
    "count_parameters",
    "clip_gradients",
]

logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch from a single integer."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Turn ``"auto"`` into the best available device.

    Preference order is **CUDA, then Apple MPS, then CPU**.  MPS handles every
    operation SAGUI uses, including the double backward that force training
    needs; its one hard limitation is that Metal has no ``float64`` at all,
    which :func:`resolve_device_and_dtype` takes care of.
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device_and_dtype(
    device: str = "auto", dtype: str | torch.dtype = "float32"
) -> tuple[torch.device, torch.dtype]:
    """Resolve a compatible ``(device, dtype)`` pair, adjusting with a warning.

    Metal cannot represent ``float64`` under any circumstances, so a request
    for double precision on MPS is downgraded to ``float32`` rather than
    failing deep inside a forward pass.  Pass ``--device cpu`` when the extra
    precision is what actually matters -- finite-difference checks, tight
    geometry optimisations, or anything comparing energies at the micro-eV
    level.
    """
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype) if isinstance(dtype, str) else dtype

    if resolved_device.type == "mps" and resolved_dtype == torch.float64:
        logger.warning(
            "float64 is not supported by the MPS backend; falling back to float32 "
            "(use --device cpu to keep double precision)"
        )
        resolved_dtype = torch.float32
    return resolved_device, resolved_dtype


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


def clip_gradients(module: torch.nn.Module, max_norm: float | None) -> torch.Tensor:
    """Clip the gradients of ``module`` and return their total norm.

    The returned norm is the one *before* clipping, so a non-finite value is a
    reliable signal that the backward pass overflowed.  That check matters more
    than it looks: ``clip_grad_norm_`` divides by the total norm, so a single
    ``inf`` gradient turns **every** parameter into ``NaN``, and the model --
    and any exponential moving average tracking it -- is destroyed for the rest
    of the run.  Callers are expected to skip the optimiser step when this
    returns a non-finite value.
    """
    parameters = [p for p in module.parameters() if p.grad is not None]
    if not parameters:
        return torch.zeros(())
    if max_norm:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad.detach()) for p in parameters])
    )
