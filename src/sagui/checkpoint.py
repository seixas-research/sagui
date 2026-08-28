"""Saving and restoring trained models.

A checkpoint is self-contained: it carries the configuration, the species
table and the dataset statistics next to the weights, so the inference tools
can rebuild the exact model without the original YAML file.  The ``task``
field says whether the weights belong to an interatomic potential or to a
generative diffusion model, and the loaders dispatch on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config
from .data.statistics import DatasetStatistics, GenerativeStatistics
from .data.ztable import ZTable
from .generative.diffusion import MaterialsDiffusion
from .models.base import InteratomicPotential
from .models.registry import build_model
from .version import __version__

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_model",
    "load_generative_model",
    "CHECKPOINT_FORMAT",
]

#: Bumped whenever the on-disk layout changes incompatibly.
CHECKPOINT_FORMAT = 2


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: Config,
    z_table: ZTable,
    stats: DatasetStatistics | GenerativeStatistics,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write a self-contained checkpoint.

    ``state_dict`` is *merged onto* the model's own weights rather than
    replacing them, so a partial override works: the EMA tracks floating-point
    tensors only, and integer buffers such as ``atomic_numbers`` still have to
    be stored.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = dict(model.state_dict())
    if state_dict is not None:
        weights.update(state_dict)

    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "sagui_version": __version__,
        "task": config.task,
        "config": config.to_dict(),
        "atomic_numbers": list(z_table.zs),
        "statistics": stats.to_dict(),
        "epoch": int(epoch),
        "metrics": dict(metrics or {}),
        "model_state_dict": {k: v.detach().cpu() for k, v in weights.items()},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Read a checkpoint file without instantiating the model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    fmt = payload.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"checkpoint format {fmt} is not supported by sagui {__version__} "
            f"(expected {CHECKPOINT_FORMAT})"
        )
    return payload


def _prepare(path: str | Path, expected_task: str) -> tuple[dict[str, Any], Config, ZTable]:
    payload = load_checkpoint(path, map_location="cpu")
    task = payload.get("task", "potential")
    if task != expected_task:
        raise ValueError(
            f"'{path}' holds a '{task}' model, but a '{expected_task}' model was requested"
        )
    return payload, Config.from_dict(payload["config"]), ZTable(payload["atomic_numbers"])


def load_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[InteratomicPotential, Config, ZTable]:
    """Rebuild a trained *potential* from a checkpoint.

    Returns the model (in ``eval`` mode, on ``device``), its configuration and
    the species table it was trained with.
    """
    payload, config, z_table = _prepare(path, "potential")
    stats = DatasetStatistics.from_dict(payload["statistics"])

    dtype = dtype or torch.get_default_dtype()
    with _default_dtype(dtype):
        model = build_model(
            config.model,
            atomic_numbers=z_table.zs,
            atomic_energies=stats.atomic_energies,
            energy_scale=stats.energy_scale,
            avg_num_neighbors=stats.avg_num_neighbors,
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config, z_table


def load_generative_model(
    path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[MaterialsDiffusion, Config, ZTable, GenerativeStatistics]:
    """Rebuild a trained *generative* diffusion model from a checkpoint."""
    payload, config, z_table = _prepare(path, "generative")
    stats = GenerativeStatistics.from_dict(payload["statistics"])

    dtype = dtype or torch.get_default_dtype()
    with _default_dtype(dtype):
        model = MaterialsDiffusion(
            config.model,
            config.diffusion,
            num_species=len(z_table),
            lattice_scale=stats.lattice_scale,
            avg_num_neighbors=stats.avg_num_neighbors,
        )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, config, z_table, stats


class _default_dtype:
    """Temporarily switch the default dtype while modules are constructed.

    Buffers (the Clebsch-Gordan tensors, the radial frequencies, ...) are
    materialised at construction time, so they must be created in the dtype the
    caller is going to run in.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype

    def __enter__(self) -> None:
        self.previous = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, *exc: object) -> None:
        torch.set_default_dtype(self.previous)
