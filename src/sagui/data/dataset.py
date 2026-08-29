"""Datasets and file readers built on top of ASE."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read
from torch.utils.data import Dataset

from .atomic_data import AtomicGraph, graph_from_atoms
from .ztable import ZTable

__all__ = ["AtomsDataset", "read_structures", "random_split"]

logger = logging.getLogger(__name__)


def read_structures(
    path: str | Path, index: str = ":", file_format: str | None = None
) -> list[Atoms]:
    """Read every structure ASE can parse from ``path`` (``.xyz``, ``.traj``, ...)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"structure file not found: {path}")
    frames = read(str(path), index=index, format=file_format)
    if isinstance(frames, Atoms):
        frames = [frames]
    if not frames:
        raise ValueError(f"no structures found in {path}")
    return list(frames)


class AtomsDataset(Dataset):
    """Lazily converts ``Atoms`` objects into :class:`AtomicGraph` graphs.

    Neighbour lists are built on access rather than up front: it keeps start-up
    instantaneous and memory flat, at the cost of recomputing them every epoch.
    Set ``cache=True`` to trade memory for speed on small datasets.
    """

    def __init__(
        self,
        frames: Sequence[Atoms],
        z_table: ZTable,
        r_max: float,
        with_labels: bool = True,
        energy_key: str | None = None,
        forces_key: str | None = None,
        stress_key: str | None = None,
        dtype: torch.dtype | None = None,
        cache: bool = False,
    ) -> None:
        self.frames = list(frames)
        self.z_table = z_table
        self.r_max = float(r_max)
        self.with_labels = with_labels
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.dtype = dtype or torch.get_default_dtype()
        self._cache: dict[int, AtomicGraph] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> AtomicGraph:
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        graph = graph_from_atoms(
            self.frames[index],
            self.z_table,
            self.r_max,
            with_labels=self.with_labels,
            energy_key=self.energy_key,
            forces_key=self.forces_key,
            stress_key=self.stress_key,
            dtype=self.dtype,
        )
        if self._cache is not None:
            self._cache[index] = graph
        return graph


def random_split(
    frames: Sequence[Atoms], valid_fraction: float, seed: int = 0
) -> tuple[list[Atoms], list[Atoms]]:
    """Deterministic train/validation split of a list of structures."""
    if not 0.0 <= valid_fraction < 1.0:
        raise ValueError(f"valid_fraction must be in [0, 1), got {valid_fraction}")
    n_valid = int(round(len(frames) * valid_fraction))
    if valid_fraction > 0.0:
        n_valid = max(1, min(n_valid, len(frames) - 1))
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(frames), generator=generator).tolist()
    valid_idx = set(order[:n_valid])
    train = [f for i, f in enumerate(frames) if i not in valid_idx]
    valid = [frames[i] for i in order[:n_valid]]
    return train, valid
