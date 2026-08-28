r"""Datasets for the generative task.

Unlike a potential -- where the geometry is given and the graph can be built
once -- the denoiser sees a *corrupted* structure, so the neighbour list
depends on the noise draw and has to be rebuilt for every sample.  Corruption
therefore happens in ``__getitem__``, which also means it runs in the data
loader's worker processes rather than on the training thread.

Each item carries the noised structure *and* the regression targets, so the
training step is a plain forward pass with no further randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import Dataset

from ..data.atomic_data import AtomicGraph, collate_graphs
from ..data.ztable import ZTable
from .corruption import MaterialsCorruption
from .structures import graph_from_arrays

__all__ = ["DiffusionBatch", "DiffusionDataset", "collate_diffusion", "random_rotation"]


@dataclass
class DiffusionBatch:
    """A batch of corrupted structures together with their denoising targets."""

    graph: AtomicGraph  # built from the corrupted structure
    t: torch.Tensor  # [G]   timestep per structure
    t_atom: torch.Tensor  # [N]   the same, broadcast to atoms
    types_0: torch.Tensor  # [N]   clean species indices
    coord_target: torch.Tensor  # [N, 3] sigma * score of the wrapped normal
    lattice_t: torch.Tensor  # [G, 3, 3] normalised, as fed to the network
    lattice_noise: torch.Tensor  # [G, 3, 3] the epsilon to be predicted

    def to(self, device: torch.device | str) -> DiffusionBatch:
        moved: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device)
        return DiffusionBatch(**moved)

    @property
    def num_graphs(self) -> int:
        return self.graph.num_graphs


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    """A uniformly distributed proper rotation (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(generator.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))  # fix the QR sign ambiguity -> Haar measure
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class DiffusionDataset(Dataset):
    """Corrupts crystals on the fly and returns denoising targets."""

    def __init__(
        self,
        frames: Sequence[Atoms],
        z_table: ZTable,
        corruption: MaterialsCorruption,
        r_max: float,
        lattice_scale: float,
        max_neighbors: int = 24,
        rotation_augmentation: bool = True,
        dtype: torch.dtype | None = None,
        seed: int = 0,
    ) -> None:
        self.frames = list(frames)
        for index, atoms in enumerate(self.frames):
            if not all(atoms.get_pbc()) or atoms.get_cell().rank < 3:
                raise ValueError(
                    f"structure {index} is not fully periodic; the generative model "
                    "diffuses a lattice and therefore needs 3D-periodic cells"
                )
        self.z_table = z_table
        self.corruption = corruption
        self.r_max = float(r_max)
        self.lattice_scale = float(lattice_scale)
        self.max_neighbors = int(max_neighbors)
        self.rotation_augmentation = bool(rotation_augmentation)
        self.dtype = dtype or torch.get_default_dtype()
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> DiffusionBatch:
        atoms = self.frames[index]
        rng = np.random.default_rng((self.seed, index, torch.randint(1 << 30, (1,)).item()))

        cell = np.array(atoms.get_cell().array, dtype=float)
        if self.rotation_augmentation:
            # Rotating the cell while keeping the fractional coordinates is a
            # relabelling of the same crystal, and makes the training
            # distribution of lattices exactly rotation-covariant.
            cell = cell @ random_rotation(rng).T

        n_atoms = len(atoms)
        frac_0 = torch.as_tensor(atoms.get_scaled_positions(), dtype=self.dtype)
        types_0 = torch.as_tensor(self.z_table.indices(atoms.get_atomic_numbers()))
        factor = self.lattice_scale * n_atoms ** (1.0 / 3.0)
        lattice_0 = torch.as_tensor(cell, dtype=self.dtype) / factor

        t = self.corruption.sample_timesteps(1)
        t_atom = t.repeat(n_atoms)

        types_t = self.corruption.corrupt_types(types_0, t_atom)
        frac_t, coord_target, _ = self.corruption.corrupt_coords(frac_0, t_atom)
        lattice_t, lattice_noise = self.corruption.corrupt_lattice(lattice_0.unsqueeze(0), t)

        graph = graph_from_arrays(
            frac_t,
            lattice_t[0] * factor,
            types_t,
            r_max=self.r_max,
            max_neighbors=self.max_neighbors,
        )
        # The graph builder may have widened a near-singular cell; the network
        # must be shown the lattice it actually sees.
        lattice_used = (graph.cell / factor).to(self.dtype)

        return DiffusionBatch(
            graph=graph,
            t=t,
            t_atom=t_atom,
            types_0=types_0,
            coord_target=coord_target,
            lattice_t=lattice_used,
            lattice_noise=lattice_noise,
        )


def collate_diffusion(samples: Sequence[DiffusionBatch]) -> DiffusionBatch:
    """Merge per-structure corrupted samples into one batch."""
    return DiffusionBatch(
        graph=collate_graphs([s.graph for s in samples]),
        t=torch.cat([s.t for s in samples]),
        t_atom=torch.cat([s.t_atom for s in samples]),
        types_0=torch.cat([s.types_0 for s in samples]),
        coord_target=torch.cat([s.coord_target for s in samples], dim=0),
        lattice_t=torch.cat([s.lattice_t for s in samples], dim=0),
        lattice_noise=torch.cat([s.lattice_noise for s in samples], dim=0),
    )
