"""Shared fixtures.

Every test runs in double precision: the equivariance and finite-difference
checks below assert agreement to ~1e-10, which single precision cannot deliver.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1234)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def cluster() -> Atoms:
    """A small, chemically mixed, non-periodic cluster."""
    rng = np.random.default_rng(0)
    positions = rng.normal(scale=1.7, size=(7, 3))
    # Push atoms apart so no pair sits on top of another.
    positions[0] += np.array([2.0, 0.0, 0.0])
    return Atoms("H3O2C2", positions=positions)


@pytest.fixture
def crystal() -> Atoms:
    """A periodic fcc cell, rattled to break the symmetry."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(stdev=0.05, seed=7)
    return atoms


@pytest.fixture
def labelled_frames() -> list[Atoms]:
    """Lennard-Jones-labelled argon clusters, as a miniature training set."""
    from ase.calculators.lj import LennardJones

    rng = np.random.default_rng(3)
    base = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 1)
    frames = []
    for _ in range(12):
        atoms = Atoms("Ar8", positions=base.get_positions()[:8])
        atoms.rattle(stdev=0.15, seed=int(rng.integers(1 << 30)))
        atoms.calc = LennardJones(sigma=3.4, epsilon=0.0104, rc=7.0)
        atoms.get_potential_energy()
        frames.append(atoms)
    return frames
