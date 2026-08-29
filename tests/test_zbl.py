"""The ZBL nuclear core: a physical constraint, so it is tested as one."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from sagui.config import ModelConfig
from sagui.data.atomic_data import collate_graphs, graph_from_atoms
from sagui.data.ztable import ZTable
from sagui.models.registry import build_model
from sagui.models.zbl import ZBLRepulsion

R_MAX = 4.0
Z_TABLE = ZTable([8])


def _pair(model, distance: float) -> dict[str, torch.Tensor]:
    atoms = Atoms(
        numbers=[8, 8],
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=np.eye(3) * 30.0,
        pbc=False,
    )
    graph = graph_from_atoms(atoms, Z_TABLE, R_MAX, with_labels=False)
    return model(collate_graphs([graph]), training=False)


def _model(zbl_cutoff):
    torch.manual_seed(0)
    return build_model(
        ModelConfig(type="strictly_local", r_max=R_MAX, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16], zbl_cutoff=zbl_cutoff),
        Z_TABLE.zs,
    ).eval()


def test_repulsion_is_monotonic_and_positive():
    zbl = ZBLRepulsion(torch.tensor([8]), cutoff=2.0)
    z = torch.full((7, 1), 8.0)
    r = torch.tensor([[0.2], [0.5], [1.0], [1.4], [1.7], [1.9], [2.0]])
    energy = zbl.pair_energy(z, z, r).squeeze(-1)
    assert (energy >= 0).all()
    assert (energy[1:] <= energy[:-1]).all()
    assert float(energy[-1]) == 0.0


def test_repulsion_grows_with_nuclear_charge():
    zbl = ZBLRepulsion(torch.tensor([1, 8, 29]), cutoff=2.0)
    r = torch.tensor([[0.5]])
    energies = [
        float(zbl.pair_energy(torch.tensor([[z]]), torch.tensor([[z]]), r))
        for z in (1.0, 8.0, 29.0)
    ]
    assert energies[0] < energies[1] < energies[2]


def test_switch_off_is_twice_differentiable():
    """The term and its first two derivatives must reach zero together."""
    zbl = ZBLRepulsion(torch.tensor([8]), cutoff=1.8)
    z = torch.tensor([[8.0]])

    def energy(r: float) -> float:
        return float(zbl.pair_energy(z, z, torch.tensor([[r]])))

    h = 1e-4
    for r in (1.79, 1.795, 1.7999):
        first = abs((energy(r + h) - energy(r - h)) / (2 * h))
        second = abs((energy(r + h) - 2 * energy(r) + energy(r - h)) / h**2)
        assert energy(r) >= 0.0
        assert first < 1.0 and second < 10.0
    assert energy(1.81) == 0.0


def test_the_core_makes_short_range_forces_repulsive():
    """The whole point: an untrained network extrapolates the wall arbitrarily."""
    with_core = _model(1.8)
    for distance in (0.4, 0.7, 1.0, 1.4):
        forces = _pair(with_core, distance)["forces"]
        # atom 1 sits at +x, so a repulsive force on it points along +x
        assert float(forces[1, 0]) > 0.0
        assert float(forces[0, 0]) < 0.0


def test_the_core_is_invisible_beyond_its_cutoff():
    """Same seed, so the networks are identical; only the core differs."""
    off, on = _model(None), _model(1.8)
    for distance in (2.0, 2.5, 3.2):
        assert float(_pair(off, distance)["energy"]) == pytest.approx(
            float(_pair(on, distance)["energy"]), abs=1e-12
        )


def test_the_core_conserves_momentum():
    forces = _pair(_model(1.8), 0.6)["forces"]
    assert torch.allclose(forces.sum(dim=0), torch.zeros(3), atol=1e-10)


def test_the_core_carries_no_parameters():
    """It is a constraint, not another thing to fit."""
    model = _model(1.8)
    assert model.zbl is not None
    assert list(model.zbl.parameters()) == []


def test_forces_still_match_finite_differences_with_the_core():
    model = _model(1.8)
    rng = np.random.default_rng(0)
    atoms = Atoms(
        numbers=[8] * 4,
        positions=rng.normal(scale=1.0, size=(4, 3)) + np.array([0.0, 0.0, 0.0]),
        cell=np.eye(3) * 30.0,
        pbc=False,
    )

    def energy(positions) -> float:
        moved = atoms.copy()
        moved.set_positions(positions)
        graph = graph_from_atoms(moved, Z_TABLE, R_MAX, with_labels=False)
        return float(model(collate_graphs([graph]), compute_forces=False, training=False)["energy"])

    graph = graph_from_atoms(atoms, Z_TABLE, R_MAX, with_labels=False)
    analytic = model(collate_graphs([graph]), training=False)["forces"].detach().numpy()

    h = 1e-6
    numerical = np.zeros_like(analytic)
    base = atoms.get_positions()
    for i in range(len(atoms)):
        for a in range(3):
            plus, minus = base.copy(), base.copy()
            plus[i, a] += h
            minus[i, a] -= h
            numerical[i, a] = -(energy(plus) - energy(minus)) / (2 * h)
    assert np.abs(analytic - numerical).max() < 1e-5
