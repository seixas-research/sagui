"""Per-element reference energies from IsolatedAtom frames (the MACE convention)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

from sagui.data.dataset import AtomsDataset
from sagui.data.ztable import ZTable
from sagui.train.stats import compute_statistics, split_isolated_atoms


def _isolated(symbol: str, energy: float, config_type: str = "IsolatedAtom") -> Atoms:
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((1, 3)))
    atoms.info["config_type"] = config_type
    return atoms


def _bulk(seed: int = 0) -> Atoms:
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    atoms.rattle(stdev=0.05, seed=seed)
    atoms.calc = EMT()
    atoms.get_potential_energy()
    return atoms


def test_isolated_frames_are_removed_and_their_energies_returned():
    frames = [_isolated("Cu", -3.75), _bulk(0), _isolated("Au", -2.1), _bulk(1)]
    kept, energies = split_isolated_atoms(frames)
    assert len(kept) == 2
    assert all(atoms.info.get("config_type") != "IsolatedAtom" for atoms in kept)
    assert energies == {29: -3.75, 79: -2.1}


def test_the_mechanism_can_be_switched_off():
    frames = [_isolated("Cu", -3.75), _bulk(0)]
    kept, energies = split_isolated_atoms(frames, config_type=None)
    assert len(kept) == 2 and energies == {}


def test_a_custom_config_type_is_honoured():
    frames = [_isolated("Cu", -3.75, config_type="atom"), _bulk(0)]
    assert split_isolated_atoms(frames, config_type="atom")[1] == {29: -3.75}
    assert split_isolated_atoms(frames, config_type="IsolatedAtom")[1] == {}


def test_a_multi_atom_frame_marked_isolated_is_kept_not_trusted():
    """It cannot define one element's reference, so it stays a training frame."""
    wrong = _bulk(0)
    wrong.info["config_type"] = "IsolatedAtom"
    kept, energies = split_isolated_atoms([wrong, _bulk(1)])
    assert len(kept) == 2 and energies == {}


def test_an_isolated_frame_without_an_energy_is_dropped():
    bare = Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 20.0, pbc=True)
    bare.info["config_type"] = "IsolatedAtom"
    kept, energies = split_isolated_atoms([bare, _bulk(0)])
    assert len(kept) == 1 and energies == {}


def test_a_measured_reference_overrides_the_composition_fit():
    """The whole point: a real single-atom calculation beats a least-squares guess."""
    frames = [_bulk(seed) for seed in range(6)]
    dataset = AtomsDataset(frames, z_table=ZTable([29]), r_max=4.0)

    fitted = compute_statistics(dataset)
    given = compute_statistics(dataset, isolated_atom_energies={29: -3.75})
    assert float(given.atomic_energies[0]) == pytest.approx(-3.75)
    assert float(fitted.atomic_energies[0]) != pytest.approx(-3.75)


def test_references_for_absent_elements_are_ignored():
    frames = [_bulk(seed) for seed in range(4)]
    dataset = AtomsDataset(frames, z_table=ZTable([29]), r_max=4.0)
    stats = compute_statistics(dataset, isolated_atom_energies={29: -3.75, 79: -2.1})
    assert stats.atomic_energies.shape == (1,)
    assert float(stats.atomic_energies[0]) == pytest.approx(-3.75)


def test_an_isolated_atom_predicts_exactly_its_reference_energy():
    """With E0 taken from the data, the model reproduces it by construction."""
    from sagui.config import ModelConfig
    from sagui.data.atomic_data import collate_graphs, graph_from_atoms
    from sagui.models.registry import build_model

    torch.manual_seed(0)
    model = build_model(
        ModelConfig(type="strictly_local", r_max=4.0, lmax=2, channels=8, num_layers=2,
                    latent_dim=16, scalar_mlp_hidden=[16]),
        atomic_numbers=[29],
        atomic_energies=[-3.75],
    ).eval()
    graph = graph_from_atoms(_isolated("Cu", -3.75), ZTable([29]), 4.0, with_labels=False)
    out = model(collate_graphs([graph]), training=False)
    assert float(out["energy"]) == pytest.approx(-3.75, abs=1e-10)
