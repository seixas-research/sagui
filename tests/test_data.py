"""Neighbour lists, graph construction and batching."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from sagui.data import (
    AtomsDataset,
    ZTable,
    build_neighbor_list,
    collate_graphs,
    graph_from_atoms,
    random_split,
)


def _brute_force_pairs(atoms: Atoms, cutoff: float) -> set[tuple[int, int]]:
    """Reference neighbour list for a non-periodic system."""
    positions = atoms.get_positions()
    pairs = set()
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i != j and np.linalg.norm(positions[j] - positions[i]) < cutoff:
                pairs.add((i, j))
    return pairs


def test_neighbor_list_matches_brute_force(cluster):
    edge_index, shifts, _ = build_neighbor_list(cluster, 3.5)
    found = {(int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1], strict=True)}
    assert found == _brute_force_pairs(cluster, 3.5)
    assert np.allclose(shifts, 0.0), "an open system has no periodic images"


def test_neighbor_list_is_symmetric(cluster):
    edge_index, _, _ = build_neighbor_list(cluster, 4.0)
    pairs = {(int(i), int(j)) for i, j in zip(edge_index[0], edge_index[1], strict=True)}
    assert all((j, i) in pairs for i, j in pairs)


def test_edge_vectors_respect_periodicity(crystal):
    """Every stored shift must reproduce a distance below the cutoff."""
    cutoff = 3.2
    edge_index, shifts, _ = build_neighbor_list(crystal, cutoff)
    positions = crystal.get_positions()
    vectors = positions[edge_index[1]] + shifts - positions[edge_index[0]]
    lengths = np.linalg.norm(vectors, axis=1)
    assert len(lengths) > 0
    assert lengths.max() < cutoff
    assert lengths.min() > 1e-6, "no self edges"


def test_periodic_neighbours_are_translation_invariant(crystal):
    """Sliding a crystal through its own cell cannot change its connectivity."""
    reference = build_neighbor_list(crystal, 3.2)[0].shape[1]
    moved = crystal.copy()
    moved.positions += moved.get_cell().array[0] * 0.37
    assert build_neighbor_list(moved, 3.2)[0].shape[1] == reference


def test_open_system_without_a_cell():
    """A molecule with a zero cell must still get a neighbour list."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
    assert atoms.get_cell().volume == 0.0
    edge_index, _, _ = build_neighbor_list(atoms, 2.0)
    assert edge_index.shape[1] == 2


def test_collate_offsets_edges_and_labels(labelled_frames):
    z_table = ZTable.from_atoms(labelled_frames)
    graphs = [graph_from_atoms(a, z_table, 5.0) for a in labelled_frames[:3]]
    batch = collate_graphs(graphs)

    assert batch.num_graphs == 3
    assert batch.num_nodes == sum(g.num_nodes for g in graphs)
    assert batch.num_edges == sum(g.num_edges for g in graphs)
    assert batch.energy.shape == (3,)
    assert batch.forces.shape == (batch.num_nodes, 3)
    # Edges never cross structure boundaries.
    assert torch.equal(batch.batch[batch.senders], batch.batch[batch.receivers])
    assert int(batch.edge_index.max()) < batch.num_nodes


def test_collate_drops_partial_labels(labelled_frames):
    z_table = ZTable.from_atoms(labelled_frames)
    labelled = graph_from_atoms(labelled_frames[0], z_table, 5.0)
    unlabelled = graph_from_atoms(labelled_frames[1], z_table, 5.0, with_labels=False)
    batch = collate_graphs([labelled, unlabelled])
    assert batch.energy is None and batch.forces is None


def test_labels_are_read_from_extxyz(labelled_frames):
    z_table = ZTable.from_atoms(labelled_frames)
    atoms = labelled_frames[0]
    graph = graph_from_atoms(atoms, z_table, 5.0)
    assert graph.energy is not None
    assert np.isclose(float(graph.energy), atoms.get_potential_energy())
    assert np.allclose(graph.forces.numpy(), atoms.get_forces())


def test_ztable_orders_species_and_reports_unknowns():
    z_table = ZTable([8, 1, 6])
    assert z_table.zs == (1, 6, 8)
    assert z_table.symbols == ("H", "C", "O")
    assert z_table.index(6) == 1
    with pytest.raises(KeyError, match="unknown to this model"):
        z_table.index(79)


def test_dataset_and_split(labelled_frames):
    z_table = ZTable.from_atoms(labelled_frames)
    dataset = AtomsDataset(labelled_frames, z_table, r_max=5.0)
    assert len(dataset) == len(labelled_frames)
    assert dataset[0].positions.dtype == torch.float64

    train, valid = random_split(labelled_frames, 0.25, seed=0)
    assert len(train) + len(valid) == len(labelled_frames)
    assert len(valid) == 3
    train_ids = {id(a) for a in train}
    assert not train_ids & {id(a) for a in valid}, "no leakage between the splits"
