"""The graph representation consumed by every SAGUI model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress

from .neighborlist import build_neighbor_list
from .ztable import ZTable

__all__ = ["AtomicGraph", "collate_graphs", "graph_from_atoms", "extract_labels"]

#: Keys searched, in order, when looking for reference labels in an ``Atoms``.
ENERGY_KEYS = ("energy", "REF_energy", "TotEnergy", "free_energy")
FORCES_KEYS = ("forces", "REF_forces", "force")
STRESS_KEYS = ("stress", "REF_stress")


@dataclass
class AtomicGraph:
    """A structure (or a batch of them) as a graph of atoms.

    A batch is *not* a separate type: :func:`collate_graphs` concatenates node
    and edge axes and records the owning structure of each atom in ``batch``.
    Because interactions are strictly local, a batch of disconnected molecules
    behaves exactly like one big structure, which keeps the models free of any
    batching logic.
    """

    positions: torch.Tensor  # [N, 3]
    species: torch.Tensor  # [N]   index into the ZTable
    edge_index: torch.Tensor  # [2, E] row 0 = receiver i, row 1 = sender j
    shifts: torch.Tensor  # [E, 3] cartesian periodic offsets
    batch: torch.Tensor  # [N]   structure index of each atom
    num_atoms: torch.Tensor  # [G]
    cell: torch.Tensor  # [G, 3, 3]
    #: ``[E, 3]`` integer lattice offsets, ``shifts = unit_shifts @ cell``.
    #: Optional because the generative path builds graphs without them; they
    #: are what lets :meth:`InteratomicPotential.forward` rebuild the shifts
    #: from a strained cell and so differentiate the energy with respect to it.
    unit_shifts: torch.Tensor | None = None  # [E, 3]
    energy: torch.Tensor | None = None  # [G]
    forces: torch.Tensor | None = None  # [N, 3]
    stress: torch.Tensor | None = None  # [G, 3, 3]

    @property
    def num_graphs(self) -> int:
        return int(self.num_atoms.shape[0])

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def receivers(self) -> torch.Tensor:
        return self.edge_index[0]

    @property
    def senders(self) -> torch.Tensor:
        return self.edge_index[1]

    def to(self, device: torch.device | str) -> AtomicGraph:
        moved: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            moved[f.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return AtomicGraph(**moved)

    def clone(self) -> AtomicGraph:
        moved = {
            f.name: (v.clone() if isinstance(v := getattr(self, f.name), torch.Tensor) else v)
            for f in fields(self)
        }
        return AtomicGraph(**moved)


def extract_labels(
    atoms: Atoms,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """Pull reference energy, forces and stress out of an ``Atoms`` object.

    Looks first in ``atoms.info`` / ``atoms.arrays`` (where the extended-XYZ
    reader puts extra columns), then falls back to the attached calculator,
    which is what ``ase.io.read`` populates for standard ``energy``/``forces``
    entries.
    """
    energy: float | None = None
    keys = (energy_key,) if energy_key else ENERGY_KEYS
    for key in keys:
        if key in atoms.info:
            energy = float(np.asarray(atoms.info[key]).reshape(()))
            break
    if energy is None and atoms.calc is not None and not energy_key:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:  # noqa: BLE001 - a missing label is not an error
            energy = None

    forces: np.ndarray | None = None
    keys = (forces_key,) if forces_key else FORCES_KEYS
    for key in keys:
        if key in atoms.arrays:
            forces = np.asarray(atoms.arrays[key], dtype=float)
            break
    if forces is None and atoms.calc is not None and not forces_key:
        try:
            forces = np.asarray(atoms.get_forces(), dtype=float)
        except Exception:  # noqa: BLE001
            forces = None

    stress: np.ndarray | None = None
    keys = (stress_key,) if stress_key else STRESS_KEYS
    for key in keys:
        if key in atoms.info:
            stress = np.asarray(atoms.info[key], dtype=float)
            break
    if stress is None and atoms.calc is not None and not stress_key:
        try:
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
        except Exception:  # noqa: BLE001
            stress = None
    if stress is not None:
        # Reference stresses come in either Voigt (6) or full (3, 3) form.
        stress = voigt_6_to_full_3x3_stress(stress) if stress.size == 6 else stress.reshape(3, 3)
    return energy, forces, stress


def graph_from_atoms(
    atoms: Atoms,
    z_table: ZTable,
    r_max: float,
    with_labels: bool = True,
    energy_key: str | None = None,
    forces_key: str | None = None,
    stress_key: str | None = None,
    dtype: torch.dtype | None = None,
) -> AtomicGraph:
    """Convert an ASE ``Atoms`` object into a single-structure :class:`AtomicGraph`."""
    dtype = dtype or torch.get_default_dtype()
    edge_index, shifts, unit_shifts = build_neighbor_list(atoms, r_max)

    energy_value, forces_value, stress_value = (None, None, None)
    if with_labels:
        energy_value, forces_value, stress_value = extract_labels(
            atoms, energy_key, forces_key, stress_key
        )

    n_atoms = len(atoms)
    return AtomicGraph(
        positions=torch.as_tensor(atoms.get_positions(), dtype=dtype),
        species=torch.as_tensor(z_table.indices(atoms.get_atomic_numbers())),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        shifts=torch.as_tensor(shifts, dtype=dtype),
        unit_shifts=torch.as_tensor(unit_shifts, dtype=dtype),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        num_atoms=torch.tensor([n_atoms], dtype=torch.long),
        cell=torch.as_tensor(atoms.get_cell().array, dtype=dtype).unsqueeze(0),
        energy=None if energy_value is None else torch.tensor([energy_value], dtype=dtype),
        forces=None if forces_value is None else torch.as_tensor(forces_value, dtype=dtype),
        stress=(
            None
            if stress_value is None
            else torch.as_tensor(stress_value, dtype=dtype).unsqueeze(0)
        ),
    )


def collate_graphs(graphs: Sequence[AtomicGraph]) -> AtomicGraph:
    """Merge single-structure graphs into one batched :class:`AtomicGraph`.

    Labels are dropped for the whole batch if *any* member is missing them, so
    that a loss term is either fully defined or absent.
    """
    if not graphs:
        raise ValueError("cannot collate an empty list of graphs")

    positions, species, edges, shifts, batch, num_atoms, cells = [], [], [], [], [], [], []
    unit_shifts: list[torch.Tensor] = []
    node_offset = 0
    for index, graph in enumerate(graphs):
        n = graph.num_nodes
        positions.append(graph.positions)
        species.append(graph.species)
        edges.append(graph.edge_index + node_offset)
        shifts.append(graph.shifts)
        if graph.unit_shifts is not None:
            unit_shifts.append(graph.unit_shifts)
        batch.append(torch.full((n,), index, dtype=torch.long))
        num_atoms.append(graph.num_atoms)
        cells.append(graph.cell)
        node_offset += n

    has_energy = all(g.energy is not None for g in graphs)
    has_forces = all(g.forces is not None for g in graphs)
    has_stress = all(g.stress is not None for g in graphs)
    has_unit_shifts = len(unit_shifts) == len(graphs)
    return AtomicGraph(
        positions=torch.cat(positions, dim=0),
        species=torch.cat(species, dim=0),
        edge_index=torch.cat(edges, dim=1),
        shifts=torch.cat(shifts, dim=0),
        unit_shifts=torch.cat(unit_shifts, dim=0) if has_unit_shifts else None,
        batch=torch.cat(batch, dim=0),
        num_atoms=torch.cat(num_atoms, dim=0),
        cell=torch.cat(cells, dim=0),
        energy=torch.cat([g.energy for g in graphs]) if has_energy else None,
        forces=torch.cat([g.forces for g in graphs], dim=0) if has_forces else None,
        stress=torch.cat([g.stress for g in graphs], dim=0) if has_stress else None,
    )
