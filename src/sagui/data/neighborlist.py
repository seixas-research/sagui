"""Neighbour-list construction, delegating the heavy lifting to a C++ backend.

The graph edges of an interatomic potential are the pairs closer than the
cutoff.  Under periodic boundary conditions a pair can appear several times
with different lattice offsets, so every edge carries the Cartesian shift
``S @ cell`` that must be added to the neighbour position.  Storing shifts
rather than wrapped coordinates keeps the edge vectors a differentiable
function of the *unwrapped* positions, which is what makes autograd forces
correct for periodic systems.

Two backends are available.  ``vesin`` (optional, ``pip install vesin``) is a
C++ implementation measured here at 36x faster than ASE on 64 atoms and 67x on
1728, and its ``(i, j, S)`` convention is identical to ours, so the edge sets
agree exactly.  ASE's pure-Python ``primitive_neighbor_list`` remains the
fallback and handles the mixed-periodicity case that ``vesin`` cannot express.
At the model speeds measured in ``sagui_performance_optimization.md`` the list
is a few per cent of an MD step with ASE and negligible with ``vesin``; the gap
widens with system size.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list

try:  # optional acceleration; the ASE path below is always available
    from vesin import NeighborList as _VesinNeighborList
except ImportError:  # pragma: no cover - exercised only without vesin installed
    _VesinNeighborList = None

__all__ = ["build_neighbor_list", "has_fast_neighbor_list"]


def has_fast_neighbor_list() -> bool:
    """Whether the ``vesin`` backend is importable."""
    return _VesinNeighborList is not None


def _sanitise_cell(positions: np.ndarray, cell: np.ndarray, pbc: tuple[bool, ...], cutoff: float):
    """Give ASE's binning algorithm a non-degenerate cell for open directions.

    Molecules usually carry a zero cell.  Any bounding box larger than the atom
    extent plus the cutoff yields identical neighbours in the non-periodic
    directions, so we synthesise one instead of failing.
    """
    cell = np.array(cell, dtype=float).reshape(3, 3)
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        extent = np.zeros(3)
        origin = np.zeros(3)
    else:
        extent = positions.max(axis=0) - positions.min(axis=0)
        origin = positions.min(axis=0)

    shift = np.zeros(3)
    for axis in range(3):
        if pbc[axis]:
            continue
        length = extent[axis] + 2.0 * cutoff + 1.0
        if np.linalg.norm(cell[axis]) < 1e-8:
            cell[axis] = 0.0
            cell[axis, axis] = length
            # Move the atoms inside the synthetic box along this direction; a
            # rigid translation leaves every interatomic vector untouched.
            shift[axis] = cutoff + 1.0 - origin[axis] if len(positions) else 0.0
    return cell, shift


def build_neighbor_list(
    atoms: Atoms, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(edge_index, shifts, unit_shifts)`` for ``atoms``.

    Parameters
    ----------
    atoms:
        Structure to analyse.
    cutoff:
        Interaction radius ``r_max`` in Angstrom.

    Returns
    -------
    edge_index:
        ``[2, E]`` integer array; row 0 is the *receiver* (central atom ``i``),
        row 1 the *sender* (neighbour ``j``).
    shifts:
        ``[E, 3]`` Cartesian offsets, so that the edge vector is
        ``positions[j] + shift - positions[i]``.
    unit_shifts:
        ``[E, 3]`` integer lattice offsets (kept for future stress support).
    """
    pbc = tuple(bool(p) for p in atoms.get_pbc())
    positions = atoms.get_positions()
    cell, translation = _sanitise_cell(positions, atoms.get_cell().array, pbc, cutoff)

    # vesin covers the fully periodic and fully open cases, which is everything
    # except slabs and wires; those keep the ASE path, whose synthetic box for
    # the open directions vesin has no way to express.
    if _VesinNeighborList is not None and (all(pbc) or not any(pbc)):
        periodic = all(pbc)
        i, j, unit_shifts = _VesinNeighborList(cutoff=cutoff, full_list=True).compute(
            points=positions + translation,
            box=cell if periodic else np.zeros((3, 3)),
            periodic=periodic,
            quantities="ijS",
        )
    else:
        i, j, unit_shifts = primitive_neighbor_list(
            "ijS",
            pbc,
            cell,
            positions + translation,
            cutoff,
            self_interaction=False,
            use_scaled_positions=False,
        )
    edge_index = np.stack([i, j]).astype(np.int64)
    unit_shifts = np.asarray(unit_shifts, dtype=np.int64)
    shifts = unit_shifts.astype(float) @ cell
    return edge_index, shifts, unit_shifts
