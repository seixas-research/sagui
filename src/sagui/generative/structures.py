r"""Turning a (possibly very noisy) ``(types, fractional coords, lattice)``
triple into a graph the network can read.

Two problems appear here that never arise for a potential:

* **degenerate lattices.**  Half way through the forward process the lattice
  is nearly a sample from a standard normal, so it can be almost singular.  A
  cell thinner than the cutoff would generate an astronomical number of
  periodic images, so the lattice is projected onto one with a minimum
  thickness *before* the graph is built.  The projection is applied
  identically during training and sampling, so the network always sees the
  same transformation of its input;
* **unbounded coordination.**  Even a valid but small cell can put hundreds of
  images inside the cutoff.  The neighbour list is therefore truncated to the
  ``max_neighbors`` closest neighbours of each atom, which bounds the memory
  per structure.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list

from ..data.atomic_data import AtomicGraph

__all__ = ["sanitize_lattice", "wrap_fractional", "graph_from_arrays", "lattice_length_scale"]


def wrap_fractional(frac: torch.Tensor) -> torch.Tensor:
    """Fold fractional coordinates back into ``[0, 1)``."""
    return frac - torch.floor(frac)


def lattice_length_scale(cell: np.ndarray, num_atoms: int) -> float:
    """``(V / N)^(1/3)``: the mean interatomic length scale of a cell."""
    volume = abs(float(np.linalg.det(np.asarray(cell, dtype=float))))
    return (volume / max(num_atoms, 1)) ** (1.0 / 3.0)


def sanitize_lattice(
    lattice: torch.Tensor, min_length: float, max_length: float = 200.0
) -> torch.Tensor:
    """Clamp the singular values of ``lattice`` into a usable range.

    The lower bound keeps the cell thicker than a fraction of the cutoff (a
    thinner one would generate an unbounded number of periodic images); the
    upper bound keeps a diverging sample from overflowing the integer binning
    inside the neighbour-list code.  Left-handed cells are flipped so the
    volume stays positive.

    Works on a single ``[3, 3]`` matrix or a batch ``[..., 3, 3]``.
    """
    device, dtype = lattice.device, lattice.dtype
    # The decomposition runs on the CPU in float64: it wants the precision, and
    # Metal has no float64 at all (its SVD would silently fall back to the CPU
    # anyway).  This is a handful of 3x3 matrices, so the round trip is free.
    work = torch.nan_to_num(lattice, nan=0.0, posinf=max_length, neginf=-max_length)
    # Move *then* cast: a combined .to(device=..., dtype=float64) is performed on
    # the source device, and Metal refuses float64 before the transfer happens.
    work = work.to("cpu").to(torch.float64)

    u, s, vh = torch.linalg.svd(work)
    s = s.clamp(float(min_length), float(max_length))
    out = (u * s.unsqueeze(-2)) @ vh
    negative = torch.linalg.det(out) < 0
    if negative.any():
        flipped = out.clone()
        flipped[..., 0, :] = -flipped[..., 0, :]
        out = torch.where(negative[..., None, None], flipped, out)
    return out.to(device=device, dtype=dtype)


def _truncate_neighbors(
    receivers: np.ndarray,
    distances: np.ndarray,
    num_atoms: int,
    max_neighbors: int,
) -> np.ndarray:
    """Indices of the ``max_neighbors`` closest edges of every receiver."""
    order = np.lexsort((distances, receivers))
    sorted_receivers = receivers[order]
    starts = np.searchsorted(sorted_receivers, np.arange(num_atoms))
    rank = np.arange(len(order)) - starts[sorted_receivers]
    return order[rank < max_neighbors]


def graph_from_arrays(
    frac: torch.Tensor,
    lattice: torch.Tensor,
    types: torch.Tensor,
    r_max: float,
    max_neighbors: int = 24,
    min_cell_length: float | None = None,
) -> AtomicGraph:
    """Build a single-structure :class:`AtomicGraph` from raw crystal arrays.

    Parameters
    ----------
    frac:
        ``[N, 3]`` fractional coordinates (wrapped internally).
    lattice:
        ``[3, 3]`` cell, rows are lattice vectors, in Angstrom.
    types:
        ``[N]`` token indices (which may include a ``[MASK]``).
    """
    dtype = frac.dtype
    min_cell_length = 0.5 * r_max if min_cell_length is None else min_cell_length
    cell = sanitize_lattice(lattice, min_cell_length)
    frac = wrap_fractional(frac)
    positions = frac @ cell

    cell_np = cell.detach().cpu().numpy().astype(float)
    positions_np = positions.detach().cpu().numpy().astype(float)
    receivers, senders, unit_shifts, distances = primitive_neighbor_list(
        "ijSd", (True, True, True), cell_np, positions_np, float(r_max), self_interaction=False
    )
    if len(receivers) > 0:
        keep = _truncate_neighbors(receivers, distances, len(frac), max_neighbors)
        receivers, senders, unit_shifts = receivers[keep], senders[keep], unit_shifts[keep]

    shifts = np.asarray(unit_shifts, dtype=float) @ cell_np
    n_atoms = int(frac.shape[0])
    return AtomicGraph(
        positions=positions,
        species=types.to(torch.long),
        edge_index=torch.as_tensor(np.stack([receivers, senders]).astype(np.int64)),
        shifts=torch.as_tensor(shifts, dtype=dtype),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        num_atoms=torch.tensor([n_atoms], dtype=torch.long),
        cell=cell.to(dtype).unsqueeze(0),
    )
