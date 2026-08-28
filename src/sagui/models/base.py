r"""Common machinery for every interatomic potential.

The base class owns everything that is *physics* rather than architecture:

* the edge vectors, built from unwrapped positions plus periodic shifts, so
  that autograd sees the true dependence of the geometry on the coordinates;
* the decomposition of the total energy into atomic contributions,

  .. math:: E = \sum_i \big[ s \cdot \varepsilon_i(\{r\}) + E^{(0)}_{Z_i} \big],

  where :math:`E^{(0)}_Z` are per-element reference energies (isolated-atom or
  least-squares fitted) and ``s`` a global scale, typically the RMS force of
  the training set.  The network therefore only has to learn a dimensionless,
  zero-centred quantity;
* the forces as exact analytic derivatives,

  .. math:: F_{i\alpha} = -\frac{\partial E}{\partial r_{i\alpha}},

  obtained with ``torch.autograd.grad``.  Differentiating the energy (rather
  than predicting a vector field directly) guarantees the forces are curl-free
  and exactly consistent with the energy, which is what makes stable molecular
  dynamics possible.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
from torch import nn

from ..data.atomic_data import AtomicGraph
from ..nn.scatter import scatter_sum

__all__ = ["InteratomicPotential"]


class InteratomicPotential(nn.Module):
    """Base class: subclasses only implement :meth:`node_energies`."""

    def __init__(
        self,
        r_max: float,
        atomic_numbers: Sequence[int],
        atomic_energies: Sequence[float] | torch.Tensor | None = None,
        energy_scale: float = 1.0,
        avg_num_neighbors: float = 1.0,
    ) -> None:
        super().__init__()
        atomic_numbers = tuple(int(z) for z in atomic_numbers)
        self.num_species = len(atomic_numbers)
        if atomic_energies is None:
            atomic_energies = torch.zeros(self.num_species)
        atomic_energies = torch.as_tensor(atomic_energies, dtype=torch.get_default_dtype())
        if atomic_energies.shape != (self.num_species,):
            raise ValueError(
                f"expected {self.num_species} atomic energies, got {tuple(atomic_energies.shape)}"
            )

        self.register_buffer("r_max", torch.tensor(float(r_max)))
        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.long))
        self.register_buffer("atomic_energies", atomic_energies)
        self.register_buffer("energy_scale", torch.tensor(float(energy_scale)))
        self.register_buffer("avg_num_neighbors", torch.tensor(float(avg_num_neighbors)))

    # ------------------------------------------------------------ interface
    def node_energies(
        self, data: AtomicGraph, vectors: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Dimensionless per-atom energy ``[N]``; implemented by subclasses.

        ``vectors`` is ``[E, 3]`` and ``lengths`` ``[E, 1]``, both already
        differentiable functions of ``data.positions``.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    @staticmethod
    def edge_vectors(
        data: AtomicGraph, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r""":math:`\vec r_{ij} = \vec r_j + \vec S_{ij} - \vec r_i` and its norm."""
        vectors = positions[data.senders] - positions[data.receivers] + data.shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        return vectors, lengths

    # -------------------------------------------------------------- forward
    def forward(
        self,
        data: AtomicGraph,
        compute_forces: bool = True,
        training: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict energies (and forces) for a batched graph.

        Returns a dict with ``energy`` ``[G]``, ``node_energy`` ``[N]`` and,
        when requested, ``forces`` ``[N, 3]``.
        """
        create_graph = self.training if training is None else bool(training)
        # Force evaluation needs a graph even inside ``torch.no_grad()``.
        context = torch.enable_grad() if compute_forces else contextlib.nullcontext()
        with context:
            positions = data.positions
            if compute_forces and not positions.requires_grad:
                positions.requires_grad_(True)

            vectors, lengths = self.edge_vectors(data, positions)
            raw = self.node_energies(data, vectors, lengths)
            node_energy = self.energy_scale * raw + self.atomic_energies[data.species]
            energy = scatter_sum(node_energy, data.batch, data.num_graphs)

            out: dict[str, torch.Tensor] = {"energy": energy, "node_energy": node_energy}
            if compute_forces:
                (gradient,) = torch.autograd.grad(
                    [energy.sum()],
                    [positions],
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    allow_unused=True,
                )
                out["forces"] = (
                    torch.zeros_like(positions) if gradient is None else -gradient
                )
        if not create_graph:
            # Nothing downstream can backpropagate through these, and keeping
            # the graph alive would pin the whole batch in memory.
            out = {key: value.detach() for key, value in out.items()}
        return out

    # ----------------------------------------------------------------- misc
    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"r_max={self.r_max.item():.2f}, species={self.num_species}, "
            f"avg_num_neighbors={self.avg_num_neighbors.item():.2f}"
        )
