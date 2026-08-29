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
import warnings
from collections.abc import Sequence

import torch
from torch import nn

from ..data.atomic_data import AtomicGraph
from ..nn.scatter import scatter_sum

__all__ = ["InteratomicPotential"]


class InteratomicPotential(nn.Module):
    """Base class: subclasses only implement :meth:`node_energies`."""

    #: Names of the ``nn.ModuleList`` attributes holding the repeated layers
    #: that :meth:`compile_layers` hands to ``torch.compile``.
    COMPILABLE_LAYERS: tuple[str, ...] = ()

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

    # -------------------------------------------------------------- compile
    def compile_layers(self, dynamic: bool = True, **compile_kwargs) -> int:
        """Hand the repeated layers to ``torch.compile``; return how many.

        Only the layers are compiled.  :meth:`forward` stays in eager mode
        because it calls :func:`torch.autograd.grad`, and the graph builder
        would break on the :class:`AtomicGraph` dataclass anyway; the layers
        take tensors in and return tensors, which is what Dynamo handles well.

        What is compiled is each layer's ``forward`` *method*, not the module.
        Wrapping the module would nest it under ``_orig_mod`` and rename every
        key in :meth:`state_dict`, silently breaking checkpoints.

        ``dynamic=True`` is the default because the edge count changes at every
        molecular-dynamics step; with static shapes each new count triggers a
        full recompilation.

        Notes
        -----
        Compilation pays off only once the tensor product is the fused
        ``"gemm"`` kind: applied to the per-path loop, ``torch.compile`` has
        been measured to make the model several times *slower*.  A warning is
        issued in that case.

        The compiled backward is the fragile part of this path.  Validate it at
        your production system size on your production hardware before relying
        on it -- keep the model usable without it, which is why this is an
        explicit call and not something ``__init__`` does.
        """
        if not self.COMPILABLE_LAYERS:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare COMPILABLE_LAYERS"
            )
        kinds = {
            type(module.tensor_product).__name__
            for name in self.COMPILABLE_LAYERS
            for module in getattr(self, name)
            if hasattr(module, "tensor_product")
        }
        if "WeightedTensorProduct" in kinds:
            warnings.warn(
                "compile_layers() with the 'loop' tensor product is typically much "
                "slower than eager; set model.tensor_product='gemm' first.",
                RuntimeWarning,
                stacklevel=2,
            )
        compiled = 0
        for name in self.COMPILABLE_LAYERS:
            for module in getattr(self, name):
                module.forward = torch.compile(module.forward, dynamic=dynamic, **compile_kwargs)
                compiled += 1
        return compiled

    # ----------------------------------------------------------------- misc
    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"r_max={self.r_max.item():.2f}, species={self.num_species}, "
            f"avg_num_neighbors={self.avg_num_neighbors.item():.2f}"
        )
