r"""MACE-style message-passing architecture.

Structure of one layer, following Batatia et al., *MACE* (NeurIPS 2022):

1. **Two-body basis.**  Each edge contributes the tensor product of the
   neighbour's features with the spherical harmonics of the bond direction,
   weighted by a learned radial function,

   .. math::
       A_i^{(l_3)} = \frac{1}{\sqrt{\bar N}} \sum_{j \in \mathcal{N}(i)}
           \big( R(r_{ij}) \, h_j \otimes Y(\hat r_{ij}) \big)^{(l_3)} .

   This is the "one-particle basis" of the Atomic Cluster Expansion; the sum
   over neighbours is what makes it permutation invariant and linear-scaling.

2. **Many-body expansion.**  Products of :math:`A_i` with itself raise the
   correlation order,

   .. math::
       B^{(\nu)}_i = \underbrace{A_i \otimes \cdots \otimes A_i}_{\nu},

   each product being projected back onto the irreducible components with the
   Clebsch-Gordan tensors.  A message that mixes ``nu = 1 .. correlation``
   therefore describes genuine ``(nu + 1)``-body interactions *within a single
   layer* -- the property that lets MACE work with only one or two layers, and
   keeps the receptive field small.

3. **Update.**  A linear map plus an element-dependent self-interaction
   (residual) gives the new node features; a readout of the invariant part
   accumulates the atomic energy.

Deviation from the published model
----------------------------------
The symmetric contraction here is a nested *channel-wise* product with learned
path weights, not the full symmetrised basis with the generalised
Clebsch-Gordan coefficients :math:`U^{L,\eta}` of the paper.  It spans the same
correlation orders with a smaller (diagonal in the channel index) coupling; see
``sagui_context.md`` for the trade-off and the roadmap entry that lifts it.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from ..config import ModelConfig
from ..data.atomic_data import AtomicGraph
from ..nn.blocks import (
    EquivariantLinear,
    SelfTensorProduct,
    SpeciesLinear,
    build_weighted_tensor_product,
    embed_scalars,
    one_hot_species,
    scalars,
)
from ..nn.o3 import SphericalLayout, spherical_harmonics
from ..nn.radial import MLP, BesselBasis, PolynomialCutoff
from ..nn.scatter import scatter_sum
from .base import InteratomicPotential
from .registry import register_model

__all__ = ["MACEModel", "SymmetricContraction", "InteractionBlock"]


class SymmetricContraction(nn.Module):
    """Many-body message from nested products of the one-particle basis ``A``."""

    def __init__(self, layout: SphericalLayout, correlation: int) -> None:
        super().__init__()
        if correlation < 1:
            raise ValueError(f"correlation must be >= 1, got {correlation}")
        self.correlation = int(correlation)
        self.mixings = nn.ModuleList(
            [EquivariantLinear(layout.lmax, layout.channels) for _ in range(self.correlation)]
        )
        self.products = nn.ModuleList(
            [
                SelfTensorProduct(layout.lmax, layout.lmax, layout.lmax, layout.channels)
                for _ in range(self.correlation - 1)
            ]
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        out = self.mixings[0](a)
        power = a
        for order, product in enumerate(self.products, start=1):
            power = product(power, a)
            out = out + self.mixings[order](power)
        return out


class InteractionBlock(nn.Module):
    """One MACE layer: gather -> many-body -> update."""

    def __init__(
        self,
        layout: SphericalLayout,
        sh_lmax: int,
        num_radial_basis: int,
        radial_mlp_hidden: Sequence[int],
        num_species: int,
        correlation: int,
        tensor_product: str = "gemm",
    ) -> None:
        super().__init__()
        self.layout = layout
        self.linear_up = EquivariantLinear(layout.lmax, layout.channels)
        self.tensor_product = build_weighted_tensor_product(
            tensor_product, layout.lmax, sh_lmax, layout.lmax, channels=layout.channels
        )
        self.radial_mlp = MLP(
            num_radial_basis,
            radial_mlp_hidden,
            self.tensor_product.num_paths * layout.channels,
        )
        self.linear_gather = EquivariantLinear(layout.lmax, layout.channels)
        self.contraction = SymmetricContraction(layout, correlation)
        self.linear_out = EquivariantLinear(layout.lmax, layout.channels)
        self.self_interaction = SpeciesLinear(num_species, layout.lmax, layout.channels)

    def forward(
        self,
        h: torch.Tensor,
        sh: torch.Tensor,
        radial: torch.Tensor,
        data: AtomicGraph,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.radial_mlp(radial).view(
            -1, self.tensor_product.num_paths, self.layout.channels
        )
        messages = self.tensor_product(self.linear_up(h)[data.senders], sh, weights)
        a = scatter_sum(messages, data.receivers, data.num_nodes) / norm
        a = self.linear_gather(a)
        b = self.contraction(a)
        return self.linear_out(b) + self.self_interaction(h, data.species)


@register_model("mace")
class MACEModel(InteratomicPotential):
    """Equivariant message-passing potential with a many-body basis."""

    COMPILABLE_LAYERS = ("interactions",)

    def __init__(
        self,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        atomic_energies=None,
        energy_scale: float = 1.0,
        avg_num_neighbors: float = 1.0,
    ) -> None:
        super().__init__(
            r_max=config.r_max,
            atomic_numbers=atomic_numbers,
            atomic_energies=atomic_energies,
            energy_scale=energy_scale,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.config = config
        self.layout = SphericalLayout(config.lmax, config.channels)
        self.sh_lmax = config.spherical_lmax

        self.species_embedding = nn.Linear(self.num_species, config.channels, bias=False)
        self.radial_basis = BesselBasis(config.r_max, config.num_radial_basis)
        self.cutoff = PolynomialCutoff(config.r_max, config.cutoff_p)

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    self.layout,
                    self.sh_lmax,
                    config.num_radial_basis,
                    config.radial_mlp_hidden,
                    self.num_species,
                    config.correlation,
                    tensor_product=config.tensor_product,
                )
                for _ in range(config.num_layers)
            ]
        )
        # Every layer reads out an energy contribution; the last one gets a
        # nonlinear head, as in MACE, since it no longer feeds another layer.
        readouts: list[nn.Module] = [
            nn.Linear(config.channels, 1, bias=False) for _ in range(config.num_layers - 1)
        ]
        readouts.append(MLP(config.channels, [config.readout_hidden], 1, final_bias=False))
        self.readouts = nn.ModuleList(readouts)

    def node_energies(
        self, data: AtomicGraph, vectors: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        dtype = vectors.dtype
        one_hot = one_hot_species(data.species, self.num_species, dtype)
        h = embed_scalars(self.species_embedding(one_hot), self.layout)

        sh = spherical_harmonics(self.sh_lmax, vectors)
        radial = self.radial_basis(lengths) * self.cutoff(lengths)
        norm = torch.sqrt(self.avg_num_neighbors.to(dtype))

        energy = vectors.new_zeros(data.num_nodes)
        for interaction, readout in zip(self.interactions, self.readouts, strict=True):
            h = interaction(h, sh, radial, data, norm)
            energy = energy + readout(scalars(h)).squeeze(-1)
        return energy
