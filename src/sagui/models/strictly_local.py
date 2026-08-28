r"""Strictly local, Allegro-style architecture (no message passing).

Message-passing potentials grow their receptive field with depth: after ``T``
layers an atom feels everything within ``T * r_max``.  That is expensive to
parallelise, because a spatial domain must exchange a halo of width
``T * r_max`` with its neighbours.  Allegro (Musaelian et al., *Nat. Commun.*
2023) removes the growth entirely: **every quantity attached to the edge
(i, j) depends only on atoms inside the cutoff sphere of the central atom i**,
no matter how many layers are stacked.  The result is a many-body potential
whose parallel decomposition needs a single-cutoff halo.

SAGUI keeps that invariant explicitly:

* a **pair embedding** ``x_ij`` from the two species and the radial basis;
* an **environment descriptor** ``e_i = 1/sqrt(N) sum_k u(r_ik) g(x_ik)``, an
  invariant summary of the neighbours of *i* only;
* ``L`` layers that update the pair tensor ``V_ij`` and the pair scalars
  ``x_ij`` using **only** ``(x_ij, V_ij, e_i)``.  Crucially the update never
  reads ``e_j``: that single restriction is what keeps the receptive field at
  exactly one cutoff for any depth;
* an energy that is a sum over *pairs*,

  .. math:: E_i = \sum_{j \in \mathcal{N}(i)} u(r_{ij})\, \phi(x_{ij}^{(L)}) ,

  with the cutoff envelope ``u`` multiplying the readout so that energies and
  forces go smoothly to zero as a neighbour leaves the sphere.

The equivariant track is updated by the same weighted tensor product used by
the message-passing model, with the weights produced by an MLP of the
invariants -- so nonlinearity enters through the scalars while the ``l > 0``
components stay exactly equivariant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from ..config import ModelConfig
from ..data.atomic_data import AtomicGraph
from ..nn.blocks import (
    EquivariantLinear,
    WeightedTensorProduct,
    invariant_features,
    num_invariants,
    one_hot_species,
)
from ..nn.o3 import SphericalLayout, spherical_harmonics
from ..nn.radial import MLP, BesselBasis, PolynomialCutoff
from ..nn.scatter import scatter_sum
from .base import InteratomicPotential
from .registry import register_model

__all__ = ["StrictlyLocalModel", "LocalLayer"]


class LocalLayer(nn.Module):
    """One Allegro-style layer acting on a single edge (plus ``e_i``)."""

    def __init__(
        self,
        layout: SphericalLayout,
        sh_lmax: int,
        latent_dim: int,
        env_dim: int,
        hidden: Sequence[int],
    ) -> None:
        super().__init__()
        self.layout = layout
        self.tensor_product = WeightedTensorProduct(layout.lmax, sh_lmax, layout.lmax)
        self.weight_mlp = MLP(
            latent_dim + env_dim,
            hidden,
            self.tensor_product.num_paths * layout.channels,
        )
        self.linear = EquivariantLinear(layout.lmax, layout.channels)
        self.scalar_mlp = MLP(
            latent_dim + num_invariants(layout) + env_dim,
            hidden,
            latent_dim,
            final_bias=True,
        )

    def forward(
        self, x: torch.Tensor, v: torch.Tensor, sh: torch.Tensor, env: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weight_mlp(torch.cat([x, env], dim=-1)).view(
            -1, self.tensor_product.num_paths, self.layout.channels
        )
        # Residual updates are averaged (1/sqrt(2)) so that the activation
        # scale does not drift as layers are stacked.
        v = (v + self.linear(self.tensor_product(v, sh, weights))) / math.sqrt(2.0)
        invariants = invariant_features(v, self.layout)
        x = (x + self.scalar_mlp(torch.cat([x, invariants, env], dim=-1))) / math.sqrt(2.0)
        return x, v


@register_model("strictly_local")
class StrictlyLocalModel(InteratomicPotential):
    """Many-body potential whose receptive field is exactly one cutoff."""

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
        latent_dim = int(config.latent_dim)
        env_dim = latent_dim

        self.radial_basis = BesselBasis(config.r_max, config.num_radial_basis)
        self.cutoff = PolynomialCutoff(config.r_max, config.cutoff_p)

        # Two-body embedding: (Z_i, Z_j, b(r_ij)) -> invariant pair features.
        self.pair_embedding = MLP(
            2 * self.num_species + config.num_radial_basis,
            config.scalar_mlp_hidden,
            latent_dim,
            final_bias=True,
        )
        # Environment of the central atom, built from two-body terms only.
        self.env_mlp = MLP(latent_dim, config.scalar_mlp_hidden, env_dim, final_bias=True)
        # Initial equivariant edge tensor: channel/degree weights times Y(r_ij).
        self.embed_weights = nn.Linear(latent_dim, config.channels * (self.layout.lmax + 1))

        self.layers = nn.ModuleList(
            [
                LocalLayer(
                    self.layout,
                    self.sh_lmax,
                    latent_dim,
                    env_dim,
                    config.scalar_mlp_hidden,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.readout = MLP(latent_dim, [config.readout_hidden], 1, final_bias=False)

    def _initial_tensor(self, x: torch.Tensor, sh: torch.Tensor) -> torch.Tensor:
        """``[E, latent] , [E, D_sh] -> [E, C, D]`` equivariant edge features."""
        weights = self.embed_weights(x).view(-1, self.layout.channels, self.layout.lmax + 1)
        blocks = [
            weights[..., l : l + 1] * sh[:, None, SphericalLayout.block(l)]
            for l in self.layout.ls
        ]
        return torch.cat(blocks, dim=-1)

    def node_energies(
        self, data: AtomicGraph, vectors: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        dtype = vectors.dtype
        one_hot = one_hot_species(data.species, self.num_species, dtype)
        envelope = self.cutoff(lengths)
        radial = self.radial_basis(lengths) * envelope
        sh = spherical_harmonics(self.sh_lmax, vectors)
        norm = torch.sqrt(self.avg_num_neighbors.to(dtype))

        x = self.pair_embedding(
            torch.cat([one_hot[data.receivers], one_hot[data.senders], radial], dim=-1)
        )
        # Invariant summary of N(i); damped by the envelope to stay smooth.
        env_nodes = scatter_sum(
            self.env_mlp(x) * envelope, data.receivers, data.num_nodes
        ) / norm

        v = self._initial_tensor(x, sh)
        for layer in self.layers:
            x, v = layer(x, v, sh, env_nodes[data.receivers])

        pair_energy = (self.readout(x) * envelope).squeeze(-1)
        return scatter_sum(pair_energy, data.receivers, data.num_nodes)
