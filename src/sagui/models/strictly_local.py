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
* an **invariant environment descriptor**
  ``e_i = 1/sqrt(kappa) sum_k u(r_ik) g(x_ik)``, a summary of the neighbours of
  *i* only;
* an **equivariant environment tensor**
  ``Yhat_i[c,lm] = 1/sqrt(kappa) sum_k u(r_ik) g_cl(x_ik) Y_lm(rhat_ik)``,
  which is what carries *angular* information about the neighbourhood;
* ``L`` layers that update the pair tensor ``V_ij`` and the pair scalars
  ``x_ij`` using **only** ``(x_ij, V_ij, Yhat_i, e_i)``.  Crucially the update
  never reads ``e_j``: that single restriction is what keeps the receptive
  field at exactly one cutoff for any depth;
* an energy that is a sum over *pairs*,

  .. math:: E_i = \sum_{j \in \mathcal{N}(i)} u(r_{ij})\, \phi(x_{ij}^{(L)}) ,

  with the cutoff envelope ``u`` multiplying the readout so that energies and
  forces go smoothly to zero as a neighbour leaves the sphere.

The equivariant track is updated by the same weighted tensor product used by
the message-passing model, with the weights produced by an MLP of the
invariants -- so nonlinearity enters through the scalars while the ``l > 0``
components stay exactly equivariant.

Why the environment *tensor* matters
------------------------------------
Coupling ``V_ij`` against ``Y(rhat_ij)`` -- the edge's own direction -- looks
natural but is a trap.  ``V_ij`` starts proportional to ``Y(rhat_ij)``, and the
Clebsch-Gordan contraction of two equivariant functions of a *single* direction
is again a function of that direction, so by induction

    V_ij[c, l, m] = a_cl(invariants) * Y_lm(rhat_ij)

at every depth.  Every rotation invariant of a single direction is a constant,
so :func:`invariant_features` would return pure scalars and the atomic energy
would depend only on the multiset of distances ``{(Z_k, r_ik)}`` -- the model
would be exactly blind to bond angles, which was measured at machine precision
before this was fixed.  Aggregating over ``N(i)`` first mixes many directions
and breaks the collapse: the ``l1 = l2 = 1 -> l3 = 0`` path alone then yields
``sum_k w_k (rhat_ij . rhat_ik)``, the cosine of the bond angle.  The sum still
ranges over ``N(i)`` only, so the receptive field is untouched.
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
    build_weighted_tensor_product,
    invariant_features,
    num_invariants,
    one_hot_species,
)
from ..nn.o3 import SphericalLayout, spherical_harmonics
from ..nn.radial import MLP, BesselBasis, PolynomialCutoff
from ..nn.scatter import scatter_sum
from .base import InteratomicPotential
from .registry import register_model

__all__ = ["StrictlyLocalModel", "LocalLayer", "EnvironmentTensor"]


class EnvironmentTensor(nn.Module):
    r"""Equivariant summary of the neighbourhood of the central atom.

    .. math::
        \hat Y_i[c, lm] = \frac{1}{\sqrt{\bar\kappa}}
            \sum_{k \in \mathcal{N}(i)}
            u_p(r_{ik})\, g_{c,l}\bigl(x_{ik}\bigr)\, Y_{lm}(\hat r_{ik})

    This is Allegro's environment embedding: a learned, per-channel and
    per-degree weighted sum of the neighbour directions.  The weights are
    functions of invariants, so the result transforms as
    :math:`\hat Y \mapsto D^{(l)}(Q)\hat Y` blockwise.

    The sum runs over the neighbours of *i* only, so an edge ``(i, j)`` that
    consumes it gains no receptive field: strict locality is preserved exactly.
    The envelope factor keeps the sum smooth as a neighbour leaves the cutoff
    sphere, as :math:`C^2` continuity of the energy requires.
    """

    def __init__(self, channels: int, sh_lmax: int, latent_dim: int) -> None:
        super().__init__()
        self.channels, self.sh_lmax = int(channels), int(sh_lmax)
        self.weights = nn.Linear(latent_dim, self.channels * (self.sh_lmax + 1))

    def forward(
        self,
        x: torch.Tensor,
        sh: torch.Tensor,
        envelope: torch.Tensor,
        receivers: torch.Tensor,
        num_nodes: int,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        """``x`` ``[E, d]``, ``sh`` ``[E, D]``, ``envelope`` ``[E, 1]`` -> ``[N, C, D]``."""
        weights = self.weights(x).view(-1, self.channels, self.sh_lmax + 1)
        blocks = [
            weights[..., l : l + 1] * sh[:, None, SphericalLayout.block(l)]
            for l in range(self.sh_lmax + 1)
        ]
        edge = torch.cat(blocks, dim=-1) * envelope[..., None]
        return scatter_sum(edge, receivers, num_nodes) / norm


class LocalLayer(nn.Module):
    """One Allegro-style layer acting on a single edge.

    Every input is a per-edge tensor, including the two environment summaries,
    which the model aggregates before the call.  Keeping all aggregation in
    :meth:`StrictlyLocalModel.node_energies` means the locality argument can be
    audited in one function, and it leaves this ``forward`` free of graph
    operations -- which is what makes it a good ``torch.compile`` target.
    """

    def __init__(
        self,
        layout: SphericalLayout,
        sh_lmax: int,
        latent_dim: int,
        env_dim: int,
        hidden: Sequence[int],
        tensor_product: str = "gemm",
    ) -> None:
        super().__init__()
        self.layout = layout
        self.tensor_product = build_weighted_tensor_product(
            tensor_product, layout.lmax, sh_lmax, layout.lmax, channels=layout.channels
        )
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
        self, x: torch.Tensor, v: torch.Tensor, y: torch.Tensor, env: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``y`` is the second tensor-product operand, ``[E, D]`` or ``[E, C, D]``."""
        weights = self.weight_mlp(torch.cat([x, env], dim=-1)).view(
            -1, self.tensor_product.num_paths, self.layout.channels
        )
        # Residual updates are averaged (1/sqrt(2)) so that the activation
        # scale does not drift as layers are stacked.
        v = (v + self.linear(self.tensor_product(v, y, weights))) / math.sqrt(2.0)
        invariants = invariant_features(v, self.layout)
        x = (x + self.scalar_mlp(torch.cat([x, invariants, env], dim=-1))) / math.sqrt(2.0)
        return x, v


@register_model("strictly_local")
class StrictlyLocalModel(InteratomicPotential):
    """Many-body potential whose receptive field is exactly one cutoff."""

    COMPILABLE_LAYERS = ("layers",)

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
        # Invariant environment of the central atom.  One per layer when
        # ``refresh_environment`` is set, so the descriptor is rebuilt from the
        # current latents instead of being frozen at the two-body stage.
        self.refresh_environment = bool(config.refresh_environment)
        num_env = config.num_layers if self.refresh_environment else 1
        self.env_mlps = nn.ModuleList(
            [
                MLP(latent_dim, config.scalar_mlp_hidden, env_dim, final_bias=True)
                for _ in range(num_env)
            ]
        )
        # Equivariant environment tensor -- the operand that carries angular
        # information into the tensor product.  See the module docstring for
        # why coupling against Y(rhat_ij) alone is not enough.
        self.environment_tensor = bool(config.environment_tensor)
        self.env_tensors = nn.ModuleList(
            [
                EnvironmentTensor(config.channels, self.sh_lmax, latent_dim)
                for _ in range(config.num_layers if self.environment_tensor else 0)
            ]
        )
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
                    tensor_product=config.tensor_product,
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

    @staticmethod
    def _invariant_environment(
        mlp: MLP,
        x: torch.Tensor,
        envelope: torch.Tensor,
        data: AtomicGraph,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        """``e_i``: invariant summary of N(i), damped by the envelope to stay smooth."""
        return scatter_sum(mlp(x) * envelope, data.receivers, data.num_nodes) / norm

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
        v = self._initial_tensor(x, sh)

        # Every aggregation in this architecture happens in this loop, and each
        # one sums over N(i) alone -- which is the whole content of the
        # receptive-field theorem.
        env_nodes = self._invariant_environment(self.env_mlps[0], x, envelope, data, norm)
        for index, layer in enumerate(self.layers):
            if self.refresh_environment and index > 0:
                env_nodes = self._invariant_environment(
                    self.env_mlps[index], x, envelope, data, norm
                )
            if self.environment_tensor:
                operand = self.env_tensors[index](
                    x, sh, envelope, data.receivers, data.num_nodes, norm
                )[data.receivers]
            else:
                operand = sh
            x, v = layer(x, v, operand, env_nodes[data.receivers])

        pair_energy = (self.readout(x) * envelope).squeeze(-1)
        return scatter_sum(pair_energy, data.receivers, data.num_nodes)
