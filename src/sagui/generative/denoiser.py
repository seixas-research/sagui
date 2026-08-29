r"""The equivariant denoiser at the heart of the generative model.

It reuses the message-passing backbone of the potential (the very same
:class:`~sagui.models.mace.InteractionBlock`) and differs only in what it
reads out.  The three heads exploit three different degrees of the equivariant
features, which is exactly what the symmetry of each target demands:

============  ==========  ============================================
target        degree      transformation under a rotation ``Q``
============  ==========  ============================================
type logits   ``l = 0``   invariant
coordinates   ``l = 1``   a vector, ``v -> Q v``
lattice       ``l = 0,2`` a symmetric tensor, ``S -> Q S Q^T``
============  ==========  ============================================

The coordinate head predicts a *Cartesian* vector and converts it to the
fractional score by contracting with the lattice, ``s_frac = v L^T``, which is
both the correct chain rule and manifestly invariant -- as a fractional
quantity must be.  The lattice head predicts a symmetric ``S`` and returns
``eps_L = Y_t S``, which transforms like the lattice itself, ``eps -> eps Q^T``.

Conditioning on the noise level uses a sinusoidal embedding of ``t`` injected
before every interaction, plus the rotation-invariant Gram matrix ``Y Y^T`` of
the current lattice, which tells the network the cell shape without breaking
any symmetry.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import ModelConfig
from ..data.atomic_data import AtomicGraph
from ..models.mace import InteractionBlock
from ..nn.blocks import embed_scalars, invariant_features, num_invariants, one_hot_species, scalars
from ..nn.o3 import (
    SphericalLayout,
    spherical_harmonics,
    spherical_to_cartesian_vector,
    spherical_to_symmetric_matrix,
)
from ..nn.radial import MLP, BesselBasis, PolynomialCutoff
from ..nn.scatter import scatter_sum

__all__ = ["EquivariantDenoiser", "SinusoidalEmbedding"]

#: The lattice head needs the degree-two block, so the denoiser never uses a
#: layout narrower than this whatever the configuration says.
MIN_LMAX = 2


class SinusoidalEmbedding(nn.Module):
    """Transformer-style positional embedding of the diffusion timestep."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"embedding dimension must be even, got {dim}")
        self.dim = int(dim)
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.get_default_dtype()) / half
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``t``: ``[G]`` in ``[0, 1]`` -> ``[G, dim]``."""
        angles = t.unsqueeze(-1) * self.frequencies * 1000.0
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def gram_invariants(lattice: torch.Tensor) -> torch.Tensor:
    r"""Rotation-invariant description of a cell: the Gram matrix plus its volume.

    ``Y Y^T`` is unchanged by ``Y -> Y Q^T``, so these seven numbers describe
    the cell *shape* without referring to any orientation.
    """
    gram = lattice @ lattice.transpose(-1, -2)
    upper = torch.stack(
        [gram[:, 0, 0], gram[:, 1, 1], gram[:, 2, 2], gram[:, 0, 1], gram[:, 0, 2], gram[:, 1, 2]],
        dim=-1,
    )
    volume = torch.linalg.det(lattice).abs().clamp_min(1e-6).log().unsqueeze(-1)
    return torch.cat([upper, volume], dim=-1)


class EquivariantDenoiser(nn.Module):
    """Predicts clean types, the coordinate score and the lattice noise."""

    def __init__(
        self,
        config: ModelConfig,
        num_tokens: int,
        num_species: int,
        num_steps: int,
        avg_num_neighbors: float = 12.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_tokens = int(num_tokens)
        self.num_species = int(num_species)
        self.num_steps = int(num_steps)
        lmax = max(int(config.lmax), MIN_LMAX)
        self.layout = SphericalLayout(lmax, config.channels)
        self.sh_lmax = max(config.spherical_lmax, MIN_LMAX)
        channels = config.channels
        time_dim = max(32, channels)

        self.register_buffer("avg_num_neighbors", torch.tensor(float(avg_num_neighbors)))
        self.token_embedding = nn.Linear(self.num_tokens, channels, bias=False)
        self.time_embedding = SinusoidalEmbedding(time_dim)
        self.time_mlp = MLP(time_dim, [time_dim], time_dim, final_bias=True)
        self.lattice_embedding = MLP(7, config.scalar_mlp_hidden, channels, final_bias=True)

        self.radial_basis = BesselBasis(config.r_max, config.num_radial_basis)
        self.cutoff = PolynomialCutoff(config.r_max, config.cutoff_p)

        self.interactions = nn.ModuleList(
            [
                InteractionBlock(
                    self.layout,
                    self.sh_lmax,
                    config.num_radial_basis,
                    config.radial_mlp_hidden,
                    self.num_tokens,
                    config.correlation,
                )
                for _ in range(config.num_layers)
            ]
        )
        # One time projection per layer: the network has to behave very
        # differently at high and low noise, and a single input-side injection
        # washes out after a couple of message-passing steps.
        self.time_projections = nn.ModuleList(
            [nn.Linear(time_dim, channels) for _ in range(config.num_layers)]
        )

        invariants = num_invariants(self.layout)
        self.type_head = MLP(invariants + time_dim, config.scalar_mlp_hidden, self.num_species,
                             final_bias=True)
        self.coord_gate = MLP(invariants + time_dim, config.scalar_mlp_hidden, channels,
                              final_bias=True)
        self.lattice_gate = MLP(invariants + time_dim, config.scalar_mlp_hidden, channels,
                                final_bias=True)
        self.lattice_scalar = MLP(invariants + time_dim, config.scalar_mlp_hidden, 1,
                                  final_bias=True)

    def forward(
        self, graph: AtomicGraph, t: torch.Tensor, lattice_norm: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        graph:
            Batched graph of the *noised* structure; ``graph.species`` holds
            the current type tokens.
        t:
            ``[G]`` integer timesteps.
        lattice_norm:
            ``[G, 3, 3]`` current normalised lattice.

        Returns
        -------
        dict with ``type_logits`` ``[N, num_species]``, ``coord_score``
        ``[N, 3]`` (fractional, in units of ``sigma * score``) and
        ``lattice_noise`` ``[G, 3, 3]``.
        """
        dtype = graph.positions.dtype
        vectors = graph.positions[graph.senders] - graph.positions[graph.receivers] + graph.shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-6)

        time = self.time_mlp(self.time_embedding(t.to(dtype) / self.num_steps))  # [G, T]
        time_nodes = time[graph.batch]
        lattice_nodes = self.lattice_embedding(gram_invariants(lattice_norm))[graph.batch]

        tokens = one_hot_species(graph.species, self.num_tokens, dtype)
        h = embed_scalars(self.token_embedding(tokens) + lattice_nodes, self.layout)

        sh = spherical_harmonics(self.sh_lmax, vectors)
        envelope = self.cutoff(lengths)
        radial = self.radial_basis(lengths) * envelope
        norm = torch.sqrt(self.avg_num_neighbors.to(dtype))

        for interaction, projection in zip(self.interactions, self.time_projections, strict=True):
            h = h + embed_scalars(projection(time_nodes), self.layout)
            h = interaction(h, sh, radial, envelope, graph, norm)

        features = torch.cat([invariant_features(h, self.layout), time_nodes], dim=-1)

        # --- types: invariant ---------------------------------------------
        type_logits = self.type_head(features)

        # --- coordinates: an l=1 vector, pulled back to fractional space ---
        vector_block = h[..., SphericalLayout.block(1)]  # [N, C, 3]
        gated = (vector_block * self.coord_gate(features).unsqueeze(-1)).sum(dim=1)
        cartesian = spherical_to_cartesian_vector(gated)  # [N, 3]
        # s_frac = s_cart L^T -- the chain rule for r = x L, and invariant.
        coord_score = torch.einsum("na,nba->nb", cartesian, lattice_norm[graph.batch])

        # --- lattice: a symmetric tensor, averaged over the structure ------
        tensor_block = h[..., SphericalLayout.block(2)]  # [N, C, 5]
        gated_2 = (tensor_block * self.lattice_gate(features).unsqueeze(-1)).sum(dim=1)
        symmetric = spherical_to_symmetric_matrix(
            self.lattice_scalar(features).squeeze(-1), gated_2
        )
        counts = graph.num_atoms.to(dtype).view(-1, 1, 1)
        pooled = scatter_sum(symmetric, graph.batch, graph.num_graphs) / counts
        lattice_noise = lattice_norm @ pooled

        return {
            "type_logits": type_logits,
            "coord_score": coord_score,
            "lattice_noise": lattice_noise,
        }

    def node_scalars(self, h: torch.Tensor) -> torch.Tensor:  # pragma: no cover - helper
        return scalars(h)
