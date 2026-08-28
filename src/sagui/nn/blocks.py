r"""Equivariant building blocks shared by every SAGUI architecture.

All feature tensors have shape ``[N, channels, (lmax + 1) ** 2]`` as described
in :class:`sagui.nn.o3.SphericalLayout`.  Three operations suffice to build
both supported architectures:

``EquivariantLinear``
    mixes channels *within* a degree -- the only linear map that commutes with
    rotations (Schur's lemma);
``WeightedTensorProduct`` / ``SelfTensorProduct``
    couple two equivariant tensors through the invariant Clebsch-Gordan
    tensors, the only source of *equivariant* nonlinearity;
``invariant_features``
    reads rotation invariants back out so that ordinary MLPs can act on them.

Normalisation convention
------------------------
Inputs are assumed to have unit variance per component.  With unit-Frobenius
Clebsch-Gordan tensors a single path contributes variance ``1 / (2 l_3 + 1)``
per output component, so each path is scaled by
``sqrt(2 l_3 + 1) / sqrt(#paths -> l_3)`` to keep activations O(1) at
initialisation without any learned normalisation layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .o3 import SphericalLayout, wigner_3j

__all__ = [
    "EquivariantLinear",
    "SpeciesLinear",
    "WeightedTensorProduct",
    "SelfTensorProduct",
    "invariant_features",
    "num_invariants",
    "scalars",
    "embed_scalars",
]


def scalars(x: torch.Tensor) -> torch.Tensor:
    """The ``l = 0`` block of an equivariant tensor: ``[N, C, D] -> [N, C]``."""
    return x[..., 0]


def embed_scalars(s: torch.Tensor, layout: SphericalLayout) -> torch.Tensor:
    """Lift invariant channels ``[N, C]`` into a full tensor with zero ``l > 0``."""
    zeros = s.new_zeros((*s.shape, layout.dim - 1))
    return torch.cat([s.unsqueeze(-1), zeros], dim=-1)


def num_invariants(layout: SphericalLayout) -> int:
    """Width of :func:`invariant_features` for ``layout``."""
    return layout.channels * (layout.lmax + 1)


def invariant_features(x: torch.Tensor, layout: SphericalLayout) -> torch.Tensor:
    r"""Extract rotation invariants: ``[N, C, D] -> [N, C * (lmax + 1)]``.

    The ``l = 0`` block is passed through unchanged; every ``l > 0`` block
    contributes its mean square :math:`\|x_l\|^2 / (2l + 1)`, which is
    invariant, smooth (unlike the norm, which is not differentiable at zero)
    and unit-scaled when the inputs are.
    """
    parts = [scalars(x)]
    for l in layout.ls[1:]:
        block = x[..., SphericalLayout.block(l)]
        parts.append(block.pow(2).sum(-1) / (2 * l + 1))
    return torch.cat(parts, dim=-1)


class EquivariantLinear(nn.Module):
    """Channel mixing applied independently to each degree ``l``.

    Weights are drawn from a unit normal and scaled by ``1 / sqrt(C_in)`` at
    call time (``e3nn`` convention), which decouples the learning rate from the
    channel count.
    """

    def __init__(self, lmax: int, channels_in: int, channels_out: int | None = None) -> None:
        super().__init__()
        channels_out = channels_in if channels_out is None else channels_out
        self.lmax = int(lmax)
        self.channels_in = int(channels_in)
        self.channels_out = int(channels_out)
        self.weight = nn.Parameter(torch.randn(self.lmax + 1, channels_out, channels_in))
        self.alpha = channels_in**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for l in range(self.lmax + 1):
            block = x[..., SphericalLayout.block(l)]
            out.append(torch.einsum("...ci,oc->...oi", block, self.weight[l]) * self.alpha)
        return torch.cat(out, dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"lmax={self.lmax}, {self.channels_in} -> {self.channels_out}"


class SpeciesLinear(nn.Module):
    """Element-dependent :class:`EquivariantLinear` (MACE's self-interaction).

    Every chemical species owns a private set of mixing weights, which lets a
    single model carry element-specific length and energy scales.
    """

    def __init__(
        self, num_species: int, lmax: int, channels_in: int, channels_out: int | None = None
    ) -> None:
        super().__init__()
        channels_out = channels_in if channels_out is None else channels_out
        self.lmax = int(lmax)
        self.weight = nn.Parameter(
            torch.randn(num_species, self.lmax + 1, channels_out, channels_in)
        )
        self.alpha = channels_in**-0.5

    def forward(self, x: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        w = self.weight[species]  # [N, lmax+1, C_out, C_in]
        out = []
        for l in range(self.lmax + 1):
            block = x[..., SphericalLayout.block(l)]
            out.append(torch.einsum("nci,noc->noi", block, w[:, l]) * self.alpha)
        return torch.cat(out, dim=-1)


class _TensorProductBase(nn.Module):
    """Shared path bookkeeping for the two tensor-product flavours.

    A path ``(l1, l2, l3)`` is kept when it satisfies the triangle inequality
    *and* conserves parity, ``(-1)^(l1 + l2 + l3) = +1``.  The parity rule is
    what makes the network equivariant under the full ``O(3)`` rather than just
    ``SO(3)``: it discards couplings such as ``1 x 1 -> 1`` (the Levi-Civita
    tensor) whose output would be a pseudo-vector.
    """

    def __init__(self, lmax_in1: int, lmax_in2: int, lmax_out: int) -> None:
        super().__init__()
        self.lmax_in1, self.lmax_in2, self.lmax_out = int(lmax_in1), int(lmax_in2), int(lmax_out)
        paths: list[tuple[int, int, int]] = []
        for l1 in range(self.lmax_in1 + 1):
            for l2 in range(self.lmax_in2 + 1):
                for l3 in range(abs(l1 - l2), min(l1 + l2, self.lmax_out) + 1):
                    if (l1 + l2 + l3) % 2 == 0:
                        paths.append((l1, l2, l3))
        if not paths:
            raise ValueError("tensor product has no valid paths")
        self.paths = tuple(paths)

        counts = {l3: sum(1 for p in paths if p[2] == l3) for _, _, l3 in paths}
        dtype = torch.get_default_dtype()
        norms = []
        for k, (l1, l2, l3) in enumerate(self.paths):
            # Non-persistent: the coefficients are reproducible constants, so
            # they are recomputed on load instead of bloating checkpoints.
            self.register_buffer(f"w3j_{k}", wigner_3j(l1, l2, l3).to(dtype), persistent=False)
            norms.append(((2 * l3 + 1) / counts[l3]) ** 0.5)
        self.path_norms = tuple(norms)

    @property
    def num_paths(self) -> int:
        return len(self.paths)

    def _w3j(self, k: int) -> torch.Tensor:
        return getattr(self, f"w3j_{k}")

    def _gather(
        self, contributions: list[torch.Tensor | None], template: torch.Tensor
    ) -> torch.Tensor:
        out = []
        for l3 in range(self.lmax_out + 1):
            acc = contributions[l3]
            if acc is None:  # degree unreachable by any path -> exactly zero
                acc = template.new_zeros((*template.shape[:-1], 2 * l3 + 1))
            out.append(acc)
        return torch.cat(out, dim=-1)

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"paths={self.num_paths}, lmax_out={self.lmax_out}"


class WeightedTensorProduct(_TensorProductBase):
    r"""Couple features with spherical harmonics using per-edge weights.

    .. math::
        (x \otimes_w Y)^{(l_3)}_{c,k} = \sum_{\text{paths}} w_{p,c}\,
            \sum_{ij} C^{l_1 l_2 l_3}_{ijk}\, x^{(l_1)}_{c,i}\, Y^{(l_2)}_j

    This is the ``uvu``-style channel-wise product of ``e3nn`` specialised to a
    second operand with a single channel (the harmonics), which is exactly the
    convolution filter of NequIP/MACE: ``w`` is produced by an MLP of the
    interatomic distance, so the filter is a learned radial function times a
    spherical harmonic -- the general form of an equivariant kernel.
    """

    def forward(self, x: torch.Tensor, sh: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x``: ``[E, C, D1]``, ``sh``: ``[E, D2]``, ``weights``: ``[E, P, C]``."""
        if weights.shape[1] != self.num_paths:
            raise ValueError(f"expected {self.num_paths} path weights, got {weights.shape[1]}")
        contributions: list[torch.Tensor | None] = [None] * (self.lmax_out + 1)
        for k, (l1, l2, l3) in enumerate(self.paths):
            term = torch.einsum(
                "eci,ej,ijk->eck",
                x[..., SphericalLayout.block(l1)],
                sh[..., SphericalLayout.block(l2)],
                self._w3j(k),
            )
            term = term * weights[:, k, :].unsqueeze(-1) * self.path_norms[k]
            contributions[l3] = term if contributions[l3] is None else contributions[l3] + term
        return self._gather(contributions, x)


class SelfTensorProduct(_TensorProductBase):
    r"""Channel-wise product of two feature tensors with learned path weights.

    Used to build the many-body terms of the MACE-style model: taking the
    product of the one-particle basis with itself raises the correlation order
    by one, so ``nu`` nested products give ``(nu + 1)``-body interactions.
    """

    def __init__(self, lmax_in1: int, lmax_in2: int, lmax_out: int, channels: int) -> None:
        super().__init__(lmax_in1, lmax_in2, lmax_out)
        self.channels = int(channels)
        self.weight = nn.Parameter(torch.randn(self.num_paths, channels))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Both inputs ``[N, C, D]``; output ``[N, C, D_out]``."""
        contributions: list[torch.Tensor | None] = [None] * (self.lmax_out + 1)
        for k, (l1, l2, l3) in enumerate(self.paths):
            term = torch.einsum(
                "eci,ecj,ijk->eck",
                x1[..., SphericalLayout.block(l1)],
                x2[..., SphericalLayout.block(l2)],
                self._w3j(k),
            )
            term = term * self.weight[k].view(1, -1, 1) * self.path_norms[k]
            contributions[l3] = term if contributions[l3] is None else contributions[l3] + term
        return self._gather(contributions, x1)


def one_hot_species(species: torch.Tensor, num_species: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot chemical-species encoding, ``[N] -> [N, num_species]``."""
    return F.one_hot(species, num_classes=num_species).to(dtype)
