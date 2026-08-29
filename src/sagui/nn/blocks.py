r"""Equivariant building blocks shared by every SAGUI architecture.

All feature tensors have shape ``[N, channels, (lmax + 1) ** 2]`` as described
in :class:`sagui.nn.o3.SphericalLayout`.  Three operations suffice to build
both supported architectures:

``EquivariantLinear``
    mixes channels *within* a degree -- the only linear map that commutes with
    rotations (Schur's lemma);
``WeightedTensorProduct`` / ``GemmWeightedTensorProduct`` / ``SelfTensorProduct``
    couple two equivariant tensors through the invariant Clebsch-Gordan
    tensors, the only source of *equivariant* nonlinearity.  The first two are
    mathematically identical and differ only in how the contraction is
    scheduled -- see :func:`build_weighted_tensor_product`;
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
    "GemmWeightedTensorProduct",
    "SelfTensorProduct",
    "TENSOR_PRODUCT_KINDS",
    "build_weighted_tensor_product",
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
    r"""Couple features with a second equivariant operand using per-edge weights.

    .. math::
        (x \otimes_w Y)^{(l_3)}_{c,k} = \sum_{\text{paths}} w_{p,c}\,
            \sum_{ij} C^{l_1 l_2 l_3}_{ijk}\, x^{(l_1)}_{c,i}\, Y^{(l_2)}_j

    This is the ``uvu``-style channel-wise product of ``e3nn`` specialised to a
    second operand with a single channel (the harmonics), which is exactly the
    convolution filter of NequIP/MACE: ``w`` is produced by an MLP of the
    interatomic distance, so the filter is a learned radial function times a
    spherical harmonic -- the general form of an equivariant kernel.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x``: ``[E, C, D1]``; ``y``: ``[E, D2]`` or ``[E, C, D2]``; ``w``: ``[E, P, C]``."""
        if weights.shape[1] != self.num_paths:
            raise ValueError(f"expected {self.num_paths} path weights, got {weights.shape[1]}")
        # A channel-less second operand is the spherical harmonic of a single
        # edge; a channelled one is an aggregated environment tensor.
        equation = "eci,ecj,ijk->eck" if y.dim() == 3 else "eci,ej,ijk->eck"
        contributions: list[torch.Tensor | None] = [None] * (self.lmax_out + 1)
        for k, (l1, l2, l3) in enumerate(self.paths):
            term = torch.einsum(
                equation,
                x[..., SphericalLayout.block(l1)],
                y[..., SphericalLayout.block(l2)],
                self._w3j(k),
            )
            term = term * weights[:, k, :].unsqueeze(-1) * self.path_norms[k]
            contributions[l3] = term if contributions[l3] is None else contributions[l3] + term
        return self._gather(contributions, x)


class GemmWeightedTensorProduct(_TensorProductBase):
    r"""Single-GEMM evaluation of :class:`WeightedTensorProduct`.

    Mathematically identical to the path loop -- the two agree to ~5e-15 in
    double precision -- but scheduled as one dense contraction instead of one
    ``einsum`` per coupling path.  The loop is dispatch-bound rather than
    FLOP-bound: profiling a 216-atom cell put it at 75% of the layer's forward
    time while performing 6% of its arithmetic, running at ~0.9 GFLOP/s beside
    scalar MLPs reaching ~200 GFLOP/s.

    The reformulation writes the outer product of the two operands once,

    .. math:: z_{c,(ij)} = x^{}_{c,i}\, y^{}_{c,j},

    and then observes that the Clebsch-Gordan contraction is a *constant*
    matrix acting on the combined ``(i, j)`` axis, so every path is evaluated
    by a single ``[E C, D_1 D_2] x [D_1 D_2, S]`` matrix product with
    :math:`S = \sum_p (2 l_3 + 1)`.  Applying the per-edge path weights and
    summing the columns into their output degrees finishes the job in five
    kernels.

    The second operand may carry a channel axis or not, so this class also
    serves products against an aggregated (per-atom) equivariant tensor.

    Memory
    ------
    The intermediate ``z`` holds ``E * channels * D_1 * D_2`` elements.  To keep
    that bounded on large systems the edge axis is processed in chunks sized so
    the intermediate stays under :attr:`INTERMEDIATE_BUDGET` elements; pass
    ``chunk_edges`` to override, or a value larger than ``E`` to disable
    chunking entirely.
    """

    #: Element budget for the ``z`` intermediate (2**26 ~ 256 MB in float32).
    INTERMEDIATE_BUDGET = 1 << 26

    def __init__(
        self,
        lmax_in1: int,
        lmax_in2: int,
        lmax_out: int,
        channels: int | None = None,
        chunk_edges: int | None = None,
    ) -> None:
        super().__init__(lmax_in1, lmax_in2, lmax_out)
        d1 = (self.lmax_in1 + 1) ** 2
        d2 = (self.lmax_in2 + 1) ** 2
        d3 = (self.lmax_out + 1) ** 2

        # cg[(i, j), column] gathers every path into one constant matrix; the
        # column bookkeeping records which path weight and which output degree
        # each column belongs to.
        n_cols = sum(2 * l3 + 1 for _, _, l3 in self.paths)
        cg = torch.zeros(d1 * d2, n_cols, dtype=torch.get_default_dtype())
        col_path: list[int] = []
        col_out: list[int] = []
        column = 0
        for k, (l1, l2, l3) in enumerate(self.paths):
            w3j = self._w3j(k) * self.path_norms[k]
            for m in range(2 * l3 + 1):
                block = torch.zeros(d1, d2, dtype=cg.dtype)
                block[l1 * l1 : (l1 + 1) ** 2, l2 * l2 : (l2 + 1) ** 2] = w3j[:, :, m]
                cg[:, column] = block.reshape(-1)
                col_path.append(k)
                col_out.append(l3 * l3 + m)
                column += 1

        # Non-persistent, exactly like the w3j buffers: reproducible constants
        # rebuilt on load rather than stored in every checkpoint.
        self.register_buffer("cg", cg, persistent=False)
        self.register_buffer("col_path", torch.tensor(col_path, dtype=torch.long), persistent=False)
        self.register_buffer("col_out", torch.tensor(col_out, dtype=torch.long), persistent=False)
        self.dim_out = d3
        if chunk_edges is None:
            width = (channels or 1) * d1 * d2
            chunk_edges = max(1, self.INTERMEDIATE_BUDGET // width)
        self.chunk_edges = int(chunk_edges)

    def _contract(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        z = (x.unsqueeze(-1) * y.unsqueeze(-2)).flatten(-2)  # [E, C, D1 * D2]
        q = z @ self.cg  # [E, C, S]
        q = q * weights.index_select(1, self.col_path).transpose(1, 2)
        out = q.new_zeros((*q.shape[:-1], self.dim_out))
        return out.index_add(-1, self.col_out, q)

    def forward(self, x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``x``: ``[E, C, D1]``; ``y``: ``[E, D2]`` or ``[E, C, D2]``; ``w``: ``[E, P, C]``."""
        if weights.shape[1] != self.num_paths:
            raise ValueError(f"expected {self.num_paths} path weights, got {weights.shape[1]}")
        if y.dim() == 2:
            y = y.unsqueeze(1)
        if x.shape[0] <= self.chunk_edges:
            return self._contract(x, y, weights)
        return torch.cat(
            [
                self._contract(xc, yc, wc)
                for xc, yc, wc in zip(
                    x.split(self.chunk_edges),
                    y.split(self.chunk_edges),
                    weights.split(self.chunk_edges),
                    strict=True,
                )
            ],
            dim=0,
        )


#: Interchangeable implementations of the weighted tensor product.  They differ
#: only in scheduling; ``"gemm"`` is faster, ``"loop"`` uses less memory and is
#: kept as the reference the fused kernel is tested against.
TENSOR_PRODUCT_KINDS: dict[str, type[_TensorProductBase]] = {
    "loop": WeightedTensorProduct,
    "gemm": GemmWeightedTensorProduct,
}


def build_weighted_tensor_product(
    kind: str, lmax_in1: int, lmax_in2: int, lmax_out: int, channels: int | None = None
) -> _TensorProductBase:
    """Instantiate the weighted tensor product named by ``kind``.

    Both kinds compute the same function and carry no learnable parameters, so
    a checkpoint trained with one loads unchanged into the other.
    """
    try:
        cls = TENSOR_PRODUCT_KINDS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown tensor product '{kind}'; available: "
            f"{', '.join(sorted(TENSOR_PRODUCT_KINDS))}"
        ) from exc
    if cls is GemmWeightedTensorProduct:
        return cls(lmax_in1, lmax_in2, lmax_out, channels=channels)
    return cls(lmax_in1, lmax_in2, lmax_out)


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
