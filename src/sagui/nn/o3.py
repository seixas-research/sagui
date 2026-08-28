r"""Self-contained real :math:`O(3)` tensor algebra.

SAGUI deliberately avoids a hard dependency on ``e3nn``; everything needed to
build E(3)-equivariant message passing is implemented here in plain PyTorch:

* **real spherical harmonics** up to ``l = 3`` in *component* normalisation,
  i.e. :math:`\sum_m Y_{lm}(\hat r)^2 = 2l + 1`;
* **Wigner-D matrices** for that basis, obtained numerically by least squares
  from the harmonics themselves;
* the **invariant three-index tensors** (real Clebsch-Gordan / Wigner 3j
  symbols) obtained as the one-dimensional null space of the equivariance
  constraint.

Deriving the D matrices *from* the spherical harmonics is the key trick: it
guarantees the Clebsch-Gordan coefficients follow whatever ordering, sign and
normalisation convention :func:`spherical_harmonics` happens to use, so there
is no convention left to get wrong.

Mathematical background
-----------------------
A vector of features transforming in the irreducible representation ``l`` obeys

.. math:: x'_m = \sum_{m'} D^l_{mm'}(R)\, x_{m'} .

The only bilinear maps ``l1 x l2 -> l3`` compatible with that transformation
law are multiples of the invariant tensor :math:`C^{l_1 l_2 l_3}` satisfying

.. math::
    C_{ijk} = \sum_{i'j'k'} D^{l_1}_{ii'} D^{l_2}_{jj'} D^{l_3}_{kk'}
              C_{i'j'k'} \qquad \forall R \in SO(3),

which exists (and is unique up to scale) iff ``|l1 - l2| <= l3 <= l1 + l2``.
Writing that condition as ``(D1 (x) D2 (x) D3 - 1) vec(C) = 0`` for a handful of
generic rotations turns the problem into a null-space computation.
"""

from __future__ import annotations

import functools
import math

import torch

__all__ = [
    "LMAX_SUPPORTED",
    "SphericalLayout",
    "spherical_harmonics",
    "rotation_matrix",
    "wigner_D",
    "wigner_3j",
    "spherical_to_cartesian_vector",
    "spherical_to_symmetric_matrix",
]

#: Highest spherical-harmonic degree implemented by :func:`spherical_harmonics`.
LMAX_SUPPORTED = 3

# Fixed, generic rotations used to pin down the invariant tensors.  They are
# hard-coded (rather than random) so that the Clebsch-Gordan coefficients are
# bit-for-bit reproducible across machines and runs -- checkpoints depend on it.
_CONSTRAINT_EULER = (
    (0.7, 1.1, 0.3),
    (2.1, 0.4, 1.7),
    (1.3, 2.5, 0.9),
    (0.2, 1.9, 2.8),
)


class SphericalLayout:
    r"""Layout of an equivariant feature tensor.

    SAGUI stores node/edge features as a dense tensor of shape
    ``[N, channels, (lmax + 1) ** 2]``: ``channels`` copies of every degree
    ``l = 0 .. lmax``, each carrying its *natural* parity ``p = (-1)^l`` (in
    ``e3nn`` notation ``channels x 0e + channels x 1o + channels x 2e + ...``).

    Restricting to natural parity is a deliberate simplification: spherical
    harmonics generate exactly those irreps, so no path is ever lost for a
    parity-even scalar target such as the energy, while the bookkeeping stays a
    single integer instead of a full irrep string.  Pseudo-scalar outputs are
    consequently *not* representable -- see the roadmap in ``sagui_context.md``.
    """

    def __init__(self, lmax: int, channels: int) -> None:
        if not 0 <= lmax <= LMAX_SUPPORTED:
            raise ValueError(f"lmax must be in [0, {LMAX_SUPPORTED}], got {lmax}")
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        self.lmax = int(lmax)
        self.channels = int(channels)

    @property
    def dim(self) -> int:
        """Size of the last axis, :math:`\\sum_{l=0}^{l_{max}} (2l+1)`."""
        return (self.lmax + 1) ** 2

    @property
    def ls(self) -> tuple[int, ...]:
        return tuple(range(self.lmax + 1))

    @staticmethod
    def block(l: int) -> slice:
        """Slice selecting the ``2l + 1`` components of degree ``l``."""
        return slice(l * l, (l + 1) * (l + 1))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SphericalLayout)
            and other.lmax == self.lmax
            and other.channels == self.channels
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        irreps = " + ".join(f"{self.channels}x{l}{'e' if l % 2 == 0 else 'o'}" for l in self.ls)
        return f"SphericalLayout({irreps})"


def spherical_harmonics(
    lmax: int,
    vectors: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    r"""Real spherical harmonics up to degree ``lmax``.

    Parameters
    ----------
    lmax:
        Maximum degree, ``0 <= lmax <= 3``.
    vectors:
        Tensor of shape ``[..., 3]``.
    normalize:
        Project onto the unit sphere first.  With ``normalize=False`` the
        returned values are the homogeneous harmonic polynomials of degree
        ``l`` (each block scales as ``|r|^l``).

    Returns
    -------
    torch.Tensor
        Shape ``[..., (lmax + 1) ** 2]``, ordered ``l = 0, 1, ...`` and within
        each degree ``m = -l ... +l``.  Component normalisation is used, so
        ``(Y[..., l**2:(l+1)**2] ** 2).sum(-1) == 2 * l + 1`` on the unit sphere.
    """
    if not 0 <= lmax <= LMAX_SUPPORTED:
        raise ValueError(f"lmax must be in [0, {LMAX_SUPPORTED}], got {lmax}")
    if vectors.shape[-1] != 3:
        raise ValueError(f"expected vectors of shape [..., 3], got {tuple(vectors.shape)}")

    if normalize:
        norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(1e-12)
        vectors = vectors / norm

    x, y, z = vectors.unbind(-1)
    r2 = x * x + y * y + z * z

    out = [torch.ones_like(x)]
    if lmax >= 1:
        c = math.sqrt(3.0)
        out += [c * y, c * z, c * x]
    if lmax >= 2:
        c2, c0 = math.sqrt(15.0), math.sqrt(5.0)
        out += [
            c2 * x * y,
            c2 * y * z,
            0.5 * c0 * (3.0 * z * z - r2),
            c2 * x * z,
            0.5 * c2 * (x * x - y * y),
        ]
    if lmax >= 3:
        a, b, c, d = math.sqrt(70.0), math.sqrt(105.0), math.sqrt(42.0), math.sqrt(7.0)
        out += [
            0.25 * a * y * (3.0 * x * x - y * y),
            b * x * y * z,
            0.25 * c * y * (5.0 * z * z - r2),
            0.5 * d * z * (5.0 * z * z - 3.0 * r2),
            0.25 * c * x * (5.0 * z * z - r2),
            0.5 * b * z * (x * x - y * y),
            0.25 * a * x * (x * x - 3.0 * y * y),
        ]
    return torch.stack(out, dim=-1)


def rotation_matrix(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    """Proper rotation from ZYZ Euler angles, as a ``float64`` ``[3, 3]`` tensor."""

    def _rz(t: float) -> torch.Tensor:
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    def _ry(t: float) -> torch.Tensor:
        c, s = math.cos(t), math.sin(t)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)

    return _rz(alpha) @ _ry(beta) @ _rz(gamma)


@functools.cache
def _sample_points(n: int = 96) -> torch.Tensor:
    """Deterministic quasi-uniform points on the unit sphere (Fibonacci lattice)."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    return torch.stack(
        [torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)],
        dim=-1,
    )


def _sh_block(l: int, vectors: torch.Tensor) -> torch.Tensor:
    """Only the degree-``l`` block of the spherical harmonics."""
    return spherical_harmonics(l, vectors)[..., l * l :]


def wigner_D(l: int, R: torch.Tensor) -> torch.Tensor:
    r"""Representation matrix of ``R`` in the real degree-``l`` basis.

    Solves :math:`Y_l(Rx) = D^l(R)\, Y_l(x)` in the least-squares sense on a
    fixed cloud of sample points.  Because the harmonics are an exact basis of
    the degree-``l`` irrep the fit is exact up to round-off, which is asserted.
    """
    R = R.to(torch.float64)
    pts = _sample_points()
    Y = _sh_block(l, pts)  # [N, 2l+1]
    Y_rot = _sh_block(l, pts @ R.transpose(-1, -2))  # [N, 2l+1]
    # Y_rot = Y @ D^T  =>  D = (pinv(Y) @ Y_rot)^T
    D = torch.linalg.lstsq(Y, Y_rot).solution.transpose(0, 1).contiguous()
    residual = (Y @ D.transpose(0, 1) - Y_rot).abs().max().item()
    if residual > 1e-9:
        raise RuntimeError(f"wigner_D(l={l}) fit failed, residual={residual:.3e}")
    return D


@functools.cache
def wigner_3j(l1: int, l2: int, l3: int) -> torch.Tensor:
    r"""Invariant tensor coupling ``l1 (x) l2 -> l3`` (real Clebsch-Gordan).

    Returned in ``float64`` with shape ``[2*l1+1, 2*l2+1, 2*l3+1]``, unit
    Frobenius norm, and a deterministic sign convention (the first component
    above a relative threshold is positive) so that repeated calls -- and
    different machines -- agree bit-for-bit.

    Raises
    ------
    ValueError
        If the triangle inequality ``|l1 - l2| <= l3 <= l1 + l2`` is violated,
        in which case no such tensor exists.
    """
    if not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError(f"({l1}, {l2}, {l3}) violates the triangle inequality")

    dims = (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)
    size = dims[0] * dims[1] * dims[2]
    eye = torch.eye(size, dtype=torch.float64)

    rows = []
    for angles in _CONSTRAINT_EULER:
        R = rotation_matrix(*angles)
        D1, D2, D3 = (wigner_D(l, R) for l in (l1, l2, l3))
        rows.append(torch.kron(torch.kron(D1, D2), D3) - eye)
    A = torch.cat(rows, dim=0)

    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    if S[-1] > 1e-8:
        raise RuntimeError(f"no invariant tensor found for ({l1}, {l2}, {l3})")
    if size > 1 and S[-2] < 1e-4:
        raise RuntimeError(f"invariant subspace for ({l1}, {l2}, {l3}) is not one-dimensional")

    flat = Vh[-1]
    flat = flat / torch.linalg.norm(flat)

    # Deterministic sign: make the first "significant" entry positive.
    significant = (flat.abs() > 1e-6 * flat.abs().max()).nonzero()[0, 0]
    if flat[significant] < 0:
        flat = -flat
    return flat.reshape(dims).contiguous()


# --------------------------------------------------------------------------
# Bridges back to Cartesian space
#
# Equivariant *inputs* are always spherical harmonics of a direction, but some
# equivariant *outputs* are naturally Cartesian: a force or a score is a
# 3-vector, a strain or a lattice update is a symmetric 3x3 matrix.  The two
# helpers below convert the l = 1 and l = 0 + l = 2 blocks into those objects
# while preserving the transformation law.
# --------------------------------------------------------------------------

#: Permutation taking the l = 1 block, ordered (y, z, x), to (x, y, z).
_L1_TO_XYZ = (2, 0, 1)


def spherical_to_cartesian_vector(block: torch.Tensor) -> torch.Tensor:
    """Turn an ``l = 1`` block ``[..., 3]`` into a Cartesian vector ``[..., 3]``.

    The harmonics of degree one are ``sqrt(3) (y, z, x)``, so the conversion is
    a permutation: if the block transforms as ``c -> D^1(R) c`` then the result
    transforms as ``v -> R v``.
    """
    if block.shape[-1] != 3:
        raise ValueError(f"expected an l=1 block of size 3, got {block.shape[-1]}")
    return block[..., _L1_TO_XYZ]


@functools.cache
def _l2_cartesian_basis() -> torch.Tensor:
    r"""Symmetric traceless matrices ``A_m`` with :math:`Y_{2m}(r) = r^T A_m r`."""
    s15, s5 = math.sqrt(15.0), math.sqrt(5.0)
    half15 = 0.5 * s15
    basis = torch.zeros(5, 3, 3, dtype=torch.float64)
    basis[0, 0, 1] = basis[0, 1, 0] = half15  # xy
    basis[1, 1, 2] = basis[1, 2, 1] = half15  # yz
    basis[2] = 0.5 * s5 * torch.diag(torch.tensor([-1.0, -1.0, 2.0], dtype=torch.float64))
    basis[3, 0, 2] = basis[3, 2, 0] = half15  # xz
    basis[4] = half15 * torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64))
    return basis


def spherical_to_symmetric_matrix(
    scalar: torch.Tensor, block: torch.Tensor
) -> torch.Tensor:
    r"""Assemble a symmetric ``3 x 3`` matrix from an ``l = 0`` and an ``l = 2`` block.

    A symmetric rank-2 tensor decomposes into its trace (a scalar, ``l = 0``)
    and its traceless part (``l = 2``), so

    .. math:: S = s\,\mathbb{1} + \sum_m c_m A_m ,

    and ``S -> R S R^T`` whenever ``s`` is invariant and ``c -> D^2(R) c``.

    Parameters
    ----------
    scalar:
        ``[...]`` invariant part.
    block:
        ``[..., 5]`` degree-two part.
    """
    if block.shape[-1] != 5:
        raise ValueError(f"expected an l=2 block of size 5, got {block.shape[-1]}")
    basis = _l2_cartesian_basis().to(dtype=block.dtype, device=block.device)
    traceless = torch.einsum("...m,mij->...ij", block, basis)
    identity = torch.eye(3, dtype=block.dtype, device=block.device)
    return scalar[..., None, None] * identity + traceless
