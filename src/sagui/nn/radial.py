r"""Radial (invariant) machinery: basis functions, cutoffs and plain MLPs.

Every distance dependence in SAGUI enters through this module.  Two properties
matter for molecular dynamics:

* **smoothness** -- energies and forces must be continuous when an atom crosses
  the cutoff sphere, which is enforced by multiplying every radial feature with
  a polynomial envelope that vanishes together with its first ``p`` derivatives
  at ``r_max``;
* **completeness** -- the Bessel basis
  :math:`b_n(r) = \sqrt{2/r_c}\,\sin(n \pi r / r_c) / r` is the radial part of
  the solutions of the Helmholtz equation in a sphere and forms an orthonormal
  set on ``[0, r_c]``, so a small number of them already resolves sharp
  features near the minimum of a pair potential.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["BesselBasis", "PolynomialCutoff", "MLP"]


class BesselBasis(nn.Module):
    r"""Orthonormal Bessel radial basis, cf. Gasteiger et al., *DimeNet* (2020)."""

    def __init__(self, r_max: float, num_basis: int = 8, trainable: bool = False) -> None:
        super().__init__()
        self.num_basis = int(num_basis)
        self.register_buffer("r_max", torch.tensor(float(r_max)))
        self.prefactor = math.sqrt(2.0 / float(r_max))
        frequencies = math.pi * torch.arange(1, self.num_basis + 1, dtype=torch.get_default_dtype())
        frequencies = frequencies / float(r_max)
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """``r``: ``[E, 1]`` -> ``[E, num_basis]``."""
        r = r.clamp_min(1e-9)
        return self.prefactor * torch.sin(self.frequencies * r) / r

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"num_basis={self.num_basis}, r_max={self.r_max.item():.3f}"


class PolynomialCutoff(nn.Module):
    r"""Smooth envelope :math:`u(r)` with :math:`u(r_c) = u'(r_c) = ... = 0`.

    .. math::
        u(x) = 1 - \tfrac{(p+1)(p+2)}{2} x^p + p(p+2) x^{p+1}
                 - \tfrac{p(p+1)}{2} x^{p+2}, \qquad x = r / r_c ,

    identically zero for ``r >= r_c``.
    """

    def __init__(self, r_max: float, p: int = 6) -> None:
        super().__init__()
        self.p = int(p)
        self.register_buffer("r_max", torch.tensor(float(r_max)))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        p = self.p
        x = r / self.r_max
        envelope = (
            1.0
            - (p + 1.0) * (p + 2.0) / 2.0 * torch.pow(x, p)
            + p * (p + 2.0) * torch.pow(x, p + 1)
            - p * (p + 1.0) / 2.0 * torch.pow(x, p + 2)
        )
        return envelope * (x < 1.0)

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"p={self.p}, r_max={self.r_max.item():.3f}"


class MLP(nn.Module):
    """Multilayer perceptron on invariant (scalar) features, SiLU activated.

    The final layer is linear and, by default, bias-free: it often feeds
    equivariant weights, where a bias would break nothing but would make the
    initial output non-zero-mean for no benefit.
    """

    def __init__(
        self,
        dim_in: int,
        hidden: Sequence[int],
        dim_out: int,
        bias: bool = True,
        final_bias: bool = False,
    ) -> None:
        super().__init__()
        dims = [int(dim_in), *(int(h) for h in hidden), int(dim_out)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            last = i == len(dims) - 2
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=(final_bias if last else bias)))
            if not last:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.dim_in, self.dim_out = dims[0], dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
