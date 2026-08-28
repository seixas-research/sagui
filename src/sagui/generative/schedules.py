r"""Noise schedules for the three modalities of a crystal.

A material is a triple ``(A, X, L)`` -- discrete atom types, fractional
coordinates on a torus, and a lattice matrix -- and each needs its own
corruption process:

============  ==========================  =========================================
modality      process                     limit distribution at ``t = T``
============  ==========================  =========================================
types ``A``   D3PM (discrete Markov)      uniform over species, or all-masked
coords ``X``  wrapped normal, VE          uniform on the unit cell (a torus)
lattice ``L`` DDPM (variance preserving)  standard normal on 3x3 matrices
============  ==========================  =========================================

All schedules are indexed ``t = 0 .. T`` with ``t = 0`` meaning *clean data*,
so ``alpha_bar[0] == 1`` and ``sigma[0] == 0``.
"""

from __future__ import annotations

import math

import torch

__all__ = ["cosine_alpha_bar", "geometric_sigmas", "betas_from_alpha_bar"]


def cosine_alpha_bar(
    num_steps: int, offset: float = 0.008, max_beta: float = 0.999
) -> torch.Tensor:
    r"""Cosine schedule of Nichol & Dhariwal (2021).

    .. math:: \bar\alpha_t = \frac{f(t)}{f(0)}, \qquad
              f(t) = \cos^2\!\Big(\frac{t/T + s}{1 + s}\frac{\pi}{2}\Big)

    Returns ``[T + 1]`` with ``alpha_bar[0] = 1``.  Compared with a linear
    schedule it destroys information more slowly at both ends, which matters
    here because the interesting structure of a crystal lives at low noise.
    Single-step ``beta`` is capped at ``max_beta`` and ``alpha_bar`` is rebuilt
    from the capped values, so the two are always mutually consistent.
    """
    t = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
    f = torch.cos((t + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
    raw = f / f[0]

    # Clamping alpha_bar directly would leave it inconsistent with the per-step
    # betas: a clamped tail is a plateau, and the first step out of that plateau
    # would have an alpha of ~0, which the reverse process amplifies by
    # 1/sqrt(alpha).  Clamp the *betas* instead and rebuild alpha_bar from them,
    # so that alpha_bar[t] == prod(1 - beta) holds exactly by construction.
    betas = torch.zeros_like(raw)
    betas[1:] = (1.0 - raw[1:] / raw[:-1]).clamp(0.0, max_beta)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return alpha_bar


def betas_from_alpha_bar(alpha_bar: torch.Tensor, max_beta: float = 0.999) -> torch.Tensor:
    r"""Per-step :math:`\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}`, ``[T + 1]``.

    ``betas[0]`` is zero by construction (no step is taken to reach ``t = 0``).
    """
    betas = torch.zeros_like(alpha_bar)
    betas[1:] = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0, max_beta)
    return betas


def geometric_sigmas(num_steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    r"""Geometric (variance-exploding) noise levels, ``[T + 1]`` with ``sigma[0] = 0``.

    Fractional coordinates are diffused with a *wrapped* normal, so the useful
    range of noise is bounded: once ``sigma`` approaches the size of the unit
    cell (1 in fractional units) the distribution is already uniform and larger
    values add nothing.  A geometric ladder spends its steps where the density
    actually changes.
    """
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError(f"expected 0 < sigma_min < sigma_max, got {sigma_min}, {sigma_max}")
    ladder = torch.logspace(
        math.log10(sigma_min), math.log10(sigma_max), num_steps, dtype=torch.float64
    )
    return torch.cat([torch.zeros(1, dtype=torch.float64), ladder])
