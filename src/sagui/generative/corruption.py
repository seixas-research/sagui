r"""The joint forward (noising) process of a crystal.

Coordinates
-----------
Fractional coordinates live on a torus, so the natural corruption is a
*wrapped* normal: :math:`x_t = (x_0 + \sigma_t \varepsilon) \bmod 1`.  Its
density is the periodic sum

.. math:: q(x_t \mid x_0) = \sum_{k \in \mathbb{Z}}
          \mathcal{N}(x_t - x_0 - k;\ 0, \sigma_t^2),

whose score -- the training target -- is a softmax-weighted average over
periodic images.  A variance-exploding ladder is used because ``sigma = 1``
already means "uniform over the cell": there is nothing beyond it.

Lattice
-------
A standard variance-preserving DDPM on the ``3 x 3`` matrix, expressed in
units of the mean interatomic distance, ``Y = L / (scale * N^(1/3))``, so that
a cubic cell maps to the identity regardless of how many atoms it holds.  No
mean is subtracted: centring on the identity would single out one orientation
and break the rotational covariance of the process, whereas an uncentred
Gaussian on ``3 x 3`` matrices is invariant under ``Y -> Y Q^T``.

Types
-----
Delegated to :class:`~sagui.generative.d3pm.D3PM`.
"""

from __future__ import annotations

import torch
from torch import nn

from .d3pm import D3PM
from .schedules import cosine_alpha_bar, geometric_sigmas
from .structures import wrap_fractional

__all__ = ["MaterialsCorruption", "wrapped_normal_score"]


def wrapped_normal_score(
    delta: torch.Tensor, sigma: torch.Tensor, num_images: int = 4
) -> torch.Tensor:
    r"""Score :math:`\nabla_{x_t} \log q(x_t \mid x_0)` of the wrapped normal.

    ``delta = x_t - x_0`` in fractional units; the sum over periodic images is
    truncated at ``+/- num_images``, which is far beyond machine precision for
    every ``sigma`` the schedule visits.
    """
    images = torch.arange(
        -num_images, num_images + 1, device=delta.device, dtype=delta.dtype
    )
    offsets = delta.unsqueeze(-1) - images  # [..., 2n+1]
    sigma = sigma.unsqueeze(-1)
    weights = torch.softmax(-0.5 * (offsets / sigma) ** 2, dim=-1)
    return -(weights * offsets).sum(-1) / sigma.squeeze(-1) ** 2


class MaterialsCorruption(nn.Module):
    """Forward process for ``(types, fractional coordinates, lattice)``."""

    def __init__(
        self,
        num_species: int,
        num_steps: int = 1000,
        transition: str = "absorbing",
        sigma_min: float = 0.005,
        sigma_max: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_steps = int(num_steps)
        self.types = D3PM(num_species, num_steps, transition)

        dtype = torch.get_default_dtype()
        self.register_buffer("sigmas", geometric_sigmas(num_steps, sigma_min, sigma_max).to(dtype))
        self.register_buffer("alpha_bar", cosine_alpha_bar(num_steps).to(dtype))

    @property
    def num_tokens(self) -> int:
        return self.types.num_tokens

    def sample_timesteps(self, num: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Uniform ``t`` in ``1 .. T``."""
        return torch.randint(1, self.num_steps + 1, (num,), device=device)

    # ------------------------------------------------------------- coords
    def corrupt_coords(
        self, frac_0: torch.Tensor, t_atom: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x_t, target, sigma)`` where ``target = sigma * score``.

        Predicting ``sigma * score`` rather than the score itself keeps the
        regression target of order one at every noise level.
        """
        sigma = self.sigmas[t_atom].unsqueeze(-1)  # [N, 1]
        noise = torch.randn_like(frac_0)
        frac_t = wrap_fractional(frac_0 + sigma * noise)
        delta = frac_t - frac_0
        target = sigma * wrapped_normal_score(delta, sigma.expand_as(delta))
        return frac_t, target, sigma

    # ------------------------------------------------------------ lattice
    def corrupt_lattice(
        self, lattice_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard DDPM step on the normalised lattice; returns ``(Y_t, eps)``."""
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1)
        noise = torch.randn_like(lattice_0)
        lattice_t = alpha_bar.sqrt() * lattice_0 + (1.0 - alpha_bar).sqrt() * noise
        return lattice_t, noise

    # -------------------------------------------------------------- types
    def corrupt_types(self, types_0: torch.Tensor, t_atom: torch.Tensor) -> torch.Tensor:
        return self.types.q_sample(types_0, t_atom)
