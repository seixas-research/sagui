r"""Ziegler-Biersack-Littmark screened-nuclear repulsion.

Training sets are built from configurations a sampler actually visits, which
means they contain almost no close contacts.  The repulsive wall a purely
data-driven potential learns is therefore an extrapolation, and it is usually
far too soft: two atoms pushed together during a hot or badly initialised
trajectory feel a finite -- sometimes attractive -- force and collapse onto one
another.  Adding a physical core is the standard cure, and it costs nothing to
evaluate.

The ZBL universal screening function

.. math::
    E^{\mathrm{ZBL}}_{ij}(r) =
        \frac{1}{4\pi\varepsilon_0}\frac{Z_i Z_j e^2}{r}\;\phi\!\left(\frac{r}{a_{ij}}\right),
    \qquad
    a_{ij} = \frac{0.46850\,\text{\AA}}{Z_i^{0.23} + Z_j^{0.23}},

.. math::
    \phi(x) = 0.18175\,e^{-3.19980x} + 0.50986\,e^{-0.94229x}
            + 0.28022\,e^{-0.40290x} + 0.02817\,e^{-0.20162x},

was fitted to Hartree-Fock calculations across the periodic table and needs no
parameters of its own -- which is the point.  It is a *constraint*, not another
thing to fit, so nothing here is learnable.

It is switched off smoothly well inside the model cutoff by the same
:math:`C^2` polynomial envelope used everywhere else, so the property that the
energy is twice differentiable survives (see the framework document,
Corollary 4.3).  The network then only ever has to learn the difference between
the reference data and this core.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.atomic_data import AtomicGraph
from ..nn.radial import PolynomialCutoff
from ..nn.scatter import scatter_sum

__all__ = ["ZBLRepulsion"]

#: :math:`e^2 / 4\pi\varepsilon_0` in eV angstrom.
COULOMB_CONSTANT = 14.399645478425668

#: Coefficients and exponents of the universal screening function.
_SCREEN_COEFFS = (0.18175, 0.50986, 0.28022, 0.02817)
_SCREEN_EXPONENTS = (3.19980, 0.94229, 0.40290, 0.20162)

#: Length scale of the screening, in angstrom.
_SCREEN_LENGTH = 0.46850
_SCREEN_POWER = 0.23


class ZBLRepulsion(nn.Module):
    """Parameter-free pair repulsion, switched off smoothly at ``cutoff``.

    Parameters
    ----------
    atomic_numbers:
        The species table, so that a species *index* can be mapped to the
        nuclear charge the ZBL formula actually needs.
    cutoff:
        Where the repulsion is switched off.  It should sit well below the
        model cutoff -- around 1.5-2 angstrom -- so that the term only ever
        acts in the region the training data does not cover.
    p:
        Order of the switching polynomial, as in :class:`PolynomialCutoff`.
    """

    def __init__(self, atomic_numbers: torch.Tensor, cutoff: float, p: int = 6) -> None:
        super().__init__()
        self.register_buffer("charges", atomic_numbers.to(torch.get_default_dtype()))
        self.switch = PolynomialCutoff(cutoff, p)
        self.cutoff = float(cutoff)

    @staticmethod
    def _screening(x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(x)
        for coefficient, exponent in zip(_SCREEN_COEFFS, _SCREEN_EXPONENTS, strict=True):
            total = total + coefficient * torch.exp(-exponent * x)
        return total

    def pair_energy(self, z_i: torch.Tensor, z_j: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Screened repulsion of one pair, ``[E, 1] -> [E, 1]``, already switched."""
        screen_length = _SCREEN_LENGTH / (z_i**_SCREEN_POWER + z_j**_SCREEN_POWER)
        safe = r.clamp_min(1e-6)
        coulomb = COULOMB_CONSTANT * z_i * z_j / safe
        return coulomb * self._screening(safe / screen_length) * self.switch(r)

    def forward(self, data: AtomicGraph, lengths: torch.Tensor) -> torch.Tensor:
        """Per-atom repulsion energy ``[N]``.

        The graph stores both directions of every pair, so the sum over
        directed edges double counts and is halved.
        """
        charges = self.charges[data.species]
        pair = self.pair_energy(
            charges[data.receivers].unsqueeze(-1),
            charges[data.senders].unsqueeze(-1),
            lengths,
        )
        return 0.5 * scatter_sum(pair.squeeze(-1), data.receivers, data.num_nodes)

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"cutoff={self.cutoff:.2f}"
