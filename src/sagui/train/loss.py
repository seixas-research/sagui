r"""Loss functions and error metrics.

The energy term is normalised per atom, so that large and small structures
contribute comparably; the force and stress terms are component-wise.  The
default weights (``1`` and ``100``) reflect the usual situation where the
forces, being ``3N`` numbers per structure and directly relevant to dynamics,
carry most of the training signal.

Two options exist beyond plain mean squares, both taken from the MACE-MP and
OMat24 recipes:

* a **Huber** residual, quadratic below ``delta`` and linear above it, so that a
  handful of badly converged reference calculations cannot dominate the
  gradient -- the failure mode that makes large public datasets hard to fit.
  ``delta`` is an *absolute* threshold, so it belongs to a term, not to the
  loss: per-atom energies, force components and stresses have residuals that
  differ by orders of magnitude, and one shared value cannot suit all three.
  Measured on MPtrj, a shared ``delta = 0.01`` improved forces by 35% while
  doubling the energy error.  Each term therefore takes its own.

  **Switching a term to Huber also re-weights it, which is easy to miss.**  For
  the same residuals the two functions differ enormously in magnitude: at an
  RMS residual of 7 eV/atom, ``Huber(0.01)`` returns ~885x less than the mean
  square.  Turning Huber on for the energy term therefore divides its effective
  ``lambda_E`` by roughly that factor.  The apparent "force improvement" from a
  shared ``delta`` on MPtrj was exactly this: the energy term shrank and the
  force term took over.  ``delta`` and ``lambda`` are entangled -- if you change
  one, re-check the other;
* a **stress** term, which needs ``compute_stress=True`` on the model and stress
  labels on the batch, and which is reported to improve materials accuracy well
  beyond the stress prediction itself.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.atomic_data import AtomicGraph

__all__ = ["EnergyForcesStressLoss", "EnergyForcesLoss", "compute_metrics"]


class EnergyForcesStressLoss(nn.Module):
    """Weighted error on per-atom energies, forces and (optionally) stress.

    Parameters
    ----------
    energy_weight, forces_weight, stress_weight:
        Relative weights.  A weight of zero drops the term entirely, which is
        also what happens when the batch carries no matching label.
    huber_delta:
        Fallback transition point for every term.  ``None`` (the default) gives
        a mean square.
    scales:
        Per-term residual scales.  Each error is divided by its scale before the
        residual is taken, which is what makes a weight of 1 mean the same thing
        for energies, forces and stress -- and what makes ``huber_delta``
        dimensionless, so ``delta = 1`` puts the crossover at the typical
        residual instead of at an absolute number that has to be rediscovered
        for every dataset.  Defaults to 1 for every term, i.e. no scaling.
    huber_delta_energy, huber_delta_forces, huber_delta_stress:
        Per-term overrides, each falling back to ``huber_delta``.  Set them from
        the residual scale of the corresponding label: a useful starting point
        is roughly the RMS of that term's residual, so the bulk of the data
        stays in the quadratic region and only outliers are linearised.  A term
        left at ``None`` keeps its mean square, so the three can be mixed.

    The weights are plain attributes so a schedule can rewrite them between
    epochs -- see :func:`sagui.train.trainer.apply_weight_switch`.
    """

    #: ``term -> (prediction key, reference attribute)``, in report order.
    FIELDS = {
        "energy": ("energy", "energy"),
        "forces": ("forces", "forces"),
        "stress": ("stress", "stress"),
        "charges": ("charges", "charges"),
        "magmoms": ("magmoms", "magmoms"),
    }
    TERMS = tuple(FIELDS)

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
        stress_weight: float = 0.0,
        charges_weight: float = 0.0,
        magmoms_weight: float = 0.0,
        huber_delta: float | None = None,
        huber_delta_energy: float | None = None,
        huber_delta_forces: float | None = None,
        huber_delta_stress: float | None = None,
        scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.energy_weight = float(energy_weight)
        self.forces_weight = float(forces_weight)
        self.stress_weight = float(stress_weight)
        self.charges_weight = float(charges_weight)
        self.magmoms_weight = float(magmoms_weight)
        overrides = {
            "energy": huber_delta_energy,
            "forces": huber_delta_forces,
            "stress": huber_delta_stress,
            "charges": None,
            "magmoms": None,
        }
        self.deltas: dict[str, float | None] = {}
        for term in self.TERMS:
            delta = overrides[term] if overrides[term] is not None else huber_delta
            if delta is not None and float(delta) <= 0.0:
                raise ValueError(f"huber delta for '{term}' must be positive, got {delta}")
            self.deltas[term] = None if delta is None else float(delta)
        scales = scales or {}
        self.scales = {term: float(scales.get(term, 1.0)) for term in self.TERMS}
        for term, value in self.scales.items():
            if value <= 0.0:
                raise ValueError(f"scale for '{term}' must be positive, got {value}")

    @property
    def wants_stress(self) -> bool:
        """Whether the model needs to be asked for a stress at all."""
        return self.stress_weight > 0.0

    def weight(self, term: str) -> float:
        return float(getattr(self, f"{term}_weight"))

    def _residual(self, error: torch.Tensor, term: str) -> torch.Tensor:
        error = error / self.scales[term]
        delta = self.deltas[term]
        if delta is None:
            return error.pow(2).mean()
        absolute = error.abs()
        quadratic = torch.minimum(absolute, torch.full_like(absolute, delta))
        return (0.5 * quadratic.pow(2) + delta * (absolute - quadratic)).mean()

    def _label(self, term: str) -> str:
        return f"{term}_{'mse' if self.deltas[term] is None else 'huber'}"

    def forward(
        self, prediction: dict[str, torch.Tensor], reference: AtomicGraph
    ) -> tuple[torch.Tensor, dict[str, float]]:
        terms: dict[str, float] = {}
        total = prediction["energy"].new_zeros(())

        for term, (key, attribute) in self.FIELDS.items():
            weight = self.weight(term)
            target = getattr(reference, attribute)
            if weight <= 0.0 or target is None or key not in prediction:
                continue
            error = prediction[key] - target
            if term == "energy":
                # Per atom, so that large and small structures count equally.
                error = error / reference.num_atoms.to(error.dtype)
            loss = self._residual(error, term)
            total = total + weight * loss
            terms[self._label(term)] = float(loss.detach())

        if not terms:
            raise ValueError(
                "batch carries neither energy nor force labels; nothing to train on"
            )
        terms["loss"] = float(total.detach())
        return total, terms


#: Backwards-compatible name from before the stress term existed.
EnergyForcesLoss = EnergyForcesStressLoss


@torch.no_grad()
def compute_metrics(
    prediction: dict[str, torch.Tensor], reference: AtomicGraph
) -> dict[str, float]:
    """Accumulator-friendly error sums (not yet averaged) for one batch."""
    out: dict[str, float] = {"n_structures": float(reference.num_graphs)}
    if reference.energy is not None:
        n_atoms = reference.num_atoms.to(prediction["energy"].dtype)
        error = (prediction["energy"] - reference.energy) / n_atoms
        out["energy_abs_sum"] = float(error.abs().sum())
        out["energy_sq_sum"] = float(error.pow(2).sum())
    if reference.forces is not None:
        error = prediction["forces"] - reference.forces
        out["forces_abs_sum"] = float(error.abs().sum())
        out["forces_sq_sum"] = float(error.pow(2).sum())
        out["n_force_components"] = float(error.numel())
    if reference.stress is not None and "stress" in prediction:
        error = prediction["stress"] - reference.stress
        out["stress_abs_sum"] = float(error.abs().sum())
        out["stress_sq_sum"] = float(error.pow(2).sum())
        out["n_stress_components"] = float(error.numel())
    return out
