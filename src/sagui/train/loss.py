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
  gradient -- the failure mode that makes large public datasets hard to fit;
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
        ``None`` (the default) gives a mean square.  A float switches to a Huber
        residual with that transition point; ``0.01`` is the MACE-MP setting.

    The weights are plain attributes so a schedule can rewrite them between
    epochs -- see :func:`sagui.train.trainer.switch_loss_phase`.
    """

    def __init__(
        self,
        energy_weight: float = 1.0,
        forces_weight: float = 100.0,
        stress_weight: float = 0.0,
        huber_delta: float | None = None,
    ) -> None:
        super().__init__()
        self.energy_weight = float(energy_weight)
        self.forces_weight = float(forces_weight)
        self.stress_weight = float(stress_weight)
        self.huber_delta = None if huber_delta is None else float(huber_delta)

    @property
    def wants_stress(self) -> bool:
        """Whether the model needs to be asked for a stress at all."""
        return self.stress_weight > 0.0

    def _residual(self, error: torch.Tensor) -> torch.Tensor:
        if self.huber_delta is None:
            return error.pow(2).mean()
        delta = self.huber_delta
        absolute = error.abs()
        quadratic = torch.minimum(absolute, torch.full_like(absolute, delta))
        return (0.5 * quadratic.pow(2) + delta * (absolute - quadratic)).mean()

    def forward(
        self, prediction: dict[str, torch.Tensor], reference: AtomicGraph
    ) -> tuple[torch.Tensor, dict[str, float]]:
        terms: dict[str, float] = {}
        total = prediction["energy"].new_zeros(())
        suffix = "mse" if self.huber_delta is None else "huber"

        if reference.energy is not None and self.energy_weight > 0.0:
            n_atoms = reference.num_atoms.to(prediction["energy"].dtype)
            loss = self._residual((prediction["energy"] - reference.energy) / n_atoms)
            total = total + self.energy_weight * loss
            terms[f"energy_{suffix}"] = float(loss.detach())

        if reference.forces is not None and self.forces_weight > 0.0:
            loss = self._residual(prediction["forces"] - reference.forces)
            total = total + self.forces_weight * loss
            terms[f"forces_{suffix}"] = float(loss.detach())

        if reference.stress is not None and self.wants_stress and "stress" in prediction:
            loss = self._residual(prediction["stress"] - reference.stress)
            total = total + self.stress_weight * loss
            terms[f"stress_{suffix}"] = float(loss.detach())

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
