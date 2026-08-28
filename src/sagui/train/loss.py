r"""Loss functions and error metrics.

The energy term is normalised per atom, so that large and small structures
contribute comparably; the force term is a plain component-wise mean square.
The default weights (``1`` and ``100``) reflect the usual situation where the
forces, being ``3N`` numbers per structure and directly relevant to dynamics,
carry most of the training signal.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.atomic_data import AtomicGraph

__all__ = ["EnergyForcesLoss", "compute_metrics"]


class EnergyForcesLoss(nn.Module):
    """Weighted mean-square error on energies (per atom) and forces."""

    def __init__(self, energy_weight: float = 1.0, forces_weight: float = 100.0) -> None:
        super().__init__()
        self.energy_weight = float(energy_weight)
        self.forces_weight = float(forces_weight)

    def forward(
        self, prediction: dict[str, torch.Tensor], reference: AtomicGraph
    ) -> tuple[torch.Tensor, dict[str, float]]:
        terms: dict[str, float] = {}
        total = prediction["energy"].new_zeros(())

        if reference.energy is not None and self.energy_weight > 0.0:
            n_atoms = reference.num_atoms.to(prediction["energy"].dtype)
            error = (prediction["energy"] - reference.energy) / n_atoms
            energy_loss = error.pow(2).mean()
            total = total + self.energy_weight * energy_loss
            terms["energy_mse"] = float(energy_loss.detach())

        if reference.forces is not None and self.forces_weight > 0.0:
            forces_loss = (prediction["forces"] - reference.forces).pow(2).mean()
            total = total + self.forces_weight * forces_loss
            terms["forces_mse"] = float(forces_loss.detach())

        if not terms:
            raise ValueError(
                "batch carries neither energy nor force labels; nothing to train on"
            )
        terms["loss"] = float(total.detach())
        return total, terms


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
    return out
