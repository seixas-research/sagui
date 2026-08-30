"""Container for the dataset-derived scales frozen into a trained model.

It lives with the data layer (rather than with the trainer) because it
describes a *dataset*, and because both the trainer and the checkpoint reader
need it -- keeping it here avoids an import cycle between the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DatasetStatistics", "GenerativeStatistics"]


@dataclass
class DatasetStatistics:
    """Reference energies, energy scale and mean coordination of a dataset."""

    #: Per-element reference energies :math:`E^{(0)}_Z`, ordered as the ZTable.
    atomic_energies: np.ndarray
    #: Global scale applied to the network output (typically the RMS force).
    energy_scale: float
    #: Mean number of neighbours within the cutoff, used to normalise sums.
    avg_num_neighbors: float
    #: RMS of each label's *residual* after the composition fit, used to put the
    #: loss terms on one scale.  Not the same as ``energy_scale``, which scales
    #: the network output: these describe how large each term's error is, so
    #: that a weight of 1 means the same thing for energies, forces and stress.
    energy_residual_rms: float = 1.0
    forces_rms: float = 1.0
    stress_rms: float = 1.0

    def to_dict(self) -> dict:
        return {
            "atomic_energies": [float(x) for x in self.atomic_energies],
            "energy_scale": float(self.energy_scale),
            "avg_num_neighbors": float(self.avg_num_neighbors),
            "energy_residual_rms": float(self.energy_residual_rms),
            "forces_rms": float(self.forces_rms),
            "stress_rms": float(self.stress_rms),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> DatasetStatistics:
        return cls(
            atomic_energies=np.asarray(raw["atomic_energies"], dtype=float),
            energy_scale=float(raw["energy_scale"]),
            avg_num_neighbors=float(raw["avg_num_neighbors"]),
            # Defaults keep checkpoints written before these existed loadable.
            energy_residual_rms=float(raw.get("energy_residual_rms", 1.0)),
            forces_rms=float(raw.get("forces_rms", 1.0)),
            stress_rms=float(raw.get("stress_rms", 1.0)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        energies = ", ".join(f"{e:.4f}" for e in self.atomic_energies)
        return (
            f"DatasetStatistics(E0=[{energies}], scale={self.energy_scale:.4f}, "
            f"avg_num_neighbors={self.avg_num_neighbors:.2f})"
        )


@dataclass
class GenerativeStatistics:
    """Scales and empirical priors needed to train and sample a diffusion model."""

    #: Mean interatomic length ``(V / N)^(1/3)`` over the training set; the unit
    #: in which lattices are diffused.
    lattice_scale: float
    #: Mean coordination of the (noised) training graphs.
    avg_num_neighbors: float
    #: Observed structure sizes, used as the prior over ``N`` when sampling.
    num_atoms: list[int]

    def to_dict(self) -> dict:
        return {
            "lattice_scale": float(self.lattice_scale),
            "avg_num_neighbors": float(self.avg_num_neighbors),
            "num_atoms": [int(n) for n in self.num_atoms],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> GenerativeStatistics:
        return cls(
            lattice_scale=float(raw["lattice_scale"]),
            avg_num_neighbors=float(raw["avg_num_neighbors"]),
            num_atoms=[int(n) for n in raw["num_atoms"]],
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        sizes = sorted(set(self.num_atoms))
        return (
            f"GenerativeStatistics(lattice_scale={self.lattice_scale:.3f} A, "
            f"avg_num_neighbors={self.avg_num_neighbors:.2f}, sizes={sizes})"
        )
