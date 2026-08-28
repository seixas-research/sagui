"""ASE calculator interface, so a trained potential can drive any ASE workflow.

    >>> from ase.optimize import BFGS
    >>> atoms.calc = SaguiCalculator("runs/sagui/best.model")
    >>> BFGS(atoms).run(fmax=0.01)

Geometry optimisation, molecular dynamics, phonons, NEB -- anything that
speaks the ASE calculator protocol works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .checkpoint import load_model
from .data.atomic_data import collate_graphs, graph_from_atoms
from .data.ztable import ZTable
from .models.base import InteratomicPotential
from .utils import resolve_device_and_dtype

__all__ = ["SaguiCalculator"]


class SaguiCalculator(Calculator):
    """Evaluate a trained SAGUI model on ``ase.Atoms`` objects."""

    implemented_properties = ["energy", "free_energy", "energies", "forces"]

    def __init__(
        self,
        model: str | Path | InteratomicPotential | None = None,
        z_table: ZTable | None = None,
        device: str = "auto",
        default_dtype: str = "float64",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Path to a checkpoint, or an already-built model (then ``z_table``
            is required).
        device:
            ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        default_dtype:
            ``float64`` by default: dynamics is sensitive to the noise floor of
            single precision, and inference is rarely the bottleneck.  Metal has
            no ``float64``, so on MPS this is downgraded to ``float32`` with a
            warning -- pass ``device="cpu"`` if the precision matters more than
            the throughput.
        """
        super().__init__(**kwargs)
        self.device, self.dtype = resolve_device_and_dtype(device, default_dtype)

        if model is None:
            raise ValueError("a model path or an InteratomicPotential instance is required")
        if isinstance(model, (str, Path)):
            self.model, self.config, self.z_table = load_model(
                model, device=self.device, dtype=self.dtype
            )
            self.r_max = float(self.config.model.r_max)
        else:
            if z_table is None:
                raise ValueError("z_table is required when passing a model instance")
            self.model = model.to(device=self.device, dtype=self.dtype).eval()
            self.z_table = z_table
            self.r_max = float(model.r_max.item())

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None

        graph = graph_from_atoms(
            self.atoms, self.z_table, self.r_max, with_labels=False, dtype=self.dtype
        )
        batch = collate_graphs([graph]).to(self.device)
        out = self.model(batch, compute_forces=True, training=False)

        self.results = {
            "energy": float(out["energy"].detach().cpu().item()),
            "energies": out["node_energy"].detach().cpu().numpy().astype(np.float64),
            "forces": out["forces"].detach().cpu().numpy().astype(np.float64),
        }
        self.results["free_energy"] = self.results["energy"]

    @torch.no_grad()
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SaguiCalculator(species={', '.join(self.z_table.symbols)}, r_max={self.r_max})"
