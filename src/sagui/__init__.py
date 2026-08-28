"""SAGUI -- equivariant graph neural-network interatomic potentials.

Quick start
-----------
Train from the command line::

    sagui-train config.yaml
    sagui-inference --model runs/sagui/best.model --input test.xyz --evaluate

or drive a trained potential from Python::

    from ase.io import read
    from sagui import SaguiCalculator

    atoms = read("structure.xyz")
    atoms.calc = SaguiCalculator("runs/sagui/best.model")
    print(atoms.get_potential_energy(), atoms.get_forces())
"""

from .calculator import SaguiCalculator
from .checkpoint import load_model, save_checkpoint
from .config import Config, DataConfig, ModelConfig, TrainingConfig
from .data import AtomicGraph, AtomsDataset, ZTable, collate_graphs, graph_from_atoms
from .models import InteratomicPotential, available_models, build_model, register_model
from .train import run_training
from .version import __version__

__all__ = [
    "AtomicGraph",
    "AtomsDataset",
    "Config",
    "DataConfig",
    "InteratomicPotential",
    "ModelConfig",
    "SaguiCalculator",
    "TrainingConfig",
    "ZTable",
    "__version__",
    "available_models",
    "build_model",
    "collate_graphs",
    "graph_from_atoms",
    "load_model",
    "register_model",
    "run_training",
    "save_checkpoint",
]
