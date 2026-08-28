"""Model architectures.

Importing this package registers every built-in architecture, so
``build_model`` can resolve ``model.type`` without further imports.
"""

from .base import InteratomicPotential
from .mace import MACEModel
from .registry import available_models, build_model, get_model_class, register_model
from .strictly_local import StrictlyLocalModel

__all__ = [
    "InteratomicPotential",
    "MACEModel",
    "StrictlyLocalModel",
    "available_models",
    "build_model",
    "get_model_class",
    "register_model",
]
