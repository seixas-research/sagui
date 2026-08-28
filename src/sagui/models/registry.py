"""Architecture registry -- the mechanism behind ``model.type`` in the YAML.

Adding an architecture means writing a module and decorating its class with
``@register_model("my_arch")``; nothing else in the code base has to change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..config import ModelConfig
from .base import InteratomicPotential

__all__ = ["register_model", "build_model", "available_models", "get_model_class"]

_REGISTRY: dict[str, type[InteratomicPotential]] = {}


def register_model(name: str) -> Callable[[type[InteratomicPotential]], type[InteratomicPotential]]:
    """Class decorator registering an architecture under ``name``."""

    def decorator(cls: type[InteratomicPotential]) -> type[InteratomicPotential]:
        key = name.lower()
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"architecture '{key}' is already registered")
        if not issubclass(cls, InteratomicPotential):
            raise TypeError(f"{cls.__name__} must subclass InteratomicPotential")
        _REGISTRY[key] = cls
        cls.architecture_name = key  # type: ignore[attr-defined]
        return cls

    return decorator


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model_class(name: str) -> type[InteratomicPotential]:
    try:
        return _REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture '{name}'; available: {', '.join(available_models())}"
        ) from exc


def build_model(
    config: ModelConfig,
    atomic_numbers: Sequence[int],
    atomic_energies: Sequence[float] | torch.Tensor | None = None,
    energy_scale: float = 1.0,
    avg_num_neighbors: float = 1.0,
) -> InteratomicPotential:
    """Instantiate the architecture named by ``config.type``.

    ``config.avg_num_neighbors``, when set, overrides the value measured on the
    dataset -- useful to keep a fine-tuned model consistent with its parent.
    """
    cls = get_model_class(config.type)
    if config.avg_num_neighbors is not None:
        avg_num_neighbors = float(config.avg_num_neighbors)
    return cls(
        config,
        atomic_numbers=atomic_numbers,
        atomic_energies=atomic_energies,
        energy_scale=energy_scale,
        avg_num_neighbors=avg_num_neighbors,
    )
