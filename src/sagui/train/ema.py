"""Exponential moving average of model weights.

Averaged weights are markedly less noisy than the last SGD iterate and almost
always give a better validation error, so SAGUI evaluates and checkpoints the
averaged model by default.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage:
    """Tracks ``shadow <- decay * shadow + (1 - decay) * weights``."""

    def __init__(self, model: nn.Module, decay: float = 0.99) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap the averaged weights into ``model``."""
        backup = {key: model.state_dict()[key].detach().clone() for key in self.shadow}
        model.load_state_dict(self.shadow, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"decay": torch.tensor(self.decay), **self.shadow}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        state = dict(state)
        self.decay = float(state.pop("decay"))
        self.shadow = {key: value.clone() for key, value in state.items()}
