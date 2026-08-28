"""Segment reductions used to move information between edges and nodes."""

from __future__ import annotations

import torch

__all__ = ["scatter_sum"]


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` into ``dim_size`` buckets given by ``index`` along axis 0.

    A dependency-free stand-in for ``torch_scatter.scatter_sum``; ``index_add_``
    is differentiable and deterministic enough for our purposes (on CUDA it is
    non-deterministic in the float-accumulation order only).
    """
    if index.dim() != 1 or index.shape[0] != src.shape[0]:
        raise ValueError(
            f"index must be 1-D with {src.shape[0]} entries, got {tuple(index.shape)}"
        )
    out = src.new_zeros((dim_size, *src.shape[1:]))
    return out.index_add(0, index, src)
