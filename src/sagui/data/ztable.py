"""Mapping between atomic numbers and the contiguous species indices used
inside the network."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

__all__ = ["ZTable"]


class ZTable:
    """Ordered set of atomic numbers seen during training.

    Networks embed species as one-hot vectors, so the *order* of this table is
    part of the model definition and is stored in every checkpoint.
    """

    def __init__(self, zs: Iterable[int]) -> None:
        self.zs: tuple[int, ...] = tuple(sorted({int(z) for z in zs}))
        if not self.zs:
            raise ValueError("ZTable cannot be empty")
        self._index = {z: i for i, z in enumerate(self.zs)}

    @classmethod
    def from_atoms(cls, frames: Iterable[Atoms]) -> ZTable:
        zs: set[int] = set()
        for atoms in frames:
            zs.update(int(z) for z in atoms.get_atomic_numbers())
        return cls(zs)

    def index(self, z: int) -> int:
        try:
            return self._index[int(z)]
        except KeyError as exc:
            known = ", ".join(self.symbols)
            raise KeyError(
                f"element {chemical_symbols[int(z)]} (Z={int(z)}) is unknown to this model; "
                f"it was trained on: {known}"
            ) from exc

    def indices(self, zs: Sequence[int] | np.ndarray) -> np.ndarray:
        return np.array([self.index(z) for z in zs], dtype=np.int64)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(chemical_symbols[z] for z in self.zs)

    def __len__(self) -> int:
        return len(self.zs)

    def __iter__(self):
        return iter(self.zs)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZTable) and other.zs == self.zs

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ZTable({', '.join(self.symbols)})"
