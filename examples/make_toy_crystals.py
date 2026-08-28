#!/usr/bin/env python
"""Generate a small set of periodic binary crystals for the generative task.

The generative model diffuses atom types, fractional coordinates *and* the
lattice, so its training data must be 3D-periodic and -- to make the discrete
part non-trivial -- contain more than one element.  Rattled and strained
rocksalt cells are about the simplest dataset with all of those properties.

    python examples/make_toy_crystals.py --output examples/crystals.xyz --frames 200
"""

from __future__ import annotations

import argparse

import numpy as np
from ase.build import bulk
from ase.io import write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/crystals.xyz")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--rattle", type=float, default=0.08, help="displacement sigma in Angstrom")
    parser.add_argument("--strain", type=float, default=0.03, help="lattice strain sigma")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    template = bulk("MgO", "rocksalt", a=4.21, cubic=True)  # 8 atoms, 4 Mg + 4 O

    frames = []
    for _ in range(args.frames):
        atoms = template.copy()
        # Anisotropic strain keeps the lattice distribution from collapsing
        # onto a single cubic cell, which would make the lattice head trivial.
        strain = np.eye(3) + args.strain * rng.normal(size=(3, 3))
        atoms.set_cell(atoms.get_cell() @ strain, scale_atoms=True)
        atoms.rattle(stdev=args.rattle, seed=int(rng.integers(1 << 30)))
        frames.append(atoms)

    write(args.output, frames, format="extxyz")
    volumes = np.array([a.get_volume() for a in frames])
    print(f"wrote {len(frames)} frames ({len(template)} atoms each) to {args.output}")
    print(f"formula: {template.get_chemical_formula()}")
    print(f"volume: {volumes.min():.2f} .. {volumes.max():.2f} A^3")


if __name__ == "__main__":
    main()
