#!/usr/bin/env python
"""Generate a small labelled dataset so the CLI can be tried out immediately.

The reference "ground truth" is a cheap classical potential (ASE's
Lennard-Jones parameterised for argon), which exercises the whole pipeline --
read structures, fit, predict -- without downloading anything or running DFT.

Configurations are rattled fcc geometries rather than random points: random
coordinates put atoms on top of each other, and the resulting ``1/r^12``
energies would swamp the training set.

    python examples/make_toy_dataset.py --output examples/toy.xyz --frames 60
    python examples/make_toy_dataset.py --output examples/toy_bulk.xyz --pbc
"""

from __future__ import annotations

import argparse

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.io import write

# Argon: sigma = 3.40 A, epsilon = 0.0104 eV; the fcc lattice constant below
# puts the nearest neighbours close to the pair minimum, 2^(1/6) sigma.
SIGMA, EPSILON, LATTICE = 3.40, 0.0104, 5.26


def _cluster(num_atoms: int) -> Atoms:
    """The ``num_atoms`` sites of an fcc lattice closest to its centre."""
    crystal = bulk("Ar", "fcc", a=LATTICE, cubic=True) * (3, 3, 3)
    positions = crystal.get_positions()
    centre = positions.mean(axis=0)
    order = np.argsort(np.linalg.norm(positions - centre, axis=1))
    atoms = Atoms(f"Ar{num_atoms}", positions=positions[order[:num_atoms]])
    atoms.center()
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="examples/toy.xyz")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--atoms", type=int, default=8, help="cluster size (ignored with --pbc)")
    parser.add_argument("--pbc", action="store_true", help="periodic bulk instead of a cluster")
    parser.add_argument("--rattle", type=float, default=0.12, help="displacement sigma in Angstrom")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    calculator = LennardJones(sigma=SIGMA, epsilon=EPSILON, rc=7.0)
    template = (
        bulk("Ar", "fcc", a=LATTICE, cubic=True) * (2, 2, 2)
        if args.pbc
        else _cluster(args.atoms)
    )

    frames = []
    for _ in range(args.frames):
        atoms = template.copy()
        # A global strain plus random displacements spans a range of densities
        # and local environments -- enough for the model to have to generalise.
        strain = 1.0 + 0.04 * rng.normal()
        if args.pbc:
            atoms.set_cell(atoms.get_cell() * strain, scale_atoms=True)
        else:
            atoms.set_positions(atoms.get_positions() * strain)
        atoms.rattle(stdev=args.rattle, seed=int(rng.integers(1 << 30)))
        atoms.calc = calculator
        atoms.get_potential_energy()  # populates energy and forces
        frames.append(atoms)

    write(args.output, frames, format="extxyz")
    energies = np.array([a.get_potential_energy() for a in frames])
    forces = np.concatenate([a.get_forces() for a in frames])
    print(f"wrote {len(frames)} frames ({len(template)} atoms each) to {args.output}")
    print(f"energy: {energies.min():.4f} .. {energies.max():.4f} eV")
    print(f"force RMS: {np.sqrt((forces ** 2).mean()):.4f} eV/A")


if __name__ == "__main__":
    main()
