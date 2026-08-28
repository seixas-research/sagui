"""``sagui-generate`` -- sample new crystal structures from a trained diffusion model."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from ..checkpoint import load_generative_model
from ..generative.diffusion import GeneratedStructure
from ..utils import resolve_device, resolve_dtype, set_seed, setup_logging
from ..version import __version__

__all__ = ["main", "build_parser", "to_atoms"]

logger = logging.getLogger("sagui-generate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sagui-generate",
        description=(
            "Generate crystal structures with a trained SAGUI diffusion model "
            "(D3PM atom types + periodic coordinate diffusion + lattice diffusion)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sagui {__version__}")
    parser.add_argument("-m", "--model", required=True, help="generative checkpoint")
    parser.add_argument("-n", "--num-structures", type=int, default=8)
    parser.add_argument(
        "--num-atoms",
        type=int,
        nargs="+",
        help="atoms per structure; repeated or cycled. Default: drawn from the training set",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="reverse steps (default: the full training schedule; fewer is faster)",
    )
    parser.add_argument("-o", "--output", default="generated.xyz", help="output structure file")
    parser.add_argument("--format", dest="file_format", help="ASE format (inferred from --output)")
    parser.add_argument("--batch-size", type=int, default=8, help="structures sampled at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--default-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def to_atoms(structure: GeneratedStructure, atomic_numbers: tuple[int, ...]) -> Atoms:
    """Convert a sampled structure into a periodic ``ase.Atoms``."""
    numbers = [atomic_numbers[int(i)] for i in structure.species]
    cell = structure.cell.numpy().astype(float)
    return Atoms(
        numbers=numbers,
        scaled_positions=structure.frac.numpy().astype(float),
        cell=cell,
        pbc=True,
    )


def _sizes(requested: list[int] | None, prior: list[int], count: int, rng) -> list[int]:
    """Pick the number of atoms of each sample."""
    if requested:
        return [requested[i % len(requested)] for i in range(count)]
    return [int(prior[i]) for i in rng.integers(0, len(prior), size=count)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(logging, args.log_level))
    set_seed(args.seed)

    dtype = resolve_dtype(args.default_dtype)
    torch.set_default_dtype(dtype)
    device = resolve_device(args.device)

    model, config, z_table, stats = load_generative_model(args.model, device=device, dtype=dtype)
    logger.info(
        "loaded generative model: %d steps, '%s' type kernel, species %s",
        model.num_steps,
        config.diffusion.type_transition,
        ", ".join(z_table.symbols),
    )

    rng = np.random.default_rng(args.seed)
    requested = args.num_atoms or config.diffusion.sample_num_atoms
    sizes = _sizes(requested, stats.num_atoms, args.num_structures, rng)
    logger.info("generating %d structure(s) with %d reverse steps",
                len(sizes), args.steps or model.num_steps)

    frames: list[Atoms] = []
    for start in range(0, len(sizes), args.batch_size):
        chunk = sizes[start : start + args.batch_size]
        structures = model.sample(chunk, device=device, num_steps=args.steps, progress=True)
        frames.extend(to_atoms(s, z_table.zs) for s in structures)
        logger.info("  %d/%d done", len(frames), len(sizes))

    print(f"\n{'#':>5}  {'formula':<20} {'natoms':>6} {'volume [A^3]':>13} {'a, b, c [A]':>26}")
    print("-" * 78)
    for index, atoms in enumerate(frames):
        lengths = atoms.cell.lengths()
        print(
            f"{index:>5}  {atoms.get_chemical_formula():<20} {len(atoms):>6} "
            f"{atoms.get_volume():>13.3f} "
            f"{lengths[0]:>8.3f} {lengths[1]:>8.3f} {lengths[2]:>8.3f}"
        )

    composition = Counter(symbol for atoms in frames for symbol in atoms.get_chemical_symbols())
    if composition:
        summary = ", ".join(f"{symbol}: {count}" for symbol, count in composition.most_common())
        print(f"\noverall composition: {summary}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write(args.output, frames, format=args.file_format)
    logger.info("wrote %d structure(s) to %s", len(frames), args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
