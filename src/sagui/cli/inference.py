"""``sagui-inference`` -- predict energies and forces with a trained model."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from torch.utils.data import DataLoader

from ..checkpoint import load_model
from ..data.atomic_data import collate_graphs, extract_labels
from ..data.dataset import AtomsDataset, read_structures
from ..utils import resolve_device_and_dtype, setup_logging
from ..version import __version__

__all__ = ["main", "build_parser"]

logger = logging.getLogger("sagui-inference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sagui-inference",
        description=(
            "Run a trained SAGUI potential on new structures and report the "
            "predicted energies and forces."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"sagui {__version__}")
    parser.add_argument(
        "-m", "--model", required=True, help="checkpoint written by sagui-train"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="structures to evaluate (.xyz, .traj, ...)"
    )
    parser.add_argument(
        "-o", "--output", help="extended-XYZ file with the predictions attached"
    )
    parser.add_argument("--index", default=":", help="ASE slice of frames to read (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--default-dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="compare against the reference labels present in the input file",
    )
    parser.add_argument("--json", dest="json_out", help="write per-structure results as JSON")
    parser.add_argument(
        "--print-forces", action="store_true", help="print the full force table per structure"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the layers (~2x faster per step, tens of seconds to warm up)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(logging, args.log_level))

    device, dtype = resolve_device_and_dtype(args.device, args.default_dtype)
    torch.set_default_dtype(dtype)
    logger.info("running on %s in %s", device, str(dtype).replace("torch.", ""))

    model, config, z_table = load_model(args.model, device=device, dtype=dtype)
    if args.compile:
        logger.info("compiling %d layers (first batch will be slow)", model.compile_layers())
    logger.info(
        "loaded '%s' model (r_max=%.2f A, species: %s)",
        config.model.type,
        config.model.r_max,
        ", ".join(z_table.symbols),
    )

    frames = read_structures(args.input, index=args.index)
    logger.info("evaluating %d structure(s) from %s", len(frames), args.input)

    dataset = AtomsDataset(
        frames,
        z_table=z_table,
        r_max=config.model.r_max,
        with_labels=False,
        dtype=dtype,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs
    )

    energies: list[float] = []
    forces: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, compute_forces=True, training=False)
        energies.extend(out["energy"].detach().cpu().tolist())
        split = torch.split(out["forces"].detach().cpu(), batch.num_atoms.tolist())
        forces.extend(f.numpy().astype(np.float64) for f in split)

    # ------------------------------------------------------------- report
    print(f"\n{'#':>5}  {'formula':<16} {'natoms':>6} {'E [eV]':>16} {'E/atom [eV]':>14} "
          f"{'|F|max [eV/A]':>14}")
    print("-" * 78)
    records = []
    for index, (atoms, energy, force) in enumerate(zip(frames, energies, forces, strict=True)):
        fmax = float(np.linalg.norm(force, axis=1).max()) if len(force) else 0.0
        formula = atoms.get_chemical_formula()
        print(
            f"{index:>5}  {formula:<16} {len(atoms):>6} {energy:>16.6f} "
            f"{energy / len(atoms):>14.6f} {fmax:>14.6f}"
        )
        records.append(
            {
                "index": index,
                "formula": formula,
                "num_atoms": len(atoms),
                "energy": energy,
                "energy_per_atom": energy / len(atoms),
                "max_force": fmax,
                "forces": force.tolist(),
            }
        )
        if args.print_forces:
            for atom_index, (symbol, vector) in enumerate(
                zip(atoms.get_chemical_symbols(), force, strict=True)
            ):
                print(
                    f"        {atom_index:>4} {symbol:<3} "
                    f"{vector[0]:>12.6f} {vector[1]:>12.6f} {vector[2]:>12.6f}"
                )

    if args.evaluate:
        _report_errors(frames, energies, forces)

    # ------------------------------------------------------------- output
    if args.output:
        out_frames = []
        for atoms, energy, force in zip(frames, energies, forces, strict=True):
            copy = atoms.copy()
            copy.calc = SinglePointCalculator(copy, energy=energy, forces=force)
            out_frames.append(copy)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write(args.output, out_frames, format="extxyz")
        logger.info("predictions written to %s", args.output)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        logger.info("per-structure results written to %s", args.json_out)

    return 0


def _report_errors(frames, energies, forces) -> None:
    """Compare predictions with whatever labels the input file carries."""
    e_abs, e_sq, n_e = 0.0, 0.0, 0
    f_abs, f_sq, n_f = 0.0, 0.0, 0
    for atoms, energy, force in zip(frames, energies, forces, strict=True):
        ref_energy, ref_forces = extract_labels(atoms)
        if ref_energy is not None:
            error = (energy - ref_energy) / len(atoms)
            e_abs += abs(error)
            e_sq += error**2
            n_e += 1
        if ref_forces is not None:
            error_f = force - np.asarray(ref_forces)
            f_abs += float(np.abs(error_f).sum())
            f_sq += float(np.square(error_f).sum())
            n_f += error_f.size

    if not n_e and not n_f:
        logger.warning("--evaluate requested but the input carries no reference labels")
        return
    print("\nerror against reference labels")
    print("-" * 78)
    if n_e:
        print(
            f"  energy   MAE = {1000 * e_abs / n_e:10.3f} meV/atom     "
            f"RMSE = {1000 * math.sqrt(e_sq / n_e):10.3f} meV/atom  ({n_e} structures)"
        )
    if n_f:
        print(
            f"  forces   MAE = {1000 * f_abs / n_f:10.3f} meV/A        "
            f"RMSE = {1000 * math.sqrt(f_sq / n_f):10.3f} meV/A     ({n_f} components)"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
