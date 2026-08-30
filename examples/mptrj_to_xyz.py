#!/usr/bin/env python
"""Extract a portion of the MPtrj dataset into an ASE-readable file.

MPtrj (the Materials Project trajectory dataset, ~1.6 M DFT frames used to
train CHGNet) ships as a *single* JSON object of roughly 11 GB:

    {"mp-1005792": {"mp-1012897-0-0": {"structure": {...}, "force": [...], ...},
                    "mp-1012897-0-1": {...}},
     "mp-1005794": {...}, ...}

``json.load`` would need tens of gigabytes of RAM, so this script streams it:
it keeps a bounded buffer, extracts complete ``"material": {...}`` entries as
they arrive with a string-aware brace matcher, and stops as soon as it has
collected the requested number of frames.  Reading a few thousand frames
touches only the first few tens of megabytes of the file.

    python examples/mptrj_to_xyz.py --input .../MPtrj_2022.9_full.json \
        --output mptrj_2k.xyz --limit 2000 --max-atoms 40

Energies
--------
``uncorrected_total_energy`` is the default: it is the raw DFT total energy,
consistent with the forces in the same record.  The Materials Project
corrections in ``corrected_total_energy`` are per-element constants, which
SAGUI's least-squares reference energies absorb exactly, so the choice does not
affect what the model has to learn -- only the meaning of the absolute number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.stress import full_3x3_to_voigt_6_stress

#: 1 GPa in eV/A^3.
GPA_TO_EV_PER_A3 = 160.21766208

#: Read granularity. Large enough that a single material always fits.
CHUNK = 8 << 20


def _match_brace(buffer: str, start: int) -> int | None:
    """Index just past the ``}`` matching the ``{`` at ``buffer[start]``.

    String-aware, so braces inside keys or values do not confuse the depth
    count.  Returns ``None`` when the buffer ends mid-object, which is the
    signal to read another chunk.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(buffer)):
        char = buffer[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def stream_materials(path: Path) -> Iterator[tuple[str, dict]]:
    """Yield ``(material_id, frames)`` pairs without loading the whole file."""
    with path.open("r", encoding="utf-8") as handle:
        buffer = handle.read(CHUNK)
        if not buffer:
            return
        cursor = buffer.index("{") + 1  # step inside the outer object

        while True:
            # Find the next key; refill the buffer if we ran out of text.
            while True:
                start = buffer.find('"', cursor)
                if start != -1:
                    break
                chunk = handle.read(CHUNK)
                if not chunk:
                    return
                buffer += chunk

            end_quote = buffer.find('"', start + 1)
            colon = buffer.find("{", end_quote)
            while end_quote == -1 or colon == -1:
                chunk = handle.read(CHUNK)
                if not chunk:
                    return
                buffer += chunk
                end_quote = buffer.find('"', start + 1)
                colon = buffer.find("{", end_quote)

            material_id = buffer[start + 1 : end_quote]
            stop = _match_brace(buffer, colon)
            while stop is None:
                chunk = handle.read(CHUNK)
                if not chunk:
                    return
                buffer += chunk
                stop = _match_brace(buffer, colon)

            try:
                frames = json.loads(buffer[colon:stop])
            except json.JSONDecodeError:
                return
            yield material_id, frames

            # Drop what we consumed so the buffer stays bounded.
            buffer = buffer[stop:]
            cursor = 0


def to_atoms(record: dict, energy_key: str) -> Atoms | None:
    """Convert one MPtrj frame into an ``ase.Atoms`` with labels attached.

    Returns ``None`` for records this pipeline cannot represent: partially
    occupied or disordered sites, or a missing energy/force label.
    """
    structure = record.get("structure")
    energy = record.get(energy_key)
    forces = record.get("force")
    if structure is None or energy is None or forces is None:
        return None

    symbols = []
    for site in structure["sites"]:
        species = site["species"]
        if len(species) != 1 or abs(species[0].get("occu", 1) - 1.0) > 1e-8:
            return None  # fractional occupancy has no place in a point cloud
        symbols.append(species[0]["element"])

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=np.array([site["abc"] for site in structure["sites"]], dtype=float),
        cell=np.array(structure["lattice"]["matrix"], dtype=float),
        pbc=True,
    )
    forces = np.array(forces, dtype=float)
    if forces.shape != (len(atoms), 3):
        return None

    labels = {"energy": float(energy), "forces": forces}
    magmom = record.get("magmom")
    if magmom is not None:
        magmom = np.asarray(magmom, dtype=float).reshape(-1)
        if magmom.shape == (len(atoms),):
            # Collinear moments, one per site; SAGUI reads them from `arrays`.
            atoms.arrays["magmoms"] = magmom
    stress = record.get("stress")
    if stress is not None:
        # MPtrj carries the VASP stress in kBar.  ASE wants eV/A^3 and the
        # opposite sign, so 1 kBar -> -0.1 / 160.21766208 eV/A^3.  The sign is
        # checked empirically by --check-stress-sign: compressed frames must
        # come out with a negative trace.
        matrix = np.array(stress, dtype=float).reshape(3, 3) * (-0.1 / GPA_TO_EV_PER_A3)
        labels["stress"] = full_3x3_to_voigt_6_stress(matrix)
    atoms.calc = SinglePointCalculator(atoms, **labels)
    atoms.info["mp_id"] = record.get("mp_id", "")
    return atoms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="MPtrj_2022.9_full.json")
    parser.add_argument("--output", default="mptrj.xyz")
    parser.add_argument("--limit", type=int, default=2000, help="frames to write")
    parser.add_argument("--max-atoms", type=int, default=40, help="skip larger structures")
    parser.add_argument(
        "--frames-per-material",
        type=int,
        default=2,
        help="cap frames taken from one material; consecutive frames of a "
             "relaxation are highly correlated, so a low value buys diversity",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        help="keep only structures made entirely of these elements",
    )
    parser.add_argument(
        "--energy-key",
        default="uncorrected_total_energy",
        choices=["uncorrected_total_energy", "corrected_total_energy"],
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    allowed = set(args.elements) if args.elements else None

    frames: list[Atoms] = []
    materials = skipped = 0
    for _material_id, records in stream_materials(path):
        materials += 1
        taken = 0
        for record in records.values():
            if len(frames) >= args.limit:
                break
            if taken >= args.frames_per_material:
                break
            atoms = to_atoms(record, args.energy_key)
            if atoms is None:
                skipped += 1
                continue
            if len(atoms) > args.max_atoms:
                continue
            if allowed and not set(atoms.get_chemical_symbols()) <= allowed:
                continue
            frames.append(atoms)
            taken += 1
        if len(frames) >= args.limit:
            break

    if not frames:
        print("error: no frames matched the filters", file=sys.stderr)
        return 1

    write(args.output, frames, format="extxyz")

    sizes = np.array([len(a) for a in frames])
    energies = np.array([a.get_potential_energy() / len(a) for a in frames])
    forces = np.concatenate([a.get_forces() for a in frames])
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    print(f"wrote {len(frames)} frames from {materials} materials to {args.output}")
    print(f"  atoms per structure : {sizes.min()}..{sizes.max()} (mean {sizes.mean():.1f})")
    print(f"  distinct elements   : {len(elements)}")
    print(f"  energy per atom     : {energies.min():.3f}..{energies.max():.3f} eV")
    print(f"  force RMS           : {np.sqrt((forces ** 2).mean()):.4f} eV/A")
    if skipped:
        print(f"  skipped {skipped} unusable record(s) (disorder or missing labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
