r"""Dataset statistics that set the model's physical scales.

A network initialised with unit-variance weights predicts numbers of order
one, while total energies are of order ``-10^3`` eV.  Rather than asking the
optimiser to discover that offset, we measure it:

* **atomic reference energies** :math:`E^{(0)}_Z` from the least-squares fit
  :math:`\min_{E^{(0)}} \sum_s \big(E_s - \sum_Z N_{sZ} E^{(0)}_Z\big)^2`, i.e.
  the best composition-only model.  What remains for the network is the
  *binding* energy, which is small and roughly zero-centred;
* an **energy scale** taken as the RMS force of the training set, so the
  gradient of the (scaled) network output starts at the right magnitude;
* the **average number of neighbours**, used to normalise sums over neighbours
  so that activations do not grow with density.

Compositions and labels are read straight from the ``Atoms`` objects, which is
cheap; only the connectivity statistics need neighbour lists, and those are
estimated from a subsample.
"""

from __future__ import annotations

import logging

import numpy as np

from ..data.atomic_data import extract_labels
from ..data.dataset import AtomsDataset
from ..data.statistics import DatasetStatistics

__all__ = ["DatasetStatistics", "compute_statistics"]

logger = logging.getLogger(__name__)


def _fit_atomic_energies(
    counts: np.ndarray, energies: np.ndarray, num_species: int
) -> np.ndarray:
    """Least-squares per-element reference energies from composition alone."""
    if len(energies) == 0:
        return np.zeros(num_species)
    if len(energies) < num_species or np.linalg.matrix_rank(counts) < num_species:
        # Under-determined (e.g. every frame has the same composition):
        # a single mean energy per atom is the best we can say.
        mean = float(energies.sum() / max(counts.sum(), 1.0))
        logger.info("composition matrix is rank deficient; using a uniform E0 = %.6f", mean)
        return np.full(num_species, mean)
    solution, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return np.asarray(solution, dtype=float)


def compute_statistics(
    dataset: AtomsDataset,
    max_samples: int | None = 500,
    fit_atomic_energies: bool = True,
) -> DatasetStatistics:
    """Measure reference energies, energy scale and neighbour count.

    ``max_samples`` bounds how many neighbour lists are built for the
    connectivity estimate; the composition fit always uses every labelled
    structure.
    """
    n_total = len(dataset)
    if n_total == 0:
        raise ValueError("cannot compute statistics on an empty dataset")
    num_species = len(dataset.z_table)

    # --- composition, energies and force magnitudes (no neighbour lists) ---
    counts_rows, energies, force_sq, n_force = [], [], 0.0, 0
    for atoms in dataset.frames:
        energy, forces = extract_labels(atoms, dataset.energy_key, dataset.forces_key)
        if energy is not None:
            row = np.zeros(num_species)
            for z in atoms.get_atomic_numbers():
                row[dataset.z_table.index(z)] += 1.0
            counts_rows.append(row)
            energies.append(energy)
        if forces is not None:
            force_sq += float(np.sum(np.square(forces)))
            n_force += int(np.size(forces))

    counts = np.asarray(counts_rows) if counts_rows else np.zeros((0, num_species))
    energies_arr = np.asarray(energies, dtype=float)

    atomic_energies = (
        _fit_atomic_energies(counts, energies_arr, num_species)
        if fit_atomic_energies
        else np.zeros(num_species)
    )

    if n_force > 0:
        energy_scale = float(np.sqrt(force_sq / n_force))
    elif len(energies_arr) > 1:
        residual = energies_arr - counts @ atomic_energies
        energy_scale = float(np.std(residual / np.maximum(counts.sum(axis=1), 1.0)))
    else:
        energy_scale = 1.0
    if not np.isfinite(energy_scale) or energy_scale < 1e-8:
        logger.warning("degenerate energy scale (%.3e); falling back to 1.0", energy_scale)
        energy_scale = 1.0

    # --- connectivity, from a strided subsample of the graphs ---
    stride = 1 if max_samples is None else max(1, n_total // max_samples)
    edges = nodes = 0
    for index in range(0, n_total, stride):
        graph = dataset[index]
        edges += graph.num_edges
        nodes += graph.num_nodes
    avg_num_neighbors = float(edges / nodes) if nodes else 0.0
    if avg_num_neighbors < 1e-6:
        logger.warning(
            "no neighbours found within r_max=%.2f A; the model cannot learn anything",
            dataset.r_max,
        )
        avg_num_neighbors = 1.0

    return DatasetStatistics(
        atomic_energies=atomic_energies,
        energy_scale=energy_scale,
        avg_num_neighbors=avg_num_neighbors,
    )
