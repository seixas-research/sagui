"""Training loop for the generative (diffusion) task.

Structurally the same as the potential trainer -- the differences are that the
corruption happens inside the dataset, that there is no notion of a "reference
force" to score against, and that the reported metrics are the three denoising
losses rather than physical errors.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..config import Config
from ..data.dataset import random_split, read_structures
from ..data.statistics import GenerativeStatistics
from ..data.ztable import ZTable
from ..generative.dataset import DiffusionDataset, collate_diffusion
from ..generative.diffusion import MaterialsDiffusion
from ..generative.structures import graph_from_arrays, lattice_length_scale
from ..utils import clip_gradients, count_parameters, resolve_device_and_dtype, set_seed
from .ema import ExponentialMovingAverage

__all__ = ["run_diffusion_training", "compute_generative_statistics", "evaluate_diffusion"]

logger = logging.getLogger(__name__)


def compute_generative_statistics(
    frames: Sequence[Atoms],
    z_table: ZTable,
    r_max: float,
    max_neighbors: int = 24,
    max_samples: int = 200,
) -> GenerativeStatistics:
    """Lattice unit, mean coordination and the empirical distribution of sizes."""
    if not frames:
        raise ValueError("cannot compute statistics on an empty dataset")

    scales, sizes = [], []
    for atoms in frames:
        sizes.append(len(atoms))
        scales.append(lattice_length_scale(atoms.get_cell().array, len(atoms)))
    lattice_scale = float(np.mean(scales))
    if not np.isfinite(lattice_scale) or lattice_scale <= 0.0:
        raise ValueError("training cells have zero volume; the generative task needs real cells")

    stride = max(1, len(frames) // max_samples)
    edges = nodes = 0
    for atoms in frames[::stride]:
        graph = graph_from_arrays(
            torch.as_tensor(atoms.get_scaled_positions(), dtype=torch.get_default_dtype()),
            torch.as_tensor(atoms.get_cell().array, dtype=torch.get_default_dtype()),
            torch.as_tensor(z_table.indices(atoms.get_atomic_numbers())),
            r_max=r_max,
            max_neighbors=max_neighbors,
        )
        edges += graph.num_edges
        nodes += graph.num_nodes

    return GenerativeStatistics(
        lattice_scale=lattice_scale,
        avg_num_neighbors=float(edges / nodes) if nodes else 1.0,
        num_atoms=sizes,
    )


def _run_epoch(
    model: MaterialsDiffusion,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """One pass over the loader; training if an optimiser is given."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = skipped = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            loss, terms = model.loss(batch)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not torch.isfinite(clip_gradients(model, max_grad_norm)):
                    # See sagui.utils.clip_gradients: stepping here would turn
                    # every parameter, and the EMA, into NaN for good.
                    skipped += 1
                    continue
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1

    if skipped:
        if not batches:
            raise RuntimeError(
                f"every one of the {skipped} batches in this epoch produced a non-finite "
                "gradient; training has diverged"
            )
        logger.warning("skipped %d batch(es) with non-finite gradients this epoch", skipped)
    return {key: value / max(batches, 1) for key, value in totals.items()}


def evaluate_diffusion(
    model: MaterialsDiffusion, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Average denoising losses over a loader.

    Note that the corruption is redrawn every epoch, so this estimate is noisy
    by construction -- it is a Monte-Carlo estimate of an expectation over
    ``t`` and over the noise, not a deterministic score.
    """
    return _run_epoch(model, loader, device)


def _format(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix} loss={metrics.get('loss', float('nan')):.5f} "
        f"(types={metrics.get('types', float('nan')):.4f} "
        f"coords={metrics.get('coords', float('nan')):.4f} "
        f"lattice={metrics.get('lattice', float('nan')):.4f})"
    )


def run_diffusion_training(config: Config) -> Path:
    """Train a generative diffusion model; returns the best checkpoint path."""
    training = config.training
    set_seed(training.seed)
    device, dtype = resolve_device_and_dtype(training.device, training.default_dtype)
    torch.set_default_dtype(dtype)

    output_dir = Path(training.output_dir) / training.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    if not config.data.train_file:
        raise ValueError("data.train_file is required")
    frames = read_structures(config.data.train_file)
    if config.data.valid_file:
        train_frames, valid_frames = frames, read_structures(config.data.valid_file)
    else:
        train_frames, valid_frames = random_split(
            frames, config.data.valid_fraction, seed=training.seed
        )
    logger.info("training on %s in %s", device, str(dtype).replace("torch.", ""))
    logger.info(
        "loaded %d training and %d validation structures", len(train_frames), len(valid_frames)
    )

    z_table = ZTable.from_atoms([*train_frames, *valid_frames])
    logger.info("species: %s", ", ".join(z_table.symbols))

    stats = compute_generative_statistics(
        train_frames, z_table, config.model.r_max, config.diffusion.max_neighbors
    )
    logger.info("%r", stats)

    model = MaterialsDiffusion(
        config.model,
        config.diffusion,
        num_species=len(z_table),
        lattice_scale=stats.lattice_scale,
        avg_num_neighbors=stats.avg_num_neighbors,
    ).to(device)
    logger.info(
        "diffusion model over %d timesteps ('%s' type kernel) with %d trainable parameters",
        config.diffusion.num_steps,
        config.diffusion.type_transition,
        count_parameters(model),
    )

    dataset_kwargs = dict(
        z_table=z_table,
        # DiffusionDataset keeps its own CPU copy: the model's buffers are on
        # the accelerator, while corruption runs on CPU in the loader workers.
        corruption=model.corruption,
        r_max=config.model.r_max,
        lattice_scale=stats.lattice_scale,
        max_neighbors=config.diffusion.max_neighbors,
        rotation_augmentation=config.diffusion.rotation_augmentation,
        dtype=dtype,
        seed=training.seed,
    )
    train_loader = DataLoader(
        DiffusionDataset(train_frames, **dataset_kwargs),
        batch_size=training.batch_size,
        shuffle=True,
        collate_fn=collate_diffusion,
        num_workers=config.data.num_workers,
    )
    valid_loader = (
        DataLoader(
            DiffusionDataset(valid_frames, **dataset_kwargs),
            batch_size=training.valid_batch_size or training.batch_size,
            shuffle=False,
            collate_fn=collate_diffusion,
            num_workers=config.data.num_workers,
        )
        if valid_frames
        else None
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training.lr_factor,
        patience=training.lr_patience,
        min_lr=training.min_lr,
    )
    ema = (
        ExponentialMovingAverage(model, training.ema_decay)
        if training.ema_decay is not None
        else None
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = output_dir / "best.model"
    last_path = output_dir / "last.model"
    started = time.time()

    for epoch in range(1, training.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer, ema, training.max_grad_norm
        )
        if valid_loader is not None:
            if ema is not None:
                with ema.average_parameters(model):
                    valid_metrics = evaluate_diffusion(model, valid_loader, device)
            else:
                valid_metrics = evaluate_diffusion(model, valid_loader, device)
            monitored = valid_metrics.get("loss", float("inf"))
        else:
            valid_metrics = {}
            monitored = train_metrics.get("loss", float("inf"))
        scheduler.step(monitored)

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(record)
        if epoch % max(1, training.log_every) == 0 or epoch == training.epochs:
            message = f"epoch {epoch:4d}/{training.epochs} | " + _format("train", train_metrics)
            if valid_metrics:
                message += " || " + _format("valid", valid_metrics)
            logger.info(message)

        weights = ema.shadow if ema is not None else None
        save_checkpoint(
            last_path, model, config, z_table, stats, epoch=epoch, metrics=record,
            state_dict=weights,
        )
        if monitored < best_loss:
            best_loss = monitored
            save_checkpoint(
                best_path, model, config, z_table, stats, epoch=epoch, metrics=record,
                state_dict=weights,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    logger.info(
        "training finished in %.1f s; best objective %.6f", time.time() - started, best_loss
    )
    logger.info("best model: %s", best_path)
    return best_path
