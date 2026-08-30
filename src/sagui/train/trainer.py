"""The training loop behind ``sagui-train``."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..checkpoint import save_checkpoint
from ..config import Config
from ..data.atomic_data import collate_graphs
from ..data.dataset import AtomsDataset, random_split, read_structures
from ..data.ztable import ZTable
from ..models.base import InteratomicPotential
from ..models.registry import build_model
from ..utils import clip_gradients, count_parameters, resolve_device_and_dtype, set_seed
from .ema import ExponentialMovingAverage
from .loss import EnergyForcesStressLoss, compute_metrics
from .stats import compute_statistics, split_isolated_atoms

__all__ = ["run_training", "evaluate"]

logger = logging.getLogger(__name__)


def _reduce(totals: dict[str, float]) -> dict[str, float]:
    """Turn accumulated squared/absolute error sums into MAE and RMSE."""
    out: dict[str, float] = {}
    n_struct = totals.get("n_structures", 0.0)
    n_forces = totals.get("n_force_components", 0.0)
    if n_struct:
        if "energy_abs_sum" in totals:
            out["energy_mae"] = totals["energy_abs_sum"] / n_struct
            out["energy_rmse"] = math.sqrt(totals["energy_sq_sum"] / n_struct)
    if n_forces:
        out["forces_mae"] = totals["forces_abs_sum"] / n_forces
        out["forces_rmse"] = math.sqrt(totals["forces_sq_sum"] / n_forces)
    n_stress = totals.get("n_stress_components", 0.0)
    if n_stress:
        out["stress_mae"] = totals["stress_abs_sum"] / n_stress
        out["stress_rmse"] = math.sqrt(totals["stress_sq_sum"] / n_stress)
    if "loss_sum" in totals and totals.get("n_batches"):
        out["loss"] = totals["loss_sum"] / totals["n_batches"]
    return out


def apply_weight_switch(
    loss_fn: EnergyForcesStressLoss,
    optimizer: torch.optim.Optimizer,
    logger: logging.Logger,
    energy_weight: float | None = None,
    forces_weight: float | None = None,
    lr_factor: float = 0.1,
) -> None:
    """Re-weight the loss for the second stage of training.

    The MACE-MP and OMat24 recipes train with ``lambda_F > lambda_E`` for most
    of the run -- forces carry ``3N`` numbers per structure and shape the local
    geometry -- and then invert the ratio for the last stretch to sharpen the
    energies, which are what thermodynamic quantities are read from.

    With no explicit weights the two are simply swapped.  Supplying them lets
    the second stage take any balance, which is what MACE's ``weight_switch_energy_weight``
    and ``weight_switch_forces_weight`` do.
    """
    if energy_weight is None and forces_weight is None:
        energy_weight, forces_weight = loss_fn.forces_weight, loss_fn.energy_weight
    if energy_weight is not None:
        loss_fn.energy_weight = float(energy_weight)
    if forces_weight is not None:
        loss_fn.forces_weight = float(forces_weight)
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * float(lr_factor)
    logger.info(
        "stage two: energy weight %.4g, force weight %.4g, lr %.3g",
        loss_fn.energy_weight,
        loss_fn.forces_weight,
        optimizer.param_groups[0]["lr"],
    )


def evaluate(
    model: InteratomicPotential,
    loader: DataLoader,
    loss_fn: EnergyForcesStressLoss,
    device: torch.device,
) -> dict[str, float]:
    """Error metrics over a whole loader (no parameter updates)."""
    model.eval()
    totals: dict[str, float] = {}
    for batch in loader:
        batch = batch.to(device)
        prediction = model(
            batch, compute_forces=True, compute_stress=loss_fn.wants_stress, training=False
        )
        _, terms = loss_fn(prediction, batch)
        contributions = compute_metrics(prediction, batch)
        contributions["loss_sum"] = terms["loss"]
        contributions["n_batches"] = 1.0
        for key, value in contributions.items():
            totals[key] = totals.get(key, 0.0) + value
    return _reduce(totals)


def _train_epoch(
    model: InteratomicPotential,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: EnergyForcesStressLoss,
    device: torch.device,
    ema: ExponentialMovingAverage | None,
    max_grad_norm: float | None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    skipped = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch, compute_forces=True, compute_stress=loss_fn.wants_stress, training=True
        )
        loss, terms = loss_fn(prediction, batch)
        loss.backward()

        if not torch.isfinite(clip_gradients(model, max_grad_norm)):
            # Stepping on a non-finite gradient would poison every parameter
            # (and the EMA) permanently.  Drop the batch and carry on.
            skipped += 1
            continue
        optimizer.step()
        if ema is not None:
            ema.update(model)

        contributions = compute_metrics(prediction, batch)
        contributions["loss_sum"] = terms["loss"]
        contributions["n_batches"] = 1.0
        for key, value in contributions.items():
            totals[key] = totals.get(key, 0.0) + value

    if skipped:
        if not totals:
            raise RuntimeError(
                f"every one of the {skipped} batches in this epoch produced a non-finite "
                "gradient; training has diverged (try a smaller learning rate, "
                "float64, or a smaller max_grad_norm)"
            )
        logger.warning(
            "skipped %d batch(es) with non-finite gradients this epoch", skipped
        )
    return _reduce(totals)


def _format(prefix: str, metrics: dict[str, float]) -> str:
    parts = [f"{prefix} loss={metrics.get('loss', float('nan')):.6f}"]
    if "energy_mae" in metrics:
        parts.append(f"E-MAE={metrics['energy_mae'] * 1000:.3f} meV/atom")
    if "forces_mae" in metrics:
        parts.append(f"F-MAE={metrics['forces_mae'] * 1000:.2f} meV/A")
    if "stress_mae" in metrics:
        parts.append(f"S-MAE={metrics['stress_mae'] * 1000:.4f} meV/A^3")
    return " | ".join(parts)


def run_training(config: Config) -> Path:
    """Train a model end to end and return the path of the best checkpoint."""
    training = config.training
    set_seed(training.seed)
    device, dtype = resolve_device_and_dtype(training.device, training.default_dtype)
    torch.set_default_dtype(dtype)

    output_dir = Path(training.output_dir) / training.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    # ------------------------------------------------------------ data
    if not config.data.train_file:
        raise ValueError("data.train_file is required")
    train_frames = read_structures(config.data.train_file)
    train_frames, isolated_energies = split_isolated_atoms(
        train_frames, config.data.isolated_atom_config_type, config.data.energy_key
    )
    if isolated_energies:
        logger.info(
            "found %d isolated-atom reference energies in %s",
            len(isolated_energies),
            config.data.train_file,
        )
    if config.data.valid_file:
        valid_frames = read_structures(config.data.valid_file)
    else:
        train_frames, valid_frames = random_split(
            train_frames, config.data.valid_fraction, seed=training.seed
        )
    # Reference frames must never reach the validation split either.
    valid_frames, _ = split_isolated_atoms(
        valid_frames, config.data.isolated_atom_config_type, config.data.energy_key
    )
    logger.info("training on %s in %s", device, str(dtype).replace("torch.", ""))
    logger.info(
        "loaded %d training and %d validation structures", len(train_frames), len(valid_frames)
    )

    z_table = ZTable.from_atoms([*train_frames, *valid_frames])
    logger.info("species: %s", ", ".join(z_table.symbols))

    dataset_kwargs = dict(
        z_table=z_table,
        r_max=config.model.r_max,
        energy_key=config.data.energy_key,
        forces_key=config.data.forces_key,
        stress_key=config.data.stress_key,
        dtype=dtype,
        cache=config.data.cache_graphs,
    )
    train_set = AtomsDataset(train_frames, **dataset_kwargs)
    valid_set = AtomsDataset(valid_frames, **dataset_kwargs) if valid_frames else None

    stats = compute_statistics(
        train_set,
        fit_atomic_energies=training.fit_atomic_energies,
        isolated_atom_energies=isolated_energies,
    )
    logger.info("%r", stats)

    train_loader = DataLoader(
        train_set,
        batch_size=training.batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    valid_loader = (
        DataLoader(
            valid_set,
            batch_size=training.valid_batch_size or training.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
            num_workers=config.data.num_workers,
        )
        if valid_set is not None and len(valid_set) > 0
        else None
    )

    # ----------------------------------------------------------- model
    model = build_model(
        config.model,
        atomic_numbers=z_table.zs,
        atomic_energies=stats.atomic_energies,
        energy_scale=stats.energy_scale,
        avg_num_neighbors=stats.avg_num_neighbors,
    ).to(device)
    logger.info(
        "architecture '%s' with %d trainable parameters",
        config.model.type,
        count_parameters(model),
    )

    loss_fn = EnergyForcesStressLoss(
        training.energy_weight,
        training.forces_weight,
        training.stress_weight,
        training.charges_weight,
        training.magmoms_weight,
        training.huber_delta,
        training.huber_delta_energy,
        training.huber_delta_forces,
        training.huber_delta_stress,
        scales=(
            {
                "energy": stats.energy_residual_rms,
                "forces": stats.forces_rms,
                "stress": stats.stress_rms,
            }
            if training.normalise_loss_terms
            else None
        ),
    )
    if training.normalise_loss_terms:
        logger.info(
            "loss terms normalised by residual RMS: energy %.4g eV/atom, forces %.4g eV/A, "
            "stress %.4g eV/A^3",
            stats.energy_residual_rms,
            stats.forces_rms,
            stats.stress_rms,
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

    # ------------------------------------------------------------ loop
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_path = output_dir / "best.model"
    last_path = output_dir / "last.model"
    started = time.time()

    switch_epoch = (
        None
        if training.weight_switch is None
        else max(1, int(training.weight_switch * training.epochs))
    )
    if switch_epoch is not None:
        logger.info(
            "stage two begins after epoch %d of %d (weight_switch=%.2f)",
            switch_epoch,
            training.epochs,
            training.weight_switch,
        )

    for epoch in range(1, training.epochs + 1):
        if switch_epoch is not None and epoch == switch_epoch + 1:
            apply_weight_switch(
                loss_fn,
                optimizer,
                logger,
                training.weight_switch_energy_weight,
                training.weight_switch_forces_weight,
                training.weight_switch_lr_factor,
            )
            # The objective itself changed, so every number tracked against the
            # old one is now meaningless.  Without this reset stage two can
            # never beat stage one's best and would never save a checkpoint,
            # and the plateau scheduler would read the jump as a catastrophe.
            best_loss = float("inf")
            scheduler.best = float("inf")
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = 0
        train_metrics = _train_epoch(
            model, train_loader, optimizer, loss_fn, device, ema, training.max_grad_norm
        )

        if valid_loader is not None:
            if ema is not None:
                with ema.average_parameters(model):
                    valid_metrics = evaluate(model, valid_loader, loss_fn, device)
            else:
                valid_metrics = evaluate(model, valid_loader, loss_fn, device)
            monitored = valid_metrics.get("loss", train_metrics.get("loss", float("inf")))
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

    elapsed = time.time() - started
    logger.info(
        "training finished in %.1f s; best objective %.6f%s",
        elapsed,
        best_loss,
        " (stage two)" if switch_epoch is not None and training.epochs > switch_epoch else "",
    )
    logger.info("best model: %s", best_path)
    return best_path
