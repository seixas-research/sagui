"""``sagui-train`` -- fit an interatomic potential to labelled structures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..config import Config
from ..models.registry import available_models
from ..train.generative import run_diffusion_training
from ..train.trainer import run_training
from ..utils import setup_logging
from ..version import __version__

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sagui-train",
        description=(
            "Train an equivariant GNN interatomic potential on any set of "
            "ASE-readable structures carrying energy and/or force labels."
        ),
        epilog=(
            "Every configuration key can be overridden from the command line, e.g.\n"
            "  sagui-train config.yaml --set model.channels=64 --set training.epochs=500"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="YAML configuration file")
    parser.add_argument(
        "--task",
        choices=["potential", "generative"],
        help=(
            "what to fit: an interatomic potential (energies and forces) or a "
            "generative diffusion model of the structures themselves"
        ),
    )
    parser.add_argument("--version", action="version", version=f"sagui {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument("--train-file", help="structures with reference labels (.xyz, .traj, ...)")
    data.add_argument("--valid-file", help="explicit validation set")
    data.add_argument("--valid-fraction", type=float, help="held-out fraction (default 0.1)")
    data.add_argument("--energy-key", help="key holding the reference energy")
    data.add_argument("--forces-key", help="key holding the reference forces")

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model-type", choices=available_models(), help="architecture (default: mace)"
    )
    model.add_argument("--r-max", type=float, help="cutoff radius in Angstrom")
    model.add_argument("--lmax", type=int, help="maximum spherical-harmonic degree")
    model.add_argument("--channels", type=int, help="feature channels per degree")
    model.add_argument("--num-layers", type=int, help="number of layers")
    model.add_argument("--correlation", type=int, help="mace: many-body correlation order")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int)
    optim.add_argument("--batch-size", type=int)
    optim.add_argument("--lr", type=float, dest="learning_rate")
    optim.add_argument("--energy-weight", type=float)
    optim.add_argument("--forces-weight", type=float)
    optim.add_argument("--seed", type=int)
    optim.add_argument("--device", help="auto, cpu, cuda, mps")
    optim.add_argument("--default-dtype", choices=["float32", "float64"])
    optim.add_argument("--output-dir", help="parent directory for runs (default: runs)")
    optim.add_argument("--name", help="run name; artefacts land in <output-dir>/<name>")

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override any configuration entry (repeatable)",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def _collect_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Map the convenience flags onto dotted configuration keys."""
    mapping = {
        "task": args.task,
        "data.train_file": args.train_file,
        "data.valid_file": args.valid_file,
        "data.valid_fraction": args.valid_fraction,
        "data.energy_key": args.energy_key,
        "data.forces_key": args.forces_key,
        "model.type": args.model_type,
        "model.r_max": args.r_max,
        "model.lmax": args.lmax,
        "model.channels": args.channels,
        "model.num_layers": args.num_layers,
        "model.correlation": args.correlation,
        "training.epochs": args.epochs,
        "training.batch_size": args.batch_size,
        "training.learning_rate": args.learning_rate,
        "training.energy_weight": args.energy_weight,
        "training.forces_weight": args.forces_weight,
        "training.seed": args.seed,
        "training.device": args.device,
        "training.default_dtype": args.default_dtype,
        "training.output_dir": args.output_dir,
        "training.name": args.name,
    }
    overrides = {key: value for key, value in mapping.items() if value is not None}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got '{item}'")
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger("sagui-train")

    config = Config.from_yaml(args.config) if args.config else Config()
    try:
        config.apply_overrides(_collect_overrides(args))
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc

    if not config.data.train_file:
        raise SystemExit(
            "no training data: pass --train-file or set data.train_file in the configuration"
        )

    logger.info("sagui %s -- task: %s", __version__, config.task)
    logger.info("training data: %s", Path(config.data.train_file).resolve())

    if config.task == "generative":
        best = run_diffusion_training(config)
        logger.info("done; sample with: sagui-generate --model %s -n 8", best)
    else:
        best = run_training(config)
        logger.info("done; run inference with: sagui-inference --model %s --input <file>", best)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
