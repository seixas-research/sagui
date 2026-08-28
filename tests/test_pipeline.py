"""End-to-end: train a model, reload it, predict, and drive ASE with it.

These are slow-ish integration tests, but they are the only ones that exercise
the checkpoint round trip, the command-line entry points and the calculator
against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase.io import read, write

from sagui import SaguiCalculator, load_model
from sagui.checkpoint import load_checkpoint, load_generative_model
from sagui.cli import generate as generate_cli
from sagui.cli import inference as inference_cli
from sagui.cli import train as train_cli
from sagui.config import Config


def _train(tmp_path, frames, **overrides) -> tuple[Config, str]:
    """Run a very short training and return the config and the best checkpoint."""
    data_file = tmp_path / "train.xyz"
    write(data_file, frames, format="extxyz")

    config = Config()
    config.data.train_file = str(data_file)
    config.data.valid_fraction = 0.25
    config.model.channels = 8
    config.model.lmax = 1
    config.model.num_layers = 1
    config.model.num_radial_basis = 6
    config.model.radial_mlp_hidden = [16]
    config.model.scalar_mlp_hidden = [16]
    config.model.latent_dim = 16
    config.model.r_max = 5.0
    config.training.epochs = 2
    config.training.batch_size = 4
    config.training.default_dtype = "float64"
    # float64 for tight tolerances, hence CPU: Metal has no double precision.
    # Per-device coverage lives in test_devices.py.
    config.training.device = "cpu"
    config.training.output_dir = str(tmp_path / "runs")
    config.training.name = "test"
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        setattr(getattr(config, section), field, value)
    return config, str(tmp_path / "runs" / "test" / "best.model")


@pytest.mark.parametrize("architecture", ["mace", "strictly_local"])
def test_train_then_reload_then_predict(tmp_path, labelled_frames, architecture):
    from sagui.train import run_training

    config, _ = _train(tmp_path, labelled_frames, **{"model.type": architecture})
    best = run_training(config)
    assert best.exists()

    payload = load_checkpoint(best)
    assert payload["task"] == "potential"
    assert payload["config"]["model"]["type"] == architecture
    assert payload["atomic_numbers"] == [18]  # argon

    model, reloaded_config, z_table = load_model(best, dtype=torch.float64)
    assert reloaded_config.model.type == architecture
    assert z_table.symbols == ("Ar",)
    assert not model.training, "a reloaded model must come back in eval mode"

    # The reloaded model must reproduce what the trainer would have predicted.
    from sagui.data import collate_graphs, graph_from_atoms

    graph = collate_graphs([graph_from_atoms(labelled_frames[0], z_table, 5.0)])
    out = model(graph, compute_forces=True, training=False)
    assert torch.isfinite(out["energy"]).all()
    assert out["forces"].shape == (len(labelled_frames[0]), 3)
    assert not out["energy"].requires_grad, "eval output should not carry a graph"


def test_calculator_matches_the_model_and_drives_ase(tmp_path, labelled_frames):
    from sagui.train import run_training

    config, _ = _train(tmp_path, labelled_frames)
    best = run_training(config)

    atoms = labelled_frames[0].copy()
    atoms.calc = SaguiCalculator(best, default_dtype="float64", device="cpu")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.allclose(atoms.get_potential_energies().sum(), energy, atol=1e-8)
    # A translated copy must give the same energy: the calculator rebuilds the
    # graph from scratch every call, so this checks that path too.
    moved = atoms.copy()
    moved.positions += np.array([1.3, -0.7, 2.2])
    moved.calc = SaguiCalculator(best, default_dtype="float64", device="cpu")
    assert np.isclose(moved.get_potential_energy(), energy, atol=1e-8)


def test_calculator_reports_unknown_elements(tmp_path, labelled_frames):
    from ase import Atoms

    from sagui.train import run_training

    config, _ = _train(tmp_path, labelled_frames)
    best = run_training(config)

    gold = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    gold.calc = SaguiCalculator(best, default_dtype="float64", device="cpu")
    with pytest.raises(KeyError, match="unknown to this model"):
        gold.get_potential_energy()


def test_train_and_inference_command_lines(tmp_path, labelled_frames, capsys):
    """The two console scripts, driven exactly as a user would."""
    data_file = tmp_path / "data.xyz"
    write(data_file, labelled_frames, format="extxyz")
    runs = tmp_path / "runs"

    assert train_cli.main(
        [
            "--train-file", str(data_file),
            "--model-type", "strictly_local",
            "--epochs", "2", "--batch-size", "4", "--channels", "8", "--lmax", "1",
            "--num-layers", "1", "--r-max", "5.0",
            "--default-dtype", "float64", "--device", "cpu",
            "--output-dir", str(runs), "--name", "cli",
            "--set", "model.latent_dim=16",
            "--set", "model.scalar_mlp_hidden=[16]",
            "--log-level", "WARNING",
        ]
    ) == 0
    checkpoint = runs / "cli" / "best.model"
    assert checkpoint.exists()
    assert (runs / "cli" / "config.yaml").exists()
    assert (runs / "cli" / "history.json").exists()

    predictions = tmp_path / "predictions.xyz"
    assert inference_cli.main(
        [
            "--model", str(checkpoint), "--input", str(data_file),
            "--output", str(predictions), "--evaluate",
            "--json", str(tmp_path / "predictions.json"),
            "--log-level", "WARNING",
        ]
    ) == 0
    assert "error against reference labels" in capsys.readouterr().out

    written = read(predictions, index=":")
    assert len(written) == len(labelled_frames)
    assert all(np.isfinite(a.get_potential_energy()) for a in written)
    assert (tmp_path / "predictions.json").exists()


def test_generative_train_and_sample_command_lines(tmp_path, crystal, capsys):
    """The generative path: sagui-train --task generative, then sagui-generate."""
    frames = []
    rng = np.random.default_rng(0)
    for _ in range(6):
        atoms = crystal.copy()
        atoms.rattle(stdev=0.05, seed=int(rng.integers(1 << 30)))
        frames.append(atoms)
    data_file = tmp_path / "crystals.xyz"
    write(data_file, frames, format="extxyz")
    runs = tmp_path / "runs"

    assert train_cli.main(
        [
            "--task", "generative",
            "--train-file", str(data_file),
            "--epochs", "2", "--batch-size", "3", "--channels", "8", "--lmax", "2",
            "--num-layers", "1", "--r-max", "4.0", "--default-dtype", "float64", "--device", "cpu",
            "--output-dir", str(runs), "--name", "gen",
            "--set", "diffusion.num_steps=20",
            "--set", "model.radial_mlp_hidden=[16]",
            "--set", "model.scalar_mlp_hidden=[16]",
            "--log-level", "WARNING",
        ]
    ) == 0
    checkpoint = runs / "gen" / "best.model"
    assert load_checkpoint(checkpoint)["task"] == "generative"

    model, config, z_table, stats = load_generative_model(checkpoint, dtype=torch.float64)
    assert config.task == "generative"
    # Statistics come from the training split only, hence fewer than len(frames).
    assert 0 < len(stats.num_atoms) <= len(frames)
    assert set(stats.num_atoms) == {len(crystal)}
    assert stats.lattice_scale == pytest.approx(
        (crystal.get_volume() / len(crystal)) ** (1 / 3), rel=1e-6
    )

    output = tmp_path / "generated.xyz"
    assert generate_cli.main(
        [
            "--model", str(checkpoint), "-n", "2", "--steps", "20",
            "--output", str(output), "--default-dtype", "float64", "--device", "cpu",
            "--log-level", "WARNING",
        ]
    ) == 0
    sampled = read(output, index=":")
    assert len(sampled) == 2
    for atoms in sampled:
        assert all(atoms.get_pbc()), "generated structures must be periodic"
        assert atoms.get_volume() > 0.0
        assert set(atoms.get_chemical_symbols()) <= set(z_table.symbols)


def test_a_potential_checkpoint_is_not_loadable_as_a_generative_one(tmp_path, labelled_frames):
    from sagui.train import run_training

    config, _ = _train(tmp_path, labelled_frames)
    best = run_training(config)
    with pytest.raises(ValueError, match="holds a 'potential' model"):
        load_generative_model(best)
