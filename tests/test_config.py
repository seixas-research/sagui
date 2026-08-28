"""Configuration parsing, overrides and error reporting."""

from __future__ import annotations

import pytest
import yaml

from sagui.config import Config, DataConfig, DiffusionConfig, ModelConfig, TrainingConfig


def test_defaults_are_a_usable_potential_configuration():
    config = Config()
    assert config.task == "potential"
    assert config.model.type == "mace"
    assert config.model.sh_lmax is None, "sh_lmax stays lazy so lmax overrides propagate"
    assert config.model.spherical_lmax == config.model.lmax


def test_yaml_round_trip(tmp_path):
    config = Config()
    config.model.channels = 48
    config.diffusion.num_steps = 250
    path = tmp_path / "config.yaml"
    config.to_yaml(path)

    reloaded = Config.from_yaml(path)
    assert reloaded.model.channels == 48
    assert reloaded.diffusion.num_steps == 250
    assert reloaded.to_dict() == config.to_dict()


def test_from_dict_reads_every_section():
    config = Config.from_dict(
        {
            "task": "generative",
            "model": {"type": "strictly_local", "lmax": 1, "channels": 16},
            "data": {"train_file": "train.xyz"},
            "training": {"epochs": 5},
            "diffusion": {"type_transition": "uniform"},
        }
    )
    assert config.task == "generative"
    assert isinstance(config.model, ModelConfig) and config.model.type == "strictly_local"
    assert isinstance(config.data, DataConfig) and config.data.train_file == "train.xyz"
    assert isinstance(config.training, TrainingConfig) and config.training.epochs == 5
    assert isinstance(config.diffusion, DiffusionConfig)
    assert config.diffusion.type_transition == "uniform"


def test_unknown_keys_are_rejected_with_a_helpful_message():
    with pytest.raises(ValueError, match="unknown key"):
        Config.from_dict({"model": {"chanels": 32}})
    with pytest.raises(ValueError, match="unknown configuration section"):
        Config.from_dict({"modle": {}})


def test_invalid_task_is_rejected():
    with pytest.raises(ValueError, match="task must be one of"):
        Config(task="regression")


def test_overrides_coerce_command_line_strings():
    config = Config().apply_overrides(
        {
            "model.channels": "64",
            "model.r_max": "6.5",
            "training.ema_decay": "0.995",
            "data.cache_graphs": "true",
            "task": "generative",
        }
    )
    assert config.model.channels == 64 and isinstance(config.model.channels, int)
    assert config.model.r_max == 6.5 and isinstance(config.model.r_max, float)
    assert config.training.ema_decay == 0.995
    assert config.data.cache_graphs is True
    assert config.task == "generative"


def test_overrides_can_clear_optional_values():
    config = Config().apply_overrides({"training.ema_decay": "none"})
    assert config.training.ema_decay is None


def test_overrides_ignore_none_and_reject_nonsense():
    config = Config()
    config.apply_overrides({"model.channels": None})  # unset CLI flag: no-op
    assert config.model.channels == ModelConfig().channels
    with pytest.raises(ValueError, match="unknown key"):
        config.apply_overrides({"model.nope": "1"})
    with pytest.raises(ValueError, match="section.key"):
        config.apply_overrides({"channels": "8"})


def test_sh_lmax_follows_lmax_through_overrides():
    config = Config().apply_overrides({"model.lmax": "3"})
    assert config.model.spherical_lmax == 3


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="configuration file not found"):
        Config.from_yaml(tmp_path / "absent.yaml")


def test_shipped_example_configs_parse():
    """The examples are documentation; they must stay loadable."""
    import pathlib

    for path in sorted(pathlib.Path("examples").glob("config_*.yaml")):
        config = Config.from_yaml(path)
        assert config.model.type in {"mace", "strictly_local"}
        assert yaml.safe_load(path.read_text()) is not None
