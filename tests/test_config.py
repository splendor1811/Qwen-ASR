"""Tests for configuration loading and YAML inheritance."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, ExperimentConfig


def test_load_base_config():
    """Test loading the base config file."""
    config = load_config("configs/base.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.model.name == "Qwen/Qwen3-ASR-1.7B"
    assert config.lora.rank == 64
    assert config.lora.alpha == 128
    assert config.training.bf16 is True
    assert config.training.gradient_checkpointing is True


def test_yaml_inheritance():
    """Test that _base_ inheritance works correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "base.yaml"
        child_path = Path(tmpdir) / "child.yaml"

        base_config = {
            "model": {"name": "base-model"},
            "training": {"learning_rate": 1e-4, "num_train_epochs": 3},
        }
        child_config = {
            "_base_": "base.yaml",
            "training": {"learning_rate": 2e-5},
        }

        with open(base_path, "w") as f:
            yaml.dump(base_config, f)
        with open(child_path, "w") as f:
            yaml.dump(child_config, f)

        config = load_config(child_path)
        assert config.model.name == "base-model"
        assert config.training.learning_rate == 2e-5
        assert config.training.num_train_epochs == 3


def test_phase_configs_load():
    """Test that all phase configs load without error."""
    for config_file in [
        "configs/phase1_large_sft.yaml",
        "configs/phase2_domain_adapt.yaml",
        "configs/phase3_competition.yaml",
    ]:
        if os.path.exists(config_file):
            config = load_config(config_file)
            assert isinstance(config, ExperimentConfig)


def test_default_values():
    """Test that default values are set correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "empty.yaml"
        with open(config_path, "w") as f:
            yaml.dump({}, f)

        config = load_config(config_path)
        assert config.data.sample_rate == 16000
        assert config.freeze.freeze_audio_encoder is True
        assert config.lora.enabled is True
