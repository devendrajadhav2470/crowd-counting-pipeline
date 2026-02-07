"""Configuration management using YAML files and dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "shanghaitech_b"
    root: str = "./data/ShanghaiTech/part_B"
    sigma: int = 15
    test_split: float = 0.1


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "csrnet"
    pretrained_backbone: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 200
    batch_size: int = 1
    lr: float = 1e-6
    momentum: float = 0.95
    weight_decay: float = 5e-4
    seed: int = 42
    num_workers: int = 4
    print_freq: int = 30
    lr_steps: list[int] = field(default_factory=lambda: [-1, 1, 100, 150])
    lr_scales: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    batch_size: int = 1


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "crowd-counting"
    entity: Optional[str] = None


@dataclass
class OutputConfig:
    """Output paths configuration."""

    dir: str = "./outputs"
    checkpoint_dir: str = "./weights"


@dataclass
class Config:
    """Top-level configuration combining all sub-configs."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file, merging with defaults.

    The config file is first merged with `configs/default.yaml` (if it exists),
    then parsed into the Config dataclass.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Fully populated Config object.
    """
    config_path = Path(config_path)

    # Try to load default config
    default_path = config_path.parent / "default.yaml"
    base_dict: dict = {}
    if default_path.exists() and default_path != config_path:
        with open(default_path) as f:
            base_dict = yaml.safe_load(f) or {}

    # Load the specified config
    with open(config_path) as f:
        override_dict = yaml.safe_load(f) or {}

    # Merge: override takes precedence over defaults
    merged = _merge_dicts(base_dict, override_dict)

    # Build Config from merged dict
    config = Config(
        dataset=DatasetConfig(**merged.get("dataset", {})),
        model=ModelConfig(**merged.get("model", {})),
        training=TrainingConfig(**merged.get("training", {})),
        evaluation=EvalConfig(**merged.get("evaluation", {})),
        wandb=WandbConfig(**merged.get("wandb", {})),
        output=OutputConfig(**merged.get("output", {})),
    )

    # Create output directories
    os.makedirs(config.output.dir, exist_ok=True)
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)

    return config

