"""Configuration helpers for clustering workflows.

This module centralizes YAML-driven configuration loading for the three
clustering scripts (supervise, normalize_and_pca, train). Each script
reads only its own block from the YAML file, avoiding accidental coupling
between different stages of the pipeline.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


def _load_block(config_path: str, block: str) -> Dict[str, Any]:
    """Load a specific configuration block from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
        block: Name of the configuration block to read (e.g. ``"supervise"``).

    Returns:
        The dictionary associated with the requested block. If the block is
        present but empty, an empty dictionary is returned.

    Raises:
        FileNotFoundError: If the YAML file cannot be found.
        ValueError: If the requested block is missing from the YAML file.
    """

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if block not in data:
        raise ValueError(
            f"Configuration block '{block}' is missing from {config_path}."
        )

    block_data = data.get(block) or {}
    if not isinstance(block_data, dict):
        raise ValueError(
            f"Configuration block '{block}' must be a mapping, got {type(block_data).__name__}."
        )

    return block_data


@dataclass
class SuperviseConfig:
    """Configuration for the supervised feature extraction stage."""

    model_name_or_path: str
    input_parquet: str
    output_dir: str
    batch_size: int = 32
    device: str = "cuda"
    num_workers: int = 0
    num_shards: int = 1
    dtype: str = "float16"
    resume: bool = False
    progress: bool = True
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if self.num_shards < 1:
            raise ValueError("num_shards must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be either 'cuda' or 'cpu'")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be one of float16, bfloat16, float32")

    @classmethod
    def from_yaml(cls, config_path: str) -> "SuperviseConfig":
        data = _load_block(config_path, "supervise")
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extras[key] = value

        cfg = cls(**init_kwargs, extra_fields=extras)
        cfg.validate()
        return cfg


@dataclass
class NormalizePCAConfig:
    """Configuration for CLS normalization and PCA transformation."""

    input_dir: str
    output_dir: str
    n_components: int = 50
    batch_size: int = 1024
    device: str = "cpu"
    overwrite: bool = False
    use_gpu_pca: bool = False
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be either 'cuda' or 'cpu'")
        if self.overwrite and not self.output_dir:
            raise ValueError("output_dir must be provided when overwrite is requested")

    @classmethod
    def from_yaml(cls, config_path: str) -> "NormalizePCAConfig":
        data = _load_block(config_path, "normalize_and_pca")
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extras[key] = value

        cfg = cls(**init_kwargs, extra_fields=extras)
        cfg.validate()
        return cfg


VALID_RUN_MODES = {"train", "predict"}


@dataclass
class TrainConfig:
    """Configuration for MiniBatchKMeans training and shard annotation."""

    input_dir: str
    centroids_out: str
    model_out: str
    run_mode: str = "train"
    pca_column: str = "pca"
    centroid_column: str = "cluster_id"
    n_clusters: int = 200
    batch_size: int = 4096
    max_iter: int = 100
    init_size: int = 100_000
    random_state: int = 42
    gpu: bool = False
    num_workers: int = 1
    sample_limit: Optional[int] = None
    progress: bool = True
    log_level: str = "INFO"
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if self.run_mode not in VALID_RUN_MODES:
            raise ValueError(
                f"Invalid run_mode '{self.run_mode}'. Must be one of {sorted(VALID_RUN_MODES)}"
            )
        if self.n_clusters <= 1:
            raise ValueError("n_clusters must be greater than 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.init_size <= 0:
            raise ValueError("init_size must be positive")
        if self.sample_limit is not None and self.sample_limit <= 0:
            raise ValueError("sample_limit must be a positive integer or null")
        if not self.centroid_column:
            raise ValueError("centroid_column must be provided to label shards")
        self.log_level = self.log_level.upper()

    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainConfig":
        data = _load_block(config_path, "train")
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extras[key] = value

        cfg = cls(**init_kwargs, extra_fields=extras)
        cfg.validate()
        return cfg


@dataclass
class PurityCounterConfig:
    """Configuration for computing cluster purity metrics."""

    shards_dir: str
    centroids_path: str
    teacher_label_column: str = "teacher_label"
    cluster_column: str = "cluster_id"
    progress: bool = True
    log_level: str = "INFO"
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if not self.shards_dir:
            raise ValueError("shards_dir must be provided")
        if not self.centroids_path:
            raise ValueError("centroids_path must be provided")
        if not self.teacher_label_column:
            raise ValueError("teacher_label_column must be provided")
        if not self.cluster_column:
            raise ValueError("cluster_column must be provided")
        self.log_level = self.log_level.upper()

    @classmethod
    def from_yaml(cls, config_path: str) -> "PurityCounterConfig":
        data = _load_block(config_path, "purity_counter")
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extras[key] = value

        cfg = cls(**init_kwargs, extra_fields=extras)
        cfg.validate()
        return cfg


@dataclass
class UpdateDatasetConfig:
    """Configuration for building filtered datasets with masks."""

    shards_dir: str
    centroids_path: str
    output_path: str
    metric_purity_threshold: float
    metric_max_prob_threshold: float
    classification_purity_threshold: float
    classification_max_prob_threshold: float
    text_column: str = "text"
    label_column: str = "label"
    cluster_id_column: str = "cluster_id"
    probs_column: str = "probs"
    progress: bool = True
    log_level: str = "INFO"
    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if not self.shards_dir:
            raise ValueError("shards_dir must be provided")
        if not self.centroids_path:
            raise ValueError("centroids_path must be provided")
        if not self.output_path:
            raise ValueError("output_path must be provided")
        if self.metric_purity_threshold < 0 or self.classification_purity_threshold < 0:
            raise ValueError("purity thresholds must be non-negative")
        if self.metric_max_prob_threshold < 0 or self.classification_max_prob_threshold < 0:
            raise ValueError("max probability thresholds must be non-negative")
        if not self.text_column:
            raise ValueError("text_column must be provided")
        if not self.label_column:
            raise ValueError("label_column must be provided")
        if not self.cluster_id_column:
            raise ValueError("cluster_id_column must be provided")
        if not self.probs_column:
            raise ValueError("probs_column must be provided")
        self.log_level = self.log_level.upper()

    @classmethod
    def from_yaml(cls, config_path: str) -> "UpdateDatasetConfig":
        data = _load_block(config_path, "update_dataset")
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}

        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extras[key] = value

        cfg = cls(**init_kwargs, extra_fields=extras)
        cfg.validate()
        return cfg


def parse_config_path(argv: Optional[Any] = None) -> argparse.Namespace:
    """Parse a single ``--config`` argument from the CLI.

    Args:
        argv: Optional custom argv for testing.

    Returns:
        The argparse namespace containing the ``config`` attribute.
    """

    parser = argparse.ArgumentParser(description="Clustering workflow configuration")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args(argv)


__all__ = [
    "SuperviseConfig",
    "NormalizePCAConfig",
    "TrainConfig",
    "PurityCounterConfig",
    "UpdateDatasetConfig",
    "parse_config_path",
]
