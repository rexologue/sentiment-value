"""Configuration utilities for metric + classifier training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import yaml


@dataclass
class OptimizerConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    warmup_steps: int = 0
    num_training_steps: Optional[int] = None
    name: str = "linear"
    num_cycles: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 256
    mixed_precision: bool = False
    gradient_clip_val: Optional[float] = None
    seed: int = 42
    resume_from: Optional[str] = None
    attention_implementation: Optional[str] = "flash_attention_2"
    supcon_temperature: float = 0.1
    classification_loss_weight: float = 1.0
    metric_loss_weight: float = 1.0


@dataclass
class NeptuneConfig:
    project: Optional[str] = None
    api_token: Optional[str] = None
    experiment_name: Optional[str] = None
    run_id: Optional[str] = None
    tags: Optional[list[str]] = None
    dependencies_path: Optional[str] = None
    env_path: Optional[str] = None


@dataclass
class DataConfig:
    parquet_path: str = "data/train.parquet"
    valid_parquet_path: Optional[str] = None
    val_ratio: float = 0.1
    num_workers: int = 0
    upsample: bool = False
    downsample: bool = False


@dataclass
class CheckpointConfig:
    checkpoints_dir: str = "checkpoints"
    save_every_n_bathces: int = 500
    save_best_by: str = "loss"


@dataclass
class MetricValidationConfig:
    recall_at_k: List[int] | None = None
    distance: str = "cos"
    knn_k: int = 10
    keep_last_n_emb_steps: int = 10


@dataclass
class Config:
    model_name: str
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    neptune: NeptuneConfig
    checkpointing: CheckpointConfig
    metric_validation: MetricValidationConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def _load(section: str, cls):
        values = raw.get(section, {})
        return cls(**values)

    model_name = raw.get("model_name", "jhu-clsp/mmBERT-base")
    data = _load("data", DataConfig)
    training = _load("training", TrainingConfig)
    optimizer = _load("optimizer", OptimizerConfig)
    scheduler = _load("scheduler", SchedulerConfig)
    neptune = _load("neptune", NeptuneConfig)
    checkpointing = _load("checkpointing", CheckpointConfig)
    metric_validation = _load("metric_validation", MetricValidationConfig)

    if metric_validation.recall_at_k is None:
        metric_validation.recall_at_k = [1, 10, 100]

    if training.supcon_temperature < 0.0:
        raise ValueError("supcon_temperature must be non-negative")

    if checkpointing.save_best_by not in {"loss", "acc", "f1"}:
        raise ValueError("save_best_by must be one of {'loss', 'acc', 'f1'}")

    return Config(
        model_name=model_name,
        data=data,
        training=training,
        optimizer=optimizer,
        scheduler=scheduler,
        neptune=neptune,
        checkpointing=checkpointing,
        metric_validation=metric_validation,
    )


__all__ = [
    "Config",
    "DataConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "NeptuneConfig",
    "CheckpointConfig",
    "MetricValidationConfig",
    "load_config",
]
