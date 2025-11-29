from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


VALID_RUN_MODES = {"train", "predict"}


@dataclass
class Config:
    input_dir: str
    output_dir: str
    centroids_out: str
    model_out: str
    run_mode: str = "train"

    pca_column: str = "cls_pca"

    n_clusters: int = 200
    batch_size: int = 4096
    max_iter: int = 100
    init_size: int = 100_000
    random_state: int = 42
    gpu: bool = False

    num_workers: int = 4
    sample_limit: Optional[int] = None
    progress: bool = True
    log_level: str = "INFO"

    extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def validate(self) -> None:
        if self.run_mode not in VALID_RUN_MODES:
            raise ValueError(f"Invalid run_mode '{self.run_mode}'. Must be one of {sorted(VALID_RUN_MODES)}")
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
        self.log_level = self.log_level.upper()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        init_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra[key] = value
        config = cls(**init_kwargs, extra_fields=extra)
        config.validate()
        return config


def parse_args(argv: Optional[Any] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniBatchKMeans clustering pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    return parser.parse_args(argv)


def load_config_from_cli(argv: Optional[Any] = None) -> Config:
    args = parse_args(argv)
    cfg = Config.from_yaml(args.config)
    return cfg
