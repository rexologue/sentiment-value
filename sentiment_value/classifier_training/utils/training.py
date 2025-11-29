from __future__ import annotations

"""Utility helpers for device setup, seeding, and tensor movement."""

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_device() -> torch.device:
    """Return a CUDA device when available, else CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path):
    """Create a directory (and parents) if it does not already exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a batch dictionary of tensors onto the specified device."""

    return {k: v.to(device) for k, v in batch.items()}


__all__ = ["set_seed", "prepare_device", "ensure_dir", "move_batch_to_device"]
