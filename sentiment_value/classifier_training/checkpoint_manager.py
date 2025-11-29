"""Checkpoint saving/loading helpers for classifier training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from shutil import copy2

from sentiment_value.classifier_training.utils.training import ensure_dir


class CheckpointManager:
    """Manage saving and loading of model, optimizer, and scheduler state."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        state: Dict,
        confusion_matrix_path: Optional[str] = None,
    ) -> Path:
        """Persist a checkpoint bundle and return its directory."""

        checkpoint_dir = self.base_dir / name

        ensure_dir(checkpoint_dir)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        if scheduler is not None:
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        with open(checkpoint_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        if confusion_matrix_path:
            dest = checkpoint_dir / Path(confusion_matrix_path).name
            copy2(confusion_matrix_path, dest)
            confusion_matrix_path = str(dest)

        return checkpoint_dir

    def load(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict:
        """Restore a checkpoint from disk and return the saved state dictionary."""

        checkpoint_path = Path(checkpoint_dir)
        model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))

        if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))

        with open(checkpoint_path / "state.json", "r", encoding="utf-8") as f:
            state = json.load(f)

        return state


__all__ = ["CheckpointManager"]
