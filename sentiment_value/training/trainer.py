"""Training loop with checkpointing and validation."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from sentiment_value.training.metrics import compute_metrics, plot_confusion_matrix
from sentiment_value.utils.training import ensure_dir, move_batch_to_device


class CheckpointManager:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR],
        state: Dict,
        confusion_matrix_path: Optional[str] = None,
    ) -> Path:
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
            Path(confusion_matrix_path).replace(dest)
            confusion_matrix_path = str(dest)
        return checkpoint_dir

    def load(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR] = None,
    ) -> Dict:
        checkpoint_path = Path(checkpoint_dir)
        model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(checkpoint_path / "scheduler.pt"))
        with open(checkpoint_path / "state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
        return state


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LambdaLR],
        device: torch.device,
        logger,
        label_encoder,
        grad_accum_steps: int = 1,
        mixed_precision: bool = False,
        gradient_clip_val: Optional[float] = None,
        checkpoints_dir: str = "checkpoints",
        save_every_n_steps: int = 500,
        save_best_by: str = "loss",
        start_state: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.label_encoder = label_encoder
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        self.checkpoint_manager = CheckpointManager(checkpoints_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_best_by = save_best_by

        self.global_step = 0
        self.start_epoch = 0
        self.best_metric = float("inf") if save_best_by == "loss" else float("-inf")
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

        if start_state:
            self.global_step = start_state.get("global_step", 0)
            self.start_epoch = start_state.get("epoch", 0)
            self.best_metric = start_state.get("best_metric", self.best_metric)

        self.model.to(self.device)

    def train(self, num_epochs: int):
        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            epoch_loss = 0.0
            for step, batch in enumerate(self.train_loader):
                batch = move_batch_to_device(batch, self.device)
                with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum_steps == 0:
                    if self.gradient_clip_val is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1
                    if self.global_step % self.save_every_n_steps == 0:
                        self._save_checkpoint(f"step_{self.global_step}", state={"epoch": epoch})

                epoch_loss += loss.item() * self.grad_accum_steps

            avg_train_loss = epoch_loss / len(self.train_loader)
            self.logger.save_metrics("train", "loss", avg_train_loss, step=self.global_step)
            val_loss, metrics, cm_path = self.validate(epoch)
            self._maybe_save_best(val_loss, metrics, cm_path, epoch)

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        preds = []
        labels = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                total_loss += loss.item()
                preds.extend(torch.argmax(logits, dim=-1).tolist())
                labels.extend(batch["labels"].tolist())

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(preds, labels)
        cm_fig = plot_confusion_matrix(labels, preds, [self.label_encoder.id_to_label[i] for i in sorted(self.label_encoder.id_to_label)])
        cm_path = self._save_confusion_matrix(cm_fig, epoch)

        self.logger.save_metrics("val", ["loss", "accuracy", "precision", "recall", "f1"], [avg_loss, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]], step=self.global_step)
        self.logger.save_plot("val", f"confusion_matrix_epoch_{epoch}", cm_fig)

        return avg_loss, metrics, cm_path

    def _save_confusion_matrix(self, fig, epoch: int) -> str:
        cm_path = self.checkpoint_manager.base_dir / f"confusion_matrix_epoch_{epoch}.png"
        fig.savefig(cm_path)
        return str(cm_path)

    def _maybe_save_best(self, val_loss: float, metrics: Dict[str, float], cm_path: str, epoch: int):
        metric = val_loss if self.save_best_by == "loss" else metrics.get("accuracy", 0.0)
        is_better = metric < self.best_metric if self.save_best_by == "loss" else metric > self.best_metric
        if is_better:
            self.best_metric = metric
            state = {"global_step": self.global_step, "best_metric": self.best_metric, "epoch": epoch}
            self._save_checkpoint("best", cm_path=cm_path, state=state)

    def _save_checkpoint(self, name: str, cm_path: Optional[str] = None, state: Optional[Dict] = None):
        base_state = {
            "global_step": self.global_step,
            "epoch": state.get("epoch", 0) if state else 0,
            "best_metric": self.best_metric,
        }
        if state:
            base_state.update(state)
        self.checkpoint_manager.save(
            name=name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=base_state,
            confusion_matrix_path=cm_path,
        )


__all__ = ["Trainer", "CheckpointManager"]
