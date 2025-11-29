"""Training loop with checkpointing and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from sentiment_value.classifier_training.utils.training import move_batch_to_device
from sentiment_value.classifier_training.checkpoint_manager import CheckpointManager
from sentiment_value.classifier_training.metrics import compute_metrics, plot_confusion_matrix


class Trainer:
    """Orchestrates training, validation, and checkpointing for classifiers."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        logger,
        label_encoder,
        grad_accum_steps: int = 1,
        mixed_precision: bool = False,
        gradient_clip_val: Optional[float] = None,
        label_smoothing: float = 0.0,
        checkpoints_dir: str = "checkpoints",
        save_every_n_bathces: int = 500,
        save_best_by: str = "loss",
        start_state: Optional[Dict] = None,
        extra_val_loader: Optional[DataLoader] = None,
    ):
        """Initialize the trainer with dataloaders, optimizer, and logger.

        Args:
            model: The classifier model to train.
            train_loader: DataLoader yielding training batches.
            val_loader: DataLoader yielding validation batches.
            optimizer: Optimizer instance for parameter updates.
            scheduler: Optional learning rate scheduler.
            device: Torch device to run training on.
            logger: Experiment logger implementing ``save_metrics`` and ``save_plot``.
            label_encoder: Encoder exposing ``id_to_label`` for confusion matrix labels.
            grad_accum_steps: Number of gradient accumulation steps.
            mixed_precision: Enable CUDA AMP when available.
            gradient_clip_val: Optional gradient clipping value.
            label_smoothing: Cross-entropy label smoothing factor.
            checkpoints_dir: Directory for saving checkpoints.
            save_every_n_bathces: Frequency (in batches) to checkpoint.
            save_best_by: Metric name used to determine the best checkpoint.
            start_state: Optional state dictionary for resuming training.
        """

        if save_best_by not in {"loss", "accuracy", "f1"}:
            raise ValueError("save_best_by must be one of {'loss', 'accuracy', 'f1'}")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.label_encoder = label_encoder
        self.extra_val_loader = extra_val_loader
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0.0, 1.0)")
        self.label_smoothing = label_smoothing
        self.checkpoint_manager = CheckpointManager(checkpoints_dir)
        self.save_every_n_batches = save_every_n_bathces
        self.save_best_by = save_best_by

        self.global_step = 0
        self.global_batch = 0
        self.start_epoch = 0
        self.best_metric = float("inf") if save_best_by == "loss" else float("-inf")
        self.use_cuda_amp = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda_amp)  # type: ignore

        if start_state:
            self.global_step = start_state.get("global_step", 0)
            self.global_batch = start_state.get("global_batch", 0)
            self.start_epoch = start_state.get("epoch", 0)
            self.best_metric = start_state.get("best_metric", self.best_metric)

        self.model.to(self.device)

    def train(self, num_epochs: int):
        """Train the model for a fixed number of epochs with validation."""

        last_cm_path: Optional[str] = None
        epoch = self.start_epoch

        try:
            for epoch in range(self.start_epoch, num_epochs):
                self.model.train()
                epoch_loss = 0.0
                latest_val_results: Optional[tuple[float, Dict[str, float], str]] = None

                with tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}",
                    dynamic_ncols=True,
                ) as progress:
                    for step, batch in progress:
                        self.global_batch += 1
                        batch = move_batch_to_device(batch, self.device)

                        autocast_context = (
                            torch.amp.autocast("cuda", enabled=True)  # type: ignore
                            if self.use_cuda_amp
                            else nullcontext()
                        )
                        with autocast_context:
                            outputs = self.model(**batch)
                            if self.label_smoothing > 0.0:
                                loss = F.cross_entropy(
                                    outputs.logits,
                                    batch["labels"],
                                    label_smoothing=self.label_smoothing,
                                )
                            else:
                                loss = outputs.loss

                            loss = loss / self.grad_accum_steps

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

                            current_lr = self.optimizer.param_groups[0].get("lr")
                            self.logger.save_metrics(
                                "train", "learning_rate", current_lr, step=self.global_step
                            )

                        batch_preds = torch.argmax(outputs.logits.detach(), dim=-1).tolist()
                        batch_labels = batch["labels"].tolist()
                        batch_metrics = compute_metrics(batch_preds, batch_labels)
                        self.logger.save_metrics(
                            "train",
                            ["f1", "accuracy", "precision", "recall"],
                            [
                                batch_metrics["f1"],
                                batch_metrics["accuracy"],
                                batch_metrics["precision"],
                                batch_metrics["recall"],
                            ],
                            step=self.global_batch,
                        )

                        epoch_loss += loss.item() * self.grad_accum_steps
                        running_train_loss = epoch_loss / (step + 1)
                        self.logger.save_metrics("train", "loss", running_train_loss, step=self.global_batch)
                        progress.set_postfix(train_loss=epoch_loss / (step + 1))

                        if self.global_batch % self.save_every_n_batches == 0:
                            val_loss, metrics, cm_path = self.validate(epoch)
                            if self.extra_val_loader is not None:
                                self.validate(epoch, loader=self.extra_val_loader, suffix="_extra")
                            latest_val_results = (val_loss, metrics, cm_path)
                            self._maybe_save_best(val_loss, metrics, cm_path, epoch)
                            checkpoint_path = self._save_checkpoint(
                                f"batch_{self.global_batch}",
                                cm_path=cm_path,
                                state={"epoch": epoch},
                            )
                            last_cm_path = cm_path
                            tqdm.write(f"Saved checkpoint at {checkpoint_path}")

                    avg_train_loss = epoch_loss / len(self.train_loader)
                    self.logger.save_metrics(
                        "train", "epoch_loss", avg_train_loss, step=self.global_batch
                    )

                    if latest_val_results is None:
                        val_loss, metrics, cm_path = self.validate(epoch)
                    else:
                        val_loss, metrics, cm_path = latest_val_results

                    if self.extra_val_loader is not None:
                        self.validate(epoch, loader=self.extra_val_loader, suffix="_extra")

                    last_cm_path = cm_path
                    progress.set_description(
                        f"Epoch {epoch + 1} | train_loss={avg_train_loss:.4f} "
                        f"val_loss={val_loss:.4f} val_f1={metrics['f1']:.4f}"
                    )
                    progress.refresh()
                    self._maybe_save_best(val_loss, metrics, cm_path, epoch)

            self._attempt_save_last_checkpoint(epoch, last_cm_path)
        except Exception:
            self._attempt_save_last_checkpoint(epoch, last_cm_path)
            raise

    def validate(
        self, epoch: int, loader: Optional[DataLoader] = None, suffix: str = ""
    ) -> tuple[float, Dict[str, float], str]:
        """Run validation and return loss, metrics, and confusion matrix path."""

        eval_loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        preds: list[int] = []
        labels: list[int] = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, self.device)
                outputs = self.model(**batch)
                logits = outputs.logits
                loss = outputs.loss
                total_loss += loss.item()
                preds.extend(torch.argmax(logits, dim=-1).tolist())
                labels.extend(batch["labels"].tolist())

        avg_loss = total_loss / len(eval_loader)
        metrics = compute_metrics(preds, labels)
        cm_fig = plot_confusion_matrix(
            labels,
            preds,
            [self.label_encoder.id_to_label[i] for i in sorted(self.label_encoder.id_to_label)],
        )
        cm_path = self._save_confusion_matrix(cm_fig, epoch, suffix)

        metric_names = ["loss", "accuracy", "precision", "recall", "f1"]
        metric_values = [
            avg_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        ]
        metric_names = [f"{name}{suffix}" for name in metric_names] if suffix else metric_names

        self.logger.save_metrics("val", metric_names, metric_values, step=self.global_batch)
        self.logger.save_plot("val", f"confusion_matrix{suffix}_epoch_{epoch}", cm_fig)

        return avg_loss, metrics, cm_path

    def _save_confusion_matrix(self, fig, epoch: int, suffix: str = "") -> str:
        """Save the confusion matrix image for later inspection."""

        cm_path = self.checkpoint_manager.base_dir / f"confusion_matrix{suffix}_epoch_{epoch}.png"
        fig.savefig(cm_path)

        return str(cm_path)

    def _maybe_save_best(self, val_loss: float, metrics: Dict[str, float], cm_path: str, epoch: int):
        """Persist the best-performing checkpoint according to the configured metric."""

        metric = val_loss if self.save_best_by == "loss" else metrics.get(self.save_best_by, 0.0)
        is_better = metric < self.best_metric if self.save_best_by == "loss" else metric > self.best_metric

        if is_better:
            self.best_metric = metric
            state = {"global_step": self.global_step, "best_metric": self.best_metric, "epoch": epoch}
            checkpoint_path = self._save_checkpoint("best", cm_path=cm_path, state=state)
            tqdm.write(
                "New best model saved to {} ({}: {:.4f})".format(
                    checkpoint_path, self.save_best_by, self.best_metric
                )
            )

    def _save_checkpoint(
        self, name: str, cm_path: Optional[str] = None, state: Optional[Dict] = None
    ) -> Path:
        """Save a checkpoint with the current training state."""

        base_state = {
            "global_step": self.global_step,
            "global_batch": self.global_batch,
            "epoch": state.get("epoch", 0) if state else 0,
            "best_metric": self.best_metric,
        }

        if state:
            base_state.update(state)

        return self.checkpoint_manager.save(
            name=name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            state=base_state,
            confusion_matrix_path=cm_path,
        )

    def _attempt_save_last_checkpoint(self, epoch: int, cm_path: Optional[str]):
        """Best-effort final checkpoint save, even if an error occurs."""

        try:
            checkpoint_path = self._save_checkpoint(
                "last", cm_path=cm_path, state={"epoch": epoch}
            )
            tqdm.write(f"Saved last checkpoint at {checkpoint_path}")
        except Exception as err:  # pragma: no cover - best effort for robustness
            tqdm.write(f"Failed to save last checkpoint: {err}")


__all__ = ["Trainer"]
