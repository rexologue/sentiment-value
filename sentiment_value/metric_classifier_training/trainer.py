"""Training loop for joint metric + classifier model."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sentiment_value.classifier_training.checkpoint_manager import CheckpointManager
from sentiment_value.classifier_training.utils.training import move_batch_to_device
from sentiment_value.metric_classifier_training.losses import masked_cross_entropy, supervised_contrastive_loss
from sentiment_value.metric_classifier_training.metrics import (
    compute_metrics,
    knn_macro_f1,
    plot_confusion_matrix,
    recall_at_k,
)


class MetricClassifierModel(nn.Module):
    """Wrapper that exposes logits and embeddings for metric learning."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout_prob: float = 0.1, reuse_head: bool = False):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_prob)
        self.use_encoder_head = reuse_head and hasattr(encoder, "config") and getattr(encoder.config, "num_labels", num_labels) == num_labels
        if not self.use_encoder_head:
            self.classifier = nn.Linear(hidden_size, num_labels)
        else:
            encoder.config.num_labels = num_labels

    def forward(self, **inputs):
        encoder_outputs = self.encoder(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(encoder_outputs, "hidden_states", None)
        pooler_output = getattr(encoder_outputs, "pooler_output", None)

        if pooler_output is not None:
            features_raw = pooler_output
        elif hidden_states is not None and len(hidden_states) > 0:
            features_raw = hidden_states[-1][:, 0]
        else:
            features_raw = encoder_outputs.last_hidden_state[:, 0]

        if self.use_encoder_head and hasattr(encoder_outputs, "logits"):
            logits = encoder_outputs.logits
        else:
            logits = self.classifier(self.dropout(features_raw))

        features_norm = torch.nn.functional.normalize(features_raw, p=2, dim=-1)

        return {"logits": logits, "features_raw": features_raw, "features_norm": features_norm}


class Trainer:
    """Orchestrates training, validation, and checkpointing for metric-classifier."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LRScheduler],
        device: torch.device,
        logger,
        label_encoder,
        supcon_temperature: float = 0.1,
        classification_loss_weight: float = 1.0,
        metric_loss_weight: float = 1.0,
        grad_accum_steps: int = 1,
        mixed_precision: bool = False,
        gradient_clip_val: Optional[float] = None,
        checkpoints_dir: str = "checkpoints",
        save_every_n_bathces: int = 500,
        save_best_by: str = "loss",
        start_state: Optional[Dict] = None,
        metric_validation_cfg: Optional[Dict] = None,
        extra_val_loader: Optional[DataLoader] = None,
    ):
        if save_best_by not in {"loss", "acc", "f1"}:
            raise ValueError("save_best_by must be one of {'loss', 'acc', 'f1'}")

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
        self.supcon_temperature = supcon_temperature
        self.classification_loss_weight = classification_loss_weight
        self.metric_loss_weight = metric_loss_weight
        self.checkpoint_manager = CheckpointManager(checkpoints_dir)
        self.save_every_n_batches = save_every_n_bathces
        self.save_best_by = save_best_by
        self.metric_validation_cfg = metric_validation_cfg or {}
        self.keep_last_n_emb_steps = self.metric_validation_cfg.get("keep_last_n_emb_steps")

        self.global_step = 0
        self.global_batch = 0
        self.start_epoch = 0
        self.best_metric = float("inf") if save_best_by == "loss" else float("-inf")
        self.use_cuda_amp = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda_amp)  # type: ignore

        self.train_embedding_bank: list[torch.Tensor] = []
        self.train_label_bank: list[torch.Tensor] = []
        self._pending_train_embeddings: list[torch.Tensor] = []
        self._pending_train_labels: list[torch.Tensor] = []

        if start_state:
            self.global_step = start_state.get("global_step", 0)
            self.global_batch = start_state.get("global_batch", 0)
            self.start_epoch = start_state.get("epoch", 0)
            self.best_metric = start_state.get("best_metric", self.best_metric)

        self.model.to(self.device)

    def train(self, num_epochs: int):
        last_cm_path: Optional[str] = None
        epoch = self.start_epoch

        try:
            for epoch in range(self.start_epoch, num_epochs):
                self.model.train()
                epoch_classification_loss = 0.0
                epoch_metric_loss = 0.0
                self._pending_train_embeddings = []
                self._pending_train_labels = []
                latest_val_results: Optional[Tuple[float, Dict[str, float], str]] = None

                with tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}",
                    dynamic_ncols=True,
                ) as progress:
                    for step, batch in progress:
                        self.global_batch += 1
                        batch = move_batch_to_device(batch, self.device)
                        classifier_mask = batch.pop("classifier_mask")
                        metric_mask = batch.pop("metric_mask")

                        autocast_context = (
                            torch.amp.autocast("cuda", enabled=True)  # type: ignore
                            if self.use_cuda_amp
                            else nullcontext()
                        )
                        with autocast_context:
                            outputs = self.model(**batch)
                            logits = outputs["logits"]
                            features_norm = outputs["features_norm"]

                            ce_loss = masked_cross_entropy(
                                logits, batch["labels"], classifier_mask, label_smoothing=self.supcon_temperature
                            )
                            metric_loss = supervised_contrastive_loss(
                                features_norm, batch["labels"], metric_mask, temperature=self.supcon_temperature
                            )
                            total_loss = (
                                self.classification_loss_weight * ce_loss
                                + self.metric_loss_weight * metric_loss
                            )
                            total_loss = total_loss / self.grad_accum_steps

                        self.scaler.scale(total_loss).backward()
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
                            self.logger.save_metrics("train", "learning_rate", current_lr, step=self.global_step)

                        if classifier_mask.sum() > 0:
                            valid_classifier = classifier_mask > 0
                            batch_preds = torch.argmax(logits.detach()[valid_classifier], dim=-1).tolist()
                            batch_labels = batch["labels"][valid_classifier].tolist()
                            batch_metrics = compute_metrics(batch_preds, batch_labels)
                        else:
                            batch_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

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

                        epoch_classification_loss += ce_loss.item()
                        epoch_metric_loss += metric_loss.item()
                        if metric_mask.sum() > 0:
                            valid_metric = metric_mask > 0
                            self._pending_train_embeddings.append(features_norm[valid_metric].detach().cpu())
                            self._pending_train_labels.append(batch["labels"][valid_metric].detach().cpu())
                        running_cls = epoch_classification_loss / (step + 1)
                        running_metric = epoch_metric_loss / (step + 1)
                        running_total = running_cls * self.classification_loss_weight + running_metric * self.metric_loss_weight
                        self.logger.save_metrics(
                            "train",
                            ["classification_loss", "metric_loss", "total_loss"],
                            [running_cls, running_metric, running_total],
                            step=self.global_batch,
                        )
                        progress.set_postfix(train_total_loss=running_total)

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

                avg_train_cls_loss = epoch_classification_loss / len(self.train_loader)
                avg_train_metric_loss = epoch_metric_loss / len(self.train_loader)
                avg_train_total = avg_train_cls_loss * self.classification_loss_weight + avg_train_metric_loss * self.metric_loss_weight
                self.logger.save_metrics(
                    "train",
                    ["epoch_classification_loss", "epoch_metric_loss", "epoch_total_loss"],
                    [avg_train_cls_loss, avg_train_metric_loss, avg_train_total],
                    step=self.global_batch,
                )

                if latest_val_results is None:
                    val_loss, metrics, cm_path = self.validate(epoch)
                else:
                    val_loss, metrics, cm_path = latest_val_results

                if self.extra_val_loader is not None:
                    self.validate(epoch, loader=self.extra_val_loader, suffix="_extra")

                last_cm_path = cm_path
                progress.set_description(
                    f"Epoch {epoch + 1} | total_loss={avg_train_total:.4f} "
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
        self._flush_pending_embedding_bank()
        eval_loader = loader or self.val_loader
        self.model.eval()
        total_cls_loss = 0.0
        total_metric_loss = 0.0
        preds: list[int] = []
        labels_list: list[int] = []
        train_embeddings = (
            torch.cat(self.train_embedding_bank, dim=0) if getattr(self, "train_embedding_bank", None) else torch.empty(0)
        )
        train_labels = (
            torch.cat(self.train_label_bank, dim=0)
            if getattr(self, "train_label_bank", None)
            else torch.empty(0, dtype=torch.long)
        )
        val_embeddings: list[torch.Tensor] = []
        val_metric_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, self.device)
                classifier_mask = batch.pop("classifier_mask")
                metric_mask = batch.pop("metric_mask")

                outputs = self.model(**batch)
                logits = outputs["logits"]
                features_norm = outputs["features_norm"]

                ce_loss = masked_cross_entropy(
                    logits, batch["labels"], classifier_mask, label_smoothing=self.supcon_temperature
                )
                metric_loss = supervised_contrastive_loss(
                    features_norm, batch["labels"], metric_mask, temperature=self.supcon_temperature
                )

                total_cls_loss += ce_loss.item()
                total_metric_loss += metric_loss.item()

                if classifier_mask.sum() > 0:
                    masked_indices = classifier_mask > 0
                    preds.extend(torch.argmax(logits[masked_indices], dim=-1).tolist())
                    labels_list.extend(batch["labels"][masked_indices].tolist())

                if metric_mask.sum() > 0:
                    valid = metric_mask > 0
                    val_embeddings.append(features_norm[valid].detach().cpu())
                    val_metric_labels.append(batch["labels"][valid].detach().cpu())

        avg_cls_loss = total_cls_loss / len(eval_loader)
        avg_metric_loss = total_metric_loss / len(eval_loader)
        total_loss = avg_cls_loss * self.classification_loss_weight + avg_metric_loss * self.metric_loss_weight

        val_emb_tensor = torch.cat(val_embeddings, dim=0) if val_embeddings else torch.empty(0)
        val_label_tensor = torch.cat(val_metric_labels, dim=0) if val_metric_labels else torch.empty(0, dtype=torch.long)

        metrics = (
            compute_metrics(preds, labels_list)
            if labels_list
            else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        )
        cm_fig = plot_confusion_matrix(
            labels_list,
            preds,
            [self.label_encoder.id_to_label[i] for i in sorted(self.label_encoder.id_to_label)],
        )
        cm_path = self._save_confusion_matrix(cm_fig, epoch, suffix)

        recall_metrics = recall_at_k(
            train_embeddings,
            train_labels,
            val_emb_tensor,
            val_label_tensor,
            self.metric_validation_cfg.get("recall_at_k", [1, 10, 100]),
            distance=self.metric_validation_cfg.get("distance", "cos"),
        )
        knn_f1 = knn_macro_f1(
            train_embeddings,
            train_labels,
            val_emb_tensor,
            val_label_tensor,
            self.metric_validation_cfg.get("knn_k", 10),
            distance=self.metric_validation_cfg.get("distance", "cos"),
        )

        metric_names = [
            "classification_loss",
            "metric_loss",
            "total_loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "knn_macro_f1",
        ] + [f"recall_at_{k}" for k in recall_metrics.keys()]
        metric_values = [
            avg_cls_loss,
            avg_metric_loss,
            total_loss,
            metrics.get("accuracy", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
            knn_f1,
        ] + list(recall_metrics.values())
        if suffix:
            metric_names = [f"{name}{suffix}" for name in metric_names]

        self.logger.save_metrics("val", metric_names, metric_values, step=self.global_batch)
        self.logger.save_plot("val", f"confusion_matrix{suffix}_epoch_{epoch}", cm_fig)

        return avg_cls_loss, metrics, cm_path

    def _flush_pending_embedding_bank(self):
        if self._pending_train_embeddings:
            new_embeddings = torch.cat(self._pending_train_embeddings, dim=0)
            new_labels = torch.cat(self._pending_train_labels, dim=0)
            if new_embeddings.numel() > 0:
                self.train_embedding_bank.append(new_embeddings)
                self.train_label_bank.append(new_labels)
                self._prune_embedding_bank()

            self._pending_train_embeddings = []
            self._pending_train_labels = []

    def _prune_embedding_bank(self):
        if self.keep_last_n_emb_steps is None or self.keep_last_n_emb_steps <= 0:
            return

        while len(self.train_embedding_bank) > self.keep_last_n_emb_steps:
            self.train_embedding_bank.pop(0)
            self.train_label_bank.pop(0)

    def _save_confusion_matrix(self, fig, epoch: int, suffix: str = "") -> str:
        cm_path = self.checkpoint_manager.base_dir / f"confusion_matrix{suffix}_epoch_{epoch}.png"
        fig.savefig(cm_path)
        return str(cm_path)

    def _maybe_save_best(self, val_loss: float, metrics: Dict[str, float], cm_path: str, epoch: int):
        metric = val_loss if self.save_best_by == "loss" else metrics.get("accuracy" if self.save_best_by == "acc" else "f1", 0.0)
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

    def _save_checkpoint(self, name: str, cm_path: Optional[str] = None, state: Optional[Dict] = None) -> Path:
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
        try:
            checkpoint_path = self._save_checkpoint("last", cm_path=cm_path, state={"epoch": epoch})
            tqdm.write(f"Saved last checkpoint at {checkpoint_path}")
        except Exception as err:  # pragma: no cover
            tqdm.write(f"Failed to save last checkpoint: {err}")


__all__ = ["Trainer", "MetricClassifierModel"]
