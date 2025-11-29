"""Entrypoint for joint metric + classifier training."""
from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.utils import is_flash_attn_2_available  # type: ignore

from sentiment_value.metric_classifier_training.data import (
    DatasetConfig,
    create_dataloaders,
    load_datasets,
    load_external_validation_dataset,
    collate_batch,
)
from sentiment_value.metric_classifier_training.trainer import MetricClassifierModel, Trainer
from sentiment_value.metric_classifier_training.utils.config import Config, load_config
from sentiment_value.classifier_training.utils.logger import NeptuneLogger
from sentiment_value.classifier_training.utils.training import ensure_dir, prepare_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a metric-aware text classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def build_scheduler(optimizer, scheduler_config, num_training_steps: int):
    total_steps = scheduler_config.num_training_steps or num_training_steps
    if total_steps is None:
        raise ValueError("num_training_steps must be provided to configure the scheduler.")
    if scheduler_config.name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_config.warmup_steps,
            num_training_steps=total_steps,
        )

    if scheduler_config.name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_config.warmup_steps,
            num_training_steps=total_steps,
            num_cycles=scheduler_config.num_cycles,
        )

    return None


def build_model(cfg: Config, num_labels: int):
    attn_implementation = cfg.training.attention_implementation
    if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
        print("FlashAttention-2 not available. Falling back to default attention implementation.")
        attn_implementation = None

    reuse_head = False
    try:
        encoder = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            attn_implementation=attn_implementation,
        )
        classifier = getattr(encoder, "classifier", None) or getattr(encoder, "score", None)
        if isinstance(classifier, nn.Linear):
            if classifier.out_features != num_labels:
                new_classifier = nn.Linear(classifier.in_features, num_labels)
                if hasattr(encoder, "classifier"):
                    encoder.classifier = new_classifier
                else:
                    encoder.score = new_classifier
            reuse_head = True
            encoder.config.num_labels = num_labels
    except Exception:
        encoder = AutoModel.from_pretrained(cfg.model_name, attn_implementation=attn_implementation)

    hidden_size = encoder.config.hidden_size
    dropout_prob = getattr(encoder.config, "hidden_dropout_prob", 0.1)
    return MetricClassifierModel(encoder, hidden_size, num_labels, dropout_prob=dropout_prob, reuse_head=reuse_head)


def main():
    args = parse_args()
    cfg: Config = load_config(args.config)

    set_seed(cfg.training.seed)
    device = prepare_device()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_cfg = DatasetConfig(
        parquet_path=cfg.data.parquet_path,
        valid_parquet_path=cfg.data.valid_parquet_path,
        max_seq_length=cfg.training.max_seq_length,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.training.seed,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        downsample=cfg.data.downsample,
    )

    train_dataset, val_dataset, label_encoder = load_datasets(data_cfg, cfg.model_name)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        tokenizer,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        upsample=cfg.data.upsample,
    )

    extra_val_loader: Optional[DataLoader] = None
    if cfg.data.valid_parquet_path:
        extra_val_dataset = load_external_validation_dataset(
            cfg.data.valid_parquet_path,
            tokenizer,
            label_encoder,
            cfg.training.max_seq_length,
        )
        extra_val_loader = DataLoader(
            extra_val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_batch(tokenizer, cfg.training.max_seq_length),
            num_workers=cfg.data.num_workers,
        )

    model = build_model(cfg, label_encoder.num_labels)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )

    total_training_steps = (len(train_loader) * cfg.training.num_epochs) // cfg.training.gradient_accumulation_steps
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_training_steps)

    ensure_dir(cfg.checkpointing.checkpoints_dir)
    logger = NeptuneLogger(cfg.neptune)
    logger.log_hyperparameters(
        {
            "model_name": cfg.model_name,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.optimizer.learning_rate,
            "num_epochs": cfg.training.num_epochs,
            "max_seq_length": cfg.training.max_seq_length,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "weight_decay": cfg.optimizer.weight_decay,
            "warmup_steps": cfg.scheduler.warmup_steps,
            "scheduler": cfg.scheduler.name,
            "scheduler_num_cycles": cfg.scheduler.num_cycles,
            "supcon_temperature": cfg.training.supcon_temperature,
            "attention_implementation": cfg.training.attention_implementation or "default",
            "upsample": cfg.data.upsample,
            "downsample": cfg.data.downsample,
            "classification_loss_weight": cfg.training.classification_loss_weight,
            "metric_loss_weight": cfg.training.metric_loss_weight,
            "metric_validation": {
                "recall_at_k": cfg.metric_validation.recall_at_k,
                "distance": cfg.metric_validation.distance,
                "knn_k": cfg.metric_validation.knn_k,
            },
        }
    )

    start_state: Optional[dict] = None
    if cfg.training.resume_from:
        checkpoint_manager = CheckpointManager(cfg.checkpointing.checkpoints_dir)
        start_state = checkpoint_manager.load(cfg.training.resume_from, model, optimizer, scheduler)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        label_encoder=label_encoder,
        supcon_temperature=cfg.training.supcon_temperature,
        classification_loss_weight=cfg.training.classification_loss_weight,
        metric_loss_weight=cfg.training.metric_loss_weight,
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        checkpoints_dir=cfg.checkpointing.checkpoints_dir,
        save_every_n_bathces=cfg.checkpointing.save_every_n_bathces,
        save_best_by=cfg.checkpointing.save_best_by,
        start_state=start_state,
        metric_validation_cfg={
            "recall_at_k": cfg.metric_validation.recall_at_k,
            "distance": cfg.metric_validation.distance,
            "knn_k": cfg.metric_validation.knn_k,
        },
        extra_val_loader=extra_val_loader,
    )

    try:
        trainer.train(cfg.training.num_epochs)
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
