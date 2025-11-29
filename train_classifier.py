"""Entrypoint for fine-tuning jhu-clsp/mmBERT-base on text classification."""
from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.utils import is_flash_attn_2_available  # type: ignore

from sentiment_value.classifier_training.trainer import Trainer
from sentiment_value.classifier_training.checkpoint_manager import CheckpointManager
from sentiment_value.classifier_training.utils.logger import NeptuneLogger
from sentiment_value.classifier_training.utils.config import Config, load_config
from sentiment_value.classifier_training.utils.training import ensure_dir, prepare_device, set_seed
from sentiment_value.classifier_training.data import (
    DatasetConfig,
    create_dataloaders,
    load_datasets,
    load_external_validation_dataset,
    collate_batch,
)


def parse_args():
    """Parse CLI arguments for the classifier training script."""

    parser = argparse.ArgumentParser(description="Train a text classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def build_scheduler(optimizer, scheduler_config, num_training_steps: int):
    """Construct the configured learning rate scheduler."""

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


def main():
    """Load configuration, prepare data/model, and launch training."""

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

    attn_implementation = cfg.training.attention_implementation
    if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
        print("FlashAttention-2 not available. Falling back to default attention implementation.")
        attn_implementation = None

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=label_encoder.num_labels,
        attn_implementation=attn_implementation,
    )

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
            "label_smoothing": cfg.training.label_smoothing,
            "attention_implementation": attn_implementation or "default",
            "upsample": cfg.data.upsample,
            "downsample": cfg.data.downsample,
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
        extra_val_loader=extra_val_loader,
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        label_smoothing=cfg.training.label_smoothing,
        checkpoints_dir=cfg.checkpointing.checkpoints_dir,
        save_every_n_bathces=cfg.checkpointing.save_every_n_bathces,
        save_best_by=cfg.checkpointing.save_best_by,
        start_state=start_state,
    )

    try:
        trainer.train(cfg.training.num_epochs)

    finally:
        logger.stop()


if __name__ == "__main__":
    main()
