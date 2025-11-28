"""Entrypoint for fine-tuning deepvk/USER-base on text classification."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from sentiment_value.utils.logger import NeptuneLogger
from sentiment_value.utils.config import Config, load_config
from sentiment_value.training.trainer import Trainer, CheckpointManager
from sentiment_value.utils.training import ensure_dir, prepare_device, set_seed
from sentiment_value.data.dataset import DatasetConfig, create_dataloaders, load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    return parser.parse_args()


def build_scheduler(optimizer, scheduler_config, num_training_steps: int):
    if scheduler_config.name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_config.warmup_steps,
            num_training_steps=num_training_steps,
        )
    
    return None


def main():
    args = parse_args()
    cfg: Config = load_config(args.config)

    set_seed(cfg.training.seed)
    device = prepare_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_cfg = DatasetConfig(
        parquet_path=cfg.data.parquet_path,
        max_seq_length=cfg.training.max_seq_length,
        val_ratio=cfg.data.val_ratio,
        seed=cfg.training.seed,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )

    train_dataset, val_dataset, label_encoder = load_datasets(data_cfg, cfg.model_name)
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        tokenizer,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=label_encoder.num_labels,
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
        grad_accum_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        checkpoints_dir=cfg.checkpointing.checkpoints_dir,
        save_every_n_steps=cfg.checkpointing.save_every_n_steps,
        save_best_by=cfg.checkpointing.save_best_by,
        start_state=start_state,
    )

    try:
        trainer.train(cfg.training.num_epochs)
        
    finally:
        logger.stop()


if __name__ == "__main__":
    main()
