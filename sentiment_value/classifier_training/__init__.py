"""Classifier training package consolidating data, metrics, and training loops."""

from sentiment_value.classifier_training.data import (
    ClassificationDataset,
    DatasetConfig,
    LabelEncoder,
    collate_batch,
    create_dataloaders,
    load_datasets,
)
from sentiment_value.classifier_training.metrics import compute_metrics, plot_confusion_matrix
from sentiment_value.classifier_training.trainer import Trainer
from sentiment_value.classifier_training.checkpoint_manager import CheckpointManager

__all__ = [
    "ClassificationDataset",
    "DatasetConfig",
    "LabelEncoder",
    "collate_batch",
    "create_dataloaders",
    "load_datasets",
    "compute_metrics",
    "plot_confusion_matrix",
    "Trainer",
    "CheckpointManager",
]
