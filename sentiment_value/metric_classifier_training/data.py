"""Dataset utilities for joint metric + classifier training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    parquet_path: str
    valid_parquet_path: str
    max_seq_length: int = 256
    seed: int = 42
    batch_size: int = 8
    shuffle: bool = True
    downsample: bool = False


class LabelEncoder:
    """Simple label encoder for mapping labels to ids and back."""

    def __init__(self, labels: Iterable):
        unique_labels = sorted({int(lbl) for lbl in labels})
        self.label_to_id: Dict[int, int] = {int(lbl): idx for idx, lbl in enumerate(unique_labels)}
        self.id_to_label: Dict[int, int] = {v: k for k, v in self.label_to_id.items()}

    def encode(self, labels: Iterable) -> List[int]:
        return [self.label_to_id[int(lbl)] for lbl in labels]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [str(self.id_to_label[int(i)]) for i in ids]

    @property
    def num_labels(self) -> int:
        return len(self.label_to_id)


class MetricClassificationDataset(Dataset):
    """Dataset storing texts, labels, and task participation masks."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        metric_mask: Sequence[int],
        classifier_mask: Sequence[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.metric_mask = list(metric_mask)
        self.classifier_mask = list(classifier_mask)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.tokenizer.is_fast:
            return {
                "text": self.texts[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "metric_mask": torch.tensor(self.metric_mask[idx], dtype=torch.float),
                "classifier_mask": torch.tensor(self.classifier_mask[idx], dtype=torch.float),
            }

        item = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )  # type: ignore

        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        item["metric_mask"] = torch.tensor(self.metric_mask[idx], dtype=torch.float)
        item["classifier_mask"] = torch.tensor(self.classifier_mask[idx], dtype=torch.float)

        return item


def collate_batch(tokenizer: AutoTokenizer, max_length: int) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Build a collate function that tokenizes a batch and stacks labels and masks."""

    def _collate(examples: List[Dict[str, torch.Tensor]]):
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        metric_mask = torch.tensor([ex["metric_mask"] for ex in examples], dtype=torch.float)
        classifier_mask = torch.tensor([ex["classifier_mask"] for ex in examples], dtype=torch.float)

        if tokenizer.is_fast and "text" in examples[0]:
            batch = tokenizer(
                [ex["text"] for ex in examples],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            features = [{k: v for k, v in ex.items() if k not in {"labels", "metric_mask", "classifier_mask"}} for ex in examples]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")

        batch["labels"] = labels
        batch["metric_mask"] = metric_mask
        batch["classifier_mask"] = classifier_mask

        return batch

    return _collate


def load_datasets(
    config: DatasetConfig,
    tokenizer_name: str,
) -> Tuple[MetricClassificationDataset, MetricClassificationDataset, LabelEncoder]:
    """Load parquet data and return train/val datasets and a label encoder."""

    train_df = pd.read_parquet(config.parquet_path)
    val_df = pd.read_parquet(config.valid_parquet_path)
    required_columns = {"text", "label", "metric_mask", "classification_mask"}

    for name, df in {"train": train_df, "validation": val_df}.items():
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"{name} parquet file must contain columns: {missing}")

    if config.downsample:
        label_counts = train_df["label"].value_counts()
        if label_counts.empty:
            raise ValueError("Cannot downsample an empty dataset")
        min_count = label_counts.min()
        train_df = (
            train_df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min_count, random_state=config.seed))
            .reset_index(drop=True)
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    labels_series = pd.concat(
        [train_df["label"].astype("int64"), val_df["label"].astype("int64")], ignore_index=True
    )
    encoder = LabelEncoder(labels_series.tolist())
    train_labels = encoder.encode(train_df["label"].astype("int64").tolist())
    val_labels = encoder.encode(val_df["label"].astype("int64").tolist())

    train_dataset = MetricClassificationDataset(
        train_df["text"].tolist(),
        train_labels,
        train_df["metric_mask"].tolist(),
        train_df["classification_mask"].tolist(),
        tokenizer,
        config.max_seq_length,
    )
    val_dataset = MetricClassificationDataset(
        val_df["text"].tolist(),
        val_labels,
        val_df["metric_mask"].tolist(),
        val_df["classification_mask"].tolist(),
        tokenizer,
        config.max_seq_length,
    )

    return train_dataset, val_dataset, encoder


def _build_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot build sampler without labels")

    max_count = max(counts.values())
    weights = [max_count / counts[label] for label in labels]
    num_samples = max_count * len(counts)

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def create_dataloaders(
    train_dataset: MetricClassificationDataset,
    val_dataset: MetricClassificationDataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    upsample: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with optional upsampling."""

    collate_fn = collate_batch(tokenizer, train_dataset.max_length)
    sampler = _build_balanced_sampler(train_dataset.labels) if upsample else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader


__all__ = [
    "DatasetConfig",
    "LabelEncoder",
    "MetricClassificationDataset",
    "collate_batch",
    "load_datasets",
    "create_dataloaders",
]
