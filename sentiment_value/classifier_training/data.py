"""Dataset utilities for text classification tasks."""
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
    """Configuration describing how to load the training and validation datasets."""

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
        unique_labels = sorted(set(labels))
        self.label_to_id: Dict[str, int] = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
        self.id_to_label: Dict[int, str] = {v: k for k, v in self.label_to_id.items()}

    def encode(self, labels: Iterable) -> List[int]:
        """Encode an iterable of labels into integer ids."""

        return [self.label_to_id[str(lbl)] for lbl in labels]

    def decode(self, ids: Iterable[int]) -> List[str]:
        """Decode integer ids back into string labels."""

        return [self.id_to_label[int(i)] for i in ids]

    @property
    def num_labels(self) -> int:
        """Return the number of unique labels."""

        return len(self.label_to_id)


class ClassificationDataset(Dataset):
    """Dataset storing texts and label ids for classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        if self.tokenizer.is_fast:
            return {"text": self.texts[idx], "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

        item = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )  # type: ignore

        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def collate_batch(
    tokenizer: AutoTokenizer, max_length: int
) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """Build a collate function that tokenizes a batch and stacks labels."""

    def _collate(examples: List[Dict[str, torch.Tensor]]):
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)

        if tokenizer.is_fast and "text" in examples[0]:
            batch = tokenizer(
                [ex["text"] for ex in examples],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            features = [{k: v for k, v in ex.items() if k != "labels"} for ex in examples]
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")

        batch["labels"] = labels

        return batch

    return _collate


def load_datasets(
    config: DatasetConfig,
    tokenizer_name: str,
) -> Tuple[ClassificationDataset, ClassificationDataset, LabelEncoder]:
    """Load parquet data and return train/val datasets and a label encoder.

    Args:
        config: Dataset configuration containing file paths and preprocessing parameters.
        tokenizer_name: Name or path of the tokenizer for consistent tokenization.

    Returns:
        A tuple of (train_dataset, val_dataset, label_encoder).
    """

    train_df = pd.read_parquet(config.parquet_path)
    val_df = pd.read_parquet(config.valid_parquet_path)

    for name, df in {"train": train_df, "validation": val_df}.items():
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{name} parquet file must contain 'text' and 'label' columns")

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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, fix_mistral_regex=True)

    encoder = LabelEncoder(list(train_df["label"].tolist()) + list(val_df["label"].tolist()))
    train_labels = encoder.encode(train_df["label"].tolist())
    val_labels = encoder.encode(val_df["label"].tolist())

    train_dataset = ClassificationDataset(train_df["text"].tolist(), train_labels, tokenizer, config.max_seq_length)
    val_dataset = ClassificationDataset(val_df["text"].tolist(), val_labels, tokenizer, config.max_seq_length)

    return train_dataset, val_dataset, encoder


def _build_balanced_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create a sampler that upsamples minority classes for balanced batches."""

    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot build sampler without labels")

    max_count = max(counts.values())
    weights = [max_count / counts[label] for label in labels]
    num_samples = max_count * len(counts)

    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def create_dataloaders(
    train_dataset: ClassificationDataset,
    val_dataset: ClassificationDataset,
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
    "ClassificationDataset",
    "collate_batch",
    "load_datasets",
    "create_dataloaders",
]
