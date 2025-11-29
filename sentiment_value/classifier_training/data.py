"""Dataset utilities for text classification tasks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    """Configuration describing how to load and split the dataset."""

    parquet_path: str
    max_seq_length: int = 256
    val_ratio: float = 0.1
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
    val_ratio: Optional[float] = None,
) -> Tuple[ClassificationDataset, ClassificationDataset, LabelEncoder]:
    """Load parquet data and return train/val datasets and a label encoder.

    Args:
        config: Dataset configuration containing file path and split parameters.
        tokenizer_name: Name or path of the tokenizer for consistent tokenization.
        val_ratio: Optional override for the validation split ratio.

    Returns:
        A tuple of (train_dataset, val_dataset, label_encoder).
    """

    df = pd.read_parquet(config.parquet_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Parquet file must contain 'text' and 'label' columns")

    if config.downsample:
        label_counts = df["label"].value_counts()
        if label_counts.empty:
            raise ValueError("Cannot downsample an empty dataset")
        min_count = label_counts.min()
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min_count, random_state=config.seed))
            .reset_index(drop=True)
        )

    val_ratio = val_ratio if val_ratio is not None else config.val_ratio
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoder = LabelEncoder(df["label"].tolist())
    encoded_labels = encoder.encode(df["label"].tolist())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        encoded_labels,
        test_size=val_ratio,
        random_state=config.seed,
        stratify=encoded_labels if encoder.num_labels > 1 else None,
    )

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, config.max_seq_length)

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


def load_external_validation_dataset(
    parquet_path: str,
    tokenizer: AutoTokenizer,
    label_encoder: LabelEncoder,
    max_seq_length: int,
) -> ClassificationDataset:
    """Load an external validation dataset using an existing tokenizer and encoder."""

    df = pd.read_parquet(parquet_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("External validation parquet must contain 'text' and 'label' columns")

    encoded_labels = label_encoder.encode(df["label"].tolist())
    return ClassificationDataset(df["text"].tolist(), encoded_labels, tokenizer, max_seq_length)


__all__ = [
    "DatasetConfig",
    "LabelEncoder",
    "ClassificationDataset",
    "collate_batch",
    "load_datasets",
    "create_dataloaders",
    "load_external_validation_dataset",
]
