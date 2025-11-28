"""Dataset utilities for text classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


@dataclass
class DatasetConfig:
    parquet_path: str
    max_seq_length: int = 256
    val_ratio: float = 0.1
    seed: int = 42
    batch_size: int = 8
    shuffle: bool = True


class LabelEncoder:
    """Simple label encoder for mapping labels to ids and back."""

    def __init__(self, labels: Iterable):
        unique_labels = sorted(set(labels))
        self.label_to_id: Dict[str, int] = {str(lbl): idx for idx, lbl in enumerate(unique_labels)}
        self.id_to_label: Dict[int, str] = {v: k for k, v in self.label_to_id.items()}

    def encode(self, labels: Iterable) -> List[int]:
        return [self.label_to_id[str(lbl)] for lbl in labels]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_label[int(i)] for i in ids]

    @property
    def num_labels(self) -> int:
        return len(self.label_to_id)


class ClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ) # type: ignore

        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def collate_batch(tokenizer: AutoTokenizer) -> Callable[[List[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    def _collate(examples: List[Dict[str, torch.Tensor]]):
        features = [{k: v for k, v in ex.items() if k != "labels"} for ex in examples]
        labels = torch.tensor([ex["labels"] for ex in examples], dtype=torch.long)
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels

        return batch

    return _collate


def load_datasets(
    config: DatasetConfig,
    tokenizer_name: str,
    val_ratio: Optional[float] = None,
) -> Tuple[ClassificationDataset, ClassificationDataset, LabelEncoder]:
    """Load parquet data and return train/val datasets and label encoder."""
    df = pd.read_parquet(config.parquet_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Parquet file must contain 'text' and 'label' columns")

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


def create_dataloaders(
    train_dataset: ClassificationDataset,
    val_dataset: ClassificationDataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    collate_fn = collate_batch(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
