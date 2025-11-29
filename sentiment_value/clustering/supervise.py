"""Supervised inference for building clustering shards.

This script reads text data from a Parquet file, runs batched inference
with a sequence classification model to capture CLS embeddings, and
writes the results into sharded Parquet outputs. Configuration is loaded
exclusively from the ``supervise`` block of the YAML file passed via
``--config``.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_value.clustering.config import SuperviseConfig, parse_config_path


class TextDataset(Dataset):
    """Simple dataset wrapper around a sequence of raw texts."""

    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def load_data(parquet_path: str) -> List[str]:
    """Load the ``text`` column from a Parquet shard."""

    df = pd.read_parquet(parquet_path, columns=["text"])
    return df["text"].tolist()


def collate_with_texts(tokenizer):
    """Collate function that preserves raw text alongside tokenized tensors.

    If tokenization fails for a batch, the batch is marked as invalid and
    will be skipped during inference.
    """

    def _collate(batch_texts: List[str]) -> Tuple[List[str], dict]:
        try:
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            return batch_texts, encoded
        except Exception as e:
            # Skip the whole batch in case of tokenization error
            print(
                f"[WARN] Skipping batch of size {len(batch_texts)} due to "
                f"tokenization error: {e}"
            )
            # We still return the raw texts so that the caller can update progress
            return batch_texts, None

    return _collate


def run_inference(
    model,
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    device: str,
    num_workers: int,
    progress: Optional[tqdm] = None,
):
    """Run batched inference to collect CLS vectors and predictions."""

    dataset = TextDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_with_texts(tokenizer),
    )

    results = []
    model.eval()
    close_progress = progress is None
    if progress is None:
        progress = tqdm(total=len(dataset), desc="Inference", unit="text")

    with torch.no_grad():
        for raw_texts, batch_inputs in loader:
            # Батч помечен как битый — пропускаем, но обновляем прогресс
            if batch_inputs is None:
                progress.update(len(raw_texts))
                continue

            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            autocast_enabled = device == "cuda" and torch.cuda.is_available()
            with torch.autocast(device_type=device, enabled=autocast_enabled, dtype=model.dtype):
                outputs = model(**batch_inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            labels = torch.argmax(logits, dim=-1)
            hidden_states = outputs.hidden_states[-1][:, 0, :]

            for text, cls_vec, prob_vec, label in zip(
                raw_texts,
                hidden_states.cpu(),
                probs.cpu(),
                labels.cpu(),
            ):
                results.append(
                    {
                        "text": text,
                        "cls": cls_vec.tolist(),
                        "probs": prob_vec.tolist(),
                        "label": int(label.item()),
                    }
                )
            progress.update(len(raw_texts))

    if close_progress:
        progress.close()

    return results


def save_shard(results: List[dict], output_dir: str, shard_idx: int) -> None:
    """Persist a single shard to disk using the conventional filename pattern."""

    shard_df = pd.DataFrame(results, columns=["text", "cls", "probs", "label"])
    shard_path = os.path.join(output_dir, f"part-{shard_idx:05d}.parquet")
    shard_df.to_parquet(shard_path, index=False)


def discover_existing_shards(output_dir: str) -> Set[int]:
    """Identify shard indices already present on disk for resume support."""

    if not os.path.isdir(output_dir):
        return set()

    shard_indices: Set[int] = set()
    for name in os.listdir(output_dir):
        if name.startswith("part-") and name.endswith(".parquet"):
            middle = name[len("part-") : -len(".parquet")]
            if middle.isdigit():
                shard_indices.add(int(middle))
    return shard_indices


def run(cfg: SuperviseConfig) -> None:
    """Execute supervised inference based on the provided configuration."""

    torch_dtype = getattr(torch, cfg.dtype)
    if cfg.device == "cpu" and torch_dtype != torch.float32:
        print("CPU execution detected; overriding dtype to float32 for compatibility.")
        torch_dtype = torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, use_fast=True, fix_mistral_regex=True
        )
    except TypeError:
        print(
            "Encountered TypeError while applying mistral regex fix; "
            "retrying with fix_mistral_regex=False for tokenizer compatibility."
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, use_fast=True, fix_mistral_regex=False
        )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path, torch_dtype=torch_dtype
    )
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = load_data(cfg.input_parquet)
    shard_size = math.ceil(len(texts) / cfg.num_shards)
    os.makedirs(cfg.output_dir, exist_ok=True)
    existing_shards = discover_existing_shards(cfg.output_dir) if cfg.resume else set()

    processed = 0
    for shard_idx in existing_shards:
        start = shard_idx * shard_size
        end = min(start + shard_size, len(texts))
        processed += max(0, end - start)

    with tqdm(
        total=len(texts),
        initial=processed,
        desc="Inference",
        unit="text",
        disable=not cfg.progress,
    ) as progress:
        for shard_idx in range(cfg.num_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(texts))
            if start >= end:
                break

            if shard_idx in existing_shards:
                progress.update(end - start)
                continue

            shard_texts = texts[start:end]
            shard_results = run_inference(
                model,
                tokenizer,
                shard_texts,
                cfg.batch_size,
                device.type,
                cfg.num_workers,
                progress,
            )
            save_shard(shard_results, cfg.output_dir, shard_idx)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint that loads configuration and runs inference."""

    args = parse_config_path(argv)
    # Each script reads only its own block to avoid coupling between stages.
    cfg = SuperviseConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
