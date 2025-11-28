import argparse
import math
import os
from typing import List, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched inference and save clustered features.")
    parser.add_argument("--model_name_or_path", required=True, help="Path or identifier of the model to load.")
    parser.add_argument(
        "--input",
        "--input_parquet",
        dest="input_parquet",
        required=True,
        help="Input Parquet file containing a 'text' column.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write output shards.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of output shards to split results into.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Floating point precision to load the model with.",
    )
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def load_data(parquet_path: str) -> List[str]:
    df = pd.read_parquet(parquet_path, columns=["text"])
    return df["text"].tolist()


def collate_with_texts(tokenizer):
    def _collate(batch_texts: List[str]) -> Tuple[List[str], dict]:
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch_texts, encoded

    return _collate


def run_inference(model, tokenizer, texts: Sequence[str], batch_size: int, device: str, num_workers: int):
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
    with torch.no_grad(), tqdm(total=len(dataset), desc="Inference", unit="text") as progress:
        for raw_texts, batch_inputs in loader:
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
    return results


def shard_and_save(results: List[dict], output_dir: str, num_shards: int):
    os.makedirs(output_dir, exist_ok=True)
    total = len(results)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")

    shard_size = math.ceil(total / num_shards)
    for shard_idx in range(num_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        if start >= end:
            break
        shard_df = pd.DataFrame(results[start:end], columns=["text", "cls", "probs", "label"])
        shard_path = os.path.join(output_dir, f"part-{shard_idx:05d}.parquet")
        shard_df.to_parquet(shard_path, index=False)


def main():
    args = parse_args()

    torch_dtype = getattr(torch, args.dtype)
    if args.device == "cpu" and torch_dtype != torch.float32:
        print("CPU execution detected; overriding dtype to float32 for compatibility.")
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, fix_mistral_regex=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, torch_dtype=torch_dtype
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = load_data(args.input_parquet)
    results = run_inference(model, tokenizer, texts, args.batch_size, device.type, args.num_workers)
    shard_and_save(results, args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
