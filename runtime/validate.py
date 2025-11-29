import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from model import OptimizedSequenceClassificationModel

LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate a PyTorch sequence classification model",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to the Hugging Face model directory",
    )
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Dataset file (CSV or Parquet) with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device preference for inference",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="JSON file with mapping rules for model answers",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile even if available",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)
    if data_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported dataset format: {data_path.suffix}")


def load_model(model_dir: Path, device_preference: str, enable_compile: bool):
    prefer_cuda = device_preference in {"auto", "cuda"}
    return OptimizedSequenceClassificationModel(
        model_dir,
        prefer_cuda=prefer_cuda,
        enable_compile=enable_compile,
    )


def iterate_batches(texts: List[str], labels: List[int], batch_size: int):
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        yield texts[start:end], labels[start:end]


def run_validation(
    model: OptimizedSequenceClassificationModel,
    texts: List[str],
    labels: List[int],
    batch_size: int,
):
    all_logits = []
    progress = tqdm(
        iterate_batches(texts, labels, batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Batches",
    )
    for batch_texts, _ in progress:
        batch_logits = model.logits(batch_texts, batch_size=batch_size)
        all_logits.append(batch_logits)
    return torch.cat(all_logits, dim=0)


def main():
    setup_logging()
    args = parse_args()

    model = load_model(args.model_dir, args.device, not args.disable_compile)
    df = load_dataset(args.data)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    # Загружаем mapping при необходимости
    map_tensor = None
    if args.mapping is not None:
        with args.mapping.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Ожидаем ключи "0", "1", ..., "N-1"
        # Явно сортируем по числовому ключу
        max_idx = max(int(k) for k in data.keys())
        mapping_list = [data[str(i)] for i in range(max_idx + 1)]
        map_tensor = torch.tensor(mapping_list, dtype=torch.long)

    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    logits = run_validation(model, texts, labels, args.batch_size)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

    # Применяем mapping, если он есть
    if map_tensor is not None:
        map_tensor = map_tensor.to(predictions.device)
        predictions = map_tensor[predictions]

    predictions = predictions.cpu().tolist()

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro")

    metrics = {"accuracy": float(acc), "macro_f1": float(macro_f1)}

    print(json.dumps(metrics, indent=2))
    LOGGER.info("Validation metrics: %s", metrics)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        LOGGER.info("Saved metrics to %s", args.output_json)


if __name__ == "__main__":
    main()
