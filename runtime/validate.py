import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ONNX sequence classification model")
    parser.add_argument("--model", required=True, type=Path, help="Path to ONNX model")
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Dataset file (CSV or Parquet) with 'text' and 'label' columns",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for validation")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON",
    )
    return parser.parse_args()


def _select_providers(device: str) -> List[str]:
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    selected = [p for p in providers if p in available]
    if not selected:
        LOGGER.warning("Requested providers %s not available. Falling back to CPU.", providers)
        return ["CPUExecutionProvider"]
    if providers[0] not in selected:
        LOGGER.warning("Primary provider %s unavailable. Using %s instead.", providers[0], selected[0])
    return selected


def load_dataset(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        return pd.read_parquet(data_path)
    if data_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported dataset format: {data_path.suffix}")


def load_resources(model_path: Path, device: str):
    model_dir = model_path.parent
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    providers = _select_providers(device)
    LOGGER.info("Initializing ONNX Runtime session with providers: %s", providers)
    session = ort.InferenceSession(str(model_path), providers=providers)
    return tokenizer, session


def iterate_batches(texts: List[str], labels: List[int], batch_size: int):
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        yield texts[start:end], labels[start:end]


def run_validation(tokenizer, session: ort.InferenceSession, texts: List[str], labels: List[int], batch_size: int):
    all_logits = []
    progress = tqdm(
        iterate_batches(texts, labels, batch_size),
        total=(len(texts) + batch_size - 1) // batch_size,
        desc="Batches",
    )
    for batch_texts, _ in progress:
        inputs = tokenizer(
            batch_texts,
            return_tensors="np",
            truncation=True,
            padding=True,
        )
        ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        outputs = session.run(None, ort_inputs)
        logits = outputs[0]
        all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def main():
    setup_logging()
    args = parse_args()

    tokenizer, session = load_resources(args.model, args.device)
    df = load_dataset(args.data)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].tolist()

    logits = run_validation(tokenizer, session, texts, labels, args.batch_size)
    probabilities = softmax(logits, axis=-1)
    predictions = np.argmax(probabilities, axis=-1)

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
