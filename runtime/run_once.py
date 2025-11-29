import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoConfig, AutoTokenizer

LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single inference with an ONNX model")
    parser.add_argument("--model", required=True, type=Path, help="Path to ONNX model file")
    parser.add_argument("--text", required=True, type=str, help="Input text to classify")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="Device to run inference on"
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


def load_resources(model_path: Path, device: str) -> Tuple[AutoTokenizer, AutoConfig, ort.InferenceSession]:
    model_dir = model_path.parent
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)

    providers = _select_providers(device)
    LOGGER.info("Initializing ONNX Runtime session with providers: %s", providers)
    session = ort.InferenceSession(str(model_path), providers=providers)
    return tokenizer, config, session


def run_inference(session: ort.InferenceSession, tokenizer: AutoTokenizer, text: str):
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]
    probabilities = softmax(logits, axis=-1)
    predictions = np.argmax(probabilities, axis=-1)
    return logits, probabilities, predictions


def main():
    setup_logging()
    args = parse_args()

    tokenizer, config, session = load_resources(args.model, args.device)
    logits, probabilities, predictions = run_inference(session, tokenizer, args.text)

    predicted_label = predictions[0]
    label_name = config.id2label.get(int(predicted_label), str(predicted_label))

    print("Predicted label:", label_name)
    print("Logits:", logits[0].tolist())
    print("Probabilities:", probabilities[0].tolist())


if __name__ == "__main__":
    main()
