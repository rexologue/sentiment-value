import argparse
import logging
from pathlib import Path

import torch

from model import OptimizedSequenceClassificationModel

LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single inference with a PyTorch sequence classification model",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to the Hugging Face model directory",
    )
    parser.add_argument("--text", required=True, type=str, help="Input text to classify")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device preference for inference",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile even if available",
    )
    return parser.parse_args()


def load_model(model_dir: Path, device_preference: str, enable_compile: bool):
    prefer_cuda = device_preference in {"auto", "cuda"}
    return OptimizedSequenceClassificationModel(
        model_dir,
        prefer_cuda=prefer_cuda,
        enable_compile=enable_compile,
    )


def run_inference(model: OptimizedSequenceClassificationModel, text: str):
    probabilities = model.probabilities(text)
    predictions = torch.argmax(probabilities, dim=-1)
    return probabilities, predictions


def main():
    setup_logging()
    args = parse_args()

    model = load_model(args.model_dir, args.device, not args.disable_compile)
    probabilities, predictions = run_inference(model, args.text)

    predicted_label = int(predictions[0])
    label_name = model.config.id2label.get(predicted_label, str(predicted_label))

    print("Predicted label:", label_name)
    print("Probabilities:", probabilities[0].tolist())


if __name__ == "__main__":
    main()
