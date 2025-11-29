import argparse
import json
import logging
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face sequence classification model to ONNX"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Path to directory with safetensors export (config, tokenizer, weights)",
    )
    parser.add_argument(
        "--onnx-out",
        required=True,
        type=Path,
        help="Destination path for the ONNX model (e.g., /path/to/model.onnx)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run the export on",
    )
    return parser.parse_args()


def _select_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def export_to_onnx(model_dir: Path, onnx_out: Path, device: torch.device) -> None:
    LOGGER.info("Loading model from %s", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    max_length = (
        tokenizer.model_max_length
        if tokenizer.model_max_length and tokenizer.model_max_length < 10000
        else 128
    )
    sample_inputs = tokenizer(
        "Dummy text for ONNX export",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items() if k in {"input_ids", "attention_mask"}}

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }

    onnx_out.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Exporting model to ONNX at %s", onnx_out)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample_inputs["input_ids"], sample_inputs["attention_mask"]),
            f=onnx_out,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=18,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False
        )

    LOGGER.info("Validating ONNX model")
    onnx_model = onnx.load(onnx_out)
    onnx.checker.check_model(onnx_model)

    io_signature = {
        "inputs": [
            {"name": inp.name, "shape": [dim.dim_param or dim.dim_value for dim in inp.type.tensor_type.shape.dim]}
            for inp in onnx_model.graph.input
        ],
        "outputs": [
            {"name": out.name, "shape": [dim.dim_param or dim.dim_value for dim in out.type.tensor_type.shape.dim]}
            for out in onnx_model.graph.output
        ],
    }
    LOGGER.info("Model I/O signature: %s", json.dumps(io_signature, indent=2))
    print(json.dumps(io_signature, indent=2))


def main():
    setup_logging()
    args = parse_args()
    device = _select_device(args.device)
    export_to_onnx(args.model_dir, args.onnx_out, device)


if __name__ == "__main__":
    main()
