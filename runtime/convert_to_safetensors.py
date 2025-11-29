"""Utilities to convert torch checkpoints to safetensors format."""

from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copy2

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a saved model.pt checkpoint to safetensors. Optionally "
            "rebuild a full Hugging Face folder using the original model files."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to the checkpoint directory or the model.pt file to convert.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where the safetensors (and optional model files) will be saved.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Optional path or model identifier for the original pretrained model. "
            "When provided, the script loads the tokenizer and config from this "
            "location and saves a full Hugging Face compatible folder alongside "
            "the converted safetensors weights."
        ),
    )
    parser.add_argument(
        "--copy-extra",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of additional files from the checkpoint directory to copy to the "
            "output directory (e.g. state.json, confusion matrix images)."
        ),
    )
    parser.add_argument(
        "--strip-encoder-prefix",
        action="store_true",
        help="Remove the 'encoder.' prefix from all keys in the state_dict."
    )
    return parser.parse_args()


def strip_encoder_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_k = k[len("encoder."):]
        else:
            new_k = k
        cleaned[new_k] = v
    return cleaned


def resolve_checkpoint_path(checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    return checkpoint_path


def save_safetensors(state_dict: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.safetensors"
    save_file(state_dict, str(output_path))
    return output_path


def build_full_model_folder(
    state_dict: dict, model_path: str, output_dir: Path
) -> None:
    num_labels = None
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None:
        num_labels = classifier_weight.shape[0]

    config = AutoConfig.from_pretrained(model_path)
    if num_labels is not None and config.num_labels != num_labels:
        config.num_labels = num_labels
        config.id2label = {i: str(i) for i in range(num_labels)}
        config.label2id = {label: idx for idx, label in config.id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config
    )
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(output_dir)


def copy_extra_files(checkpoint_dir: Path, output_dir: Path, extras: list[str]) -> None:
    for extra in extras:
        source = checkpoint_dir / extra
        if source.exists():
            copy2(source, output_dir / source.name)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    output_dir = Path(args.output_dir)

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if args.strip_encoder_prefix:
        state_dict = strip_encoder_prefix(state_dict)

    save_safetensors(state_dict, output_dir)

    if args.model_path:
        build_full_model_folder(state_dict, args.model_path, output_dir)

    if args.copy_extra:
        checkpoint_dir = checkpoint_path.parent
        copy_extra_files(checkpoint_dir, output_dir, args.copy_extra)

    print(f"Converted checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
