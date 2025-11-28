# Sentiment Value Training Pipeline

This repository provides a production-ready training pipeline for fine-tuning [`deepvk/USER-base`](https://huggingface.co/deepvk/USER-base) on text classification datasets stored in a single Parquet file with `text` and `label` columns.

## Features
- Parquet dataset loader with automatic label encoding, configurable max sequence length, and train/validation split.
- Optional upsampling of minority classes to match the largest class using a balanced sampler.
- Efficient tokenization, padding, and batching for PyTorch dataloaders.
- Training loop with gradient accumulation, optional mixed precision, gradient clipping, and Neptune logging.
- Validation metrics (accuracy, precision, recall, F1, confusion matrix) with PNG export and logging.
- Periodic and best-checkpoint saving with optimizer/scheduler states and training metadata.
- Single YAML configuration to control hyperparameters, data paths, checkpointing, and Neptune options.

## Quickstart
1. Install dependencies (e.g., `pip install -r requirements.txt` with `torch`, `transformers`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `neptune`).
2. Copy `config.example.yaml` to `config.yaml` and adjust paths and parameters for your environment.
3. Run training:
   ```bash
   python train.py --config config.yaml
   ```

Checkpoints and confusion matrix images are saved to the configured `checkpoints_dir`. Neptune logs capture hyperparameters, losses, metrics, and confusion matrices for each validation run.

## Perfomance

For better perfomance it is recommended to install `flash-attn` after another requirements by following coomnads: 

```bash
python -m pip install ninja
python -m pip install --no-build-isolation -v flash-attn
```
