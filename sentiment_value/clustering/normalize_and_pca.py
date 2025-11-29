"""Normalize CLS vectors and apply PCA using YAML configuration."""
from __future__ import annotations

import os
from glob import glob
from typing import Iterable, List, Optional, Sequence

import importlib
import importlib.util
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sentiment_value.clustering.config import NormalizePCAConfig, parse_config_path


def discover_shards(input_dir: str) -> List[str]:
    """Return a sorted list of Parquet shards in ``input_dir``."""

    pattern = os.path.join(input_dir, "*.parquet")
    return sorted(glob(pattern))


def count_rows(shard_paths: Sequence[str]) -> int:
    """Count total rows across shards to size the progress bar."""

    total = 0
    for path in tqdm(shard_paths, desc="Counting rows", unit="shard"):
        df = pd.read_parquet(path, columns=["cls"])
        total += len(df)
    return total


def resolve_array_module(device: str):
    """Resolve the array module (NumPy or CuPy) and optional cuML PCA class."""

    if device != "cuda":
        return np, None, None

    cupy_spec = importlib.util.find_spec("cupy")
    cuml_spec = importlib.util.find_spec("cuml")
    cupy_module = importlib.import_module("cupy") if cupy_spec is not None else None
    cuml_pca_cls = None
    if cuml_spec is not None:
        cuml_decomp = importlib.import_module("cuml.decomposition")
        cuml_pca_cls = getattr(cuml_decomp, "PCA", None)

    return (cupy_module if cupy_module is not None else np), cupy_module, cuml_pca_cls


def l2_normalize(batch, xp):
    """Apply L2 normalization row-wise for a batch of vectors."""

    norms = xp.linalg.norm(batch, axis=1, keepdims=True)
    norms = xp.maximum(norms, xp.asarray(1e-12, dtype=batch.dtype))
    return batch / norms


def iterate_batches(cls_values: Sequence[Sequence[float]], batch_size: int, xp) -> Iterable:
    """Yield mini-batches of CLS vectors as arrays backed by ``xp``."""

    batch: List[Sequence[float]] = []
    for row in cls_values:
        batch.append(row)
        if len(batch) >= batch_size:
            array = xp.asarray(batch, dtype=xp.float32)
            yield array
            batch = []
    if batch:
        array = xp.asarray(batch, dtype=xp.float32)
        yield array


def fit_pca(
    shard_paths: Sequence[str],
    batch_size: int,
    n_components: int,
    xp,
    cupy_module,
    cuml_pca_cls,
    total_rows: Optional[int] = None,
    use_gpu_pca: bool = False,
):
    """Fit PCA incrementally (CPU) or fully on GPU depending on configuration."""

    if use_gpu_pca and cupy_module is not None and cuml_pca_cls is not None:
        normalized_batches = []
        progress = tqdm(total=total_rows, desc="Fitting PCA (GPU)", unit="vec")
        for path in shard_paths:
            df = pd.read_parquet(path, columns=["cls"])
            for batch in iterate_batches(df["cls"].tolist(), batch_size, cupy_module):
                normalized = l2_normalize(batch, cupy_module)
                normalized_batches.append(normalized)
                progress.update(len(batch))
        progress.close()
        if not normalized_batches:
            raise ValueError("No data found to fit PCA.")
        full_matrix = cupy_module.concatenate(normalized_batches, axis=0)
        pca_model = cuml_pca_cls(n_components=n_components)
        pca_model.fit(full_matrix)
        return pca_model

    from sklearn.decomposition import IncrementalPCA

    pca_model = IncrementalPCA(n_components=n_components)
    progress = tqdm(total=total_rows, desc="Fitting PCA (CPU)", unit="vec")
    for path in shard_paths:
        df = pd.read_parquet(path, columns=["cls"])
        for batch in iterate_batches(df["cls"].tolist(), batch_size, xp):
            normalized = l2_normalize(batch, xp)
            cpu_batch = normalized.get() if hasattr(normalized, "get") else normalized
            pca_model.partial_fit(cpu_batch)
            progress.update(len(batch))
    progress.close()
    return pca_model


def transform_shard(
    shard_path: str,
    output_path: str,
    pca_model,
    batch_size: int,
    xp,
    cupy_module,
    use_gpu_pca: bool,
):
    """Normalize and transform a shard, then persist the PCA vectors."""

    df = pd.read_parquet(shard_path)
    cls_values = df["cls"].tolist()
    pca_results: List[Sequence[float]] = []

    progress = tqdm(total=len(df), desc=f"Transform {os.path.basename(shard_path)}", unit="vec")
    for batch in iterate_batches(cls_values, batch_size, xp):
        normalized = l2_normalize(batch, xp)
        if use_gpu_pca and cupy_module is not None and hasattr(pca_model, "transform"):
            pca_batch = pca_model.transform(normalized)
            cpu_batch = pca_batch.get() if hasattr(pca_batch, "get") else pca_batch
        else:
            cpu_batch = normalized.get() if hasattr(normalized, "get") else normalized
            pca_batch = pca_model.transform(cpu_batch)
        pca_results.extend(pca_batch.tolist())
        progress.update(len(batch))
    progress.close()

    df["pca"] = pca_results
    df.to_parquet(output_path, index=False)


def run(cfg: NormalizePCAConfig) -> None:
    """Execute normalization and PCA using configuration from YAML."""

    if cfg.overwrite and cfg.output_dir is None:
        cfg.output_dir = cfg.input_dir
    if not cfg.overwrite and cfg.output_dir is None:
        raise ValueError("output_dir must be provided when not overwriting input shards.")

    xp, cupy_module, cuml_pca_cls = resolve_array_module(cfg.device)
    shard_paths = discover_shards(cfg.input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found in {cfg.input_dir}")

    total_rows = count_rows(shard_paths)
    pca_model = fit_pca(
        shard_paths,
        batch_size=cfg.batch_size,
        n_components=cfg.n_components,
        xp=xp,
        cupy_module=cupy_module,
        cuml_pca_cls=cuml_pca_cls,
        total_rows=total_rows,
        use_gpu_pca=cfg.use_gpu_pca,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    for shard_path in shard_paths:
        filename = os.path.basename(shard_path)
        output_path = os.path.join(cfg.output_dir, filename)
        transform_shard(
            shard_path,
            output_path,
            pca_model,
            batch_size=cfg.batch_size,
            xp=xp,
            cupy_module=cupy_module,
            use_gpu_pca=cfg.use_gpu_pca,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for the normalize and PCA stage."""

    args = parse_config_path(argv)
    cfg = NormalizePCAConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
