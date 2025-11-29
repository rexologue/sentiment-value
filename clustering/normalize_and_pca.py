import argparse
import os
from glob import glob
from typing import Iterable, List, Optional, Sequence

import importlib
import importlib.util
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def discover_shards(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, "*.parquet")
    return sorted(glob(pattern))


def count_rows(shard_paths: Sequence[str]) -> int:
    total = 0
    for path in tqdm(shard_paths, desc="Counting rows", unit="shard"):
        df = pd.read_parquet(path, columns=["cls"])
        total += len(df)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize CLS vectors and apply PCA across shards.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input parquet shards.")
    parser.add_argument(
        "--output_dir",
        help="Directory to write updated shards. Defaults to input_dir when --overwrite is set.",
    )
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components.")
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for normalization and PCA transformation.")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if importlib.util.find_spec("cupy") is not None else "cpu",
        help="Device to run normalization on (PCA may still be CPU-bound).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite shards in place. When set, output_dir defaults to input_dir.",
    )
    parser.add_argument(
        "--use_gpu_pca",
        action="store_true",
        help="Attempt GPU-accelerated PCA with cuML when available (loads all data into memory).",
    )
    return parser.parse_args()


def resolve_array_module(device: str):
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
    norms = xp.linalg.norm(batch, axis=1, keepdims=True)
    norms = xp.maximum(norms, xp.asarray(1e-12, dtype=batch.dtype))
    return batch / norms


def iterate_batches(cls_values: Sequence[Sequence[float]], batch_size: int, xp) -> Iterable:
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


def main():
    args = parse_args()

    if args.overwrite and args.output_dir is None:
        args.output_dir = args.input_dir
    if not args.overwrite and args.output_dir is None:
        raise ValueError("--output_dir must be provided when not overwriting input shards.")

    xp, cupy_module, cuml_pca_cls = resolve_array_module(args.device)
    shard_paths = discover_shards(args.input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found in {args.input_dir}")

    total_rows = count_rows(shard_paths)
    pca_model = fit_pca(
        shard_paths,
        batch_size=args.batch_size,
        n_components=args.n_components,
        xp=xp,
        cupy_module=cupy_module,
        cuml_pca_cls=cuml_pca_cls,
        total_rows=total_rows,
        use_gpu_pca=args.use_gpu_pca,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    for shard_path in shard_paths:
        filename = os.path.basename(shard_path)
        output_path = os.path.join(args.output_dir, filename)
        transform_shard(
            shard_path,
            output_path,
            pca_model,
            batch_size=args.batch_size,
            xp=xp,
            cupy_module=cupy_module,
            use_gpu_pca=args.use_gpu_pca,
        )


if __name__ == "__main__":
    main()
