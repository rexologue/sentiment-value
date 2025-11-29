"""I/O utilities for clustering shards."""

import os
import uuid
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def list_parquet_shards(input_dir: str) -> List[str]:
    """Return sorted Parquet shard paths from a directory."""

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    shards = [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.endswith(".parquet")
    ]

    return shards


def _validate_vector(vec: object, expected_dim: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Validate and coerce a vector, ensuring consistent dimensionality."""

    if vec is None:
        return None, expected_dim

    if isinstance(vec, (list, tuple)):
        arr = np.asarray(vec, dtype=np.float32)

    else:
        try:
            arr = np.asarray(vec, dtype=np.float32)
        except Exception:
            return None, expected_dim

    if arr.ndim != 1:
        return None, expected_dim
    if arr.size == 0:
        return None, expected_dim
    if expected_dim is None:
        expected_dim = arr.size
    if arr.size != expected_dim:
        return None, expected_dim

    return arr, expected_dim


def iter_vector_batches(
    shard_path: str,
    column: str,
    batch_size: int,
    expected_dim: Optional[int],
    sample_limit: Optional[int],
    logger,
) -> Tuple[Iterable[np.ndarray], Optional[int], int]:
    """Yield batches of vectors from a shard, validating shape as we stream."""

    file = pq.ParquetFile(shard_path)
    yielded = 0

    for batch in file.iter_batches(columns=[column], batch_size=batch_size):
        vectors: List[np.ndarray] = []
        for value in batch.column(0).to_pylist():
            vec, expected_dim = _validate_vector(value, expected_dim)
            if vec is None:
                logger.warning("Skipping invalid vector in %s", shard_path)
                continue

            vectors.append(vec)
            yielded += 1

            if sample_limit is not None and yielded >= sample_limit:
                break

        if vectors:
            yield np.stack(vectors, axis=0), expected_dim, len(vectors)

        if sample_limit is not None and yielded >= sample_limit:
            break


def load_shard(shard_path: str, columns: List[str]) -> pd.DataFrame:
    """Load specific columns from a Parquet shard into a DataFrame."""
    try:
        table = pq.read_table(shard_path, columns=columns)
    except KeyError as exc:
        available = pq.ParquetFile(shard_path).schema.names
        missing = [col for col in columns if col not in available]
        raise ValueError(
            f"Missing columns {missing} in shard {shard_path}. Available columns: {available}"
        ) from exc

    return table.to_pandas()


def write_parquet_atomic(df: pd.DataFrame, output_path: str) -> None:
    """Write Parquet atomically by swapping a temporary file into place."""

    directory = os.path.dirname(output_path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_name = f".{uuid.uuid4().hex}.parquet.tmp"
    temp_path = os.path.join(directory, temp_name)
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, output_path)


__all__ = [
    "list_parquet_shards",
    "iter_vector_batches",
    "load_shard",
    "write_parquet_atomic",
]
