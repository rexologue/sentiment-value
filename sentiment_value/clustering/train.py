from __future__ import annotations

import os
from functools import partial
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sentiment_value.clustering.config import Config, load_config_from_cli
from sentiment_value.clustering.log_utils import get_logger, setup_logging
from sentiment_value.clustering.io import iter_vector_batches, list_parquet_shards, load_shard, write_parquet_atomic

try:  # Optional GPU support
    from cuml.cluster import MiniBatchKMeans as CuMiniBatchKMeans # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CuMiniBatchKMeans = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as exc:  # pragma: no cover - sklearn should be present
    raise RuntimeError("scikit-learn is required for clustering") from exc


LOGGER = get_logger(__name__)


def create_kmeans(cfg: Config):
    if cfg.gpu:
        if CuMiniBatchKMeans is None:
            LOGGER.warning("GPU clustering requested but CuML is not available; falling back to CPU")

        else:
            LOGGER.info("Using CuML MiniBatchKMeans on GPU")
            return CuMiniBatchKMeans(
                n_clusters=cfg.n_clusters,
                batch_size=cfg.batch_size,
                max_iter=cfg.max_iter,
                init_size=cfg.init_size,
                random_state=cfg.random_state,
                verbose=0,
            )

    LOGGER.info("Using scikit-learn MiniBatchKMeans on CPU")
    return MiniBatchKMeans(
        n_clusters=cfg.n_clusters,
        batch_size=cfg.batch_size,
        max_iter=cfg.max_iter,
        init_size=cfg.init_size,
        random_state=cfg.random_state,
        verbose=0,
    )


def train_model(cfg: Config) -> Tuple[object, int]:
    shard_paths = list_parquet_shards(cfg.input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.input_dir}")

    model = create_kmeans(cfg)
    expected_dim: Optional[int] = None
    total_vectors = 0

    shard_progress = tqdm(shard_paths, desc="Training shards", disable=not cfg.progress)
    for shard_path in shard_progress:
        if cfg.sample_limit is not None and total_vectors >= cfg.sample_limit:
            break

        batch_iter = iter_vector_batches(
            shard_path,
            cfg.pca_column,
            cfg.batch_size,
            expected_dim,
            cfg.sample_limit - total_vectors if cfg.sample_limit is not None else None,
            LOGGER,
        )

        for vectors, expected_dim, batch_count in batch_iter:
            model.partial_fit(vectors)
            total_vectors += batch_count
            shard_progress.set_postfix(vectors=total_vectors)

            if cfg.sample_limit is not None and total_vectors >= cfg.sample_limit:
                LOGGER.info("Reached sample_limit=%s; stopping training", cfg.sample_limit)
                shard_progress.close()
                return model, expected_dim or 0

    return model, expected_dim or 0


def save_centroids(model, path: str) -> None:
    centers = getattr(model, "cluster_centers_", None)
    if centers is None:
        raise ValueError("Model does not expose cluster_centers_.")
    
    centers = np.asarray(centers)

    df = pd.DataFrame(
        {
            "centroid_id": np.arange(len(centers), dtype=np.int64),
            "center": centers.tolist(),
        }
    )

    write_parquet_atomic(df, path)


def save_model(model, path: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def _predict_for_shard(
    model,
    shard_path: str,
    output_dir: str,
    pca_column: str,
    expected_dim: Optional[int] = None,
) -> Tuple[str, Optional[int]]:
    
    df = load_shard(shard_path, columns=["text", pca_column])
    valid_vectors: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, vec in enumerate(df[pca_column].tolist()):
        arr, expected_dim = _validate_for_prediction(vec, expected_dim)
        if arr is None:
            LOGGER.warning("Skipping invalid vector in %s at row %s", shard_path, idx)
            continue

        valid_vectors.append(arr)
        valid_indices.append(idx)

    if valid_vectors:
        vectors = np.stack(valid_vectors, axis=0)
        preds = model.predict(vectors)

    else:
        preds = np.array([], dtype=np.int64)

    centroid_col = np.full(len(df), -1, dtype=np.int64)
    for pos, row_idx in enumerate(valid_indices):
        centroid_col[row_idx] = int(preds[pos])

    df = df.assign(centroid_id=centroid_col)
    output_path = os.path.join(output_dir, os.path.basename(shard_path))
    write_parquet_atomic(df, output_path)

    return output_path, expected_dim


def _validate_for_prediction(vec: object, expected_dim: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if vec is None:
        return None, expected_dim
    
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        return None, expected_dim
    
    if expected_dim is None:
        expected_dim = arr.size

    if arr.size != expected_dim:
        return None, expected_dim
    
    return arr, expected_dim


_GLOBAL_MODEL = None
_GLOBAL_PCA_COLUMN = None


def _predict_worker(shard_path: str, output_dir: str) -> str:
    assert _GLOBAL_MODEL is not None
    output_path, _ = _predict_for_shard(
        _GLOBAL_MODEL, shard_path, output_dir, _GLOBAL_PCA_COLUMN
    )

    return output_path


def _init_predict_worker(model_path: str, pca_column: str) -> None:
    global _GLOBAL_MODEL, _GLOBAL_PCA_COLUMN
    _GLOBAL_MODEL = joblib.load(model_path)
    _GLOBAL_PCA_COLUMN = pca_column


def predict_shards(cfg: Config, model) -> List[str]:
    shard_paths = list_parquet_shards(cfg.input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.input_dir}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    expected_dim = getattr(model, "n_features_in_", None)
    if expected_dim is None:
        centers = getattr(model, "cluster_centers_", None)
        if centers is not None:
            expected_dim = np.asarray(centers).shape[1]

    if cfg.num_workers <= 1:
        outputs = []
        iterator = tqdm(shard_paths, desc="Predict shards", disable=not cfg.progress)

        for shard_path in iterator:
            output_path, expected_dim = _predict_for_shard(
                model, shard_path, cfg.output_dir, cfg.pca_column, expected_dim
            )

            outputs.append(output_path)

        return outputs

    from multiprocessing import Pool

    LOGGER.info("Using %s workers for prediction", cfg.num_workers)
    worker_init = partial(_init_predict_worker, cfg.model_out, cfg.pca_column)

    with Pool(processes=cfg.num_workers, initializer=worker_init) as pool:
        iterator = pool.imap_unordered(partial(_predict_worker, output_dir=cfg.output_dir), shard_paths)
        results = list(tqdm(iterator, total=len(shard_paths), desc="Predict shards", disable=not cfg.progress))

    return results


def run(cfg: Config) -> None:
    setup_logging(cfg.log_level)
    LOGGER.info("Loaded config: %s", cfg)

    if cfg.run_mode == "train":
        LOGGER.info("Starting training mode")
        model, vector_dim = train_model(cfg)
        LOGGER.info("Training finished; vector_dim=%s", vector_dim)
        save_model(model, cfg.model_out)
        LOGGER.info("Model saved to %s", cfg.model_out)
        save_centroids(model, cfg.centroids_out)
        LOGGER.info("Centroids saved to %s", cfg.centroids_out)
        predict_shards(cfg, model)

    else:
        LOGGER.info("Starting prediction-only mode")

        if not os.path.isfile(cfg.model_out):
            raise FileNotFoundError(f"Model file not found: {cfg.model_out}")
        
        model = joblib.load(cfg.model_out)
        predict_shards(cfg, model)


if __name__ == "__main__":
    configuration = load_config_from_cli()
    run(configuration)
