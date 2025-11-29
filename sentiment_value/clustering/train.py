"""MiniBatchKMeans training and shard annotation driven by YAML configuration."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

from functools import partial
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sentiment_value.clustering.config import TrainConfig, parse_config_path
from sentiment_value.clustering.log_utils import get_logger, setup_logging
from sentiment_value.clustering.io import iter_vector_batches, list_parquet_shards, write_parquet_atomic

try:  # Optional GPU support
    from cuml.cluster import MiniBatchKMeans as CuMiniBatchKMeans  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CuMiniBatchKMeans = None

try:
    from sklearn.cluster import MiniBatchKMeans
except ImportError as exc:  # pragma: no cover - sklearn should be present
    raise RuntimeError("scikit-learn is required for clustering") from exc


LOGGER = get_logger(__name__)


def create_kmeans(cfg: TrainConfig):
    """Instantiate a MiniBatchKMeans model honoring CPU/GPU preference."""

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


def train_model(cfg: TrainConfig) -> Tuple[object, int]:
    """Train MiniBatchKMeans using vectors streamed from shards."""

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


def _extract_cluster_centers(model) -> np.ndarray:
    """Retrieve cluster centers from the fitted model in a robust manner."""

    centers = None
    if hasattr(model, "cluster_centers_"):
        centers = getattr(model, "cluster_centers_")
    elif hasattr(model, "cluster_centers"):
        centers = getattr(model, "cluster_centers")

    if centers is None:
        raise ValueError("Trained model does not expose cluster centers for export.")

    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2:
        raise ValueError("Cluster centers should form a 2D array.")

    return centers


def save_centroids(model, path: str, centroid_column: str = "cluster_id") -> None:
    """Persist cluster centroids with their identifiers to Parquet."""

    centers = _extract_cluster_centers(model)
    df = pd.DataFrame(
        {
            centroid_column: np.arange(len(centers), dtype=np.int64),
            "center": centers.tolist(),
        }
    )

    write_parquet_atomic(df, path)


def save_model(model, path: str) -> None:
    """Save the fitted clustering model to disk using joblib."""

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def _validate_for_prediction(vec: object, expected_dim: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Validate and coerce a vector before prediction."""

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


def _predict_for_shard(
    model,
    shard_path: str,
    pca_column: str,
    centroid_column: str,
    expected_dim: Optional[int] = None,
) -> Tuple[str, Optional[int]]:
    """Assign cluster ids for a single shard and rewrite it atomically."""

    df = pd.read_parquet(shard_path)
    if pca_column not in df.columns:
        raise ValueError(f"Column '{pca_column}' not found in shard {shard_path}")

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

    if centroid_column in df.columns:
        LOGGER.info(
            "Overwriting existing '%s' column in shard %s with new assignments",
            centroid_column,
            shard_path,
        )

    df = df.assign(**{centroid_column: centroid_col})
    write_parquet_atomic(df, shard_path)

    return shard_path, expected_dim


_GLOBAL_MODEL = None
_GLOBAL_PCA_COLUMN = None
_GLOBAL_CENTROID_COLUMN = None


def _predict_worker(shard_path: str) -> str:
    assert _GLOBAL_MODEL is not None
    output_path, _ = _predict_for_shard(
        _GLOBAL_MODEL, shard_path, _GLOBAL_PCA_COLUMN, _GLOBAL_CENTROID_COLUMN
    )
    return output_path


def _init_predict_worker(model_path: str, pca_column: str, centroid_column: str) -> None:
    global _GLOBAL_MODEL, _GLOBAL_PCA_COLUMN, _GLOBAL_CENTROID_COLUMN
    _GLOBAL_MODEL = joblib.load(model_path)
    _GLOBAL_PCA_COLUMN = pca_column
    _GLOBAL_CENTROID_COLUMN = centroid_column


def predict_shards(cfg: TrainConfig, model) -> List[str]:
    """Annotate all shards with cluster assignments in place."""

    shard_paths = list_parquet_shards(cfg.input_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.input_dir}")

    expected_dim = getattr(model, "n_features_in_", None)
    if expected_dim is None:
        centers = _extract_cluster_centers(model)
        expected_dim = centers.shape[1]

    if cfg.num_workers <= 1:
        outputs = []
        iterator = tqdm(shard_paths, desc="Predict shards", disable=not cfg.progress)
        for shard_path in iterator:
            output_path, expected_dim = _predict_for_shard(
                model, shard_path, cfg.pca_column, cfg.centroid_column, expected_dim
            )
            outputs.append(output_path)
        return outputs

    from multiprocessing import Pool

    LOGGER.info("Using %s workers for prediction", cfg.num_workers)
    worker_init = partial(_init_predict_worker, cfg.model_out, cfg.pca_column, cfg.centroid_column)

    with Pool(processes=cfg.num_workers, initializer=worker_init) as pool:
        iterator = pool.imap_unordered(_predict_worker, shard_paths)
        results = list(
            tqdm(iterator, total=len(shard_paths), desc="Predict shards", disable=not cfg.progress)
        )

    return results


def run(cfg: TrainConfig) -> None:
    """Run training and/or prediction depending on the configured mode."""

    setup_logging(cfg.log_level)
    LOGGER.info("Loaded config: %s", cfg)

    if cfg.run_mode == "train":
        LOGGER.info("Starting training mode")
        model, vector_dim = train_model(cfg)
        LOGGER.info("Training finished; vector_dim=%s", vector_dim)
        save_model(model, cfg.model_out)
        LOGGER.info("Model saved to %s", cfg.model_out)
        save_centroids(model, cfg.centroids_out, cfg.centroid_column)
        LOGGER.info("Centroids saved to %s", cfg.centroids_out)
        predict_shards(cfg, model)

    else:
        LOGGER.info("Starting prediction-only mode")

        if not os.path.isfile(cfg.model_out):
            raise FileNotFoundError(f"Model file not found: {cfg.model_out}")

        model = joblib.load(cfg.model_out)
        predict_shards(cfg, model)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for clustering training and annotation."""

    args = parse_config_path(argv)
    cfg = TrainConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
