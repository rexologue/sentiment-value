"""Post-process clustered shards to build filtered datasets with masks."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from sentiment_value.clustering.config import UpdateDatasetConfig, parse_config_path
from sentiment_value.clustering.io import list_parquet_shards, load_shard, write_parquet_atomic
from sentiment_value.clustering.log_utils import get_logger, setup_logging


LOGGER = get_logger(__name__)


def _load_purity_map(centroids_path: str, cluster_column: str) -> dict[int, float]:
    centroids_df = pd.read_parquet(centroids_path)
    missing = [col for col in (cluster_column, "purity") if col not in centroids_df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in centroids file {centroids_path}. Cannot attach purity."
        )

    purity_map: dict[int, float] = {}
    for cluster_value, purity in zip(
        centroids_df[cluster_column].tolist(), centroids_df["purity"].tolist()
    ):
        try:
            cluster_key = int(cluster_value)
        except (TypeError, ValueError):
            LOGGER.warning("Skipping non-integer cluster id '%s' in centroids", cluster_value)
            continue
        purity_map[cluster_key] = float(purity)

    if not purity_map:
        raise ValueError("No valid cluster_id entries found in centroids file")

    return purity_map


def _max_prob_from_value(value: object) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return float("nan")

    if array.size == 0:
        return float("nan")

    return float(np.max(array))


def _validate_columns(df: pd.DataFrame, required: Iterable[str], shard_path: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in shard {shard_path}. Cannot update dataset."
        )


def _process_shard(
    shard_path: str,
    purity_map: dict[int, float],
    cfg: UpdateDatasetConfig,
    global_purity_threshold: float,
    global_max_prob_threshold: float,
) -> tuple[pd.DataFrame, int, int, int]:
    columns: List[str] = [cfg.text_column, cfg.label_column, cfg.cluster_id_column]
    if cfg.probs_column not in columns:
        columns.append(cfg.probs_column)

    df = load_shard(shard_path, columns)
    _validate_columns(
        df,
        [cfg.text_column, cfg.label_column, cfg.cluster_id_column, cfg.probs_column],
        shard_path,
    )

    total_rows = len(df)
    df["purity"] = df[cfg.cluster_id_column].map(purity_map)

    purity_mask = df["purity"] >= global_purity_threshold
    dropped_purity = int(total_rows - purity_mask.sum())
    df = df[purity_mask].copy()

    df["max_prob"] = df[cfg.probs_column].apply(_max_prob_from_value)

    max_prob_mask = df["max_prob"] >= global_max_prob_threshold
    dropped_max_prob = int(len(df) - max_prob_mask.sum())
    df = df[max_prob_mask].copy()

    df["metric_mask"] = (
        (df["purity"] >= cfg.metric_purity_threshold)
        & (df["max_prob"] >= cfg.metric_max_prob_threshold)
    ).astype(int)

    df["classification_mask"] = (
        (df["purity"] >= cfg.classification_purity_threshold)
        & (df["max_prob"] >= cfg.classification_max_prob_threshold)
    ).astype(int)

    final_df = df[[cfg.text_column, cfg.label_column, "metric_mask", "classification_mask"]]

    return final_df, total_rows, dropped_purity, dropped_max_prob


def run(cfg: UpdateDatasetConfig) -> None:
    setup_logging(cfg.log_level)
    LOGGER.info("Loaded config: %s", cfg)

    global_purity_threshold = min(cfg.metric_purity_threshold, cfg.classification_purity_threshold)
    global_max_prob_threshold = min(cfg.metric_max_prob_threshold, cfg.classification_max_prob_threshold)

    LOGGER.info(
        "Using global thresholds purity>=%.4f, max_prob>=%.4f",
        global_purity_threshold,
        global_max_prob_threshold,
    )

    purity_map = _load_purity_map(cfg.centroids_path, cfg.cluster_id_column)

    shard_paths = list_parquet_shards(cfg.shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.shards_dir}")

    total_rows = 0
    dropped_purity_total = 0
    dropped_max_prob_total = 0
    outputs: List[pd.DataFrame] = []

    iterator = tqdm(shard_paths, desc="Update dataset shards", disable=not cfg.progress)
    for shard_path in iterator:
        shard_df, shard_total, shard_dropped_purity, shard_dropped_max_prob = _process_shard(
            shard_path, purity_map, cfg, global_purity_threshold, global_max_prob_threshold
        )

        total_rows += shard_total
        dropped_purity_total += shard_dropped_purity
        dropped_max_prob_total += shard_dropped_max_prob
        outputs.append(shard_df)

    if outputs:
        final_df = pd.concat(outputs, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=[cfg.text_column, cfg.label_column, "metric_mask", "classification_mask"])

    final_rows = len(final_df)
    metric_fraction = (
        final_df["metric_mask"].sum() / final_rows if final_rows else 0.0
    )
    classification_fraction = (
        final_df["classification_mask"].sum() / final_rows if final_rows else 0.0
    )

    LOGGER.info("Total input rows: %s", total_rows)
    LOGGER.info("Dropped by purity: %s", dropped_purity_total)
    LOGGER.info("Dropped by max_prob: %s", dropped_max_prob_total)
    LOGGER.info("Final dataset rows: %s", final_rows)
    LOGGER.info(
        "Mask fractions - metric: %.4f, classification: %.4f",
        metric_fraction,
        classification_fraction,
    )

    write_parquet_atomic(final_df, cfg.output_path)
    LOGGER.info("Wrote updated dataset to %s", cfg.output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_config_path(argv)
    cfg = UpdateDatasetConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
