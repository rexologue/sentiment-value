"""Compute cluster purity metrics and attach them to centroid data."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from sentiment_value.clustering.config import PurityCounterConfig, parse_config_path
from sentiment_value.clustering.io import (
    list_parquet_shards,
    load_shard,
    write_parquet_atomic,
)
from sentiment_value.clustering.log_utils import get_logger, setup_logging


LOGGER = get_logger(__name__)


def _accumulate_label_counts(
    shard_paths: Sequence[str],
    cluster_column: str,
    label_column: str,
    progress: bool,
) -> Dict[int, Counter]:
    counts: Dict[int, Counter] = defaultdict(Counter)
    iterator = tqdm(shard_paths, desc="Purity shards", disable=not progress)

    for shard_path in iterator:
        df = load_shard(shard_path, [cluster_column, label_column])
        missing = [col for col in (cluster_column, label_column) if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing} in shard {shard_path}. Cannot compute purity."
            )

        initial_rows = len(df)
        df = df.dropna(subset=[cluster_column, label_column])
        dropped = initial_rows - len(df)
        if dropped:
            LOGGER.warning(
                "Dropped %s rows without %s/%s from shard %s",
                dropped,
                cluster_column,
                label_column,
                shard_path,
            )

        grouped = df.groupby([cluster_column, label_column]).size()
        for (cluster_id, label), count in grouped.items():
            try:
                cluster_key = int(cluster_id)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping non-integer cluster id '%s' in shard %s", cluster_id, shard_path
                )
                continue

            counts[cluster_key][label] += int(count)

    return counts


def _compute_purity(cluster_counts: Mapping[int, Counter]) -> Dict[int, float]:
    purity: Dict[int, float] = {}

    for cluster_id, counter in cluster_counts.items():
        total = sum(counter.values())
        if total == 0:
            purity[cluster_id] = float("nan")
            continue

        dominant = max(counter.values())
        purity[cluster_id] = dominant / float(total)

    return purity


def _write_purity_to_centroids(
    centroids_path: str, cluster_column: str, purity: Mapping[int, float]
) -> None:
    centroids_df = pd.read_parquet(centroids_path)
    if cluster_column not in centroids_df.columns:
        raise ValueError(
            f"Column '{cluster_column}' not found in centroids file {centroids_path}"
        )

    purity_df = (
        pd.DataFrame({cluster_column: list(purity.keys()), "purity": list(purity.values())})
        .drop_duplicates(subset=[cluster_column])
        .set_index(cluster_column)
    )

    missing = set(centroids_df[cluster_column]) - set(purity_df.index)
    if missing:
        LOGGER.warning(
            "No purity values found for %s cluster(s) present in centroids: %s",
            len(missing),
            sorted(missing),
        )

    centroid_indexed = centroids_df.set_index(cluster_column)
    centroid_indexed["purity"] = purity_df["purity"]
    centroid_indexed = centroid_indexed.reset_index()

    write_parquet_atomic(centroid_indexed, centroids_path)


def run(cfg: PurityCounterConfig) -> None:
    setup_logging(cfg.log_level)
    LOGGER.info("Loaded config: %s", cfg)

    shard_paths = list_parquet_shards(cfg.shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.shards_dir}")

    LOGGER.info(
        "Computing purity using cluster_column='%s' and teacher_label_column='%s'",
        cfg.cluster_column,
        cfg.teacher_label_column,
    )
    counts = _accumulate_label_counts(
        shard_paths, cfg.cluster_column, cfg.teacher_label_column, cfg.progress
    )
    purity = _compute_purity(counts)
    LOGGER.info("Computed purity for %s clusters", len(purity))

    _write_purity_to_centroids(cfg.centroids_path, cfg.cluster_column, purity)
    LOGGER.info("Wrote purity scores to %s", cfg.centroids_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_config_path(argv)
    cfg = PurityCounterConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
