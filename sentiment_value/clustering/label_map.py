"""Build a cluster_id -> label map using majority labels per cluster."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, Mapping, Optional, Sequence

import yaml
from tqdm.auto import tqdm

from sentiment_value.clustering.config import ClusterLabelMapConfig, parse_config_path
from sentiment_value.clustering.io import list_parquet_shards, load_shard
from sentiment_value.clustering.log_utils import get_logger, setup_logging


LOGGER = get_logger(__name__)


def _accumulate_label_counts(
    shard_paths: Sequence[str],
    cluster_column: str,
    label_column: str,
    progress: bool,
) -> Dict[int, Counter]:
    counts: Dict[int, Counter] = defaultdict(Counter)
    iterator = tqdm(shard_paths, desc="Label map shards", disable=not progress)

    for shard_path in iterator:
        df = load_shard(shard_path, [cluster_column, label_column])
        missing = [col for col in (cluster_column, label_column) if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing} in shard {shard_path}. Cannot build label map."
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
                label_value = int(label)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping non-integer cluster_id/label pair (%s, %s) in shard %s",
                    cluster_id,
                    label,
                    shard_path,
                )
                continue

            counts[cluster_key][label_value] += int(count)

    return counts


def _majority_label(counter: Counter) -> Optional[int]:
    if not counter:
        return None

    most_common = counter.most_common()
    top_count = most_common[0][1]
    candidates = [label for label, count in most_common if count == top_count]
    return int(min(candidates))


def _compute_label_map(cluster_counts: Mapping[int, Counter]) -> Dict[int, int]:
    label_map: Dict[int, int] = {}

    for cluster_id, counter in cluster_counts.items():
        label = _majority_label(counter)
        if label is None:
            LOGGER.warning("Cluster %s has no valid labels; skipping", cluster_id)
            continue

        label_map[int(cluster_id)] = label

    return label_map


def _write_label_map(mapping: Mapping[int, int], output_path: str) -> None:
    if not mapping:
        raise ValueError("Computed label map is empty; nothing to write")

    if output_path.lower().endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(mapping, f, allow_unicode=True, sort_keys=True)


def run(cfg: ClusterLabelMapConfig) -> None:
    setup_logging(cfg.log_level)
    LOGGER.info("Loaded config: %s", cfg)

    shard_paths = list_parquet_shards(cfg.shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No Parquet shards found in {cfg.shards_dir}")

    LOGGER.info(
        "Building cluster label map from %s shards using cluster_column='%s' and label_column='%s'",
        len(shard_paths),
        cfg.cluster_column,
        cfg.label_column,
    )
    counts = _accumulate_label_counts(
        shard_paths, cfg.cluster_column, cfg.label_column, cfg.progress
    )
    label_map = _compute_label_map(counts)

    LOGGER.info("Computed labels for %s clusters", len(label_map))
    _write_label_map(label_map, cfg.output_path)
    LOGGER.info("Saved cluster label map to %s", cfg.output_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_config_path(argv)
    cfg = ClusterLabelMapConfig.from_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
