"""Metric utilities for metric-classifier training."""
from __future__ import annotations

from typing import Dict, Sequence

import torch

from sentiment_value.classifier_training.metrics import compute_metrics, plot_confusion_matrix


def recall_at_k(train_embeddings: torch.Tensor, train_labels: torch.Tensor, val_embeddings: torch.Tensor, val_labels: torch.Tensor, ks: Sequence[int], distance: str = "cos") -> Dict[int, float]:
    """Compute Recall@K for validation embeddings against a training bank."""

    if train_embeddings.numel() == 0 or val_embeddings.numel() == 0:
        return {k: 0.0 for k in ks}

    if distance == "cos":
        similarities = torch.matmul(val_embeddings, train_embeddings.T)
        sorted_indices = torch.argsort(similarities, dim=1, descending=True)
    else:
        # l2 distance
        dists = torch.cdist(val_embeddings, train_embeddings, p=2)
        sorted_indices = torch.argsort(dists, dim=1, descending=False)

    results: Dict[int, float] = {}
    for k in ks:
        topk = sorted_indices[:, :k]
        topk_labels = train_labels[topk]
        matches = (topk_labels == val_labels.unsqueeze(1)).any(dim=1).float()
        results[k] = matches.mean().item()

    return results


__all__ = ["compute_metrics", "plot_confusion_matrix", "recall_at_k"]
