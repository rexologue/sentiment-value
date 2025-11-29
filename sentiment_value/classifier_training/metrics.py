"""Metric utilities and plotting helpers for classifier training."""

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


AVERAGING = "macro"  # consistent averaging across metrics


def compute_metrics(preds: Iterable[int], labels: Iterable[int]) -> Dict[str, float]:
    """Compute standard classification metrics on predictions and labels."""

    preds = np.asarray(list(preds))
    labels = np.asarray(list(labels))

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average=AVERAGING, zero_division=0)),
        "recall": float(recall_score(labels, preds, average=AVERAGING, zero_division=0)),
        "f1": float(f1_score(labels, preds, average=AVERAGING, zero_division=0)),
    }

    return metrics


def plot_confusion_matrix(
    labels: Iterable[int],
    preds: Iterable[int],
    class_names: Iterable[str],
) -> plt.Figure:  # type: ignore
    """Plot a normalized confusion matrix for visualization and logging."""

    labels = list(labels)
    preds = list(preds)
    class_names = list(class_names)

    num_classes = len(class_names)

    if not all(0 <= x < num_classes for x in labels):
        raise ValueError(f"Some labels are out of range 0..{num_classes-1}")
    if not all(0 <= x < num_classes for x in preds):
        raise ValueError(f"Some preds are out of range 0..{num_classes-1}")

    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(num_classes)),
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            text_color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{val * 100:.1f}%",
                ha="center",
                va="center",
                color=text_color,
            )

    fig.tight_layout()
    return fig


__all__ = ["compute_metrics", "plot_confusion_matrix", "AVERAGING"]
