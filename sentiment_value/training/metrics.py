"""Validation metrics utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

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
    preds = np.asarray(list(preds))
    labels = np.asarray(list(labels))
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average=AVERAGING, zero_division=0)),
        "recall": float(recall_score(labels, preds, average=AVERAGING, zero_division=0)),
        "f1": float(f1_score(labels, preds, average=AVERAGING, zero_division=0)),
    }
    return metrics


def plot_confusion_matrix(labels: Iterable[int], preds: Iterable[int], class_names: Iterable[str]) -> plt.Figure:
    cm = confusion_matrix(list(labels), list(preds), labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return fig


__all__ = ["compute_metrics", "plot_confusion_matrix", "AVERAGING"]
