"""Loss functions for joint training."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    """Compute cross-entropy on masked examples."""

    per_example_loss = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
    valid = mask > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    loss = (per_example_loss * mask).sum() / mask.sum()
    return loss


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Compute Supervised Contrastive Loss on a masked subset."""

    valid = mask > 0
    if valid.sum() <= 1:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    features = features[valid]
    labels = labels[valid]

    similarity_matrix = torch.matmul(features, features.T) / max(temperature, 1e-8)
    logits_mask = torch.ones_like(similarity_matrix) - torch.eye(features.size(0), device=features.device)
    exp_logits = torch.exp(similarity_matrix) * logits_mask

    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * logits_mask
    positive_logits = exp_logits * positive_mask

    denominator = exp_logits.sum(dim=1, keepdim=True)
    log_prob = torch.log((positive_logits.sum(dim=1, keepdim=True) + 1e-12) / (denominator + 1e-12))

    num_positives = positive_mask.sum(dim=1)
    valid_positions = num_positives > 0
    if valid_positions.sum() == 0:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    loss = -(log_prob[valid_positions].sum() / num_positives[valid_positions].sum())
    return loss


__all__ = ["masked_cross_entropy", "supervised_contrastive_loss"]
