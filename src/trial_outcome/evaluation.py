"""Evaluation metrics for predicted probabilities of trial success."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true: ArrayLike, y_prob: ArrayLike, threshold: float = 0.5
) -> dict[str, float]:
    """Score predicted success probabilities against binary outcomes.

    Ranking metrics (ROC AUC, PR AUC) and probability-quality metrics (Brier score,
    log loss) use the raw probabilities; the rest use ``y_prob >= threshold``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "positive_rate": float(y_true.mean()),
        "threshold": threshold,
    }
