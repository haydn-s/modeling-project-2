"""Plotting helpers for model evaluation and presentation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from trial_outcome.config import FIGURES_DIR


def plot_model_diagnostics(
    y_true: ArrayLike, y_prob: ArrayLike, name: str | None = None, n_bins: int = 10
) -> Figure:
    """ROC curve, precision-recall curve, and calibration curve side by side."""
    fig, (ax_roc, ax_pr, ax_cal) = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    RocCurveDisplay.from_predictions(y_true, y_prob, name=name, ax=ax_roc, plot_chance_level=True)
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, name=name, ax=ax_pr, plot_chance_level=True
    )
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=n_bins, name=name, ax=ax_cal)
    ax_roc.set_title("ROC curve")
    ax_pr.set_title("Precision-recall curve")
    ax_cal.set_title("Calibration")
    return fig


def save_figure(fig: Figure, name: str, figures_dir: Path = FIGURES_DIR, dpi: int = 200) -> Path:
    """Save ``fig`` to ``reports/figures/<name>.png`` and return the path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
