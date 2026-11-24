"""Plotting helpers for model evaluation and presentation figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator
from numpy.typing import ArrayLike
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from trial_outcome.config import FIGURES_DIR

# Categorical series colors in fixed order; these three stay distinguishable with color-vision
# deficiencies. Color follows the model: the logit is always slot 1, XGBoost slot 2.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
MODEL_COLORS = {"logit": SERIES_COLORS[0], "xgboost": SERIES_COLORS[1]}
MUTED = "#898781"

STYLE = {
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#52514e",
    "axes.titlecolor": "#0b0b0b",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": "#52514e",
    "ytick.labelcolor": "#52514e",
    "text.color": "#0b0b0b",
    "legend.frameon": False,
    "lines.linewidth": 2,
}

METRIC_LABELS = {
    "roc_auc": "ROC AUC (higher is better)",
    "pr_auc": "PR AUC (higher is better)",
    "brier": "Brier score (lower is better)",
    "log_loss": "Log loss (lower is better)",
}


def plot_model_comparison(
    y_true: ArrayLike, probabilities: Mapping[str | None, ArrayLike], n_bins: int = 10
) -> Figure:
    """ROC, precision-recall, and calibration curves for up to three models on the same trials."""
    if not 1 <= len(probabilities) <= len(SERIES_COLORS):
        raise ValueError(f"compare between 1 and {len(SERIES_COLORS)} models")
    with plt.rc_context(STYLE):
        fig, (ax_roc, ax_pr, ax_cal) = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
        chance = {"color": MUTED, "linestyle": ":", "linewidth": 1}
        for i, ((name, y_prob), color) in enumerate(
            zip(probabilities.items(), SERIES_COLORS, strict=False)
        ):
            first = i == 0
            curve = {"color": color}
            RocCurveDisplay.from_predictions(
                y_true,
                y_prob,
                name=name,
                ax=ax_roc,
                curve_kwargs=curve,
                plot_chance_level=first,
                chance_level_kw=chance,
            )
            PrecisionRecallDisplay.from_predictions(
                y_true,
                y_prob,
                name=name,
                ax=ax_pr,
                curve_kwargs=curve,
                plot_chance_level=first,
                chance_level_kw=chance,
            )
            if first:
                ax_cal.plot([0, 1], [0, 1], label="Perfectly calibrated", **chance)
            CalibrationDisplay.from_predictions(
                y_true,
                y_prob,
                n_bins=n_bins,
                name=name,
                ax=ax_cal,
                ref_line=False,
                color=color,
                marker="o",
                markersize=5,
            )
        ax_roc.set_title("ROC curve")
        ax_pr.set_title("Precision-recall curve")
        ax_cal.set_title("Calibration")
        ax_cal.set_xlabel("Mean predicted probability of success")
        ax_cal.set_ylabel("Fraction of trials that succeeded")
        ax_cal.legend(loc="upper left")
    return fig


def plot_model_diagnostics(
    y_true: ArrayLike, y_prob: ArrayLike, name: str | None = None, n_bins: int = 10
) -> Figure:
    """ROC curve, precision-recall curve, and calibration curve side by side."""
    return plot_model_comparison(y_true, {name: y_prob}, n_bins=n_bins)


def plot_chain_length(
    metrics: pd.DataFrame,
    metric_names: Sequence[str] = ("roc_auc", "pr_auc", "brier", "log_loss"),
) -> Figure:
    """XGBoost scores by boosting chain length, against the selected logit.

    ``metrics`` holds one phase's rows from :func:`trial_outcome.baseline.compare_phase`.
    XGBoost is drawn per chain length and the selected logit as a flat reference line; solid
    lines are the validation split and dashed lines the test split. A dotted vertical line marks
    the chain length selected on validation.
    """
    xgboost = metrics[metrics["model"] == "xgboost"]
    logit = metrics[(metrics["model"] == "logit") & metrics["selected"]]
    best_C = logit["C"].iloc[0]
    best_n = int(xgboost.loc[xgboost["selected"], "n_trees"].iloc[0])
    lengths = np.sort(xgboost["n_trees"].unique().astype(int))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained", sharex=True)
        for ax, metric in zip(axes.flat, metric_names, strict=False):
            for split, linestyle in (("valid", "-"), ("test", "--")):
                rows = xgboost[xgboost["split"] == split].sort_values("n_trees")
                ax.plot(
                    rows["n_trees"].astype(int),
                    rows[metric],
                    color=MODEL_COLORS["xgboost"],
                    linestyle=linestyle,
                    marker="o",
                    markersize=5,
                    label=f"XGBoost, {split}",
                )
                ax.axhline(
                    logit.loc[logit["split"] == split, metric].iloc[0],
                    color=MODEL_COLORS["logit"],
                    linestyle=linestyle,
                    label=f"Logit (C={best_C:g}), {split}",
                )
            ax.axvline(
                best_n,
                color=MUTED,
                linestyle=":",
                linewidth=1,
                label=f"Selected chain length ({best_n})",
            )
            ax.set_title(METRIC_LABELS.get(metric, metric))
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(lengths))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticklabels([str(n) for n in lengths])
        for ax in axes[-1]:
            ax.set_xlabel("Boosting chain length (number of trees)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
    return fig


def save_figure(fig: Figure, name: str, figures_dir: Path = FIGURES_DIR, dpi: int = 200) -> Path:
    """Save ``fig`` to ``reports/figures/<name>.png`` and return the path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
