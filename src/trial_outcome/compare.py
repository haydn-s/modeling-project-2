"""Compare the three tuned models on each TOP phase: logit, XGBoost, and random forest.

For each phase, every model is tuned on TOP's train and valid splits only
(:mod:`trial_outcome.tuning`): by default with 5-fold cross-validation over both, or with
``--tuning valid`` on the valid split alone, as in :mod:`trial_outcome.baseline`. The selected
models are then scored once on the test split.
Test scores come with paired bootstrap confidence intervals: each resample draws the same test
trials for every model, so the interval for a difference between two models accounts for
their scores moving together.

    python -m trial_outcome.compare --dataset-root data/processed/top_2026-11-22

Outputs go to ``--output-dir`` (default ``reports/models/<dataset root name>``), replacing a
previous run's files:

- ``summary.csv``: each selected model's settings, its tuning score (pooled cross-validation
  or valid, ``split == "tuning"``), and its test score
- ``test_intervals.csv``: each model's test ROC AUC, PR AUC, and Brier score with 95% intervals
- ``test_differences.csv``: XGBoost and the random forest minus the logit, with 95% intervals
- ``tuning_<model>_phase_<P>.csv``: every tuning candidate's validation scores
- ``logit_coefficients_phase_<P>.csv`` and ``importances_<model>_phase_<P>.csv``
- ``figures/test_diagnostics_phase_<P>.png``: ROC, PR, and calibration curves, three models
- ``figures/test_roc_auc.png``: test ROC AUC with 95% intervals, every phase and model
- ``run.json``: configuration, input file hashes, and library versions

The selected models are saved to ``--models-dir`` as ``<model>_phase_<P>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from trial_outcome.baseline import dataset_path, load_phase
from trial_outcome.config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from trial_outcome.data.top import LABELS_DIR, PHASES, labels_filename
from trial_outcome.evaluation import evaluate_probabilities
from trial_outcome.modeling import (
    DEFAULT_C_GRID,
    DEFAULT_CHAIN_LENGTHS,
    feature_importances,
    logit_coefficients,
    predict_success,
    save_model,
)
from trial_outcome.preprocessing import DEFAULT_MIN_COUNT
from trial_outcome.tuning import (
    DEFAULT_FOLDS,
    DEFAULT_FOREST_CANDIDATES,
    DEFAULT_TUNING_MODE,
    DEFAULT_XGBOOST_CANDIDATES,
    SELECTION_METRIC,
    TUNING_MODES,
    TuningResult,
    make_folds,
    tune_logit,
    tune_random_forest,
    tune_xgboost,
)
from trial_outcome.visualization import (
    MUTED,
    SERIES_COLORS,
    STYLE,
    plot_model_comparison,
    save_figure,
)

# Fixed order: the logit is the reference model, and colors follow this order in every figure.
MODELS = ("logit", "xgboost", "random_forest")
MODEL_NAMES = {"logit": "Logit", "xgboost": "XGBoost", "random_forest": "Random forest"}
MODEL_COLORS = dict(zip(MODELS, SERIES_COLORS, strict=True))
MODEL_MARKERS = {"logit": "o", "xgboost": "s", "random_forest": "D"}
REFERENCE_MODEL = "logit"

SUMMARY_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss"]
INTERVAL_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
}
DEFAULT_BOOTSTRAP = 2000
CONFIDENCE = 0.95


def tune_phase(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    chain_lengths: Sequence[int] = DEFAULT_CHAIN_LENGTHS,
    xgboost_candidates: int = DEFAULT_XGBOOST_CANDIDATES,
    forest_candidates: int = DEFAULT_FOREST_CANDIDATES,
    min_count: int = DEFAULT_MIN_COUNT,
    random_state: int = RANDOM_SEED,
    tuning: str = DEFAULT_TUNING_MODE,
    n_folds: int = DEFAULT_FOLDS,
) -> dict[str, TuningResult]:
    """Tune all three models on one phase's train and valid splits, in :data:`MODELS` order.

    All three models are tuned on the same folds and refitted on the same rows.
    """
    folds, refit = make_folds(splits["train"], splits["valid"], tuning, n_folds, random_state)
    return {
        "logit": tune_logit(folds, refit, c_grid, min_count),
        "xgboost": tune_xgboost(
            folds, refit, xgboost_candidates, chain_lengths, min_count, random_state
        ),
        "random_forest": tune_random_forest(
            folds, refit, forest_candidates, min_count, random_state
        ),
    }


def bootstrap_test(
    y_true: pd.Series | np.ndarray,
    probabilities: dict[str, np.ndarray],
    n_resamples: int = DEFAULT_BOOTSTRAP,
    random_state: int = RANDOM_SEED,
    reference: str = REFERENCE_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap intervals for each model's test scores and for differences vs. reference.

    Every resample draws the same trials (with replacement) for all models. Resamples that
    contain only one outcome class are redrawn, since ROC and PR AUC need both. Returns
    ``(intervals, differences)``: one row per model and metric, and one row per non-reference
    model and metric with ``model - reference``. ``share_above_zero`` is the fraction of
    resamples where that difference was positive (higher AUC is better, higher Brier is worse).
    """
    if reference not in probabilities:
        raise ValueError(f"reference model {reference!r} has no predictions")
    y = np.asarray(y_true)
    n = len(y)
    rng = np.random.default_rng(random_state)
    draws = {(m, k): np.empty(n_resamples) for m in probabilities for k in INTERVAL_METRICS}
    done = 0
    while done < n_resamples:
        rows = rng.integers(0, n, size=n)
        if y[rows].min() == y[rows].max():
            continue
        for model, y_prob in probabilities.items():
            for metric, score in INTERVAL_METRICS.items():
                draws[model, metric][done] = score(y[rows], np.asarray(y_prob)[rows])
        done += 1

    low, high = (1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2
    intervals, differences = [], []
    for model, y_prob in probabilities.items():
        for metric, score in INTERVAL_METRICS.items():
            values = draws[model, metric]
            intervals.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": score(y, np.asarray(y_prob)),
                    "ci_low": np.quantile(values, low),
                    "ci_high": np.quantile(values, high),
                }
            )
            if model == reference:
                continue
            delta = values - draws[reference, metric]
            differences.append(
                {
                    "model": model,
                    "reference": reference,
                    "metric": metric,
                    "difference": score(y, np.asarray(y_prob))
                    - score(y, np.asarray(probabilities[reference])),
                    "ci_low": np.quantile(delta, low),
                    "ci_high": np.quantile(delta, high),
                    "share_above_zero": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(intervals), pd.DataFrame(differences)


def plot_test_auc(intervals: pd.DataFrame) -> Figure:
    """Test ROC AUC with 95% intervals: one panel per phase, one dot and whisker per model.

    Every dot is labeled with its value, and each model keeps its color and marker shape in
    every panel, so models are identifiable without relying on color.
    """
    rows = intervals[intervals["metric"] == "roc_auc"]
    phases = list(dict.fromkeys(rows["phase"]))
    low = min(0.5, float(rows["ci_low"].min())) - 0.02
    high = min(1.0, float(rows["ci_high"].max()) + 0.04)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(phases), figsize=(4.2 * len(phases), 4.4), layout="constrained", sharey=True
        )
        axes = np.atleast_1d(axes)
        for ax, phase in zip(axes, phases, strict=True):
            phase_rows = rows[rows["phase"] == phase].set_index("model")
            for x, model in enumerate(m for m in MODELS if m in phase_rows.index):
                row = phase_rows.loc[model]
                ax.vlines(x, row["ci_low"], row["ci_high"], color=MODEL_COLORS[model], lw=2)
                ax.plot(
                    x,
                    row["estimate"],
                    marker=MODEL_MARKERS[model],
                    markersize=9,
                    color=MODEL_COLORS[model],
                    markeredgecolor=STYLE["axes.facecolor"],
                    markeredgewidth=2,
                    linestyle="none",
                    label=MODEL_NAMES[model],
                )
                ax.annotate(
                    f"{row['estimate']:.3f}",
                    (x, row["estimate"]),
                    xytext=(10, 0),
                    textcoords="offset points",
                    va="center",
                    color=STYLE["axes.labelcolor"],
                    fontsize=9,
                )
            ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
            ax.set_xticks(range(len(phase_rows)))
            ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS if m in phase_rows.index])
            ax.set_xlim(-0.5, len(phase_rows) - 0.2)
            ax.set_ylim(low, high)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"Phase {phase}")
        axes[0].set_ylabel("Test ROC AUC (higher is better)")
        fig.suptitle("Test ROC AUC with 95% bootstrap intervals (dotted line: chance)")
    return fig


def run(
    dataset_root: Path,
    labels_dir: Path = LABELS_DIR,
    output_dir: Path | None = None,
    models_dir: Path = MODELS_DIR,
    phases: Sequence[str] = PHASES,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    chain_lengths: Sequence[int] = DEFAULT_CHAIN_LENGTHS,
    xgboost_candidates: int = DEFAULT_XGBOOST_CANDIDATES,
    forest_candidates: int = DEFAULT_FOREST_CANDIDATES,
    min_count: int = DEFAULT_MIN_COUNT,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    random_state: int = RANDOM_SEED,
    tuning: str = DEFAULT_TUNING_MODE,
    n_folds: int = DEFAULT_FOLDS,
) -> pd.DataFrame:
    """Tune and test the three models on each phase, write every output, return the summary."""
    dataset_root, labels_dir = Path(dataset_root), Path(labels_dir)
    output_dir = Path(output_dir or REPORTS_DIR / "models" / dataset_root.name)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_intervals, all_differences = [], [], []
    for phase in phases:
        splits = load_phase(dataset_root, labels_dir, phase)
        results = tune_phase(
            splits,
            c_grid,
            chain_lengths,
            xgboost_candidates,
            forest_candidates,
            min_count,
            random_state,
            tuning,
            n_folds,
        )
        X_test, y_test = splits["test"]
        # The rows scored while tuning: the valid split, or train and valid under cross-validation.
        y_tuning = (
            splits["valid"][1]
            if tuning == "valid"
            else pd.concat([splits["train"][1], splits["valid"][1]])
        )
        test_probabilities = {}
        for model, result in results.items():
            result.candidates.to_csv(output_dir / f"tuning_{model}_phase_{phase}.csv", index=False)
            save_model(result.model, f"{model}_phase_{phase}", models_dir)
            described = {"phase": phase, "model": model, "setting": result.setting}
            tuned = result.candidates.loc[result.candidates["selected"]].iloc[0]
            summaries.append(
                described
                | {"split": "tuning", "n_trials": len(y_tuning), "positive_rate": y_tuning.mean()}
                | {metric: tuned[f"tuning_{metric}"] for metric in SUMMARY_METRICS}
            )
            test_probabilities[model] = predict_success(result.model, X_test)
            scores = evaluate_probabilities(y_test, test_probabilities[model])
            summaries.append(
                described
                | {"split": "test", "n_trials": len(y_test), "positive_rate": y_test.mean()}
                | {metric: scores[metric] for metric in SUMMARY_METRICS}
            )

        logit_coefficients(results["logit"].model).to_csv(
            output_dir / f"logit_coefficients_phase_{phase}.csv", index=False
        )
        for model in ("xgboost", "random_forest"):
            feature_importances(results[model].model).to_csv(
                output_dir / f"importances_{model}_phase_{phase}.csv", index=False
            )

        intervals, differences = bootstrap_test(
            y_test, test_probabilities, n_bootstrap, random_state
        )
        all_intervals.append(intervals.assign(phase=phase))
        all_differences.append(differences.assign(phase=phase))

        fig = plot_model_comparison(
            y_test,
            {MODEL_NAMES[model]: test_probabilities[model] for model in MODELS},
        )
        fig.suptitle(f"Phase {phase} test set: {len(y_test)} trials, {y_test.mean():.0%} succeeded")
        save_figure(fig, f"test_diagnostics_phase_{phase}", figures_dir)
        plt.close(fig)

    summary = pd.DataFrame(summaries)
    intervals = _phase_first(pd.concat(all_intervals, ignore_index=True))
    differences = _phase_first(pd.concat(all_differences, ignore_index=True))
    summary.to_csv(output_dir / "summary.csv", index=False)
    intervals.to_csv(output_dir / "test_intervals.csv", index=False)
    differences.to_csv(output_dir / "test_differences.csv", index=False)

    fig = plot_test_auc(intervals)
    save_figure(fig, "test_roc_auc", figures_dir)
    plt.close(fig)

    _write_run_record(
        output_dir / "run.json",
        dataset_root=dataset_root,
        labels_dir=labels_dir,
        phases=phases,
        config={
            "selection_metric": SELECTION_METRIC,
            "tuning": tuning,
            "n_folds": n_folds if tuning == "cv" else None,
            "c_grid": sorted(set(c_grid)),
            "chain_lengths": sorted(set(chain_lengths)),
            "xgboost_candidates": xgboost_candidates,
            "forest_candidates": forest_candidates,
            "min_count": min_count,
            "n_bootstrap": n_bootstrap,
            "confidence": CONFIDENCE,
            "random_state": random_state,
        },
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Tune and compare the logit, XGBoost, and random forest on the TOP datasets."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="directory holding phase_<P>/"
    )
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, help="default: reports/models/<root name>")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    parser.add_argument("--chain-lengths", nargs="+", type=int, default=list(DEFAULT_CHAIN_LENGTHS))
    parser.add_argument(
        "--xgboost-candidates",
        type=int,
        default=DEFAULT_XGBOOST_CANDIDATES,
        help="XGBoost settings to try, including the baseline defaults",
    )
    parser.add_argument(
        "--forest-candidates",
        type=int,
        default=DEFAULT_FOREST_CANDIDATES,
        help="random forest settings to try, including the defaults",
    )
    parser.add_argument(
        "--tuning",
        choices=TUNING_MODES,
        default=DEFAULT_TUNING_MODE,
        help="cv: k-fold cross-validation over train + valid; valid: the valid split alone",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="folds for --tuning cv")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be at least 1")
    if args.tuning == "cv" and args.folds < 2:
        parser.error("--folds must be at least 2")

    summary = run(
        args.dataset_root,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        phases=args.phases,
        c_grid=args.c_grid,
        chain_lengths=args.chain_lengths,
        xgboost_candidates=args.xgboost_candidates,
        forest_candidates=args.forest_candidates,
        min_count=args.min_count,
        n_bootstrap=args.bootstrap,
        random_state=args.seed,
        tuning=args.tuning,
        n_folds=args.folds,
    )
    test = summary[summary["split"] == "test"].drop(columns=["split", "setting"])
    print(test.to_string(index=False, float_format="%.3f"))


def _phase_first(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["phase", *frame.columns.drop("phase")]]


def _write_run_record(
    path: Path,
    dataset_root: Path,
    labels_dir: Path,
    phases: Sequence[str],
    config: dict[str, Any],
) -> None:
    inputs = {}
    for phase in phases:
        files = [dataset_path(dataset_root, phase), labels_dir / labels_filename(phase)]
        inputs[phase] = {str(file): _sha256(file) for file in files}
    record = {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs,
        "config": config,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    path.write_text(json.dumps(record, indent=2) + "\n")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if __name__ == "__main__":
    main()
