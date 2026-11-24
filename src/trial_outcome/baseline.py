"""Baseline comparison on the TOP datasets: regularized logit vs. XGBoost by chain length.

For each phase, every model is fitted on TOP's train split and scored on train, valid, and test.
The logit's ``C`` and XGBoost's chain length (number of boosting rounds) are each chosen by
validation log loss. Report the selected models' test scores. The other candidates' test scores
are only for drawing the comparison curves; picking a model by them would leak the test set.

XGBoost is fitted once per phase at the longest chain length, and each shorter length is scored
from that chain's first trees, which gives the same predictions as fitting the shorter chain.

    python -m trial_outcome.baseline --dataset-root data/processed/top_2026-11-22

Outputs go to ``--output-dir`` (default ``reports/baseline/<dataset root name>``), replacing a
previous run's files:

- ``metrics.csv``: every phase, model, setting, and split, with a ``selected`` flag
- ``summary.csv``: the selected logit and XGBoost for each phase, on valid and test
- ``logit_coefficients_phase_<P>.csv``: the selected logit's coefficients, largest first
- ``figures/chain_length_phase_<P>.png`` and ``figures/test_diagnostics_phase_<P>.png``
- ``run.json``: configuration, input file hashes, and library versions

The selected models are saved to ``--models-dir`` as ``baseline_{logit,xgboost}_phase_<P>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from trial_outcome.config import MODELS_DIR, REPORTS_DIR
from trial_outcome.data.top import LABELS_DIR, PHASES, SPLITS, labels_filename, load_splits
from trial_outcome.evaluation import evaluate_probabilities
from trial_outcome.modeling import (
    DEFAULT_C_GRID,
    DEFAULT_CHAIN_LENGTHS,
    XGBOOST_DEFAULTS,
    logit_coefficients,
    make_logit,
    make_xgboost,
    predict_success,
    save_model,
)
from trial_outcome.preprocessing import DEFAULT_MIN_COUNT
from trial_outcome.visualization import plot_chain_length, plot_model_comparison, save_figure

SELECTION_METRIC = "log_loss"
SUMMARY_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss"]

Splits = dict[str, tuple[pd.DataFrame, pd.Series]]


@dataclass(frozen=True)
class PhaseComparison:
    """One phase's scores, plus the logit and XGBoost selected on validation log loss."""

    metrics: pd.DataFrame
    logit: Pipeline
    xgboost: Pipeline
    best_C: float
    best_n_trees: int


def dataset_path(dataset_root: Path, phase: str) -> Path:
    return Path(dataset_root) / f"phase_{phase}" / "dataset.parquet"


def load_phase(dataset_root: Path, labels_dir: Path, phase: str) -> Splits:
    """Split one phase's ``dataset.parquet`` into TOP's train, valid, and test partitions."""
    dataset = pd.read_parquet(dataset_path(dataset_root, phase))
    split = load_splits(Path(labels_dir) / labels_filename(phase)).reindex(dataset.index)
    if split.isna().any():
        missing = split.index[split.isna()]
        raise ValueError(
            f"phase {phase}: {len(missing)} trials have no TOP split: {missing[:3].tolist()}"
        )
    features, labels = dataset.drop(columns="label"), dataset["label"]
    return {name: (features[split == name], labels[split == name]) for name in SPLITS}


def compare_phase(
    splits: Splits,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    chain_lengths: Sequence[int] = DEFAULT_CHAIN_LENGTHS,
    min_count: int = DEFAULT_MIN_COUNT,
    xgboost_params: dict[str, Any] | None = None,
) -> PhaseComparison:
    """Fit every candidate on the train split, score it on every split, and select on valid.

    Ties go to the earlier candidate, so list ``c_grid`` from strongest regularization and
    ``chain_lengths`` from shortest.
    """
    xgboost_params = xgboost_params or {}
    X_train, y_train = splits["train"]
    rows = []

    logits = {C: make_logit(C, min_count).fit(X_train, y_train) for C in c_grid}
    for C, model in logits.items():
        rows += _score(splits, model, "logit", C=C)

    longest = make_xgboost(max(chain_lengths), min_count, **xgboost_params).fit(X_train, y_train)
    for n_trees in chain_lengths:
        rows += _score(splits, longest, "xgboost", n_trees=n_trees)

    metrics = pd.DataFrame(rows)
    metrics["n_trees"] = metrics["n_trees"].astype("Int64")
    valid = metrics[metrics["split"] == "valid"]
    best = {
        name: valid.loc[valid.loc[valid["model"] == name, SELECTION_METRIC].idxmin()]
        for name in ("logit", "xgboost")
    }
    best_C, best_n_trees = float(best["logit"]["C"]), int(best["xgboost"]["n_trees"])
    selected = np.where(
        metrics["model"] == "logit",
        metrics["C"] == best_C,
        metrics["n_trees"].fillna(0) == best_n_trees,
    )
    metrics.insert(metrics.columns.get_loc("split") + 1, "selected", selected)
    return PhaseComparison(
        metrics=metrics,
        logit=logits[best_C],
        # Same trees as the first best_n_trees of `longest`, saved without the unused rest.
        xgboost=make_xgboost(best_n_trees, min_count, **xgboost_params).fit(X_train, y_train),
        best_C=best_C,
        best_n_trees=best_n_trees,
    )


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """The selected models' validation and test scores, one row per phase, model, and split."""
    selected = metrics[metrics["selected"] & metrics["split"].isin(["valid", "test"])].copy()
    selected["setting"] = [
        f"C={row.C:g}" if row.model == "logit" else f"n_trees={row.n_trees}"
        for row in selected.itertuples()
    ]
    columns = ["phase", "model", "setting", "split", "n_trials", "positive_rate"]
    return selected[columns + SUMMARY_METRICS].reset_index(drop=True)


def run(
    dataset_root: Path,
    labels_dir: Path = LABELS_DIR,
    output_dir: Path | None = None,
    models_dir: Path = MODELS_DIR,
    phases: Sequence[str] = PHASES,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    chain_lengths: Sequence[int] = DEFAULT_CHAIN_LENGTHS,
    min_count: int = DEFAULT_MIN_COUNT,
    xgboost_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare the models on each phase, write every output, and return the summary."""
    dataset_root, labels_dir = Path(dataset_root), Path(labels_dir)
    output_dir = Path(output_dir or REPORTS_DIR / "baseline" / dataset_root.name)
    c_grid, chain_lengths = sorted(set(c_grid)), sorted(set(chain_lengths))
    if not c_grid or c_grid[0] <= 0:
        raise ValueError("c_grid must hold positive values")
    if not chain_lengths or chain_lengths[0] < 1:
        raise ValueError("chain_lengths must hold positive integers")
    xgboost_params = XGBOOST_DEFAULTS | (xgboost_params or {})
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for phase in phases:
        splits = load_phase(dataset_root, labels_dir, phase)
        result = compare_phase(splits, c_grid, chain_lengths, min_count, xgboost_params)
        metrics = result.metrics.assign(phase=phase)
        all_metrics.append(metrics[["phase", *result.metrics.columns]])

        logit_coefficients(result.logit).to_csv(
            output_dir / f"logit_coefficients_phase_{phase}.csv", index=False
        )
        save_model(result.logit, f"baseline_logit_phase_{phase}", models_dir)
        save_model(result.xgboost, f"baseline_xgboost_phase_{phase}", models_dir)

        fig = plot_chain_length(result.metrics)
        fig.suptitle(f"Phase {phase}: XGBoost by boosting chain length vs. regularized logit")
        save_figure(fig, f"chain_length_phase_{phase}", figures_dir)
        plt.close(fig)

        X_test, y_test = splits["test"]
        fig = plot_model_comparison(
            y_test,
            {
                f"Logit (C={result.best_C:g})": predict_success(result.logit, X_test),
                f"XGBoost ({result.best_n_trees} trees)": predict_success(result.xgboost, X_test),
            },
        )
        fig.suptitle(f"Phase {phase} test set: {len(y_test)} trials, {y_test.mean():.0%} succeeded")
        save_figure(fig, f"test_diagnostics_phase_{phase}", figures_dir)
        plt.close(fig)

    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(metrics)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    _write_run_record(
        output_dir / "run.json",
        dataset_root=dataset_root,
        labels_dir=labels_dir,
        phases=phases,
        config={
            "selection_metric": SELECTION_METRIC,
            "c_grid": c_grid,
            "chain_lengths": chain_lengths,
            "min_count": min_count,
            "xgboost_params": xgboost_params,
        },
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare a regularized logit with XGBoost chain lengths on the TOP datasets."
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True, help="directory holding phase_<P>/"
    )
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    parser.add_argument(
        "--output-dir", type=Path, help="default: reports/baseline/<dataset root name>"
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument(
        "--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID), help="logit C values"
    )
    parser.add_argument(
        "--chain-lengths",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHAIN_LENGTHS),
        help="XGBoost numbers of trees",
    )
    parser.add_argument("--learning-rate", type=float, default=XGBOOST_DEFAULTS["learning_rate"])
    parser.add_argument("--max-depth", type=int, default=XGBOOST_DEFAULTS["max_depth"])
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="fewest training trials a category or condition needs to get its own column",
    )
    args = parser.parse_args(argv)

    summary = run(
        args.dataset_root,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        phases=args.phases,
        c_grid=args.c_grid,
        chain_lengths=args.chain_lengths,
        min_count=args.min_count,
        xgboost_params={"learning_rate": args.learning_rate, "max_depth": args.max_depth},
    )
    print(summary.to_string(index=False, float_format="%.3f"))


def _score(
    splits: Splits, model: Pipeline, name: str, C: float = np.nan, n_trees: int | None = None
) -> list[dict[str, Any]]:
    rows = []
    for split, (X, y) in splits.items():
        scores = evaluate_probabilities(y, predict_success(model, X, n_trees))
        settings = {"model": name, "C": C, "n_trees": n_trees}
        rows.append(settings | {"split": split, "n_trials": len(y)} | scores)
    return rows


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
