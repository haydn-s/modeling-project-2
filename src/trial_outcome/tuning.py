"""Hyperparameter tuning for the three compared models, without touching the test split.

Candidates are scored on one or more *folds*, each a (fit, score) pair of row sets, and the
candidate with the lowest log loss on the pooled scored rows is selected. Two ways to build
folds (:func:`make_folds`):

- ``"valid"``: one fold: fit on TOP's train split, score on its valid split. This is the rule
  :mod:`trial_outcome.baseline` uses. The selected model is refitted on train.
- ``"cv"``: stratified k-fold cross-validation over train and valid together. TOP's train and
  valid splits are a random 90/10 split of the same trials, so pooling them is sound, and
  every learning trial helps choose the settings. The valid split alone is small (117 trials
  in phase I), and a search over many candidates can fit its noise. The selected model is
  refitted on train and valid.

Models:

- **Logit:** the inverse regularization strength ``C`` from a fixed grid.
- **XGBoost:** a random search over tree depth, learning rate, row and column subsampling,
  and regularization. Each candidate is fitted once per fold at the longest chain length,
  and its chain length (number of trees) is chosen by scoring its first ``n`` trees, as in the
  baseline. The baseline's default settings are always the first candidate.
- **Random forest:** a random search over features per split, leaf size, and depth. The
  default settings are always the first candidate.

Each fold's preprocessor is fitted on that fold's fitting rows only and shared by the
candidates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline

from trial_outcome.config import RANDOM_SEED
from trial_outcome.evaluation import evaluate_probabilities
from trial_outcome.modeling import (
    DEFAULT_C_GRID,
    DEFAULT_CHAIN_LENGTHS,
    RANDOM_FOREST_DEFAULTS,
    XGBOOST_DEFAULTS,
    make_logit,
    make_random_forest,
    make_xgboost,
)
from trial_outcome.preprocessing import DEFAULT_MIN_COUNT, make_preprocessor

SELECTION_METRIC = "log_loss"
TUNING_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss"]
TUNING_MODES = ("cv", "valid")
DEFAULT_TUNING_MODE = "cv"
DEFAULT_FOLDS = 5

XGBOOST_SEARCH_SPACE = {
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "min_child_weight": [1.0, 2.0, 5.0, 10.0, 20.0],
    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0],
    "reg_alpha": [0.0, 0.1, 1.0, 5.0],
    "gamma": [0.0, 0.5, 1.0, 2.0],
}
RANDOM_FOREST_SEARCH_SPACE = {
    "max_features": ["sqrt", 0.1, 0.25, 0.5],
    "min_samples_leaf": [1, 3, 5, 10, 20, 40],
    "max_depth": [None, 6, 10, 16],
}
DEFAULT_XGBOOST_CANDIDATES = 30
DEFAULT_FOREST_CANDIDATES = 20

Split = tuple[pd.DataFrame, pd.Series]
Fold = tuple[Split, Split]


@dataclass(frozen=True)
class TuningResult:
    """The selected model, refitted on the learning rows, and every candidate's scores.

    ``candidates`` has one row per candidate (per candidate and chain length for XGBoost),
    with its settings, its scores on the pooled scored rows of the folds (columns prefixed
    ``tuning_``), and a ``selected`` flag on exactly one row.
    """

    model: Pipeline
    params: dict[str, Any]
    candidates: pd.DataFrame

    @property
    def setting(self) -> str:
        """The selected settings as one short readable string, e.g. ``C=0.1``."""
        return ", ".join(f"{key}={_format(value)}" for key, value in self.params.items())


def make_folds(
    train: Split,
    valid: Split,
    mode: str = DEFAULT_TUNING_MODE,
    n_folds: int = DEFAULT_FOLDS,
    random_state: int = RANDOM_SEED,
) -> tuple[list[Fold], Split]:
    """Return the tuning folds and the rows to refit the selected model on.

    ``"valid"`` gives the single train -> valid fold and refits on train. ``"cv"`` gives
    ``n_folds`` stratified folds over train and valid together and refits on both.
    """
    if mode == "valid":
        return [(train, valid)], train
    if mode != "cv":
        raise ValueError(f"mode must be one of {TUNING_MODES}, got {mode!r}")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    X = pd.concat([train[0], valid[0]])
    y = pd.concat([train[1], valid[1]])
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = [
        ((X.iloc[fit], y.iloc[fit]), (X.iloc[score], y.iloc[score]))
        for fit, score in splitter.split(X, y)
    ]
    return folds, (X, y)


def tune_logit(
    folds: Sequence[Fold],
    refit: Split,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    min_count: int = DEFAULT_MIN_COUNT,
) -> TuningResult:
    """Select the logit's ``C`` by pooled log loss (ties go to stronger regularization)."""
    c_grid = sorted(set(c_grid))
    if not c_grid or c_grid[0] <= 0:
        raise ValueError("c_grid must hold positive values")
    rows = []
    for C in c_grid:
        y_true, y_prob = _pooled(
            folds, lambda fit, X, C=C: make_logit(C, min_count).fit(*fit).predict_proba(X)[:, 1]
        )
        rows.append({"C": C, **_scores(y_true, y_prob)})
    candidates = _mark_selected(pd.DataFrame(rows))
    best_C = float(candidates.loc[candidates["selected"], "C"].item())
    return TuningResult(make_logit(best_C, min_count).fit(*refit), {"C": best_C}, candidates)


def tune_xgboost(
    folds: Sequence[Fold],
    refit: Split,
    n_candidates: int = DEFAULT_XGBOOST_CANDIDATES,
    chain_lengths: Sequence[int] = DEFAULT_CHAIN_LENGTHS,
    min_count: int = DEFAULT_MIN_COUNT,
    random_state: int = RANDOM_SEED,
    search_space: Mapping[str, Sequence[Any]] = XGBOOST_SEARCH_SPACE,
) -> TuningResult:
    """Random search over XGBoost settings and chain length, selected on pooled log loss.

    ``n_candidates`` counts the baseline defaults, which are always tried first. Ties go to the
    earlier candidate and the shorter chain.
    """
    chain_lengths = sorted(set(chain_lengths))
    if not chain_lengths or chain_lengths[0] < 1:
        raise ValueError("chain_lengths must hold positive integers")
    settings = _candidates(XGBOOST_DEFAULTS, search_space, n_candidates, random_state)
    encoded = _preprocessed(folds, min_count)
    y_true = np.concatenate([np.asarray(y) for _, (_, y) in encoded])

    rows = []
    for i, params in enumerate(settings):
        by_length = {n: [] for n in chain_lengths}
        for (X_fit, y_fit), (X_score, _) in encoded:
            booster = make_xgboost(max(chain_lengths), min_count, random_state, **params)[-1]
            booster.fit(X_fit, y_fit)
            for n in chain_lengths:
                by_length[n].append(booster.predict_proba(X_score, iteration_range=(0, n))[:, 1])
        for n, parts in by_length.items():
            scores = _scores(y_true, np.concatenate(parts))
            rows.append({"candidate": i, **params, "n_trees": n, **scores})
    candidates = _mark_selected(pd.DataFrame(rows))

    best = candidates.loc[candidates["selected"]].iloc[0]
    params = dict(settings[int(best["candidate"])])
    n_trees = int(best["n_trees"])
    model = make_xgboost(n_trees, min_count, random_state, **params).fit(*refit)
    return TuningResult(model, {"n_trees": n_trees, **params}, candidates)


def tune_random_forest(
    folds: Sequence[Fold],
    refit: Split,
    n_candidates: int = DEFAULT_FOREST_CANDIDATES,
    min_count: int = DEFAULT_MIN_COUNT,
    random_state: int = RANDOM_SEED,
    search_space: Mapping[str, Sequence[Any]] = RANDOM_FOREST_SEARCH_SPACE,
    n_estimators: int = RANDOM_FOREST_DEFAULTS["n_estimators"],
) -> TuningResult:
    """Random search over random forest settings, selected on pooled log loss.

    ``n_candidates`` counts the default settings, which are always tried first. The number of
    trees is fixed: more trees only reduce noise, so it is not worth tuning.
    """
    defaults = {k: v for k, v in RANDOM_FOREST_DEFAULTS.items() if k != "n_estimators"}
    settings = _candidates(defaults, search_space, n_candidates, random_state)
    encoded = _preprocessed(folds, min_count)
    y_true = np.concatenate([np.asarray(y) for _, (_, y) in encoded])

    rows = []
    for i, params in enumerate(settings):
        parts = []
        for (X_fit, y_fit), (X_score, _) in encoded:
            forest = make_random_forest(
                min_count, random_state, n_estimators=n_estimators, **params
            )[-1]
            parts.append(forest.fit(X_fit, y_fit).predict_proba(X_score)[:, 1])
        rows.append({"candidate": i, **params, **_scores(y_true, np.concatenate(parts))})
    candidates = _mark_selected(pd.DataFrame(rows))

    best = int(candidates.loc[candidates["selected"], "candidate"].item())
    params = {"n_estimators": n_estimators, **settings[best]}
    model = make_random_forest(min_count, random_state, **params).fit(*refit)
    return TuningResult(model, params, candidates)


def _candidates(
    defaults: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    n_candidates: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """The defaults first, then distinct random draws from the search space."""
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    space = {key: list(values) for key, values in search_space.items()}
    first = {key: defaults[key] for key in space if key in defaults}
    settings = [first]
    size = math.prod(len(values) for values in space.values())
    sampler = ParameterSampler(space, n_iter=min(4 * n_candidates, size), random_state=random_state)
    for params in sampler:
        if len(settings) >= n_candidates:
            break
        params = {key: params[key] for key in space}
        if params not in settings:
            settings.append(params)
    return settings


def _preprocessed(folds: Sequence[Fold], min_count: int) -> list[tuple[Any, Any]]:
    """Encode each fold with a preprocessor fitted on that fold's fitting rows only."""
    encoded = []
    for (X_fit, y_fit), (X_score, y_score) in folds:
        preprocessor = make_preprocessor(min_count).fit(X_fit, y_fit)
        encoded.append(
            ((preprocessor.transform(X_fit), y_fit), (preprocessor.transform(X_score), y_score))
        )
    return encoded


def _pooled(
    folds: Sequence[Fold], predict: Callable[[Split, pd.DataFrame], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Each fold's scored-row predictions from a model fitted on its fitting rows, pooled."""
    y_true = np.concatenate([np.asarray(score[1]) for _, score in folds])
    y_prob = np.concatenate([predict(fit, score[0]) for fit, score in folds])
    return y_true, y_prob


def _scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    scores = evaluate_probabilities(y_true, y_prob)
    return {f"tuning_{metric}": scores[metric] for metric in TUNING_METRICS}


def _mark_selected(candidates: pd.DataFrame) -> pd.DataFrame:
    """Flag the first row with the lowest pooled log loss."""
    best = candidates[f"tuning_{SELECTION_METRIC}"].idxmin()
    candidates["selected"] = candidates.index == best
    return candidates


def _format(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)
