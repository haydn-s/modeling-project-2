"""Model definitions, prediction, and persistence.

Each model is a scikit-learn pipeline whose first step is
:func:`~trial_outcome.preprocessing.make_preprocessor`, so the models see the same features and
differ only in the learner. Fit a pipeline on the training rows only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from trial_outcome.config import MODELS_DIR, RANDOM_SEED
from trial_outcome.preprocessing import DEFAULT_MIN_COUNT, make_preprocessor

# Inverse regularization strengths for the logit, strongest regularization first.
DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
# Boosting chain lengths (number of trees) to compare, shortest first.
DEFAULT_CHAIN_LENGTHS = (10, 25, 50, 100, 200, 400, 800, 1600)

XGBOOST_DEFAULTS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

RANDOM_FOREST_DEFAULTS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_depth": None,
}


def make_logit(C: float = 1.0, min_count: int = DEFAULT_MIN_COUNT) -> Pipeline:
    """L2-regularized logistic regression; a smaller ``C`` regularizes more strongly.

    Classes are not reweighted, so the predicted probabilities keep the training success rate.
    """
    return Pipeline(
        [
            ("preprocess", make_preprocessor(min_count)),
            ("model", LogisticRegression(C=C, max_iter=10_000)),
        ]
    )


def make_xgboost(
    n_estimators: int = max(DEFAULT_CHAIN_LENGTHS),
    min_count: int = DEFAULT_MIN_COUNT,
    random_state: int = RANDOM_SEED,
    **params: Any,
) -> Pipeline:
    """Gradient-boosted trees with ``n_estimators`` boosting rounds (the chain length).

    ``params`` override :data:`XGBOOST_DEFAULTS` or set other ``XGBClassifier`` options. To
    compare chain lengths, fit the longest chain once and score its first ``n`` trees with
    :func:`predict_success`.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        **(XGBOOST_DEFAULTS | params),
    )
    return Pipeline([("preprocess", make_preprocessor(min_count)), ("model", model)])


def make_random_forest(
    min_count: int = DEFAULT_MIN_COUNT,
    random_state: int = RANDOM_SEED,
    n_jobs: int = -1,
    **params: Any,
) -> Pipeline:
    """Random forest of fully grown or leaf-size-limited trees, averaged into a probability.

    ``params`` override :data:`RANDOM_FOREST_DEFAULTS` or set other ``RandomForestClassifier``
    options. Larger ``min_samples_leaf`` gives smoother, less extreme probabilities, which
    matters when models are selected and scored by log loss. Classes are not reweighted, so the
    probabilities stay on the training success rate, as for the logit.
    """
    model = RandomForestClassifier(
        random_state=random_state, n_jobs=n_jobs, **(RANDOM_FOREST_DEFAULTS | params)
    )
    return Pipeline([("preprocess", make_preprocessor(min_count)), ("model", model)])


def feature_importances(model: Pipeline) -> pd.DataFrame:
    """Impurity (random forest) or gain (XGBoost) importances of a fitted tree pipeline.

    Importances are normalized to sum to 1 and sorted largest first. They show what the trees
    split on, not the direction of an effect; use the logit's coefficients for direction.
    """
    names = model[:-1].get_feature_names_out()
    importances = pd.DataFrame(
        {"feature": names, "importance": model[-1].feature_importances_.astype(float)}
    )
    total = importances["importance"].sum()
    if total > 0:
        importances["importance"] /= total
    order = importances["importance"].sort_values(ascending=False, kind="stable").index
    return importances.loc[order].reset_index(drop=True)


def predict_success(model: Pipeline, X: pd.DataFrame, n_trees: int | None = None) -> np.ndarray:
    """Return each trial's predicted probability of success.

    For an XGBoost pipeline, ``n_trees`` scores only the first ``n_trees`` boosting rounds.
    """
    if n_trees is None:
        return model.predict_proba(X)[:, 1]
    n_fitted = model[-1].get_booster().num_boosted_rounds()
    if not 1 <= n_trees <= n_fitted:
        raise ValueError(f"n_trees must be between 1 and {n_fitted}, got {n_trees}")
    return model.predict_proba(X, iteration_range=(0, n_trees))[:, 1]


def logit_coefficients(model: Pipeline) -> pd.DataFrame:
    """Coefficients of a fitted logit pipeline, largest effect first.

    Each coefficient is the change in log-odds of success per unit of its feature: one standard
    deviation for numeric features, presence for 0/1 features.
    """
    names = model[:-1].get_feature_names_out()
    coefficients = pd.DataFrame({"feature": names, "coefficient": model[-1].coef_[0]})
    order = coefficients["coefficient"].abs().sort_values(ascending=False, kind="stable").index
    return coefficients.loc[order].reset_index(drop=True)


def save_model(model: Any, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Persist ``model`` to ``models/<name>.joblib`` and return the path."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Any:
    """Load a model saved with :func:`save_model`."""
    return joblib.load(models_dir / f"{name}.joblib")
