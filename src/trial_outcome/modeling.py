"""Model training and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from trial_outcome.config import MODELS_DIR


def train_model(features: pd.DataFrame, labels: pd.Series) -> Any:
    """Fit a model that predicts the probability a trial succeeds."""
    raise NotImplementedError("Model not chosen yet.")


def save_model(model: Any, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Persist ``model`` to ``models/<name>.joblib`` and return the path."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str, models_dir: Path = MODELS_DIR) -> Any:
    """Load a model saved with :func:`save_model`."""
    return joblib.load(models_dir / f"{name}.joblib")
