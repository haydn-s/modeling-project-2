import numpy as np
import pytest

from trial_outcome.evaluation import evaluate_probabilities
from trial_outcome.modeling import load_model, save_model
from trial_outcome.visualization import plot_model_diagnostics, save_figure


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(0.6 * y_true + rng.normal(0.2, 0.2, size=200), 0, 1)
    return y_true, y_prob


def test_perfect_predictions():
    metrics = evaluate_probabilities([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0])

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["positive_rate"] == 0.5


def test_threshold_only_affects_thresholded_metrics(predictions):
    y_true, y_prob = predictions
    low = evaluate_probabilities(y_true, y_prob, threshold=0.1)
    high = evaluate_probabilities(y_true, y_prob, threshold=0.9)

    assert low["roc_auc"] == high["roc_auc"]
    assert low["brier"] == high["brier"]
    assert low["recall"] > high["recall"]


def test_diagnostics_figure_saves(tmp_path, predictions):
    fig = plot_model_diagnostics(*predictions, name="toy")

    path = save_figure(fig, "diagnostics", figures_dir=tmp_path)

    assert len(fig.axes) == 3
    assert path.exists()


def test_model_round_trip(tmp_path):
    model = {"weights": [1, 2, 3]}
    save_model(model, "toy", models_dir=tmp_path)
    assert load_model("toy", models_dir=tmp_path) == model
