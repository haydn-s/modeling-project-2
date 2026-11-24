import json

import numpy as np
import pandas as pd
import pytest

from trial_outcome.baseline import compare_phase, load_phase, main
from trial_outcome.modeling import load_model, make_xgboost, predict_success

C_GRID = (0.01, 1.0)
CHAIN_LENGTHS = (5, 20)


@pytest.fixture
def top_phase(tmp_path, labeled_features):
    """A phase I dataset and its TOP labels file: 200 train, 50 valid, and 50 test trials."""
    dataset_root, labels_dir = tmp_path / "top", tmp_path / "labels"
    (dataset_root / "phase_I").mkdir(parents=True)
    labels_dir.mkdir()
    labeled_features.to_parquet(dataset_root / "phase_I" / "dataset.parquet")
    split = ["train"] * 200 + ["valid"] * 50 + ["test"] * 50
    labels = pd.DataFrame(
        {"nct_id": labeled_features.index, "label": labeled_features["label"], "split": split}
    )
    labels.to_csv(labels_dir / "phase_I.csv", index=False)
    return dataset_root, labels_dir


def test_load_phase_splits_features_from_labels(top_phase, labeled_features):
    splits = load_phase(*top_phase, "I")

    assert {name: len(X) for name, (X, _) in splits.items()} == {
        "train": 200,
        "valid": 50,
        "test": 50,
    }
    X_test, y_test = splits["test"]
    assert "label" not in X_test.columns
    assert y_test.equals(labeled_features["label"].iloc[250:])


def test_load_phase_rejects_trials_without_a_split(top_phase):
    dataset_root, labels_dir = top_phase
    labels = pd.read_csv(labels_dir / "phase_I.csv")
    labels.iloc[:-1].to_csv(labels_dir / "phase_I.csv", index=False)

    with pytest.raises(ValueError, match="1 trials have no TOP split"):
        load_phase(dataset_root, labels_dir, "I")


def test_compare_phase_selects_by_validation_log_loss(top_phase):
    splits = load_phase(*top_phase, "I")

    result = compare_phase(splits, C_GRID, CHAIN_LENGTHS, min_count=5)
    metrics = result.metrics

    assert len(metrics) == (len(C_GRID) + len(CHAIN_LENGTHS)) * 3
    assert metrics["selected"].sum() == 2 * 3
    valid = metrics[metrics["split"] == "valid"]
    for model in ("logit", "xgboost"):
        rows = valid[valid["model"] == model]
        assert rows.loc[rows["selected"], "log_loss"].item() == rows["log_loss"].min()
    assert result.best_C == result.logit[-1].C
    assert result.best_n_trees == result.xgboost[-1].n_estimators


def test_selected_xgboost_scores_match_its_metrics_row(top_phase):
    splits = load_phase(*top_phase, "I")
    X_test, y_test = splits["test"]

    result = compare_phase(splits, C_GRID, CHAIN_LENGTHS, min_count=5)
    row = result.metrics.query("model == 'xgboost' and selected and split == 'test'")
    longest = make_xgboost(max(CHAIN_LENGTHS), min_count=5).fit(*splits["train"])

    np.testing.assert_allclose(
        predict_success(result.xgboost, X_test),
        predict_success(longest, X_test, n_trees=result.best_n_trees),
    )
    assert row["n_trials"].item() == len(y_test)


def test_main_writes_every_output(tmp_path, top_phase, capsys):
    dataset_root, labels_dir = top_phase
    output_dir, models_dir = tmp_path / "report", tmp_path / "models"

    main(
        [
            *("--dataset-root", str(dataset_root), "--labels-dir", str(labels_dir)),
            *("--output-dir", str(output_dir), "--models-dir", str(models_dir)),
            *("--phases", "I", "--min-count", "5"),
            *("--c-grid", *map(str, C_GRID), "--chain-lengths", *map(str, CHAIN_LENGTHS)),
        ]
    )

    summary = pd.read_csv(output_dir / "summary.csv")
    assert summary[["model", "split"]].values.tolist() == [
        ["logit", "valid"],
        ["logit", "test"],
        ["xgboost", "valid"],
        ["xgboost", "test"],
    ]
    assert "n_trees=" in capsys.readouterr().out
    assert len(pd.read_csv(output_dir / "metrics.csv")) == 12
    assert not pd.read_csv(output_dir / "logit_coefficients_phase_I.csv").empty
    for figure in ("chain_length_phase_I.png", "test_diagnostics_phase_I.png"):
        assert (output_dir / "figures" / figure).stat().st_size > 0
    record = json.loads((output_dir / "run.json").read_text())
    assert record["config"]["chain_lengths"] == list(CHAIN_LENGTHS)
    assert len(record["inputs"]["I"]) == 2
    for name in ("baseline_logit_phase_I", "baseline_xgboost_phase_I"):
        assert load_model(name, models_dir=models_dir) is not None
