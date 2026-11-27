import json

import numpy as np
import pandas as pd
import pytest

from trial_outcome.compare import MODELS, bootstrap_test, main
from trial_outcome.modeling import load_model


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


def test_bootstrap_intervals_contain_estimates_and_pair_the_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=300)
    good = np.clip(0.5 * y + rng.uniform(0, 0.5, size=300), 0, 1)
    noise = rng.uniform(size=300)

    intervals, differences = bootstrap_test(
        y, {"logit": noise, "xgboost": good, "random_forest": noise}, n_resamples=300
    )

    assert len(intervals) == 3 * 3
    assert (intervals["ci_low"] <= intervals["estimate"]).all()
    assert (intervals["estimate"] <= intervals["ci_high"]).all()
    auc = differences.set_index(["model", "metric"])
    # A clearly better model is better in every resample...
    assert auc.loc[("xgboost", "roc_auc"), "ci_low"] > 0
    assert auc.loc[("xgboost", "roc_auc"), "share_above_zero"] == 1.0
    # ...and identical predictions differ by exactly zero in every resample, because they're paired.
    same = auc.loc[("random_forest", "roc_auc")]
    assert same["ci_low"] == same["ci_high"] == same["difference"] == 0


def test_bootstrap_requires_reference_model():
    with pytest.raises(ValueError, match="reference"):
        bootstrap_test([0, 1], {"xgboost": np.array([0.2, 0.8])})


@pytest.mark.parametrize("tuning, tuning_trials", [("cv", 250), ("valid", 50)])
def test_main_writes_every_output(tmp_path, top_phase, capsys, tuning, tuning_trials):
    dataset_root, labels_dir = top_phase
    output_dir, models_dir = tmp_path / "report", tmp_path / "models"

    main(
        [
            *("--dataset-root", str(dataset_root), "--labels-dir", str(labels_dir)),
            *("--output-dir", str(output_dir), "--models-dir", str(models_dir)),
            *("--phases", "I", "--min-count", "5", "--bootstrap", "50"),
            *("--c-grid", "0.01", "1", "--chain-lengths", "5", "20"),
            *("--xgboost-candidates", "2", "--forest-candidates", "2"),
            *("--tuning", tuning, "--folds", "3"),
        ]
    )

    summary = pd.read_csv(output_dir / "summary.csv")
    assert summary[["model", "split"]].values.tolist() == [
        [model, split] for model in MODELS for split in ("tuning", "test")
    ]
    assert "random_forest" in capsys.readouterr().out
    tuned = summary[summary["split"] == "tuning"]
    assert (tuned["n_trials"] == tuning_trials).all()
    assert len(pd.read_csv(output_dir / "test_intervals.csv")) == 3 * 3
    differences = pd.read_csv(output_dir / "test_differences.csv")
    assert set(differences["model"]) == {"xgboost", "random_forest"}
    for model in MODELS:
        candidates = pd.read_csv(output_dir / f"tuning_{model}_phase_I.csv")
        assert candidates["selected"].sum() == 1
        assert load_model(f"{model}_phase_I", models_dir=models_dir) is not None
    assert not pd.read_csv(output_dir / "logit_coefficients_phase_I.csv").empty
    for model in ("xgboost", "random_forest"):
        assert not pd.read_csv(output_dir / f"importances_{model}_phase_I.csv").empty
    for figure in ("test_diagnostics_phase_I.png", "test_roc_auc.png"):
        assert (output_dir / "figures" / figure).stat().st_size > 0
    record = json.loads((output_dir / "run.json").read_text())
    assert record["config"]["xgboost_candidates"] == 2
    assert record["config"]["tuning"] == tuning
    assert len(record["inputs"]["I"]) == 2


@pytest.mark.parametrize("bad", [["--bootstrap", "0"], ["--folds", "1"], ["--tuning", "x"]])
def test_main_rejects_bad_arguments(top_phase, bad):
    dataset_root, _ = top_phase
    with pytest.raises(SystemExit):
        main(["--dataset-root", str(dataset_root), *bad])
