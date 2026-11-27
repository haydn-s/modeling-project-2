import numpy as np
import pytest

from trial_outcome.modeling import (
    RANDOM_FOREST_DEFAULTS,
    XGBOOST_DEFAULTS,
    feature_importances,
    make_random_forest,
    make_xgboost,
    predict_success,
)
from trial_outcome.tuning import (
    RANDOM_FOREST_SEARCH_SPACE,
    XGBOOST_SEARCH_SPACE,
    _candidates,
    make_folds,
    tune_logit,
    tune_random_forest,
    tune_xgboost,
)

CHAIN_LENGTHS = (5, 20)


@pytest.fixture
def valid_folds(splits):
    """The single train -> valid fold, refitting on train."""
    return make_folds(splits["train"], splits["valid"], mode="valid")


@pytest.fixture
def splits(labeled_features):
    """200 train, 50 valid, and 50 test trials from the synthetic labeled table."""
    X, y = labeled_features.drop(columns="label"), labeled_features["label"]
    return {
        "train": (X.iloc[:200], y.iloc[:200]),
        "valid": (X.iloc[200:250], y.iloc[200:250]),
        "test": (X.iloc[250:], y.iloc[250:]),
    }


def test_candidates_start_with_defaults_and_are_distinct():
    settings = _candidates(XGBOOST_DEFAULTS, XGBOOST_SEARCH_SPACE, 10, random_state=0)

    assert len(settings) == 10
    assert settings[0] == {
        k: XGBOOST_DEFAULTS[k] for k in XGBOOST_SEARCH_SPACE if k in XGBOOST_DEFAULTS
    }
    assert len({tuple(sorted(s.items(), key=str)) for s in settings}) == 10
    assert settings == _candidates(XGBOOST_DEFAULTS, XGBOOST_SEARCH_SPACE, 10, random_state=0)


def test_candidates_rejects_empty_search():
    with pytest.raises(ValueError, match="at least 1"):
        _candidates(XGBOOST_DEFAULTS, XGBOOST_SEARCH_SPACE, 0, random_state=0)


def test_tune_logit_selects_lowest_validation_log_loss(valid_folds):
    result = tune_logit(*valid_folds, c_grid=(0.01, 1.0), min_count=5)

    candidates = result.candidates
    assert candidates["selected"].sum() == 1
    best = candidates.loc[candidates["selected"]]
    assert best["tuning_log_loss"].item() == candidates["tuning_log_loss"].min()
    assert result.model[-1].C == result.params["C"] == best["C"].item()
    assert result.setting == f"C={result.params['C']:g}"


def test_tune_xgboost_selects_candidate_and_chain_length(valid_folds):
    result = tune_xgboost(*valid_folds, n_candidates=3, chain_lengths=CHAIN_LENGTHS, min_count=5)

    candidates = result.candidates
    assert len(candidates) == 3 * len(CHAIN_LENGTHS)
    assert candidates["selected"].sum() == 1
    best = candidates.loc[candidates["selected"]].iloc[0]
    assert best["tuning_log_loss"] == candidates["tuning_log_loss"].min()
    assert result.model[-1].n_estimators == result.params["n_trees"] == best["n_trees"]
    # The first candidate is the baseline's default settings.
    first = candidates[candidates["candidate"] == 0].iloc[0]
    assert first["learning_rate"] == XGBOOST_DEFAULTS["learning_rate"]
    assert first["max_depth"] == XGBOOST_DEFAULTS["max_depth"]


def test_tuned_xgboost_reproduces_its_validation_score(splits, valid_folds):
    result = tune_xgboost(*valid_folds, n_candidates=3, chain_lengths=CHAIN_LENGTHS, min_count=5)
    X_valid, y_valid = splits["valid"]
    params = {k: v for k, v in result.params.items() if k != "n_trees"}
    longest = make_xgboost(max(CHAIN_LENGTHS), 5, **params).fit(*splits["train"])

    # The refitted model gives the same predictions the search scored.
    np.testing.assert_allclose(
        predict_success(result.model, X_valid),
        predict_success(longest, X_valid, n_trees=result.params["n_trees"]),
        rtol=1e-6,
    )


def test_tune_random_forest_selects_and_refits(splits, valid_folds):
    result = tune_random_forest(*valid_folds, n_candidates=3, min_count=5, n_estimators=50)

    candidates = result.candidates
    assert len(candidates) == 3
    assert candidates["selected"].sum() == 1
    best = candidates.loc[candidates["selected"]]
    assert best["tuning_log_loss"].item() == candidates["tuning_log_loss"].min()
    forest = result.model[-1]
    assert forest.n_estimators == 50
    for key in RANDOM_FOREST_SEARCH_SPACE:
        assert getattr(forest, key) == result.params[key]

    X_valid, _ = splits["valid"]
    refit = make_random_forest(5, **result.params).fit(*splits["train"])
    np.testing.assert_allclose(
        predict_success(result.model, X_valid), predict_success(refit, X_valid)
    )


def test_random_forest_probabilities_are_valid(splits):
    model = make_random_forest(5, n_estimators=50).fit(*splits["train"])
    y_prob = predict_success(model, splits["test"][0])

    assert y_prob.shape == (50,)
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
    assert model[-1].min_samples_leaf == RANDOM_FOREST_DEFAULTS["min_samples_leaf"]


@pytest.mark.parametrize("builder", [make_random_forest, make_xgboost])
def test_feature_importances_sum_to_one_and_name_features(splits, builder):
    model = (builder(5, n_estimators=20) if builder is make_random_forest else builder(20, 5)).fit(
        *splits["train"]
    )

    importances = feature_importances(model)

    assert importances["importance"].sum() == pytest.approx(1.0)
    assert importances["importance"].is_monotonic_decreasing
    assert set(importances["feature"]) == set(model[:-1].get_feature_names_out())


def test_valid_mode_is_one_fold_refitting_on_train(splits):
    folds, refit = make_folds(splits["train"], splits["valid"], mode="valid")

    assert len(folds) == 1
    ((fit, score),) = folds
    assert fit[0].index.equals(splits["train"][0].index)
    assert score[0].index.equals(splits["valid"][0].index)
    assert refit[0].index.equals(splits["train"][0].index)


def test_cv_folds_partition_train_and_valid_and_never_touch_test(splits):
    folds, refit = make_folds(splits["train"], splits["valid"], mode="cv", n_folds=5)

    learning = splits["train"][0].index.append(splits["valid"][0].index)
    assert refit[0].index.equals(learning)
    scored = [score[0].index for _, score in folds]
    assert sorted(i for index in scored for i in index) == sorted(learning)
    test_ids = set(splits["test"][0].index)
    for fit, score in folds:
        assert not set(fit[0].index) & set(score[0].index)
        assert not (set(fit[0].index) | set(score[0].index)) & test_ids
        # Stratified: every fold keeps roughly the overall success rate.
        assert abs(score[1].mean() - refit[1].mean()) < 0.1


@pytest.mark.parametrize("mode, n_folds", [("holdout", 5), ("cv", 1)])
def test_make_folds_rejects_bad_settings(splits, mode, n_folds):
    with pytest.raises(ValueError):
        make_folds(splits["train"], splits["valid"], mode=mode, n_folds=n_folds)


def test_cv_tuning_scores_every_learning_trial_and_refits_on_all(splits):
    folds, refit = make_folds(splits["train"], splits["valid"], mode="cv", n_folds=3)

    result = tune_random_forest(folds, refit, n_candidates=2, min_count=5, n_estimators=30)

    # The refitted forest saw all 250 learning trials, not only the 200 train trials.
    assert result.model[-1].n_features_in_ == len(result.model[:-1].get_feature_names_out())
    assert result.model[:-1].n_features_in_ == refit[0].shape[1]
    np.testing.assert_allclose(
        predict_success(result.model, splits["test"][0]),
        predict_success(make_random_forest(5, **result.params).fit(*refit), splits["test"][0]),
    )
