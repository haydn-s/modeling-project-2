import numpy as np
import pytest

from trial_outcome.modeling import (
    load_model,
    logit_coefficients,
    make_logit,
    make_xgboost,
    predict_success,
    save_model,
)


@pytest.fixture
def training_data(labeled_features):
    return labeled_features.drop(columns="label"), labeled_features["label"]


@pytest.mark.parametrize("make_model", [make_logit, lambda: make_xgboost(n_estimators=20)])
def test_models_predict_probabilities(training_data, make_model):
    X, y = training_data

    probabilities = predict_success(make_model().fit(X, y), X)

    assert probabilities.shape == (len(X),)
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_first_trees_of_a_long_chain_match_a_short_chain(training_data):
    X, y = training_data
    long_chain = make_xgboost(n_estimators=30).fit(X, y)
    short_chain = make_xgboost(n_estimators=10).fit(X, y)

    np.testing.assert_allclose(
        predict_success(long_chain, X, n_trees=10), predict_success(short_chain, X)
    )
    assert not np.allclose(predict_success(long_chain, X), predict_success(short_chain, X))


@pytest.mark.parametrize("n_trees", [0, 11])
def test_predict_success_rejects_unfitted_chain_lengths(training_data, n_trees):
    X, y = training_data
    model = make_xgboost(n_estimators=10).fit(X, y)

    with pytest.raises(ValueError, match="between 1 and 10"):
        predict_success(model, X, n_trees=n_trees)


def test_stronger_regularization_shrinks_logit_coefficients(training_data):
    X, y = training_data

    strong = logit_coefficients(make_logit(C=0.01).fit(X, y))
    weak = logit_coefficients(make_logit(C=100.0).fit(X, y))

    assert strong["coefficient"].abs().sum() < weak["coefficient"].abs().sum()


def test_logit_coefficients_recover_the_true_effects(training_data):
    X, y = training_data

    coefficients = logit_coefficients(make_logit(C=1.0).fit(X, y))
    effects = coefficients.set_index("feature")["coefficient"]

    assert coefficients["coefficient"].abs().is_monotonic_decreasing
    assert effects["sponsor_class_INDUSTRY"] > 0
    assert effects["minimum_age_years"] < 0
    assert set(coefficients["feature"].head(2)) == {"sponsor_class_INDUSTRY", "minimum_age_years"}


def test_fitted_pipeline_round_trips(tmp_path, training_data):
    X, y = training_data
    model = make_logit().fit(X, y)

    save_model(model, "logit", models_dir=tmp_path)

    np.testing.assert_array_equal(
        predict_success(load_model("logit", models_dir=tmp_path), X), predict_success(model, X)
    )
