import numpy as np
import pandas as pd
import pytest

from trial_outcome.features import FEATURE_DTYPES
from trial_outcome.preprocessing import (
    CATEGORICAL_FEATURES,
    MULTI_VALUED_FEATURES,
    NUMERIC_FEATURES,
    MultiValueBinarizer,
    make_preprocessor,
)


def test_feature_groups_cover_the_feature_table_once():
    groups = CATEGORICAL_FEATURES + MULTI_VALUED_FEATURES + NUMERIC_FEATURES

    assert sorted(groups) == sorted(FEATURE_DTYPES)


def test_binarizer_vocabulary_comes_from_fit_rows_only():
    train = pd.DataFrame({"conditions": ["Asthma|COPD", "asthma", "COPD"]})
    later = pd.DataFrame({"conditions": [" ASTHMA | Lupus", "", pd.NA]})

    binarizer = MultiValueBinarizer(min_count=1).fit(train)

    assert binarizer.get_feature_names_out().tolist() == ["conditions=asthma", "conditions=copd"]
    assert binarizer.transform(later).tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]


def test_binarizer_drops_values_below_min_count():
    train = pd.DataFrame({"phases": ["PHASE1", "PHASE1|PHASE2", "PHASE2", "PHASE3"]})

    binarizer = MultiValueBinarizer(min_count=2).fit(train)

    assert binarizer.get_feature_names_out().tolist() == ["phases=phase1", "phases=phase2"]


def test_binarizer_rejects_different_columns():
    binarizer = MultiValueBinarizer().fit(pd.DataFrame({"phases": ["PHASE1"]}))

    with pytest.raises(ValueError, match="columns"):
        binarizer.transform(pd.DataFrame({"conditions": ["Asthma"]}))


def test_preprocessor_encodes_every_feature_type(labeled_features):
    X = labeled_features.drop(columns="label")

    preprocessor = make_preprocessor(min_count=5).fit(X.iloc[:200])
    matrix = preprocessor.transform(X.iloc[200:])
    names = preprocessor.get_feature_names_out().tolist()

    assert matrix.shape == (100, len(names))
    assert np.isfinite(matrix).all()
    assert len(set(names)) == len(names)
    for name in [
        "allocation_NA",  # the registry's "not applicable", distinct from a missing value
        "allocation_missing",
        "primary_purpose_SCREENING",
        "healthy_volunteers_missing",
        "conditions=asthma",
        "conditions=breast cancer",
        "intervention_types=biological",
        "minimum_age_years",
        "missingindicator_maximum_age_years",
    ]:
        assert name in names


def test_preprocessor_groups_rare_and_unseen_categories(labeled_features):
    X = labeled_features.drop(columns="label")
    unseen = X.iloc[:1].copy()
    unseen["sponsor_class"] = "NEW_CLASS"
    unseen["conditions"] = "A Condition Never Seen"

    preprocessor = make_preprocessor(min_count=150).fit(X)
    names = preprocessor.get_feature_names_out().tolist()
    row = dict(zip(names, preprocessor.transform(unseen)[0], strict=True))

    assert "primary_purpose_infrequent_sklearn" in names  # SCREENING is in ~1/3 of rows
    assert row["sponsor_class_infrequent_sklearn"] == 1.0
    assert not [name for name in names if name.startswith("conditions=")]
