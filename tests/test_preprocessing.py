import numpy as np
import pandas as pd
import pytest

from trial_outcome.data.top import TOP_FEATURE_DTYPES
from trial_outcome.features import FEATURE_DTYPES
from trial_outcome.preprocessing import (
    CATEGORICAL_FEATURES,
    MULTI_VALUED_FEATURES,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    TOP_MULTI_VALUED_FEATURES,
    MultiValueBinarizer,
    TextComponents,
    make_preprocessor,
)


def test_feature_groups_cover_the_feature_table_once():
    groups = CATEGORICAL_FEATURES + MULTI_VALUED_FEATURES + NUMERIC_FEATURES

    assert sorted(groups) == sorted(FEATURE_DTYPES)
    assert sorted(TOP_MULTI_VALUED_FEATURES + TEXT_FEATURES) == sorted(TOP_FEATURE_DTYPES)


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


def test_preprocessor_adds_top_columns_only_when_present(labeled_features, top_labeled_features):
    registry = make_preprocessor(min_count=5).fit(labeled_features.drop(columns="label"))
    with_top = make_preprocessor(min_count=5, text_components=4).fit(
        top_labeled_features.drop(columns="label").iloc[:200]
    )
    registry_names = registry.get_feature_names_out().tolist()
    names = with_top.get_feature_names_out().tolist()

    assert names[: len(registry_names)] == registry_names
    assert names[len(registry_names) :] == [
        "icd_chapters=blood_immune",
        "icd_chapters=neoplasms",
        "icd_chapters=respiratory",
        "icd_categories=c50",
        "icd_categories=d63",
        "icd_categories=j45",
        *(f"criteria_text_{i}" for i in range(4)),
    ]
    matrix = with_top.transform(top_labeled_features.drop(columns="label").iloc[200:])
    assert matrix.shape == (100, len(names))
    assert np.isfinite(matrix).all()


def test_preprocessor_fitted_with_top_columns_requires_them(top_labeled_features):
    X = top_labeled_features.drop(columns="label")
    preprocessor = make_preprocessor(min_count=5, text_components=4).fit(X)

    with pytest.raises(ValueError, match="missing"):
        preprocessor.transform(X.drop(columns="criteria"))


def test_text_components_vocabulary_comes_from_fit_rows_only():
    train = pd.DataFrame({"criteria": ["adults with asthma", "adults who smoke", pd.NA] * 4})
    later = pd.DataFrame({"criteria": ["children with lupus", pd.NA]})

    text = TextComponents(n_components=2, min_df=2).fit(train)

    # Stop words ("with", "who") are dropped; two-word phrases are kept.
    assert sorted(text.vectorizer_.vocabulary_) == [
        "adults",
        "adults asthma",
        "adults smoke",
        "asthma",
        "smoke",
    ]
    assert text.get_feature_names_out().tolist() == ["criteria_text_0", "criteria_text_1"]
    # Unseen words carry no information: the same row as missing text.
    np.testing.assert_allclose(*text.transform(later))


def test_text_components_are_capped_by_the_vocabulary():
    # "adults" is in every row, above the 90% limit, so 4 terms remain: 3 components at most.
    train = pd.DataFrame({"criteria": ["adults with asthma", "adults who smoke"] * 2})

    text = TextComponents(n_components=50, min_df=2).fit(train)

    assert len(text.vectorizer_.vocabulary_) == 4
    assert text.transform(train).shape == (4, 3)
    with pytest.raises(ValueError, match="too small"):
        TextComponents(min_df=2).fit(pd.DataFrame({"criteria": ["asthma"] * 4}))


def test_text_components_reject_other_columns():
    text = TextComponents(n_components=1, min_df=1).fit(
        pd.DataFrame({"criteria": ["adults asthma", "asthma smokers"]})
    )

    with pytest.raises(ValueError, match="columns"):
        text.transform(pd.DataFrame({"notes": ["a b"]}))
