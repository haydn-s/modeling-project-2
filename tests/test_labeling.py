import pandas as pd
import pytest

from trial_outcome.labeling import build_labels, validate_labels


def test_valid_labels_pass_through():
    labels = pd.DataFrame({"nct_id": ["NCT00000001", "NCT00000002"], "label": [1, 0]})
    assert validate_labels(labels) is labels


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (pd.DataFrame({"nct_id": ["NCT00000001"]}), "missing columns"),
        (pd.DataFrame({"nct_id": ["NCT00000001", None], "label": [1, 0]}), "missing nct_id"),
        (pd.DataFrame({"nct_id": ["NCT00000001", "NCT00000001"], "label": [1, 0]}), "duplicate"),
        (pd.DataFrame({"nct_id": ["NCT00000001", "NCT00000002"], "label": [1, 2]}), "only 0 and 1"),
        (pd.DataFrame({"nct_id": ["NCT00000001"], "label": [None]}), "only 0 and 1"),
        (pd.DataFrame({"nct_id": [""], "label": [1]}), "invalid nct_id"),
    ],
)
def test_invalid_labels_raise(labels, message):
    with pytest.raises(ValueError, match=message):
        validate_labels(labels)


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_load_labels_sorts_and_drops_source_columns(tmp_path, extension):
    labels = pd.DataFrame(
        {"nct_id": ["NCT00000002", "NCT00000001"], "label": [0, 1], "notes": ["a", "b"]}
    )
    path = tmp_path / f"labels.{extension}"
    getattr(labels, f"to_{extension}")(path, index=False)
    actual = build_labels(path)
    assert actual.to_dict("list") == {"nct_id": ["NCT00000001", "NCT00000002"], "label": [1, 0]}
    assert actual["label"].dtype == "int64"


def test_labels_reject_unsupported_file_types(tmp_path):
    with pytest.raises(ValueError, match=".csv or .parquet"):
        build_labels(tmp_path / "labels.json")
