import pandas as pd
import pytest

from trial_outcome.labeling import validate_labels


def test_valid_labels_pass_through():
    labels = pd.DataFrame({"nct_id": ["NCT1", "NCT2"], "label": [1, 0]})
    assert validate_labels(labels) is labels


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (pd.DataFrame({"nct_id": ["NCT1"]}), "missing columns"),
        (pd.DataFrame({"nct_id": ["NCT1", None], "label": [1, 0]}), "missing nct_id"),
        (pd.DataFrame({"nct_id": ["NCT1", "NCT1"], "label": [1, 0]}), "duplicate"),
        (pd.DataFrame({"nct_id": ["NCT1", "NCT2"], "label": [1, 2]}), "only 0 and 1"),
    ],
)
def test_invalid_labels_raise(labels, message):
    with pytest.raises(ValueError, match=message):
        validate_labels(labels)
