"""Trial outcome labels.

The project brief allows either approach:

1. Use an existing labeled set of trials, stored under ``data/external/``.
2. Label a set of trials ourselves from ClinicalTrials.gov records.

Whichever we choose, :func:`build_labels` must return a DataFrame that passes
:func:`validate_labels`: one row per trial, with columns ``nct_id`` and ``label``
(1 = success, 0 = failure). Everything downstream depends only on that contract, so the
labeling source can change without touching features or models.
"""

from __future__ import annotations

import pandas as pd

ID_COLUMN = "nct_id"
LABEL_COLUMN = "label"


def build_labels(*args, **kwargs) -> pd.DataFrame:
    """Return a ``nct_id`` / ``label`` DataFrame for the trials we model."""
    raise NotImplementedError("Labeling approach not chosen yet; see the module docstring.")


def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Check the labels contract and return ``labels`` unchanged if it holds."""
    missing = {ID_COLUMN, LABEL_COLUMN} - set(labels.columns)
    if missing:
        raise ValueError(f"labels is missing columns: {sorted(missing)}")
    if labels[ID_COLUMN].isna().any():
        raise ValueError(f"labels has missing {ID_COLUMN} values")
    duplicated = labels[ID_COLUMN].duplicated()
    if duplicated.any():
        examples = labels.loc[duplicated, ID_COLUMN].head().tolist()
        raise ValueError(f"labels has duplicate {ID_COLUMN} values, e.g. {examples}")
    if not labels[LABEL_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{LABEL_COLUMN} must contain only 0 and 1")
    return labels
