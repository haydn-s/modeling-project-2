"""Load explicit trial outcome labels from ``data/external/``.

The source may be a published labeled dataset or labels the team curates. Record the
source and outcome definition when building a dataset; registry completion alone does
not establish clinical success. :func:`build_labels` returns one row per trial with
columns ``nct_id`` and ``label`` (1 = success, 0 = failure). Downstream code depends only
on this contract, so the labeling source can change without changing features or models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trial_outcome.data.validation import validate_nct_ids

ID_COLUMN = "nct_id"
LABEL_COLUMN = "label"


def build_labels(path: str | Path) -> pd.DataFrame:
    """Load an explicitly labeled CSV or Parquet file; never infer success from status.

    Extra source columns are ignored. IDs must be canonical NCT identifiers and labels
    must be binary (1 = success, 0 = failure), with one nonmissing label per trial.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        labels = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    elif path.suffix.lower() == ".parquet":
        labels = pd.read_parquet(path)
    else:
        raise ValueError("labels must be a .csv or .parquet file")
    validate_labels(labels)
    return (
        labels[[ID_COLUMN, LABEL_COLUMN]]
        .astype({ID_COLUMN: "string", LABEL_COLUMN: "int64"})
        .sort_values(ID_COLUMN)
        .reset_index(drop=True)
    )


def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Check the labels contract and return ``labels`` unchanged if it holds."""
    missing = {ID_COLUMN, LABEL_COLUMN} - set(labels.columns)
    if missing:
        raise ValueError(f"labels is missing columns: {sorted(missing)}")
    validate_nct_ids(labels[ID_COLUMN], source="labels")
    if not labels[LABEL_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{LABEL_COLUMN} must contain only 0 and 1")
    return labels
