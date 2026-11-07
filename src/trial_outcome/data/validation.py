"""Shared checks for the identifiers used to join trial data."""

import pandas as pd


def validate_nct_ids(ids: pd.Series | pd.Index, *, source: str) -> None:
    """Require unique ClinicalTrials.gov identifiers (NCT followed by eight digits)."""
    values = pd.Series(ids, dtype=object)
    if values.isna().any():
        raise ValueError(f"{source} has missing nct_id values")
    valid = values.map(lambda value: isinstance(value, str) and len(value) == 11)
    valid &= values.astype("string").str.fullmatch(r"NCT[0-9]{8}", na=False)
    if not valid.all():
        raise ValueError(f"{source} has invalid nct_id values: {values[~valid].head().tolist()}")
    duplicated = values.duplicated()
    if duplicated.any():
        raise ValueError(
            f"{source} has duplicate nct_id values: {values[duplicated].head().tolist()}"
        )
