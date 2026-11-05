"""Feature engineering: raw ClinicalTrials.gov study records -> model-ready table.

Output contract for :func:`build_features`: a DataFrame indexed by ``nct_id`` with one
column per feature, ready to join with the labels from :mod:`trial_outcome.labeling`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

_MISSING = object()


def get_path(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a nested field from a study record using a dotted path.

    >>> study = {"protocolSection": {"statusModule": {"overallStatus": "COMPLETED"}}}
    >>> get_path(study, "protocolSection.statusModule.overallStatus")
    'COMPLETED'
    >>> get_path(study, "protocolSection.designModule.phases", default=[])
    []
    """
    node: Any = record
    for key in path.split("."):
        if not isinstance(node, Mapping):
            return default
        node = node.get(key, _MISSING)
        if node is _MISSING:
            return default
    return node


def nct_id(study: Mapping[str, Any]) -> str | None:
    """Return a study's NCT identifier."""
    return get_path(study, "protocolSection.identificationModule.nctId")


def build_features(studies: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return a feature table indexed by ``nct_id``."""
    raise NotImplementedError("Feature set not defined yet.")
