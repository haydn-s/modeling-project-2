"""Feature engineering: raw ClinicalTrials.gov study records -> model-ready table.

Output contract for :func:`build_features`: a DataFrame indexed by ``nct_id`` with one
column per feature, ready to join with the labels from :mod:`trial_outcome.labeling`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from trial_outcome.data.validation import validate_nct_ids

_MISSING = object()

# An explicit allowlist prevents result/status fields from becoming predictors.
# These are current registry values, not historical registration-time snapshots.
_STRING_PATHS = {
    "study_type": "designModule.studyType",
    "allocation": "designModule.designInfo.allocation",
    "intervention_model": "designModule.designInfo.interventionModel",
    "primary_purpose": "designModule.designInfo.primaryPurpose",
    "masking": "designModule.designInfo.maskingInfo.masking",
    "sponsor_class": "sponsorCollaboratorsModule.leadSponsor.class",
    "sex": "eligibilityModule.sex",
}
FEATURE_DTYPES = {
    **dict.fromkeys(_STRING_PATHS, "string"),
    "phases": "string",
    "conditions": "string",
    "intervention_types": "string",
    "healthy_volunteers": "boolean",
    "minimum_age_years": "Float64",
    "maximum_age_years": "Float64",
    "arm_count": "Int64",
}


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
    """Extract typed design/eligibility features, sorted and indexed by ``nct_id``.

    Missing values remain missing. Multi-valued categories are sorted, deduplicated,
    and joined with ``|``. Status, results, dates, free text, and enrollment are excluded.
    Encoding and imputation must be fitted on the training partition downstream.
    """
    rows = []
    for study in studies:
        row = {"nct_id": nct_id(study)}
        for name, path in _STRING_PATHS.items():
            value = get_path(study, f"protocolSection.{path}")
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{row['nct_id']}: {path} must be a string")
            row[name] = value
        row["phases"] = _categories(get_path(study, "protocolSection.designModule.phases"))
        row["conditions"] = _categories(
            get_path(study, "protocolSection.conditionsModule.conditions")
        )
        interventions = get_path(study, "protocolSection.armsInterventionsModule.interventions")
        if interventions is not None:
            _object_list(interventions, "interventions")
        row["intervention_types"] = _categories(
            [item["type"] for item in interventions if item.get("type") is not None]
            if interventions is not None
            else None
        )
        healthy = get_path(study, "protocolSection.eligibilityModule.healthyVolunteers")
        if healthy is not None and not isinstance(healthy, bool):
            raise ValueError(f"{row['nct_id']}: healthyVolunteers must be a boolean")
        row["healthy_volunteers"] = healthy
        for bound in ("minimum", "maximum"):
            row[f"{bound}_age_years"] = _age_years(
                get_path(study, f"protocolSection.eligibilityModule.{bound}Age")
            )
        arms = get_path(study, "protocolSection.armsInterventionsModule.armGroups")
        if arms is not None:
            _object_list(arms, "armGroups")
        row["arm_count"] = len(arms) if arms is not None else None
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["nct_id", *FEATURE_DTYPES])
    validate_nct_ids(frame["nct_id"], source="studies")
    frame["nct_id"] = frame["nct_id"].astype("string")
    return frame.set_index("nct_id").astype(FEATURE_DTYPES).sort_index()


def _object_list(value: Any, name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{name} must be a list of objects")


def _categories(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("multi-valued categories must be lists of strings")
    return "|".join(sorted(set(value)))


def _age_years(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s+(years?|months?|weeks?|days?|hours?|minutes?)", str(value), re.I
    )
    if not match:
        raise ValueError(f"invalid eligibility age: {value!r}")
    factors = {
        "year": 1,
        "month": 1 / 12,
        "week": 7 / 365.25,
        "day": 1 / 365.25,
        "hour": 1 / (365.25 * 24),
        "minute": 1 / (365.25 * 24 * 60),
    }
    return float(match[1]) * factors[match[2].lower().rstrip("s")]
