from copy import deepcopy

import pandas as pd
import pytest

from trial_outcome.features import FEATURE_DTYPES, build_features, get_path, nct_id

STUDY = {
    "protocolSection": {
        "identificationModule": {"nctId": "NCT01234567"},
        "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 0}},
    }
}


def test_get_path_reads_nested_values():
    assert get_path(STUDY, "protocolSection.designModule.phases") == ["PHASE2"]


def test_get_path_keeps_falsy_values():
    assert get_path(STUDY, "protocolSection.designModule.enrollmentInfo.count", default=-1) == 0


def test_get_path_returns_default_for_missing_or_non_mapping():
    assert get_path(STUDY, "protocolSection.statusModule.overallStatus", "?") == "?"
    assert get_path(STUDY, "protocolSection.designModule.phases.0", "?") == "?"


def test_nct_id():
    assert nct_id(STUDY) == "NCT01234567"


def test_build_features_types_missing_values_and_sorted_categories(study_records):
    frame = build_features(reversed(study_records))

    assert frame.index.tolist() == ["NCT00000001", "NCT00000002", "NCT00000003"]
    assert frame.index.name == "nct_id"
    assert frame.dtypes.astype(str).to_dict() == FEATURE_DTYPES
    first = frame.loc["NCT00000001"]
    assert first["phases"] == "PHASE1|PHASE2"
    assert first["conditions"] == "Asthma"
    assert first["intervention_types"] == "BEHAVIORAL|DRUG"
    assert not first["healthy_volunteers"]
    assert first["minimum_age_years"] == 0.5
    assert pd.isna(first["maximum_age_years"])
    assert first["arm_count"] == 2
    assert frame.loc["NCT00000002", "arm_count"] == 0
    assert frame.loc["NCT00000002", "minimum_age_years"] == 0
    assert pd.isna(frame.loc["NCT00000003", "arm_count"])
    assert pd.isna(frame.loc["NCT00000002", "healthy_volunteers"])


def test_outcome_fields_and_enrollment_do_not_change_features(study_records):
    expected = build_features(study_records)
    modified = deepcopy(study_records)
    for study in modified:
        study["protocolSection"]["statusModule"] = {"overallStatus": "WITHDRAWN"}
        study["protocolSection"]["designModule"]["enrollmentInfo"] = {
            "count": 999,
            "type": "ESTIMATED",
        }
        study["resultsSection"] = {"new_result": "success"}
        study["hasResults"] = False
    pd.testing.assert_frame_equal(build_features(modified), expected)


@pytest.mark.parametrize("trial_id", [None, "", "NCT1", "nct00000001", 123])
def test_build_features_rejects_invalid_ids(trial_id):
    record = {"protocolSection": {"identificationModule": {"nctId": trial_id}}}
    with pytest.raises(ValueError, match="nct_id"):
        build_features([record])


def test_build_features_rejects_duplicates(study_records):
    with pytest.raises(ValueError, match="duplicate"):
        build_features([study_records[0], study_records[0]])


@pytest.mark.parametrize(
    "age,expected", [("18 Years", 18), ("12 Months", 1), ("7 Days", 7 / 365.25)]
)
def test_age_units(age, expected, study_records):
    study_records[0]["protocolSection"]["eligibilityModule"]["minimumAge"] = age
    assert build_features(study_records).loc["NCT00000001", "minimum_age_years"] == expected


def test_invalid_age_fails(study_records):
    study_records[0]["protocolSection"]["eligibilityModule"]["minimumAge"] = "unknown"
    with pytest.raises(ValueError, match="invalid eligibility age"):
        build_features(study_records)


def test_empty_features_keep_schema():
    frame = build_features([])
    assert frame.empty
    assert frame.dtypes.astype(str).to_dict() == FEATURE_DTYPES
