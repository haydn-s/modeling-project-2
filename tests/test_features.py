from trial_outcome.features import get_path, nct_id

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
