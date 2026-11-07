import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend for tests / CI


@pytest.fixture
def study_records():
    """Synthetic API v2 records covering complete, sparse, and excluded studies."""
    return [
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "studyFirstPostDateStruct": {"date": "2010-01-01"},
                    "lastUpdatePostDateStruct": {"date": "2020-01-01"},
                },
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE2", "PHASE1", "PHASE2"],
                    "designInfo": {"allocation": "RANDOMIZED"},
                    "enrollmentInfo": {"count": 100, "type": "ACTUAL"},
                },
                "conditionsModule": {"conditions": ["Asthma", "Asthma"]},
                "armsInterventionsModule": {
                    "interventions": [{"type": "DRUG"}, {"type": "BEHAVIORAL"}],
                    "armGroups": [{"label": "Treatment"}, {"label": "Control"}],
                },
                "eligibilityModule": {
                    "healthyVolunteers": False,
                    "sex": "ALL",
                    "minimumAge": "6 Months",
                    "maximumAge": "N/A",
                },
            },
            "hasResults": True,
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": []}},
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000002"},
                "statusModule": {"overallStatus": "TERMINATED", "whyStopped": "Test reason"},
                "designModule": {"studyType": "INTERVENTIONAL"},
                "armsInterventionsModule": {"armGroups": []},
                "eligibilityModule": {"minimumAge": "0 Years"},
            }
        },
        {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000003"},
                "designModule": {"studyType": "OBSERVATIONAL"},
            }
        },
    ]
