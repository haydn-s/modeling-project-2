import matplotlib
import numpy as np
import pandas as pd
import pytest

from trial_outcome.features import FEATURE_DTYPES

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


@pytest.fixture
def labeled_features():
    """A synthetic ``dataset.parquet`` table whose label depends on sponsor class and age."""
    rng = np.random.default_rng(0)
    n = 300
    sponsor = rng.choice(["INDUSTRY", "OTHER", "NIH"], size=n)
    age = rng.uniform(18, 65, size=n)
    log_odds = 1.5 * (sponsor == "INDUSTRY") - 0.08 * (age - 40)
    frame = pd.DataFrame(
        {
            "study_type": "INTERVENTIONAL",
            "allocation": rng.choice(["RANDOMIZED", "NA", None], size=n),
            "intervention_model": rng.choice(["PARALLEL", "SINGLE_GROUP"], size=n),
            "primary_purpose": rng.choice(["TREATMENT", "TREATMENT", "SCREENING"], size=n),
            "masking": rng.choice(["NONE", "DOUBLE"], size=n),
            "sponsor_class": sponsor,
            "sex": "ALL",
            "phases": rng.choice(["PHASE2", "PHASE1|PHASE2"], size=n),
            "conditions": rng.choice(["Asthma", "asthma|COPD", "Breast  Cancer", None], size=n),
            "intervention_types": rng.choice(["DRUG", "BIOLOGICAL|DRUG"], size=n),
            "healthy_volunteers": rng.choice([True, False, None], size=n),
            "minimum_age_years": age,
            "maximum_age_years": np.where(rng.uniform(size=n) < 0.5, np.nan, 80.0),
            "arm_count": rng.choice([1, 2, 3, None], size=n),
        },
        index=pd.Index([f"NCT{i:08d}" for i in range(1, n + 1)], name="nct_id"),
    ).astype(FEATURE_DTYPES)
    frame["label"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return frame


@pytest.fixture
def top_labeled_features(labeled_features):
    """``labeled_features`` plus TOP's ICD and criteria columns, as in a v2 ``dataset.parquet``.

    Successful trials mention "placebo controlled" in their criteria more often.
    """
    rng = np.random.default_rng(1)
    words = np.array(["adults", "asthma", "consent", "pregnant", "smokers", "renal", "hepatic"])
    criteria = [
        "Inclusion Criteria: "
        + " ".join(rng.choice(words, size=6))
        + (" placebo controlled" if rng.uniform() < (0.7 if label else 0.2) else "")
        + "\nExclusion Criteria: "
        + " ".join(rng.choice(words, size=3))
        for label in labeled_features["label"]
    ]
    frame = labeled_features.drop(columns="label")
    frame["icd_chapters"] = pd.array(
        rng.choice(["neoplasms", "blood_immune|neoplasms", "respiratory", ""], size=len(frame)),
        dtype="string",
    )
    frame["icd_categories"] = pd.array(
        rng.choice(["C50", "C50|D63", "J45", ""], size=len(frame)), dtype="string"
    )
    frame["criteria"] = pd.array(criteria, dtype="string")
    frame.iloc[::50, frame.columns.get_loc("criteria")] = pd.NA
    frame["label"] = labeled_features["label"]
    return frame
