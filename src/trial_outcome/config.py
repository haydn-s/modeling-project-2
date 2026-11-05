"""Project-wide paths and constants.

Import these instead of hard-coding paths so notebooks and scripts work from any directory::

    from trial_outcome.config import RAW_DATA_DIR
"""

from pathlib import Path

# src/trial_outcome/config.py -> repo root (valid for the editable install used in this repo)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # untouched downloads (e.g. ClinicalTrials.gov JSON)
EXTERNAL_DATA_DIR = DATA_DIR / "external"  # third-party data, e.g. an existing labeled set
INTERIM_DATA_DIR = DATA_DIR / "interim"  # cleaned / joined intermediate tables
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # final model-ready tables

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42
