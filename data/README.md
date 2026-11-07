# Data

Data files are **not committed**: everything under these folders except `.gitkeep` is gitignored. Each teammate regenerates the data locally with the code or notebooks in this repo. Paths are available in code from `trial_outcome.config`.

| Folder | Contents | Written by |
| --- | --- | --- |
| `raw/` | Untouched downloads: ClinicalTrials.gov API pulls, or the full dataset from the provided notebook | `trial_outcome.data.ctgov`, download notebook |
| `external/` | Third-party data, such as an existing labeled set of trials | Manual download |
| `interim/` | Cleaned, filtered, or joined intermediate tables | Pipeline code |
| `processed/` | Typed feature tables, optional labels, audit, and manifest | `trial_outcome.data.pipeline` |

Treat `raw/` and `external/` as read-only. Rebuild anything downstream from them.

## Pipeline contract

Run `python -m trial_outcome.data.pipeline --help` for the CLI. The required input is a local JSON Lines snapshot with one full ClinicalTrials.gov API v2 study object per line. The build defaults to `INTERVENTIONAL` studies; `OBSERVATIONAL` and `ANY` are also supported. Missing study types are excluded by a specific-type filter and retained by `ANY`. Duplicate or invalid identifiers, malformed records, empty cohorts, invalid labels, and labels with no cohort overlap fail the build instead of silently producing unusable data.

Each run writes a **new** directory under `data/processed/`. Tables are sorted by a unique `nct_id` index (`NCT` plus eight digits). The directory appears only after every output is written successfully.

| Output | Contents |
| --- | --- |
| `features.parquet` | All included trials and the typed feature columns below, including unlabeled trials |
| `cohort_audit.parquet` | All raw trial IDs, study type, status, registry dates, inclusion flag, and exclusion reason; adds `has_label` when labels are supplied |
| `dataset.parquet` | Only included trials with supplied labels; feature columns plus integer `label` |
| `unmatched_labels.parquet` | Supplied labels excluded from the joined dataset, with `not_in_raw` or `excluded_from_cohort` reasons |
| `manifest.json` | UTC build time, schema version, input paths and SHA-256 hashes, cohort configuration, label description, counts, feature types/missingness, output hashes, and limitations |

The last two Parquet files are written only when `--labels` is supplied. Unlabeled rows are counted but never assigned a negative label. Extra label-source columns are dropped before joining, so they cannot accidentally become predictors. Input labels must already use canonical IDs and binary 0/1 values; duplicate labels are rejected even when the values agree. `--label-description` must identify the external source and explain what success and failure mean.

```python
import pandas as pd

features = pd.read_parquet("data/processed/breast_cancer_sample/features.parquet")
# Once real labels have been supplied:
dataset = pd.read_parquet("data/processed/breast_cancer_labeled/dataset.parquet")
X = dataset.drop(columns="label")
y = dataset["label"]
```

## Initial features

The extraction paths follow the [ClinicalTrials.gov API v2 study structure](https://clinicaltrials.gov/data-api/about-api/study-data-structure). All paths below are within `protocolSection`.

| Feature(s) | Source | Representation |
| --- | --- | --- |
| `study_type` | `designModule.studyType` | Nullable string |
| `allocation`, `intervention_model`, `primary_purpose` | `designModule.designInfo` | Nullable strings |
| `masking` | `designModule.designInfo.maskingInfo.masking` | Nullable string |
| `sponsor_class` | `sponsorCollaboratorsModule.leadSponsor.class` | Nullable string |
| `sex`, `healthy_volunteers` | `eligibilityModule` | Nullable string / boolean |
| `minimum_age_years`, `maximum_age_years` | `eligibilityModule.minimumAge` / `maximumAge` | Nullable floats; months / 12, days / 365.25, equivalent week/hour/minute conversions; `N/A` is missing |
| `phases` | `designModule.phases` | Sorted unique values joined with `\|` |
| `conditions` | `conditionsModule.conditions` | Sorted unique values joined with `\|` |
| `intervention_types` | `armsInterventionsModule.interventions[].type` | Sorted unique values joined with `\|` |
| `arm_count` | `armsInterventionsModule.armGroups` | Nullable integer |

Missing lists remain missing; explicitly empty lists become an empty category string or a zero count. False booleans and zero ages are preserved. Multi-valued strings are an initial representation; choose suitable categorical or multi-label encoding when implementing the model.

Status, termination reasons, results, registry dates, enrollment (actual or estimated), and narrative descriptions are excluded from the feature allowlist. Status and dates are kept only in the separate audit table. **This is not a guarantee against temporal leakage:** current API records may have been edited after registration, including their design and eligibility fields. Before claiming prospective performance, define a prediction time and obtain appropriate historical snapshots or assess remaining leakage. No completion-status proxy label is provided.

Feature values are not imputed, scaled, or encoded here. Decide the split strategy after selecting the cohort and label source, then fit preprocessing only on training data. The initial builder holds the snapshot and feature tables in memory; start with a bounded download before attempting a full-registry build.

## Provenance

Record every dataset here so results can be reproduced.

Keep the exact API query, download command, and retrieval date with each raw snapshot. The processed manifest fingerprints the saved inputs; it cannot reconstruct an unrecorded download query or recover an unavailable label source. Raw data, external labels, and generated outputs remain gitignored.

| File | Source | Retrieved | Query / notes |
| --- | --- | --- | --- |
| `raw/ctgov_asthma_smoke_2026-11-07.jsonl` | ClinicalTrials.gov API v2 | 2026-11-07 | `python -m trial_outcome.data.ctgov --param query.cond=asthma --max-studies 100 --out data/raw/ctgov_asthma_smoke_2026-11-07.jsonl`; live validation sample, not a selected modeling cohort |

The validation sample was built with `python -m trial_outcome.data.pipeline --raw data/raw/ctgov_asthma_smoke_2026-11-07.jsonl --output-dir data/processed/asthma_smoke_2026-11-07`. It contained 100 studies: 70 interventional studies were included and 30 were excluded. No clinical-success labels were supplied. Registry data and default result ordering can change, so a fresh download may differ; the local manifest records this snapshot's exact hash.
