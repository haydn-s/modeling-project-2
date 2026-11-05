# Data

Data files are **not committed**: everything under these folders except `.gitkeep` is gitignored. Each teammate regenerates the data locally with the code or notebooks in this repo. Paths are available in code from `trial_outcome.config`.

| Folder | Contents | Written by |
| --- | --- | --- |
| `raw/` | Untouched downloads: ClinicalTrials.gov API pulls, or the full dataset from the provided notebook | `trial_outcome.data.ctgov`, download notebook |
| `external/` | Third-party data, such as an existing labeled set of trials | Manual download |
| `interim/` | Cleaned, filtered, or joined intermediate tables | Pipeline code |
| `processed/` | Final model-ready tables (features + labels) | Pipeline code |

Treat `raw/` and `external/` as read-only. Rebuild anything downstream from them.

## Provenance

Record every dataset here so results can be reproduced.

| File | Source | Retrieved | Query / notes |
| --- | --- | --- | --- |
| | | | |
