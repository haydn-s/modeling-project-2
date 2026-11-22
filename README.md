# AIPI 520 — Modeling Project 2: Clinical Trial Outcome Prediction

Team 9: Haydn Stucker, Sabrina Grewal

## Project

**Objective:** build a model that predicts the probability that a clinical trial succeeds.

- **Data:** ClinicalTrials.gov, pulled through the API (`trial_outcome.data.ctgov`) or downloaded in full with the provided notebook
- **Labels:** an existing labeled set of trials, or a set we label ourselves (`trial_outcome.labeling`)
- **Model:** any algorithm

### Deliverables

- [ ] **Presentation** (10 min max): objective, data used, labeling approach, data pipeline, features, models, evaluation approach, performance, and interpretation. Explain the decisions we made and why.
- [ ] **Writeup** (2–4 pages, `docs/writeup.md`): our approach and how we applied what we learned in class. *No AI use allowed for the writeup.*
- [ ] **Code repo** (this repository)

## Repository layout

```
├── data/                   # local only (gitignored); see data/README.md
│   ├── raw/                #   untouched downloads
│   ├── external/           #   third-party data, e.g. an existing labeled set
│   ├── interim/            #   intermediate tables
│   └── processed/          #   model-ready features + labels
├── docs/writeup.md         # project writeup (human-written)
├── models/                 # trained model artifacts (gitignored)
├── notebooks/              # exploration; see notebooks/README.md for naming
├── reports/figures/        # figures for the presentation
├── src/trial_outcome/      # importable project package
│   ├── config.py           #   paths and constants
│   ├── data/ctgov.py        #   ClinicalTrials.gov API v2 client + CLI
│   ├── data/pipeline.py    #   raw snapshot -> features, audit, optional labeled dataset
│   ├── data/top.py         #   TOP benchmark labels -> per-phase labeled datasets
│   ├── labeling.py         #   load/validate external labels (1 = success, 0 = failure)
│   ├── features.py         #   study records -> feature table
│   ├── modeling.py         #   training and model persistence
│   ├── evaluation.py       #   ROC AUC, PR AUC, Brier, log loss, thresholded metrics
│   └── visualization.py    #   ROC / PR / calibration plots, figure saving
└── tests/                  # pytest suite (also runs the docstring examples in src/)
```

The data pipeline extracts an initial set of design and eligibility features and joins them to an explicitly labeled dataset. Labels and the train/test split come from the TOP benchmark (see [Building the TOP labeled datasets](#building-the-top-labeled-datasets)). The model is still to be chosen; `modeling.train_model` is a placeholder.

### Getting data from ClinicalTrials.gov

```bash
# How many trials match?
python -m trial_outcome.data.ctgov --count-only --param query.cond="breast cancer"
```

```bash
# Download them as JSON Lines into data/raw/
python -m trial_outcome.data.ctgov \
  --param query.cond="breast cancer" \
  --param "filter.overallStatus=COMPLETED|TERMINATED" \
  --out data/raw/ctgov_breast_cancer.jsonl
```

Any [API v2 query parameter](https://clinicaltrials.gov/data-api/api) can be passed with `--param`. Record each pull in `data/README.md`.

### Building a dataset

After activating the development environment, download a small snapshot and build the initial feature tables:

```bash
python -m trial_outcome.data.ctgov \
  --param query.cond="breast cancer" \
  --max-studies 500 \
  --out data/raw/ctgov_breast_cancer_sample.jsonl

python -m trial_outcome.data.pipeline \
  --raw data/raw/ctgov_breast_cancer_sample.jsonl \
  --output-dir data/processed/breast_cancer_sample
```

The default cohort includes interventional studies. Use `--study-type OBSERVATIONAL` or `--study-type ANY` to change it. The build runs entirely from the saved snapshot, writes Parquet files plus a manifest, and refuses to overwrite an existing output directory. Use a new output directory for another run. Downloads also write atomically, so a failed request preserves any previous snapshot; use a distinct raw filename to retain a previous successful pull.

To include clinical-success labels, supply a CSV or Parquet file with unique `nct_id` values and a binary `label` column, together with its source and outcome definition:

```bash
python -m trial_outcome.data.pipeline \
  --raw data/raw/ctgov_breast_cancer_sample.jsonl \
  --labels data/external/trial_labels.csv \
  --label-description "SOURCE: replace with the dataset citation and success/failure definition" \
  --output-dir data/processed/breast_cancer_labeled
```

The labels file must be obtained separately. Completed status is **not** treated as clinical success. Without a labels file, the pipeline produces features and an audit for exploration; it does not fabricate a target. See [data/README.md](data/README.md) for the output contract, feature definitions, and remaining modeling decisions.

### Building the TOP labeled datasets

Our labels come from **TOP** (Trial Outcome Prediction), the benchmark released with the HINT paper (Fu et al., 2022, *Patterns*). It holds 11,601 phase I–III trials labeled success (1) or failure (0); the labels come from IQVIA. TOP's split is by date: train and valid hold trials that completed before 2014, and test holds trials that started in 2014 or later. Using TOP's split means we train on the past, predict the future, and score on the same test trials as HINT.

One command downloads everything and builds one labeled dataset per phase:

```bash
make top-data
```

It runs three steps, each of which can also be run alone:

1. `python -m trial_outcome.data.top labels`: downloads the nine TOP files at a pinned commit into `data/external/top/`, checks their checksums, and writes `data/interim/top_labels/phase_{I,II,III}.csv` plus `nct_ids.txt`.
2. `python -m trial_outcome.data.ctgov --ids-file data/interim/top_labels/nct_ids.txt --out data/raw/ctgov_top_<date>.jsonl`: downloads the current registry record for each TOP trial, 200 IDs per request (about 60 requests).
3. `python -m trial_outcome.data.top datasets --raw ... --output-root data/processed/top_<date>`: runs the pipeline once per phase into `phase_I/`, `phase_II/`, and `phase_III/`.

`<date>` is `TOP_DATE`, pinned in the `Makefile` to `2026-11-22`, the snapshot recorded in `data/README.md`. Re-running `make top-data` skips steps whose output already exists, so a second run does nothing. For a fresh registry snapshot, run `make top-data TOP_DATE=$(date +%F)` and record it in `data/README.md`. The pipeline never overwrites a build, so to rebuild from the same snapshot, pass `TOP_OUTPUT=data/processed/<new name>`. Each labels file keeps TOP's `split`, `status`, and `why_stop` columns for analysis; the build reads only `nct_id` and `label`, so they never reach `dataset.parquet`. To split a dataset for modeling:

```python
import pandas as pd
from trial_outcome.data.top import load_splits

dataset = pd.read_parquet("data/processed/top_2026-11-22/phase_II/dataset.parquet")
split = load_splits("data/interim/top_labels/phase_II.csv").reindex(dataset.index)
train, valid, test = (dataset[split == name] for name in ("train", "valid", "test"))
```

Trials registered as phase 1/2 or 2/3 appear in both phases, always with the same label, and never in one phase's test set and another phase's training set (the loader checks both). TOP is licensed for non-commercial academic use.

## Development setup

Requires Python 3.12 (see `.python-version`). [uv](https://docs.astral.sh/uv/) is used if it's installed; otherwise the standard `venv` + `pip` path is used.

```bash
make setup
```

This creates a virtual environment in `dev/`, installs `requirements-dev.txt` (the ML / data-science stack plus tooling), and installs the git hooks.

With uv, PyTorch is installed for your hardware: the CUDA build if an NVIDIA driver is found, otherwise the CPU build (on macOS, the standard build, which supports Apple's MPS GPU). To force a build, run `make setup TORCH_BACKEND=cpu` (or a CUDA version such as `cu128`). Without uv, pip installs PyPI's default torch, which on Linux is the multi-GB CUDA build.

Activate the environment with:

```bash
source dev/bin/activate
```

In Jupyter or VS Code, choose `dev/bin/python` as the kernel / interpreter.

| File | Purpose |
| --- | --- |
| `requirements.txt` | Runtime libraries: numpy, pandas, scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, Optuna, matplotlib, seaborn, plotly, JupyterLab, ... |
| `requirements-dev.txt` | Everything above plus ruff, pre-commit, and pytest |

## Everyday commands

| Command | What it does |
| --- | --- |
| `make format` | Auto-fix lint issues and format Python files and notebooks with ruff |
| `make lint` | Run every pre-commit hook against the whole repo |
| `make test` | Run the test suite with pytest |

## Conventions

### Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org), enforced by [commitlint](https://commitlint.js.org) through a `commit-msg` hook and again in CI:

```
<type>(<optional scope>): <subject>

feat(data): add ClinicalTrials.gov API client
fix(features): handle trials with missing enrollment
docs: describe labeling approach
```

Allowed types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, `test`. Pull request titles must follow the same format, because they become the commit message on a squash merge.

### Formatting and linting

[ruff](https://docs.astral.sh/ruff/) formats and lints Python files and Jupyter notebooks. It runs automatically on every commit and in CI; configuration lives in `pyproject.toml`. Other pre-commit hooks trim trailing whitespace, check YAML/TOML, and block files larger than 1 MB (keep datasets out of git).

### Branching

`main` is protected (see `scripts/protect-main.sh`):

- Every change lands through a pull request. Direct pushes, force pushes, and deleting `main` are all blocked.
- The CI checks `lint`, `commitlint`, and `test` must pass, and the branch must be up to date with `main`.
- A PR needs one approving review, and all review conversations must be resolved.
