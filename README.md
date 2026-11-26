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
├── reports/baseline/       # baseline model comparison outputs (make baseline)
├── reports/models/         # tuned three-model comparison outputs (make models)
├── src/trial_outcome/      # importable project package
│   ├── config.py           #   paths and constants
│   ├── data/ctgov.py        #   ClinicalTrials.gov API v2 client + CLI
│   ├── data/pipeline.py    #   raw snapshot -> features, audit, optional labeled dataset
│   ├── data/top.py         #   TOP labels, ICD codes, criteria text -> per-phase datasets
│   ├── labeling.py         #   load/validate external labels (1 = success, 0 = failure)
│   ├── features.py         #   study records -> feature table; ICD-10 chapters
│   ├── preprocessing.py    #   feature table -> model matrix (encoding, imputation, text)
│   ├── modeling.py         #   logit, XGBoost, and random forest pipelines, persistence
│   ├── baseline.py         #   logit vs. XGBoost chain-length comparison + CLI
│   ├── tuning.py           #   cross-validated tuning: logit, XGBoost, random forest
│   ├── compare.py          #   tuned three-model comparison on the test sets + CLI
│   ├── evaluation.py       #   ROC AUC, PR AUC, Brier, log loss, thresholded metrics
│   └── visualization.py    #   ROC / PR / calibration and chain-length plots, figure saving
└── tests/                  # pytest suite (also runs the docstring examples in src/)
```

The data pipeline extracts design and eligibility features from each trial's registry record, adds TOP's ICD-10 disease codes and eligibility criteria text, and joins them to an explicitly labeled dataset. Labels and the train/test split come from the TOP benchmark (see [Building the TOP labeled datasets](#building-the-top-labeled-datasets)). The baseline models are an L2-regularized logistic regression and XGBoost (see [Baseline models](#baseline-models)). The final comparison tunes three models (logit, XGBoost, and random forest) and tests them (see [Tuned model comparison](#tuned-model-comparison)).

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

1. `python -m trial_outcome.data.top labels`: downloads the nine TOP files at a pinned commit into `data/external/top/`, checks their checksums, and writes `data/interim/top_labels/phase_{I,II,III}.csv` plus `nct_ids.txt`. It also writes `data/interim/top_features.parquet`, one row per trial with TOP's ICD-10 chapters and categories and its eligibility criteria text.
2. `python -m trial_outcome.data.ctgov --ids-file data/interim/top_labels/nct_ids.txt --out data/raw/ctgov_top_<date>.jsonl`: downloads the current registry record for each TOP trial, 200 IDs per request (about 60 requests).
3. `python -m trial_outcome.data.top datasets --raw ... --output-root data/processed/top_<date>_v2`: runs the pipeline once per phase into `phase_I/`, `phase_II/`, and `phase_III/`, adding the TOP feature table to the registry features. `--registry-only` leaves it out.

`<date>` is `TOP_DATE`, pinned in the `Makefile` to `2026-11-22`, the snapshot recorded in `data/README.md`. The current build is `top_2026-11-22_v2`; the first build, `top_2026-11-22`, has the registry features only, and its reports are kept for comparison. Re-running `make top-data` skips steps whose output already exists, so a second run does nothing. For a fresh registry snapshot, run `make top-data TOP_DATE=$(date +%F)` and record it in `data/README.md`. The pipeline never overwrites a build, so to rebuild from the same snapshot, pass `TOP_OUTPUT=data/processed/<new name>`. Each labels file keeps TOP's `split`, `status`, and `why_stop` columns for analysis; the build reads only `nct_id` and `label`, so they never reach `dataset.parquet`. To split a dataset for modeling:

```python
import pandas as pd
from trial_outcome.data.top import load_splits

dataset = pd.read_parquet("data/processed/top_2026-11-22_v2/phase_II/dataset.parquet")
split = load_splits("data/interim/top_labels/phase_II.csv").reindex(dataset.index)
train, valid, test = (dataset[split == name] for name in ("train", "valid", "test"))
```

Trials registered as phase 1/2 or 2/3 appear in both phases, always with the same label, and never in one phase's test set and another phase's training set (the loader checks both). TOP is licensed for non-commercial academic use.

### Baseline models

`make baseline` compares two models on each phase:

- **Regularized logit:** L2-regularized logistic regression. Classes are not reweighted, so its probabilities stay calibrated to the training success rate.
- **XGBoost:** gradient-boosted trees, at several boosting chain lengths (numbers of trees).

```bash
make baseline
# or, to change the candidates:
python -m trial_outcome.baseline --dataset-root data/processed/top_2026-11-22_v2 \
  --chain-lengths 10 25 50 100 200 400 800 1600 --learning-rate 0.05 --max-depth 4
```

Both models share the preprocessing in `trial_outcome.preprocessing`, so they see the same features:

- Categorical columns are one-hot encoded, with missing values as their own category.
- Multi-valued columns are split into one 0/1 column per value. A condition, intervention type, or category needs at least `--min-count` training trials (default 10) to get its own column.
- Numeric columns are median-imputed, given a missing-value indicator, and standardized.
- TOP's ICD-10 chapters (e.g. neoplasms) and three-character categories (e.g. C50, breast cancer) are multi-valued columns like the conditions.
- TOP's eligibility criteria text becomes 50 dense columns: TF-IDF weights of its words and two-word phrases (each in at least `--min-count` training trials), reduced by truncated SVD (latent semantic analysis) and standardized.

Every step is fitted on the training split only. The TOP columns are encoded only when the dataset has them, so builds made with `--registry-only` still work. In 5-fold cross-validation over train and valid, adding the ICD codes and the criteria text each lowered log loss in every phase, for both the logit and XGBoost. TOP's drug names and counts of listed criteria did not, so they are not used.

How the models are fitted, selected, and scored:

1. Each model is fitted on TOP's train split.
2. The logit's `C` and XGBoost's chain length are each chosen by **validation log loss**.
3. Every candidate is scored on train, valid, and test.

Report the selected models' test scores. The other candidates' test scores are only there to draw the curves; choosing a model by them would leak the test set. XGBoost is fitted once at the longest chain length, and each shorter length is scored from its first trees. This gives exactly the same predictions as fitting the shorter chain (`tests/test_modeling.py` checks this).

Outputs go to `reports/baseline/<build name>/`:

- `summary.csv`: the selected models' validation and test scores
- `metrics.csv`: every candidate on every split
- `logit_coefficients_phase_<P>.csv`: the selected logit's coefficients
- `figures/`: chain-length curves and test ROC / PR / calibration plots
- `run.json`: configuration, input hashes, and library versions

The selected models are saved to `models/baseline_{logit,xgboost}_phase_<P>.joblib`.

### Tuned model comparison

`make models` tunes three models on each phase and compares them on TOP's test split:

- **Regularized logit:** the baseline's logit, with `C` tuned.
- **XGBoost:** a random search over depth, learning rate, subsampling, and regularization (30 settings by default, starting with the baseline's), each at every chain length.
- **Random forest:** a random search over features per split, leaf size, and depth (20 settings by default, starting with 500 trees, `sqrt` features, leaf size 5).

All three use the same preprocessing as the baseline, and every model is selected by log loss.

```bash
make models
# or, to change the search:
python -m trial_outcome.compare --dataset-root data/processed/top_2026-11-22_v2 \
  --tuning cv --folds 5 --xgboost-candidates 30 --forest-candidates 20
```

**Tuning never sees the test split.** By default (`--tuning cv`), each model is tuned with stratified 5-fold cross-validation over TOP's train and valid splits together, then refitted on both. TOP's train and valid splits are a random 90/10 split of the same pre-2014 trials, so pooling them is sound. Selecting among many settings on the valid split alone (117 trials in phase I) tends to fit its noise and overstate the winner. `--tuning valid` tunes on the valid split instead and refits on train, like `make baseline`.

**Testing:** each selected model is scored once on the test split. Test scores come with 95% paired bootstrap intervals (2,000 resamples, the same resampled trials for every model). `test_differences.csv` gives XGBoost minus the logit and the random forest minus the logit. When a difference's interval includes 0, the test set does not show that the two models differ.

Outputs go to `reports/models/<build name>/`:

- `summary.csv`: the selected models' settings, tuning scores (`split == "tuning"`), and test scores
- `test_intervals.csv` and `test_differences.csv`: test scores and differences with 95% intervals
- `tuning_<model>_phase_<P>.csv`: every candidate's tuning scores
- `logit_coefficients_phase_<P>.csv` and `importances_{xgboost,random_forest}_phase_<P>.csv`
- `figures/test_roc_auc.png` and `figures/test_diagnostics_phase_<P>.png`
- `run.json`: configuration, input hashes, and library versions

The selected models are saved to `models/{logit,xgboost,random_forest}_phase_<P>.joblib`. A full run of all three phases took about 15 minutes on 2 CPU cores; more cores make it faster. `--tuning valid` is about 2.5 times faster.

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
| `make baseline` | Compare the regularized logit with XGBoost chain lengths on each TOP phase |
| `make models` | Tune logit, XGBoost, and random forest; compare them on each TOP test set |

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
