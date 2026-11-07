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
│   ├── labeling.py         #   load/validate external labels (1 = success, 0 = failure)
│   ├── features.py         #   study records -> feature table
│   ├── modeling.py         #   training and model persistence
│   ├── evaluation.py       #   ROC AUC, PR AUC, Brier, log loss, thresholded metrics
│   └── visualization.py    #   ROC / PR / calibration plots, figure saving
└── tests/                  # pytest suite (also runs the docstring examples in src/)
```

The data pipeline extracts an initial set of design and eligibility features and can join an explicitly labeled dataset. Choosing the clinical-success label source, train/test split, and model remains project work; `modeling.train_model` is still a placeholder.

### Getting data from ClinicalTrials.gov

```bash
# How many trials match?
python -m trial_outcome.data.ctgov --out /dev/null --count-only --param query.cond="breast cancer"
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

## Development setup

Requires Python 3.12 (see `.python-version`). [uv](https://docs.astral.sh/uv/) is used if it's installed; otherwise the standard `venv` + `pip` path is used.

```bash
make setup
```

This creates a virtual environment in `dev/`, installs `requirements-dev.txt` (the ML / data-science stack plus tooling), and installs the git hooks. Activate it with:

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
