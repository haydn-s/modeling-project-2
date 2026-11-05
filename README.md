# AIPI 520 — Modeling Project 2: Clinical Trial Outcome Prediction

Team 9: Haydn Stucker, Sabrina Grewal

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
