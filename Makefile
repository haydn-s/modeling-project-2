VENV := dev
BIN := $(VENV)/bin
PYTHON_VERSION := $(shell cat .python-version)
UV := $(shell command -v uv 2>/dev/null)
# PyTorch build for uv to install: auto (CUDA if an NVIDIA driver is found,
# else CPU), cpu, or a CUDA version such as cu128. CI always uses cpu.
TORCH_BACKEND ?= auto

# TOP labeled datasets: `make top-data`. TOP_DATE names the registry snapshot and the build, and is
# pinned to the snapshot recorded in data/README.md so re-runs reuse it. For a fresh snapshot, pass
# TOP_DATE=$$(date +%F); to rebuild from the same snapshot, pass a new TOP_OUTPUT.
TOP_DATE ?= 2026-11-22
TOP_LABELS := data/interim/top_labels
TOP_SNAPSHOT ?= data/raw/ctgov_top_$(TOP_DATE).jsonl
TOP_OUTPUT ?= data/processed/top_$(TOP_DATE)
# Phases are built in order (I, II, III), so phase III's manifest marks a finished build.
TOP_BUILT := $(TOP_OUTPUT)/phase_III/manifest.json

.PHONY: help setup hooks lint format test notebook top-data baseline models

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-8s %s\n", $$1, $$2}'

setup: ## Create the `dev` venv, install dependencies, and install git hooks
ifdef UV
	uv venv $(VENV) --python $(PYTHON_VERSION) --seed --allow-existing
	uv pip install --python $(BIN)/python --torch-backend $(TORCH_BACKEND) -r requirements-dev.txt
else
	python$(PYTHON_VERSION) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
endif
	$(MAKE) hooks

hooks: ## Install the pre-commit and commit-msg git hooks
	$(BIN)/pre-commit install

lint: ## Run every pre-commit hook against all files
	$(BIN)/pre-commit run --all-files

format: ## Auto-fix lint issues and format Python + notebooks with ruff
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test: ## Run the test suite
	$(BIN)/pytest

notebook: ## Launch JupyterLab using the dev venv
	$(BIN)/jupyter lab

top-data: $(TOP_BUILT) ## Download TOP labels + their registry records; build one dataset per phase

baseline: $(TOP_BUILT) ## Compare the regularized logit with XGBoost chain lengths on each TOP phase
	$(BIN)/python -m trial_outcome.baseline --dataset-root $(TOP_OUTPUT) --labels-dir $(TOP_LABELS)

models: $(TOP_BUILT) ## Tune logit, XGBoost, and random forest; compare them on each TOP test set
	$(BIN)/python -m trial_outcome.compare --dataset-root $(TOP_OUTPUT) --labels-dir $(TOP_LABELS)

$(TOP_BUILT): $(TOP_SNAPSHOT)
	$(BIN)/python -m trial_outcome.data.top datasets \
		--raw $(TOP_SNAPSHOT) --labels-dir $(TOP_LABELS) --output-root $(TOP_OUTPUT)

$(TOP_LABELS)/nct_ids.txt:
	$(BIN)/python -m trial_outcome.data.top labels --out-dir $(TOP_LABELS)

$(TOP_SNAPSHOT): $(TOP_LABELS)/nct_ids.txt
	$(BIN)/python -m trial_outcome.data.ctgov --ids-file $< --out $@
