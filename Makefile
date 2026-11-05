VENV := dev
BIN := $(VENV)/bin
PYTHON_VERSION := $(shell cat .python-version)
UV := $(shell command -v uv 2>/dev/null)

.PHONY: help setup hooks lint format test

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-8s %s\n", $$1, $$2}'

setup: ## Create the `dev` venv, install dependencies, and install git hooks
ifdef UV
	uv venv $(VENV) --python $(PYTHON_VERSION) --seed --allow-existing
	uv pip install --python $(BIN)/python -r requirements-dev.txt
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
