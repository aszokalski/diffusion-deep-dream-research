# GLOBALS
# In Make, we don't need quotes for simple strings
project_name := diffusion-deep-dream-research
python_version := 3.12

# Shell setup to ensure bash syntax works (like pipefail)
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# .PHONY tells Make that these are commands, not file names
.PHONY: help update_packages lock requirements create_environment clean lint format test submodules models _check-deps

# COMMANDS

# Default target runs help
.DEFAULT_GOAL := help

# A script to extract comments starting with '##' to create a help menu
help:
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## Update lockfile and sync dependencies
update_packages: lock requirements

## Compile requirements.in to requirements.lock using uv
lock:
	@echo "Locking dependencies from requirements.in..."
	uv pip compile requirements.in -o requirements.lock

## Sync environment with locked requirements and install project
requirements:
	@echo "Syncing dependencies..."
	uv pip sync requirements.lock
	@echo "Installing project in editable mode..."
	uv pip install --no-deps -e .

## Helper to check for envsubst
_check-deps:
	@command -v envsubst >/dev/null || { echo "❌ Error: 'envsubst' is required. Install 'gettext'."; exit 1; }

## Destroy and recreate the Conda environment
create_environment: _check-deps
	@echo "Creating Conda Env: $(project_name) (Python $(python_version))"
	@# Remove any existing temp file
	@rm -f .environment.tmp.yml
	@# Remove existing environment (using \ to continue the shell command)
	@echo "Removing old environment if it exists..."
	@if conda env list | grep -q "$(project_name)"; then \
		echo "Removing existing Conda environment '$(project_name)'..."; \
		conda env remove -n "$(project_name)" -y; \
	fi
	@# Render template. Note: We export variables on the same line for the shell execution
	@ENV_NAME="$(project_name)" PYTHON_VERSION="$(python_version)" envsubst < environment.yml.tpl > .environment.tmp.yml
	@# Create environment
	@conda env create -f .environment.tmp.yml
	@# Cleanup
	@rm .environment.tmp.yml
	@echo "Conda environment created."
	@echo "NEXT STEPS:"
	@echo "    1. conda activate $(project_name)"
	@echo "    2. make requirements"

## Remove cache and temporary files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .environment.tmp.yml

## Check code style with Ruff
lint:
	ruff format --check
	ruff check

## Fix code style with Ruff
format:
	ruff check --fix
	ruff format

## Run tests with pytest
test:
	python -m pytest tests

## Initialize/Update git submodules
submodules:
	git submodule update --init --recursive

## Pull models via python script
models:
	python scripts/00_pull_models.py