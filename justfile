# GLOBALS
project_name := "diffusion-deep-dream-research"
python_version := "3.12"

# Helper to check for envsubst
_check-deps:
    @command -v envsubst >/dev/null || { echo "❌ Error: 'envsubst' is required. Install 'gettext'."; exit 1; }

# COMMANDS

default:
    @just --list

update_packages:
    @just lock
    @just requirements

lock:
    @echo "Locking dependencies from requirements.in..."
    uv pip compile requirements.in -o requirements.lock


requirements:
    @echo "Syncing dependencies..."
    uv pip sync requirements.lock
    @echo "Installing project in editable mode..."
    uv pip install --no-deps -e .


create_environment: _check-deps
    @echo "Creating Conda Env: {{project_name}} (Python {{python_version}})"

    # Remove any existing temp file
    rm -f .environment.tmp.yml

    # 1. Remove existing environment (ignore error if it doesn't exist)
    @echo "Removing old environment if it exists..."
    if conda env list | grep -q "{{project_name}}"; then \
        echo "🗑️  Removing existing Conda environment '{{project_name}}'..."; \
        conda env remove -n "{{project_name}}" -y; \
    fi

    # 2. Render template with variables passed INLINE (Fixes the shell issue)
    ENV_NAME="{{project_name}}" PYTHON_VERSION="{{python_version}}" envsubst < environment.yml.tpl > .environment.tmp.yml

    # 3. Create environment
    conda env create -f .environment.tmp.yml

    # Cleanup
    rm .environment.tmp.yml

    @echo "Conda environment created."
    @echo "NEXT STEPS:"
    @echo "    1. conda activate {{project_name}}"
    @echo "    2. just requirements"

clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete
    rm .environment.tmp.yml

lint:
    ruff format --check
    ruff check

format:
    ruff check --fix
    ruff format

test:
    python -m pytest tests

submodules:
    git submodule update --init --recursive

models:
    python scripts/00_pull_models.py