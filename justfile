# GLOBALS
project_name := "diffusion-deep-dream-research"
python_version := "3.12"
python_interpreter := "python"

# COMMANDS

# List available commands
default:
    @just --list

# Install Python dependencies
requirements:
    uv sync

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Lint using ruff (use `just format` to do formatting)
lint:
    ruff format --check
    ruff check

# Format source code with ruff
format:
    ruff check --fix
    ruff format

# Run tests
test:
    python -m pytest tests

# Download git submodules
submodules:
    git submodule update --init --recursive

models:
    python diffusion_deep_dream_research/models.py

# Set up Python interpreter environment
create_environment:
    uv venv --python {{python_version}}
    @echo ">>> New uv virtual environment created. Activate with:"
    @echo ">>> Windows: .\\.venv\\Scripts\\activate"
    @echo ">>> Unix/macOS: source ./.venv/bin/activate"

