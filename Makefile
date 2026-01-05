PROJECT_NAME := diffusion-deep-dream-research
PLG_CONDA_BASE := $(PLG_GROUPS_STORAGE)/plggailpwmm/aszokalski/.conda
.PHONY: help submodules setup reset clean update configure-plg-conda

# Default target: lists commands
help:
	@echo "Available commands:"
	@echo "  make setup       - Update submodules and Conda environment"
	@echo "  make update      - Update Python packages"
	@echo "  make reset       - Delete and recreate the environment"
	@echo "  make clean       - Remove Python cache files"
	@echo "  make submodules  - Initialize/update git submodules"

submodules:
	git submodule update --init --recursive

# Installs or updates dependencies from environment.yml
# 'submodules' is listed as a dependency so it runs first
setup: submodules
	@echo "Updating Conda Environment: $(PROJECT_NAME)..."
	conda env update -n $(PROJECT_NAME) -f environment.yml --prune

update:
	@echo "updating packages"
	pip install -e .

# Deletes the environment and re-runs setup
reset:
	@echo "Removing environment..."
	conda env remove -n $(PROJECT_NAME) -y || true
	$(MAKE) setup

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
