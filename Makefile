PROJECT_NAME := diffusion-deep-dream-research-test
.PHONY: help env remove-env submodules install clean

# Default target: lists commands
help:
	@echo "Available commands:"
	@echo "  make env		  - Set up conda environment"
	@echo "  make remove-env  - Remove the conda environment"
	@echo "  make install     - Install all dependencies"
	@echo "  make clean       - Remove Python cache files"
	@echo "  make submodules  - Initialize/update git submodules"
	@echo "  make test        - Run the test suite"

env:
	conda create -n $(PROJECT_NAME)-env python=3.12 -y -vv
	@echo "Conda environment '$(PROJECT_NAME)-env' created."
	@echo "To activate the environment, run: conda activate $(PROJECT_NAME)-env"


remove-env:
	conda env remove -n $(PROJECT_NAME)-env -vv || true

submodules:
	git submodule update --init --recursive

install: submodules
	@echo "Installing packages"
	pip install -e .

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

test:
	pytest tests/