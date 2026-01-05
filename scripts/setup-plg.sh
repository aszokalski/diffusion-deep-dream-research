#!/bin/bash

PROJECT_NAME="${1:-diffusion-deep-dream-research}"
BASE_PATH="${PLG_GROUPS_STORAGE}/plggailpwmm/aszokalski/.conda"

echo "Setting up CUDA"
module load CUDA/12.8.0

echo "Setting up miniconda"
module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"

echo "Creating Conda directories..."
mkdir -p "$BASE_PATH/pkgs"
mkdir -p "$BASE_PATH/envs"


echo "Configuring Conda paths..."
conda config --add pkgs_dirs "$BASE_PATH/pkgs"
conda config --add envs_dirs "$BASE_PATH/envs"


conda env update -f "environment.yml" -n "${PROJECT_NAME}" --prune -vv --solver=libmamba
conda activate "$PROJECT_NAME"
echo "Done."