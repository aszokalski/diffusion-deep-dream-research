#!/bin/bash

echo "Setting up CUDA"
module load CUDA/12.8.0

echo "Setting up miniconda"
module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"

echo "Creating Conda directories..."
mkdir -p "$SCRATCH/.conda/pkgs"
mkdir -p "$SCRATCH/.conda/envs"


echo "Configuring Conda paths..."
conda config --add pkgs_dirs "$SCRATCH/.conda/pkgs"
conda config --add envs_dirs "$SCRATCH/.conda/envs"

echo "Done."