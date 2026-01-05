#!/bin/bash

# 1. Define variables (pass Project Name as first argument, or default)
PROJECT_NAME="${1:-diffusion-deep-dream-research}"
BASE_PATH="${PLG_GROUPS_STORAGE}/plggailpwmm/aszokalski/.conda"

# 2. Check for required env var
if [ -z "$PLG_GROUPS_STORAGE" ]; then
    echo "❌ Error: PLG_GROUPS_STORAGE is not set."
    return 1 2>/dev/null || exit 1
fi

# 3. Create directories
echo "Creating Conda directories..."
mkdir -p "$BASE_PATH/pkgs"
mkdir -p "$BASE_PATH/envs"

# 4. Configure Conda paths
echo "Configuring Conda paths..."
conda config --add pkgs_dirs "$BASE_PATH/pkgs"
conda config --add envs_dirs "$BASE_PATH/envs"

# 5. ACTIVATE
# We need to initialize conda for this shell context first
eval "$(conda shell.bash hook)"

echo "Activating environment: $PROJECT_NAME"
conda activate "$PROJECT_NAME"

if [ $? -eq 0 ]; then
    echo "✅ Success! Environment '$PROJECT_NAME' is active."
else
    echo "⚠️  Environment '$PROJECT_NAME' not found. Creating it might be the next step."
fi