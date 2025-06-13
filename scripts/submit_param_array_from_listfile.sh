#!/bin/bash

# User provides the parameter list file path
PARAM_LIST_PATH="/home/mokari27/workspace/infectio-mesa/scripts/parameter_combinations.txt"

# Hardcoded SAVE_ROOT (where results will go)
SAVE_ROOT="/home/mokari27/workspace/infectio-mesa/output/dVGFdF11/wide_plausible_100x/"

# Hardcoded -c flag path for the final simulation run
CONFIG_PATH="/home/mokari27/workspace/infectio-mesa/examples/vacv-epithelial/dVGFdF11_default_config.ini"

# Number of experiments per parameter set
NUM_EXPERIMENTS=100

# Validate that the param list exists
if [ ! -f "$PARAM_LIST_PATH" ]; then
    echo "Parameter list file not found at: $PARAM_LIST_PATH"
    exit 1
fi

# Extract name and parent for renaming the parameter list
PARENT_DIR=$(dirname "$SAVE_ROOT")
BASENAME=$(basename "$SAVE_ROOT")
DEST_TXT_FILE="${PARENT_DIR}/${BASENAME}.txt"

# Copy the parameter list file to renamed destination
mkdir -p "$SAVE_ROOT"
cp "$PARAM_LIST_PATH" "$DEST_TXT_FILE"

# Count total combinations
NUM_PARAMS=$(wc -l < "$PARAM_LIST_PATH")
TOTAL_TASKS=$((NUM_PARAMS * NUM_EXPERIMENTS))

# Replace placeholders in SLURM template and submit
sed "s|#SBATCH --array=0-9999|#SBATCH --array=0-$((TOTAL_TASKS - 1))|; s|__SAVE_ROOT__|$SAVE_ROOT|; s|__CONFIG_PATH__|$CONFIG_PATH|; s|__PARAM_LIST_PATH__|$PARAM_LIST_PATH|; s|__NUM_EXPERIMENTS__|$NUM_EXPERIMENTS|" scripts/run_array_template.sh > scripts/tmp_run.sbatch

# Submit with the final simulation -c flag and hardcoded configuration path
sbatch scripts/tmp_run.sbatch