#!/bin/bash

# 48hrs min walltime:
#SBATCH --time 00:30:00

# 1 node:
#SBATCH --nodes 1

# 1 slot for MPI ranks per node:
#SBATCH --ntasks-per-node 1

# 4 CPUs per MPI rank:
#SBATCH --cpus-per-task 2

# 4GB of RAM per CPU:
#SBATCH --mem-per-cpu=8000

# jobname:
#SBATCH -J infectio-param-array

# stdouts
#SBATCH --output=output/logs/out_%A_%a.log
#SBATCH --error=output/logs/err_%A_%a.log
#SBATCH --array=0-9999

# Set the SAVE_ROOT dynamically
SAVE_ROOT="__SAVE_ROOT__"

# Path to parameter list file
PARAM_LIST_PATH="__PARAM_LIST_PATH__"

# Hardcoded -c flag for the final simulation run (this is where the config is used)
CONFIG_PATH="__CONFIG_PATH__"

# Number of experiments per parameter set
NUM_EXPERIMENTS="__NUM_EXPERIMENTS__"

# Read the parameter combinations into an array
IFS=$'\n' read -d '' -r -a PARAMS < "$PARAM_LIST_PATH"

# Calculate parameter index and experiment index
OFFSET=0
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
VAL_INDEX=$((GLOBAL_TASK_ID / NUM_EXPERIMENTS))
EXP_INDEX=$((GLOBAL_TASK_ID % NUM_EXPERIMENTS))

# Get the parameter string
PARAM_STRING="${PARAMS[$VAL_INDEX]}"
RANDOM_SUFFIX=$((RANDOM % 10000))

# Create save name
SAVE_NAME="${PARAM_STRING}/${RANDOM_SUFFIX}"

# Extract the parameters from the string and generate flags
eval $(echo "$PARAM_STRING" | awk -F'-' '{
    for (i=1; i<=NF; i++) {
        split($i, kv, "=");
        printf("%s=%s;", kv[1], kv[2]);
    }
}')

module load git python/3.11
cd $SLURM_SUBMIT_DIR
source ./venv/bin/activate

# Run the Python script with the received arguments
python ./infectio/models/vacv/run.py \
    "-c" "$CONFIG_PATH" \
    "--save_name" "$SAVE_NAME" \
    "--save_root" "$SAVE_ROOT" \
    "$@"