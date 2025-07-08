#!/bin/bash

#SBATCH --time 00:30:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu=8000
#SBATCH -J infectio-param-array
#SBATCH --output=/home/mokari27/workspace/infectio-mesa/slurm_logs/%A_%a.out
#SBATCH --error=/home/mokari27/workspace/infectio-mesa/slurm_logs/%A_%a.err
#SBATCH --array=0-9999

# --- Dynamically set variables from the creator script ---
SAVE_ROOT="__SAVE_ROOT__"
PARAM_LIST_PATH="__PARAM_LIST_PATH__"
NUM_EXPERIMENTS="__NUM_EXPERIMENTS__"
# This is now an optional base config file
BASE_CONFIG_PATH="__CONFIG_PATH__"


# --- Create a temporary file for this job's unique parameters ---
# mktemp creates a securely named temporary file
TMP_CONFIG_PATH=$(mktemp)

# This 'trap' command ensures the temporary file is deleted when the script exits
trap 'rm -f "$TMP_CONFIG_PATH"' EXIT

# --- Read parameters from file ---
IFS=$'\n' read -d '' -r -a PARAMS < "$PARAM_LIST_PATH"

# --- Calculate task-specific indices ---
OFFSET=0
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + OFFSET))
VAL_INDEX=$((GLOBAL_TASK_ID / NUM_EXPERIMENTS))
EXP_INDEX=$((GLOBAL_TASK_ID % NUM_EXPERIMENTS))

# --- Get this job's parameter string ---
PARAM_STRING="${PARAMS[$VAL_INDEX]}"

# --- Write the parameters into the temporary config file ---
# First, copy the base config so we have all default values
cat "$BASE_CONFIG_PATH" > "$TMP_CONFIG_PATH"
# Then, append the new, overriding parameters
echo "" >> "$TMP_CONFIG_PATH" # Add a newline for safety
echo "# Parameters for this specific job" >> "$TMP_CONFIG_PATH"
echo "$PARAM_STRING" | tr -d '\r' | tr '-' '\n' | while IFS='=' read -r key value; do
    if [ -n "$key" ]; then
        # Write 'key = value' to the temp config file
        echo "$key = $value" >> "$TMP_CONFIG_PATH"
    fi
done

# --- Set up the environment ---
module load git python/3.11
cd $SLURM_SUBMIT_DIR
source ./venv/bin/activate

# --- Create a unique save name ---
RANDOM_SUFFIX=$((RANDOM % 10000))
SAVE_NAME="${PARAM_STRING}/${RANDOM_SUFFIX}"

# --- Run the Python script with a SINGLE -c flag ---
# All job-specific parameters are now neatly inside the temporary config file.
python ./infectio/models/vacv/run.py \
    -c "$TMP_CONFIG_PATH" \
    --save_name "$SAVE_NAME" \
    --save_root "$SAVE_ROOT"