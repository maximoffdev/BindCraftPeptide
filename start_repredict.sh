#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:a30:1
#SBATCH --output=logs/boltz_%A_%a.log
#SBATCH --error=logs/boltz_%A_%a.log
#SBATCH --job-name="boltz_pred"
#SBATCH --array=1-5

################################################################################
# CONFIGURATION
################################################################################
# The base directory where design folders already exist
BASE_DESIGN_PATH="./designs_output"

# Mapping the Array Task ID to the existing subfolder
SUBFOLDER_NAME="task_${SLURM_ARRAY_TASK_ID}"
CURRENT_DESIGN_PATH="${BASE_DESIGN_PATH}/${SUBFOLDER_NAME}"

# Path to the new Boltz environment
ENV_PATH="/home/sc.uni-leipzig.de/user/.conda/envs/boltz"

################################################################################
# PREPARATION
################################################################################
# Verify directory exists before proceeding
if [ ! -d "$CURRENT_DESIGN_PATH" ]; then
    echo "Error: Directory $CURRENT_DESIGN_PATH does not exist. Skipping."
    exit 1
fi

# Load modules
module load Anaconda3

# Critical for HPC stability: ignore home directory .local packages
export PIP_NO_USER_INSTALL=1
export PYTHONNOUSERSITE=1

# Initialize Conda and activate Boltz environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH"

# Prioritize environment binaries
export PATH="$ENV_PATH/bin:$PATH"

################################################################################
# EXECUTION
################################################################################
echo "Starting Boltz Repredict Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Target Directory: ${CURRENT_DESIGN_PATH}"
echo "Using Python: $(which python)"

# Execute boltz_repredict.py
# We use srun to ensure proper GPU allocation and the specific env python
srun "$ENV_PATH/bin/python" -u ./boltz_repredict.py \
    --settings './example/paul/settings_scaffold.json' \
    --design_path "$CURRENT_DESIGN_PATH"

echo "Boltz Task ${SLURM_ARRAY_TASK_ID} completed."