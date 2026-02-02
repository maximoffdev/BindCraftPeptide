#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=paula
#SBATCH --gres=gpu:a30:1
#SBATCH --output=logs/design_%A_%a.log
#SBATCH --error=logs/design_%A_%a.log
#SBATCH --job-name="pep_design"
#SBATCH --array=1-5

################################################################################
# CONFIGURATION
################################################################################
# 1. Specify the number of subfolders in the --array line above (e.g., 1-10)
# 2. Specify the base directory for designs:
BASE_DESIGN_PATH="./designs_output"

# 3. Define the subfolder name based on the task ID
# This creates names like: ./designs_output/task_1, ./designs_output/task_2, etc.
SUBFOLDER_NAME="task_${SLURM_ARRAY_TASK_ID}"
CURRENT_DESIGN_PATH="${BASE_DESIGN_PATH}/${SUBFOLDER_NAME}"

################################################################################
# PREPARATION
################################################################################
# Create the specific subfolder for this task
mkdir -p "$CURRENT_DESIGN_PATH"

# Load modules and environment
module load Anaconda3
export PIP_NO_USER_INSTALL=1
export PYTHONNOUSERSITE=1

# Use 'source' instead of 'conda init' for better stability inside scripts
eval "$(conda shell.bash hook)"
conda activate BindCraft

################################################################################
# EXECUTION
################################################################################
echo "Starting Job Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output Directory: ${CURRENT_DESIGN_PATH}"

# Execute python with the specific subfolder path
~/.conda/envs/BindCraft/bin/python -u ./design.py \
    --settings './example/paul/settings_scaffold.json' \
    --advanced './settings_advanced/test_settings_peptide_betasheet_4stage_multimer.json' \
    --filters './settings_filters/no_filters.json' \
    --design_path "$CURRENT_DESIGN_PATH"

echo "Task ${SLURM_ARRAY_TASK_ID} completed."