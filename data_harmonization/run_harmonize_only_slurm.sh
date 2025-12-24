#!/bin/bash
## SLURM directives
#SBATCH --job-name=CyberSART_Harmonize
#SBATCH --output=logs/harmonize_%A_%a.out
#SBATCH --error=logs/harmonize_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/data_harmonization

# =============================================================================
# CYBERSART DATA HARMONIZATION - SLURM ARRAY JOB
# =============================================================================
# This script runs the data harmonization pipeline on the cluster.
# It processes BrainVision files and converts them to BIDS-compliant format.
#
# Array size = subjects * tasks
# 42 subjects (02..43) * 4 tasks = 168 jobs -> --array=0-167
# =============================================================================

#SBATCH --array=0-167

# -----------------------------------------------------------------------------
# CONFIGURATION - Modify these arrays to match your data
# -----------------------------------------------------------------------------
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

# Compute array dimensions
n_subjects=${#subjects[@]}
n_tasks=${#tasks[@]}

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
mkdir -p logs

module load proxy

echo "=========================================="
echo "CYBERSART DATA HARMONIZATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

# Activate Python environment
echo "Activating Python environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate eeg
else
    module load anaconda3 || module load miniconda3 || module load conda || true
    conda activate eeg || true
fi

# Constrain threading to reduce memory pressure
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MNE_MEMMAP_MIN_SIZE=1M
export MNE_LOGGING_LEVEL=WARNING
export KMP_AFFINITY=disabled
export ORT_DISABLE_THREAD_AFFINITY=1

# -----------------------------------------------------------------------------
# COMPUTE INDICES FROM ARRAY TASK ID
# -----------------------------------------------------------------------------
# Layout: for each subject, iterate tasks
# index = subject_idx * n_tasks + task_idx

subject_idx=$((SLURM_ARRAY_TASK_ID / n_tasks))
task_idx=$((SLURM_ARRAY_TASK_ID % n_tasks))

# Validate indices
if (( subject_idx < 0 || subject_idx >= n_subjects )); then
    echo "Subject index $subject_idx out of range ($n_subjects). Exiting."
    exit 0
fi
if (( task_idx < 0 || task_idx >= n_tasks )); then
    echo "Task index $task_idx out of range ($n_tasks). Exiting."
    exit 0
fi

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "Processing: sub-$subject task-$task"

# -----------------------------------------------------------------------------
# RUN HARMONIZATION
# -----------------------------------------------------------------------------
SCRIPT_DIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/data_harmonization"
HARM_SCRIPT="$SCRIPT_DIR/data_harmonization.py"
CONFIG_PATH="$SCRIPT_DIR/config.yaml"

echo "[HARMONIZE] Starting harmonization..."
echo "  Script: $HARM_SCRIPT"
echo "  Config: $CONFIG_PATH"

srun --cpu-bind=none --ntasks=1 python "$HARM_SCRIPT" \
    --config "$CONFIG_PATH" \
    --subject "$subject" \
    --task "$task"

exit_code=$?

echo "=========================================="
echo "Harmonization completed with exit code: $exit_code"
echo "Finished: $(date)"
echo "=========================================="

exit $exit_code


