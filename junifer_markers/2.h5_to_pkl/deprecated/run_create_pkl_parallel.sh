#!/bin/bash
#SBATCH --job-name=pkl_create
#SBATCH --output=logs/pkl_parallel_%A_%a.out
#SBATCH --error=logs/pkl_parallel_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-60  # 60 parallel jobs (334 files / 60 = ~6 files per job)

# ==============================================================================
# Parallel PKL Creation Script for SLURM - OPTIMIZED for Speed
# ==============================================================================
# This script divides the 334 element-desc combinations into 60 parallel jobs
# Each job processes ~6 files independently
#
# Expected runtime: ~3 seconds × 6 files = ~18 seconds per job
# Total pipeline time: ~18-30 seconds (including SLURM overhead)
#
# Usage:
#   sbatch run_create_pkl_parallel.sh
#
# Monitor:
#   squeue -u $USER | grep pkl_create
#   tail -f logs/pkl_parallel_*.out
# ==============================================================================

# Configuration
SCRIPT_DIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/junifer_markers/2.h5_to_pkl"
ELEMENTS_FILE="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/junifer_markers/1.markers_h5_creation/elements"
N_JOBS=60  # Total number of parallel jobs (must match --array above)

# Activate environment
echo "=========================================="
echo "SLURM Array Job: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID} / ${N_JOBS}"
echo "Node: ${HOSTNAME}"
echo "=========================================="

# Load proxy module (if needed)
module load proxy || echo "Warning: Could not load proxy module"

# Activate conda environment
source ~/.bashrc
conda activate junifer || {
    echo "ERROR: Could not activate junifer conda environment"
    exit 1
}

echo "Using Python: $(which python3)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Change to script directory
cd "${SCRIPT_DIR}" || exit 1

# Run the parallel worker for this task
echo "Starting PKL creation..."
python3 batch_create_pkl_parallel.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --n-jobs ${N_JOBS} \
    --elements-file "${ELEMENTS_FILE}" \
    --desc both \
    --force

EXIT_CODE=$?

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed with exit code: ${EXIT_CODE}"
echo "=========================================="

exit ${EXIT_CODE}
