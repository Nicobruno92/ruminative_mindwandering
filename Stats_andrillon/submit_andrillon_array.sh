#!/bin/bash
#SBATCH --job-name=andrillon_cluster
#SBATCH --output=logs/andrillon_%A_%a.out
#SBATCH --error=logs/andrillon_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Andrillon 2020 Cluster-Permutation Pipeline - SLURM Array Job
# Each array task processes one marker in parallel
#
# IMPORTANT: Do NOT submit this script directly!
# Use submit_parallel_andrillon.sh instead, which automatically
# determines the correct number of markers and sets the array range.
#
# Manual usage (if needed):
#   1. Determine number of markers: python get_marker_list.py --format count
#   2. Submit: sbatch --array=0-N submit_andrillon_array.sh (where N = count - 1)

echo "=========================================="
echo "Andrillon 2020 Pipeline - Array Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
# Adjust environment name as needed
echo "Activating conda environment..."
module load proxy  # Load proxy if on cluster

# Initialize conda for bash (required for SLURM)
eval "$(conda shell.bash hook)"
conda activate eeg  # or your appropriate environment

# Set working directory
WORK_DIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats_andrillon"
cd $WORK_DIR

echo "Working directory: $(pwd)"

# Configuration
CONFIG_FILE="config_andrillon.yaml"
MARKER_INDEX=${SLURM_ARRAY_TASK_ID}

echo "=========================================="
echo "Processing marker index: $MARKER_INDEX"
echo "=========================================="

# Run pipeline with marker index (script will handle marker selection)
# -u flag: unbuffered output (prints appear immediately)
python -u run_andrillon_pipeline.py \
    --config $CONFIG_FILE \
    --marker-index $MARKER_INDEX

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "SUCCESS: Marker $MARKER completed"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Marker $MARKER failed with exit code $EXIT_CODE"
    echo "=========================================="
fi

exit $EXIT_CODE
