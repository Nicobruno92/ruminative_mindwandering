#!/bin/bash
#SBATCH --job-name=conn_aggregate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/connectivity_aggregate_%j.out
#SBATCH --error=logs/connectivity_aggregate_%j.err

# =============================================================================
# CONNECTIVITY ANALYSIS - AGGREGATE RESULTS
# =============================================================================
# Combines results from parallel band jobs and generates multi-band figures.
# This job runs after all band array jobs complete.
#
# Usage:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> Statistics_connectivity/aggregate_results.sh
# =============================================================================

set -e

# Thread restrictions
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
export MKL_DYNAMIC=FALSE

# Load modules
module load proxy

# Activate conda environment
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

# Plotting backend
export MPLBACKEND=Agg
export MNE_LOGGING_LEVEL=WARNING

# Working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Configuration
CONFIG_FILE="Statistics_connectivity/config.yaml"
SCRIPT_DIR="Statistics_connectivity"

echo "=========================================="
echo "Connectivity Analysis - Aggregation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

# Check config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Run aggregation script
srun python ${SCRIPT_DIR}/aggregate_and_plot.py --config ${CONFIG_FILE}

echo "=========================================="
echo "Aggregation completed at $(date)"
echo "=========================================="
