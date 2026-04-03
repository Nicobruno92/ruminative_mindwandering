#!/bin/bash
#SBATCH --job-name=conn_band
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/connectivity_band_%A_%a.out
#SBATCH --error=logs/connectivity_band_%A_%a.err

# =============================================================================
# CONNECTIVITY ANALYSIS - PARALLEL BAND EXECUTION
# =============================================================================
# Runs connectivity analysis for one frequency band per array task.
# This allows processing all 4 bands (theta, alpha, beta, gamma) in parallel.
#
# Usage:
#   sbatch Statistics_connectivity/submit_band_array.sh
#   sbatch Statistics_connectivity/submit_band_array.sh --array=0-1  # Only theta, alpha
#
# Array indices:
#   0 = theta
#   1 = alpha
#   2 = beta
#   3 = gamma
# =============================================================================

set -e

# Thread restrictions for numpy/scipy (allow joblib parallelism)
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
LOG_DIR="logs"

# Create log directory
mkdir -p ${LOG_DIR}

# Map array index to band
BANDS=(theta alpha beta gamma)
BAND=${BANDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Connectivity Analysis - Band: ${BAND}"
echo "=========================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

# Check config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Run analysis for this band
srun python ${SCRIPT_DIR}/run_pipeline.py \
    --config ${CONFIG_FILE} \
    --band ${BAND}

echo "=========================================="
echo "Band ${BAND} completed at $(date)"
echo "=========================================="
