#!/bin/bash
#SBATCH --job-name=lmm_marker_array
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lmm_marker_%A_%a.out
#SBATCH --error=logs/lmm_marker_%A_%a.err

# LMM-Based Spatial Cluster Permutation Pipeline - SLURM Array Job
# This script parallelizes marker processing across SLURM array tasks
# Each array task processes one marker independently
#
# IMPORTANT: CPU allocation
# - cpus-per-task should match the n_jobs setting in config.yaml
# - If n_jobs=-1 (use all cores), set cpus-per-task to desired number
# - If n_jobs=1 (sequential), set cpus-per-task=1
# - Current setting: cpus-per-task=4 (matches n_jobs=-1 with 4 cores)
#
# Usage:
#   1. Configure markers in Statistics/config.yaml (set project.markers)
#   2. Set clustering.n_jobs in config.yaml (-1 for all cores, or specific number)
#   3. Adjust --cpus-per-task above to match your n_jobs setting
#   4. Run: sbatch Statistics/submit_marker_array.sh
#   5. The script will auto-detect the number of markers and submit array jobs

# Set threading environment variables to prevent nested parallelism
# We use joblib for parallel processing at the permutation level,
# so we disable threading in numerical libraries to avoid conflicts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled
export ORT_DISABLE_THREAD_AFFINITY=1
export ONNX_DISABLE_THREADPOOL_AFFINITY=1
export ORT_NUM_THREADS=1
export ONNXRUNTIME_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
export MKL_DYNAMIC=FALSE
export TORCH_NUM_THREADS=1

# Note: These settings prevent nested parallelism (library-level threading)
# while allowing joblib to parallelize at the process level (permutations)

# Load required modules
module load proxy

# Activate conda environment with proper error handling
echo "Attempting to activate Python environment..."
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

# Additional MNE and plotting configuration
export MPLBACKEND=Agg
export MNE_MEMMAP_MIN_SIZE=1M
export MNE_LOGGING_LEVEL=WARNING

# Set working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Configuration
CONFIG_FILE="Statistics/config.yaml"
SCRIPT_DIR="Statistics"
LOG_DIR="logs"
RESULTS_DIR="results/lmm_cluster"

# Create directories if they don't exist
mkdir -p ${LOG_DIR}
mkdir -p ${RESULTS_DIR}

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "LMM Marker Array Job"
echo "=========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Config file: ${CONFIG_FILE}"
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Check if script directory exists
if [ ! -d "${SCRIPT_DIR}" ]; then
    echo "ERROR: Script directory ${SCRIPT_DIR} not found!"
    exit 1
fi

# The marker index is provided by SLURM_ARRAY_TASK_ID
MARKER_INDEX=${SLURM_ARRAY_TASK_ID}

echo "Processing marker index: ${MARKER_INDEX}"

# Build CLI arguments
# PREDICTOR_OF_INTEREST may be injected via --export by submit_parallel_markers.sh
CLI_ARGS="--config ${CONFIG_FILE} --marker-index ${MARKER_INDEX}"
if [ -n "${PREDICTOR_OF_INTEREST:-}" ]; then
    CLI_ARGS="${CLI_ARGS} --predictor-of-interest ${PREDICTOR_OF_INTEREST}"
    echo "Predictor of interest: ${PREDICTOR_OF_INTEREST}"
fi

# Run the pipeline for this specific marker
echo "Starting pipeline execution..."
srun --cpu-bind=none --ntasks=1 python ${SCRIPT_DIR}/run_pipeline.py ${CLI_ARGS}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Marker ${MARKER_INDEX} completed successfully at $(date)"
else
    echo "✗ Marker ${MARKER_INDEX} failed with exit code ${EXIT_CODE} at $(date)"
    exit ${EXIT_CODE}
fi

echo "=========================================="
echo "Array task completed at $(date)"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="
