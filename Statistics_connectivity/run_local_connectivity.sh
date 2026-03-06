#!/bin/bash
# Local execution script for Connectivity Statistics Pipeline

set -e  # Exit on error

# 1. Environment Setup (Optimization settings used across pipeline)
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

# Configuration
CONFIG_FILE="Statistics_connectivity/config.yaml"
SCRIPT_DIR="Statistics_connectivity"
LOG_DIR="logs_connectivity"

mkdir -p ${LOG_DIR}

echo "=========================================="
echo "LMM Connectivity Pipeline - Local Execution"
echo "=========================================="

# Activate Conda
echo "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
fi

# Check config
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

echo "Starting pipeline..."
python ${SCRIPT_DIR}/run_pipeline.py --config ${CONFIG_FILE}

echo "=========================================="
echo "Local Pipeline Completed Successfully!"
echo "=========================================="
