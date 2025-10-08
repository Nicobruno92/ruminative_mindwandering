#!/bin/bash

# SLURM script to batch process H5+FIF outputs to PKL files
# This runs after the main pipeline has completed
#
# Usage:
#   sbatch batch_create_pkl.sh
# Or with custom settings:
#   JOBNAME=CYBERSART_features sbatch batch_create_pkl.sh

#SBATCH --job-name=create_pkl
#SBATCH --output=logs/create_pkl_%j.out
#SBATCH --error=logs/create_pkl_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -euo pipefail

# Parameters (overridable)
JOBNAME=${JOBNAME:-CYBERSART_features}
WORKDIR=${WORKDIR:-/network/iss/home/nicolas.bruno/Junifer}
CONDA_ENV=${CONDA_ENV:-junifer}
ELEMENTS_FILE=${ELEMENTS_FILE:-${WORKDIR}/junifer_jobs/${JOBNAME}/elements}
PIPELINE_DIR=${PIPELINE_DIR:-${WORKDIR}/junifer_markers/2.h5_to_pkl}

cd "$WORKDIR"

echo "[INFO] Starting PKL batch creation at $(date)"
echo "[INFO] Job name: $JOBNAME"
echo "[INFO] Working directory: $WORKDIR"
echo "[INFO] Elements file: $ELEMENTS_FILE"
echo "[INFO] Pipeline directory: $PIPELINE_DIR"

# Load conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

# Activate environment
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV"
fi

# Set Python path
export PYTHONPATH="${WORKDIR}:${PIPELINE_DIR}:${PYTHONPATH:-}"

# Navigate to pipeline directory
cd "${PIPELINE_DIR}"

# Run the batch processing script
echo "[INFO] Running batch PKL creation..."
python batch_create_pkl_from_pipeline.py \
  --elements-file "$ELEMENTS_FILE" \
  --desc both

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "[INFO] PKL batch creation completed successfully at $(date)"
else
  echo "[ERROR] PKL batch creation failed with exit code $EXIT_CODE at $(date)"
fi

exit $EXIT_CODE
