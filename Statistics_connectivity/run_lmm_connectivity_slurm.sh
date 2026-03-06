#!/bin/bash
#SBATCH --job-name=connectivity_lmm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs_connectivity/lmm_%j.out
#SBATCH --error=logs_connectivity/lmm_%j.err

# Connectivity Statistics Pipeline - SLURM script
# Usage: sbatch Statistics_connectivity/run_lmm_pipeline_slurm.sh

# Set threading environment variables early
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

# Load required modules
module load proxy

# Activate conda environment
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
export MNE_LOGGING_LEVEL=WARNING

# Set working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Configuration
CONFIG_FILE="Statistics_connectivity/config.yaml"
SCRIPT_DIR="Statistics_connectivity"
LOG_DIR="logs_connectivity"

# Create directories if they don't exist
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Connectivity LMM Pipeline - Cluster Run"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Config file: ${CONFIG_FILE}"
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Run the pipeline
srun python ${SCRIPT_DIR}/run_pipeline.py --config ${CONFIG_FILE}

echo "=========================================="
echo "Job completed at $(date)"
echo "=========================================="
