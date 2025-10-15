#!/bin/bash
#SBATCH --job-name=lmm_cluster_pipeline
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/lmm_pipeline_%j.out
#SBATCH --error=logs/lmm_pipeline_%j.err

# LMM-Based Spatial Cluster Permutation Pipeline
# SLURM script for running the pipeline on the cluster
# Usage: sbatch run_lmm_pipeline.sh [--all-markers] [--markers "marker1" "marker2"]

# Set threading environment variables VERY early to prevent any library conflicts
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

# Default arguments
ANALYSIS_MODE="single"
MARKERS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all-markers)
            ANALYSIS_MODE="all"
            shift
            ;;
        --markers)
            ANALYSIS_MODE="specific"
            shift
            MARKERS="$1"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Set log file names
LOG_FILE="${LOG_DIR}/lmm_pipeline_${TIMESTAMP}.log"
ERROR_FILE="${LOG_DIR}/lmm_pipeline_${TIMESTAMP}.err"

echo "=========================================="
echo "LMM-Based Spatial Cluster Permutation Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Config file: ${CONFIG_FILE}"
echo "Analysis mode: ${ANALYSIS_MODE}"
echo "Log file: ${LOG_FILE}"
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

# Function to run the pipeline
run_pipeline() {
    local mode=$1
    local markers=$2
    
    echo "Starting pipeline execution..."
    echo "Mode: ${mode}"
    
    case ${mode} in
        "single")
            echo "Running single marker analysis..."
            srun --cpu-bind=none --ntasks=1 python ${SCRIPT_DIR}/run_pipeline.py --config ${CONFIG_FILE}
            ;;
        "all")
            echo "Running all markers analysis..."
            srun --cpu-bind=none --ntasks=1 python ${SCRIPT_DIR}/run_pipeline.py --config ${CONFIG_FILE} --all-markers
            ;;
        "specific")
            echo "Running specific markers: ${markers}"
            srun --cpu-bind=none --ntasks=1 python ${SCRIPT_DIR}/run_pipeline.py --config ${CONFIG_FILE} --markers ${markers}
            ;;
        *)
            echo "ERROR: Unknown analysis mode: ${mode}"
            exit 1
            ;;
    esac
}

# Run the pipeline with error handling
echo "Starting pipeline at $(date)" | tee ${LOG_FILE}

if run_pipeline ${ANALYSIS_MODE} "${MARKERS}" 2>&1 | tee -a ${LOG_FILE}; then
    echo "Pipeline completed successfully at $(date)" | tee -a ${LOG_FILE}
    
    # Check if results were created
    if [ -d "${RESULTS_DIR}" ] && [ "$(ls -A ${RESULTS_DIR})" ]; then
        echo "Results found in ${RESULTS_DIR}:" | tee -a ${LOG_FILE}
        ls -la ${RESULTS_DIR} | tee -a ${LOG_FILE}
        
        # Count result files
        PKL_COUNT=$(find ${RESULTS_DIR} -name "*.pkl" | wc -l)
        CSV_COUNT=$(find ${RESULTS_DIR} -name "*.csv" | wc -l)
        PNG_COUNT=$(find ${RESULTS_DIR} -name "*.png" | wc -l)
        
        echo "Summary of generated files:" | tee -a ${LOG_FILE}
        echo "  - Pickle files: ${PKL_COUNT}" | tee -a ${LOG_FILE}
        echo "  - CSV files: ${CSV_COUNT}" | tee -a ${LOG_FILE}
        echo "  - PNG files: ${PNG_COUNT}" | tee -a ${LOG_FILE}
        
        # Show pipeline summary if it exists
        if [ -f "${RESULTS_DIR}/pipeline_summary.csv" ]; then
            echo "Pipeline summary:" | tee -a ${LOG_FILE}
            head -10 ${RESULTS_DIR}/pipeline_summary.csv | tee -a ${LOG_FILE}
        fi
        
    else
        echo "WARNING: No results found in ${RESULTS_DIR}" | tee -a ${LOG_FILE}
    fi
    
else
    echo "Pipeline failed at $(date)" | tee -a ${LOG_FILE}
    echo "Check the error log for details." | tee -a ${LOG_FILE}
    exit 1
fi

echo "=========================================="
echo "Job completed at $(date)"
echo "Log file: ${LOG_FILE}"
echo "Results directory: ${RESULTS_DIR}"
echo "=========================================="
