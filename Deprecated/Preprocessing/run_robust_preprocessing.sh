#!/bin/bash
#SBATCH --job-name=RobustEEGPreprocessing
#SBATCH --output=logs/robust_preprocessing_%A_%a.out
#SBATCH --error=logs/robust_preprocessing_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=48G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-167  # Adjust array size (subjects * tasks)

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load proxy

# Try to activate the ML environment
echo "Attempting to activate Python environment..."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "Using miniconda3 path"
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Using anaconda3 path"
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif command -v conda >/dev/null 2>&1; then
    echo "Using conda from PATH"
    eval "$(conda shell.bash hook)"
    conda activate eeg
else
    echo "Using module method"
    module load anaconda3 || module load miniconda3 || module load conda || echo "No conda module found"
    conda activate eeg || echo "Failed to activate eeg environment"
fi

# Verify that required packages are available
echo "Checking required Python packages..."
python -c "import mne, mne_icalabel, autoreject, pyprep, yaml" || {
    echo "ERROR: Missing required packages. Please install:"
    echo "conda install -c conda-forge mne mne-icalabel autoreject pyprep"
    echo "pip install pyyaml"
    exit 1
}

# Check optional packages
echo "Checking optional packages..."
python -c "import structlog" && echo "✅ structlog available" || echo "⚠️  structlog not available - using fallback logging"

# Create configs directory if it doesn't exist
mkdir -p Preprocessing/configs

# Check if config file exists
if [ ! -f "Preprocessing/configs/config_preprocessing.yaml" ]; then
    echo "ERROR: Configuration file not found!"
    echo "Please ensure the configuration file is available before running."
    exit 1
fi

# Define subjects and tasks
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

# Calculate subject-task pair from SLURM_ARRAY_TASK_ID
subject_idx=$((SLURM_ARRAY_TASK_ID / 4))
task_idx=$((SLURM_ARRAY_TASK_ID % 4))

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "Processing subject $subject for task $task using Robust TWO-COPY ICA Pipeline"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set memory limit for Python - optimize for memory not CPU
export PYTHONHASHSEED=0
export MKL_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16

# 🔧 OPTIMIZATION: Fix ONNX runtime threading issues for ICLabel
export ORT_NUM_THREADS=1
export ONNXRUNTIME_NUM_INTER_OP_THREADS=1
export ONNXRUNTIME_NUM_INTRA_OP_THREADS=1

# 🔧 OPTIMIZATION: Additional threading controls for ICLabel stability
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

# Memory optimization
export MALLOC_TRIM_THRESHOLD_=65536
export PYTHONMALLOC=malloc

# Monitor memory usage
echo "Starting preprocessing at $(date)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $7}')"

# Run the robust preprocessing script with ultra-fast optimization
python Preprocessing/robust_preprocessing_two_copy_ica.py $subject $task --ultra-fast

# Check exit status
if [ $? -eq 0 ]; then
    echo "Preprocessing completed successfully at $(date)"
else
    echo "Preprocessing failed at $(date)" >&2
    exit 1
fi 