#!/bin/bash
#SBATCH --job-name=MNE-BIDS-Pipeline
#SBATCH --output=logs/mne_bids_pipeline_%A_%a.out
#SBATCH --error=logs/mne_bids_pipeline_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=32G
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
python -c "import mne, mne_bids_pipeline, autoreject, yaml" || {
    echo "ERROR: Missing required packages. Please install:"
    echo "pip install mne-bids-pipeline"
    echo "conda install -c conda-forge mne autoreject"
    exit 1
}

# Check MNE-BIDS-Pipeline version
echo "MNE-BIDS-Pipeline version:"
python -c "import mne_bids_pipeline; print(mne_bids_pipeline.__version__)"

# Define subjects and tasks
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

# Calculate subject-task pair from SLURM_ARRAY_TASK_ID
subject_idx=$((SLURM_ARRAY_TASK_ID / 4))
task_idx=$((SLURM_ARRAY_TASK_ID % 4))

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "Processing subject $subject for task $task using MNE-BIDS-Pipeline"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set memory limit for Python
export PYTHONHASHSEED=0
export MKL_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export OMP_NUM_THREADS=16

echo "Running MNE-BIDS-Pipeline at $(date)"

# Run the MNE-BIDS-Pipeline using the wrapper script
python Preprocessing/run_mne_bids_wrapper.py $subject $task

# Check exit status
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "MNE-BIDS-Pipeline completed successfully at $(date)"
    
    # Update status CSV
    python -c "
import pandas as pd
import os

# Create status entry
status_df = pd.DataFrame([{
    'subject': '$subject', 
    'task': '$task', 
    'data': 'eeg', 
    'status': 'preprocessed_mne_bids',
    'pipeline': 'mne-bids-pipeline'
}])

# Update status file
derivatives_folder = '/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids'
os.makedirs(derivatives_folder, exist_ok=True)
status_path = os.path.join(derivatives_folder, 'preprocessing_status.csv')

if os.path.exists(status_path):
    existing_df = pd.read_csv(status_path)
    status_df = pd.concat([existing_df, status_df], ignore_index=True)
    
status_df.to_csv(status_path, index=False)
print(f'Status updated for subject $subject, task $task')
"
else
    echo "MNE-BIDS-Pipeline failed at $(date)" >&2
    
    # Update status CSV with failure
    python -c "
import pandas as pd
import os

# Create status entry
status_df = pd.DataFrame([{
    'subject': '$subject', 
    'task': '$task', 
    'data': 'eeg', 
    'status': 'failed_mne_bids',
    'pipeline': 'mne-bids-pipeline'
}])

# Update status file
derivatives_folder = '/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_mne_bids'
os.makedirs(derivatives_folder, exist_ok=True)
status_path = os.path.join(derivatives_folder, 'preprocessing_status.csv')

if os.path.exists(status_path):
    existing_df = pd.read_csv(status_path)
    status_df = pd.concat([existing_df, status_df], ignore_index=True)
    
status_df.to_csv(status_path, index=False)
print(f'Failure status updated for subject $subject, task $task')
"
    exit 1
fi 