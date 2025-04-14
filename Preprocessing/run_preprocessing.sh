#!/bin/bash
#SBATCH --job-name=EEGPreprocessing
#SBATCH --output=logs/preprocessing_%A_%a.out
#SBATCH --error=logs/preprocessing_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/home/nicolas.bruno/depressed_mindwandering/
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

# Define subjects and tasks
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

# Calculate subject-task pair from SLURM_ARRAY_TASK_ID
subject_idx=$((SLURM_ARRAY_TASK_ID / 4))
task_idx=$((SLURM_ARRAY_TASK_ID % 4))

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "Processing subject $subject for task $task"

# Run the Python script with the current subject and task
python Preprocessing/cluster_preprocessing.py $subject $task
