#!/bin/bash
#SBATCH --job-name=mneMarkers
#SBATCH --output=logs/mne_markers_%A_%a.out
#SBATCH --error=logs/mne_markers_%A_%a.err
#SBATCH --array=2-42
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load proxy

# Try to activate the Python environment
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

# Get subject ID from SLURM array task ID
SUBJECT_NUM=$SLURM_ARRAY_TASK_ID
SUBJECT=$(printf "%02d" $SUBJECT_NUM)

echo "Processing subject: $SUBJECT (Task ID: $SLURM_ARRAY_TASK_ID)"

# List of tasks to process
TASKS=("Sart1" "Sart2" "Sart3" "Sart4")

# Process all tasks for this subject
for TASK in "${TASKS[@]}"; do
    echo "Computing MNE markers for subject $SUBJECT, task $TASK"
    
    # Run the Python script for this specific subject and task
    python -u MNE_markers/run_mne_markers.py --subject $SUBJECT --task $TASK
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed subject $SUBJECT, task $TASK"
    else
        echo "ERROR: Failed to process subject $SUBJECT, task $TASK"
    fi
done

echo "Finished processing all tasks for subject: $SUBJECT" 