#!/bin/bash
#SBATCH --job-name=erpAnalysis
#SBATCH --output=logs/preprocessing_%A_%a.out
#SBATCH --error=logs/preprocessing_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=32G
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


echo "Computing ERPs"

# Run the Python script with the current subject and task
python ERPs/run_ERP_analysis.py
