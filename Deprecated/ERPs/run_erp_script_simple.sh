#!/bin/bash
#SBATCH --job-name=erpAnalysis
#SBATCH --output=logs/erp_analysis_%j.out
#SBATCH --error=logs/erp_analysis_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

# Add error handling
set -e  # Exit on any error

# Print debug information
echo "=== ERP ANALYSIS DEBUG INFO ==="
echo "Current working directory: $(pwd)"
echo "Date/time: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-Not set}"
echo "User: $(whoami)"
echo "Available memory: $(free -h | head -2)"
echo "Available CPUs: $(nproc)"
echo "==============================="

# Create logs directory if it doesn't exist
mkdir -p logs
echo "Logs directory created/verified"

# Load required modules with error checking
echo "Loading proxy module..."
if module load proxy; then
    echo "Proxy module loaded successfully"
else
    echo "Warning: Failed to load proxy module, continuing anyway"
fi

# Try to activate the Python environment
echo "Attempting to activate Python environment..."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "Using miniconda3 path"
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
    echo "Conda environment activated via miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "Using anaconda3 path"
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
    echo "Conda environment activated via anaconda3"
elif command -v conda >/dev/null 2>&1; then
    echo "Using conda from PATH"
    eval "$(conda shell.bash hook)"
    conda activate eeg
    echo "Conda environment activated from PATH"
else
    echo "Using module method"
    if module load anaconda3 || module load miniconda3 || module load conda; then
        echo "Conda module loaded successfully"
        if conda activate eeg; then
            echo "Conda environment activated via module"
        else
            echo "ERROR: Failed to activate eeg environment"
            exit 1
        fi
    else
        echo "ERROR: No conda module found and conda not in PATH"
        exit 1
    fi
fi

# Verify Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "Available Python packages:"
pip list | head -10

echo "Starting ERP analysis..."

# Run the Python script
python ./ERPs/run_ERP_analysis.py

echo "ERP analysis completed successfully" 