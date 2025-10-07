#!/bin/bash
#SBATCH --job-name=clusterLMM
#SBATCH --output=logs/cluster_lmm_%A_%a.out
#SBATCH --error=logs/cluster_lmm_%A_%a.err
#SBATCH --array=1-33
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
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

# Define all markers
MARKERS=("a" "a_n" "b" "b_n" "cnv" "d" "d_n" "g" "g_n" "kurtosis" "mean_amp" "msf" "n1" "p1" "p1_amp" "p1_lat" "p2" "p2p_amp" "p3a" "p3a_amp" "p3a_lat" "p3b" "p3b_amp" "p3b_lat" "rms" "se" "sef90" "sef95" "skew" "std" "t" "t_n" "t_n_ratio" "var")

# Get marker from SLURM array task ID
MARKER_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
MARKER=${MARKERS[$MARKER_INDEX]}

echo "Processing marker: $MARKER (Task ID: $SLURM_ARRAY_TASK_ID)"

# Run the Python script for this specific marker
python -u /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Stats/onoff/run_cluster_mixedlm.py --marker $MARKER

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Successfully processed marker $MARKER"
else
    echo "ERROR: Failed to process marker $MARKER"
fi

echo "Finished processing marker: $MARKER" 