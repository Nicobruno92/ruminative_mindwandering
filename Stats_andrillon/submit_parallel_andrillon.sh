#!/bin/bash
# Helper script to submit SLURM array job for Andrillon pipeline
# Automatically determines the number of markers and submits array job

# Configuration
CONFIG_FILE="Stats_andrillon/config_andrillon.yaml"
ARRAY_SCRIPT="Stats_andrillon/submit_andrillon_array.sh"
SCRIPT_DIR="Stats_andrillon"

# Set working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "Andrillon Pipeline - Parallel Submission"
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Check if array script exists
if [ ! -f "${ARRAY_SCRIPT}" ]; then
    echo "ERROR: Array script ${ARRAY_SCRIPT} not found!"
    exit 1
fi

# Load modules
module load proxy

# Activate conda environment
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
  echo "ERROR: Could not activate conda environment"
  exit 1
fi

# Get number of markers from config using Python
echo "Detecting number of markers from config..."
N_MARKERS=$(python -c "
import yaml
import sys
from pathlib import Path

# Add Stats_andrillon to path
sys.path.insert(0, 'Stats_andrillon')
sys.path.insert(0, 'Statistics')

from reader import get_available_markers

# Load config
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

features_root = config['project']['features_root']
marker_types = config['project'].get('marker_types', None)
markers_config = config['project'].get('markers', 'all')

# Determine number of markers
if isinstance(markers_config, str) and markers_config.lower() == 'all':
    # Get all markers
    available_markers_dict = get_available_markers(features_root, marker_types)
    n_markers = sum(len(markers) for markers in available_markers_dict.values())
elif isinstance(markers_config, list):
    # Count specified markers
    n_markers = len(markers_config)
else:
    n_markers = 0

print(n_markers)
")

if [ -z "${N_MARKERS}" ] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "ERROR: Could not determine number of markers or no markers found!"
    echo "Please check your config_andrillon.yaml file."
    exit 1
fi

echo "Found ${N_MARKERS} markers to process"
echo "Submitting SLURM array job with ${N_MARKERS} tasks..."

# Submit array job with correct number of tasks
# Array indices: 0 to N_MARKERS-1
ARRAY_RANGE="0-$((N_MARKERS-1))"

# Submit the array job and capture the job ID
ARRAY_JOB_OUTPUT=$(sbatch --array=${ARRAY_RANGE} ${ARRAY_SCRIPT})
ARRAY_EXIT_CODE=$?

if [ ${ARRAY_EXIT_CODE} -ne 0 ]; then
    echo "✗ Failed to submit array job"
    exit 1
fi

# Extract job ID from sbatch output
ARRAY_JOB_ID=$(echo ${ARRAY_JOB_OUTPUT} | awk '{print $NF}')

echo "✓ Array job submitted successfully"
echo "  Job ID: ${ARRAY_JOB_ID}"
echo "  Array range: ${ARRAY_RANGE}"
echo "  Total tasks: ${N_MARKERS}"
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/andrillon_${ARRAY_JOB_ID}_*.out"
echo "=========================================="
