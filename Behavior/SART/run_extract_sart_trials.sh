#!/bin/bash
#SBATCH --job-name=extract_sart_trials
#SBATCH --output=logs/extract_sart_trials_%j.out
#SBATCH --error=logs/extract_sart_trials_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=medium

# Extract SART behavioral trial data from BIDS events.tsv files
# This script processes all subjects and tasks

# Activate conda environment
source ~/.bashrc
conda activate base

# Set paths
SCRIPT_DIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Behavior"
BIDS_RAW="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/raw"
BIDS_OUT="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS"

# Create logs directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/logs"

# Run the extraction script
echo "Starting SART trial extraction..."
echo "BIDS Raw Root: ${BIDS_RAW}"
echo "BIDS Output Root: ${BIDS_OUT}"
echo "Date: $(date)"

python "${SCRIPT_DIR}/SART/extract_sart_trials_from_bids.py" \
    --bids-raw-root "${BIDS_RAW}" \
    --bids-output-root "${BIDS_OUT}"

echo "Extraction complete: $(date)"
