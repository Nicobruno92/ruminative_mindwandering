#!/bin/bash
#SBATCH --job-name=EEGPreprocessing
#SBATCH --output=logs/preprocessing_%A_%a.out
#SBATCH --error=logs/preprocessing_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/lustre/iss02/home/nicolas.bruno/depressed_mindwandering
module load anaconda
module load CUDA/12.1
activate my_eeg

echo "Computing ERPs"

# Run the Python script with the current subject and task
python ERPs/run_ERP_analysis.py
