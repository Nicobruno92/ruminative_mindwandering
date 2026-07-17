#!/bin/bash
#SBATCH --job-name=combined_fig
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=mw_classification_pipeline/logs/combined_figure_%j.out
#SBATCH --error=mw_classification_pipeline/logs/combined_figure_%j.err

# =============================================================================
# run_combined_figure_slurm.sh — generate_combined_classification_figure.py
# =============================================================================
# Runs the combined WS/LOSO classification figure generator on a dedicated
# compute node, so kaleido's headless-Chrome image export isn't starved by
# CPU contention on the shared login node.
#
# USAGE (from project root):
#   sbatch mw_classification_pipeline/scripts/run_combined_figure_slurm.sh
#
# MONITOR:
#   squeue -u $USER
#   tail -f mw_classification_pipeline/logs/combined_figure_<JOBID>.out
# =============================================================================

set -euo pipefail

export PYTHONUNBUFFERED=1

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
mkdir -p mw_classification_pipeline/logs

PYTHON=/network/iss/home/nicolas.bruno/miniforge3/envs/plots/bin/python

"$PYTHON" mw_classification_pipeline/scripts/generate_combined_classification_figure.py
