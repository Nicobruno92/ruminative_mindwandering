#!/bin/bash
# =============================================================================
# Within-Subject spatial decoding — TRUE-RUN array (one task per dimension).
# Computes the n_runs-averaged true group-mean AUC for ALL 64 channels of one
# dimension, in-process (per-channel cache loaded once). Run AFTER precompute.
# =============================================================================
#SBATCH --job-name=ws_sp_true
#SBATCH --output=logs/ws_sp_true_%A_%a.out
#SBATCH --error=logs/ws_sp_true_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(on_vs_off_within_median valence_within_median selfother_within_median time_within_median confidence_within_median)
CONTRAST=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}

echo "TRUE task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST (all channels)"
cd "$HERE/../.."
"$PYTHON" "$HERE/run_within_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST"
