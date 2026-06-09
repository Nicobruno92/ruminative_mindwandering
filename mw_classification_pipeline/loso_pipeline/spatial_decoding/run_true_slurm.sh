#!/bin/bash
# =============================================================================
# LOSO spatial decoding — TRUE-RUN array (one task per dimension).
#
# Each task computes the n_runs-averaged true AUC for ALL 64 channels of one
# dimension, in-process (the data is loaded once from the per-channel cache).
# Writes true/true_per_channel_auc.csv under the dimension's results dir.
#
# Run AFTER precompute_cache.sh so the cache exists (avoids a cold ~10-min load).
# =============================================================================
#SBATCH --job-name=loso_sp_true
#SBATCH --output=logs/loso_sp_true_%A_%a.out
#SBATCH --error=logs/loso_sp_true_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(ON_vs_OFF_within_median valence_within_median selfother_within_median time_within_median confidence_within_median)
CONTRAST=${CONTRASTS[$SLURM_ARRAY_TASK_ID]}

echo "TRUE task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST (all channels)"
cd "$HERE/../.."
"$PYTHON" "$HERE/run_loso_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST"
