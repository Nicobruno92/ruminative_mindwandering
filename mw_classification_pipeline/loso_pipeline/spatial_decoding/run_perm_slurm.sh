#!/bin/bash
# =============================================================================
# LOSO spatial decoding — PERMUTATION array (max-statistic null).
#
# Axis = (dimension × permutation block). Each task runs PERMS_PER_JOB within-
# subject shuffles; each shuffle is scored across ALL 64 channels (n_runs=1) so
# the per-permutation max-over-channels AUC is a valid family-wise null draw.
# Writes perms/perm-{P}.csv per shuffle.
#
# N (= permutation_runs) and PERMS_PER_JOB are read from / set here; the array
# size MUST equal 5 * ceil(N / PERMS_PER_JOB). With N=500, PERMS_PER_JOB=20 →
# 25 blocks/dim × 5 dims = 125 tasks (array 0-124).
#
# Run AFTER precompute_cache.sh.
# =============================================================================
#SBATCH --job-name=loso_sp_perm
#SBATCH --output=logs/loso_sp_perm_%A_%a.out
#SBATCH --error=logs/loso_sp_perm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-124

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$HERE/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(ON_vs_OFF_within_median valence_within_median selfother_within_median time_within_median confidence_within_median)
# N must match permutation_runs in config.yaml; PERMS_PER_JOB sets the block size.
N=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['permutation_runs'])")
PERMS_PER_JOB=20
BLOCKS_PER_DIM=$(( (N + PERMS_PER_JOB - 1) / PERMS_PER_JOB ))

DIM_IDX=$(( SLURM_ARRAY_TASK_ID / BLOCKS_PER_DIM ))
BLOCK_IDX=$(( SLURM_ARRAY_TASK_ID % BLOCKS_PER_DIM ))
CONTRAST=${CONTRASTS[$DIM_IDX]}
PERM_START=$(( BLOCK_IDX * PERMS_PER_JOB ))
PERM_END=$(( PERM_START + PERMS_PER_JOB - 1 ))
if [ $PERM_END -ge $N ]; then PERM_END=$(( N - 1 )); fi

echo "PERM task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST perms $PERM_START..$PERM_END (N=$N)"
cd "$HERE/../.."
for P in $(seq $PERM_START $PERM_END); do
  "$PYTHON" "$HERE/run_loso_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST" --perm_idx "$P"
done
