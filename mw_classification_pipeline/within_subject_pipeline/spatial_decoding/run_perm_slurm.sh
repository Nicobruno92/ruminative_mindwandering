#!/bin/bash
# =============================================================================
# Within-Subject spatial decoding — PERMUTATION array (max-statistic null).
# Axis = (dimension × permutation block). Each task runs PERMS_PER_JOB within-
# subject shuffles; each shuffle scored across ALL 64 channels (n_runs=1).
# Array size MUST equal 7 * ceil(N / PERMS_PER_JOB). N=500, PERMS_PER_JOB=20 →
# 25 blocks/dim × 7 = 175 tasks (array 0-174).
#
# SUBMIT FROM the mw_classification_pipeline/ root:
#     sbatch within_subject_pipeline/spatial_decoding/run_perm_slurm.sh
# Run AFTER scripts/precompute_spatial_cache.py.
# =============================================================================
#SBATCH --job-name=ws_sp_perm
#SBATCH --output=within_subject_pipeline/spatial_decoding/logs/ws_sp_perm_%A_%a.out
#SBATCH --error=within_subject_pipeline/spatial_decoding/logs/ws_sp_perm_%A_%a.err
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-174

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the mw_classification_pipeline/ root}"
SD="within_subject_pipeline/spatial_decoding"
CONFIG="$SD/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(on_vs_off_within_median valence_within_median selfother_within_median time_within_median confidence_within_median valence_sq time_sq)
N=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['n_permutations'])")
PERMS_PER_JOB=20
BLOCKS_PER_DIM=$(( (N + PERMS_PER_JOB - 1) / PERMS_PER_JOB ))

DIM_IDX=$(( SLURM_ARRAY_TASK_ID / BLOCKS_PER_DIM ))
BLOCK_IDX=$(( SLURM_ARRAY_TASK_ID % BLOCKS_PER_DIM ))
CONTRAST=${CONTRASTS[$DIM_IDX]}
PERM_START=$(( BLOCK_IDX * PERMS_PER_JOB ))
PERM_END=$(( PERM_START + PERMS_PER_JOB - 1 ))
if [ $PERM_END -ge $N ]; then PERM_END=$(( N - 1 )); fi

echo "PERM task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST perms $PERM_START..$PERM_END (N=$N)"
for P in $(seq $PERM_START $PERM_END); do
  "$PYTHON" "$SD/run_within_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST" --perm_idx "$P"
done
