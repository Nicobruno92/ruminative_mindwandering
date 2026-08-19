#!/bin/bash
# =============================================================================
# Within-Subject spatial decoding — TRUE-RUN array, axis (dimension × channel)
# = 768 tasks (12 dims × 64 channels). Each task computes the n_runs-averaged
# true group-mean AUC for ONE electrode of one dimension and writes a
# true/channel-{CH}.csv shard.
#
# valence_sq_res/time_sq_res/valence_sq_res_cross/time_sq_res_cross are NOT
# included — the quadratic construct was dropped from the paper 2026-08-13
# (see within_subject_pipeline/config.yaml run_contrasts); valence_sq/time_sq
# themselves stay for now as already-computed results predating that call.
#
# SUBMIT FROM the mw_classification_pipeline/ root:
#     sbatch within_subject_pipeline/spatial_decoding/run_true_slurm.sh
# Run AFTER scripts/precompute_spatial_cache.py.
# =============================================================================
#SBATCH --job-name=ws_sp_true
#SBATCH --output=within_subject_pipeline/spatial_decoding/logs/ws_sp_true_%A_%a.out
#SBATCH --error=within_subject_pipeline/spatial_decoding/logs/ws_sp_true_%A_%a.err
#SBATCH --partition=compute
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-767

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the mw_classification_pipeline/ root}"
SD="within_subject_pipeline/spatial_decoding"
CONFIG="$SD/config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"

CONTRASTS=(on_vs_off_within_median valence_within_median selfother_within_median time_within_median confidence_within_median valence_sq time_sq onoff_within_median_res valence_within_median_res selfother_within_median_res time_within_median_res confidence_within_median_res)
CHANNELS=(AF3 AF4 AF7 AF8 AFz C1 C2 C3 C4 C5 C6 CP1 CP2 CP3 CP4 CP5 CP6 CPz Cz F1 \
          F2 F3 F4 F5 F6 F7 F8 FC1 FC2 FC3 FC4 FC5 FC6 FT10 FT7 FT8 FT9 Fp1 Fp2 Fz \
          Iz O1 O2 Oz P1 P2 P3 P4 P5 P6 P7 P8 PO3 PO4 PO7 PO8 POz Pz T7 T8 TP10 TP7 TP8 TP9)
N_CH=${#CHANNELS[@]}
DIM_IDX=$(( SLURM_ARRAY_TASK_ID / N_CH ))
CH_IDX=$(( SLURM_ARRAY_TASK_ID % N_CH ))
CONTRAST=${CONTRASTS[$DIM_IDX]}
CHANNEL=${CHANNELS[$CH_IDX]}

echo "TRUE task $SLURM_ARRAY_TASK_ID → contrast=$CONTRAST channel=$CHANNEL"
"$PYTHON" "$SD/run_within_spatial_decoding.py" --config "$CONFIG" --contrast "$CONTRAST" --channel "$CHANNEL"
