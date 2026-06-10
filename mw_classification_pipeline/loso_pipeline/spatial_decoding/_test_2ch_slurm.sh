#!/bin/bash
# Real 2-electrode end-to-end test (production params) on a compute node. Throwaway.
# SUBMIT FROM mw_classification_pipeline/:  sbatch loso_pipeline/spatial_decoding/_test_2ch_slurm.sh
#SBATCH --job-name=sp_2ch_test
#SBATCH --output=logs/sp_2ch_test_%j.out
#SBATCH --error=logs/sp_2ch_test_%j.err
#SBATCH --partition=compute
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
# SLURM copies the batch script to a spool dir, so BASH_SOURCE is useless for locating
# the repo. Use the submission directory (must be the mw_classification_pipeline/ root).
cd "${SLURM_SUBMIT_DIR:?submit from mw_classification_pipeline/}"
SD="loso_pipeline/spatial_decoding"
CFG="$SD/_test_2ch_config.yaml"
PYTHON="$HOME/miniforge3/envs/ML/bin/python"
CONTRAST=ON_vs_OFF_within_median
RD="results/MW_Classification/SpatialDecoding/LOSO/$CONTRAST/all/rf"

echo "=== node: $(hostname) | cores: $(nproc) | cwd: $(pwd) ==="
echo "=== precompute cache ==="
"$PYTHON" scripts/precompute_spatial_cache.py --config "$CFG"
echo "=== TRUE (Cz,Pz; n_runs=20) ==="
time "$PYTHON" "$SD/run_loso_spatial_decoding.py" --config "$CFG" --contrast "$CONTRAST"
echo "=== PERMS (2 channels each), 4-way parallel, cv_n_jobs=1 ==="
N=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('$CFG'))['permutation_runs'])")
time seq 0 $((N-1)) | xargs -P 4 -I{} "$PYTHON" \
  "$SD/run_loso_spatial_decoding.py" --config "$CFG" --contrast "$CONTRAST" --perm_idx {} \
  > logs/sp_2ch_perms.out 2>&1
echo "=== MERGE (alpha=0.05) ==="
"$PYTHON" "$SD/merge_spatial_results.py" --results_dir "$RD" --alpha 0.05
echo "=== per_channel_metrics.csv ==="; cat "$RD/per_channel_metrics.csv"
echo "=== n perm files ==="; ls "$RD/perms" | wc -l
echo "### 2CH REAL TEST DONE ###"
