#!/bin/bash
#SBATCH --job-name=mw_lmm_diag
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mw_lmm_diag_%j.out
#SBATCH --error=logs/mw_lmm_diag_%j.err

# LMM assumption diagnostics for the Andrillon cluster-permutation pipeline.
#
# Refits the OBSERVED model only (no permutations) for every predictor x marker
# and writes per-channel residual diagnostics, a per-predictor summary table and
# a summary figure under <output_path>/<model_folder>/assumption_diagnostics/.
#
# --cpus-per-task must match andrillon_clustering.n_jobs in the config: loky
# reads the node's physical core count rather than the cgroup allocation, so a
# mismatch oversubscribes the allocated cores.

set -euo pipefail

# Load modules
module load proxy

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
fi

# Ensure the env's python wins on PATH
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export MPLBACKEND=Agg

PROJECT_ROOT=<PATH_TO_YOUR_REPO_ROOT>
CONFIG="${PROJECT_ROOT}/Stats_andrillon/config_andrillon.yaml"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "LMM ASSUMPTION DIAGNOSTICS"
echo "=========================================="
echo "Start time: $(date)"
echo "Config: ${CONFIG}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo ""

python Stats_andrillon/lmm_assumption_diagnostics.py \
    --config "${CONFIG}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}"

echo ""
echo "✓ Assumption diagnostics completed at $(date)"
echo "=========================================="
