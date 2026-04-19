#!/bin/bash
#SBATCH --job-name=cluster_erp
#SBATCH --output=logs/cluster_erp_%j.out
#SBATCH --error=logs/cluster_erp_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

# =============================================================
# Plot group ERP for significant LMM cluster channels
# (On-task vs Off-task, configurable classification method)
#
# Usage (interactive):
#   bash ERPs_new/run_cluster_erp.sh
#   bash ERPs_new/run_cluster_erp.sh --method within_subject_median
#   bash ERPs_new/run_cluster_erp.sh --method fixed --threshold 50
#
# Usage (SLURM):
#   sbatch ERPs_new/run_cluster_erp.sh
#   sbatch ERPs_new/run_cluster_erp.sh --method global_median
# =============================================================

set -euo pipefail

mkdir -p logs

module load proxy || true

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
else
  module load anaconda3 || module load miniconda3 || module load conda || true
  conda activate eeg || true
fi

export MPLBACKEND=Agg

echo "============================================================"
echo "Cluster ERP pipeline"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Default: use classification_method from config.yaml
# Override via CLI args passed through, e.g.:
#   bash run_cluster_erp.sh --method fixed --threshold 50
srun --cpu-bind=none --ntasks=1 python ERPs_new/plot_cluster_erp.py \
    --config ERPs_new/config.yaml \
    "$@"

echo "============================================================"
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results: results/ERPs_new/cluster_erp/"
echo "============================================================"
