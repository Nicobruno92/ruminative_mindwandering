#!/usr/bin/env bash
# =============================================================================
# run_merge.sh — Aggregate LOSO results after all SLURM jobs complete
# =============================================================================
# Run this script after all true-run and perm jobs are finished to produce
# the consolidated summary files and p-values.
#
# USAGE:
#   bash run_merge.sh                     # uses config.yaml
#   bash run_merge.sh path/to/config.yaml
# =============================================================================

set -euo pipefail

CONFIG="${1:-config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('slurm', {}).get('conda_env', 'ML'))
")

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "Merging results from: $CONFIG"
python merge_loso_results.py --config "$CONFIG"
