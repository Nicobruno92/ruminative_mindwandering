#!/bin/bash
# SLURM runner for step 3 — group-level topomap visualization.
# Single-node job. Runs after step 2 (aggregation) is complete.
#
# Usage:
#   sbatch slurm_topomaps.sh
#   sbatch --dependency=afterany:JOBID slurm_topomaps.sh
#   EPOCH_TYPE=sleep MARKER=psd_bands_alpha sbatch slurm_topomaps.sh

#SBATCH --job-name=WM_topomaps
#SBATCH --output=logs/topomaps_%j.out
#SBATCH --error=logs/topomaps_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=8G

set -euo pipefail

WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/junifer_markers/3.topomaps}
CONDA_ENV=${CONDA_ENV:-eeg}     # `eeg` env has matplotlib + mne + yaml + pandas
SHELL_KIND=${SHELL:-bash}
CONFIG=${CONFIG:-${WORKDIR}/config.yaml}
LOG_DIR=${LOG_DIR:-${WORKDIR}/logs}
EPOCH_TYPE=${EPOCH_TYPE:-}
MARKER=${MARKER:-}

mkdir -p "$LOG_DIR"

# Robust conda activation
set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.${SHELL_KIND} hook)"
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export MPLBACKEND=${MPLBACKEND:-Agg}
export MNE_LOGGING_LEVEL=${MNE_LOGGING_LEVEL:-WARNING}

if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || echo "[WARN] Could not activate $CONDA_ENV"
fi
set -u

PYTHON_BIN="${CONDA_PREFIX:-$HOME/miniforge3/envs/$CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi
echo "[INFO] Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

cd "$WORKDIR"

CMD=("$PYTHON_BIN" visualize_junifer_topomaps.py --config "$CONFIG")
if [ -n "$EPOCH_TYPE" ]; then
  CMD+=(--epoch-type "$EPOCH_TYPE")
fi
if [ -n "$MARKER" ]; then
  CMD+=(--marker "$MARKER")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"
echo "[INFO] Done"
