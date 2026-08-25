#!/bin/bash
# SLURM runner for step 0 — per-subject PTP thresholds for slow-wave detection.
# Single-node job: detection across all subjects fits comfortably in one task.
#
# Usage:
#   sbatch slurm_compute_thresholds.sh
#
# To process a single subject:
#   SUBJECT=sub-03 sbatch slurm_compute_thresholds.sh

#SBATCH --job-name=WM_sw_thresholds
#SBATCH --output=logs/sw_thresholds_%j.out
#SBATCH --error=logs/sw_thresholds_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=16G

set -euo pipefail

WORKDIR=${WORKDIR:-<PATH_TO_YOUR_REPO_ROOT>/junifer_markers/0.compute_sw_thresholds}
CONDA_ENV=${CONDA_ENV:-junifer-eeg-2}
SHELL_KIND=${SHELL:-bash}
CONFIG=${CONFIG:-${WORKDIR}/config.yaml}
SUBJECT=${SUBJECT:-}
LOG_DIR=${LOG_DIR:-${WORKDIR}/logs}

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

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export MNE_LOGGING_LEVEL=${MNE_LOGGING_LEVEL:-WARNING}

if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || echo "[WARN] Could not activate $CONDA_ENV"
fi
set -u

# Resolve python interpreter from the env explicitly so this works whether or
# not `conda activate` succeeds in the SLURM shell environment. Falls back to
# `python` on PATH if the env-specific interpreter is missing.
PYTHON_BIN="${CONDA_PREFIX:-$HOME/miniforge3/envs/$CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi
echo "[INFO] Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

cd "$WORKDIR"

if [ -n "$SUBJECT" ]; then
  echo "[INFO] Computing thresholds for $SUBJECT"
  "$PYTHON_BIN" compute_thresholds.py --config "$CONFIG" --subject "$SUBJECT"
else
  echo "[INFO] Computing thresholds for all subjects in blinding"
  "$PYTHON_BIN" compute_thresholds.py --config "$CONFIG"
fi

echo "[INFO] Done"
