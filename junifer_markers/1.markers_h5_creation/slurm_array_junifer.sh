#!/bin/bash

# SLURM array runner for Junifer elements
# Maps each SLURM_ARRAY_TASK_ID to a single element (subject,task) from the
# auto-generated junifer_jobs/<jobname>/elements file and runs that element.
#
# Usage (sbatch):
#   sbatch --array=0-(N-1) slurm_array_junifer.sh
# Environment variables you can override with --export or inline before sbatch:
#   JOBNAME           Default: CYBERSART_features
#   WORKDIR           Default: /network/iss/home/nicolas.bruno/Junifer
#   CONDA_ENV         Default: junifer
#   CONFIG            Default: ${WORKDIR}/junifer_jobs/${JOBNAME}/config.yaml
#   ELEMENTS_FILE     Default: ${WORKDIR}/junifer_jobs/${JOBNAME}/elements.csv
#   LOG_DIR           Default: ${WORKDIR}/logs
#   PYTHONPATH_EXTRA  Default: ${WORKDIR}
#   SHELL             Default: zsh
#
# Example:
#   sbatch --array=0-167 slurm_array_junifer.sh

#SBATCH --job-name=CYBERSART_markers
#SBATCH --output=logs/CYBERSART_markers_%A_%a.out
#SBATCH --error=logs/CYBERSART_markers_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=8G

set -euo pipefail

# Parameters (overridable)
CONFIG_TYPE=${CONFIG_TYPE:-state}  # "state", "evoked" or "sleep" (matches desc in elements.csv)
JOBNAME=${JOBNAME:-CYBERSART_${CONFIG_TYPE}}
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering}
CONDA_ENV=${CONDA_ENV:-junifer}
SHELL_KIND=${SHELL:-zsh}
CONFIG=${CONFIG:-${WORKDIR}/junifer_markers/1.markers_h5_creation/config_${CONFIG_TYPE}.yaml}
ELEMENTS_FILE=${ELEMENTS_FILE:-${WORKDIR}/junifer_markers/1.markers_h5_creation/elements.csv}
LOG_DIR=${LOG_DIR:-${WORKDIR}/logs}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-/network/iss/home/nicolas.bruno/Junifer}

mkdir -p "$LOG_DIR"

# Optional: load modules if your site requires it (edit as needed)
# module load anaconda3 || module load miniconda3 || true

# Robust conda activation across clusters
# Temporarily disable unbound variable check for conda sourcing and activation
set +u
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.${SHELL_KIND} hook)"
else
  echo "[WARN] Could not source conda.sh; relying on preconfigured environment"
fi

# Constrain threading to reduce cluster contention (tune as needed)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export MPLBACKEND=${MPLBACKEND:-Agg}
export MNE_MEMMAP_MIN_SIZE=${MNE_MEMMAP_MIN_SIZE:-1M}
export MNE_LOGGING_LEVEL=${MNE_LOGGING_LEVEL:-WARNING}
export KMP_AFFINITY=${KMP_AFFINITY:-disabled}
export ORT_DISABLE_THREAD_AFFINITY=${ORT_DISABLE_THREAD_AFFINITY:-1}

# Activate environment
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || echo "[WARN] Could not activate $CONDA_ENV; proceeding"
fi
# Re-enable unbound variable check after conda setup
set -u

# Ensure Python can import local junifer_eeg package
export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"

cd "$WORKDIR"

if [ ! -f "$ELEMENTS_FILE" ]; then
  echo "[ERROR] Elements CSV file not found: $ELEMENTS_FILE"
  echo "Run: junifer queue config.yaml --overwrite --verbose info"
  exit 1
fi

# Map array index (0-based) to the corresponding line among rows matching this CONFIG_TYPE
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}
ELEMENT=$(awk -F',' -v d="${CONFIG_TYPE}" -v idx="$TASK_ID" '
  NR==1 {next}  # skip header
  $3==d {
    if (c==idx) {
      print $0
      exit
    }
    c++
  }
' "$ELEMENTS_FILE" | tr -d '\r')

if [ -z "$ELEMENT" ]; then
  echo "[ERROR] No element for index $TASK_ID for CONFIG_TYPE=${CONFIG_TYPE} in $ELEMENTS_FILE"
  exit 1
fi

# Parse comma-separated subject,task,desc for logging
SUBJECT=$(echo "$ELEMENT" | awk -F, '{print $1}')
TASK=$(echo "$ELEMENT" | awk -F, '{print $2}')
DESC=$(echo "$ELEMENT" | awk -F, '{print $3}')

echo "[INFO] Job ${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA} [${CONFIG_TYPE}] -> element: $SUBJECT,$TASK,$DESC"

# Run the specific element - pass the full element string as expected by Junifer
set -x
junifer run "$CONFIG" --verbose info --element "$ELEMENT"
set +x

echo "[INFO] Done [${CONFIG_TYPE}]: $SUBJECT,$TASK"
