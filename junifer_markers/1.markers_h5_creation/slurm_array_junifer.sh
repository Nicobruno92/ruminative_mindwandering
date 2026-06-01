#!/bin/bash

# SLURM array runner for Junifer elements - Wandering Mind Project
# Discovers elements dynamically by scanning the derivatives directory.
# No elements CSV files needed - everything is generated in memory.
#
# Usage (sbatch):
#   CONFIG_TYPE=evoked sbatch --array=0-$(python discover_elements.py -d evoked -c)-1 slurm_array_junifer.sh
#
# Or let run_full_pipeline.sh handle it automatically.

#SBATCH --job-name=WM_markers
#SBATCH --output=logs/WM_markers_%A_%a.out
#SBATCH --error=logs/WM_markers_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=8G

set -euo pipefail

# Parameters (overridable)
CONFIG_TYPE=${CONFIG_TYPE:-state}  # "state", "evoked" or "sleep"
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/junifer_markers/1.markers_h5_creation}
CONDA_ENV=${CONDA_ENV:-junifer-eeg-2}
SHELL_KIND=${SHELL:-zsh}
CONFIG=${CONFIG:-${WORKDIR}/config_${CONFIG_TYPE}.yaml}
LOG_DIR=${LOG_DIR:-${WORKDIR}/logs}
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-}

mkdir -p "$LOG_DIR"

# Robust conda activation across clusters
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

# Constrain threading to reduce cluster contention
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
set -u

# junifer_eeg / reader_picnic / sart_datagrabber_picnic are installed editable
# in the junifer-eeg-2 env, so no PYTHONPATH override is needed by default.
if [ -n "${PYTHONPATH_EXTRA}" ]; then
  export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
fi

cd "$WORKDIR"

# Get task ID
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}

# Discover element dynamically using Python script
ELEMENT=$(python discover_elements.py --desc "${CONFIG_TYPE}" --index "${TASK_ID}")

if [ -z "$ELEMENT" ]; then
  echo "[ERROR] No element found for index $TASK_ID and CONFIG_TYPE=${CONFIG_TYPE}"
  exit 1
fi

# Parse comma-separated element for logging and filename construction
# discover_elements.py outputs dynamic fields (3 or 4) depending on whether
# the dataset uses sessions. CYBERSART/depressed_mindwandering does NOT.
NUM_FIELDS=$(echo "$ELEMENT" | awk -F, '{print NF}')

if [ "$NUM_FIELDS" -eq 4 ]; then
    # Format: subject,session,task,desc
    SUBJECT=$(echo "$ELEMENT" | awk -F, '{print $1}')
    SESSION=$(echo "$ELEMENT" | awk -F, '{print $2}')
    TASK=$(echo "$ELEMENT" | awk -F, '{print $3}')
    DESC=$(echo "$ELEMENT" | awk -F, '{print $4}')

    JUNIFER_ELEMENT="$ELEMENT"
    OUTPUT_FILENAME="element_${SUBJECT}_${SESSION}_${TASK}_${DESC}_markers.h5"

elif [ "$NUM_FIELDS" -eq 3 ]; then
    # Format: subject,task,desc (no session — this project)
    SUBJECT=$(echo "$ELEMENT" | awk -F, '{print $1}')
    TASK=$(echo "$ELEMENT" | awk -F, '{print $2}')
    DESC=$(echo "$ELEMENT" | awk -F, '{print $3}')
    SESSION=""

    JUNIFER_ELEMENT="$ELEMENT"
    OUTPUT_FILENAME="element_${SUBJECT}_${TASK}_${DESC}_markers.h5"
else
    echo "[ERROR] Unexpected element format (expected 3 or 4 fields): $ELEMENT"
    exit 1
fi

echo "[INFO] Job ${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA} [${CONFIG_TYPE}] -> element: $ELEMENT"

# Extract storage URI from config to determine output directory
OUTPUT_URI=$(grep "uri:" "$CONFIG" | head -n 1 | awk '{print $2}')
OUTPUT_DIR=$(dirname "$OUTPUT_URI")
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# Clean up existing file to force Junifer to write a new one
if [ -f "$OUTPUT_FILE" ]; then
  echo "[INFO] Removing existing marker file: $OUTPUT_FILE"
  rm -f "$OUTPUT_FILE"
fi

# Run the specific element
set -x
junifer run "$CONFIG" --verbose info --element "$JUNIFER_ELEMENT"
set +x

echo "[INFO] Done [${CONFIG_TYPE}]: $SUBJECT,$SESSION,$TASK,$DESC"
