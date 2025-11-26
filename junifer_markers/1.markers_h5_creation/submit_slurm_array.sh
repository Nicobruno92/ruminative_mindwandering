#!/bin/bash

# Helper to prepare Junifer jobs and submit SLURM arrays for state, evoked, and sleep epochs.
# It will:
#  1) Ensure the job folders exist (optionally regenerate with --queue)
#  2) Determine the number of elements for each config
#  3) Submit sbatch arrays for state, evoked, and sleep configs
#
# Usage:
#  ./submit_slurm_array.sh [--queue]
# Environment variables (override as needed):
#  WORKDIR=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
#  CONDA_ENV=junifer
#  PARTITION=default
#  CPUS=4 MEM=8G TIME=08:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering}
CONDA_ENV=${CONDA_ENV:-junifer}
PARTITION=${PARTITION:-}
CPUS=${CPUS:-4}
MEM=${MEM:-8G}
TIME=${TIME:-08:00:00}

QUEUE_FLAG="no"
if [[ "${1:-}" == "--queue" ]]; then
  QUEUE_FLAG="yes"
fi

ALL_JOB_IDS=()

cd "$WORKDIR"

# Optionally (re)create the master elements.csv via a single global junifer queue
if [[ "$QUEUE_FLAG" == "yes" ]]; then
  echo "[INFO] (Re)creating global elements.csv via junifer queue (config.yaml)"
  export PYTHONPATH="/network/iss/home/nicolas.bruno/Junifer:${PYTHONPATH:-}"
  # Temporarily disable unbound variable check for conda sourcing and activation
  set +u
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV" || true
  fi
  set -u
  junifer queue "junifer_markers/1.markers_h5_creation/config.yaml" --overwrite --verbose info
fi

# Ensure master elements.csv exists
MASTER_ELEMENTS="junifer_markers/1.markers_h5_creation/elements.csv"
if [ ! -f "$MASTER_ELEMENTS" ]; then
  echo "[ERROR] Master elements file not found: $MASTER_ELEMENTS"
  echo "       Run with --queue or generate it via 'junifer queue config.yaml'"
  exit 1
fi

# Function to submit a job for a specific config
submit_job() {
  local CONFIG_TYPE=$1  # "state", "evoked" or "sleep" (matches desc in elements.csv)
  local YAML="junifer_markers/1.markers_h5_creation/config_${CONFIG_TYPE}.yaml"
  local JOBNAME="CYBERSART_${CONFIG_TYPE}"
  
  echo ""
  echo "========================================"
  echo "Processing ${CONFIG_TYPE} epochs"
  echo "========================================"

  # Count how many rows in the master elements.csv match this desc
  echo "[INFO] Counting elements in $MASTER_ELEMENTS for desc='${CONFIG_TYPE}' (excluding header)"
  local N
  N=$(awk -F',' -v d="${CONFIG_TYPE}" 'NR>1 && $3==d {c++} END{print c+0}' "$MASTER_ELEMENTS")
  if [[ "$N" -le 0 ]]; then
    echo "[WARN] No elements found for ${CONFIG_TYPE} (desc='${CONFIG_TYPE}') in $MASTER_ELEMENTS"
    return 0
  fi
  local END=$((N - 1))
  echo "[INFO] Submitting array 0-$END ($N elements) for ${CONFIG_TYPE}"
  
  # Optional: Delete existing HDF5 file to force overwrite
  H5_FILE="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/junifer/markers_${CONFIG_TYPE}.h5"
  if [[ "${OVERWRITE:-no}" == "yes" ]] && [[ -f "$H5_FILE" ]]; then
    echo "[WARN] Deleting existing HDF5 file: $H5_FILE"
    rm -f "$H5_FILE"
  fi
  
  SBATCH_ARGS=()
  # Inject resources via sbatch overrides
  SBATCH_ARGS+=("--array=0-${END}")
  SBATCH_ARGS+=("--cpus-per-task=${CPUS}")
  SBATCH_ARGS+=("--mem=${MEM}")
  SBATCH_ARGS+=("--time=${TIME}")
  if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS+=("--partition=${PARTITION}")
  fi
  SBATCH_ARGS+=("junifer_markers/1.markers_h5_creation/slurm_array_junifer.sh")
  
  # Export key environment vars to the job
  export JOBNAME WORKDIR CONDA_ENV CONFIG_TYPE
  export ELEMENTS_FILE="$MASTER_ELEMENTS"
  
  set -x
  JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" | awk '{print $4}')
  set +x
  
  echo "[INFO] Submitted ${CONFIG_TYPE} array job: $JOB_ID"
  ALL_JOB_IDS+=("$JOB_ID")
}

# Submit jobs for state, evoked, and sleep epochs
submit_job "state" "$@"
submit_job "evoked" "$@"
submit_job "sleep" "$@"

if ((${#ALL_JOB_IDS[@]} > 0)); then
  STEP1_IDS_JOINED=$(IFS=:; echo "${ALL_JOB_IDS[*]}")
  echo "Submitted array job: ${STEP1_IDS_JOINED}"
fi

echo ""
echo "========================================"
echo "All jobs submitted successfully!"
echo "========================================"