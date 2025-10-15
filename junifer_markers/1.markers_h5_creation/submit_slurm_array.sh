#!/bin/bash

# Helper to prepare Junifer jobs and submit a SLURM array.
# It will:
#  1) Ensure the job folder exists (optionally regenerate with --queue)
#  2) Determine the number of elements
#  3) Submit sbatch with the correct array range
#
# Usage:
#  ./submit_slurm_array.sh [--queue]
# Environment variables (override as needed):
#  JOBNAME=CYBERSART_features
#  WORKDIR=/network/iss/home/nicolas.bruno/Junifer
#  CONDA_ENV=junifer
#  YAML=config.yaml
#  PARTITION=default
#  CPUS=4 MEM=8G TIME=08:00:00

set -euo pipefail

JOBNAME=${JOBNAME:-CYBERSART_features}
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering}
CONDA_ENV=${CONDA_ENV:-eeg}
YAML=${YAML:-junifer_markers/1.markers_h5_creation/config.yaml}
PARTITION=${PARTITION:-}
CPUS=${CPUS:-4}
MEM=${MEM:-8G}
TIME=${TIME:-08:00:00}

cd "$WORKDIR"

if [[ "${1:-}" == "--queue" ]]; then
  echo "[INFO] (Re)creating job folder via junifer queue"
  # Configure PYTHONPATH so junifer finds junifer_eeg
  export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
  # Activate conda if available
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
  junifer queue "$YAML" --overwrite --verbose info
fi

ELEMENTS_FILE="junifer_markers/1.markers_h5_creation/elements"
if [ ! -f "$ELEMENTS_FILE" ]; then
  echo "[ERROR] Elements file not found: $ELEMENTS_FILE"
  echo "Run with --queue to generate it."
  exit 1
fi

echo "[INFO] Counting elements in $ELEMENTS_FILE"
N=$(wc -l < "$ELEMENTS_FILE" | tr -d ' ')
if [[ "$N" -le 0 ]]; then
  echo "[ERROR] No elements found"
  exit 1
fi
END=$((N - 1))
echo "[INFO] Submitting array 0-$END ($N elements)"

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
export JOBNAME WORKDIR CONDA_ENV

set -x
JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" | awk '{print $4}')
set +x

echo "[INFO] Submitted array job: $JOB_ID"

# With single_output=false, each element creates its own H5 file
# No collection step needed!

# Optionally submit PKL creation job after array completes
if [[ "${CREATE_PKL:-yes}" == "yes" ]]; then
  echo "[INFO] Submitting PKL creation job to run after array completes..."
  
  set -x
  PKL_JOB_ID=$(sbatch --dependency=afterok:${JOB_ID} \
         --job-name="${JOBNAME}_pkl" \
         --output="${WORKDIR}/logs/${JOBNAME}_pkl_%j.out" \
         --error="${WORKDIR}/logs/${JOBNAME}_pkl_%j.err" \
         ${WORKDIR}/junifer_markers/2.h5_to_pkl/batch_create_pkl.sh | awk '{print $4}')
  set +x
  
  echo "[INFO] PKL creation job ${PKL_JOB_ID} will run automatically after array job ${JOB_ID} completes"
  echo "[INFO] Pipeline chain: ${JOB_ID} (H5 creation) → ${PKL_JOB_ID} (PKL creation)"
else
  echo "[INFO] PKL creation disabled (set CREATE_PKL=yes to enable)"
fi
