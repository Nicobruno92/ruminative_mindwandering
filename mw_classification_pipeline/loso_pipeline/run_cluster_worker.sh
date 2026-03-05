#!/usr/bin/env bash
# =============================================================================
# run_cluster_worker.sh — SLURM worker (one array element)
# =============================================================================
# Submitted by run_cluster.sh as a SLURM array element.
# Reads its (model, contrast, family) from logs/.combinations.txt via
# SLURM_ARRAY_TASK_ID. All parameters come from config.yaml.
#
# DO NOT submit this script directly — use run_cluster.sh instead.
# =============================================================================
# Note: --chdir is passed explicitly by run_cluster.sh as an absolute path.
# SLURM_SUBMIT_DIR is used as a fallback to guarantee the correct working directory
# even if the script is copied to a temp location before execution.

set -euo pipefail

# Use SLURM_SUBMIT_DIR (set by SLURM at submission time) as the authoritative
# working directory — reliable even when SLURM copies the script to a tmpdir.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"
# If CONFIG was passed as an absolute path (by run_cluster.sh), use it as-is;
# otherwise resolve relative to SCRIPT_DIR.
[[ "$CONFIG" != /* ]] && CONFIG="$SCRIPT_DIR/$CONFIG"

# ---------------------------------------------------------------------------
# Resolve this array task's (model, contrast, family) combination
# ---------------------------------------------------------------------------
COMBINATIONS_FILE="logs/.combinations.txt"
if [ ! -f "$COMBINATIONS_FILE" ]; then
    echo "ERROR: Combination file not found: $COMBINATIONS_FILE"
    echo "Run run_cluster.sh first to generate it."
    exit 1
fi

COMBO=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$COMBINATIONS_FILE")
if [ -z "$COMBO" ]; then
    echo "ERROR: No combination at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

MODEL_T=$(echo "$COMBO" | cut -d: -f1)
CONTRAST=$(echo "$COMBO" | cut -d: -f2)
FAMILY=$(echo "$COMBO" | cut -d: -f3)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
CONDA_ENV=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('slurm', {}).get('conda_env', 'ML'))
")

# Activate conda robustly
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: Cannot find conda. Activate $CONDA_ENV manually."
    exit 1
fi
conda activate "$CONDA_ENV"

# Propagate SLURM CPU count to threading libraries
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------
echo "========================================================"
echo " MW Classification Pipeline — SLURM Worker"
echo "========================================================"
echo " Job ID        : ${SLURM_JOB_ID:-local}"
echo " Array task    : ${SLURM_ARRAY_TASK_ID:-0}"
echo " Combination   : $COMBO"
echo " Model         : $MODEL_T"
echo " Contrast      : $CONTRAST"
echo " Family        : $FAMILY"
echo " Config        : $CONFIG"
echo " Host          : $(hostname)"
echo " CPUs          : ${SLURM_CPUS_PER_TASK:-?}"
echo " Conda env     : $CONDA_ENV"
echo " Started       : $(date)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python run_loso_classification.py \
    --config    "$CONFIG" \
    --contrast  "$CONTRAST" \
    --family    "$FAMILY" \
    --model_type "$MODEL_T"

EXIT_CODE=$?

echo "========================================================"
if [ $EXIT_CODE -ne 0 ]; then
    echo " FAILED: $MODEL_T | $CONTRAST | $FAMILY (exit $EXIT_CODE)"
    exit $EXIT_CODE
else
    echo " SUCCESS: $MODEL_T | $CONTRAST | $FAMILY"
    echo " Finished : $(date)"
fi
echo "========================================================"
