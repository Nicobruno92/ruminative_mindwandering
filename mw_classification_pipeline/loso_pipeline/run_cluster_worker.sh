#!/usr/bin/env bash
# =============================================================================
# run_cluster_worker.sh — SLURM worker (one array element)
# =============================================================================
# Submitted by run_cluster.sh as part of a true-run or perm array.
# Reads its (model, contrast, family, run/perm index) from the combinations
# file via SLURM_ARRAY_TASK_ID.
#
# Arguments:
#   $1 — path to config.yaml
#   $2 — mode: "true" (true-run array) or "perm" (permutation array)
#   $3 — combination-index offset (for arrays chunked under MaxArraySize)
#
# DO NOT submit this script directly — use run_cluster.sh instead.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"
MODE="${2:-true}"   # "true" or "perm"
OFFSET="${3:-0}"    # combination-index offset for chunked arrays

[[ "$CONFIG" != /* ]] && CONFIG="$SCRIPT_DIR/$CONFIG"

# ---------------------------------------------------------------------------
# Resolve this array task's combination
# ---------------------------------------------------------------------------
if [ "$MODE" = "true" ]; then
    COMBINATIONS_FILE="logs/.true_combinations.txt"
else
    COMBINATIONS_FILE="logs/.perm_combinations.txt"
fi

if [ ! -f "$COMBINATIONS_FILE" ]; then
    echo "ERROR: Combination file not found: $COMBINATIONS_FILE"
    echo "Run run_cluster.sh first to generate it."
    exit 1
fi

COMBO_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))
COMBO=$(sed -n "$((COMBO_INDEX + 1))p" "$COMBINATIONS_FILE")
if [ -z "$COMBO" ]; then
    echo "ERROR: No combination at index $COMBO_INDEX"
    exit 1
fi

# Combo format: model:contrast:family:run_N  OR  model:contrast:family:perm_N
MODEL_T=$(echo "$COMBO" | cut -d: -f1)
CONTRAST=$(echo "$COMBO" | cut -d: -f2)
FAMILY=$(echo "$COMBO"   | cut -d: -f3)
IDX_FIELD=$(echo "$COMBO" | cut -d: -f4)   # e.g. "run_3" or "perm_7"
IDX_NUM="${IDX_FIELD#*_}"                  # strip "run_" or "perm_"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
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
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: Cannot find conda. Activate $CONDA_ENV manually."
    exit 1
fi
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"

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
echo " Array task    : ${SLURM_ARRAY_TASK_ID:-0} (offset $OFFSET → combo index $COMBO_INDEX)"
echo " Mode          : $MODE"
echo " Combination   : $COMBO"
echo " Model         : $MODEL_T"
echo " Contrast      : $CONTRAST"
echo " Family        : $FAMILY"
echo " Index         : $IDX_NUM"
echo " Config        : $CONFIG"
echo " Host          : $(hostname)"
echo " CPUs          : ${SLURM_CPUS_PER_TASK:-?}"
echo " Conda env     : $CONDA_ENV"
echo " Started       : $(date)"
echo "========================================================"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if [ "$MODE" = "true" ]; then
    python run_loso_classification.py \
        --config     "$CONFIG" \
        --contrast   "$CONTRAST" \
        --family     "$FAMILY" \
        --model_type "$MODEL_T" \
        --run_idx    "$IDX_NUM" \
        --skip_permutation
else
    python run_loso_classification.py \
        --config     "$CONFIG" \
        --contrast   "$CONTRAST" \
        --family     "$FAMILY" \
        --model_type "$MODEL_T" \
        --perm_idx   "$IDX_NUM"
fi

EXIT_CODE=$?

echo "========================================================"
if [ $EXIT_CODE -ne 0 ]; then
    echo " FAILED: $MODEL_T | $CONTRAST | $FAMILY | $MODE $IDX_NUM (exit $EXIT_CODE)"
    exit $EXIT_CODE
else
    echo " SUCCESS: $MODEL_T | $CONTRAST | $FAMILY | $MODE $IDX_NUM"
    echo " Finished : $(date)"
fi
echo "========================================================"
