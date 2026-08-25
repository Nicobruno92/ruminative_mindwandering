#!/usr/bin/env bash
# =============================================================================
# run_cluster_worker.sh — Within-Subject MW Pipeline (SLURM worker)
# =============================================================================
# Arguments: $1 = config path, $2 = mode ("true" or "perm")
#            $3 = combination-index offset (for arrays chunked under MaxArraySize)
# DO NOT submit directly — use run_cluster.sh.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"
MODE="${2:-true}"
OFFSET="${3:-0}"
[[ "$CONFIG" != /* ]] && CONFIG="$SCRIPT_DIR/$CONFIG"

if [ "$MODE" = "true" ]; then
    COMBINATIONS_FILE="logs/.true_combinations.txt"
else
    COMBINATIONS_FILE="logs/.perm_combinations.txt"
fi

if [ ! -f "$COMBINATIONS_FILE" ]; then
    echo "ERROR: $COMBINATIONS_FILE not found. Run run_cluster.sh first."
    exit 1
fi

COMBO_INDEX=$((SLURM_ARRAY_TASK_ID + OFFSET))
COMBO=$(sed -n "$((COMBO_INDEX + 1))p" "$COMBINATIONS_FILE")
[ -z "$COMBO" ] && { echo "ERROR: No combination at index $COMBO_INDEX"; exit 1; }

MODEL_T=$(echo "$COMBO" | cut -d: -f1)
CONTRAST=$(echo "$COMBO" | cut -d: -f2)
FAMILY=$(echo "$COMBO"   | cut -d: -f3)
IDX_FIELD=$(echo "$COMBO" | cut -d: -f4)
IDX_NUM="${IDX_FIELD#*_}"

# ---------------------------------------------------------------------------
CONDA_ENV=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('slurm', {}).get('conda_env', 'ML'))
")

module load proxy 2>/dev/null || true

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda > /dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: Cannot find conda."; exit 1
fi
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export PYTHONUNBUFFERED=1

echo "========================================================"
echo " MW Within-Subject Pipeline — SLURM Worker"
echo "========================================================"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Array    : ${SLURM_ARRAY_TASK_ID:-0} (offset $OFFSET → combo index $COMBO_INDEX)"
echo " Mode     : $MODE"
echo " Combo    : $COMBO"
echo " Model    : $MODEL_T | Contrast: $CONTRAST | Family: $FAMILY | Idx: $IDX_NUM"
echo " CPUs     : ${SLURM_CPUS_PER_TASK:-?}"
echo " Started  : $(date)"
echo "========================================================"

if [ "$MODE" = "true" ]; then
    python run_within_subject_classification.py \
        --config     "$CONFIG" \
        --contrast   "$CONTRAST" \
        --family     "$FAMILY" \
        --model_type "$MODEL_T" \
        --run_idx    "$IDX_NUM" \
        --skip_permutation
else
    python run_within_subject_classification.py \
        --config     "$CONFIG" \
        --contrast   "$CONTRAST" \
        --family     "$FAMILY" \
        --model_type "$MODEL_T" \
        --perm_idx   "$IDX_NUM"
fi

EXIT_CODE=$?
echo "========================================================"
[ $EXIT_CODE -ne 0 ] && echo " FAILED: $MODE $IDX_NUM (exit $EXIT_CODE)" && exit $EXIT_CODE
echo " SUCCESS: $MODEL_T | $CONTRAST | $FAMILY | $MODE $IDX_NUM"
echo " Finished: $(date)"
echo "========================================================"
