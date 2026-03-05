#!/usr/bin/env bash
# =============================================================================
# run_local.sh — MW Classification Pipeline (local/sequential)
# =============================================================================
# Runs all combinations of (model × contrasts × feature families) defined in
# config.yaml, sequentially. All parameters come from config.yaml — no
# redundant settings here.
#
# USAGE:
#   bash run_local.sh                    # uses config.yaml
#   bash run_local.sh path/to/config.yaml
#   bash run_local.sh config.yaml --skip-permutation
#
# =============================================================================

set -euo pipefail

CONFIG="${1:-config.yaml}"
SKIP_PERM_FLAG=""
if [[ "${2:-}" == "--skip-permutation" ]]; then
    SKIP_PERM_FLAG="--skip_permutation"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Validate config
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Read all parameters and build combination list from config via Python
# ---------------------------------------------------------------------------
read -r CONDA_ENV MODEL COMBINATIONS <<< "$(python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

conda_env = cfg.get('slurm', {}).get('conda_env', 'ML')
model = cfg.get('model_type', 'lr')
contrasts = list(cfg.get('label_contrasts', {}).keys())
families = cfg.get('run_families', ['all'])

combos = [f'{model}:{c}:{fam}' for c in contrasts for fam in families]
print(conda_env, model, ','.join(combos))
print(f'Combos: {len(combos)} ({model} x {len(contrasts)} contrasts x {len(families)} families)', file=sys.stderr)
")"

# Print header
echo "==========================================="
echo " MW Classification Pipeline — Local Run"
echo "==========================================="
echo " Config  : $CONFIG"
echo " Env     : $CONDA_ENV"
echo " Model   : $MODEL"
echo " Started : $(date)"
echo "==========================================="
echo ""

# Run each combination
IFS=',' read -ra COMBO_ARRAY <<< "$COMBINATIONS"
TOTAL=${#COMBO_ARRAY[@]}
IDX=0

for COMBO in "${COMBO_ARRAY[@]}"; do
    MODEL_T=$(echo "$COMBO" | cut -d: -f1)
    CONTRAST=$(echo "$COMBO" | cut -d: -f2)
    FAMILY=$(echo "$COMBO" | cut -d: -f3)
    IDX=$((IDX + 1))

    LOG_FILE="${LOG_DIR}/local_${CONTRAST}_${FAMILY}_${MODEL_T}_${TIMESTAMP}.log"

    echo "[$IDX/$TOTAL] Model=$MODEL_T | Contrast=$CONTRAST | Family=$FAMILY"
    echo "  → $LOG_FILE"

    conda run -n "$CONDA_ENV" --no-capture-output \
        python run_loso_classification.py \
            --config "$CONFIG" \
            --contrast "$CONTRAST" \
            --family  "$FAMILY" \
            --model_type "$MODEL_T" \
            $SKIP_PERM_FLAG \
    2>&1 | tee "$LOG_FILE"

    echo ""
done

echo "==========================================="
echo " All $TOTAL combinations done."
echo " Finished : $(date)"
echo "==========================================="
