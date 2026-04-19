#!/bin/bash
# =============================================================================
# SUBMIT PREDICTOR LOOP
# =============================================================================
# Reads lmm.predictor_of_interest from config.yaml.
# When the value is a list, submits one independent SLURM job-set
# (markers array + MCC + report) per predictor via submit_parallel_markers.sh.
# When the value is a single string, forwards straight to submit_parallel_markers.sh.
#
# Usage:
#   bash Statistics/submit_predictor_loop.sh [--config <path>]
#
# Options:
#   --config <path>   Path to config yaml (default: Statistics/config.yaml)
#
# Each predictor produces results in its own directory:
#   results/lmm_cluster/{fixed_effects}__target_{predictor}/
#
# Monitor submitted jobs with: squeue -u $USER
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
CONFIG_FILE="Statistics/config.yaml"
DEPENDENCY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash Statistics/submit_predictor_loop.sh [--config <path>] [--dependency <job_id>]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

module load proxy 2>/dev/null || true

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate eeg
fi

SCRIPT_DIR="Statistics"

# ---------------------------------------------------------------------------
# Read predictor list from config
# ---------------------------------------------------------------------------
echo "=========================================="
echo "LMM Predictor Loop Submission"
echo "Config: ${CONFIG_FILE}"
echo "=========================================="

PREDICTORS=$(python -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
poi = config['lmm'].get('predictor_of_interest', 'auto')
if isinstance(poi, str):
    print(poi)
else:
    for p in poi:
        print(p)
")

if [ -z "${PREDICTORS}" ]; then
    echo "ERROR: Could not read predictor_of_interest from ${CONFIG_FILE}"
    exit 1
fi

N_PREDICTORS=$(echo "${PREDICTORS}" | wc -l)
echo "Found ${N_PREDICTORS} predictor(s) to process:"
echo "${PREDICTORS}" | while read P; do echo "  - ${P}"; done
echo ""

# ---------------------------------------------------------------------------
# Submit one job-set per predictor
# ---------------------------------------------------------------------------
SUBMITTED=()

while IFS= read -r PREDICTOR; do
    [[ -z "${PREDICTOR}" ]] && continue
    echo "------------------------------------------"
    echo "Submitting jobs for predictor: ${PREDICTOR}"
    echo "------------------------------------------"
    
    CMD="bash ${SCRIPT_DIR}/submit_parallel_markers.sh --predictor \"${PREDICTOR}\""
    if [ -n "${DEPENDENCY}" ]; then
        CMD="${CMD} --dependency \"${DEPENDENCY}\""
    fi
    eval ${CMD}
    
    SUBMITTED+=("${PREDICTOR}")
    echo ""
done <<< "${PREDICTORS}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "All predictor submissions complete."
echo "Submitted ${#SUBMITTED[@]} predictor(s):"
for P in "${SUBMITTED[@]}"; do
    OUTPUT_PATH=$(python -c "
import yaml
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
print(config['project']['output_path'])
")
    FIXED=$(python -c "
import yaml, sys
sys.path.append('${SCRIPT_DIR}')
from helpers import get_model_folder_name
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
print(get_model_folder_name(config['lmm']['formula'], '${P}'))
")
    echo "  [${P}]  →  ${OUTPUT_PATH}/${FIXED}/"
done
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
