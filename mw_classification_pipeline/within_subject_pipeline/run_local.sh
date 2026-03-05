#!/bin/bash
# ==============================================================================
# LOCAL Within-Subject Mind-Wandering Classification Runner
# ==============================================================================
# This script runs the within-subject pipeline locally for all families defined
# in config.yaml sequentially.
# It enforces strict log capturing and reproducibility.
# ==============================================================================

# Parse arguments
DRY_RUN=false
MODEL="rf"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --model) MODEL="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ------------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PIPELINE_DIR}" || exit 1

CONFIG_FILE="${PIPELINE_DIR}/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Ensure output directory exists (parse from YAML)
RESULTS_ROOT=$(awk -F '"' '/results_dir:/ {print $2}' "$CONFIG_FILE")
if [ -z "$RESULTS_ROOT" ]; then
    # Fallback to single quotes if double quotes weren't used
    RESULTS_ROOT=$(awk -F "'" '/results_dir:/ {print $2}' "$CONFIG_FILE")
fi
if [ -z "$RESULTS_ROOT" ]; then
    RESULTS_ROOT="results/MW_Classification/WithinSubject"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PIPELINE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------------------------
# PARSE YAML FOR RUN_FAMILIES
# ------------------------------------------------------------------------------
get_yq() {
    # If standard yq is not installed, fallback to python
    python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    cfg = yaml.safe_load(f)
for fam in cfg.get('run_families', []):
    print(fam)
"
}

FAMILIES=$(get_yq)
if [ -z "$FAMILIES" ]; then
    echo "ERROR: Could not parse run_families from config.yaml"
    exit 1
fi

echo "============================================================================="
echo " WITHIN-SUBJECT MIND-WANDERING CLASSIFICATION PIPELINE (LOCAL)"
echo "============================================================================="
echo " Time         : $(date)"
echo " Config       : ${CONFIG_FILE}"
echo " Model Target : ${MODEL}"
echo " Target Fams  : $(echo ${FAMILIES} | tr '\n' ' ')"
echo " Dry Run      : ${DRY_RUN}"
echo "============================================================================="

# ------------------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------------------
for FAMILY in ${FAMILIES}; do
    echo -e "\n-----------------------------------------------------------------------------"
    echo " Running Family: ${FAMILY}"
    echo "-----------------------------------------------------------------------------"
    
    LOG_FILE="${LOG_DIR}/ws_${FAMILY}_${MODEL}_${TIMESTAMP}.log"
    CMD="python run_within_subject_classification.py --config config.yaml --family ${FAMILY} --model_type ${MODEL}"
    
    if [ "$DRY_RUN" = true ]; then
        CMD="${CMD} --dry_run"
        echo " [DRY RUN EXECUTION]: $CMD"
        $CMD 2>&1 | tee "${LOG_FILE}"
    else
        echo " Command: $CMD"
        echo " Logging to: ${LOG_FILE}"
        # Run with strictly unbuffered output to tee
        PYTHONUNBUFFERED=1 $CMD 2>&1 | tee "${LOG_FILE}"
        
        # Check exit status
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
             echo " ERROR: Pipeline failed for family ${FAMILY}. Check ${LOG_FILE}"
             # Continue to next family rather than failing completely
        fi
    fi
done

echo -e "\n============================================================================="
echo " ALL LOCAL WITHIN-SUBJECT PIPELINES COMPLETED."
echo "============================================================================="
