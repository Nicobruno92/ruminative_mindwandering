#!/bin/bash
# =============================================================================
# SUBMIT MODERATION LOOP
# =============================================================================
# Reads moderation.moderators from config_moderation.yaml.
# Submits one independent SLURM job-set (markers array + results) per moderator
# via submit_parallel_markers.sh, with auto-generated interaction formulas.
#
# After all moderator jobs complete, runs cross-moderator FDR correction.
#
# Usage:
#   bash Statistics/submit_moderation_loop.sh [--config <path>]
#
# Options:
#   --config <path>   Path to moderation config yaml (default: Statistics/config_moderation.yaml)
#
# Each moderator produces results in its own directory:
#   results/lmm_cluster/{fixed_effects}__target_{onoff:moderator}/
#
# Monitor submitted jobs with: squeue -u $USER
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
MOD_CONFIG="Statistics/config_moderation.yaml"
BASE_CONFIG="Statistics/config.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            MOD_CONFIG="$2"
            shift 2
            ;;
        --base-config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash Statistics/submit_moderation_loop.sh [--config <path>] [--base-config <path>]"
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
# Read moderators and base predictor from config
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Moderation Analysis Loop Submission"
echo "Config: ${MOD_CONFIG}"
echo "Base config: ${BASE_CONFIG}"
echo "=========================================="

MODERATORS=$(python -c "
import yaml
with open('${MOD_CONFIG}') as f:
    config = yaml.safe_load(f)
for m in config['moderation']['moderators']:
    print(m)
")

BASE_PREDICTOR=$(python -c "
import yaml
with open('${MOD_CONFIG}') as f:
    config = yaml.safe_load(f)
print(config['moderation']['base_predictor'])
")

if [ -z "${MODERATORS}" ]; then
    echo "ERROR: Could not read moderators from ${MOD_CONFIG}"
    exit 1
fi

N_MODERATORS=$(echo "${MODERATORS}" | wc -l)
echo "Base predictor: ${BASE_PREDICTOR}"
echo "Found ${N_MODERATORS} moderator(s) to process:"
echo "${MODERATORS}" | while read M; do echo "  - ${M}"; done
echo ""

# ---------------------------------------------------------------------------
# Submit one job-set per moderator
# ---------------------------------------------------------------------------
SUBMITTED=()
ALL_JIDS=""

while IFS= read -r MODERATOR; do
    [[ -z "${MODERATOR}" ]] && continue
    
    # Build formula and interaction term
    FORMULA="power ~ ${BASE_PREDICTOR} * ${MODERATOR} + (1|subject)"
    INTERACTION="${BASE_PREDICTOR}:${MODERATOR}"
    
    echo "------------------------------------------"
    echo "Submitting jobs for moderator: ${MODERATOR}"
    echo "  Formula: ${FORMULA}"
    echo "  Predictor of interest: ${INTERACTION}"
    echo "------------------------------------------"
    
    # Create temporary config in project directory (not /tmp for cluster compatibility)
    TMP_CONFIG="${SCRIPT_DIR}/.tmp_mod_config_${MODERATOR}.yaml"
    python -c "
import yaml
import sys

try:
    # Load base config
    with open('${BASE_CONFIG}') as f:
        config = yaml.safe_load(f)

    # Load moderation config for overrides
    with open('${MOD_CONFIG}') as f:
        mod_config = yaml.safe_load(f)

    # Apply moderation-specific LMM settings
    if 'lmm' in mod_config:
        for key, val in mod_config['lmm'].items():
            if key not in ['formula', 'predictor_of_interest']:
                config['lmm'][key] = val

    # Apply clustering overrides if present
    if 'clustering' in mod_config and mod_config['clustering'] is not None:
        for key, val in mod_config['clustering'].items():
            config['clustering'][key] = val

    # Set formula and predictor for this moderator
    config['lmm']['formula'] = '${FORMULA}'
    config['lmm']['predictor_of_interest'] = '${INTERACTION}'

    # Write temporary config
    with open('${TMP_CONFIG}', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print('Config written to ${TMP_CONFIG}')
except Exception as e:
    print(f'ERROR creating config: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [ ! -f "${TMP_CONFIG}" ]; then
        echo "  ✗ Failed to create temporary config for '${MODERATOR}'"
        continue
    fi
    
    # Submit via the standard parallel markers script
    # Capture all output to parse job IDs
    SUBMIT_OUTPUT=$(bash ${SCRIPT_DIR}/submit_parallel_markers.sh \
        --config "${TMP_CONFIG}" \
        --predictor "${INTERACTION}" 2>&1)
    
    echo "${SUBMIT_OUTPUT}"
    
    # Extract all job IDs from output (array job, MCC job, report job)
    JIDS=$(echo "${SUBMIT_OUTPUT}" | grep -oP 'Submitted batch job \K[0-9]+' | tr '\n' ':')
    
    if [ -n "${JIDS}" ]; then
        # Remove trailing ':'
        JIDS="${JIDS%:}"
        echo "  ✓ Submitted jobs ${JIDS} for moderator '${MODERATOR}'"
        ALL_JIDS="${ALL_JIDS}:${JIDS}"
        SUBMITTED+=("${MODERATOR}")
    else
        echo "  ✗ Failed to submit jobs for moderator '${MODERATOR}'"
    fi
    
    # Clean up temporary config
    rm -f "${TMP_CONFIG}"
    
    echo ""
done <<< "${MODERATORS}"

# ---------------------------------------------------------------------------
# Submit cross-moderator FDR summary job
# ---------------------------------------------------------------------------
if [ ${#SUBMITTED[@]} -gt 0 ]; then
    echo "------------------------------------------"
    echo "Submitting cross-moderator FDR summary job"
    echo "------------------------------------------"
    
    # Remove leading ':'
    DEP_JIDS="${ALL_JIDS#:}"
    
    if [ -n "${DEP_JIDS}" ]; then
        FDR_JID=$(sbatch \
            --parsable \
            --job-name="mod_fdr_summary" \
            --dependency="afterok:${DEP_JIDS}" \
            --time=00:30:00 \
            --mem=8G \
            --cpus-per-task=1 \
            --output="logs/moderation_fdr_summary_%j.out" \
            --error="logs/moderation_fdr_summary_%j.err" \
            --wrap="
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
module load proxy 2>/dev/null || true
source \$HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source \$HOME/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate eeg
python ${SCRIPT_DIR}/run_moderation_pipeline.py \
    --config ${MOD_CONFIG} \
    --base-config ${BASE_CONFIG}
")
        echo "  ✓ FDR summary job ${FDR_JID} will run after all moderator jobs complete."
    else
        echo "  ⚠ No jobs to wait for, skipping FDR summary."
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "All moderator submissions complete."
echo "Submitted ${#SUBMITTED[@]} moderator(s):"
for M in "${SUBMITTED[@]}"; do
    INTERACTION="${BASE_PREDICTOR}:${M}"
    OUTPUT_PATH=$(python -c "
import yaml
with open('${BASE_CONFIG}') as f:
    config = yaml.safe_load(f)
print(config['project']['output_path'])
")
    FIXED=$(python -c "
import yaml, sys
sys.path.append('${SCRIPT_DIR}')
from helpers import get_model_folder_name
print(get_model_folder_name('power ~ ${BASE_PREDICTOR} * ${M} + (1|subject)', '${INTERACTION}'))
")
    echo "  [${M}]  →  ${OUTPUT_PATH}/${FIXED}/"
done
echo "Monitor with: squeue -u \$USER"
echo "=========================================="
