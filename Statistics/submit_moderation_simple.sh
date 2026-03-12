#!/bin/bash
# =============================================================================
# SIMPLE MODERATION SUBMISSION
# =============================================================================
# Simplified version that submits all moderators sequentially without
# complex job tracking. Use this when the cluster is busy.
#
# Usage:
#   bash Statistics/submit_moderation_simple.sh
#
# Monitor: squeue -u $USER
# =============================================================================

set -e

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

module load proxy 2>/dev/null || true

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
fi

SCRIPT_DIR="Statistics"
MOD_CONFIG="${SCRIPT_DIR}/config_moderation.yaml"
BASE_CONFIG="${SCRIPT_DIR}/config.yaml"

echo "=========================================="
echo "SIMPLE MODERATION SUBMISSION"
echo "=========================================="

# Read moderators
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

echo "Base predictor: ${BASE_PREDICTOR}"
echo "Moderators: $(echo ${MODERATORS} | tr '\n' ' ')"
echo ""

# Submit each moderator
while IFS= read -r MOD; do
    [[ -z "${MOD}" ]] && continue
    
    FORMULA="power ~ ${BASE_PREDICTOR} * ${MOD} + (1|subject)"
    INTERACTION="${BASE_PREDICTOR}:${MOD}"
    
    echo "------------------------------------------"
    echo "Moderator: ${MOD}"
    echo "Formula: ${FORMULA}"
    echo "Interaction: ${INTERACTION}"
    echo "------------------------------------------"
    
    # Create temp config
    TMP_CONFIG="${SCRIPT_DIR}/.tmp_mod_${MOD}.yaml"
    python -c "
import yaml
with open('${BASE_CONFIG}') as f:
    config = yaml.safe_load(f)
with open('${MOD_CONFIG}') as f:
    mod_config = yaml.safe_load(f)
if 'lmm' in mod_config:
    for key, val in mod_config['lmm'].items():
        if key not in ['formula', 'predictor_of_interest']:
            config['lmm'][key] = val
if 'clustering' in mod_config and mod_config['clustering']:
    for key, val in mod_config['clustering'].items():
        config['clustering'][key] = val
config['lmm']['formula'] = '${FORMULA}'
config['lmm']['predictor_of_interest'] = '${INTERACTION}'
with open('${TMP_CONFIG}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"
    
    # Submit
    bash ${SCRIPT_DIR}/submit_parallel_markers.sh \
        --config "${TMP_CONFIG}" \
        --predictor "${INTERACTION}"
    
    # Cleanup
    rm -f "${TMP_CONFIG}"
    
    echo ""
    sleep 2
    
done <<< "${MODERATORS}"

echo "=========================================="
echo "All moderators submitted"
echo "Monitor with: squeue -u \$USER"
echo ""
echo "After all jobs complete, run FDR correction:"
echo "  python Statistics/run_moderation_pipeline.py"
echo "=========================================="
