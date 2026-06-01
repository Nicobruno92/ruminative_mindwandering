#!/bin/bash
# Local test for Junifer pipeline
set -euo pipefail

PROJECT_DIR="/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
DERIVATIVES_DIR="/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"
H5_CREATION_DIR="${PROJECT_DIR}/junifer_markers/1.markers_h5_creation"

echo "==========================================="
echo "Local Junifer Pipeline Test"
echo "==========================================="
cd "$H5_CREATION_DIR"

# Create local config
cat > pipeline_config_local.yaml << EOF
datadir: /Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives
pattern: "{subject}/eeg/{subject}_task-{task}_desc-{desc}_epo.fif"
subjects: []
tasks: []
descriptions:
  - evoked
  - state
  - sleep
EOF

echo "[STEP 1] Testing element discovery..."
for desc in evoked state sleep; do
    COUNT=$(python discover_elements.py --config pipeline_config_local.yaml --desc "$desc" --count 2>/dev/null || echo "0")
    echo "  desc=$desc: $COUNT elements"
done

echo ""
echo "[STEP 2] Validating configs..."
for f in config_state.yaml config_evoked.yaml config_sleep.yaml; do
    python -c "import yaml; yaml.safe_load(open('$f'))" 2>/dev/null && echo "✓ $f" || echo "❌ $f"
done

echo ""
echo "[STEP 3] First element test..."
FIRST=$(python discover_elements.py --config pipeline_config_local.yaml --desc evoked --index 0 2>/dev/null || echo "")
if [ -n "$FIRST" ]; then
    echo "First element: $FIRST"
    SUBJECT=$(echo "$FIRST" | cut -d',' -f1)
    TASK=$(echo "$FIRST" | cut -d',' -f2)
    EPOCH_FILE="${DERIVATIVES_DIR}/${SUBJECT}/eeg/${SUBJECT}_task-${TASK}_desc-evoked_epo.fif"
    [ -f "$EPOCH_FILE" ] && echo "✓ Epoch file exists" || echo "❌ File not found"
else
    echo "❌ No elements found"
fi

echo ""
echo "==========================================="
echo "Done! To run junifer:"
echo "cd $H5_CREATION_DIR"
echo "junifer run config_evoked.yaml --element \"$FIRST\""
