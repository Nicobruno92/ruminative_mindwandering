#!/bin/bash
# Quick test script to verify optimizations are working

echo "========================================"
echo "Testing PKL Creation Performance"
echo "========================================"
echo ""

# Activate conda environment
echo "Activating junifer conda environment..."
source ~/.bashrc
conda activate junifer || {
    echo "ERROR: Could not activate junifer conda environment"
    echo "Please ensure conda is configured and junifer env exists"
    exit 1
}

echo "Using Python: $(which python3)"
echo ""

# Configuration
TEST_ELEMENT="sub-02"
TEST_TASK="Sart1"
TEST_DESC="evoked"

H5_PATH="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/junifer/element_${TEST_ELEMENT}_${TEST_TASK}_${TEST_DESC}_markers.h5"
FIF_PATH="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives/${TEST_ELEMENT}/eeg/${TEST_ELEMENT}_task-${TEST_TASK}_desc-${TEST_DESC}_epo.fif"
OUTPUT_PATH="/tmp/test_${TEST_ELEMENT}_${TEST_TASK}_${TEST_DESC}_markers.pkl"

echo "Test files:"
echo "  H5:  ${H5_PATH}"
echo "  FIF: ${FIF_PATH}"
echo "  OUT: ${OUTPUT_PATH}"
echo ""

# Check files exist
if [ ! -f "${H5_PATH}" ]; then
    echo "ERROR: H5 file not found!"
    exit 1
fi

if [ ! -f "${FIF_PATH}" ]; then
    echo "ERROR: FIF file not found!"
    exit 1
fi

echo "Running PKL creation (should take ~3 seconds)..."
echo ""

# Time the execution
START_TIME=$(date +%s)

python3 -c "
from create_pkl_from_h5_fif import create_pkl_from_h5_fif
create_pkl_from_h5_fif(
    '${H5_PATH}',
    '${FIF_PATH}',
    '${OUTPUT_PATH}',
    {'subject': '${TEST_ELEMENT}', 'task': '${TEST_TASK}', 'desc': '${TEST_DESC}'}
)
"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS!"
    echo "   Time: ${ELAPSED} seconds"
    
    if [ ${ELAPSED} -le 5 ]; then
        echo "   🚀 EXCELLENT! Optimizations are working!"
    elif [ ${ELAPSED} -le 10 ]; then
        echo "   ✓ Good! Much better than 20 minutes."
    else
        echo "   ⚠️  Slower than expected. Check for issues."
    fi
    
    # Check output file
    if [ -f "${OUTPUT_PATH}" ]; then
        SIZE=$(du -h "${OUTPUT_PATH}" | cut -f1)
        echo "   Output: ${OUTPUT_PATH} (${SIZE})"
    fi
else
    echo "❌ FAILED with exit code ${EXIT_CODE}"
    echo "   Check error messages above"
fi
echo "========================================"
