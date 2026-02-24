#!/bin/bash
# =============================================================================
# Parallel H5 to PKL Conversion using SLURM Job Arrays
# =============================================================================
# This script auto-detects the number of elements and submits as array job.
# Just run: bash batch_convert_h5_to_pkl_parallel.sh
# =============================================================================

set -euo pipefail

# Configuration
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/wandering-mind}
PIPELINE_CONFIG="${WORKDIR}/Analysis/eeg/junifer_markers/1.markers_h5_creation/pipeline_config.yaml"
DISCOVER_SCRIPT="${WORKDIR}/Analysis/eeg/junifer_markers/1.markers_h5_creation/discover_elements.py"
CONVERTER_SCRIPT="${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/h5_to_pkl_converter.py"
H5_FEATURES_DIR="${WORKDIR}/data/features/junifer"
PKL_FEATURES_DIR="${WORKDIR}/data/features"
THIS_SCRIPT="${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/batch_convert_h5_to_pkl_parallel.sh"

# Conda setup
CONDA_ENV=${CONDA_ENV:-junifer}

set +u
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV" 2>/dev/null || true
set -u

# Force overwrite mode
FORCE_OVERWRITE=true

# =============================================================================
# LAUNCHER MODE: If not inside SLURM, count elements and submit array job
# =============================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Counting total elements..."
    
    # Read descriptions from config
    DESCRIPTIONS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('${PIPELINE_CONFIG}'))['descriptions']))"))
    
    TOTAL=0
    for desc in "${DESCRIPTIONS[@]}"; do
        COUNT=$(python "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${desc}" --count)
        echo "  ${desc}: ${COUNT} elements"
        TOTAL=$((TOTAL + COUNT))
    done
    
    if [ $TOTAL -eq 0 ]; then
        echo "ERROR: No elements found!"
        exit 1
    fi
    
    echo "========================================="
    echo "Total elements to process: ${TOTAL}"
    echo "Submitting SLURM array job 1-${TOTAL}..."
    echo "========================================="
    
    # Create logs directory
    mkdir -p "${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/logs"
    
    # Submit as array job
    sbatch \
        --job-name=h5_to_pkl \
        --output="${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/logs/h5_to_pkl_%A_%a.out" \
        --error="${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/logs/h5_to_pkl_%A_%a.err" \
        --time=01:00:00 \
        --mem=8G \
        --cpus-per-task=1 \
        --array=1-${TOTAL} \
        "${THIS_SCRIPT}"
    
    exit 0
fi

# =============================================================================
# WORKER MODE: Running inside SLURM array job
# =============================================================================
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))

# Read descriptions from config
DESCRIPTIONS=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('${PIPELINE_CONFIG}'))['descriptions']))"))

# Find which description type and index within that type
CURRENT_OFFSET=0
FOUND=false

for desc in "${DESCRIPTIONS[@]}"; do
    COUNT=$(python "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${desc}" --count)
    
    if [ $TASK_ID -lt $((CURRENT_OFFSET + COUNT)) ]; then
        # This task belongs to this description type
        LOCAL_INDEX=$((TASK_ID - CURRENT_OFFSET))
        ELEMENT=$(python "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${desc}" --index "${LOCAL_INDEX}")
        FOUND=true
        break
    fi
    
    CURRENT_OFFSET=$((CURRENT_OFFSET + COUNT))
done

if [ "$FOUND" = false ]; then
    echo "ERROR: Task ID ${SLURM_ARRAY_TASK_ID} out of range"
    exit 1
fi

# Parse element
IFS=',' read -r subject session task desc <<< "${ELEMENT}"

echo "=========================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Processing: ${subject}, ${session}, ${task}, ${desc}"
if [ "$FORCE_OVERWRITE" = true ]; then
    echo "Mode: FORCE (overwriting existing files)"
fi
echo "=========================================="

# Create logs directory
mkdir -p "${WORKDIR}/Analysis/eeg/junifer_markers/2.h5_to_pkl/logs"

# Build H5 file pattern - Junifer names files like: element_sub-XX_ses-X_task-X_desc_markers.h5
H5_FILE="${H5_FEATURES_DIR}/element_${subject}_${session}_${task}_${desc}_markers.h5"

if [ ! -f "${H5_FILE}" ]; then
    echo "Warning: H5 file not found: ${H5_FILE}"
    echo "Skipping..."
    exit 0
fi

# Run converter
cd "${WORKDIR}"
if [ "$FORCE_OVERWRITE" = true ]; then
    python "${CONVERTER_SCRIPT}" \
        --h5-file "${H5_FILE}" \
        --output-dir "${PKL_FEATURES_DIR}" \
        --force
else
    python "${CONVERTER_SCRIPT}" \
        --h5-file "${H5_FILE}" \
        --output-dir "${PKL_FEATURES_DIR}"
fi

echo "Done: ${subject}, ${session}, ${task}, ${desc}"
