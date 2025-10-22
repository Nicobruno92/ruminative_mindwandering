#!/bin/bash
#SBATCH --job-name=h5_to_pkl
#SBATCH --output=logs/h5_to_pkl_%j.out
#SBATCH --error=logs/h5_to_pkl_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# =============================================================================
# Batch convert H5 + FIF files to PKL format with BIDS structure
# =============================================================================
# This script reads the elements file and processes each subject/task/desc
# combination, converting Junifer H5 markers + MNE FIF metadata to PKL files
# 
# By default, OVERWRITES existing PKL files (force mode)
# 
# Output structure:
#   BIDS_ROOT/features/sub-XX/eeg/junifer/sub-XX_task-SartX_desc-[evoked,state]_markers.pkl
# =============================================================================

# Configuration
FORCE_OVERWRITE=true  # Set to false to skip existing files
BIDS_ROOT="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS"
LOCAL_ROOT="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
DERIVATIVES_DIR="${BIDS_ROOT}/derivatives"
H5_FEATURES_DIR="${BIDS_ROOT}/features/junifer"
PKL_FEATURES_DIR="${BIDS_ROOT}/features"
ELEMENTS_FILE="${LOCAL_ROOT}/junifer_markers/1.markers_h5_creation/elements.csv"
CONVERTER_SCRIPT="${LOCAL_ROOT}/junifer_markers/2.h5_to_pkl/h5_to_pkl_converter.py"

# Description types to process
DESC_TYPES=("evoked" "state")

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate appropriate environment
echo "Activating environment..."
source activate junifer

# Check if converter script exists
if [ ! -f "${CONVERTER_SCRIPT}" ]; then
    echo "Error: Converter script not found: ${CONVERTER_SCRIPT}"
    exit 1
fi

# Check if elements file exists
if [ ! -f "${ELEMENTS_FILE}" ]; then
    echo "Error: Elements file not found: ${ELEMENTS_FILE}"
    exit 1
fi

# Initialize counters
total=0
success=0
failed=0
skipped=0

echo "=========================================="
echo "Starting batch H5 to PKL conversion"
if [ "$FORCE_OVERWRITE" = true ]; then
    echo "Mode: FORCE (overwriting existing files)"
else
    echo "Mode: SKIP (skipping existing files)"
fi
echo "=========================================="
echo "BIDS ROOT: ${BIDS_ROOT}"
echo "Elements file: ${ELEMENTS_FILE}"
echo "Converter script: ${CONVERTER_SCRIPT}"
echo ""

# Read elements CSV file and process each line (skip header)
while IFS=',' read -r subject task desc; do
    # Skip empty lines
    if [ -z "${subject}" ] || [ -z "${task}" ] || [ -z "${desc}" ]; then
        continue
    fi
    
    # Remove any whitespace
    subject=$(echo "${subject}" | xargs)
    task=$(echo "${task}" | xargs)
    desc=$(echo "${desc}" | xargs)
    
    echo "Processing: ${subject}, ${task}, ${desc}"
    
    # Process the specific description from CSV
    {
        # Note: desc is now read from CSV, not looped
    ((total++))
    
    # Construct file paths
    fif_file="${DERIVATIVES_DIR}/${subject}/eeg/${subject}_task-${task}_desc-${desc}_epo.fif"
    h5_file="${H5_FEATURES_DIR}/element_${subject}_${task}_${desc}_markers.h5"
    pkl_file="${PKL_FEATURES_DIR}/${subject}/eeg/junifer/${subject}_task-${task}_desc-${desc}_markers.pkl"
    
    # Check if input files exist
    if [ ! -f "${fif_file}" ]; then
        echo "  ⚠️  FIF file not found: ${fif_file}"
        ((failed++))
        continue
    fi
    
    if [ ! -f "${h5_file}" ]; then
        echo "  ⚠️  H5 file not found: ${h5_file}"
        ((failed++))
        continue
    fi
    
    # Check if output already exists (only skip if force mode is off)
    if [ -f "${pkl_file}" ] && [ "$FORCE_OVERWRITE" = false ]; then
        echo "  ℹ️  PKL file already exists, skipping: ${pkl_file}"
        ((skipped++))
        continue
    fi
    
    # Notify if overwriting
    if [ -f "${pkl_file}" ] && [ "$FORCE_OVERWRITE" = true ]; then
        echo "  🔄  Overwriting existing PKL file"
    fi
    
    # Create output directory
    mkdir -p "$(dirname "${pkl_file}")"
    
    # Get input file sizes
    fif_size=$(du -h "${fif_file}" | cut -f1)
    h5_size=$(du -h "${h5_file}" | cut -f1)
    
    # Run conversion
    echo "  Converting: ${subject}_task-${task}_desc-${desc}"
    echo "    FIF: ${fif_file} (${fif_size})"
    echo "    H5:  ${h5_file} (${h5_size})"
    echo "    PKL: ${pkl_file}"
    
    if python "${CONVERTER_SCRIPT}" "${fif_file}" "${h5_file}" "${pkl_file}"; then
        # Get output file size
        pkl_size=$(du -h "${pkl_file}" | cut -f1)
        echo "  ✓  Success - PKL size: ${pkl_size}"
        ((success++))
    else
        echo "  ❌  Failed"
        ((failed++))
    fi
    echo ""
    }
    
done < <(tail -n +2 "${ELEMENTS_FILE}")

# Print summary
echo "=========================================="
echo "Conversion Summary"
echo "=========================================="
echo "Total elements processed: ${total}"
echo "Successful: ${success}"
echo "Failed: ${failed}"
echo "Skipped (already exist): ${skipped}"
echo "=========================================="

# Exit with error if any failed
if [ ${failed} -gt 0 ]; then
    exit 1
else
    exit 0
fi
