#!/bin/bash
#SBATCH --job-name=h5_to_pkl_parallel
#SBATCH --output=logs/h5_to_pkl_%A_%a.out
#SBATCH --error=logs/h5_to_pkl_%A_%a.err
#SBATCH --array=1-332%20
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# =============================================================================
# Parallel H5 to PKL Conversion using SLURM Job Arrays
# =============================================================================
# This script uses SLURM job arrays to process multiple elements in parallel
# Each array task processes ONE element (subject + task + desc combination)
# 
# --array=1-332%20 means:
#   - 332 total tasks (one per line in elements file)
#   - Maximum 20 running simultaneously (%20)
#
# Elements file format: subject,task,desc (e.g., sub-31,Sart4,evoked)
# Adjust %20 to control parallelism (e.g., %30 for 30 parallel jobs)
# =============================================================================

# Configuration
BIDS_ROOT="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS"
LOCAL_ROOT="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
DERIVATIVES_DIR="${BIDS_ROOT}/derivatives"
H5_FEATURES_DIR="${BIDS_ROOT}/features/junifer"
PKL_FEATURES_DIR="${BIDS_ROOT}/features"
ELEMENTS_FILE="${LOCAL_ROOT}/junifer_markers/1.markers_h5_creation/elements.csv"
CONVERTER_SCRIPT="${LOCAL_ROOT}/junifer_markers/2.h5_to_pkl/h5_to_pkl_converter.py"

# Force overwrite mode
FORCE_OVERWRITE=true

# Get the line corresponding to this array task (skip header, so add 1)
# Array task IDs start at 1, and we need to skip the header line
LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
ELEMENT_LINE=$(sed -n "${LINE_NUM}p" "${ELEMENTS_FILE}")

if [ -z "${ELEMENT_LINE}" ]; then
    echo "Error: Could not read line ${SLURM_ARRAY_TASK_ID} from ${ELEMENTS_FILE}"
    exit 1
fi

# Parse subject, task, and desc from the line (3 columns)
# Remove any carriage returns (CRLF) first
ELEMENT_LINE="${ELEMENT_LINE//$'\r'/}"
# Parse using IFS - use printf to avoid here-string (no temp file needed)
IFS=',' read -r subject task desc < <(printf '%s\n' "${ELEMENT_LINE}")
# Trim leading/trailing whitespace only
subject="${subject#"${subject%%[![:space:]]*}"}"
subject="${subject%"${subject##*[![:space:]]}"}"
task="${task#"${task%%[![:space:]]*}"}"
task="${task%"${task##*[![:space:]]}"}"
desc="${desc#"${desc%%[![:space:]]*}"}"
desc="${desc%"${desc##*[![:space:]]}"}"

# Validate that all fields were parsed correctly
if [ -z "${subject}" ] || [ -z "${task}" ] || [ -z "${desc}" ]; then
    echo "=========================================="
    echo "ERROR: Failed to parse element line ${SLURM_ARRAY_TASK_ID}"
    echo "=========================================="
    echo "Raw line: [${ELEMENT_LINE}]"
    echo "Parsed values:"
    echo "  subject: [${subject}]"
    echo "  task: [${task}]"
    echo "  desc: [${desc}]"
    echo "Line hex dump:"
    echo "${ELEMENT_LINE}" | od -c
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "Processing: ${subject}, ${task}, ${desc}"
if [ "$FORCE_OVERWRITE" = true ]; then
    echo "Mode: FORCE (overwriting existing files)"
fi
echo "=========================================="

# Activate appropriate environment
source activate junifer

# Initialize counters for this task
success=0
failed=0

# Process the single desc from the elements file
echo ""
echo "Processing desc=${desc}..."

# Construct file paths
fif_file="${DERIVATIVES_DIR}/${subject}/eeg/${subject}_task-${task}_desc-${desc}_epo.fif"
h5_file="${H5_FEATURES_DIR}/element_${subject}_${task}_${desc}_markers.h5"
pkl_file="${PKL_FEATURES_DIR}/${subject}/eeg/junifer/${subject}_task-${task}_desc-${desc}_markers.pkl"

# Check if input files exist
if [ ! -f "${fif_file}" ]; then
    echo "  ⚠️  FIF file not found: ${fif_file}"
    ((failed++))
else
    if [ ! -f "${h5_file}" ]; then
        echo "  ⚠️  H5 file not found: ${h5_file}"
        ((failed++))
    else
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
    fi
fi

# Print summary for this task
echo ""
echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} Summary"
echo "=========================================="
echo "Element: ${subject}, ${task}, ${desc}"
echo "Successful: ${success}/1"
echo "Failed: ${failed}/1"
echo "=========================================="

# Exit with error if any failed
if [ ${failed} -gt 0 ]; then
    exit 1
else
    exit 0
fi
