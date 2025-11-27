#!/bin/bash

# =============================================================================
# Launch Parallel H5 to PKL Conversion
# =============================================================================
# This script launches the parallel processing using SLURM job arrays
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Cancel any existing h5_to_pkl jobs
echo "Checking for existing h5_to_pkl jobs..."
existing_jobs=$(squeue -n h5_to_pkl_parallel -u $USER -h -o "%A" 2>/dev/null)
if [ ! -z "${existing_jobs}" ]; then
    echo "Cancelling existing jobs: ${existing_jobs}"
    scancel -n h5_to_pkl_parallel -u $USER
    sleep 2
else
    echo "No existing jobs found."
fi

# Submit the parallel job array
echo ""
echo "=========================================="
echo "Launching Parallel H5 to PKL Conversion"
echo "=========================================="

# Determine number of elements from master elements.csv
LOCAL_ROOT="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
ELEMENTS_FILE="${LOCAL_ROOT}/junifer_markers/1.markers_h5_creation/elements.csv"

if [ ! -f "${ELEMENTS_FILE}" ]; then
    echo "Error: Elements file not found: ${ELEMENTS_FILE}"
    exit 1
fi

total_elements=$(tail -n +2 "${ELEMENTS_FILE}" | wc -l)
if [ "${total_elements}" -le 0 ]; then
    echo "Error: No elements found in ${ELEMENTS_FILE}"
    exit 1
fi

ARRAY_RANGE="1-${total_elements}%20"

echo "Script: batch_convert_h5_to_pkl_parallel.sh"
echo "Jobs: ${total_elements} array tasks (one per element)"
echo "Parallelism: Max 20 simultaneous jobs"
echo "Time per job: 1 hour"
echo "Memory per job: 8GB"
echo "Array range: ${ARRAY_RANGE}"
echo ""

JOB_ID=$(sbatch --parsable --array=${ARRAY_RANGE} batch_convert_h5_to_pkl_parallel.sh)

if [ $? -eq 0 ]; then
    echo "✓ Job array submitted successfully!"
    echo "  Job ID: ${JOB_ID}"
    echo "  Array tasks: 1-${total_elements}"
    echo ""
    echo "=========================================="
    echo "Monitoring Commands"
    echo "=========================================="
    echo "Check job status:"
    echo "  squeue -j ${JOB_ID}"
    echo ""
    echo "Check all array tasks:"
    echo "  squeue -j ${JOB_ID} -r"
    echo ""
    echo "Count running/pending tasks:"
    echo "  squeue -j ${JOB_ID} -t RUNNING | wc -l"
    echo "  squeue -j ${JOB_ID} -t PENDING | wc -l"
    echo ""
    echo "Watch progress in real-time:"
    echo "  watch -n 5 'squeue -j ${JOB_ID} -t RUNNING | wc -l'"
    echo ""
    echo "View logs for a specific task (e.g., task 1):"
    echo "  tail -f logs/h5_to_pkl_${JOB_ID}_1.out"
    echo ""
    echo "Check for failures:"
    echo "  grep -l 'Failed' logs/h5_to_pkl_${JOB_ID}_*.out"
    echo ""
    echo "=========================================="
else
    echo "❌ Failed to submit job array"
    exit 1
fi
