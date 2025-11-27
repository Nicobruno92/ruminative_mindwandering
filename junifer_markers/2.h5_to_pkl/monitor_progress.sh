#!/bin/bash

# =============================================================================
# Monitor Progress of Parallel H5 to PKL Conversion
# =============================================================================

# Get the most recent job array
JOB_ID=$(squeue -n h5_to_pkl_parallel -u $USER -h -o "%A" | head -1 | cut -d'_' -f1)

if [ -z "${JOB_ID}" ]; then
    # Check recent jobs in logs
    latest_log=$(ls -t logs/h5_to_pkl_*.out 2>/dev/null | head -1)
    if [ ! -z "${latest_log}" ]; then
        JOB_ID=$(basename "${latest_log}" | cut -d'_' -f3)
        echo "No active jobs found. Showing status of most recent job: ${JOB_ID}"
        echo ""
    else
        echo "No h5_to_pkl_parallel jobs found."
        exit 1
    fi
else
    JOB_ID=$(echo "${JOB_ID}" | head -1)
fi

echo "=========================================="
echo "Parallel H5 to PKL Conversion - Progress"
echo "=========================================="
echo "Job ID: ${JOB_ID}"
echo ""

# Get job array status
running=$(squeue -j ${JOB_ID} -t RUNNING -h 2>/dev/null | wc -l)
pending=$(squeue -j ${JOB_ID} -t PENDING -h 2>/dev/null | wc -l)
# Determine expected total number of tasks from elements.csv (one per element)
LOCAL_ROOT="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering"
ELEMENTS_FILE="${LOCAL_ROOT}/junifer_markers/1.markers_h5_creation/elements.csv"
if [ -f "${ELEMENTS_FILE}" ]; then
    total=$(tail -n +2 "${ELEMENTS_FILE}" | wc -l)
else
    echo "Warning: elements.csv not found at ${ELEMENTS_FILE}, falling back to array logs"
    total=$(squeue -j ${JOB_ID} -h 2>/dev/null | wc -l)
fi

completed=$((total - running - pending))

echo "Status:"
echo "  Total tasks: ${total}"
echo "  Completed: ${completed}"
echo "  Running: ${running}"
echo "  Pending: ${pending}"
echo ""

# Calculate progress percentage
if [ "${total}" -gt 0 ]; then
    progress=$((completed * 100 / total))
else
    progress=0
fi
echo "Progress: ${completed}/${total} (${progress}%)"

# Progress bar
bar_length=40
filled=$((progress * bar_length / 100))
printf "["
for ((i=0; i<bar_length; i++)); do
    if [ $i -lt $filled ]; then
        printf "="
    else
        printf " "
    fi
done
printf "] ${progress}%%\n"

echo ""
echo "=========================================="

# Check for failures in completed logs
if [ -d "logs" ]; then
    failed_count=$(grep -l "Failed: [1-2]/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
    if [ ${failed_count} -gt 0 ]; then
        echo "⚠️  Warning: ${failed_count} tasks reported failures"
        echo ""
        echo "Failed tasks:"
        grep -l "Failed: [1-2]/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | while read log; do
            task_id=$(basename "${log}" | sed 's/.*_\([0-9]*\)\.out/\1/')
            element=$(grep "Element:" "${log}" | head -1 | sed 's/Element: //')
            echo "  Task ${task_id}: ${element}"
        done
        echo ""
    fi
    
    # Count successful completions
    success_count=$(grep -l "Successful: 2/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
    echo "✓ Successfully completed tasks: ${success_count}"
fi

echo ""
echo "=========================================="
echo "Commands"
echo "=========================================="
echo "Refresh this status:"
echo "  ./monitor_progress.sh"
echo ""
echo "Watch in real-time:"
echo "  watch -n 5 ./monitor_progress.sh"
echo ""
echo "View a specific task log (e.g., task 1):"
echo "  tail -f logs/h5_to_pkl_${JOB_ID}_1.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel -j ${JOB_ID}"
echo "=========================================="
