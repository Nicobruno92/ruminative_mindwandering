#!/bin/bash

# =============================================================================
# Generate Final Summary Report
# =============================================================================

# Find most recent job ID
JOB_ID=$(ls -t logs/h5_to_pkl_*.out 2>/dev/null | head -1 | sed 's/.*h5_to_pkl_\([0-9]*\)_.*/\1/')

if [ -z "${JOB_ID}" ]; then
    echo "No job logs found."
    exit 1
fi

echo "=========================================="
echo "Final Summary Report"
echo "=========================================="
echo "Job ID: ${JOB_ID}"
echo ""

# Count all logs for this job
total_logs=$(ls logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
echo "Total tasks executed: ${total_logs}"

# Count successes (both files converted)
success_full=$(grep -l "Successful: 2/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
echo "✓ Fully successful (2/2): ${success_full}"

# Count partial successes (only 1 file converted)
success_partial=$(grep -l "Successful: 1/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
if [ ${success_partial} -gt 0 ]; then
    echo "⚠️  Partial success (1/2): ${success_partial}"
fi

# Count failures (0 files converted)
failed=$(grep -l "Successful: 0/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | wc -l)
if [ ${failed} -gt 0 ]; then
    echo "❌ Complete failures (0/2): ${failed}"
fi

echo ""
echo "=========================================="
echo "Detailed Results"
echo "=========================================="

# List partial failures
if [ ${success_partial} -gt 0 ]; then
    echo ""
    echo "Partial failures (1/2 converted):"
    grep -l "Successful: 1/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | while read log; do
        task_id=$(basename "${log}" | sed 's/.*_\([0-9]*\)\.out/\1/')
        element=$(grep "Element:" "${log}" | tail -1 | sed 's/.*Element: //')
        echo "  Task ${task_id}: ${element}"
        # Show which desc failed
        grep "⚠️" "${log}" | head -2 | sed 's/^/    /'
    done
fi

# List complete failures
if [ ${failed} -gt 0 ]; then
    echo ""
    echo "Complete failures (0/2 converted):"
    grep -l "Successful: 0/2" logs/h5_to_pkl_${JOB_ID}_*.out 2>/dev/null | while read log; do
        task_id=$(basename "${log}" | sed 's/.*_\([0-9]*\)\.out/\1/')
        element=$(grep "Element:" "${log}" | tail -1 | sed 's/.*Element: //')
        echo "  Task ${task_id}: ${element}"
        # Show why it failed
        grep "⚠️\|❌" "${log}" | head -4 | sed 's/^/    /'
    done
fi

echo ""
echo "=========================================="
echo "PKL Files Generated"
echo "=========================================="

# Count actual PKL files created
BIDS_ROOT="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS"
PKL_DIR="${BIDS_ROOT}/features"

total_pkl=$(find "${PKL_DIR}" -name "*_markers.pkl" -type f 2>/dev/null | wc -l)
echo "Total PKL files in ${PKL_DIR}: ${total_pkl}"

# Count by description type
evoked_pkl=$(find "${PKL_DIR}" -name "*_desc-evoked_markers.pkl" -type f 2>/dev/null | wc -l)
state_pkl=$(find "${PKL_DIR}" -name "*_desc-state_markers.pkl" -type f 2>/dev/null | wc -l)

echo "  - desc-evoked: ${evoked_pkl}"
echo "  - desc-state: ${state_pkl}"

echo ""
echo "Expected: 167 elements × 2 descriptions = 334 PKL files"
echo "Actual: ${total_pkl} PKL files"

if [ ${total_pkl} -eq 334 ]; then
    echo "✓ All PKL files generated successfully!"
else
    missing=$((334 - total_pkl))
    echo "⚠️  Missing ${missing} PKL files"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo "1. Review any failures above"
echo "2. Re-run failed tasks if needed"
echo "3. Proceed to probe aggregation:"
echo "   cd ../3.aggregate_probes"
echo "=========================================="
