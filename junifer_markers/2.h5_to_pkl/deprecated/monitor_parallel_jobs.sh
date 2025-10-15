#!/bin/bash
# Monitor the parallel PKL creation jobs

echo "======================================"
echo "PKL Creation Jobs Monitor"
echo "======================================"
echo ""

# Check if jobs are running
echo "Active jobs:"
squeue -u $USER | grep pkl_create || echo "  No jobs found"
echo ""

# Count output files
LOG_DIR="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/logs"
OUTPUT_DIR="/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features"

if [ -d "${LOG_DIR}" ]; then
    echo "Log files:"
    echo "  Out files: $(ls ${LOG_DIR}/pkl_parallel_*.out 2>/dev/null | wc -l)"
    echo "  Err files: $(ls ${LOG_DIR}/pkl_parallel_*.err 2>/dev/null | wc -l)"
    echo ""
    
    # Check for errors in log files
    echo "Checking for errors..."
    ERROR_COUNT=$(grep -c "ERROR:" ${LOG_DIR}/pkl_parallel_*.out 2>/dev/null | awk -F: '{sum+=$2} END {print sum}')
    echo "  Total errors found: ${ERROR_COUNT:-0}"
    echo ""
fi

# Count completed PKL files
if [ -d "${OUTPUT_DIR}" ]; then
    echo "Output PKL files:"
    PKL_COUNT=$(find ${OUTPUT_DIR} -name "*_markers.pkl" 2>/dev/null | wc -l)
    echo "  Total PKL files created: ${PKL_COUNT}/334"
    echo ""
fi

# Show last few lines of recent log
echo "Recent progress (last log file):"
LATEST_LOG=$(ls -t ${LOG_DIR}/pkl_parallel_*.out 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "  From: ${LATEST_LOG}"
    tail -n 10 "${LATEST_LOG}" | sed 's/^/  /'
else
    echo "  No log files found yet"
fi

echo ""
echo "======================================"
echo "To see live updates:"
echo "  watch -n 5 ./monitor_parallel_jobs.sh"
echo "To cancel all jobs:"
echo "  scancel -u \$USER -n pkl_create"
echo "======================================"
