#!/bin/bash
#
# Cancel current jobs and resubmit with fixed code
#

echo "Cancelling current Andrillon jobs..."

# Find all running/pending jobs with name "andrillon"
JOBS=$(squeue -u $USER -n andrillon -h -o "%A")

if [ -z "$JOBS" ]; then
    echo "No andrillon jobs found running"
else
    echo "Found jobs: $JOBS"
    for JOB in $JOBS; do
        echo "Cancelling job $JOB..."
        scancel $JOB
    done
    echo "All jobs cancelled"
fi

echo ""
echo "Waiting 5 seconds for jobs to clear..."
sleep 5

echo ""
echo "Resubmitting with fixed code..."
bash submit_parallel_andrillon.sh

echo ""
echo "Done! Check job status with: squeue -u \$USER"
