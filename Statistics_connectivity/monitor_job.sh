#!/bin/bash
# Monitor CPU usage of running connectivity job
# Usage: bash Statistics_connectivity/monitor_job.sh <JOBID>

JOBID=${1:-1902552}

echo "Monitoring job ${JOBID}..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Get node where job is running
    NODE=$(squeue -j ${JOBID} -h -o "%N")
    
    if [ -z "$NODE" ]; then
        echo "Job ${JOBID} is no longer running"
        break
    fi
    
    # Get CPU usage from sstat
    echo "=== $(date) ==="
    sstat -j ${JOBID} --format=JobID,AveCPU,MaxRSS,NTasks -P 2>/dev/null || \
        echo "Job still initializing or sstat not available..."
    
    # Alternative: check with sacct
    sacct -j ${JOBID} --format=JobID,Elapsed,AveCPU,MaxRSS,State -P 2>/dev/null | tail -1
    
    echo ""
    sleep 10
done
