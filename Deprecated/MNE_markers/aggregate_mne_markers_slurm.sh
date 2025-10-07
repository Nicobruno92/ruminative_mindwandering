#!/bin/bash
#SBATCH --job-name=aggMneOptimized
#SBATCH --output=logs/agg_mne_optimized_%A_%a.out
#SBATCH --error=logs/agg_mne_optimized_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-1

INPUT_DIR="results/mne_markers"
OUTPUT_DIR="results/aggregated_mne_markers"
SCRIPT="MNE_markers/aggregate_mne_markers.py"

set -e

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Define condition sets to process in parallel
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # Job 0: Only onoff condition
    CONDITIONS_SETS=("onoff")
    JOB_NAME="onoff_only"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # Job 1: All conditions combined
    CONDITIONS_SETS=("onoff valence confidence time selfother")
    JOB_NAME="all_conditions"
fi

echo "Starting optimized MNE marker aggregation job: $JOB_NAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Conditions sets: ${CONDITIONS_SETS[@]}"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script $SCRIPT not found!"
    exit 1
fi

# Count MNE marker files
num_files=$(find "$INPUT_DIR" -name "*_mne_markers.csv" | wc -l)
echo "Found $num_files MNE marker files to process"
if [ "$num_files" -eq 0 ]; then
    echo "ERROR: No MNE marker files found to process."
    exit 1
fi

echo "Checking disk space..."
available_space=$(df -k . | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 10485760 ]; then  # Less than 10 GB
    echo "ERROR: Not enough disk space available. Only $available_space KB free."
    exit 1
fi

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate eeg
else
    echo "ERROR: Could not activate conda environment."
    exit 1
fi

# Set optimized threading environment variables
export MKL_NUM_THREADS=4  # Reduced to allow for parallel file processing
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-$USER

# Calculate optimal number of workers based on available CPUs and memory
# Leave some CPUs for system and use conservative memory per worker
MAX_WORKERS=$((SLURM_CPUS_PER_TASK / 2))  # 2 CPUs per worker (reduced from 4)
if [ "$MAX_WORKERS" -lt 4 ]; then
    MAX_WORKERS=4  # Minimum 4 workers for good parallelism
elif [ "$MAX_WORKERS" -gt 16 ]; then
    MAX_WORKERS=16  # Increased cap for better performance with more cores
fi

echo "Using $MAX_WORKERS parallel workers for file processing"

# Function to run aggregation for a specific condition set and level
run_aggregation() {
    local conditions="$1"
    local level="$2"
    local conditions_clean=$(echo "$conditions" | tr ' ' '_')
    
    echo "Processing conditions: '$conditions' at level: $level"
    
    python -u "$SCRIPT" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --conditions $conditions \
        --aggregate_level "$level" \
        --trials_before_probe 5 \
        --max_workers "$MAX_WORKERS" \
        --only_go_correct \
        --quiet
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Successfully completed: conditions='$conditions', level=$level"
    else
        echo "ERROR: Failed for conditions='$conditions', level=$level (exit code: $exit_code)"
        return $exit_code
    fi
}

# Process all condition sets for this job
for conditions in "${CONDITIONS_SETS[@]}"; do
    echo "Starting aggregation for conditions: '$conditions'"
    
    # Run all three aggregation levels in parallel using background processes
    echo "Launching probe-level aggregation in background..."
    run_aggregation "$conditions" "probe" &
    PROBE_PID=$!
    
    echo "Launching task-level aggregation in background..."
    run_aggregation "$conditions" "task" &
    TASK_PID=$!
    
    echo "Launching subject-level aggregation in background..."
    run_aggregation "$conditions" "subject" &
    SUBJECT_PID=$!
    
    # Wait for all three levels to complete and check their exit status
    echo "Waiting for all aggregation levels to complete for conditions: '$conditions'"
    
    wait $PROBE_PID
    PROBE_STATUS=$?
    echo "Probe-level aggregation completed with status: $PROBE_STATUS"
    
    wait $TASK_PID
    TASK_STATUS=$?
    echo "Task-level aggregation completed with status: $TASK_STATUS"
    
    wait $SUBJECT_PID
    SUBJECT_STATUS=$?
    echo "Subject-level aggregation completed with status: $SUBJECT_STATUS"
    
    # Check if any failed
    if [ $PROBE_STATUS -ne 0 ] || [ $TASK_STATUS -ne 0 ] || [ $SUBJECT_STATUS -ne 0 ]; then
        echo "ERROR: At least one aggregation level failed for conditions: '$conditions'"
        echo "Probe status: $PROBE_STATUS, Task status: $TASK_STATUS, Subject status: $SUBJECT_STATUS"
        exit 1
    else
        echo "All aggregation levels completed successfully for conditions: '$conditions'"
    fi
    
    # Brief pause to let system recover
    sleep 10
done

echo "All condition sets completed successfully for job: $JOB_NAME"
echo "MNE marker aggregation complete at $(date)"

# Clean up any temporary files or processes
cleanup() {
    echo "Cleaning up background processes..."
    jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "Aggregation job $JOB_NAME completed successfully!" 