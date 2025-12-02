#!/bin/bash

# =============================================================================
# Full Junifer Pipeline Orchestrator
# =============================================================================
# This script runs the complete 3-step Junifer pipeline with job dependencies:
#   Step 1: Create H5 markers (submit_slurm_array.sh)
#   Step 2: Convert H5 to PKL (launch_parallel.sh)
#   Step 3: Aggregate markers by probe (run_aggregate_slurm.sh)
#
# Each step waits for all jobs from the previous step to complete successfully.
#
# Usage:
#   ./run_full_pipeline.sh                    # Run standalone
#   ./run_full_pipeline.sh --dependency=JOB_ID # Run after JOB_ID completes
# =============================================================================

set -euo pipefail

# Parse command line arguments
DEPENDENCY_JOB_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dependency=*)
            DEPENDENCY_JOB_ID="${1#*=}"
            shift
            ;;
        --dependency)
            DEPENDENCY_JOB_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dependency=JOB_ID]"
            echo "  --dependency=JOB_ID  Wait for JOB_ID to complete before starting pipeline"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Configuration
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering}
cd "$WORKDIR"

# Create logs directory for this orchestrator
LOGS_DIR="junifer_markers/logs"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Junifer Pipeline Orchestrator"
echo "=========================================="
echo "Working directory: $WORKDIR"
echo "Start time: $(date)"
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "Dependency: Waiting for Job ${DEPENDENCY_JOB_ID} to complete"
fi
echo ""

# =============================================================================
# STEP 1: Create H5 markers
# =============================================================================
echo "=========================================="
echo "STEP 1: Creating H5 markers"
echo "=========================================="

cd "$WORKDIR/junifer_markers/1.markers_h5_creation"

# Check if we need to modify the submit script to include dependency
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "Adding dependency on Job ${DEPENDENCY_JOB_ID} to Step 1"
    # Create a temporary modified submit script with dependency
    TEMP_SUBMIT_SCRIPT="submit_slurm_array_with_dependency.sh"
    cp submit_slurm_array.sh "$TEMP_SUBMIT_SCRIPT"
    
    # Add dependency to SBATCH_ARGS array in the submit script
    # We need to add it before the script path argument (line 88)
    # Add dependency flag after the partition check but before the script path
    # Use afterany to handle jobs that may have already completed
    sed -i "/SBATCH_ARGS+=(\"junifer_markers/i SBATCH_ARGS+=(\"--dependency=afterany:${DEPENDENCY_JOB_ID}\")" "$TEMP_SUBMIT_SCRIPT"
    
    # The submit_slurm_array.sh script already calls sbatch internally
    # We need to capture its output to get the job ID
    # Redirect stderr to avoid verbose output from set -x
    STEP1_OUTPUT=$(./"$TEMP_SUBMIT_SCRIPT" 2>&1)
    STEP1_JOB_ID=$(echo "$STEP1_OUTPUT" | grep "Submitted array job:" | awk '{print $NF}')
    
    # Clean up temporary script
    rm -f "$TEMP_SUBMIT_SCRIPT"
else
    # The submit_slurm_array.sh script already calls sbatch internally
    # We need to capture its output to get the job ID
    # Redirect stderr to avoid verbose output from set -x
    STEP1_OUTPUT=$(./submit_slurm_array.sh 2>&1)
    STEP1_JOB_ID=$(echo "$STEP1_OUTPUT" | grep "Submitted array job:" | awk '{print $NF}')
fi

if [ -z "$STEP1_JOB_ID" ]; then
    echo "❌ ERROR: Failed to extract Step 1 job ID"
    echo "Output: $STEP1_OUTPUT"
    exit 1
fi

echo "✓ Step 1 submitted: Job ID ${STEP1_JOB_ID}"
echo ""

# =============================================================================
# STEP 2: Convert H5 to PKL (depends on Step 1)
# =============================================================================
echo "=========================================="
echo "STEP 2: Converting H5 to PKL"
echo "=========================================="
echo "Dependency: Waiting for Job ${STEP1_JOB_ID} to complete"

cd "$WORKDIR/junifer_markers/2.h5_to_pkl"

# Submit step 2 with dependency on step 1 completion
# Step 1 is an array job, so we use afterany to wait for ALL array elements
# afterany handles jobs that complete before dependency is registered
ELEMENTS_FILE="$WORKDIR/junifer_markers/1.markers_h5_creation/elements.csv"

if [ ! -f "${ELEMENTS_FILE}" ]; then
    echo "❌ ERROR: Elements file not found for Step 2: ${ELEMENTS_FILE}"
    exit 1
fi

total_elements=$(tail -n +2 "${ELEMENTS_FILE}" | wc -l)

if [ "${total_elements}" -le 0 ]; then
    echo "❌ ERROR: No elements found in ${ELEMENTS_FILE} for Step 2"
    exit 1
fi

ARRAY_RANGE="1-${total_elements}%20"

echo "Submitting Step 2 with dependency afterany:${STEP1_JOB_ID} and array range ${ARRAY_RANGE}..."
STEP2_JOB_ID=$(sbatch --parsable --dependency=afterany:${STEP1_JOB_ID} --array=${ARRAY_RANGE} batch_convert_h5_to_pkl_parallel.sh)

if [ -z "$STEP2_JOB_ID" ]; then
    echo "❌ ERROR: Failed to submit Step 2"
    exit 1
fi

echo "✓ Step 2 submitted: Job ID ${STEP2_JOB_ID}"
echo "  Will start after Job ${STEP1_JOB_ID} completes successfully"
echo ""

# =============================================================================
# STEP 3: Aggregate markers by probe (depends on Step 2)
# =============================================================================
echo "=========================================="
echo "STEP 3: Aggregating markers by probe"
echo "=========================================="
echo "Dependency: Waiting for Job ${STEP2_JOB_ID} to complete"

cd "$WORKDIR/junifer_markers/3.aggregate_probes"

# Create logs directory for step 3
mkdir -p logs

# Submit step 3 with dependency on step 2 completion
# Step 2 is an array job, so we use afterany to wait for ALL array elements
# afterany handles jobs that complete before dependency is registered
STEP3_JOB_ID=$(sbatch --parsable --dependency=afterany:${STEP2_JOB_ID} run_aggregate_slurm.sh)

if [ -z "$STEP3_JOB_ID" ]; then
    echo "❌ ERROR: Failed to submit Step 3"
    exit 1
fi

echo "✓ Step 3 submitted: Job ID ${STEP3_JOB_ID}"
echo "  Will start after Job ${STEP2_JOB_ID} completes successfully"
echo ""

# =============================================================================
# Summary
# =============================================================================
cd "$WORKDIR"

echo "=========================================="
echo "Pipeline Submitted Successfully!"
echo "=========================================="
echo "Job Chain:"
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "  Dependency Job:         ${DEPENDENCY_JOB_ID} → must complete first"
fi
echo "  Step 1 (H5 creation):    ${STEP1_JOB_ID} (array job)"
echo "  Step 2 (H5 to PKL):      ${STEP2_JOB_ID} (array job) → waits for Step 1 array"
echo "  Step 3 (Aggregation):    ${STEP3_JOB_ID} → waits for Step 2 array"
echo ""
echo "=========================================="
echo "Monitoring Commands"
echo "=========================================="
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "Check dependency job:"
    echo "  squeue -j ${DEPENDENCY_JOB_ID}"
    echo ""
    echo "Check all jobs (dependency + pipeline):"
    echo "  squeue -j ${DEPENDENCY_JOB_ID},${STEP1_JOB_ID},${STEP2_JOB_ID},${STEP3_JOB_ID}"
    echo ""
    echo "Monitor all jobs in real-time:"
    echo "  watch -n 10 'squeue -j ${DEPENDENCY_JOB_ID},${STEP1_JOB_ID},${STEP2_JOB_ID},${STEP3_JOB_ID}'"
    echo ""
    echo "Cancel entire pipeline (dependency will continue):"
    echo "  scancel ${STEP1_JOB_ID} ${STEP2_JOB_ID} ${STEP3_JOB_ID}"
else
    echo "Check all pipeline jobs:"
    echo "  squeue -j ${STEP1_JOB_ID},${STEP2_JOB_ID},${STEP3_JOB_ID}"
    echo ""
    echo "Monitor all jobs in real-time:"
    echo "  watch -n 10 'squeue -j ${STEP1_JOB_ID},${STEP2_JOB_ID},${STEP3_JOB_ID}'"
    echo ""
    echo "Cancel entire pipeline:"
    echo "  scancel ${STEP1_JOB_ID} ${STEP2_JOB_ID} ${STEP3_JOB_ID}"
fi
echo ""
echo "Check Step 1 status:"
echo "  squeue -j ${STEP1_JOB_ID}"
echo ""
echo "Check Step 2 status:"
echo "  squeue -j ${STEP2_JOB_ID}"
echo ""
echo "Check Step 3 status:"
echo "  squeue -j ${STEP3_JOB_ID}"
echo ""
echo "=========================================="
echo "Pipeline started at: $(date)"
echo "=========================================="
