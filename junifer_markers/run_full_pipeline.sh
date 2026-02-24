#!/bin/bash

# =============================================================================
# Full Junifer Pipeline Orchestrator - Wandering Mind Project
# =============================================================================
# This script runs the complete 2-step Junifer pipeline with job dependencies:
#   Step 1: Create H5 markers (slurm_array_junifer.sh)
#   Step 2: Aggregate markers by probe (run_aggregate_slurm.sh)
#
# The aggregation step reads directly from H5 files using JuniferHDF5Reader,
# bypassing the intermediate PKL conversion step for efficiency.
#
# Elements are discovered dynamically by scanning the derivatives directory.
# No elements CSV files needed - everything is generated in memory.
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

# Create logs directory
LOGS_DIR="junifer_markers/logs"
mkdir -p "$LOGS_DIR"

echo "=========================================="
echo "Junifer Pipeline Orchestrator - Wandering Mind"
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

# Check config file exists
if [ ! -f "pipeline_config.yaml" ]; then
    echo "❌ ERROR: pipeline_config.yaml not found"
    exit 1
fi

# Read description types from config file
CONFIG_TYPES=($(python -c "import yaml; print(' '.join(yaml.safe_load(open('pipeline_config.yaml'))['descriptions']))"))

VALID_CONFIG_TYPES=()
STEP1_JOB_IDS=()
TOTAL_ELEMENTS=0

echo "Discovering elements from derivatives directory..."

for config_type in "${CONFIG_TYPES[@]}"; do
    # Count elements dynamically using Python script
    elements_count=$(python discover_elements.py --desc "${config_type}" --count 2>/dev/null || echo "0")
    
    if [ "${elements_count}" -gt 0 ]; then
        VALID_CONFIG_TYPES+=("$config_type")
        TOTAL_ELEMENTS=$((TOTAL_ELEMENTS + elements_count))
        echo "  ✓ ${config_type}: ${elements_count} elements found"
    else
        echo "  ⚠ ${config_type}: no elements found, skipping"
    fi
done

if [ ${#VALID_CONFIG_TYPES[@]} -eq 0 ]; then
    echo "❌ ERROR: No elements found for any config type"
    exit 1
fi

echo ""

for config_type in "${VALID_CONFIG_TYPES[@]}"; do
    # Get element count for array specification
    elements_count=$(python discover_elements.py --desc "${config_type}" --count)
    MAX_INDEX=$((elements_count - 1))
    ARRAY_SPEC="0-${MAX_INDEX}%20"  # Limit to 20 concurrent jobs
    
    echo "Submitting Step 1 for ${config_type} markers (${elements_count} elements, array=${ARRAY_SPEC})..."
    
    if [ -n "$DEPENDENCY_JOB_ID" ]; then
        STEP1_OUTPUT=$(CONFIG_TYPE=${config_type} sbatch --parsable --array="${ARRAY_SPEC}" --dependency=afterany:${DEPENDENCY_JOB_ID} slurm_array_junifer.sh 2>&1)
    else
        STEP1_OUTPUT=$(CONFIG_TYPE=${config_type} sbatch --parsable --array="${ARRAY_SPEC}" slurm_array_junifer.sh 2>&1)
    fi
    
    STEP1_JOB_ID=$(echo "$STEP1_OUTPUT" | grep -E '^[0-9]+' | head -1)
    if [ -z "$STEP1_JOB_ID" ]; then
        echo "❌ ERROR: Failed to submit Step 1 for ${config_type}"
        echo "Output: $STEP1_OUTPUT"
        exit 1
    fi
    
    STEP1_JOB_IDS+=("$STEP1_JOB_ID")
    echo "✓ Step 1 (${config_type}) submitted: Job ID ${STEP1_JOB_ID}"
done

# Combine all Step 1 job IDs for dependency tracking
STEP1_JOB_ID=$(IFS=','; echo "${STEP1_JOB_IDS[*]}")
echo ""
echo "✓ All Step 1 jobs submitted: ${STEP1_JOB_ID}"
echo ""

# =============================================================================
# STEP 2: Aggregate markers by probe (depends on Step 1)
# =============================================================================
echo "=========================================="
echo "STEP 2: Aggregating markers by probe"
echo "=========================================="
echo "Dependency: Waiting for Jobs ${STEP1_JOB_ID} to complete"
echo "Note: Reads directly from H5 files using JuniferHDF5Reader"

cd "$WORKDIR/junifer_markers/2.aggregate_probes"
mkdir -p logs

echo "Submitting Step 2 launcher (it will auto-submit the aggregation array when it runs)..."

STEP2_JOB_ID=$(bash run_aggregate_slurm.sh --parsable --dependency=afterany:${STEP1_JOB_ID})

if [ -z "$STEP2_JOB_ID" ]; then
    echo "❌ ERROR: Failed to submit Step 2"
    exit 1
fi

echo "✓ Step 2 submitted: Job ID ${STEP2_JOB_ID}"
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
    echo "  Dependency:           ${DEPENDENCY_JOB_ID}"
fi
echo "  Step 1 (H5 creation): ${STEP1_JOB_ID} (${#VALID_CONFIG_TYPES[@]} types: ${VALID_CONFIG_TYPES[*]})"
echo "  Step 2 (Aggregation): ${STEP2_JOB_ID}"
echo ""
echo "Total elements: ${TOTAL_ELEMENTS}"
echo ""
echo "Monitor: squeue -j ${STEP1_JOB_ID},${STEP2_JOB_ID}"
echo "Cancel:  scancel ${STEP1_JOB_ID//,/ } ${STEP2_JOB_ID}"
echo ""
echo "Started at: $(date)"
echo "=========================================="
