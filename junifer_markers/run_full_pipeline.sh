#!/bin/bash

# =============================================================================
# Full Junifer Pipeline Orchestrator — depressed_mindwandering Project
# =============================================================================
# Runs the complete 4-step pipeline with SLURM job dependencies:
#   Step 0: Compute per-subject SW PTP thresholds (slurm_compute_thresholds.sh)
#   Step 1: Create H5 markers              (slurm_array_junifer.sh)
#   Step 2: Aggregate markers by probe     (run_aggregate_slurm.sh)
#   Step 3: Group-level topomap figures    (slurm_topomaps.sh)
#
# Step 1 (sleep) depends on Step 0 because config_sleep.yaml has
# ptp_thresholds_strict: true — without the CSV, every sleep element crashes.
# Step 1 (evoked/state) does not need Step 0 but is chained via the same
# dependency for simplicity. Step 3 depends on Step 2.
#
# Elements are discovered dynamically by scanning the derivatives directory.
#
# Usage:
#   ./run_full_pipeline.sh                    # Run standalone (steps 0, 1, 2, 3)
#   ./run_full_pipeline.sh --skip-step0       # Skip SW thresholds (CSV must exist)
#   ./run_full_pipeline.sh --skip-topomaps    # Skip Step 3
#   ./run_full_pipeline.sh --dependency=JOB_ID # Wait for external JOB_ID first
# =============================================================================

set -euo pipefail

# Parse command line arguments
DEPENDENCY_JOB_ID=""
SKIP_STEP0=0
SKIP_TOPOMAPS=0
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
        --skip-step0)
            SKIP_STEP0=1
            shift
            ;;
        --skip-topomaps)
            SKIP_TOPOMAPS=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dependency=JOB_ID] [--skip-step0] [--skip-topomaps]"
            echo "  --dependency=JOB_ID  Wait for JOB_ID to complete before starting pipeline"
            echo "  --skip-step0         Skip SW threshold computation (CSV must already exist)"
            echo "  --skip-topomaps      Skip Step 3 (group-level topomap figures)"
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
# STEP 0: Compute per-subject SW PTP thresholds
# =============================================================================
STEP0_JOB_ID=""
if [ "$SKIP_STEP0" -eq 1 ]; then
    echo "=========================================="
    echo "STEP 0: SKIPPED (--skip-step0)"
    echo "=========================================="
    echo "Assuming pooled_sw_thresholds.csv already exists at the path"
    echo "configured in 1.markers_h5_creation/config_sleep.yaml."
    echo ""
else
    echo "=========================================="
    echo "STEP 0: Computing SW PTP thresholds"
    echo "=========================================="

    cd "$WORKDIR/junifer_markers/0.compute_sw_thresholds"
    mkdir -p logs

    if [ -n "$DEPENDENCY_JOB_ID" ]; then
        STEP0_OUTPUT=$(sbatch --parsable --dependency=afterany:${DEPENDENCY_JOB_ID} slurm_compute_thresholds.sh 2>&1)
    else
        STEP0_OUTPUT=$(sbatch --parsable slurm_compute_thresholds.sh 2>&1)
    fi

    STEP0_JOB_ID=$(echo "$STEP0_OUTPUT" | grep -E '^[0-9]+' | head -1)
    if [ -z "$STEP0_JOB_ID" ]; then
        echo "❌ ERROR: Failed to submit Step 0"
        echo "Output: $STEP0_OUTPUT"
        exit 1
    fi
    echo "✓ Step 0 submitted: Job ID ${STEP0_JOB_ID}"
    echo ""
fi

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

# Step 1 jobs depend on Step 0 (sleep needs the CSV; chaining all for simplicity)
# If --dependency was passed AND step 0 ran, chain is: external -> step0 -> step1.
# If --skip-step0, fall back to the external dependency for step 1.
if [ -n "$STEP0_JOB_ID" ]; then
    STEP1_DEP="$STEP0_JOB_ID"
else
    STEP1_DEP="$DEPENDENCY_JOB_ID"
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
    
    if [ -n "$STEP1_DEP" ]; then
        STEP1_OUTPUT=$(CONFIG_TYPE=${config_type} sbatch --parsable --array="${ARRAY_SPEC}" --dependency=afterany:${STEP1_DEP} slurm_array_junifer.sh 2>&1)
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

# Combine all Step 1 job IDs for SLURM dependency tracking.
# SLURM uses `:` to chain multiple deps within one type (afterany:A:B:C),
# while `,` separates expressions of different types. We need AND-of-arrays,
# so colons.
STEP1_JOB_ID=$(IFS=':'; echo "${STEP1_JOB_IDS[*]}")
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
# STEP 3: Group-level topomap figures (depends on Step 2)
# =============================================================================
STEP3_JOB_ID=""
if [ "$SKIP_TOPOMAPS" -eq 1 ]; then
    echo "=========================================="
    echo "STEP 3: SKIPPED (--skip-topomaps)"
    echo "=========================================="
    echo ""
else
    echo "=========================================="
    echo "STEP 3: Generating group-level topomaps"
    echo "=========================================="
    echo "Dependency: Waiting for Job ${STEP2_JOB_ID} to complete"

    cd "$WORKDIR/junifer_markers/3.topomaps"
    mkdir -p logs

    STEP3_OUTPUT=$(sbatch --parsable --dependency=afterany:${STEP2_JOB_ID} slurm_topomaps.sh 2>&1)
    STEP3_JOB_ID=$(echo "$STEP3_OUTPUT" | grep -E '^[0-9]+' | head -1)

    if [ -z "$STEP3_JOB_ID" ]; then
        echo "❌ ERROR: Failed to submit Step 3"
        echo "Output: $STEP3_OUTPUT"
        exit 1
    fi

    echo "✓ Step 3 submitted: Job ID ${STEP3_JOB_ID}"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
cd "$WORKDIR"

echo "=========================================="
echo "Pipeline Submitted Successfully!"
echo "=========================================="
echo "Job Chain:"
if [ -n "$DEPENDENCY_JOB_ID" ]; then
    echo "  External dep:           ${DEPENDENCY_JOB_ID}"
fi
if [ -n "$STEP0_JOB_ID" ]; then
    echo "  Step 0 (SW thresholds): ${STEP0_JOB_ID}"
fi
echo "  Step 1 (H5 creation):   ${STEP1_JOB_ID} (${#VALID_CONFIG_TYPES[@]} types: ${VALID_CONFIG_TYPES[*]})"
echo "  Step 2 (Aggregation):   ${STEP2_JOB_ID}"
if [ -n "$STEP3_JOB_ID" ]; then
    echo "  Step 3 (Topomaps):      ${STEP3_JOB_ID}"
fi
echo ""
echo "Total elements: ${TOTAL_ELEMENTS}"
echo ""
# For display only: squeue/scancel expect commas, but STEP1_JOB_ID is colon-separated.
STEP1_JOB_ID_DISPLAY="${STEP1_JOB_ID//:/,}"
ALL_JOBS="${STEP0_JOB_ID:+${STEP0_JOB_ID},}${STEP1_JOB_ID_DISPLAY},${STEP2_JOB_ID}${STEP3_JOB_ID:+,${STEP3_JOB_ID}}"
echo "Monitor: squeue -j ${ALL_JOBS}"
echo "Cancel:  scancel ${ALL_JOBS//,/ }"
echo ""
echo "Started at: $(date)"
echo "=========================================="
