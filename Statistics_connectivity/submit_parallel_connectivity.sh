#!/bin/bash

# =============================================================================
# PARALLEL CONNECTIVITY ANALYSIS - MASTER SUBMISSION SCRIPT
# =============================================================================
# Submits connectivity analysis jobs in parallel:
#   1. Array job for all 4 bands (theta, alpha, beta, gamma) in parallel
#   2. Aggregation job to combine results and generate figures (depends on array)
#
# Usage:
#   bash Statistics_connectivity/submit_parallel_connectivity.sh
#   bash Statistics_connectivity/submit_parallel_connectivity.sh --bands "0-1"  # Only theta, alpha
#
# This leverages cluster parallelism to run 4 bands simultaneously instead
# of sequentially, reducing total runtime from ~8h to ~2h.
# =============================================================================

set -euo pipefail

# Parse arguments
BAND_ARRAY="0-3"  # Default: all 4 bands
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bands)
            BAND_ARRAY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="Statistics_connectivity"
LOG_DIR="logs"

# Create log directory
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "Parallel Connectivity Analysis Submission"
echo "=========================================="
echo "Submitting array job for bands: ${BAND_ARRAY}"
echo "  0 = theta"
echo "  1 = alpha"
echo "  2 = beta"
echo "  3 = gamma"
echo "=========================================="

# Submit array job for bands
BAND_JOB=$(sbatch --parsable --array=${BAND_ARRAY} ${SCRIPT_DIR}/submit_band_array.sh)

echo "Submitted band array job: ${BAND_JOB}"

# Submit aggregation job (depends on all band jobs completing)
AGG_JOB=$(sbatch --parsable --dependency=afterok:${BAND_JOB} ${SCRIPT_DIR}/aggregate_results.sh)

echo "Submitted aggregation job: ${AGG_JOB} (depends on ${BAND_JOB})"

echo "=========================================="
echo "Submission complete!"
echo "=========================================="
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/connectivity_band_${BAND_JOB}_*.out"
echo ""
echo "Job chain:"
echo "  ${BAND_JOB} (array) -> ${AGG_JOB} (aggregate)"
echo "=========================================="
