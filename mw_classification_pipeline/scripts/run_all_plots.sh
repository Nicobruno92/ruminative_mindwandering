#!/usr/bin/env bash
# =============================================================================
# run_all_plots.sh — Dynamically generate comparison plots for all completed runs
# =============================================================================
# Scans the MW_Classification results tree and runs generate_pipeline_plots.py
# for every model directory that contains a true_runs/ subdirectory.
# No hardcoded paths — works for any model (rf, lr, iforest, xgb, ...) and
# any contrast/family combination present on disk.
#
# USAGE (from project root):
#   bash mw_classification_pipeline/scripts/run_all_plots.sh
#   bash mw_classification_pipeline/scripts/run_all_plots.sh --top_n 30
#   bash mw_classification_pipeline/scripts/run_all_plots.sh --results_root /path/to/MW_Classification
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/generate_pipeline_plots.py"

# Default results root (relative to pipeline root)
RESULTS_ROOT="$PIPELINE_ROOT/results/MW_Classification"

TOP_N=20

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --top_n)    TOP_N="$2";         shift 2 ;;
        --results_root) RESULTS_ROOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Python executable — full path to ML conda env
PYTHON="/network/iss/home/nicolas.bruno/miniforge3/envs/ML/bin/python"
if [ ! -f "$PYTHON" ]; then
    PYTHON="$(which python)"
fi

if [ ! -d "$RESULTS_ROOT" ]; then
    echo "Results root not found: $RESULTS_ROOT"
    exit 1
fi

echo "==========================================="
echo " MW Classification — Dynamic Plot Generation"
echo "==========================================="
echo " Python      : $PYTHON"
echo " Results root: $RESULTS_ROOT"
echo " Top-N       : $TOP_N features"
echo " Started     : $(date)"
echo "==========================================="

DONE=0
SKIPPED=0
FAILED=0

# Find all model-level directories that contain true_runs/
# A model dir looks like: .../LOSO/CONTRAST/FAMILY/MODEL_TYPE/
while IFS= read -r -d '' TRUE_RUNS_DIR; do
    MODEL_DIR="$(dirname "$TRUE_RUNS_DIR")"
    LABEL="${MODEL_DIR#$RESULTS_ROOT/}"

    echo ""
    echo "============================================================"
    echo "  $LABEL"
    echo "============================================================"

    if "$PYTHON" "$PLOT_SCRIPT" \
        --results_dir "$MODEL_DIR" \
        --top_n_features "$TOP_N"; then
        DONE=$((DONE + 1))
    else
        echo "  [FAILED] $LABEL"
        FAILED=$((FAILED + 1))
    fi
done < <(find "$RESULTS_ROOT" -mindepth 1 -type d -name "true_runs" -print0 | sort -z)

echo ""
echo "==========================================="
echo " Done: $DONE   Failed: $FAILED"
echo " Finished : $(date)"
echo "==========================================="

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
