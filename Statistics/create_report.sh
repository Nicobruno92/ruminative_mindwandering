#!/bin/bash
#
# Generate comprehensive summary report for LMM cluster analysis results
#
# Usage:
#   bash Statistics/create_report.sh [model_directory]
#
# Example:
#   bash Statistics/create_report.sh results/lmm_cluster/onoff
#

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_summary_report.py"

# Check if model directory is provided
if [ $# -eq 0 ]; then
    echo "Error: Model directory not specified"
    echo ""
    echo "Usage: bash $0 <model_directory>"
    echo ""
    echo "Example:"
    echo "  bash $0 results/lmm_cluster/onoff"
    echo ""
    exit 1
fi

MODEL_DIR="$1"

# Check if directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Directory not found: $MODEL_DIR"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "================================================================================"
echo "GENERATING SUMMARY REPORT"
echo "================================================================================"
echo "Model directory: $MODEL_DIR"
echo "Python script:   $PYTHON_SCRIPT"
echo ""

# Run the report generation
python "$PYTHON_SCRIPT" "$MODEL_DIR" "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "REPORT GENERATION COMPLETED SUCCESSFULLY"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "REPORT GENERATION FAILED (exit code: $exit_code)"
    echo "================================================================================"
fi

exit $exit_code
