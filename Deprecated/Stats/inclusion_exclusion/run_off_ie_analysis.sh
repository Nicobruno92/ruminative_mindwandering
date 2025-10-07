#!/bin/bash

# Run LMM FDR Analysis - OFF Task Inclusion vs Exclusion Comparison
# This script compares OFF task conditions between inclusion and exclusion conditions

echo "=========================================="
echo "LMM FDR Analysis - OFF Task I/E Comparison"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Check if the Python script exists
PYTHON_SCRIPT="$SCRIPT_DIR/lmm_fdr_analysis_off_inclusion_exclusion.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if required data files exist
CSV_FILE="$PROJECT_ROOT/results/aggregated_mne_markers/aggregated_mne_markers_onoff_valence_confidence_time_selfother_5trials_go_correct_probe.csv"
METADATA_FILE="$PROJECT_ROOT/metadata_experiment.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: Aggregated MNE markers file not found: $CSV_FILE"
    exit 1
fi

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    exit 1
fi

echo "Input files verified."

# Create output directory
OUT_DIR="$PROJECT_ROOT/results/off_inclusion_exclusion_comparison"
mkdir -p "$OUT_DIR"
echo "Output directory: $OUT_DIR"

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the analysis
echo ""
echo "Starting OFF Task Inclusion vs Exclusion analysis..."
echo "This compares OFF task conditions between inclusion and exclusion conditions"
echo "Based on order (IE/EI) and task (Sart2/Sart4)"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the Python script
python3 "$PYTHON_SCRIPT"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Analysis completed successfully!"
    echo "=========================================="
    echo "Results saved to: $OUT_DIR"
    
    # List the generated files
    echo ""
    echo "Generated files:"
    find "$OUT_DIR" -type f -name "*.csv" -o -name "*.png" | sort
    
else
    echo ""
    echo "=========================================="
    echo "ERROR: Analysis failed!"
    echo "=========================================="
    exit 1
fi

echo ""
echo "Analysis complete." 