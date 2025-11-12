#!/bin/bash
# Manual script to apply multiple comparisons correction to existing results
# Use this if you need to re-run MCC without re-running the entire pipeline

# Usage: bash Statistics/apply_mcc_manual.sh <model_dir>
# Example: bash Statistics/apply_mcc_manual.sh results/lmm_cluster/onoff

if [ $# -eq 0 ]; then
    echo "Usage: bash Statistics/apply_mcc_manual.sh <model_dir>"
    echo "Example: bash Statistics/apply_mcc_manual.sh results/lmm_cluster/onoff"
    exit 1
fi

MODEL_DIR=$1

echo "=========================================="
echo "MANUAL MCC APPLICATION"
echo "=========================================="
echo "Model directory: ${MODEL_DIR}"
echo ""

# Check if directory exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo "✗ Error: Directory not found: ${MODEL_DIR}"
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
fi

# Run MCC post-processing
python Statistics/apply_mcc_postprocessing.py "${MODEL_DIR}" --config Statistics/config.yaml

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ MCC application completed successfully"
    echo ""
    echo "Next steps:"
    echo "  1. Check corrected summaries in marker directories:"
    echo "     ${MODEL_DIR}/*/cluster_summary_corrected.csv"
    echo "  2. Generate summary report:"
    echo "     python Statistics/generate_summary_report.py ${MODEL_DIR}"
else
    echo ""
    echo "✗ MCC application failed with exit code ${EXIT_CODE}"
fi

echo "=========================================="
