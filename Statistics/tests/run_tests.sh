#!/bin/bash
#
# Quick test script for LMM-based spatial cluster permutation pipeline
# This runs the comprehensive test suite with minimal configuration
#

set -e  # Exit on error

echo "======================================================================"
echo "LMM-Based Spatial Cluster Permutation Pipeline - Quick Test"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please ensure Python 3.7+ is installed."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Using Python $PYTHON_VERSION"
echo ""

# Check for required packages
echo "Checking dependencies..."
python -c "
import sys
missing = []
try:
    import numpy
    import pandas
    import yaml
    import mne
    import statsmodels
    import sklearn
    import matplotlib
    import seaborn
except ImportError as e:
    missing.append(str(e).split()[-1])

if missing:
    print(f'Error: Missing packages: {missing}')
    print('Install with: pip install numpy pandas pyyaml mne statsmodels scikit-learn matplotlib seaborn')
    sys.exit(1)
else:
    print('✓ All dependencies found')
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "======================================================================"
echo "Running comprehensive test suite..."
echo "======================================================================"
echo ""
echo "This will:"
echo "  1. Generate synthetic EEG data with realistic structure"
echo "  2. Test pipeline with multiple effect sizes (null, small, medium, large)"
echo "  3. Validate sensitivity and specificity"
echo "  4. Save detailed results to test_results/"
echo ""
echo "Estimated time: 5-10 minutes"
echo ""

# Run the comprehensive test
python test_pipeline_comprehensive.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Test suite completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Results saved to: $SCRIPT_DIR/test_results/"
    echo ""
    echo "To view results:"
    echo "  cd test_results/comprehensive_*/"
    echo "  ls -lh  # See all test directories"
    echo "  cat test_summary.csv  # View summary"
    echo ""
    echo "To clean up:"
    echo "  rm -rf test_results/"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ Test suite failed"
    echo "======================================================================"
    echo ""
    echo "Check error messages above for details"
    echo ""
    exit 1
fi
