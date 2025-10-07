#!/bin/bash

# Script to run LMM FDR analysis comparing OFF task conditions between groups 1 and 2
# This script runs the analysis for depressed patients using metadata to identify groups

echo "Starting OFF Task Group Comparison Analysis..."
echo "Comparing Group 1 vs Group 2 for OFF task conditions only"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the Python analysis script
python3 lmm_fdr_analysis_off_groups.py

echo "Analysis completed!"
echo "Results saved to: ../../results/off_depressed_group_comparison/" 