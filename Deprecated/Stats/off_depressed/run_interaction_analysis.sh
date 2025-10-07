#!/bin/bash

# Script to run LMM interaction analysis: OnOff * Group
# This script examines the interaction between on-task/off-task conditions and groups

echo "Starting OnOff * Group Interaction Analysis..."
echo "Model: marker ~ onoff_label * group_label"
echo "Effects: Main effect of OnOff, Main effect of Group, Interaction effect"

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the Python analysis script
python3 lmm_fdr_interaction_analysis.py

echo "Interaction analysis completed!"
echo "Results saved to: ../../results/onoff_group_interaction_analysis/" 