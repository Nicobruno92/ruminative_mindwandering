#!/bin/bash
# Regenerate combined ERP+Beta plots with horizontal layout (A and B side by side)

echo "=================================================="
echo "Regenerating Combined ERP+Beta Plots"
echo "New format: A (ERP) and B (Beta) side by side"
echo "=================================================="

cd "$(dirname "$0")"

# Activate eeg environment
echo "Activating eeg environment..."
conda activate eeg

# Run the combined plotting script
echo "Running combined plotting..."
python combined_erp_beta_plots.py --config config.yaml

echo ""
echo "✅ Done! Check results/ERPs_new/combined_erp_beta/"
echo "   - Individual plots: combined_erp_beta_roi_{roi}.png"
echo "   - Summary plot: combined_erp_beta_summary.png"
