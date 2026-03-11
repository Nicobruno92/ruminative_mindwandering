#!/bin/bash
## SLURM directives
#SBATCH --job-name=ERPsComplete
#SBATCH --output=logs/erps_complete_%j.out
#SBATCH --error=logs/erps_complete_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

mkdir -p logs

module load proxy || true

echo "Activating Python environment..."
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
else
  module load anaconda3 || module load miniconda3 || module load conda || true
  conda activate eeg || true
fi

export MPLBACKEND=Agg

echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=== Running Complete ERP Analysis Pipeline ==="
echo "Step 1: Generate probe evoked potentials..."

# Step 1: Generate individual probe evoked potentials
srun --cpu-bind=none --ntasks=1 python ERPs_new/make_probe_evokeds.py \
  --config ERPs_new/config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: Probe evoked generation failed!"
    exit 1
fi

echo "Step 2: Run Linear Mixed Model analysis..."

# Install statsmodels if not available
python -c "import statsmodels" 2>/dev/null || pip install --user statsmodels

# Step 2: Run LMM analysis FIRST
srun --cpu-bind=none --ntasks=1 python ERPs_new/lmm_analysis.py \
  --config ERPs_new/config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: LMM analysis failed!"
    exit 1
fi

echo "Step 3: Generate ERP figures with LMM results..."

# Step 3: Generate ERP figures (individual and group) with LMM results
srun --cpu-bind=none --ntasks=1 python ERPs_new/make_erp_figures.py \
  --config ERPs_new/config.yaml

if [ $? -ne 0 ]; then
    echo "ERROR: ERP figure generation failed!"
    exit 1
fi

echo "=== Complete ERP Analysis Pipeline Finished Successfully! ==="
echo ""
echo "Results saved in:"
echo "  - Individual probe evokeds: /network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/"
echo "  - LMM analysis: results/ERPs_new/lmm/"
echo "  - ERP figures with significant windows: results/ERPs_new/participants/ and results/ERPs_new/group/"
echo ""
echo "Check the following for results:"
echo "  - LMM windowed results: results/ERPs_new/lmm/lmm_results.csv"
echo "  - LMM temporal results: results/ERPs_new/lmm/lmm_temporal_results.csv"
echo "  - LMM visualizations: results/ERPs_new/lmm/*.png"
echo "  - Individual ERP plots: results/ERPs_new/participants/sub-XX/figures/"
echo "  - Group classic ERP plots: results/ERPs_new/group/*_erps_classic.png"
echo "  - Group beta time-course plots: results/ERPs_new/group/*_beta_timecourse.png"
echo ""
echo "Plot types generated depend on config.yaml plotting.plot_type setting:"
echo "  - 'classic_erp': Traditional ERP averages (option A)"
echo "  - 'beta_timecourse': Time-course of LMM betas (option B)"
echo "  - 'both': Both plot types"
echo "END:   $(date '+%Y-%m-%d %H:%M:%S')"
