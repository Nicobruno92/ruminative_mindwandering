#!/bin/bash
## SLURM directives
#SBATCH --job-name=ERPfigs
#SBATCH --output=logs/erp_complete_%j.out
#SBATCH --error=logs/erp_complete_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

mkdir -p logs

module load proxy || true

echo "Attempting to activate Python environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
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
echo "Running complete ERP analysis: individual participants + group figures"

# Run the complete pipeline: individual subjects + group aggregation
srun --cpu-bind=none --ntasks=1 python ERPs_new/make_erp_figures.py \
  --config ERPs_new/config.yaml

echo "Complete ERP analysis finished"
echo "END:   $(date '+%Y-%m-%d %H:%M:%S')"

