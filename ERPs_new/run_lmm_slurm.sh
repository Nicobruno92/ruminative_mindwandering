#!/bin/bash
## SLURM directives
#SBATCH --job-name=ERPsLMM
#SBATCH --output=logs/erps_lmm_%j.out
#SBATCH --error=logs/erps_lmm_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

mkdir -p logs

module load proxy || true
module load julia

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
echo "Running LMM analysis on individual probe epochs"

# Install statsmodels if not available
python -c "import statsmodels" 2>/dev/null || pip install --user statsmodels

srun --cpu-bind=none --ntasks=1 python ERPs_new/lmm_analysis.py \
  --config ERPs_new/config.yaml

echo "LMM analysis completed"
echo "END:   $(date '+%Y-%m-%d %H:%M:%S')"
