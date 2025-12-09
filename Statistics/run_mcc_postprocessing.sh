#!/bin/bash
#SBATCH --job-name=lmm_mcc
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/lmm_mcc_%j.out
#SBATCH --error=logs/lmm_mcc_%j.err

# Load modules
module load proxy

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

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "MULTIPLE COMPARISONS CORRECTION"
echo "=========================================="
echo "Start time: $(date)"
echo "Model directory: /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff_valence_selfother_time_time_on_task"
echo ""

python Statistics/apply_mcc_postprocessing.py /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff_valence_selfother_time_time_on_task --config Statistics/config.yaml

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ MCC post-processing completed successfully at $(date)"
else
    echo ""
    echo "✗ MCC post-processing failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
