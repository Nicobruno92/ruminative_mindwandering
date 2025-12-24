#!/bin/bash
#SBATCH --job-name=andrillon_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=logs/andrillon_report_%j.out
#SBATCH --error=logs/andrillon_report_%j.err

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

export MPLBACKEND=Agg

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "GENERATING ANDRILLON SUMMARY REPORT"
echo "=========================================="
echo "Start time: $(date)"
echo "Model directory: /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/andrillon_cluster/onoff_valence_selfother_time_time_on_task_confidence"
echo ""

python Statistics/generate_summary_report.py /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/andrillon_cluster/onoff_valence_selfother_time_time_on_task_confidence

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Andrillon report generation completed successfully at $(date)"
else
    echo ""
    echo "✗ Andrillon report generation failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
