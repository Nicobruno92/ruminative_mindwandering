#!/bin/bash
#SBATCH --job-name=andrillon_marker
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=logs/andrillon_marker_%A_%a.out
#SBATCH --error=logs/andrillon_marker_%A_%a.err

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
echo "ANDRILLON MARKER JOB"
echo "=========================================="
echo "Start time: $(date)"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo ""

python Stats_andrillon/run_andrillon_pipeline.py \
  --config Stats_andrillon/config_andrillon.yaml \
  --marker-index ${SLURM_ARRAY_TASK_ID}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Andrillon marker job completed successfully at $(date)"
else
    echo ""
    echo "✗ Andrillon marker job failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
