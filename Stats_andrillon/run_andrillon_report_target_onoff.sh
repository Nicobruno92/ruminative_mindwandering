#!/bin/bash
#SBATCH --job-name=mw_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mw_report_%j.out
#SBATCH --error=logs/mw_report_%j.err

# One-off report job for the `onoff` predictor model (target_onoff), submitted
# after re-running its 9 previously-failed markers (array job for indices
# 17,18,21,24,28,30,35,55,56). Runs only on afterok, so it will not regenerate
# a report on incomplete data.

# Load modules
module load proxy

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
fi

# Ensure the env's python wins on PATH
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

export MPLBACKEND=Agg

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

MODEL_DIR="results/andrillon_cluster/onoff_valence_selfother_time_time_on_task_confidence__target_onoff"
CONFIG_FILE="Stats_andrillon/config_andrillon.yaml"

echo "=========================================="
echo "RUNNING MCC POST-PROCESSING"
echo "=========================================="
python Statistics/apply_mcc_postprocessing.py ${MODEL_DIR} \
    --config ${CONFIG_FILE}

echo ""
echo "=========================================="
echo "GENERATING SUMMARY REPORT"
echo "=========================================="
echo "Start time: $(date)"
echo "Model directory: ${MODEL_DIR}"
echo ""

python Statistics/generate_summary_report.py ${MODEL_DIR}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Report generation completed successfully at $(date)"
else
    echo ""
    echo "✗ Report generation failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
exit ${EXIT_CODE}
