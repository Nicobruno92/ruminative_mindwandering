#!/bin/bash
#SBATCH --job-name=mw_marker
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/mw_marker_%A_%a.out
#SBATCH --error=logs/mw_marker_%A_%a.err

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

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "MARKER JOB"
echo "=========================================="
echo "Start time: $(date)"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo ""

# Build arguments for Python script
CLI_ARGS="--config Stats_andrillon/config_andrillon.yaml --marker-index ${SLURM_ARRAY_TASK_ID}"

# PREDICTOR_OF_INTEREST may be injected via --export by the submit script
if [ -n "${PREDICTOR_OF_INTEREST:-}" ]; then
    CLI_ARGS="${CLI_ARGS} --predictor-of-interest ${PREDICTOR_OF_INTEREST}"
    echo "Predictor of interest: ${PREDICTOR_OF_INTEREST}"
fi

python Stats_andrillon/run_andrillon_pipeline.py ${CLI_ARGS}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Marker job completed successfully at $(date)"
else
    echo ""
    echo "✗ Marker job failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="

# Propagate the Python exit code so SLURM records FAILED (not COMPLETED 0:0)
# when a marker dies. This is what makes the report's afterok dependency
# meaningful: a report is only generated when every marker actually succeeded.
exit ${EXIT_CODE}
