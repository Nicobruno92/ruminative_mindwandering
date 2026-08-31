#!/bin/bash
#SBATCH --job-name=mw_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mw_report_%j.out
#SBATCH --error=logs/mw_report_%j.err

# Reusable report job: MCC post-processing + summary report for a single,
# already-complete Andrillon model directory.
#
# Usage:
#   sbatch Stats_andrillon/run_andrillon_report.sh <MODEL_DIR> [CONFIG_FILE]
#
#   MODEL_DIR    Path to results/andrillon_cluster/<model_folder> (absolute or
#                relative to the project root).
#   CONFIG_FILE  Optional; defaults to Stats_andrillon/config_andrillon.yaml.

MODEL_DIR="$1"
CONFIG_FILE="${2:-Stats_andrillon/config_andrillon.yaml}"

if [ -z "${MODEL_DIR}" ]; then
    echo "ERROR: MODEL_DIR argument is required."
    echo "Usage: sbatch Stats_andrillon/run_andrillon_report.sh <MODEL_DIR> [CONFIG_FILE]"
    exit 2
fi

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

cd <PATH_TO_YOUR_REPO_ROOT>

if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

echo "=========================================="
echo "RUNNING MCC POST-PROCESSING"
echo "=========================================="
python Statistics/apply_mcc_postprocessing.py "${MODEL_DIR}" \
    --config "${CONFIG_FILE}"
MCC_EXIT=$?

echo ""
echo "=========================================="
echo "GENERATING SUMMARY REPORT"
echo "=========================================="
echo "Start time: $(date)"
echo "Model directory: ${MODEL_DIR}"
echo ""

python Statistics/generate_summary_report.py "${MODEL_DIR}"
REPORT_EXIT=$?

# Fail the job if either step failed, so sacct records it and any afterok
# dependency is respected.
if [ ${MCC_EXIT} -ne 0 ]; then
    EXIT_CODE=${MCC_EXIT}
else
    EXIT_CODE=${REPORT_EXIT}
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Report generation completed successfully at $(date)"
else
    echo ""
    echo "✗ Report generation failed (MCC=${MCC_EXIT}, report=${REPORT_EXIT}) at $(date)"
fi

echo "=========================================="
exit ${EXIT_CODE}
