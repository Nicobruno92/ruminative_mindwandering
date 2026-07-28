#!/bin/bash
# Helper script to submit SLURM array job for parallel Andrillon-marker processing
# This script automatically determines the number of markers from config_andrillon.yaml
# and submits an array job with the correct number of tasks, followed by a
# dependent summary-report generation job.

# Configuration
CONFIG_FILE="Stats_andrillon/config_andrillon.yaml"
SCRIPT_DIR="Stats_andrillon"

# ---------------------------------------------------------------------------
# Optional arguments:
#   --predictor <name>  Override lmm.predictor_of_interest
#   --config <path>     Use custom config file
# ---------------------------------------------------------------------------
PREDICTOR_OVERRIDE=""
CONFIG_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictor)
            PREDICTOR_OVERRIDE="$2"
            shift 2
            ;;
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash Stats_andrillon/submit_parallel_andrillon_markers.sh [--predictor <name>] [--config <path>]"
            exit 1
            ;;
    esac
done

# Use custom config if provided
if [ -n "${CONFIG_OVERRIDE}" ]; then
    CONFIG_FILE="${CONFIG_OVERRIDE}"
fi

# Set working directory to project root on the cluster
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Ensure logs directory exists (required for sbatch output/error files)
mkdir -p logs

echo "=========================================="
echo "Parallel Marker Submission"
if [ -n "${PREDICTOR_OVERRIDE}" ]; then
    echo "Predictor of interest: ${PREDICTOR_OVERRIDE}"
fi
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Activate conda environment to run Python
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

# Ensure the env's python wins on PATH (some shells prepend /usr/bin after activate)
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
fi

# Get number of markers from config using Python (via get_marker_list)
echo "Detecting number of markers from config..."
N_MARKERS=$(python -c "
import yaml
import sys
import logging
from pathlib import Path

# Suppress logging to avoid polluting stdout
logging.disable(logging.CRITICAL)

sys.path.append('${SCRIPT_DIR}')
from andrillon_pipeline import get_marker_list

CONFIG_PATH = Path('${CONFIG_FILE}')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

markers = get_marker_list(config)
print(len(markers))
")
PYTHON_EXIT_CODE=$?

if [ ${PYTHON_EXIT_CODE} -ne 0 ] || [ -z "${N_MARKERS}" ]; then
    echo "ERROR: Python failed to detect markers (see traceback above)"
    exit 1
fi

if [ -z "${N_MARKERS}" ] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "ERROR: Could not determine number of markers or no markers found!"
    echo "Please check your ${CONFIG_FILE} file."
    exit 1
fi

echo "Found ${N_MARKERS} markers to process"
echo "Submitting SLURM array job with ${N_MARKERS} tasks..."

# Array indices: 0 to N_MARKERS-1
ARRAY_RANGE="0-$((N_MARKERS-1))"

# Create Andrillon array job script
ARRAY_SCRIPT="${SCRIPT_DIR}/run_andrillon_marker_array.sh"

cat > ${ARRAY_SCRIPT} << EOFARRAY
#!/bin/bash
#SBATCH --job-name=mw_marker
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=logs/mw_marker_%A_%a.out
#SBATCH --error=logs/mw_marker_%A_%a.err

# Load modules
module load proxy

# Activate conda environment
if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "\$(conda shell.bash hook)"
  conda activate eeg
fi

# Ensure the env's python wins on PATH
if [ -n "\${CONDA_PREFIX:-}" ]; then
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
fi

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "MARKER JOB"
echo "=========================================="
echo "Start time: \$(date)"
echo "Task ID: \${SLURM_ARRAY_TASK_ID}"
echo ""

# Build arguments for Python script
CLI_ARGS="--config ${CONFIG_FILE} --marker-index \${SLURM_ARRAY_TASK_ID}"

# PREDICTOR_OF_INTEREST may be injected via --export by the submit script
if [ -n "\${PREDICTOR_OF_INTEREST:-}" ]; then
    CLI_ARGS="\${CLI_ARGS} --predictor-of-interest \${PREDICTOR_OF_INTEREST}"
    echo "Predictor of interest: \${PREDICTOR_OF_INTEREST}"
fi

python Stats_andrillon/run_andrillon_pipeline.py \${CLI_ARGS}

EXIT_CODE=\$?

if [ \${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Marker job completed successfully at \$(date)"
else
    echo ""
    echo "✗ Marker job failed with exit code \${EXIT_CODE} at \$(date)"
fi

echo "=========================================="

# Propagate the Python exit code so SLURM records FAILED (not COMPLETED 0:0)
# when a marker dies. This is what makes the report's afterok dependency
# meaningful: a report is only generated when every marker actually succeeded.
exit \${EXIT_CODE}
EOFARRAY

# Make array script executable
chmod +x ${ARRAY_SCRIPT}

# Submit the array job and capture the job ID
# Propagate the predictor override via SLURM --export so each array task knows it
SBATCH_EXPORT="ALL"
if [ -n "${PREDICTOR_OVERRIDE}" ]; then
    SBATCH_EXPORT="ALL,PREDICTOR_OF_INTEREST=${PREDICTOR_OVERRIDE}"
fi
ARRAY_JOB_OUTPUT=$(sbatch --array=${ARRAY_RANGE} --export=${SBATCH_EXPORT} --cpus-per-task=16 --mem=24G --time=8:00:00 ${ARRAY_SCRIPT})
ARRAY_EXIT_CODE=$?

if [ ${ARRAY_EXIT_CODE} -ne 0 ]; then
    echo "✗ Failed to submit Andrillon array job"
    exit 1
fi

# Extract job ID from sbatch output
ARRAY_JOB_ID=$(echo ${ARRAY_JOB_OUTPUT} | awk '{print $NF}')

echo "✓ Array job submitted successfully"
echo "  Job ID: ${ARRAY_JOB_ID}"
echo "  Array range: ${ARRAY_RANGE}"
echo "  Total tasks: ${N_MARKERS}"
echo ""

# Get model folder name and output path from Andrillon config.
# IMPORTANT: resolve the formula via _resolve_formula_for_predictor so that
# per-predictor overrides (e.g. the interaction model for "onoff:confidence")
# produce the SAME folder name the marker pipeline writes to. Using the default
# formula here makes the report point at a non-existent additive-named dir.
MODEL_FOLDER=$(python -c "
import yaml
import sys
import os
from pathlib import Path
sys.path.append(os.path.join('${SCRIPT_DIR}', '..', 'Statistics'))
sys.path.append('${SCRIPT_DIR}')
from helpers import get_model_folder_name
from andrillon_pipeline import _resolve_formula_for_predictor

CONFIG_PATH = Path('${CONFIG_FILE}')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

predictor = '${PREDICTOR_OVERRIDE}' or config['lmm'].get('predictor_of_interest', 'auto')
if predictor and predictor.strip().lower() != 'auto':
    formula = _resolve_formula_for_predictor(config, predictor)
else:
    formula = config['lmm']['formula']
model_folder = get_model_folder_name(formula, predictor)
print(model_folder)
")

OUTPUT_PATH=$(python -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config['project']['output_path'])
")

MODEL_DIR="${OUTPUT_PATH}/${MODEL_FOLDER}"

# ========================================
# STEP 2: Submit report generation job
# ========================================
echo "Submitting summary report job (will run after all markers complete)..."

REPORT_SCRIPT="${SCRIPT_DIR}/run_andrillon_report_generation.sh"

cat > ${REPORT_SCRIPT} << EOFREPORT
#!/bin/bash
#SBATCH --job-name=mw_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mw_report_%j.out
#SBATCH --error=logs/mw_report_%j.err

# Load modules
module load proxy

# Activate conda environment
if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate eeg
elif [ -f "\$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "\$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate eeg
elif command -v conda >/dev/null 2>&1; then
  eval "\$(conda shell.bash hook)"
  conda activate eeg
fi

# Ensure the env's python wins on PATH
if [ -n "\${CONDA_PREFIX:-}" ]; then
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
fi

export MPLBACKEND=Agg

cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "RUNNING MCC POST-PROCESSING"
echo "=========================================="
python Statistics/apply_mcc_postprocessing.py MODEL_DIR_PLACEHOLDER \
    --config ${CONFIG_FILE}

echo ""
echo "=========================================="
echo "GENERATING SUMMARY REPORT"
echo "=========================================="
echo "Start time: \$(date)"
echo "Model directory: MODEL_DIR_PLACEHOLDER"
echo ""

python Statistics/generate_summary_report.py MODEL_DIR_PLACEHOLDER

EXIT_CODE=\$?

if [ \${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Report generation completed successfully at \$(date)"
else
    echo ""
    echo "✗ Report generation failed with exit code \${EXIT_CODE} at \$(date)"
fi

echo "=========================================="

# Propagate the exit code so a failed report is recorded as FAILED in sacct.
exit \${EXIT_CODE}
EOFREPORT

# Replace placeholder with actual model directory
sed -i "s|MODEL_DIR_PLACEHOLDER|${MODEL_DIR}|g" ${REPORT_SCRIPT}

# Make report script executable
chmod +x ${REPORT_SCRIPT}

# Submit report job with dependency on array job completion
REPORT_JOB_OUTPUT=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} ${REPORT_SCRIPT})
REPORT_EXIT_CODE=$?

if [ ${REPORT_EXIT_CODE} -eq 0 ]; then
    REPORT_JOB_ID=$(echo ${REPORT_JOB_OUTPUT} | awk '{print $NF}')
    echo "✓ Report generation job submitted successfully"
    echo "  Job ID: ${REPORT_JOB_ID}"
    echo "  Dependency: Will run after job ${ARRAY_JOB_ID} completes"
    echo ""
else
    echo "⚠ Warning: Failed to submit report generation job"
    echo "  You can generate the report manually later with:"
    echo "  python Statistics/generate_summary_report.py ${MODEL_DIR}"
    echo ""
fi

echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in:"
echo "  - Array jobs: logs/mw_marker_${ARRAY_JOB_ID}_*.out"
echo "  - Report job: logs/mw_report_*.out"

echo "=========================================="
