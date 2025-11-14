#!/bin/bash
# Helper script to submit SLURM array job for parallel Andrillon-marker processing
# This script automatically determines the number of markers from config_andrillon.yaml
# and submits an array job with the correct number of tasks, followed by a
# dependent summary-report generation job.

# Configuration
CONFIG_FILE="Stats_andrillon/config_andrillon.yaml"
SCRIPT_DIR="Stats_andrillon"

# Set working directory to project root on the cluster
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

# Ensure logs directory exists (required for sbatch output/error files)
mkdir -p logs

echo "=========================================="
echo "Andrillon Parallel Marker Submission"
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
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate eeg
fi

# Get number of markers from config using Python (via get_marker_list)
echo "Detecting number of markers from Andrillon config..."
N_MARKERS=$(python -c "
import yaml
import sys
from pathlib import Path

sys.path.append('${SCRIPT_DIR}')
from andrillon_pipeline import get_marker_list

CONFIG_PATH = Path('${CONFIG_FILE}')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

markers = get_marker_list(config)
print(len(markers))
")

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

cat > ${ARRAY_SCRIPT} << 'EOFARRAY'
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
EOFARRAY

# Make array script executable
chmod +x ${ARRAY_SCRIPT}

# Submit the array job and capture the job ID
ARRAY_JOB_OUTPUT=$(sbatch --array=${ARRAY_RANGE} ${ARRAY_SCRIPT})
ARRAY_EXIT_CODE=$?

if [ ${ARRAY_EXIT_CODE} -ne 0 ]; then
    echo "✗ Failed to submit Andrillon array job"
    exit 1
fi

# Extract job ID from sbatch output
ARRAY_JOB_ID=$(echo ${ARRAY_JOB_OUTPUT} | awk '{print $NF}')

echo "✓ Andrillon array job submitted successfully"
echo "  Job ID: ${ARRAY_JOB_ID}"
echo "  Array range: ${ARRAY_RANGE}"
echo "  Total tasks: ${N_MARKERS}"
echo ""

# Get model folder name and output path from Andrillon config
MODEL_FOLDER=$(python -c "
import yaml
from pathlib import Path

CONFIG_PATH = Path('${CONFIG_FILE}')
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

formula = config['lmm']['formula']
fixed_effects = formula.split('~')[1].split('(')[0].strip()
predictors = [p.strip() for p in fixed_effects.split('+') if p.strip()]
model_folder = '_'.join(predictors)
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
echo "Submitting Andrillon summary report job (will run after all markers complete)..."

REPORT_SCRIPT="${SCRIPT_DIR}/run_andrillon_report_generation.sh"

cat > ${REPORT_SCRIPT} << 'EOFREPORT'
#!/bin/bash
#SBATCH --job-name=andrillon_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
echo "Model directory: MODEL_DIR_PLACEHOLDER"
echo ""

python Statistics/generate_summary_report.py MODEL_DIR_PLACEHOLDER

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Andrillon report generation completed successfully at $(date)"
else
    echo ""
    echo "✗ Andrillon report generation failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
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
    echo "✓ Andrillon report generation job submitted successfully"
    echo "  Job ID: ${REPORT_JOB_ID}"
    echo "  Dependency: Will run after job ${ARRAY_JOB_ID} completes"
    echo ""
else
    echo "⚠ Warning: Failed to submit Andrillon report generation job"
    echo "  You can generate the report manually later with:"
    echo "  python Statistics/generate_summary_report.py ${MODEL_DIR}"
    echo ""
fi

echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in:"
echo "  - Array jobs: logs/andrillon_marker_${ARRAY_JOB_ID}_*.out"
echo "  - Report job: logs/andrillon_report_*.out"

echo "=========================================="
