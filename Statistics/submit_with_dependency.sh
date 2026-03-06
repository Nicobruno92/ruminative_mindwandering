#!/bin/bash

# =============================================================================
# Submit Statistics Job with Dependency
# =============================================================================
# This script allows you to manually submit the statistics analysis
# with a dependency on another job (e.g., the aggregation step).
#
# Usage:
#   ./submit_with_dependency.sh <JOB_ID>
#
# Example:
#   ./submit_with_dependency.sh 1179400
#
# This will submit the statistics job to run after job 1179400 completes.
# =============================================================================

set -euo pipefail

# Check if job ID was provided
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "ERROR: No job ID provided"
    echo "=========================================="
    echo "Usage: $0 <JOB_ID>"
    echo ""
    echo "Example:"
    echo "  $0 1179400"
    echo ""
    echo "This will submit the statistics job to run after the specified job completes."
    exit 1
fi

DEPENDENCY_JOB_ID=$1

# Configuration
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering}
CONFIG_FILE="Statistics/config.yaml"
ARRAY_SCRIPT="Statistics/submit_marker_array.sh"
SCRIPT_DIR="Statistics"

cd "$WORKDIR"

echo "=========================================="
echo "Submit Statistics with Dependency"
echo "=========================================="
echo "Dependency Job ID: ${DEPENDENCY_JOB_ID}"
echo "Working directory: ${WORKDIR}"
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Check if array script exists
if [ ! -f "${ARRAY_SCRIPT}" ]; then
    echo "❌ ERROR: Array script ${ARRAY_SCRIPT} not found!"
    exit 1
fi

# Check if dependency job exists
if ! squeue -j ${DEPENDENCY_JOB_ID} &>/dev/null && ! sacct -j ${DEPENDENCY_JOB_ID} --format=JobID -n &>/dev/null; then
    echo "⚠ WARNING: Job ${DEPENDENCY_JOB_ID} not found in queue or history"
    echo "Proceeding anyway (job may have completed or been cancelled)"
    echo ""
fi

# Activate conda environment to run Python
echo "Activating conda environment..."
set +u
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
set -u

# Get number of markers from config using Python
echo "Detecting number of markers from config..."
N_MARKERS=$(python -c "
import yaml
import sys
sys.path.append('${SCRIPT_DIR}')
from reader import get_available_markers

# Load config
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

features_root = config['project']['features_root']
selected_markers = config['project'].get('selected_markers', {})

# Determine number of marker-type combinations
n_markers = 0
if selected_markers:
    available_markers_dict = get_available_markers(features_root)
    for marker_type, configured_markers in selected_markers.items():
        if marker_type in available_markers_dict:
            if isinstance(configured_markers, str) and configured_markers.lower() == 'all':
                n_markers += len(available_markers_dict[marker_type])
            elif isinstance(configured_markers, list):
                # Only count markers that actually exist
                available_type_markers = available_markers_dict[marker_type]
                n_markers += sum(1 for m in configured_markers if m in available_type_markers)

print(n_markers)
")

if [ -z "${N_MARKERS}" ] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "❌ ERROR: Could not determine number of markers or no markers found!"
    echo "Please check your config.yaml file."
    exit 1
fi

echo "✓ Found ${N_MARKERS} markers to process"
echo ""

# Submit array job with dependency
ARRAY_RANGE="0-$((N_MARKERS-1))"

echo "=========================================="
echo "Submitting Statistics Job"
echo "=========================================="
echo "Array range: ${ARRAY_RANGE}"
echo "Total tasks: ${N_MARKERS}"
echo "Dependency: afterok:${DEPENDENCY_JOB_ID}"
echo ""

# Submit the array job with dependency and capture the job ID
ARRAY_JOB_OUTPUT=$(sbatch --parsable --dependency=afterok:${DEPENDENCY_JOB_ID} --array=${ARRAY_RANGE} ${ARRAY_SCRIPT})
ARRAY_EXIT_CODE=$?

if [ ${ARRAY_EXIT_CODE} -ne 0 ]; then
    echo "❌ Failed to submit array job"
    exit 1
fi

ARRAY_JOB_ID=${ARRAY_JOB_OUTPUT}

echo "✓ Statistics job submitted successfully"
echo "  Job ID: ${ARRAY_JOB_ID}"
echo "  Will start after Job ${DEPENDENCY_JOB_ID} completes"
echo ""

# Submit report generation job that depends on array job completion
echo "=========================================="
echo "Submitting Report Generation"
echo "=========================================="
echo "Dependency: afterok:${ARRAY_JOB_ID}"
echo ""

# Get model folder name from config
MODEL_FOLDER=$(python -c "
import yaml
import sys
sys.path.append('${SCRIPT_DIR}')
from helpers import extract_fixed_effects_from_formula

with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

formula = config['lmm']['formula']
model_folder = extract_fixed_effects_from_formula(formula)
print(model_folder)
")

OUTPUT_PATH=$(python -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config['project']['output_path'])
")

MODEL_DIR="${OUTPUT_PATH}/${MODEL_FOLDER}"

# Create report generation script
REPORT_SCRIPT="Statistics/run_report_generation.sh"

cat > ${REPORT_SCRIPT} << 'EOFSCRIPT'
#!/bin/bash
#SBATCH --job-name=lmm_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/lmm_report_%j.out
#SBATCH --error=logs/lmm_report_%j.err

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
echo "GENERATING SUMMARY REPORT"
echo "=========================================="
echo "Start time: $(date)"
echo "Model directory: MODEL_DIR_PLACEHOLDER"
echo ""

python Statistics/generate_summary_report.py MODEL_DIR_PLACEHOLDER

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ Report generation completed successfully at $(date)"
else
    echo ""
    echo "✗ Report generation failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
EOFSCRIPT

# Replace placeholder with actual model directory
sed -i "s|MODEL_DIR_PLACEHOLDER|${MODEL_DIR}|g" ${REPORT_SCRIPT}

# Make script executable
chmod +x ${REPORT_SCRIPT}

# Submit report job with dependency on array job completion
REPORT_JOB_OUTPUT=$(sbatch --parsable --dependency=afterok:${ARRAY_JOB_ID} ${REPORT_SCRIPT})
REPORT_EXIT_CODE=$?

if [ ${REPORT_EXIT_CODE} -eq 0 ]; then
    REPORT_JOB_ID=${REPORT_JOB_OUTPUT}
    echo "✓ Report generation job submitted successfully"
    echo "  Job ID: ${REPORT_JOB_ID}"
    echo "  Will run after Job ${ARRAY_JOB_ID} completes"
    echo ""
else
    echo "⚠ Warning: Failed to submit report generation job"
    echo "  You can generate the report manually later with:"
    echo "  bash Statistics/create_report.sh ${MODEL_DIR}"
    echo ""
fi

echo "=========================================="
echo "Job Chain Summary"
echo "=========================================="
echo "Dependency chain:"
echo "  Job ${DEPENDENCY_JOB_ID} (external) → Job ${ARRAY_JOB_ID} (statistics) → Job ${REPORT_JOB_ID} (report)"
echo ""
echo "Monitor jobs:"
echo "  squeue -j ${DEPENDENCY_JOB_ID},${ARRAY_JOB_ID},${REPORT_JOB_ID}"
echo ""
echo "Check logs:"
echo "  - Statistics: logs/lmm_marker_${ARRAY_JOB_ID}_*.out"
echo "  - Report: logs/lmm_report_${REPORT_JOB_ID}.out"
echo ""
echo "=========================================="
