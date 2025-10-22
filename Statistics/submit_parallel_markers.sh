#!/bin/bash
# Helper script to submit SLURM array job for parallel marker processing
# This script automatically determines the number of markers from config.yaml
# and submits an array job with the correct number of tasks

# Configuration
CONFIG_FILE="Statistics/config.yaml"
ARRAY_SCRIPT="Statistics/submit_marker_array.sh"
SCRIPT_DIR="Statistics"

# Set working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "LMM Parallel Marker Submission"
echo "=========================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# Check if array script exists
if [ ! -f "${ARRAY_SCRIPT}" ]; then
    echo "ERROR: Array script ${ARRAY_SCRIPT} not found!"
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
marker_types = config['project'].get('marker_types', None)
markers_config = config['project'].get('markers', 'all')

# Determine number of marker-type combinations
if isinstance(markers_config, str) and markers_config.lower() == 'all':
    # Get all markers, counting each (marker, type) combination separately
    available_markers_dict = get_available_markers(features_root, marker_types)
    n_markers = sum(len(markers) for markers in available_markers_dict.values())
elif isinstance(markers_config, list):
    # For each marker, count how many types it exists in
    available_markers_dict = get_available_markers(features_root, marker_types)
    n_markers = 0
    for marker_name in markers_config:
        for marker_type, type_markers in available_markers_dict.items():
            if marker_name in type_markers:
                n_markers += 1
else:
    n_markers = 0

print(n_markers)
")

if [ -z "${N_MARKERS}" ] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "ERROR: Could not determine number of markers or no markers found!"
    echo "Please check your config.yaml file."
    exit 1
fi

echo "Found ${N_MARKERS} markers to process"
echo "Submitting SLURM array job with ${N_MARKERS} tasks..."

# Submit array job with correct number of tasks
# Array indices: 0 to N_MARKERS-1
ARRAY_RANGE="0-$((N_MARKERS-1))"

# Submit the array job and capture the job ID
ARRAY_JOB_OUTPUT=$(sbatch --array=${ARRAY_RANGE} ${ARRAY_SCRIPT})
ARRAY_EXIT_CODE=$?

if [ ${ARRAY_EXIT_CODE} -ne 0 ]; then
    echo "✗ Failed to submit array job"
    exit 1
fi

# Extract job ID from sbatch output (format: "Submitted batch job 12345")
ARRAY_JOB_ID=$(echo ${ARRAY_JOB_OUTPUT} | awk '{print $NF}')

echo "✓ Array job submitted successfully"
echo "  Job ID: ${ARRAY_JOB_ID}"
echo "  Array range: ${ARRAY_RANGE}"
echo "  Total tasks: ${N_MARKERS}"
echo ""

# Submit report generation job that depends on array job completion
echo "Submitting report generation job (will run after all markers complete)..."

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
    echo "  bash Statistics/create_report.sh ${MODEL_DIR}"
    echo ""
fi

echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in:"
echo "  - Array jobs: logs/lmm_marker_${ARRAY_JOB_ID}_*.out"
echo "  - Report job: logs/lmm_report_*.out"

echo "=========================================="
