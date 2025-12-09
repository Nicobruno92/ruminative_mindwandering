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

# Run Python and capture both stdout and stderr separately
# NOTE: Using lightweight marker counting (reads only one CSV header per marker type)
# instead of loading all data, to avoid OOM on login nodes
PYTHON_OUTPUT=$(python -c "
import yaml
import sys
import os
from pathlib import Path
from collections import defaultdict

try:
    # Load config
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)

    features_root = config['project']['features_root']
    marker_types_filter = config['project'].get('marker_types', None)
    markers_config = config['project'].get('markers', 'all')

    # Lightweight marker discovery: scan filenames and read ONE csv header per type
    # instead of loading all data into memory
    features_path = Path(features_root)
    
    # Find all CSV files matching the pattern
    pattern = '**/sub-*_task-*_desc-probe-*_*_aggMarkers.csv'
    csv_files = list(features_path.glob(pattern))
    
    if len(csv_files) == 0:
        print(f'No CSV files found in {features_root}', file=sys.stderr)
        sys.exit(1)
    
    # Group files by marker type and find one sample file per type
    sample_files_by_type = {}
    for f in csv_files:
        filename = f.name
        if '_aggMarkers.csv' in filename:
            marker_type = filename.split('_desc-probe-')[1].split('_aggMarkers.csv')[0]
            if '_' in marker_type:
                marker_type = marker_type.split('_')[-1]
            else:
                marker_type = 'mixed'
            
            if marker_types_filter is None or marker_type in marker_types_filter:
                if marker_type not in sample_files_by_type:
                    sample_files_by_type[marker_type] = f
    
    # Read marker names from ONE sample file per type (memory efficient)
    import pandas as pd
    markers_by_type = {}
    for marker_type, sample_file in sample_files_by_type.items():
        # Read only the 'marker' column from one file
        df = pd.read_csv(sample_file, usecols=['marker'])
        markers_by_type[marker_type] = sorted(df['marker'].unique().tolist())
    
    # Count markers
    if isinstance(markers_config, str) and markers_config.lower() == 'all':
        n_markers = sum(len(markers) for markers in markers_by_type.values())
    elif isinstance(markers_config, list):
        n_markers = 0
        for marker_name in markers_config:
            for marker_type, type_markers in markers_by_type.items():
                if marker_name in type_markers:
                    n_markers += 1
    else:
        n_markers = 0

    print(n_markers)
except Exception as e:
    import traceback
    print(f'PYTHON_ERROR: {e}', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
" 2>&1)

PYTHON_EXIT_CODE=$?

# Check if Python failed
if [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Python script failed!"
    echo "Exit code: ${PYTHON_EXIT_CODE}"
    echo "Python executable: $(which python)"
    echo "Python version: $(python --version 2>&1)"
    echo "Working directory: $(pwd)"
    echo "--- Python output ---"
    echo "${PYTHON_OUTPUT}"
    echo "--- End output ---"
    exit 1
fi

# Extract the number (last line of output, in case there are warnings)
N_MARKERS=$(echo "${PYTHON_OUTPUT}" | tail -n 1)

if [ -z "${N_MARKERS}" ] || ! [[ "${N_MARKERS}" =~ ^[0-9]+$ ]] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "ERROR: Could not determine number of markers or no markers found!"
    echo "Python output: ${PYTHON_OUTPUT}"
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

# Get model folder name and output path from config
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

# ========================================
# STEP 1: Submit MCC post-processing job
# ========================================
echo "Submitting multiple comparisons correction job (will run after all markers complete)..."

# Create MCC post-processing script
MCC_SCRIPT="Statistics/run_mcc_postprocessing.sh"

cat > ${MCC_SCRIPT} << 'EOFMCC'
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
echo "Model directory: MODEL_DIR_PLACEHOLDER"
echo ""

python Statistics/apply_mcc_postprocessing.py MODEL_DIR_PLACEHOLDER --config Statistics/config.yaml

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✓ MCC post-processing completed successfully at $(date)"
else
    echo ""
    echo "✗ MCC post-processing failed with exit code ${EXIT_CODE} at $(date)"
fi

echo "=========================================="
EOFMCC

# Replace placeholder with actual model directory
sed -i "s|MODEL_DIR_PLACEHOLDER|${MODEL_DIR}|g" ${MCC_SCRIPT}

# Make script executable
chmod +x ${MCC_SCRIPT}

# Submit MCC job with dependency on array job completion
MCC_JOB_OUTPUT=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} ${MCC_SCRIPT})
MCC_EXIT_CODE=$?

if [ ${MCC_EXIT_CODE} -eq 0 ]; then
    MCC_JOB_ID=$(echo ${MCC_JOB_OUTPUT} | awk '{print $NF}')
    echo "✓ MCC post-processing job submitted successfully"
    echo "  Job ID: ${MCC_JOB_ID}"
    echo "  Dependency: Will run after job ${ARRAY_JOB_ID} completes"
    echo ""
else
    echo "⚠ Warning: Failed to submit MCC post-processing job"
    echo "  You can run MCC manually later with:"
    echo "  python Statistics/apply_mcc_postprocessing.py ${MODEL_DIR}"
    echo ""
    MCC_JOB_ID=""
fi

# ========================================
# STEP 2: Submit report generation job
# ========================================
echo "Submitting report generation job (will run after MCC completes)..."

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

# Submit report job with dependency on MCC job completion (if MCC job was submitted)
if [ -n "${MCC_JOB_ID}" ]; then
    REPORT_JOB_OUTPUT=$(sbatch --dependency=afterok:${MCC_JOB_ID} ${REPORT_SCRIPT})
    DEPENDENCY_MSG="Will run after MCC job ${MCC_JOB_ID} completes"
else
    # Fallback: depend on array job if MCC job failed to submit
    REPORT_JOB_OUTPUT=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} ${REPORT_SCRIPT})
    DEPENDENCY_MSG="Will run after array job ${ARRAY_JOB_ID} completes (MCC skipped)"
fi

REPORT_EXIT_CODE=$?

if [ ${REPORT_EXIT_CODE} -eq 0 ]; then
    REPORT_JOB_ID=$(echo ${REPORT_JOB_OUTPUT} | awk '{print $NF}')
    echo "✓ Report generation job submitted successfully"
    echo "  Job ID: ${REPORT_JOB_ID}"
    echo "  Dependency: ${DEPENDENCY_MSG}"
    echo ""
else
    echo "⚠ Warning: Failed to submit report generation job"
    echo "  You can generate the report manually later with:"
    echo "  python Statistics/generate_summary_report.py ${MODEL_DIR}"
    echo ""
fi

echo "Monitor jobs with: squeue -u $USER"
echo "Check logs in:"
echo "  - Array jobs: logs/lmm_marker_${ARRAY_JOB_ID}_*.out"
if [ -n "${MCC_JOB_ID}" ]; then
    echo "  - MCC job: logs/lmm_mcc_${MCC_JOB_ID}.out"
fi
echo "  - Report job: logs/lmm_report_*.out"

echo "=========================================="
