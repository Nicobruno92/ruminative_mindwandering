#!/bin/bash
# Helper script to submit SLURM array job for parallel marker processing
# This script automatically determines the number of markers from config.yaml
# and submits an array job with the correct number of tasks

# Configuration
CONFIG_FILE="Statistics/config.yaml"
ARRAY_SCRIPT="Statistics/submit_marker_array.sh"
SCRIPT_DIR="Statistics"

# ---------------------------------------------------------------------------
# Optional arguments:
#   --predictor <name>  Override lmm.predictor_of_interest
#   --config <path>     Use custom config file (default: Statistics/config.yaml)
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
            echo "Usage: bash Statistics/submit_parallel_markers.sh [--predictor <name>] [--config <path>]"
            exit 1
            ;;
    esac
done

# Use custom config if provided
if [ -n "${CONFIG_OVERRIDE}" ]; then
    CONFIG_FILE="${CONFIG_OVERRIDE}"
fi

# Set working directory
cd /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

echo "=========================================="
echo "LMM Parallel Marker Submission"
if [ -n "${PREDICTOR_OVERRIDE}" ]; then
    echo "Predictor of interest: ${PREDICTOR_OVERRIDE}"
fi
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
from pathlib import Path

try:
    # Load config
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)

    features_root = config['project']['features_root']
    # selected_markers and feature_families are top-level keys, not under 'project'
    selected_markers = config.get('selected_markers', {})
    feature_families = config.get('feature_families', {})
    
    if not selected_markers:
        print('No selected_markers configured in config.yaml', file=sys.stderr)
        sys.exit(1)
    if not feature_families:
        print('No feature_families configured in config.yaml', file=sys.stderr)
        sys.exit(1)

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
    import re
    import pandas as pd
    sample_files_by_type = {}
    for f in csv_files:
        m = re.search(r'_desc-probe-\d+_(\w+)_aggMarkers\.csv$', f.name)
        if m:
            mtype = m.group(1)
            if mtype not in sample_files_by_type:
                sample_files_by_type[mtype] = f
    
    # Read available marker names from ONE sample file per type (memory-efficient)
    markers_by_type = {}
    for mtype, sample_file in sample_files_by_type.items():
        df = pd.read_csv(sample_file, usecols=['marker'])
        markers_by_type[mtype] = sorted(df['marker'].unique().tolist())
    
    # Count markers: resolve feature_families -> fragments -> substring matching
    # (mirrors the logic in run_pipeline.py)
    n_markers = 0
    for marker_type, family_list in selected_markers.items():
        if marker_type not in markers_by_type:
            continue

        available = markers_by_type[marker_type]

        # Special case: ['all'] -> include every available marker
        if isinstance(family_list, list) and len(family_list) == 1 and str(family_list[0]).lower() == 'all':
            n_markers += len(available)
            continue

        # Collect fragments from all referenced families
        fragments = []
        null_family = False
        for family_name in family_list:
            if family_name not in feature_families:
                print(f'Family "{family_name}" not defined in feature_families', file=sys.stderr)
                sys.exit(1)
            ffrags = feature_families[family_name]
            if ffrags is None:  # null -> all features
                null_family = True
                break
            fragments.extend(ffrags)

        if null_family:
            n_markers += len(available)
        else:
            matched = [mk for mk in available if any(frag in mk for frag in fragments)]
            n_markers += len(matched)

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
# Propagate the predictor override via SLURM --export so each array task knows it
SBATCH_EXPORT="ALL"
if [ -n "${PREDICTOR_OVERRIDE}" ]; then
    SBATCH_EXPORT="ALL,PREDICTOR_OF_INTEREST=${PREDICTOR_OVERRIDE}"
fi
ARRAY_JOB_OUTPUT=$(sbatch --array=${ARRAY_RANGE} --export=${SBATCH_EXPORT} ${ARRAY_SCRIPT})
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
# Use the predictor override (if any) to match the directory that run_pipeline.py will create
MODEL_FOLDER=$(python -c "
import yaml
import sys
sys.path.append('${SCRIPT_DIR}')
from helpers import get_model_folder_name

with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

formula = config['lmm']['formula']
predictor = '${PREDICTOR_OVERRIDE}' or config['lmm'].get('predictor_of_interest', 'auto')
# If config still has a list (not overridden), fall back to 'auto'
if isinstance(predictor, list):
    predictor = 'auto'
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
