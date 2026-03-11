#!/bin/bash
# Local execution script for LMM-Based Spatial Cluster Permutation Pipeline
# Replaces SLURM array jobs with a local sequential loop.

set -e  # Exit on error

# 1. Environment Setup
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_AFFINITY=disabled
export ORT_DISABLE_THREAD_AFFINITY=1
export ONNX_DISABLE_THREADPOOL_AFFINITY=1
export ORT_NUM_THREADS=1
export ONNXRUNTIME_NUM_THREADS=1
export MKL_THREADING_LAYER=sequential
export MKL_DYNAMIC=FALSE
export TORCH_NUM_THREADS=1

# Configuration
CONFIG_FILE="Statistics/config.yaml"
SCRIPT_DIR="Statistics"
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Optional argument: --predictor <name>
# When provided, overrides lmm.predictor_of_interest from config.yaml.
# ---------------------------------------------------------------------------
PREDICTOR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictor)
            PREDICTOR_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash Statistics/run_local_pipeline.sh [--predictor <name>]"
            exit 1
            ;;
    esac
done

mkdir -p ${LOG_DIR}

echo "=========================================="
echo "LMM Local Pipeline Execution"
if [ -n "${PREDICTOR_OVERRIDE}" ]; then
    echo "Predictor of interest: ${PREDICTOR_OVERRIDE}"
fi
echo "=========================================="

# Activate Conda
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

# Check config
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

# 2. Detect Markers (Python Snippet from submit_parallel_markers.sh)
echo "Detecting number of markers configuration..."
PYTHON_OUTPUT=$(python -c "
import yaml
import sys
from pathlib import Path

try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)

    features_root = config['project']['features_root']
    selected_markers = config['project'].get('selected_markers', {})
    
    if not selected_markers:
        print('No selected_markers configured in config.yaml', file=sys.stderr)
        sys.exit(1)
        
    # Lightweight marker discovery logic
    features_path = Path(features_root)
    pattern = '**/sub-*_task-*_desc-probe-*_*_aggMarkers.csv'
    csv_files = list(features_path.glob(pattern))
    
    if len(csv_files) == 0:
        print(f'No CSV files found in {features_root}', file=sys.stderr)
        sys.exit(1)
        
    sample_files_by_type = {}
    for f in csv_files:
        filename = f.name
        if '_aggMarkers.csv' in filename:
            marker_type = filename.split('_desc-probe-')[1].split('_aggMarkers.csv')[0]
            if '_' in marker_type:
                marker_type = marker_type.split('_')[-1]
            else:
                marker_type = 'mixed'
            
            if marker_type in selected_markers:
                if marker_type not in sample_files_by_type:
                    sample_files_by_type[marker_type] = f

    import pandas as pd
    markers_by_type = {}
    for marker_type, sample_file in sample_files_by_type.items():
        df = pd.read_csv(sample_file, usecols=['marker'])
        markers_by_type[marker_type] = sorted(df['marker'].unique().tolist())
    
    n_markers = 0
    for marker_type, configured_markers in selected_markers.items():
        if marker_type not in markers_by_type:
            continue
            
        available_type_markers = markers_by_type[marker_type]
        if isinstance(configured_markers, str) and configured_markers.lower() == 'all':
            n_markers += len(available_type_markers)
        elif isinstance(configured_markers, list):
            for marker_name in configured_markers:
                if marker_name in available_type_markers:
                    n_markers += 1
    print(n_markers)
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
")

N_MARKERS=$(echo "${PYTHON_OUTPUT}" | tail -n 1)

if [ -z "${N_MARKERS}" ] || ! [[ "${N_MARKERS}" =~ ^[0-9]+$ ]] || [ "${N_MARKERS}" -eq 0 ]; then
    echo "ERROR: Could not determine number of markers. Python output: ${PYTHON_OUTPUT}"
    exit 1
fi

echo "Found ${N_MARKERS} markers to process."

# 3. Sequential Loop for Markers
echo "Starting sequential processing..."

for (( i=0; i<N_MARKERS; i++ ))
do
    echo "------------------------------------------"
    echo "Processing Marker Index: $i / $((N_MARKERS-1))"
    echo "------------------------------------------"
    CLI_ARGS="--config ${CONFIG_FILE} --marker-index $i"
    if [ -n "${PREDICTOR_OVERRIDE}" ]; then
        CLI_ARGS="${CLI_ARGS} --predictor-of-interest ${PREDICTOR_OVERRIDE}"
    fi
    python ${SCRIPT_DIR}/run_pipeline.py ${CLI_ARGS}
done

echo "------------------------------------------"
echo "All markers processed."

# 4. Get Model Directory (for Post-processing)
MODEL_FOLDER=$(python -c "
import yaml
import sys
sys.path.append('${SCRIPT_DIR}')
from helpers import get_model_folder_name

with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)

formula = config['lmm']['formula']
predictor = '${PREDICTOR_OVERRIDE}' or config['lmm'].get('predictor_of_interest', 'auto')
if isinstance(predictor, list):
    predictor = 'auto'
print(get_model_folder_name(formula, predictor))
")

OUTPUT_PATH=$(python -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
print(config['project']['output_path'])
")
MODEL_DIR="${OUTPUT_PATH}/${MODEL_FOLDER}"

# 5. MCC Post-processing
echo "=========================================="
echo "Running MCC Post-processing..."
echo "Model Dir: ${MODEL_DIR}"
python Statistics/apply_mcc_postprocessing.py "${MODEL_DIR}" --config Statistics/config.yaml

# 6. Report Generation
echo "=========================================="
echo "Generating Summary Report..."
python Statistics/generate_summary_report.py "${MODEL_DIR}"

echo "=========================================="
echo "Local Pipeline Completed Successfully!"
echo "Model Results: ${MODEL_DIR}"
