#!/bin/bash
# Local execution script for Andrillon Pipeline
# Replaces SLURM array jobs with a local sequential loop.

set -e  # Exit on error

# ---------------------------------------------------------------------------
# Optional argument: --predictor <name> / --config <path>
# ---------------------------------------------------------------------------
CONFIG_FILE="Stats_andrillon/config_andrillon.yaml"
PREDICTOR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictor)
            PREDICTOR_OVERRIDE="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash Stats_andrillon/run_local_andrillon_pipeline.sh [--predictor <name>] [--config <path>]"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="Stats_andrillon"

echo "=========================================="
echo "Andrillon Local Pipeline Execution"
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

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file ${CONFIG_FILE} not found!"
    exit 1
fi

echo "Detecting number of markers configuration..."
PYTHON_OUTPUT=$(python -c "
import yaml, sys, logging
from pathlib import Path
logging.disable(logging.CRITICAL)
sys.path.append('${SCRIPT_DIR}')
from andrillon_pipeline import get_marker_list

try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    print(len(get_marker_list(config)))
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
    python ${SCRIPT_DIR}/run_andrillon_pipeline.py ${CLI_ARGS}
done

echo "------------------------------------------"
echo "All markers processed."

# 4. Get Model Directory (for Post-processing)
MODEL_FOLDER=$(python -c "
import yaml, sys, os
sys.path.append(os.path.join('${SCRIPT_DIR}', '..', 'Statistics'))
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

# 5. Report Generation
echo "=========================================="
echo "Generating Summary Report..."
export MPLBACKEND=Agg
python Statistics/generate_summary_report.py "${MODEL_DIR}"

echo "=========================================="
echo "Local Pipeline Completed Successfully!"
echo "Model Results: ${MODEL_DIR}"
