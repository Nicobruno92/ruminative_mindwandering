#!/bin/bash
#SBATCH --job-name=WM_aggregate
#SBATCH --output=logs/aggregate_%A_%a.out
#SBATCH --error=logs/aggregate_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# =============================================================================
# Aggregate Junifer markers by probe - Wandering Mind Project
# =============================================================================
# Elements are discovered dynamically from the data directory.
# No hardcoded subjects/sessions/tasks - reads from pipeline_config.yaml
# =============================================================================

set -euo pipefail

DEPENDENCY=""
PARSABLE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --dependency=*)
            DEPENDENCY="${1#*=}"
            shift
            ;;
        --dependency)
            DEPENDENCY="${2-}"
            shift 2
            ;;
        --parsable)
            PARSABLE=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Configuration
# Get script directory (where this file is located)
REAL_SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR="$( cd "$( dirname "$REAL_SCRIPT_PATH" )" && pwd )"

# Fallback for Slurm spooling: if SCRIPT_DIR is in /var/spool/slurmd, use SLURM_SUBMIT_DIR
if [[ "$SCRIPT_DIR" == *"/var/spool/slurmd"* ]] && [ -n "${SLURM_SUBMIT_DIR-}" ]; then
    # If the script is in the submit dir or a known subfolder
    if [ -f "${SLURM_SUBMIT_DIR}/run_aggregate_slurm.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    elif [ -f "${SLURM_SUBMIT_DIR}/junifer_markers/2.aggregate_probes/run_aggregate_slurm.sh" ]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/junifer_markers/2.aggregate_probes"
    fi
fi

# Deduce WORKDIR (project root) by going up 2 levels
# SCRIPT_DIR is .../junifer_markers/2.aggregate_probes
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

PIPELINE_CONFIG="${WORKDIR}/junifer_markers/1.markers_h5_creation/pipeline_config.yaml"
DISCOVER_SCRIPT="${WORKDIR}/junifer_markers/1.markers_h5_creation/discover_elements.py"
AGG_CONFIG="${SCRIPT_DIR}/config.yaml"

# reader_picnic / junifer_eeg / sart_datagrabber_picnic are installed editable
# in the junifer-eeg-2 env, so PYTHONPATH override is only kept as an escape hatch.
PYTHONPATH_EXTRA=${PYTHONPATH_EXTRA:-}
if [ -n "${PYTHONPATH_EXTRA}" ]; then
  export PYTHONPATH="${PYTHONPATH_EXTRA}:${PYTHONPATH:-}"
fi

# Conda setup
CONDA_ENV=${CONDA_ENV:-junifer-eeg-2}

set +u
source ~/.bashrc 2>/dev/null || true
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV" 2>/dev/null || true
set -u

# Resolve python interpreter from the env explicitly so this works whether or
# not `conda activate` succeeded. Falls back to `python` on PATH only if the
# env-specific interpreter is missing — and we error out loudly later if that
# python lacks pandas/etc.
PYTHON_BIN="${CONDA_PREFIX:-$HOME/miniforge3/envs/$CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python)"
fi
echo "[INFO] Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

# -----------------------------------------------------------------------------
# Auto-array launcher
# If this script is submitted without --array, it will resubmit itself as an
# array job with the correct size (one task per unique subject-session-task).
# -----------------------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID-}" ]; then
    if [ ${PARSABLE} -eq 0 ]; then
        echo "[LAUNCHER] Not an array job. Computing array size and resubmitting..."
    fi

    FIRST_DESC=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('${PIPELINE_CONFIG}'))['descriptions'][0])")

    # Detect element format (3 or 4 fields) — sessions optional
    SAMPLE_LINE=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | head -1)
    NUM_FIELDS=$(echo "${SAMPLE_LINE}" | awk -F',' '{print NF}')

    if [ "${NUM_FIELDS}" -eq 4 ]; then
        # subject,session,task,desc -> unique by first 3 fields
        N_COMBOS=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | \
            cut -d',' -f1-3 | sort -u | wc -l | tr -d ' ')
    elif [ "${NUM_FIELDS}" -eq 3 ]; then
        # subject,task,desc -> unique by first 2 fields
        N_COMBOS=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | \
            cut -d',' -f1-2 | sort -u | wc -l | tr -d ' ')
    else
        echo "[LAUNCHER] ERROR: Unexpected element format: ${SAMPLE_LINE}"
        exit 1
    fi

    if [ -z "${N_COMBOS}" ] || [ "${N_COMBOS}" -le 0 ]; then
        echo "[LAUNCHER] ERROR: Could not determine number of subject-task combinations"
        exit 1
    fi

    if [ ${PARSABLE} -eq 0 ]; then
        echo "[LAUNCHER] Found ${N_COMBOS} unique (subject, task) combinations"
        echo "[LAUNCHER] Submitting array: 0-$((N_COMBOS - 1))"
    fi

    DEP_OPT=()
    if [ -n "${DEPENDENCY}" ]; then
        DEP_OPT=(--dependency="${DEPENDENCY}")
    fi

    set +e
    ARRAY_JOB_ID=$(sbatch --parsable "${DEP_OPT[@]}" \
        --array=0-$((N_COMBOS - 1)) \
        --output="${SCRIPT_DIR}/logs/aggregate_%A_%a.out" \
        --error="${SCRIPT_DIR}/logs/aggregate_%A_%a.err" \
        "$0")
    SBATCH_EXIT=$?
    set -e

    if [ ${SBATCH_EXIT} -eq 0 ] && [ -n "${ARRAY_JOB_ID}" ]; then
        if [ ${PARSABLE} -eq 1 ]; then
            echo "${ARRAY_JOB_ID}"
        else
            echo "[LAUNCHER] Array submitted successfully as Job ID ${ARRAY_JOB_ID}"
            
            # Submit finalize job
            FINALIZE_JOB_ID=$(sbatch --dependency=afterok:${ARRAY_JOB_ID} \
                --job-name=WM_agg_final \
                --output="${SCRIPT_DIR}/logs/aggregate_finalize_%j.out" \
                --error="${SCRIPT_DIR}/logs/aggregate_finalize_%j.err" \
                --mem=8G \
                --time=00:30:00 \
                --cpus-per-task=1 \
                --wrap="export PYTHONPATH=\"${PYTHONPATH_EXTRA}:${PYTHONPATH:-}\"; source ~/.bashrc 2>/dev/null; if [ -f \"$HOME/miniforge3/etc/profile.d/conda.sh\" ]; then . \"$HOME/miniforge3/etc/profile.d/conda.sh\"; elif [ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]; then . \"$HOME/miniconda3/etc/profile.d/conda.sh\"; elif [ -f \"$HOME/anaconda3/etc/profile.d/conda.sh\" ]; then . \"$HOME/anaconda3/etc/profile.d/conda.sh\"; fi; conda activate \"$CONDA_ENV\" 2>/dev/null || true; cd \"$WORKDIR\"; \"$PYTHON_BIN\" \"$SCRIPT_DIR/aggregate_markers_by_probe.py\" --config \"$AGG_CONFIG\" --finalize")
            
            echo "[LAUNCHER] Finalize job submitted as Job ID ${FINALIZE_JOB_ID##* }"
            echo "[LAUNCHER] Exiting launcher."
        fi
        exit 0
    fi

    if [ -z "${SLURM_JOB_ID-}" ]; then
        echo "[LAUNCHER] ERROR: sbatch array submission failed (exit=${SBATCH_EXIT})." >&2
        exit 1
    fi

    echo "[LAUNCHER] WARNING: sbatch array submission failed. Falling back to running TASK_ID=0 in this job." >&2
fi

# Get task ID (0-indexed)
TASK_ID=${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}

# For aggregation we need unique subject-(session-)task combinations (not per desc).
# Detect element format dynamically (3 fields = no session, 4 fields = with session).
FIRST_DESC=$("$PYTHON_BIN" -c "import yaml; print(yaml.safe_load(open('${PIPELINE_CONFIG}'))['descriptions'][0])")

SAMPLE_LINE=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | head -1)
NUM_FIELDS=$(echo "${SAMPLE_LINE}" | awk -F',' '{print NF}')

if [ "${NUM_FIELDS}" -eq 4 ]; then
    COMBO=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | \
            cut -d',' -f1-3 | sort -u | sed -n "$((TASK_ID + 1))p")
elif [ "${NUM_FIELDS}" -eq 3 ]; then
    COMBO=$("$PYTHON_BIN" "${DISCOVER_SCRIPT}" --config "${PIPELINE_CONFIG}" --desc "${FIRST_DESC}" --list | \
            cut -d',' -f1-2 | sort -u | sed -n "$((TASK_ID + 1))p")
else
    echo "[ERROR] Unexpected element format: ${SAMPLE_LINE}"
    exit 1
fi

if [ -z "${COMBO}" ]; then
    echo "[INFO] Task ID ${TASK_ID} out of range (likely fewer combos than array size). Exiting."
    exit 0
fi

# Parse combination dynamically based on field count
COMBO_FIELDS=$(echo "${COMBO}" | awk -F',' '{print NF}')

if [ "${COMBO_FIELDS}" -eq 3 ]; then
    IFS=',' read -r subject session task <<< "${COMBO}"
    SUBJECT_NUM="${subject#sub-}"
    SESSION_LETTER="${session#ses-}"
elif [ "${COMBO_FIELDS}" -eq 2 ]; then
    IFS=',' read -r subject task <<< "${COMBO}"
    SUBJECT_NUM="${subject#sub-}"
    SESSION_LETTER=""
else
    echo "[ERROR] Unexpected combo format: ${COMBO}"
    exit 1
fi

echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Subject: $SUBJECT_NUM"
echo "Task: $task"
if [ -n "${SESSION_LETTER}" ]; then
    echo "Session: $SESSION_LETTER"
fi
echo "=================================================="

# Run aggregation script
cd "$WORKDIR"
if [ -n "${SESSION_LETTER}" ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_markers_by_probe.py" \
        --config "$AGG_CONFIG" \
        --subject "$SUBJECT_NUM" \
        --session "$SESSION_LETTER" \
        --task "$task"
else
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_markers_by_probe.py" \
        --config "$AGG_CONFIG" \
        --subject "$SUBJECT_NUM" \
        --task "$task"
fi

echo "Done: ${SUBJECT_NUM}, ${task}"
