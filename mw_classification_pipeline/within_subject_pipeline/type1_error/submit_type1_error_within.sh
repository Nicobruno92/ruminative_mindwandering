#!/usr/bin/env bash
# =============================================================================
# SLURM Array Submission — Type I Error Evaluation (Within-Subject)
#
# Submits one SLURM array job per (contrast × family × model) combination.
# Each array element runs one simulation (--sim_idx = SLURM_ARRAY_TASK_ID).
# After all jobs complete, run with --aggregate to collect results.
#
# USAGE:
#   # Submit all simulations:
#   bash submit_type1_error_within.sh
#
#   # Aggregate results after jobs finish:
#   bash submit_type1_error_within.sh --aggregate
#
# CONFIGURATION:
#   All pipeline params come from ../config.yaml (parent) and
#   config_type1_error.yaml (simulation-specific overrides).
#   Edit n_simulations in config_type1_error.yaml to change array size.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_CONFIG="${SCRIPT_DIR}/../config.yaml"
SIM_CONFIG="${SCRIPT_DIR}/config_type1_error.yaml"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_type1_error_within.py"

# Read parameters via python
CONDA_ENV=$(python3 -c "import yaml; c=yaml.safe_load(open('${PARENT_CONFIG}')); print(c.get('slurm',{}).get('conda_env','ML'))")
N_SIMULATIONS=$(python3 -c "import yaml; c=yaml.safe_load(open('${SIM_CONFIG}')); print(c['simulation']['n_simulations'])")
ARRAY_END=$((N_SIMULATIONS - 1))

# Read contrasts from within-subject parent config (uses run_contrasts key)
CONTRASTS=$(python3 -c "
import yaml
c = yaml.safe_load(open('${PARENT_CONFIG}'))
v = c.get('run_contrasts', ['on_vs_off_within_median'])
print(' '.join(v if isinstance(v, list) else [v]))
")

FAMILIES=$(python3 -c "
import yaml
c = yaml.safe_load(open('${PARENT_CONFIG}'))
v = c.get('run_families', ['all'])
print(' '.join(v if isinstance(v, list) else [v]))
")

MODELS=$(python3 -c "
import yaml
c = yaml.safe_load(open('${PARENT_CONFIG}'))
v = c.get('classifiers', {}).get('run_models', ['rf'])
print(' '.join(v if isinstance(v, list) else [v]))
")

# SLURM resource settings (within-subject is lighter than LOSO)
SLURM_TIME=$(python3 -c "import yaml; c=yaml.safe_load(open('${PARENT_CONFIG}')); print(c.get('slurm',{}).get('time','06:00:00'))")
SLURM_MEM=$(python3 -c "import yaml; c=yaml.safe_load(open('${PARENT_CONFIG}')); print(c.get('slurm',{}).get('mem','32G'))")
SLURM_CPUS=$(python3 -c "import yaml; c=yaml.safe_load(open('${PARENT_CONFIG}')); print(c.get('slurm',{}).get('cpus_per_task','16'))")

MINIFORGE="${HOME}/miniforge3"
PYTHON_BIN="${MINIFORGE}/envs/${CONDA_ENV}/bin/python"

# ── Aggregate mode ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "--aggregate" ]]; then
    echo "Aggregating Type I error results (Within-Subject)..."
    for CONTRAST in $CONTRASTS; do
        for FAMILY in $FAMILIES; do
            for MODEL in $MODELS; do
                echo "  ${CONTRAST} / ${FAMILY} / ${MODEL}"
                "${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
                    --config "${SIM_CONFIG}" \
                    --contrast "${CONTRAST}" \
                    --family "${FAMILY}" \
                    --model_type "${MODEL}" \
                    --aggregate
            done
        done
    done
    echo "Done."
    exit 0
fi

# ── Submit SLURM array jobs ────────────────────────────────────────────────
echo "Submitting Type I error simulations (Within-Subject)"
echo "  N simulations : ${N_SIMULATIONS} (array 0–${ARRAY_END})"
echo "  Contrasts     : ${CONTRASTS}"
echo "  Families      : ${FAMILIES}"
echo "  Models        : ${MODELS}"
echo "  SLURM config  : time=${SLURM_TIME} mem=${SLURM_MEM} cpus=${SLURM_CPUS}"
echo ""

for CONTRAST in $CONTRASTS; do
    for FAMILY in $FAMILIES; do
        for MODEL in $MODELS; do
            JOB_NAME="t1e_ws_${CONTRAST:0:8}_${FAMILY:0:6}_${MODEL}"
            LOG_DIR="${SCRIPT_DIR}/logs"
            mkdir -p "${LOG_DIR}"

            sbatch \
                --job-name="${JOB_NAME}" \
                --array="0-${ARRAY_END}" \
                --time="${SLURM_TIME}" \
                --mem="${SLURM_MEM}" \
                --cpus-per-task="${SLURM_CPUS}" \
                --output="${LOG_DIR}/${JOB_NAME}_%A_%a.out" \
                --error="${LOG_DIR}/${JOB_NAME}_%A_%a.err" \
                --wrap="
source '${MINIFORGE}/etc/profile.d/conda.sh'
conda activate ${CONDA_ENV}
cd '${SCRIPT_DIR}'
'${PYTHON_BIN}' '${PYTHON_SCRIPT}' \
    --config '${SIM_CONFIG}' \
    --contrast '${CONTRAST}' \
    --family '${FAMILY}' \
    --model_type '${MODEL}' \
    --sim_idx \${SLURM_ARRAY_TASK_ID}
"
            echo "  Submitted: ${JOB_NAME} [array 0-${ARRAY_END}]"
        done
    done
done

echo ""
echo "After all jobs complete, run:"
echo "  bash submit_type1_error_within.sh --aggregate"
