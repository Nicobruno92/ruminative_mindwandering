#!/usr/bin/env bash
# =============================================================================
# run_cluster.sh — MW Classification Pipeline (SLURM launcher)
# =============================================================================
# Submits SEPARATE SLURM arrays for true runs and permutations so that every
# run/perm executes in parallel on its own node.
#
# Architecture:
#   • Array A: (model × contrast × family × run_idx)  — one job per true run
#   • Array B: (model × contrast × family × perm_idx) — one job per permutation
#   • After A+B complete: run merge_loso_results.py to aggregate results
#
# USAGE:
#   bash run_cluster.sh                     # uses config.yaml
#   bash run_cluster.sh path/to/config.yaml
#
# MONITOR:
#   squeue -u $USER
#   cat logs/slurm_<JOBID>_<ARRAYID>.out
# =============================================================================

set -euo pipefail

CONFIG="${1:-config.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

mkdir -p logs

# ---------------------------------------------------------------------------
# Parse config: extract all parameters in one Python call
# ---------------------------------------------------------------------------
eval "$(python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

slurm      = cfg.get('slurm', {})
model_raw  = cfg.get('model_type', 'lr')
models     = model_raw if isinstance(model_raw, list) else [model_raw]
contrasts  = cfg.get('run_contrasts', list(cfg.get('label_contrasts', {}).keys()))
families   = cfg.get('run_families', ['all'])
n_runs     = cfg.get('n_runs', 10)
n_perms    = cfg.get('permutation_runs', 10)

# True-run combinations: model:contrast:family:run_N
true_combos = [
    f'{m}:{c}:{fam}:run_{r}'
    for m in models for c in contrasts for fam in families
    for r in range(n_runs)
]
# Perm combinations: model:contrast:family:perm_N
perm_combos = [
    f'{m}:{c}:{fam}:perm_{p}'
    for m in models for c in contrasts for fam in families
    for p in range(n_perms)
]

import sys
print(f'Models    : {models}',    file=sys.stderr)
print(f'Contrasts : {contrasts}', file=sys.stderr)
print(f'Families  : {families}',  file=sys.stderr)
print(f'n_runs    : {n_runs}',    file=sys.stderr)
print(f'n_perms   : {n_perms}',   file=sys.stderr)
print(f'True jobs : {len(true_combos)}',  file=sys.stderr)
print(f'Perm jobs : {len(perm_combos)}',  file=sys.stderr)
print(f'Time      : {slurm.get(\"time\", \"24:00:00\")}', file=sys.stderr)
print(f'Memory    : {slurm.get(\"mem\", \"32G\")}',        file=sys.stderr)
print(f'CPUs      : {slurm.get(\"cpus_per_task\", 8)}',   file=sys.stderr)

with open('logs/.true_combinations.txt', 'w') as f:
    f.write('\n'.join(true_combos) + '\n')
with open('logs/.perm_combinations.txt', 'w') as f:
    f.write('\n'.join(perm_combos) + '\n')

print(f'SLURM_TIME={slurm.get(\"time\", \"24:00:00\")}')
print(f'SLURM_MEM={slurm.get(\"mem\", \"32G\")}')
print(f'SLURM_CPUS={slurm.get(\"cpus_per_task\", 8)}')
print(f'SLURM_ENV={slurm.get(\"conda_env\", \"ML\")}')
print(f'SLURM_JOBNAME={slurm.get(\"job_name\", \"mw_class\")}')
print(f'N_TRUE={len(true_combos)}')
print(f'N_PERM={len(perm_combos)}')
")" || exit 1

echo ""

# ---------------------------------------------------------------------------
# Submit true-run array (one job per run)
# ---------------------------------------------------------------------------
TRUE_JOB_ID=""
if [ "$N_TRUE" -gt 0 ]; then
    TRUE_ARRAY="0-$((N_TRUE-1))"
    echo "Submitting true-run array: ${SLURM_JOBNAME}_true [${TRUE_ARRAY}] (${N_TRUE} jobs)"
    TRUE_JOB_ID=$(sbatch \
        --job-name="${SLURM_JOBNAME}_true" \
        --time="$SLURM_TIME" \
        --mem="$SLURM_MEM" \
        --cpus-per-task="$SLURM_CPUS" \
        --array="$TRUE_ARRAY" \
        --chdir="$SCRIPT_DIR" \
        --output="logs/slurm_%A_%a_true.out" \
        --error="logs/slurm_%A_%a_true.err" \
        "$SCRIPT_DIR/run_cluster_worker.sh" "$SCRIPT_DIR/$CONFIG" true \
        | awk '{print $NF}')
    echo "  → Job ID: $TRUE_JOB_ID"
fi

# ---------------------------------------------------------------------------
# Submit permutation array (one job per perm)
# ---------------------------------------------------------------------------
PERM_JOB_ID=""
if [ "$N_PERM" -gt 0 ]; then
    PERM_ARRAY="0-$((N_PERM-1))"
    echo "Submitting perm array:      ${SLURM_JOBNAME}_perm  [${PERM_ARRAY}] (${N_PERM} jobs)"
    PERM_JOB_ID=$(sbatch \
        --job-name="${SLURM_JOBNAME}_perm" \
        --time="$SLURM_TIME" \
        --mem="$SLURM_MEM" \
        --cpus-per-task="$SLURM_CPUS" \
        --array="$PERM_ARRAY" \
        --chdir="$SCRIPT_DIR" \
        --output="logs/slurm_%A_%a_perm.out" \
        --error="logs/slurm_%A_%a_perm.err" \
        "$SCRIPT_DIR/run_cluster_worker.sh" "$SCRIPT_DIR/$CONFIG" perm \
        | awk '{print $NF}')
    echo "  → Job ID: $PERM_JOB_ID"
fi

echo ""
echo "Submitted $N_TRUE true-run + $N_PERM perm jobs."
echo ""
echo "After all jobs complete, run:"
echo "  bash run_merge.sh  # or: python merge_loso_results.py --config $CONFIG"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs in     : logs/"
