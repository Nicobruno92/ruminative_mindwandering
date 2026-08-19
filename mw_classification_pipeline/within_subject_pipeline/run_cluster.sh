#!/usr/bin/env bash
# =============================================================================
# run_cluster.sh — Within-Subject MW Classification Pipeline (SLURM launcher)
# =============================================================================
# Submits SEPARATE arrays for true runs and permutations so every run/perm
# executes in parallel on its own node.
#
# Architecture:
#   • Array A: (model × contrast × family × run_idx)  — one job per true run
#   • Array B: (model × contrast × family × perm_idx) — one job per permutation
#
# After all jobs complete, aggregate with:
#   python ../scripts/recompute_headline_numbers.py
# (There is no merge_ws_results.py; it was referenced here but never existed.
# recompute_headline_numbers.py reads true_runs/run_*/ and permuted_runs/run_*/
# directly, which is the source of truth — the per-model consolidated CSVs are
# deliberately not written by shard jobs.)
#
# USAGE:
#   bash run_cluster.sh                     # uses config.yaml
#   bash run_cluster.sh path/to/config.yaml
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

eval "$(python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

slurm     = cfg.get('slurm', {})
models    = cfg.get('classifiers', {}).get('run_models', ['rf'])
contrasts = cfg.get('run_contrasts', ['on_vs_off_within_median'])
families  = cfg.get('run_families', ['all'])
n_runs    = cfg.get('n_runs', 10)
n_perms   = cfg.get('n_permutations', 10)

true_combos = [
    f'{m}:{c}:{fam}:run_{r}'
    for m in models for c in contrasts for fam in families
    for r in range(n_runs)
]
perm_combos = [
    f'{m}:{c}:{fam}:perm_{p}'
    for m in models for c in contrasts for fam in families
    for p in range(n_perms)
]

print(f'Models    : {models}',    file=sys.stderr)
print(f'Contrasts : {contrasts}', file=sys.stderr)
print(f'Families  : {families}',  file=sys.stderr)
print(f'n_runs    : {n_runs}',    file=sys.stderr)
print(f'n_perms   : {n_perms}',   file=sys.stderr)
print(f'True jobs : {len(true_combos)}', file=sys.stderr)
print(f'Perm jobs : {len(perm_combos)}', file=sys.stderr)
print(f'Max array : {slurm.get(\"max_array_size\", 2000)}', file=sys.stderr)

with open('logs/.true_combinations.txt', 'w') as f:
    f.write('\n'.join(true_combos) + '\n')
with open('logs/.perm_combinations.txt', 'w') as f:
    f.write('\n'.join(perm_combos) + '\n')

print(f'SLURM_TIME={slurm.get(\"time\", \"24:00:00\")}')
print(f'SLURM_MEM={slurm.get(\"mem\", \"32G\")}')
print(f'SLURM_CPUS={slurm.get(\"cpus_per_task\", 32)}')
print(f'SLURM_JOBNAME={slurm.get(\"job_name\", \"mw_ws_clf\")}')
print(f'N_TRUE={len(true_combos)}')
print(f'N_PERM={len(perm_combos)}')
print(f'MAX_ARRAY_SIZE={slurm.get(\"max_array_size\", 2000)}')
")" || exit 1

echo ""

# Submit a SLURM array, splitting into multiple sbatch calls if the
# combination count exceeds the cluster's MaxArraySize limit
# (check with: scontrol show config | grep MaxArraySize).
#
# SLURM requires every array task ID to be < MaxArraySize (the limit applies
# to the index value itself, not just the array's element count), so each
# chunk is submitted as "0-(chunk_size-1)" and the combination-index offset
# is passed to run_cluster_worker.sh as $3.
submit_chunked_array() {
    local label="$1"       # sbatch --job-name
    local mode="$2"        # "true" or "perm" (passed through to worker)
    local n_total="$3"     # number of combinations
    local out_suffix="$4"  # "true" or "perm" (log filename suffix)

    local offset=0 chunk_size job_id
    while [ "$offset" -lt "$n_total" ]; do
        chunk_size=$((n_total - offset))
        [ "$chunk_size" -gt "$MAX_ARRAY_SIZE" ] && chunk_size=$MAX_ARRAY_SIZE
        echo "Submitting ${label} array: offset=${offset} [0-$((chunk_size - 1))] (${chunk_size} jobs)"
        job_id=$(sbatch \
            --job-name="${label}" \
            --time="$SLURM_TIME" \
            --mem="$SLURM_MEM" \
            --cpus-per-task="$SLURM_CPUS" \
            --ntasks=1 \
            --array="0-$((chunk_size - 1))" \
            --chdir="$SCRIPT_DIR" \
            --output="logs/slurm_%A_%a_${out_suffix}.out" \
            --error="logs/slurm_%A_%a_${out_suffix}.err" \
            "$SCRIPT_DIR/run_cluster_worker.sh" "$SCRIPT_DIR/$CONFIG" "$mode" "$offset" \
            | awk '{print $NF}')
        echo "  → Job ID: $job_id"
        offset=$((offset + chunk_size))
    done
}

if [ "$N_TRUE" -gt 0 ]; then
    submit_chunked_array "${SLURM_JOBNAME}_true" true "$N_TRUE" true
fi

if [ "$N_PERM" -gt 0 ]; then
    submit_chunked_array "${SLURM_JOBNAME}_perm" perm "$N_PERM" perm
fi

echo ""
echo "Submitted $N_TRUE true-run + $N_PERM perm jobs."
echo "Monitor: squeue -u \$USER | Logs: logs/"
