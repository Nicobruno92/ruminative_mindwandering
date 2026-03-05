#!/usr/bin/env bash
# =============================================================================
# run_cluster.sh — MW Classification Pipeline (SLURM launcher)
# =============================================================================
# Reads the config, computes combination count, and submits to SLURM as a
# dynamic array. All parameters (SLURM resources, model, families…) come
# from config.yaml — nothing is hardcoded here.
#
# USAGE  (run from the project root or the pipeline directory):
#   bash run_cluster.sh                     # uses config.yaml
#   bash run_cluster.sh path/to/config.yaml
#
# WHAT IT DOES:
#   1. Reads config → derives N = n_contrasts × n_families  combinations
#   2. Submits: sbatch --array=0-(N-1) run_cluster_worker.sh config.yaml
#   Each array job resolves its own (contrast, family) from its SLURM index.
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

# NOTE: `module load proxy` is intentionally NOT called here — the launcher
# only reads a YAML and calls sbatch (no internet-dependent operations).
# The worker (run_cluster_worker.sh) loads proxy before conda activation.

mkdir -p logs

# ---------------------------------------------------------------------------
# Parse config once: print summary, write combinations, emit SLURM params
# ---------------------------------------------------------------------------
# All values are extracted in a single Python call to avoid spawning the
# interpreter twice and to keep the launcher fast.
eval "$(python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

slurm = cfg.get('slurm', {})
model = cfg.get('model_type', 'lr')
contrasts = list(cfg.get('label_contrasts', {}).keys())
families = cfg.get('run_families', ['all'])

combos = [f'{model}:{c}:{fam}' for c in contrasts for fam in families]
n = len(combos)

# Write combinations file (trailing newline so wc -l returns correct count)
with open('logs/.combinations.txt', 'w') as out:
    out.write('\n'.join(combos) + '\n')

# Print human-readable summary to stderr so it reaches the terminal
import sys
print(f'Model     : {model}', file=sys.stderr)
print(f'Contrasts : {contrasts}', file=sys.stderr)
print(f'Families  : {families}', file=sys.stderr)
print(f'Combos    : {n}', file=sys.stderr)
print(f'Time      : {slurm.get(\"time\", \"24:00:00\")}', file=sys.stderr)
print(f'Memory    : {slurm.get(\"mem\", \"32G\")}', file=sys.stderr)
print(f'CPUs      : {slurm.get(\"cpus_per_task\", 8)}', file=sys.stderr)
print(f'Conda env : {slurm.get(\"conda_env\", \"ML\")}', file=sys.stderr)
print(f'Array range: 0-{n-1}', file=sys.stderr)

# Emit shell variable assignments captured by eval
print(f'SLURM_TIME={slurm.get(\"time\", \"24:00:00\")}')
print(f'SLURM_MEM={slurm.get(\"mem\", \"32G\")}')
print(f'SLURM_CPUS={slurm.get(\"cpus_per_task\", 8)}')
print(f'SLURM_ENV={slurm.get(\"conda_env\", \"ML\")}')
print(f'SLURM_JOBNAME={slurm.get(\"job_name\", \"mw_class\")}')
print(f'N_COMBOS={n}')
")" || exit 1

ARRAY_RANGE="0-$((N_COMBOS-1))"

echo ""
echo "Submitting SLURM array: $ARRAY_RANGE"
sbatch \
    --job-name="$SLURM_JOBNAME" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --cpus-per-task="$SLURM_CPUS" \
    --array="$ARRAY_RANGE" \
    --chdir="$SCRIPT_DIR" \
    --output="logs/slurm_%A_%a.out" \
    --error="logs/slurm_%A_%a.err" \
    "$SCRIPT_DIR/run_cluster_worker.sh" "$SCRIPT_DIR/$CONFIG"

echo ""
echo "Submitted $N_COMBOS jobs."
echo "Monitor with: squeue -u \$USER"
echo "Logs in     : logs/"
