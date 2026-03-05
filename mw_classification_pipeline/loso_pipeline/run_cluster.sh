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

module load proxy 2>/dev/null || true  # required for cluster network

mkdir -p logs

# ---------------------------------------------------------------------------
# Read SLURM params + combination count from config
# ---------------------------------------------------------------------------
python3 -c "
import yaml, sys

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

slurm = cfg.get('slurm', {})
model = cfg.get('model_type', 'lr')
contrasts = list(cfg.get('label_contrasts', {}).keys())
families = cfg.get('run_families', ['all'])

combos = [f'{model}:{c}:{fam}' for c in contrasts for fam in families]
n = len(combos)

print(f'Model     : {model}')
print(f'Contrasts : {contrasts}')
print(f'Families  : {families}')
print(f'Combos    : {n}')
print(f'Time      : {slurm.get(\"time\", \"24:00:00\")}')
print(f'Memory    : {slurm.get(\"mem\", \"32G\")}')
print(f'CPUs      : {slurm.get(\"cpus_per_task\", 8)}')
print(f'Partition : {slurm.get(\"partition\", \"normal\")}')
print(f'Conda env : {slurm.get(\"conda_env\", \"ML\")}')

# Write combos to a temp file for the worker
with open('logs/.combinations.txt', 'w') as out:
    out.write('\n'.join(combos))

print(f'Array range: 0-{n-1}')
" || exit 1

# Count combinations
N_COMBOS=$(wc -l < logs/.combinations.txt)
ARRAY_RANGE="0-$((N_COMBOS-1))"

# Read SLURM params from config
read SLURM_TIME SLURM_MEM SLURM_CPUS SLURM_PARTITION SLURM_ENV SLURM_JOBNAME <<< "$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
s = cfg.get('slurm', {})
print(
    s.get('time', '24:00:00'),
    s.get('mem', '32G'),
    s.get('cpus_per_task', 8),
    s.get('partition', 'normal'),
    s.get('conda_env', 'ML'),
    s.get('job_name', 'mw_class'),
)
")"

echo ""
echo "Submitting SLURM array: $ARRAY_RANGE"
sbatch \
    --job-name="$SLURM_JOBNAME" \
    --time="$SLURM_TIME" \
    --mem="$SLURM_MEM" \
    --cpus-per-task="$SLURM_CPUS" \
    --partition="$SLURM_PARTITION" \
    --array="$ARRAY_RANGE" \
    --output="logs/slurm_%A_%a.out" \
    --error="logs/slurm_%A_%a.err" \
    run_cluster_worker.sh "$CONFIG"

echo ""
echo "Submitted $N_COMBOS jobs."
echo "Monitor with: squeue -u \$USER"
echo "Logs in     : logs/"
