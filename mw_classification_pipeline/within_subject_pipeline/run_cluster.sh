#!/bin/bash
#SBATCH --job-name=mw_ws_clf
#SBATCH --output=logs/ws_%a_%A.out
#SBATCH --error=logs/ws_%a_%A.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

# ==============================================================================
# SLURM Within-Subject Mind-Wandering Classification Runner
# ==============================================================================
# This script submits a job array to run classification for each feature family.
# It uses the ML conda environment.
#
# USAGE:
#   sbatch run_cluster.sh
# ==============================================================================

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ML

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PIPELINE_DIR}" || exit 1

CONFIG_FILE="${PIPELINE_DIR}/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

mkdir -p logs

# Extract all run_families using python snippet
FAMILIES=($(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    cfg = yaml.safe_load(f)
for fam in cfg.get('run_families', []):
    print(fam)
"))

NUM_FAMILIES=${#FAMILIES[@]}

if [ $NUM_FAMILIES -eq 0 ]; then
    echo "ERROR: No families found in config.yaml under 'run_families'"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Array syntax handling
# SLURM_ARRAY_TASK_ID goes from 0 to N-1
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Run this script via sbatch:"
    echo "sbatch --array=0-$((NUM_FAMILIES-1)) run_cluster.sh"
    exit 1
fi

# Make sure array ID is within bounds
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_FAMILIES" ]; then
    echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is out of bounds (max: $((NUM_FAMILIES-1)))."
    exit 0
fi

CURRENT_FAMILY=${FAMILIES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================================="
echo " WITHIN-SUBJECT MIND-WANDERING CLASSIFICATION PIPELINE (CLUSTER)"
echo "============================================================================="
echo " Job ID       : $SLURM_ARRAY_JOB_ID"
echo " Array ID     : $SLURM_ARRAY_TASK_ID"
echo " Node         : $SLURMD_NODENAME"
echo " CPUs         : $SLURM_CPUS_PER_TASK"
echo " Start Time   : $(date)"
echo "-----------------------------------------------------------------------------"
echo " Processing Family: $CURRENT_FAMILY"
echo "============================================================================="

# For cluster, enforcing deterministic output buffers
PYTHONUNBUFFERED=1 python run_within_subject_classification.py \
    --config config.yaml \
    --family "${CURRENT_FAMILY}"

echo "============================================================================="
echo " CLUSTER JOB COMPLETED."
echo " End Time     : $(date)"
echo "============================================================================="
