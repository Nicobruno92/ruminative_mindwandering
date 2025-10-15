#!/bin/bash
#SBATCH --job-name=aggregate_markers
#SBATCH --output=logs/aggregate_markers_%A_%a.out
#SBATCH --error=logs/aggregate_markers_%A_%a.err
#SBATCH --array=0-167  # 42 subjects * 4 tasks = 168 combinations
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Aggregate Junifer markers by probe
# This is a SLURM array job that processes each subject-task combination in parallel

# Configuration
WORKDIR=${WORKDIR:-/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/}
CONDA_ENV=${CONDA_ENV:-junifer}
SCRIPT_DIR="$WORKDIR/junifer_markers/3.aggregate_probes"
CONFIG="$SCRIPT_DIR/config.yaml"

# Load modules
module load proxy

# Activate conda environment
source ~/.bashrc
conda activate $CONDA_ENV

# Subject and task arrays
SUBJECTS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
TASKS=(Sart1 Sart2 Sart3 Sart4)

# Calculate subject and task from array index
N_TASKS=${#TASKS[@]}
SUBJECT_IDX=$((SLURM_ARRAY_TASK_ID / N_TASKS))
TASK_IDX=$((SLURM_ARRAY_TASK_ID % N_TASKS))

SUBJECT=${SUBJECTS[$SUBJECT_IDX]}
TASK=${TASKS[$TASK_IDX]}

echo "=================================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $SUBJECT (index $SUBJECT_IDX)"
echo "Task: $TASK (index $TASK_IDX)"
echo "=================================================="

# Run aggregation script
cd "$WORKDIR"
python "$SCRIPT_DIR/aggregate_markers_by_probe.py" \
    --config "$CONFIG" \
    --subject "$SUBJECT" \
    --task "$TASK"

echo "Done!"
