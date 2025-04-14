#!/bin/bash
#SBATCH --job-name=niceMarkersBySubject
#SBATCH --output=logs/markers_by_subject_%A_%a.out
#SBATCH --error=logs/markers_by_subject_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-41   # Set to number of subjects minus 1

set -e

MODES="all"
MAX_WORKERS=32
SCRIPT="NICE_markers/run_markers.py"
OUTPUT_DIR="results/nice_markers"
DERIVATIVES_DIR="/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_nico"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script $SCRIPT not found!"
    exit 1
fi
if [ ! -d "$DERIVATIVES_DIR" ]; then
    echo "ERROR: Derivatives directory $DERIVATIVES_DIR not found!"
    exit 1
fi

# Dynamically generate the subject list from the derivatives_nico directory
SUBJECTS=($(ls -d $DERIVATIVES_DIR/sub-* | xargs -n 1 basename | sort))
NUM_SUBJECTS=${#SUBJECTS[@]}

if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_SUBJECTS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds number of subjects $NUM_SUBJECTS"
    exit 1
fi

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SUBJECT" ]; then
    echo "ERROR: No subject found for SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing subject: $SUBJECT"

# Activate conda environment (robust method)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate eeg
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate eeg
else
    echo "ERROR: Could not activate conda environment."
    exit 1
fi

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-$USER

echo "Starting NICE markers run for $SUBJECT at $(date)"
echo "MODES: $MODES"
echo "MAX_WORKERS: $MAX_WORKERS"

python -u "$SCRIPT" --modes "$MODES" --max-workers $MAX_WORKERS --subject "$SUBJECT"

echo "NICE markers run complete for $SUBJECT at $(date)" 