#!/bin/bash
#SBATCH --job-name=niceMarkersBySubject
#SBATCH --output=logs/markers_by_subject_%A_%a.out
#SBATCH --error=logs/markers_by_subject_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-42   # Replace XX with the number of subjects minus 1

set -e

MODES="all"
MAX_WORKERS=32
SCRIPT="NICE_markers/run_markers.py"
SUBJECT_LIST="NICE_markers/subject_list.txt"
OUTPUT_DIR="results/nice_markers"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script $SCRIPT not found!"
    exit 1
fi
if [ ! -f "$SUBJECT_LIST" ]; then
    echo "ERROR: Subject list $SUBJECT_LIST not found!"
    exit 1
fi

SUBJECT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SUBJECT_LIST")
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