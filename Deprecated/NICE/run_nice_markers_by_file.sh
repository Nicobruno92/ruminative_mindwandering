#!/bin/bash
#SBATCH --job-name=niceMarkersByFile
#SBATCH --output=logs/nice_markers_%A_%a.out
#SBATCH --error=logs/nice_markers_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-163  # 41 subjects x 4 tasks = 164 files (0-indexed)

set -e

SCRIPT="NICE/compute_nice_markers_per_epoch.py"
OUTPUT_DIR="results/nice_markers"
DERIVATIVES_DIR="/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_nico"

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Build an array of all subject-task pairs
declare -a FILE_PAIRS
INDEX=0

for SUBJECT_DIR in $DERIVATIVES_DIR/sub-*; do
    SUBJECT=$(basename "$SUBJECT_DIR" | sed 's/sub-//')
    for TASK_FILE in "$SUBJECT_DIR"/eeg/*_task-*_desc-autoPreproc_eeg.fif; do
        if [ -f "$TASK_FILE" ]; then
            FILENAME=$(basename "$TASK_FILE")
            TASK=$(echo "$FILENAME" | grep -o "task-[^_]*" | sed 's/task-//')
            if [ -n "$TASK" ]; then
                FILE_PAIRS[$INDEX]="$SUBJECT:$TASK"
                INDEX=$((INDEX + 1))
            fi
        fi
    done
done

TOTAL_FILES=${#FILE_PAIRS[@]}
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_FILES" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds number of files $TOTAL_FILES"
    exit 1
fi

PAIR=${FILE_PAIRS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=$(echo "$PAIR" | cut -d':' -f1)
TASK=$(echo "$PAIR" | cut -d':' -f2)

echo "Processing: subject $SUBJECT, task $TASK"

# Activate conda environment
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

echo "Starting NICE markers for $SUBJECT $TASK at $(date)"

python -u "$SCRIPT" --subject "$SUBJECT" --task "$TASK" --output-dir "$OUTPUT_DIR" --reduction-mode all

echo "NICE markers complete for $SUBJECT $TASK at $(date)"