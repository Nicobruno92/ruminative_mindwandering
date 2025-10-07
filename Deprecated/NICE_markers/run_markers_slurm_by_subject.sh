#!/bin/bash
#SBATCH --job-name=niceMarkersByFile
#SBATCH --output=logs/markers_by_file_%A_%a.out
#SBATCH --error=logs/markers_by_file_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-163  # 41 subjects x 4 tasks = 164 files (0-indexed)

set -e

# Set modes to compute all marker types
MODES="all"
MAX_WORKERS=8  # Reduced from 32 to be more reasonable per file
SCRIPT="NICE_markers/run_markers.py"  # Using the existing script that we modified
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

# Dynamically generate the list of all subject-task pairs
echo "Finding all subject-task pairs..."
# Build an array of all subject-task pairs
declare -a FILE_PAIRS
INDEX=0

for SUBJECT_DIR in $DERIVATIVES_DIR/sub-*; do
    SUBJECT=$(basename "$SUBJECT_DIR")
    
    # Check for EEG files for each task
    for TASK_FILE in "$SUBJECT_DIR"/eeg/*_task-*_desc-autoPreproc_eeg.fif; do
        if [ -f "$TASK_FILE" ]; then
            # Extract task name from filename
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
echo "Found $TOTAL_FILES subject-task pairs"

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_FILES" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID exceeds number of files $TOTAL_FILES"
    exit 1
fi

# Get the current subject-task pair
PAIR=${FILE_PAIRS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=$(echo "$PAIR" | cut -d':' -f1)
TASK=$(echo "$PAIR" | cut -d':' -f2)

echo "Processing file: $SUBJECT task $TASK (array task ID: $SLURM_ARRAY_TASK_ID)"

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

echo "Starting NICE markers run for $SUBJECT task $TASK at $(date)"
echo "MODES: $MODES"
echo "MAX_WORKERS: $MAX_WORKERS"

# Run the modified script for the current file
python -u "$SCRIPT" --modes "$MODES" --max-workers $MAX_WORKERS --subject "$SUBJECT" --task "$TASK"

echo "NICE markers run complete for $SUBJECT task $TASK at $(date)" 