#!/bin/bash
#SBATCH --job-name=aggMarkers
#SBATCH --output=logs/agg_markers_%A_%a.out
#SBATCH --error=logs/agg_markers_%A_%a.err
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/
#SBATCH --array=0-2

INPUT_DIR="results/nice_markers"
OUTPUT_DIR="results/aggregated_markers"
SCRIPT="NICE_markers/aggregate_markers.py"

set -e

mkdir -p logs
mkdir -p "$OUTPUT_DIR"

FILE_TYPES=("whole_brain" "per_roi" "per_electrode") # "whole_brain" "per_roi" "per_electrode"
FILE_TYPE=${FILE_TYPES[$SLURM_ARRAY_TASK_ID]}

if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: Script $SCRIPT not found!"
    exit 1
fi

num_files=$(find "$INPUT_DIR" -name "*_${FILE_TYPE}.csv" | wc -l)
echo "Found $num_files ${FILE_TYPE} files to process"
if [ "$num_files" -eq 0 ]; then
    echo "ERROR: No $FILE_TYPE files found to process."
    exit 1
fi

echo "Checking disk space..."
available_space=$(df -k . | awk 'NR==2 {print $4}')
if [ "$available_space" -lt 5242880 ]; then
    echo "ERROR: Not enough disk space available. Only $available_space KB free."
    exit 1
fi

# Activate conda environment (use the most reliable method for your cluster)
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

echo "Starting task-level aggregation..."
python -u "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --file_type "$FILE_TYPE" \
    --conditions onoff valence selfother time confidence \
    --aggregate_level task \
    --trials_before_probe 5 \
    --max_files_per_batch 5 \
    --quiet

echo "Starting subject-level aggregation..."
python -u "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --file_type "$FILE_TYPE" \
    --conditions onoff valence selfother time confidence \
    --aggregate_level subject \
    --trials_before_probe 5 \
    --max_files_per_batch 5 \
    --quiet

echo "Starting probe-level aggregation..."
python -u "$SCRIPT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --file_type "$FILE_TYPE" \
    --conditions onoff valence selfother time confidence \
    --aggregate_level probe \
    --trials_before_probe 5 \
    --max_files_per_batch 5 \
    --quiet

echo "Aggregation complete for $FILE_TYPE at $(date)"


# If you get OOM errors for per_electrode or per_roi, try lowering --max_files_per_batch (e.g., to 2 or 1)
# Example:
# python -u "$SCRIPT" \
#     --input_dir "$INPUT_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --file_type "$FILE_TYPE" \
#     --conditions onoff valence selfother time confidence \
#     --aggregate_level task \
#     --trials_before_probe 5 \
#     --max_files_per_batch 2 \
#     --quiet 