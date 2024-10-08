#!/bin/bash
#SBATCH --job-name=EEGPreprocessing
#SBATCH --output=logs/preprocessing_%A_%a.out
#SBATCH --error=logs/preprocessing_%A_%a.err
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/network/lustre/iss02/home/nicolas.bruno/
module load conda
activate my_eeg

# Define subjects and tasks
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

# Calculate subject-task pair from SLURM_ARRAY_TASK_ID
subject_idx=$((SLURM_ARRAY_TASK_ID / 4))
task_idx=$((SLURM_ARRAY_TASK_ID % 4))

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "Processing subject $subject for task $task"

# Load necessary modules (if applicable)
module load python/3.8  # Adjust based on your environment

# Run the Python script with the current subject and task
python preprocess_subject_task.py $subject $task
