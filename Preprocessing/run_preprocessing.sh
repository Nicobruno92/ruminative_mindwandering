#!/bin/bash
#SBATCH --job-name=EEGPreprocessing
#SBATCH --output=logs/preprocessing_%A_%a.out
#SBATCH --error=logs/preprocessing_%A_%a.err
#SBATCH --cpus-per-task=5
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/network/lustre/iss02/home/nicolas.bruno/
#SBATCH --array=0-167  # Adjust array size (subjects * tasks)
module load anaconda
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

# Run the Python script with the current subject and task
python depressed_mindwandering/Preprocessing/cluster_preprocessing,py $subject $task
