#!/bin/bash
## SLURM directives
#SBATCH --job-name=ERPProbes
#SBATCH --output=logs/erps_probe_%A_%a.out
#SBATCH --error=logs/erps_probe_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

# Array size must be subjects * tasks; adjust if subjects/tasks change in config
# 42 subjects (02..43) * 4 tasks = 168 elements -> 0..167
#SBATCH --array=0-167

mkdir -p logs

module load proxy || true

echo "Attempting to activate Python environment..."
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
  module load anaconda3 || module load miniconda3 || module load conda || true
  conda activate eeg || true
fi

# Constrain threading to reduce memory pressure and use non-interactive backends
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MNE_MEMMAP_MIN_SIZE=1M
export MNE_LOGGING_LEVEL=WARNING

# Define subjects and tasks (keep in sync with ERPs_new/config.yaml)
subjects=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43)
tasks=('Sart1' 'Sart2' 'Sart3' 'Sart4')

subject_idx=$((SLURM_ARRAY_TASK_ID / 4))
task_idx=$((SLURM_ARRAY_TASK_ID % 4))

if (( subject_idx < 0 || subject_idx >= ${#subjects[@]} )); then
  echo "Subject index $subject_idx out of range (${#subjects[@]}). Exiting."
  exit 0
fi
if (( task_idx < 0 || task_idx >= ${#tasks[@]} )); then
  echo "Task index $task_idx out of range (${#tasks[@]}). Exiting."
  exit 0
fi

subject=${subjects[$subject_idx]}
task=${tasks[$task_idx]}

echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Generating per-probe ERPs for subject $subject task $task"

# Use srun to avoid CPU binding issues; no binding and one task
srun --cpu-bind=none --ntasks=1 python ERPs_new/make_probe_evokeds.py \
  --config ERPs_new/config.yaml \
  --subject "$subject" \
  --task "$task"
echo "END:   $(date '+%Y-%m-%d %H:%M:%S')"


