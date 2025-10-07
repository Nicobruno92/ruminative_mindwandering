#!/bin/bash
#SBATCH --job-name=lmm_parallel
#SBATCH --output=logs/lmm_parallel_%A_%a.out
#SBATCH --error=logs/lmm_parallel_%A_%a.err
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

# =============================================================================
# Parallel LMM Analysis - Each array job runs one DV on one node
# 
# This distributes 16 analyses (8 DVs × 2 datasets) across cluster nodes
# Each job runs independently, dramatically reducing total wall time
# =============================================================================

mkdir -p logs

module load proxy || true

echo "Activating Python environment..."
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate ML
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate ML
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate ML
else
  module load anaconda3 || module load miniconda3 || module load conda || true
  conda activate ML || true
fi

export MPLBACKEND=Agg

# Define all DV-dataset combinations
DVS=("onoff" "valence" "time" "selfother" "confidence" "PC1" "PC2" "PC3")
DATASETS=("full" "ie")

# Calculate which DV and dataset this array task should process
TOTAL_DVS=${#DVS[@]}
DV_INDEX=$((SLURM_ARRAY_TASK_ID / 2))
DATASET_INDEX=$((SLURM_ARRAY_TASK_ID % 2))

DV=${DVS[$DV_INDEX]}
DATASET=${DATASETS[$DATASET_INDEX]}

echo "=================================="
echo "Parallel LMM Analysis"
echo "=================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 32G"
echo "Start time: $(date)"
echo "=================================="
echo "Analyzing: DV=$DV, Dataset=$DATASET"
echo "=================================="

# Set environment variables for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run single DV analysis
srun --cpu-bind=none --ntasks=1 python Behavior/Probe_analysis/lmm_single_dv.py \
  --dv $DV \
  --dataset $DATASET

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
