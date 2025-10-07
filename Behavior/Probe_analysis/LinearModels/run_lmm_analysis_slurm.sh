#!/bin/bash
#SBATCH --job-name=lmm_exhaustive
#SBATCH --output=logs/lmm_exhaustive_%j.out
#SBATCH --error=logs/lmm_exhaustive_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --chdir=/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/

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

echo "=================================="
echo "LMM Exhaustive Search Analysis"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "=================================="

echo "Python version: $(python --version 2>&1)"
python - <<'PY'
import pandas, numpy, statsmodels
print(f"pandas: {pandas.__version__}")
print(f"numpy: {numpy.__version__}")
print(f"statsmodels: {statsmodels.__version__}")
PY

# Set environment variables for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running LMM analysis with exhaustive search..."
echo "Using $SLURM_CPUS_PER_TASK cores for parallel model evaluation"
echo ""

# Run analysis with srun from project root
srun --cpu-bind=none --ntasks=1 python Behavior/Probe_analysis/lmm_beta_analysis_enhanced.py

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
