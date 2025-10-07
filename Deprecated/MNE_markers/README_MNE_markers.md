# MNE-based Spectral and ERP Markers

This directory contains scripts to compute spectral and ERP markers using MNE's built-in functions instead of the NICE library. The output format matches the requested CSV structure.

## Features

### Spectral Markers
- **Band Powers**: delta (d), theta (t), alpha (a), beta (b), gamma (g)
- **Normalized Band Powers**: delta_n (d_n), theta_n (t_n), alpha_n (a_n), beta_n (b_n), gamma_n (g_n)
- **Spectral Summary**: spectral entropy (se), median spectral frequency (msf), spectral edge frequencies (sef90, sef95)

### ERP Components
- **Classic Components**: CNV, P1, N1, P2, P3a, P3b (based on standard time windows)
- **Time-domain Features**: mean amplitude, peak-to-peak, RMS, standard deviation, variance, kurtosis, skewness

## Files
root path cluster = /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

- `compute_mne_markers.py`: Main computation script
- `run_mne_markers.py`: Simple runner script
- `run_mne_markers_slurm.sh`: SLURM job script

## Usage

### Command Line (Single Subject/Task)
```bash
# From project root directory:
python MNE_markers/run_mne_markers.py --subject 02 --task Sart1

# Or from MNE_markers directory:
cd MNE_markers
python run_mne_markers.py --subject 02 --task Sart1
```

### Command Line (All Subjects/Tasks)
```bash
# From project root directory:
python MNE_markers/compute_mne_markers.py

# Or from MNE_markers directory:
cd MNE_markers  
python compute_mne_markers.py
```

### SLURM Array Job (All Subjects/Tasks in Parallel)
```bash
# Submit array job - each subject runs on a different node/core
sbatch MNE_markers/run_mne_markers_slurm.sh

# Check job status
squeue -u $USER

# Check specific array job status
squeue -j <JOB_ID>
```

### Python API
```python
from MNE_markers.compute_mne_markers import process_one_subject

process_one_subject(
    root='/network/iss/cenir/analyse/meeg/CYBERSART/',
    subject='02',
    task='Sart1',
    data_type='eeg',
    desc='autoPreproc',
    output_dir='results/mne_markers',
    tmin=0,
    tmax=2
)
```

## Output Format

The script generates CSV files with the following structure:

| marker | channel | value | subject_id | task | epoch | event_id | event_name |
|--------|---------|-------|------------|------|-------|----------|------------|
| a_n | Fp1 | 0.123 | 02 | Sart1 | 0 | 311003734 | go/correct/... |
| a_n | Fp1 | 0.145 | 02 | Sart1 | 1 | 1300503969 | go/correct/... |
| ... | ... | ... | ... | ... | ... | ... | ... |

Each row represents one marker value for one electrode in one epoch.

## Parameters

- `--subject`: Subject ID (e.g., "02")
- `--task`: Task name (e.g., "Sart1") 
- `--root`: Root data directory (default: "/network/iss/cenir/analyse/meeg/CYBERSART/")
- `--output-dir`: Output directory (default: "results/mne_markers")
- `--tmin`: Start time in seconds (default: 0)
- `--tmax`: End time in seconds (default: 2)

## SLURM Array Job Details

The SLURM script uses array jobs for parallel processing:

- **Array range**: `--array=2-42` (subjects 02 to 42)
- **Parallel execution**: Each subject runs on a separate node/core
- **Total jobs**: 41 parallel jobs (one per subject)
- **Tasks per job**: Each job processes 4 tasks (Sart1, Sart2, Sart3, Sart4) sequentially
- **Log files**: Separate log files for each subject (`mne_markers_<JOB_ID>_<SUBJECT_NUM>.out`)

### Advantages:
- ✅ **Much faster**: All subjects run in parallel instead of sequential
- ✅ **Resource efficient**: Each job uses only 4 CPUs and 16GB RAM
- ✅ **Fault tolerant**: If one subject fails, others continue
- ✅ **Easy monitoring**: Individual logs per subject

## Notes

- The script automatically removes EOG channels if present
- Uses MNE's Welch method for power spectral density estimation
- ERP components use the same time windows as the NICE pipeline
- Compatible with both newer and older MNE versions
- Output files are named: `sub-{subject}_task-{task}_mne_markers.csv` 