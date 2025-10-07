# MNE Marker Aggregation Scripts

This directory contains scripts for aggregating MNE-based EEG markers computed by the `compute_mne_markers.py` script.

## Overview

The aggregation process takes individual MNE marker files and aggregates them by:
1. **Probe level**: Groups trials by proximity to thought probes
2. **Task level**: Aggregates across different conditions within tasks  
3. **Subject level**: Aggregates across tasks and subjects

## Files

root path cluster = /network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering

- `aggregate_mne_markers.py` - Main aggregation script
- `aggregate_mne_markers_slurm.sh` - SLURM job script for cluster execution
- `README_MNE_aggregation.md` - This documentation

## Input Data Structure

The scripts expect MNE marker files with this structure:
```
marker,channel,value,subject_id,task,epoch,event_id,event_name
d,Fp1,2.543e-13,02,Sart1,0,311003734,go/correct/onoff96/selfother8/valence94/time49/confidence97/average62/-40/probe15
```

## Key Features

### Condition Extraction
The script automatically extracts conditions from `event_name`:
- **Trial type**: go/nogo
- **Correctness**: correct/incorrect
- **Mind-wandering conditions**: onoff, selfother, valence, time, confidence
- **Probe information**: distance to probe, probe number

### Aggregation Levels
1. **Probe Level**: Groups trials within N steps of each probe
2. **Task Level**: Aggregates by conditions within each task
3. **Subject Level**: Aggregates across tasks for each subject

### Memory Optimization
- Chunked file processing to handle large datasets
- Configurable batch sizes
- Automatic garbage collection
- Progress tracking

## Usage

### Local Testing
```bash
# Test with a single condition
python aggregate_mne_markers.py \
    --input_dir results/mne_markers \
    --output_dir results/aggregated_mne_markers \
    --conditions onoff \
    --aggregate_level task \
    --trials_before_probe 5 \
    --max_files_per_batch 2
```

### Cluster Execution
```bash
# Submit SLURM array job (processes 3 aggregation levels in parallel)
sbatch MNE_markers/aggregate_mne_markers_slurm.sh
```

The SLURM script creates 3 parallel jobs:
- Job 0: Probe-level aggregation
- Job 1: Task-level aggregation  
- Job 2: Subject-level aggregation

Each job processes multiple conditions:
- Individual conditions: onoff, valence, selfother, time, confidence
- Combined conditions: onoff + valence
- Go/correct filtering (task level only)

## Parameters

### Command Line Arguments
- `--input_dir`: Directory containing MNE marker CSV files (default: results/mne_markers)
- `--output_dir`: Output directory (default: results/aggregated_mne_markers)
- `--conditions`: Conditions to aggregate by (e.g., onoff valence)
- `--aggregate_level`: Level to aggregate (probe/task/subject)
- `--trials_before_probe`: Number of trials before probe to include (default: 5)
- `--only_go_correct`: Include only go/correct trials
- `--max_files_per_batch`: Files to process per batch (default: 3)
- `--quiet`: Disable progress bars for cleaner logs

### SLURM Configuration
- **Memory**: 128GB (handles large EEG datasets)
- **CPUs**: 16 cores for parallel processing
- **Time**: 48 hours (generous for large datasets)
- **Array**: 3 jobs (one per aggregation level)

## Output Files

Output files follow this naming pattern:
```
aggregated_mne_markers_{condition}_{trials}trials_{level}.csv
```

Examples:
- `aggregated_mne_markers_onoff_5trials_task.csv`
- `aggregated_mne_markers_valence_5trials_probe.csv`
- `aggregated_mne_markers_onoff_5trials_go_correct_task.csv`

## Output Structure

Aggregated files contain:
```
subject_id,task,marker,channel,probe_number,onoff_label,mean,std,count
02,Sart1,delta,Fp1,15,high,1.23e-12,2.45e-13,25
```

Where:
- `mean`: Average marker value
- `std`: Standard deviation
- `count`: Number of trials averaged
- Condition columns (e.g., `onoff_label`): high/low labels

## Memory Considerations

The MNE markers are per-electrode data, which creates much larger datasets than NICE markers:
- **Input files**: 14-97MB per subject/task (vs ~1MB for NICE)
- **Processing**: Uses chunking and reduced batch sizes (2 files vs 5)
- **Memory**: 128GB allocation (vs 64GB for NICE)

## Monitoring

Check job progress:
```bash
# Check job status
squeue -u $USER

# Monitor output logs
tail -f logs/agg_mne_markers_*.out

# Check for errors
tail -f logs/agg_mne_markers_*.err
```

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `--max_files_per_batch` to 1
2. **Missing Files**: Ensure MNE marker computation completed successfully
3. **Condition Extraction**: Check that event_name format matches expected pattern

### Performance Tips
- Use smaller batch sizes for large datasets
- Monitor disk space (requires >10GB free)
- Consider processing subsets of subjects for testing

## Example Workflow

1. **Generate MNE markers**:
   ```bash
   sbatch MNE_markers/run_mne_markers_slurm.sh
   ```

2. **Aggregate markers**:
   ```bash
   sbatch MNE_markers/aggregate_mne_markers_slurm.sh
   ```

3. **Check results**:
   ```bash
   ls results/aggregated_mne_markers/
   ```

The aggregated data can then be used for statistical analysis, machine learning, or visualization of EEG patterns related to mind-wandering states. 