# Junifer Marker Aggregation by Probe

This directory contains scripts to aggregate per-epoch Junifer markers into per-probe markers for mind-wandering analysis.

## Overview

This pipeline is the third step in the Junifer marker extraction workflow:

```
Pipeline 1: H5 Creation → Pipeline 2: PKL Creation → **Pipeline 3: Probe Aggregation**
```

It reads per-epoch marker PKL files and aggregates them by probe, similar to the ERP probe aggregation pipeline (`ERPs_new/make_probe_evokeds.py`), but adapted for marker data.

## Key Features

### Two Marker Types with Different Aggregation Strategies

1. **Evoked Markers** (time-locked topographies: P1, N1, P2, P3a, P3b)
   - Aggregate only the **5 trials closest to the probe** (-5 to -1)
   - Filter for **go/correct trials only** (trial-locked responses)
   - Apply outlier detection using RMS-based z-score thresholding
   - Same approach as ERP analysis

2. **State Markers** (spectral, connectivity, information theory)
   - Aggregate **all trials before the probe** (captures sustained mental state)
   - **No trial-type filtering** (state events have trial_type='unknown')
   - Apply outlier detection using the same robust method
   - Includes: psd_bands, psd_relative, wsmi_*, PE_*, kolmogorov_complexity

### Robust Outlier Detection

- Uses broader baseline range (-10 to -1) for computing statistics
- Z-score threshold configurable (default: 3.0)
- Same method as ERP pipeline for consistency

## Directory Structure

```
3.aggregate_probes/
├── README.md                           # This file
├── config.yaml                         # Configuration file
├── aggregate_markers_by_probe.py       # Main aggregation script
└── run_aggregate_slurm.sh             # SLURM array job script
```

## Input Files

### Required Inputs

1. **Per-epoch marker PKL files** (from Pipeline 2):
   ```
   features/{subject}/eeg/junifer/{subject}_task-{task}_desc-evoked_markers.pkl
   features/{subject}/eeg/junifer/{subject}_task-{task}_desc-state_markers.pkl
   ```

2. **Preprocessed epochs** (for metadata and outlier detection):
   ```
   derivatives/{subject}/eeg/{subject}_task-{task}_desc-evoked_epo.fif
   derivatives/{subject}/eeg/{subject}_task-{task}_desc-evoked_events.tsv
   ```

### Input Structure

Per-epoch marker PKL files contain:
- `markers`: Dictionary of marker data with per-epoch values
  - Evoked markers: `data[epoch_XXXX][channel]`
  - Spectral markers: `data[channel][band_epoch_XXXX]`
  - Connectivity markers: `data[ch1-ch2][epoch_XXXX]`
- `annotations`: Epoch-level behavioral annotations
- `info`: Metadata (n_epochs, channels, etc.)

## Output Files

### Per-probe Aggregated Markers

```
features/{subject}/eeg/junifer/{subject}_task-{task}_desc-probe-{NNN}_{LABEL}_markers.pkl
```

Where:
- `{NNN}`: Probe number (001, 002, etc.)
- `{LABEL}`: onTask or offTask based on onoff rating

### Output Structure

```python
{
    "markers": {
        "P1": {"Fp1": 0.123, "Fz": 0.456, ...},        # Evoked marker
        "psd_bands": {"Fp1": 0.789, "Fz": 0.234, ...}, # State marker
        "wsmi_theta": {...},
        ...
    },
    "metadata": {
        "subject": "02",
        "task": "Sart1",
        "probe_number": 15,
        "label": "onTask",
        "n_evoked_trials": 5,
        "n_state_trials": 23,
        "outlier_detection": {...}
    },
    "info": {
        "created_at": "2025-10-08 ...",
        "storage_type": "aggregated_probe_markers",
        "n_markers": 14,
        "marker_names": ["P1", "N1", ...]
    }
}
```

### Summary CSV

```
features/junifer_probe_aggregation_summary.csv
```

Contains one row per probe with:
- subject, task, probe_number, label
- n_markers, n_evoked_trials, n_state_trials
- output_path

## Usage

### Quick Start

```bash
# On cluster
cd /network/iss/home/nicolas.bruno/Junifer

# Submit SLURM array job for all subjects/tasks
sbatch junifer_markers/3.aggregate_probes/run_aggregate_slurm.sh

# Monitor progress
squeue -u $USER
watch squeue -u $USER

# Check logs
tail -f logs/aggregate_markers_*.out
```

### Single Subject/Task

```bash
# Activate environment
conda activate junifer

# Run aggregation
python junifer_markers/3.aggregate_probes/aggregate_markers_by_probe.py \
    --config junifer_markers/3.aggregate_probes/config.yaml \
    --subject 31 \
    --task Sart4
```

### Custom Configuration

```bash
# Use custom config file
python aggregate_markers_by_probe.py --config my_custom_config.yaml
```

## Configuration

Edit `config.yaml` to customize:

### Key Parameters

```yaml
trial_selection:
  only_go_correct: true
  evoked_distance_min: -5   # For evoked markers
  evoked_distance_max: -1
  state_distance_min: -999  # For state markers (all trials)
  state_distance_max: -1
  min_required_distances: 3

outlier_detection:
  epoch_z_threshold: 3.0
  baseline_distance_min: -10
  baseline_distance_max: -1
  min_baseline_epochs: 5

marker_types:
  evoked: ["P1", "N1", "P2", "P3a", "P3b"]
  state: ["psd_bands", "psd_relative", "wsmi_theta", ...]
```

## Processing Pipeline

### For Each Subject-Task Combination:

1. **Load Data**
   - Load per-epoch marker PKL files (evoked and state)
   - Load preprocessed epochs for metadata
   - Parse events.tsv to identify probes

2. **For Each Probe**:
   - Label as onTask/offTask based on onoff rating (threshold=50)
   
   - **For Evoked Markers**:
     - Select trials -5 to -1
     - Detect outliers using RMS z-score (baseline: -10 to -1)
     - Aggregate valid epochs (average across epochs)
   
   - **For State Markers**:
     - Select all trials before probe
     - Detect outliers using same method
     - Aggregate valid epochs (average across epochs)
   
   - Save aggregated markers to PKL file

3. **Generate Summary**
   - Create CSV with probe-level statistics

## Dependencies

### Python Packages
- numpy
- pandas
- mne
- pickle (standard library)
- pyyaml

### Project Dependencies
- `ERPs_new/helpers.py`: For event parsing, probe labeling, outlier detection

### Environment
```bash
conda activate junifer
```

## SLURM Resource Requirements

- **Array size**: 168 jobs (42 subjects × 4 tasks)
- **Per job**: 4 CPUs, 16GB RAM, 4 hours
- **Total walltime**: ~4 hours (all parallel)

## Differences from ERP Pipeline

| Aspect | ERP Pipeline | Marker Aggregation Pipeline |
|--------|-------------|----------------------------|
| Input format | `.fif` epochs | `.pkl` marker files |
| Output format | `.fif` evoked | `.pkl` aggregated markers |
| Data structure | Time × Channels | Channels (scalar per marker) |
| Evoked aggregation | Average waveforms | Average marker values |
| State aggregation | N/A | Average across all pre-probe trials |
| Baseline correction | Optional on waveforms | Not applicable (scalars) |

## Troubleshooting

### Common Issues

**1. Missing input PKL files**
```bash
# Check if PKL files were created in Pipeline 2
ls -lh features/sub-*/eeg/junifer/*_markers.pkl

# If missing, run Pipeline 2 first
```

**2. Import errors**
```bash
# Ensure ERPs_new is in Python path
export PYTHONPATH=/path/to/Junifer:$PYTHONPATH

# Test import
python -c "from ERPs_new.helpers import load_yaml_config; print('OK')"
```

**3. Insufficient trials**
```
[WARN] Insufficient trials: 2 < 3
```
- Reduce `min_required_distances` in config
- Check if preprocessing filtered too many epochs

**4. Memory issues**
```bash
# Increase memory allocation
MEM=32G sbatch run_aggregate_slurm.sh
```

### Validation

**Check output files**:
```bash
# Count generated probe files
ls -1 features/sub-*/eeg/junifer/*_desc-probe-*_markers.pkl | wc -l

# Inspect a sample file
python -c "
import pickle
with open('features/sub-31/eeg/junifer/sub-31_task-Sart4_desc-probe-015_onTask_markers.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Markers:', list(data['markers'].keys()))
    print('N evoked trials:', data['metadata']['n_evoked_trials'])
    print('N state trials:', data['metadata']['n_state_trials'])
"
```

**Check summary**:
```bash
# View summary CSV
cat features/junifer_probe_aggregation_summary.csv | head -20
```

## Integration with Analysis Pipeline

### Downstream Analysis

The aggregated probe markers can be used for:

1. **Linear Mixed Models**: Test differences between onTask/offTask
2. **Classification**: Train ML models on probe-level markers
3. **Correlation Analysis**: Relate markers to behavioral measures
4. **Clustering**: Group probes by marker patterns

### Example: Load and Analyze

```python
import pickle
import pandas as pd
import numpy as np

# Load aggregated marker
with open('features/sub-31/eeg/junifer/sub-31_task-Sart4_desc-probe-015_onTask_markers.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract marker values for a specific channel
p3b_fz = data['markers']['P3b']['Fz']
theta_power_fz = data['markers']['psd_bands']['Fz']

# Get metadata
label = data['metadata']['label']
n_trials = data['metadata']['n_evoked_trials']

print(f"Probe label: {label}")
print(f"P3b amplitude at Fz: {p3b_fz:.4f}")
print(f"Theta power at Fz: {theta_power_fz:.4f}")
```

## Performance Optimization

### Parallel Execution
- Use SLURM array for parallel processing across subjects/tasks
- Each job is independent and can run concurrently

### Resource Tuning
- Default: 4 CPUs, 16GB RAM per job
- Adjust if needed based on dataset size:
  ```bash
  CPUS=8 MEM=32G sbatch run_aggregate_slurm.sh
  ```

### Monitoring
```bash
# Check job status
squeue -u $USER

# View resource usage
sacct -j <job_id> --format=JobID,Elapsed,MaxRSS,State

# Monitor in real-time
watch -n 10 squeue -u $USER
```

## Testing

### Test on Single Subject
```bash
# Test with one subject/task
python aggregate_markers_by_probe.py \
    --config config.yaml \
    --subject 02 \
    --task Sart1
```

### Verify Output
```bash
# Check if output file exists
ls -lh features/sub-02/eeg/junifer/*_desc-probe-*_markers.pkl

# Count probes
ls -1 features/sub-02/eeg/junifer/*_desc-probe-*_markers.pkl | wc -l
```

## Future Enhancements

Potential improvements:
1. Add support for ROI-level aggregation (e.g., frontal, central, posterior)
2. Include additional outlier detection methods (e.g., peak-to-peak, amplitude)
3. Generate HTML reports with marker visualizations per probe
4. Add optional temporal smoothing for state markers
5. Support for custom marker weighting schemes

## Related Documentation

- [Pipeline 1: H5 Markers Creation](../1.markers_h5_creation/README.md)
- [Pipeline 2: PKL Creation](../2.h5_to_pkl/README.md)
- [ERP Probe Aggregation](../../ERPs_new/README.md)
- [Main Junifer Markers README](../README.md)
