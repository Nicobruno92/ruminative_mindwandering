---
trigger: always_on
---

# EEG Mind-Wandering Analysis - Agent Instructions
description: EEG mind-wandering pipeline: BIDS data I/O, preprocessing, feature extraction, statistics. Use for loading/saving EEG data, running analyses, filtering trials, classifying mental states, batch jobs, or fixing path/config errors.Retry


## Critical Rules
1. **NEVER hardcode paths** → use `utils/bids_compliance.py`
2. **NEVER hardcode parameters** → load from `config.yaml`
3. **Subject IDs**: zero-padded strings `"02"` to `"43"` (no "01")
4. **Tasks**: `["Sart1", "Sart2", "Sart3", "Sart4"]` (case-sensitive)
5. **Pre-probe analysis**: filter distance `-5` to `-1`

## Essential Imports
```python
import yaml
from utils.bids_compliance import read_epochs, save_evokeds, load_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

config = yaml.safe_load(open('config.yaml'))
```

## Data Locations
```
_RAW_DATA/{sub}/                          → Raw BrainVision
BIDS/raw/{sub}/eeg/                       → BIDS raw
BIDS/derivatives/{sub}/eeg/               → Preprocessed (*_desc-autoPreproc_epo.fif)
BIDS/features/{sub}/eeg/                  → Features
  ├── junifer/                            → Spectral/connectivity (*_markers.pkl)
  └── mne_evokeds/                        → ERPs (*_desc-probe-{N}_{label}_ave.fif)
results/{pipeline}/                       → Outputs
```

## Standard Workflow
```python
# Load & filter
epochs = read_epochs("04", "Sart1", config['project']['derivatives_root'])
epochs = filter_epochs_by_distance_to_probe(epochs, distance=5)  # -5 to -1

# Classify on/off-task
classified = classify_onoff_epochs(epochs, split='median')
evoked_on = classified['high'].average()
evoked_off = classified['low'].average()

# Save
save_evokeds([evoked_on, evoked_off], "04", "Sart1", 
             config['project']['features_root'],
             desc="probe-015", labels=['onTask', 'offTask'])
```

## Event Structure

Format: `go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`

- **Dimensions**: 0-100 scale (0=off-task, 100=on-task)
- **Distance**: negative=before probe, positive=after
- **Use**: `-5` to `-1` for mental state prediction

## Commands
```bash
# Single subject
python Preprocessing_pipeline_new/preprocessing_pipeline.py \
  --config config.yaml --subject 04 --task Sart1

python ERPs_new/make_probe_evokeds.py \
  --config config.yaml --subject 04 --task Sart1

# Batch (SLURM)
sbatch Preprocessing_pipeline_new/run_preprocessing_slurm.sh
sbatch ERPs_new/run_complete_erp_pipeline.sh

# Junifer (sequential)
cd junifer_markers/1.markers_h5_creation && sbatch slurm_array_junifer.sh
cd ../2.h5_to_pkl && sbatch batch_convert_h5_to_pkl_parallel.sh
cd ../3.aggregate_probes && sbatch run_aggregate_slurm.sh
```

## SLURM Array Index

`array_idx = subject_idx * 4 + task_idx`

Example: Sub-04 (idx=2), Sart2 (idx=1) → `2*4+1 = 9`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check zero-padding (`"04"`), case (`"Sart1"`), BIDS functions |
| No epochs match | Verify distance filter, check `*_desc-autoPreproc_epo.fif` exists |
| Memory error | Add `gc.collect()`, use `max_files_per_batch` parameter |
| Missing config param | Check correct config file, use `config['project']['key']` path |

## Key Functions
```python
# I/O
read_epochs(subject, task, derivatives_root)
save_evokeds(evokeds, subject, task, features_root, desc, labels)
load_evokeds(subject, task, features_root, desc, labels)

# Analysis
filter_epochs_by_distance_to_probe(epochs, distance)  # Keep -N to -1
classify_onoff_epochs(epochs, split='median')         # {'high': on, 'low': off}
```

## Validation Checklist

- [ ] Config loaded with `yaml.safe_load()`
- [ ] Subject is string: `"04"` not `4`
- [ ] Task case-correct: `"Sart1"` not `"sart1"`
- [ ] BIDS functions used (no hardcoded paths)
- [ ] Pre-probe filter applied for mental state
- [ ] Dependencies met (preprocessing before features)

## Config Locations
```
Preprocessing_pipeline_new/config.yaml    → ICA, artifacts
ERPs_new/config.yaml                      → ROIs, baseline
Statistics/config.yaml                    → LMM, thresholds
junifer_markers/*/config.yaml             → Markers, bands
```

## Quick Tests
```python
# Check preprocessing status
from pathlib import Path
fpath = Path(config['project']['derivatives_root']) / f"sub-04/eeg/sub-04_task-Sart1_desc-autoPreproc_epo.fif"
print(fpath.exists())

# Validate subject/task
assert subject in [f"{i:02d}" for i in range(2, 44)]
assert task in ['Sart1', 'Sart2', 'Sart3', 'Sart4']
```