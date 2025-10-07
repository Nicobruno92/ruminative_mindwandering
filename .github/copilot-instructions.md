# EEG Mind-Wandering Analysis Codebase Guide

## Project Overview
This is a multi-center EEG dataset analyzing mind-wandering states using SART (Sustained Attention to Response Task) with thought probes. The pipeline processes raw BrainVision files through BIDS-compliant preprocessing, extracts neural markers, and analyzes ERPs to classify on-task vs off-task mental states.

## Architecture & Data Flow

### Core Pipeline Structure
```
Raw BrainVision (.vhdr) → BIDS/raw → BIDS/derivatives → BIDS/features → results/
```

1. **Data Harmonization** (`Preprocessing_pipeline_new/`): Converts raw BrainVision files to BIDS format with consistent event recoding
2. **Preprocessing** (`Preprocessing_pipeline_new/preprocessing_pipeline.py`): Deterministic preprocessing with ICA, artifact rejection, epoching
3. **Feature Extraction**: Multiple analysis streams (ERPs, spectral analysis, NICE markers, time-frequency)
4. **Statistical Analysis**: Linear mixed models, classification, visualization

### Key Data Types & Locations
- **Raw data**: `_RAW_DATA/` - BrainVision files per subject/task
- **BIDS derivatives**: `BIDS/derivatives/sub-XX/eeg/` - Preprocessed epochs with suffix `desc-autoPreproc_eeg.fif`
- **Features**: `BIDS/features/sub-XX/eeg/` - Per-probe ERPs with descriptors like `desc-probe-015_onTask_ave.fif`
- **Results**: `results/` - Aggregated analyses, figures, statistical outputs

## Critical Development Patterns

### Environment Management
Always activate the appropriate conda environment before running:
- `eeg` - Main EEG analysis environment (preprocessing, ERPs, NICE markers)
- `plots` - Visualization-specific tasks
- `ML` - Machine learning analyses
- `nlp` - Natural language processing
- `base` - Basic operations

On cluster: `module load proxy` before environment activation.

### BIDS Compliance & File Naming
All I/O uses BIDS-like naming via `utils/bids_compliance.py`:
```python
from utils.bids_compliance import read_epochs, save_evokeds, make_bids_basename
```
Pattern: `sub-XX_task-YY_desc-ZZ_suffix.extension`

### Configuration-Driven Design
All pipelines use YAML configs (never hardcode parameters):
- `Preprocessing_pipeline_new/config.yaml` - Main preprocessing parameters
- `ERPs_new/config.yaml` - ERP analysis parameters
- Configs specify paths, thresholds, subject lists, tasks

### Event Structure & Parsing
Events follow structured format: `go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`

Key parsing handled by `utils/trigger_correction.py`:
- Trial type: `go`/`nogo`
- Correctness: `correct`/`incorrect`
- Thought dimensions: `onoff`, `selfother`, `valence`, `time`, `confidence`, `average` (0-100 scale)
- Distance to probe: negative before probe (`-5`), positive after (`+1`)
- Probe number: `probe15`
- Binary labels automatically appended: `ontask`/`offtask`, dimension binaries

## Common Workflows

### Running Single Subject/Task
```bash
# Preprocessing
python Preprocessing_pipeline_new/preprocessing_pipeline.py --config config.yaml --subject 04 --task Sart1

# ERP analysis
python ERPs_new/make_probe_evokeds.py --config ERPs_new/config.yaml --subject 04 --task Sart1
```

### SLURM Cluster Execution
Use provided SLURM scripts for batch processing:
- `run_preprocessing.sh` - Array job for all subject/task combinations
- `NICE_markers/run_markers_slurm.sh` - Parallel marker computation
- Array indices calculated as: `subject_idx * 4 + task_idx` (4 tasks per subject)

### Data Loading Patterns
```python
# Load preprocessed epochs
epochs, events = read_epochs(derivatives_folder, subject, task, "eeg", desc="autoPreproc")

# Filter by distance to probe
filtered_epochs = filter_epochs_by_distance_to_probe(epochs, distance=5)

# Classify on-task vs off-task
classified_epochs = classify_onoff_epochs(filtered_epochs, split='median')
```

## Project-Specific Conventions

### Subject/Task Conventions
- Subjects: `02` to `43` (zero-padded strings)
- Tasks: `Sart1`, `Sart2`, `Sart3`, `Sart4`
- File paths use `sub-XX` format but variables often use `XX` format

### Distance to Probe Logic
- Negative distances (`-5` to `-1`): trials before thought probe
- Positive distances (`+1` to `+N`): trials after thought probe
- Distance `0`: probe response itself
- Most analyses focus on pre-probe trials (`-5` to `-1`)

### ROI Definitions
Multiple ROI schemes depending on analysis:
- **ROIs**: frontal, central, posterior (defined in configs)
- **Electrode-wise**: Individual channel analysis

### Memory & Performance
- Use `gc.collect()` after large operations
- Prefer parallel processing with `joblib.Parallel` or `ProcessPoolExecutor`
- SLURM jobs typically use 32 CPUs, 32GB RAM, 36-72h time limits
- For large datasets, process in batches (`max_files_per_batch` parameter)

## Integration Points

### Cross-Component Dependencies
- All analysis modules depend on `utils/` for BIDS I/O, trigger correction, preprocessing helpers
- ERPs depend on preprocessed epochs from `Preprocessing_pipeline_new/`
- Statistical analyses depend on feature extraction outputs
- NICE markers can run independently but use same preprocessing outputs

### External Dependencies
- **MNE-Python**: Core EEG analysis library
- **AutoReject**: Automated artifact rejection
- **ICLabel**: ICA component classification
- **NICE**: Neural network-based EEG markers
- **pyprep**: Bad channel detection

## Debugging & Common Issues

### File Not Found Errors
Check BIDS naming consistency and ensure preprocessing completed successfully. Files should follow exact naming patterns.

### Memory Issues
Reduce batch sizes, use `del` statements, call `gc.collect()`, or run single subject/task instead of batches.

### Environment Issues
Always activate correct conda environment and run `module load proxy` on cluster before package installation.

### SLURM Array Jobs
Verify array indices match actual subject/task combinations. Use `echo` statements to debug subject/task assignment in SLURM scripts.

This codebase prioritizes scientific reproducibility through deterministic processing, comprehensive configuration files, and BIDS compliance throughout the analysis pipeline.
