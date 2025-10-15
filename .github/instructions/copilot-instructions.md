# EEG Mind-Wandering Analysis: Complete Development Guide

## Project Overview
Multi-center EEG dataset investigating mind-wandering using SART (Sustained Attention to Response Task) with thought probes. The pipeline implements a complete workflow from raw BrainVision files through BIDS-compliant preprocessing, neural marker extraction, and statistical analysis to classify on-task vs off-task mental states.

**Study Design**: 42 subjects × 4 SART sessions with periodic thought probes assessing mind-wandering dimensions (on-task/off-task, self/other, valence, temporal focus, confidence).

## Architecture & Data Flow

### Complete Pipeline Overview
```
┌───────────────────────────────────────────────────────────────┐
│  Raw BrainVision (.vhdr, .vmrk, .eeg)                         │
│  Location: _RAW_DATA/{subject}/CYBERSART_*_{task}.vhdr       │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  PIPELINE 1: Data Harmonization & BIDS Conversion             │
│  Script: Preprocessing_pipeline_new/preprocessing_pipeline.py │
│  • Converts raw BrainVision → BIDS format                     │
│  • Standardizes event codes across sites                      │
│  • Recodes triggers with behavioral metadata                  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  BIDS/raw/{subject}/eeg/                                      │
│  {subject}_task-{task}_eeg.{vhdr,vmrk,eeg}                   │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  PIPELINE 2: Deterministic Preprocessing                      │
│  Script: Preprocessing_pipeline_new/preprocessing_pipeline.py │
│  • Bad channel detection (PyPREP)                             │
│  • ICA decomposition with ICLabel classification              │
│  • AutoReject for artifact rejection                          │
│  • Epoching around stimuli with probe metadata                │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────────┐
│  BIDS/derivatives/{subject}/eeg/                              │
│  {subject}_task-{task}_desc-autoPreproc_epo.fif              │
│  • Clean epochs with embedded probe metadata                  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ├─────────────────────────────────────────┐
                       │                                         │
                       ▼                                         ▼
┌──────────────────────────────────┐  ┌───────────────────────────────────┐
│  PIPELINE 3A: ERP Analysis       │  │  PIPELINE 3B: Junifer Markers     │
│  Dir: ERPs_new/                  │  │  Dir: junifer_markers/            │
│  • Extract per-probe evokeds     │  │  • Spectral power (bands)         │
│  • Classify on/off-task          │  │  • Connectivity (WSMI)            │
│  • ROI averaging                 │  │  • Info theory (PE, KC)           │
│  • Linear mixed models           │  │  • ERP components (P1,N1,P3)      │
└──────────────────┬───────────────┘  └──────────────┬────────────────────┘
                   │                                  │
                   ▼                                  ▼
┌──────────────────────────────────┐  ┌───────────────────────────────────┐
│  BIDS/features/{subject}/eeg/    │  │  BIDS/features/{subject}/eeg/     │
│  {subject}_task-{task}_          │  │  junifer/{subject}_task-{task}_   │
│  desc-probe-{NNN}_{label}_       │  │  desc-probe-{NNN}_{label}_        │
│  ave.fif                         │  │  markers.pkl                      │
└──────────────────┬───────────────┘  └──────────────┬────────────────────┘
                   │                                  │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  PIPELINE 4: Statistics     │
                   │  Dir: Statistics/           │
                   │  • Linear mixed models      │
                   │  • Cluster permutation      │
                   │  • ML classification        │
                   └──────────────┬──────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────┐
                   │  results/{pipeline_name}/   │
                   │  • Grand averages           │
                   │  • Statistical maps         │
                   │  • Figures & reports        │
                   └─────────────────────────────┘
```

### Key Data Locations & File Types
- **Raw data**: `_RAW_DATA/{subject}/CYBERSART_*_{task}.vhdr` - Original BrainVision recordings
- **BIDS raw**: `BIDS/raw/{subject}/eeg/` - BIDS-formatted raw data with harmonized events
- **BIDS derivatives**: `BIDS/derivatives/{subject}/eeg/` - Preprocessed epochs (`desc-autoPreproc_epo.fif`)
- **ERP features**: `BIDS/features/{subject}/eeg/` - Per-probe evoked responses (`desc-probe-015_onTask_ave.fif`)
- **Junifer features**: `BIDS/features/{subject}/eeg/junifer/` - Neural markers in PKL format
- **Results**: `results/{pipeline}/` - Analysis outputs, figures, statistical reports

### 2. BIDS Compliance & File Naming
**ALL I/O operations MUST use BIDS-compliant paths via `utils/bids_compliance.py`**:

```python
from utils.bids_compliance import (
    read_epochs,           # Load preprocessed epochs
    save_evokeds,          # Save evoked responses
    load_evokeds,          # Load evoked responses
    make_bids_basename,    # Generate BIDS-compliant filenames
    save_psd_epochs,       # Save power spectral density
    read_psd_epochs,       # Load power spectral density
    save_tfr_epochs,       # Save time-frequency representations
    read_tfr_epochs        # Load time-frequency representations
)
```

**BIDS Naming Pattern**: `sub-{XX}_task-{TASK}_desc-{DESCRIPTOR}_suffix.extension`

**Examples**:
- Preprocessed epochs: `sub-04_task-Sart1_desc-autoPreproc_epo.fif`
- Per-probe evoked: `sub-04_task-Sart1_desc-probe-015_onTask_ave.fif`
- Junifer markers: `sub-04_task-Sart1_desc-probe-015_onTask_markers.pkl`

**Critical Rules**:
1. Never hardcode paths - always use BIDS helper functions
2. Use zero-padded subject IDs (`"02"` not `2` or `"2"`)
3. Task names are case-sensitive: `Sart1`, `Sart2`, `Sart3`, `Sart4`
4. Descriptors should be lowercase with hyphens: `desc-auto-preproc`, not `desc-AutoPreproc`

### 3. Configuration-Driven Design
**ALL pipelines use YAML configurations - NEVER hardcode parameters**:

```python
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access parameters
subjects = config['subjects']
tasks = config['tasks']
bids_root = config['project']['bids_root']
```

**Main Configuration Files**:
- `Preprocessing_pipeline_new/config.yaml` - Preprocessing parameters (filtering, ICA, artifact rejection)
- `ERPs_new/config.yaml` - ERP analysis parameters (trial selection, ROI definitions, baseline)
- `Statistics/config.yaml` - Statistical analysis parameters (LMM formulas, clustering thresholds)
- `junifer_markers/*/config.yaml` - Feature extraction settings (markers, bands, connectivity)

**Configuration Best Practices**:
1. Always check if parameter exists in config before hardcoding
2. Document all new parameters with comments
3. Use consistent naming: `snake_case` for keys
4. Include default values and valid ranges in comments
5. Maintain reproducibility: document random seeds, thresholds, file paths

### 4. Event Structure & Parsing
**Events follow structured format**: `go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15`

**Event Components**:
- **Trial type**: `go` or `nogo`
- **Correctness**: `correct` or `incorrect`
- **Thought dimensions** (0-100 scale):
  - `onoff` - On-task (100) to off-task (0)
  - `selfother` - Self-oriented (100) to other-oriented (0)
  - `valence` - Positive (100) to negative (0)
  - `time` - Future (100) to past (0)
  - `confidence` - Confident (100) to uncertain (0)
  - `average` - Mean of all dimensions
- **Distance to probe**: Negative before probe (`-5` to `-1`), positive after probe (`+1` to `+N`)
- **Probe number**: `probe15` (unique identifier for each thought probe)

**Automatic Binary Labels**:
- `onTask` / `offTask` - Based on threshold (typically median or 50)
- Dimension-specific binaries appended automatically

**Key Parsing Utilities** (in `utils/trigger_correction.py`):
```python
class TriggerCorrector:
    def recode_annotations(self):
        # Recodes raw triggers into structured format
        # Handles go/nogo classification, correctness, distance calculations
        
    def annotate_thought_probe_markers(self):
        # Inserts THOUGHT_PROBE markers at S31 triggers
        
    def annotate_rest_periods(self):
        # Marks rest periods as BAD_rest for exclusion
```

**Usage Pattern**:
```python
from utils.analysis_helpers import classify_onoff_epochs, filter_epochs_by_distance_to_probe

# Filter trials by distance to probe
epochs_preprobe = filter_epochs_by_distance_to_probe(epochs, distance=5)  # -5 to -1

# Classify on-task vs off-task
classified = classify_onoff_epochs(epochs_preprobe, split='median')
on_task_epochs = classified['high']
off_task_epochs = classified['low']
```

## Common Workflows

### Running Single Subject/Task
```bash
# 1. Preprocessing (from scratch to clean epochs)
python Preprocessing_pipeline_new/preprocessing_pipeline.py \
    --config Preprocessing_pipeline_new/config.yaml \
    --subject 04 \
    --task Sart1

# 2. ERP analysis (extract per-probe evoked responses)
python ERPs_new/make_probe_evokeds.py \
    --config ERPs_new/config.yaml \
    --subject 04 \
    --task Sart1

# 3. Generate ERP figures
python ERPs_new/make_erp_figures.py \
    --config ERPs_new/config.yaml \
    --subjects 04

# 4. Run LMM analysis
python ERPs_new/lmm_analysis.py \
    --config ERPs_new/config.yaml
```

### SLURM Cluster Execution
Use provided SLURM scripts for batch processing:

**Preprocessing**:
```bash
# Run all subjects/tasks as array job
sbatch Preprocessing_pipeline_new/run_preprocessing_slurm.sh
```

**ERP Pipeline**:
```bash
# Complete ERP workflow (evokeds + figures + LMM)
sbatch ERPs_new/run_complete_erp_pipeline.sh

# Individual steps
sbatch ERPs_new/run_evokeds.sh          # Extract per-probe evokeds
sbatch ERPs_new/run_erp_figures.sh      # Generate figures
sbatch ERPs_new/run_lmm_slurm.sh        # LMM analysis
```

**Junifer Markers**:
```bash
# Pipeline 1: Create H5 markers
cd junifer_markers/1.markers_h5_creation
sbatch slurm_array_junifer.sh

# Pipeline 2: Convert H5 to PKL
cd junifer_markers/2.h5_to_pkl
sbatch batch_convert_h5_to_pkl_parallel.sh

# Pipeline 3: Aggregate by probe
cd junifer_markers/3.aggregate_probes
sbatch run_aggregate_slurm.sh
```

**Array Job Indices**:
- Calculated as: `subject_idx * 4 + task_idx` (4 tasks per subject)
- Subject 04, Task 1 (Sart1) = `(2 * 4) + 0 = 8`
- Subject 04, Task 2 (Sart2) = `(2 * 4) + 1 = 9`

### Data Loading Patterns
```python
from utils.bids_compliance import read_epochs, load_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs
```

## Project-Specific Conventions

### Subject/Task Conventions
- **Subjects**: `"02"` to `"43"` (zero-padded strings, no subject "01")
- **Tasks**: `"Sart1"`, `"Sart2"`, `"Sart3"`, `"Sart4"` (case-sensitive)
- **BIDS paths**: Use `sub-XX` format (e.g., `sub-04`)
- **Python variables**: Often use just `"XX"` format without `sub-` prefix

### Distance to Probe Logic
- **Negative distances** (`-5` to `-1`): Trials before thought probe
- **Positive distances** (`+1` to `+N`): Trials after thought probe (recovery/response period)
- **Distance `0`**: Probe response itself (usually excluded from ERP analysis)
- **Most analyses**: Focus on pre-probe trials (`-5` to `-1`) for predicting mental state

### Thought Probe Dimensions & Scales
All dimensions use **0-100 continuous scale**:
- **onoff**: 100 = fully on-task, 0 = fully off-task
- **selfother**: 100 = self-oriented, 0 = other-oriented
- **valence**: 100 = positive, 0 = negative
- **time**: 100 = future-oriented, 0 = past-oriented
- **confidence**: 100 = confident in rating, 0 = uncertain
- **average**: Mean across all dimensions

**Binary Classification**:
# Classify on-task vs off-task
- Threshold typically at 50 or median split
- `> 50` or `> median` → "high" or "on-task"
- `< 50` or `< median` → "low" or "off-task"
- Value `== 50` typically excluded
classified = classify_onoff_epochs(filtered_epochs, split='median')
on_task = classified['high']
off_task = classified['low']

# Filter by distance to probe (keep only -5 to -1)
filtered_epochs = filter_epochs_by_distance_to_probe(epochs, distance=5)

### Memory & Performance
- **Memory management**: Use `gc.collect()` after large operations
- **Parallel processing**: Prefer `joblib.Parallel` or `ProcessPoolExecutor` for subject/task loops
- **SLURM resources**: 
  - Typical allocation: 32 CPUs, 32GB RAM
  - Time limits: 36-72 hours for full pipeline
  - Array jobs: More efficient than single large jobs
- **Batch processing**: Use `max_files_per_batch` parameter for large datasets to avoid memory overflow

## Integration Points

### Cross-Component Dependencies
- **Core utilities** (`utils/`): All analysis modules depend on these for:
  - BIDS I/O operations (`bids_compliance.py`)
  - Event recoding and trigger correction (`trigger_correction.py`)
  - Preprocessing helpers (`preprocessing_helpers.py`)
  - Analysis helpers (`analysis_helpers.py`)
  - Custom exceptions (`exceptions.py`)
  - Logging utilities (`log_preprocessing.py`)

- **Preprocessing → Feature Extraction**:
  - ERPs depend on `desc-autoPreproc` epochs from preprocessing pipeline
  - Junifer markers use same preprocessed epochs
  - All downstream analyses require completed preprocessing

- **Feature Extraction → Statistics**:
  - Statistical analyses consume outputs from ERP and Junifer pipelines
  - Per-probe aggregated markers enable probe-level comparisons
  - Both time-domain (ERPs) and frequency-domain (spectral) features

- **Independent Components**:
  - NICE markers can run independently (deprecated but functional)
  - Behavioral analyses in `Behavior/` use probe responses directly
  - Each statistical analysis can run independently once features extracted


### Path Resolution & BIDS Structure
The codebase uses consistent path resolution through configuration files:

```python
# Typical path hierarchy from config
project:
  data_root: "/path/to/_RAW_DATA"           # Input: raw BrainVision files
  bids_root: "/path/to/BIDS"                # BIDS root directory
  raw_root: "/path/to/BIDS/raw"             # BIDS raw data (converted)
  derivatives_root: "/path/to/BIDS/derivatives"  # Preprocessed data
  features_root: "/path/to/BIDS/features"   # Extracted features
  results_root: "results/{pipeline_name}"   # Analysis outputs
```

**Directory Structure**:
```
BIDS/
├── raw/
│   └── sub-XX/
│       └── eeg/
│           └── sub-XX_task-YY_eeg.{vhdr,vmrk,eeg}
├── derivatives/
│   └── sub-XX/
│       └── eeg/
│           └── sub-XX_task-YY_desc-autoPreproc_epo.fif
└── features/
    └── sub-XX/
        └── eeg/
            └── junifer/
                └── sub-XX_task-YY_desc-probe-NNN_LABEL_markers.pkl
            └── mne_evokeds/
                └── sub-XX_task-YY_desc-probe-NNN_LABEL_markers.pkl
```
