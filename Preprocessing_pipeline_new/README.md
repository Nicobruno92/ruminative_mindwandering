## Harmonization Pipeline (EEG) – Developer Guide

### Overview
This pipeline harmonizes multi-center EEG recordings into BIDS raw format, with consistent event recoding, montage assignment, optional metadata integration, and resampling. It is driven by a YAML configuration and is modular for reuse in later processing steps.

### Key Components
- `data_harmonization.py`: Main entry point. Orchestrates I/O using the YAML config.
- `bids_compliance.py`:
  - Class `BIDSCompliance` with `write_raw` and `read_raw` for BIDS I/O and `dataset_description.json` management.
- `io_helpers.py`:
  - `DataFinder`: Locates BrainVision `.vhdr` input (supports flexible subject formats and extensions).
  - `read_brainvision_safe`: Reads `.vhdr` after sanitizing `.vmrk/.eeg` references.
  - `MontageHelper`: Applies custom montage and sets EOG channel types if present.
  - `EventRecoder`: Recodes annotations using `utils/trigger_correction.py` and preserves original onsets/durations.
  - `maybe_set_meas_date_from_name`: Optionally populates measurement date from filename.
- `metadata_helpers.py`:
  - `load_metadata`, `write_participants`: Create `participants.tsv/.json` in BIDS root from `metadata_experiment.csv`.
- Montages: `CACS-64_withREF.bvef`, `CACS-64_REF.bvef` live in this folder and are applied as configured.

### Requirements
- Python 3.10+
- Packages: `mne`, `mne-bids`, `pyyaml`, `numpy`, `pandas`
- BrainVision files: `.vhdr` + matching `.vmrk` and `.eeg` files in per-subject directories

### Installation
Use your existing environment or create one:
```
conda create -n eeg python=3.10 -y
conda activate eeg
pip install mne mne-bids pyyaml numpy pandas
```

### Configuration (config.yaml)
Top-level keys and example values:
```
project:
  dataset_name: "EEG Multicenter Harmonized Dataset"
  data_root: "/path/to/_RAW_DATA"              # input location
  raw_root: "/path/to/BIDS/raw"                # BIDS raw output root
  derivatives_root: "/path/to/derivatives"     # derivatives output root
  brainvision_pattern: "CYBERSART_*_{task}.vhdr"
  montage: "CACS-64_withREF.bvef"               # or CACS-64_REF.bvef
  sr_target: 250
  line_freq_hz: 50
  overwrite: true
  metadata_csv: "/path/to/metadata_experiment.csv"  # optional

tasks: ["Sart1", "Sart2", "Sart3", "Sart4"]
subjects: ["02", "03", "04", "05", ...]

# Thresholds for binarizing thought probe dimensions
thought_thresholds:
  onoff: 50
  selfother: 50
  valence: 50
  time: 50
  confidence: 50
  average: 50
```
Notes:
- `subjects` can be a list of strings (e.g. "02") or a single string like "(02 03 04)".
- `brainvision_pattern` is expanded per task; loader also tries `.VHDR` and wildcard forms, and subject variants `sub-02`, `sub-S02`, `sub-S002`, etc.

### Run
- Batch for all subjects/tasks in YAML:
```
python Preprocessing_pipeline_new/data_harmonization.py --config ./Preprocessing_pipeline_new/config.yaml
python Preprocessing_pipeline_new/preprocessing_pipeline.py --config ./Preprocessing_pipeline_new/config.yaml
```
- Single subject/task:
```
python Preprocessing_pipeline_new/data_harmonization.py --config ./Preprocessing_pipeline_new/config.yaml --subject 02 --task Sart1
python Preprocessing_pipeline_new/preprocessing_pipeline.py --config ./Preprocessing_pipeline_new/config.yaml --subject 02 --task Sart1
```

### What it does
- Finds and reads BrainVision `.vhdr` via `read_brainvision_safe` (fixes malformed header references to `.vmrk/.eeg`).
- Sets measurement date from filename if present.
- Applies montage (`CACS-64_*`) and marks `VEOG`/`HEOG` as EOG if present.
- Recodes events via `utils/trigger_correction.TriggerCorrector`, preserving original onsets/durations (`orig_time`). Recoding includes aggregating THOUGHT_PROBE answers, adding ontask/offtask labels for trials, and appending binarized tokens based on `thought_thresholds`.
- Resampling: only `sr_target` from config is honored (single source of truth).
- Writes BIDS raw (BrainVision) into `raw_root` using `BIDSCompliance`.
- Optionally writes `participants.tsv` and `.json` to `raw_root` from `metadata_experiment.csv`.

### Automatic preprocessing (continuous → epochs) - Deterministic Pipeline
- **Deterministic processing**: No fallback methods - pipeline uses exactly one specified method per operation.
- **Scientifically replicable**: All operations are reproducible with fixed random seeds and explicit parameters.
- Immediate notch filtering with zero-phase FIR; PSDs pre/post in report.
- Respects harmonized annotations and excludes `BAD_rest` periods from ICA fitting and epoching.
- Detects bad channels (flat/saturation + pyprep), delays interpolation to post-ICA.
- Sets average reference as projector early; applied after ICA.
- Two-copy strategy: Copy A for ICA (filtered and resampled), Copy B for analysis (ERP filter).
- **Single ICA method**: Uses only the method specified in config (no fallbacks to other methods).
- **Deterministic filtering**: Uses zero-phase FIR filtering consistently throughout pipeline.
- **Explicit interpolation**: Uses "accurate" mode interpolation without fallbacks.
- Automatic IC exclusion via EOG/ECG/muscle detection + ICLabel (with threading controls).
- Saves ICA-corrected raw to derivatives, optional CSD computation.
- Creates evoked epochs and pre-probe state windows; AutoReject applied deterministically.
- **Error transparency**: Pipeline fails explicitly on errors rather than silently falling back.
- Quantitative QA + HTML report in `derivatives/sub-*/reports/`.

### Pass/Fail summary and inclusion criteria
- The pipeline now computes a pass/fail decision per subject-task and records it in:
  - Per-run QA CSV: `derivatives/sub-<ID>/sub-<ID>_task-<TASK>_qa_metrics.csv`
  - Global summary CSV: `derivatives/qa_summary.csv` (safe for concurrent jobs)
- Criteria are configurable in `config.yaml` under `pass_criteria`:
```
pass_criteria:
  fail_on_flags: [
    "too_many_bad_channels",
    "too_many_bad_epochs",
    "too_many_ica_excluded",
    "low_line_noise_suppression",
    "no_ocular_ic_detected_with_eog"
  ]
  require_all_flags_absent: false
```
- A run fails if any of the flags listed in `fail_on_flags` are present. If `require_all_flags_absent` is true, any flag triggers failure.
- Use `qa_summary.csv` to select subjects for subsequent steps by filtering `passed == True`.

#### What each flag token means and thresholds (config-driven)
- max_bad_channels_ratio: number of bad channels exceeds `qa_thresholds.max_bad_channels_ratio` × total EEG channels.
- max_bad_epochs_ratio: fraction of evoked epochs rejected exceeds `qa_thresholds.max_bad_epochs_ratio`.
- max_ica_excluded_ratio: fraction of ICA components excluded exceeds `max_ica_excluded_ratio`.
- min_line_noise_db_improvement: reduction in 50 Hz band power (dB) from pre- to post-notch is below `qa_thresholds.min_line_noise_db_improvement`.
- ocular_ic_required: EOG channels present but no ocular component detected by ICA heuristics.

**Deterministic Pipeline Configuration Requirements:**
- `ica.method`: Single ICA method (e.g., "infomax", "picard", "fastica") - no fallbacks
- `ica.random_state`: Fixed random seed for reproducible ICA decomposition
- `autoreject.random_state`: Fixed random seed for reproducible epoch rejection
- All filtering uses zero-phase FIR for consistent phase behavior

Relevant config keys:
- `qa_thresholds.max_bad_channels_ratio` (e.g., 0.20)
- `qa_thresholds.max_bad_epochs_ratio` (e.g., 0.35)
- `qa_thresholds.max_annot_dropped_epochs_ratio` (e.g., 0.50)
- `max_ica_excluded_ratio` (e.g., 0.3)
- `qa_thresholds.min_line_noise_db_improvement` (e.g., 3.0)
- `ica.method` (required: "infomax", "picard", or "fastica")
- `ica.random_state` (required: integer for reproducibility)

### Harmonized annotations and labels
- **BAD_rest**: Added for each rest period starting at `Stimulus/S 31` and ending at the first of `Stimulus/S 41`, `Stimulus/S 42`, or `Stimulus/S 43` thereafter. Duration spans the full rest interval. This preserves the original annotations and adds new ones.
- **THOUGHT_PROBE**: Inserted at each `Stimulus/S 31` onset and recoded to aggregate all probe answers with probe number:
  - Format: `THOUGHT_PROBE/onoffXX/selfotherYY/valenceZZ/timeTT/confidenceCC/averageAA/probeN`
- **Task-state label (trials only)**: For `go`/`nogo` annotations that include an `onoff` value, `/ontask` is appended when `onoff >= threshold`, `/offtask` when below. Threshold comes from `thought_thresholds.onoff` (default 50). Labels are inserted without breaking distance/probe/trial tokens.
- **Binarized tokens**: For trials and `THOUGHT_PROBE` aggregates, binarized tokens are appended per dimension using thresholds in `thought_thresholds`:
  - `onoffBin{0|1}`, `selfotherBin{0|1}`, `valenceBin{0|1}`, `timeBin{0|1}`, `confidenceBin{0|1}`, `averageBin{0|1}`
  - Trials also receive `gonogoBin1` for `go` and `gonogoBin0` for `nogo`.
  - Bin tokens are inserted before the distance (`+N`/`-N`), `probeN`, and trial index tokens to preserve positional parsing.

Examples:
```
go/correct/onoff91/selfother51/valence72/time68/confidence76/average71/-1/probe15/ontask/onoffBin1/selfotherBin1/valenceBin1/timeBin1/confidenceBin1/averageBin1/gonogoBin1/1
THOUGHT_PROBE/onoff91/selfother51/valence72/time68/confidence76/average71/probe15/onoffBin1/selfotherBin1/valenceBin1/timeBin1/confidenceBin1/averageBin1
```

Notes:
- All new annotations are appended using `raw.set_annotations(...)` and retain `orig_time` consistency.
- Events are built via `mne.events_from_annotations`, so downstream epoching remains compatible.

### Troubleshooting
- **Pipeline failures**: The deterministic pipeline fails explicitly on errors rather than using fallbacks. Check error messages for specific issues:
  - ICA convergence failures: Adjust `max_iter` or try a different `method` in config
  - Filter failures: Ensure data quality is sufficient for zero-phase filtering
  - AutoReject failures: Check epoch count and data quality; may need different CV parameters
- Conflicting BIDSVersion warning: occurs if `dataset_description.json` already exists in `raw_root`. Point `raw_root` to a new empty directory for a clean conversion.
- File not found: ensure `data_root` and `brainvision_pattern` are correct. The loader already tries subject/extension variants; if still failing, check real filenames.
- Fiducial/nasion warning: harmless for many analyses; indicates unknown head transform as expected with custom montages.
- OverflowError on annotations: addressed by preserving original onsets/durations and only replacing descriptions.
- **Reproducibility**: Ensure all random seeds are set in config (`ica.random_state`, `autoreject.random_state`) for identical results across runs.

### Development Guidelines
- Keep modules cohesive:
  - BIDS I/O logic in `bids_compliance.py` (class-based API).
  - Input discovery, event recoding, montage, and safe header reading in `io_helpers.py`.
  - Metadata handling in `metadata_helpers.py`.
- Prefer adding new features behind config flags/keys.
- Follow consistent naming and typing; avoid 1–2 char variable names.
- When modifying event logic, verify with `mne.events_from_annotations` and sanity-check counts.

### Maintenance
- Dependency updates: ensure compatibility with `mne`/`mne-bids` API changes.
- Add unit or smoke tests for:
  - `read_brainvision_safe` (synthetic headers with broken references)
  - `EventRecoder` outputs vs. expected recoded labels
  - `DataFinder` pattern/subject variants
- Document new YAML keys in this README and provide examples.
- For new montages, place `.bvef` files in this folder and update `montage` in config.

### Extending
- Adding tasks/modalities: extend `tasks` in YAML; reuse `BIDSCompliance` for writing other datatypes if needed.
- Derivatives: current step writes raw; subsequent steps can write to `derivatives_root` and reuse the same subject/task iteration pattern.


