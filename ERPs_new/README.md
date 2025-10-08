### ERP per-probe Evoked pipeline (ERPs_new)

This pipeline reads preprocessed evoked epochs produced by `Preprocessing_pipeline_new`, selects trials per probe based on configurable rules, averages them to obtain one Evoked per probe, labels probes on-task/off-task from the `onoff` rating embedded in event descriptions, runs statistical analysis with **multiple comparison correction**, and generates two types of ERP visualizations following a BIDS-like structure.

## Key Features

### ✨ Multiple Comparison Correction (NEW!)
- **Cluster permutation testing**: Recommended for temporal ERP data
- **FDR correction**: False Discovery Rate (Benjamini-Hochberg)
- **Bonferroni correction**: Conservative point-wise correction
- **Configurable**: Easy selection via config file
- **Visualization**: Automatic cluster highlighting in plots

👉 **See [CLUSTER_PERMUTATION_GUIDE.md](./CLUSTER_PERMUTATION_GUIDE.md) for detailed documentation**

## Plot Types Available

The pipeline now supports two complementary types of ERP visualization:

### Option A: Classic ERP Plots (Traditional Approach)
- Shows raw ERP averages without baseline subtraction
- Displays the actual neural signal with natural offsets between conditions
- Suitable for classic ERP interpretation and visualization
- Statistical comparisons are informed by LMM analysis with baseline as covariate
- File suffix: `*_erps_classic.png`

### Option B: Beta Time-course Plots (Statistical Effects)
- Shows temporal evolution of statistical effects (beta coefficients from LMM)
- Displays sliding window analysis of condition differences over time
- Highlights significant time periods where on-task vs off-task differ
- Represents pure statistical effects independent of baseline
- **Now includes cluster permutation results with colored cluster regions**
- File suffix: `*_beta_timecourse.png`

Configure in `config.yaml`:
```yaml
plotting:
  plot_type: "both"  # Options: "classic_erp", "beta_timecourse", "both"

multiple_comparison_correction:
  method: "cluster_permutation"  # Options: "none", "fdr", "bonferroni", "cluster_permutation"
  cluster_permutation:
    n_permutations: 1000
    threshold: "auto"
    tail: 0  # 0=two-tailed, 1=right, -1=left
  
lmm_analysis:
  formula: "amplitude ~ condition_code + baseline_centered"  # Default formula
  # Alternative formulas:
  # "amplitude ~ condition_code"                             # No baseline covariate
  # "amplitude ~ condition_code * baseline_centered"         # Include interaction
```

### LMM Formula Configuration

The pipeline supports configurable Linear Mixed Model formulas for maximum flexibility:

**Default Formula**: `amplitude ~ condition_code + baseline_centered`
- Models amplitude as a function of condition (on-task vs off-task) while controlling for baseline
- `condition_code`: 0 = on-task, 1 = off-task  
- `baseline_centered`: mean-centered baseline amplitude
- Random effects: always `(1|subject)`

**Alternative Formulas**:
1. **No baseline control**: `"amplitude ~ condition_code"`
   - Simple condition effect without baseline correction
   - Use when baseline differences are not a concern

2. **Interaction model**: `"amplitude ~ condition_code * baseline_centered"`  
   - Tests if condition effect depends on baseline levels
   - Includes main effects and interaction term

3. **Quadratic baseline**: `"amplitude ~ condition_code + baseline_centered + I(baseline_centered**2)"`
   - Non-linear baseline relationship
   - Useful when baseline effects are not linear

**Note**: The formula must include `condition_code` for condition comparisons to be extracted. Other terms are optional.

### What it does
- Reads input epochs from `BIDS/derivatives/sub-XX/eeg/sub-XX_task-YY_desc-evoked_epo.fif` and its `*_events.tsv`/`*.json`.
- Parses event `description` strings of the form `go/correct/onoff99/.../-5/probe15` to extract:
  - `trial_type` (go/nogo), `correctness` (correct/incorrect)
  - ratings: `onoff`, `selfother`, `valence`, `time`, `confidence`, `average`
  - `distance_to_probe` (negative integer), `probe_number`
- Filters to go/correct trials within a configurable distance window, then groups by `probe_number` and averages trials to obtain one Evoked per probe.
- Labels probes: onTask if `onoff > threshold`, offTask if `onoff < threshold` (threshold configurable).
- Saves outputs to `BIDS/features/sub-XX/eeg`:
  - `sub-XX_task-YY_desc-probe-###_(onTask|offTask)_ave.fif`
  - Sidecar JSON with metadata including the exact `distances_used`
  - `sub-XX_task-YY_probe_evokeds.csv` summary
  - `sub-XX_task-YY_probe_trials.html` report with the full annotation rows per probe

### Directory layout
- `ERPs_new/config.yaml`: central configuration
- `ERPs_new/helpers.py`: reusable I/O, parsing, selection, saving, and report helpers
- `ERPs_new/make_probe_evokeds.py`: CLI entry point

### Dependencies
Use your existing conda env where MNE and pandas are available (example name `eeg`). Required packages:
- `mne`, `numpy`, `pandas`, `pyyaml`

If needed with conda (example):
```bash
conda install -n eeg -c conda-forge mne pandas pyyaml numpy
```

### Configuration (ERPs_new/config.yaml)
- `project.derivatives_root`: absolute path to BIDS derivatives (inputs)
- `project.features_root`: absolute path to BIDS features (outputs)
- `project.bids_root`: optional, used by helpers when wrapping BIDS utilities
- `project.input_evoked_desc`: descriptor of input epochs, default `evoked`
- `subjects`, `tasks`: arrays processed by default; can be overridden by CLI flags
- `trial_selection`:
  - `only_go_correct` (bool): keep only go/correct trials
  - `distance_min`, `distance_max` (int): inclusive window, e.g., `-5..-1`
  - `min_required_distances` (int): require at least N distinct distances in-range for a probe to be kept; set `0` to disable
- `labeling.onoff_threshold` (int): onTask if `onoff > threshold`, offTask if `< threshold`
- `output.desc_prefix` (str): base of the output `desc` (e.g., `probe-015_onTask`)
- `output.save_summary_csv` (bool): write per-subject CSV summary
- `output.overwrite` (bool): allow overwriting existing files
- `evoked.apply_baseline` (bool) and `evoked.baseline`: optional extra baseline before averaging
- `plotting`:
  - `plot_type` (str): "classic_erp", "beta_timecourse", or "both" 
  - `time_step` (float): time step for sliding window LMM (default: 0.01s)
  - `window_width` (float): width of sliding window (default: 0.05s)
  - `smooth_betas` (bool): apply smoothing to beta time-course (default: true)
  - `smoothing_window` (float): smoothing window width (default: 0.02s)
- `lmm_analysis`:
  - `formula` (str): LMM formula specification (default: "amplitude ~ condition_code + baseline_centered")
  - Available variables: `amplitude` (DV), `condition_code` (0=onTask, 1=offTask), `baseline_centered`
  - Groups variable is always `subject`
  - Alternative formulas: 
    - `"amplitude ~ condition_code"` (no baseline covariate)
    - `"amplitude ~ condition_code * baseline_centered"` (include interaction)
    - `"amplitude ~ condition_code + baseline_centered + I(baseline_centered**2)"` (quadratic baseline)

### Running

#### Complete Pipeline (Recommended)
Run the complete pipeline (probe evokeds → LMM → figures):
```bash
sbatch ERPs_new/run_complete_erp_pipeline.sh
```

#### Individual Steps

**Step 1: Generate probe evokeds**
```bash
python ERPs_new/make_probe_evokeds.py --config ERPs_new/config.yaml --subject 04 --task Sart4
```

**Step 2: Run LMM analysis** 
```bash
python ERPs_new/lmm_analysis.py --config ERPs_new/config.yaml
```

**Step 3: Generate ERP figures**
```bash
# Generate both types of plots (default)
python ERPs_new/make_erp_figures.py --config ERPs_new/config.yaml

# Generate only classic ERP plots
python ERPs_new/make_erp_figures.py --config ERPs_new/config.yaml --plot-type classic_erp

# Generate only beta time-course plots  
python ERPs_new/make_erp_figures.py --config ERPs_new/config.yaml --plot-type beta_timecourse
```

### Outputs

#### Probe Evokeds
- Evoked FIF per probe with descriptor: `probe-###_(onTask|offTask)`
- JSON sidecar includes:
  - `probe_number`, `label`, `n_trials`
  - `distances_used`: exact list of trial distances aggregated, e.g., `[-5, -3, -2, -1]`
  - Standard MNE evoked metadata (`Nave`, `SamplingFrequency`, `Tmin`, `Tmax`)
- Summary CSV per subject: number of trials per probe, label, and paths
- HTML report per subject/task listing all selected trials per probe with full annotation columns

#### Statistical Analysis
- `results/ERPs_new/lmm/lmm_results.csv`: Windowed LMM results for predefined time windows
- `results/ERPs_new/lmm/lmm_temporal_results.csv`: Point-by-point LMM results for beta time-course
- `results/ERPs_new/lmm/lmm_*.png`: Heatmaps and visualizations of statistical results

#### ERP Figures  
- **Individual plots**: `results/ERPs_new/participants/sub-XX/figures/`
- **Group classic ERPs**: `results/ERPs_new/group/*_erps_classic.png` and `*_interactive.html`
- **Group beta time-courses**: `results/ERPs_new/group/*_beta_timecourse.png` and `*_interactive.html`
- **QA reports**: `results/ERPs_new/group/qa_report_roi-*.csv`

### Event format expectations
This pipeline currently targets the legacy description format with binary splits, e.g.:
```
go/correct/onoff99/selfother72/valence47/time71/confidence71/average72/-5/probe15
```
It derives `distance_to_probe` from the `/-X/` token and `probe_number` from `/probeY`. If your dataset uses the newer categorical probe format, extend `enrich_events_with_parsed_fields` accordingly.

### Troubleshooting
- No trials selected:
  - Check the printed diagnostics: available distances within the window for go/correct; lower `min_required_distances` or widen the window to `[-5,-1]`.
  - Verify the input epochs exist for that subject/task under `BIDS/derivatives`.
- Missing input files:
  - Run the preprocessing pipeline first for the subject/task, or restrict to available ones.
- Overwrite errors:
  - Set `output.overwrite: true` in `config.yaml`.

### Outlier Detection
The pipeline includes robust outlier detection with two methods:

**Sequential Multi-Criteria Method (Recommended)**:
- **Step 1**: Peak-to-peak rejection (removes epochs with excessive peak-to-peak amplitude)
- **Step 2**: Maximum amplitude rejection (removes epochs exceeding amplitude threshold)  
- **Step 3**: Variance-based Z-score detection (removes epochs with extreme variance)
- More conservative and effective than single-criteria methods
- Configurable thresholds in `outlier_detection` section

**Simple Z-Score Method**:
- Single criterion based on mean amplitude Z-score
- Faster but less comprehensive
- Fallback option when sequential detection is disabled

Configure in `config.yaml`:
```yaml
outlier_detection:
  use_sequential_detection: true  # Enable sequential method
  reject_ptp: 200e-6             # Peak-to-peak threshold (Volts)
  reject_amp: 150e-6             # Amplitude threshold (Volts)
  epoch_z_threshold: 4.0         # Z-score threshold for variance
```

### Extending the pipeline (P100/P300 extraction)
- Add a module (for example, `components.py`) that loads the saved per-probe Evokeds and computes ROI-based P100/P300 latency/amplitude using windows from a dedicated YAML section (e.g., `erp_windows` and channel ROIs).
- Save results to `BIDS/features` as CSV per probe and a subject-level summary.
- Optionally add figures overlaying onTask vs offTask per ROI.

### Code structure and style
- All parameters come from YAML; do not hard-code thresholds in code.
- Keep helpers small and reusable; avoid deep nesting and prefer early returns.
- Add docstrings for non-trivial functions.
- Match existing formatting; avoid reformatting unrelated code.

### Maintainership notes
- Inputs are tightly coupled to the preprocessing pipeline’s output naming. If you change `desc` values upstream, reflect it in `project.input_evoked_desc`.
- The event parsing relies on stable tokens in `description`. If acquisition/annotation changes, update `enrich_events_with_parsed_fields` and add tests on a small sample events.tsv.
- The HTML report is meant for quick human QA; the JSON sidecars are the source of truth for programmatic usage.


