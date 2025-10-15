# LMM-Based Spatial Cluster Permutation Testing Pipeline

This pipeline implements linear mixed model (LMM) based spatial cluster permutation testing for EEG probe marker data from the Junifer pipeline. It analyzes continuous `onoff` values (mind-wandering levels) across EEG channels using spatial cluster permutation testing.

## Overview

The pipeline performs the following steps:
1. **Data Loading**: Load aggregated probe marker data from Junifer pipeline output
2. **Channel-wise LMM**: Fit linear mixed models for each EEG channel independently using continuous `onoff` predictor
3. **Cluster Formation**: Identify spatially contiguous clusters of channels exceeding threshold
4. **Permutation Testing**: Generate null distribution by permuting `onoff` values within subjects
5. **Visualization**: Create topographic maps and statistical plots organized by marker type

## Pipeline Structure

```
Statistics/
├── config.yaml              # Configuration file (YAML)
├── reader.py                # Data loading module
├── lmm_model.py             # Linear mixed model implementation
├── cluster_test.py          # Spatial cluster permutation testing
├── plot_results.py          # Visualization functions
├── run_pipeline.py          # Main pipeline executor
├── run_lmm_pipeline.sh      # SLURM cluster script
├── example_usage.py         # Usage examples
└── test_pipeline.py         # Test script
```

## Configuration

All parameters are specified in `config.yaml`:

### Key Parameters

- **LMM Formula**: R-style formula using continuous `onoff` predictor (e.g., `"power ~ onoff + (1|subject)"`)
- **Predictor of Interest**: `onoff` - continuous mind-wandering level (0-100)
- **Threshold**: T-statistic threshold for cluster formation
- **Permutations**: Number of permutations for null distribution
- **Alpha**: Significance level (default: 0.05)
- **Montage**: Channel positions file (`CACS-64_REF.bvef`)

### Example Configuration

```yaml
project:
  features_root: "/path/to/features"
  output_path: "/path/to/results"
  montage_path: "Preprocessing_pipeline_new/CACS-64_REF.bvef"

lmm:
  formula: "power ~ onoff + (1|subject)"
  predictor_of_interest: "onoff"
  method: "REML"
  
clustering:
  threshold: 2.0
  n_permutations: 1000
  alpha: 0.05
```

## Usage

### Local Usage

```bash
# Activate eeg environment
conda activate eeg

# Run pipeline with default config
python Statistics/run_pipeline.py

# Run all markers
python Statistics/run_pipeline.py --all-markers

# Run specific markers
python Statistics/run_pipeline.py --markers "EEG_psd_bands_spectralpower_alpha" "EEG_psd_bands_spectralpower_theta"

# Run with custom config
python Statistics/run_pipeline.py --config path/to/config.yaml
```

### SLURM Cluster Usage

```bash
# Submit single marker analysis
sbatch Statistics/run_lmm_pipeline.sh

# Submit all markers analysis
sbatch Statistics/run_lmm_pipeline.sh --all-markers

# Submit specific markers analysis
sbatch Statistics/run_lmm_pipeline.sh --markers "EEG_psd_bands_spectralpower_alpha"

# Check job status
squeue -u $USER

# View output logs
tail -f logs/lmm_pipeline_*.log
```

### Data Format

The pipeline loads aggregated probe marker data from the Junifer pipeline output:

**Input (CSV files):**
- Location: `features_root/sub-XX/eeg/junifer/`
- Pattern: `sub-XX_task-YY_desc-probe-NNN_[evoked|state]_aggMarkers.csv`
- Format: Long format with one row per channel-marker combination per probe

**Required columns:**
- `subject`: subject identifier (required for random effects)
- `task`: task identifier
- `probe_number`: probe number
- `marker_type`: "evoked" or "state" (processed independently)
- `marker`: marker name (e.g., "EEG_psd_bands_spectralpower_alpha")
- `channel`: EEG channel name
- `value`: marker value
- `onoff`: continuous mind-wandering level (0-100) - **main predictor**

**Example data structure:**
```csv
subject,task,probe_number,marker_type,marker,channel,value,onoff,valence,confidence
02,Sart1,1,state,EEG_psd_bands_spectralpower_alpha,Fp1,0.123,75.5,60.2,85.1
02,Sart1,1,state,EEG_psd_bands_spectralpower_alpha,Fp2,0.145,75.5,60.2,85.1
```

## Methodology

### Linear Mixed Models

For each channel independently:
- Fit LMM using statsmodels with `method='REML'` for optimal mixed-effects estimation
- Extract t-statistic for continuous `onoff` predictor (mind-wandering level)
- Handle convergence failures silently (t=0, p=1)
- Uses actual montage file (`CACS-64_REF.bvef`) for channel positions

### Spatial Clustering

- Compute channel adjacency using actual montage positions from `CACS-64_REF.bvef`
- Identify spatially contiguous clusters exceeding threshold
- Cluster statistic = sum of t-values within cluster
- Uses MNE's standard adjacency computation

### Permutation Testing

- Permute continuous `onoff` values **within subjects** to preserve within-subject structure
- Recompute LMM and identify clusters for each permutation
- Generate null distribution of maximum cluster statistics
- P-value = proportion of permutations where max_cluster_stat ≥ observed_cluster_stat
- Deterministic with fixed random seeds for reproducibility

## Output

The pipeline generates:

### 1. Results Files (organized by marker type)
For each marker, files are saved with type-specific naming:
- `{marker_type}_{marker_name}_results.pkl`: Complete results dictionary
- `{marker_type}_{marker_name}_cluster_summary.csv`: Cluster details and significance
- `{marker_type}_{marker_name}_t_statistics.csv`: Per-channel t-statistics and p-values
- `{marker_type}_{marker_name}_topomap.png`: Topographic map with significant clusters

### 2. Summary Files
- `pipeline_summary.csv`: Overview of all processed markers
- `summary_{marker_type}_markers.csv`: Results grouped by marker type (evoked/state)
- `analysis_summary_{marker_type}.csv`: Statistical summary for each marker type

### 3. Example Output Structure
```
results/
├── pipeline_summary.csv
├── summary_evoked_markers.csv
├── summary_state_markers.csv
├── analysis_summary_evoked.csv
├── analysis_summary_state.csv
├── evoked_EEG_psd_bands_spectralpower_alpha_results.pkl
├── evoked_EEG_psd_bands_spectralpower_alpha_cluster_summary.csv
├── evoked_EEG_psd_bands_spectralpower_alpha_t_statistics.csv
├── state_EEG_psd_bands_spectralpower_alpha_results.pkl
└── state_EEG_psd_bands_spectralpower_alpha_cluster_summary.csv
```

## Key Features

- **Deterministic**: Fixed random seeds ensure reproducibility across runs
- **Continuous Analysis**: Uses continuous `onoff` predictor (0-100 mind-wandering scale)
- **Type-Specific Processing**: State and evoked markers processed independently
- **Organized Output**: Results saved with marker type prefixes for easy analysis
- **Montage-Aware**: Uses actual `CACS-64_REF.bvef` montage for spatial operations
- **Cluster-Ready**: Optimized for SLURM cluster execution
- **Robust**: Handles convergence failures and edge cases gracefully
- **Modular**: Each component can be imported and used independently

## Requirements

```
numpy
pandas
mne
statsmodels
matplotlib
scipy
pyyaml
```

All dependencies are included in the `eeg` conda environment.

## Implementation Notes

### Convergence Handling
LMM convergence failures are handled gracefully:
- Failed models: t=0, p=1
- No fallback methods (deterministic approach)
- Warnings suppressed for cleaner output

### Within-Subject Permutation
Predictor values are permuted within each subject to preserve:
- Within-subject correlation structure
- Subject-specific random effects
- Overall data distribution

### Spatial Adjacency
Channel adjacency is computed using MNE's standard approach:
- Based on Euclidean distance in sensor space
- Respects montage geometry
- Compatible with all standard montages

## Example Workflow

```python
from reader import load_all_probe_data, prepare_data_for_lmm
from lmm_model import run_lmm_per_channel
from cluster_test import get_channel_adjacency, spatial_cluster_permutation_test
from plot_results import create_results_report

# Load aggregated probe data
df_all = load_all_probe_data(
    features_root="/path/to/features",
    subjects=["02", "03", "04"],
    tasks=["Sart1", "Sart2"],
    marker_types=["evoked", "state"]
)

# Prepare data for specific marker
power_data, df_behavioral, channels = prepare_data_for_lmm(
    df=df_all,
    marker_name="EEG_psd_bands_spectralpower_alpha",
    formula="power ~ onoff + (1|subject)"
)

# Get adjacency using actual montage
adjacency, ch_names = get_channel_adjacency('CACS-64_REF.bvef', channels)

# Run LMM for each channel
t_stats, p_values = run_lmm_per_channel(
    power_data, df_behavioral, 
    formula="power ~ onoff + (1|subject)",
    predictor_of_interest="onoff"
)

# Cluster permutation test
clusters, cluster_stats, cluster_p = spatial_cluster_permutation_test(
    t_stats, power_data, df_behavioral,
    formula="power ~ onoff + (1|subject)",
    predictor_of_interest="onoff",
    adjacency=adjacency,
    threshold=2.0,
    n_permutations=1000
)

# Generate organized visualizations
create_results_report(t_stats, clusters, cluster_stats, cluster_p, 
                     info, threshold=2.0, alpha=0.05, 
                     marker_name="EEG_psd_bands_spectralpower_alpha",
                     output_dir='results')
```

## References

- **MNE-Python**: Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.
- **Cluster Permutation**: Maris & Oostenveld (2007). Nonparametric statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods.
- **Linear Mixed Models**: Bates et al. (2015). Fitting Linear Mixed-Effects Models Using lme4. Journal of Statistical Software.

## SLURM Cluster Usage

The pipeline includes a SLURM script (`run_lmm_pipeline.sh`) optimized for cluster execution:

### Script Features
- **Resource Management**: Requests appropriate CPU, memory, and time
- **Logging**: Comprehensive logging with timestamps
- **Error Handling**: Graceful failure handling and reporting
- **Result Validation**: Checks output files and provides summary
- **Flexible Execution**: Supports single, all, or specific marker analysis

### SLURM Parameters
```bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
```

### Monitoring Jobs
```bash
# Check job status
squeue -u $USER

# View real-time output
tail -f logs/lmm_pipeline_*.log

# Cancel job if needed
scancel <job_id>
```

## Support

For questions or issues:
1. Check configuration file format and paths
2. Verify aggregated probe marker data exists in features directory
3. Ensure `onoff` column exists and contains continuous values (0-100)
4. Check SLURM job logs for detailed error messages
5. Verify montage file (`CACS-64_REF.bvef`) is accessible
6. Check convergence of LMM models (review warnings if any)
