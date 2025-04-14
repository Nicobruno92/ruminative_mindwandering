# LMM-Based Spatial Cluster Permutation Testing Pipeline

This pipeline implements linear mixed model (LMM) based spatial cluster permutation testing for EEG spectral markers, following established methodologies from MNE-Python and standard statistical practices.

## Overview

The pipeline performs the following steps:
1. **Data Loading**: Load spectral power data and behavioral variables
2. **Channel-wise LMM**: Fit linear mixed models for each EEG channel independently
3. **Cluster Formation**: Identify spatially contiguous clusters of channels exceeding threshold
4. **Permutation Testing**: Generate null distribution by permuting predictor within subjects
5. **Visualization**: Create topographic maps and statistical plots

## Pipeline Structure

```
Statistics/
├── config.yaml          # Configuration file (YAML)
├── reader.py            # Data loading module
├── lmm_model.py         # Linear mixed model implementation
├── cluster_test.py      # Spatial cluster permutation testing
├── plot_results.py      # Visualization functions
└── run_pipeline.py      # Main pipeline executor
```

## Configuration

All parameters are specified in `config.yaml`:

### Key Parameters

- **LMM Formula**: R-style formula (e.g., `"power ~ 1 + onoff + (1|subject)"`)
- **Predictor of Interest**: Variable to extract t-statistics from
- **Threshold**: T-statistic threshold for cluster formation
- **Permutations**: Number of permutations for null distribution
- **Alpha**: Significance level (default: 0.05)

### Example Configuration

```yaml
lmm:
  formula: "power ~ 1 + onoff + (1|subject)"
  predictor_of_interest: "onoff"
  method: "lbfgs"
  
clustering:
  threshold: 2.0
  n_permutations: 1000
  alpha: 0.05
```

## Usage

### Basic Usage

```bash
# Activate eeg environment
conda activate eeg

# Run pipeline with default config
python Statistics/run_pipeline.py

# Run with custom config
python Statistics/run_pipeline.py --config path/to/config.yaml
```

### Data Format

The pipeline expects data in the following format:

**Input (pickle file):**
- `power_data`: numpy array with shape `(n_observations, n_channels)`
- `df_behavioral`: pandas DataFrame with:
  - `subject`: subject identifier (required for random effects)
  - Additional columns: predictor variables as specified in formula

**Example:**
```python
import pickle
import numpy as np
import pandas as pd

# Create example data
power_data = np.random.randn(1000, 64)  # 1000 observations, 64 channels
df_behavioral = pd.DataFrame({
    'subject': np.repeat(range(1, 21), 50),  # 20 subjects, 50 obs each
    'onoff': np.random.randint(0, 2, 1000),  # Binary predictor
    'distance': np.random.randn(1000)        # Continuous predictor
})

# Save as pickle
with open('data.pkl', 'wb') as f:
    pickle.dump({'power_data': power_data, 'behavioral_data': df_behavioral}, f)
```

## Methodology

### Linear Mixed Models

For each channel independently:
- Fit LMM using statsmodels with `method='lbfgs'` for deterministic optimization
- Extract t-statistic for predictor of interest
- Handle convergence failures silently (t=0, p=1)

### Spatial Clustering

- Compute channel adjacency using `mne.channels.find_ch_adjacency()`
- Identify spatially contiguous clusters exceeding threshold
- Cluster statistic = sum of t-values within cluster

### Permutation Testing

- Permute predictor variable **within subjects** to preserve within-subject structure
- Recompute LMM and identify clusters for each permutation
- Generate null distribution of maximum cluster statistics
- P-value = proportion of permutations where max_cluster_stat ≥ observed_cluster_stat

## Output

The pipeline generates:

### 1. Results Pickle (`results.pkl`)
Complete results dictionary including:
- T-statistics and p-values per channel
- Cluster information (channels, statistics, p-values)
- Configuration parameters
- MNE Info object for visualization

### 2. CSV Files
- `cluster_summary.csv`: Cluster details and significance
- `t_statistics.csv`: Per-channel t-statistics and p-values

### 3. Figures
- `cluster_test_topomap.png`: Topographic map with significant clusters
- `cluster_test_details.png`: Cluster statistics and p-values
- `cluster_test_distribution.png`: T-statistic distribution

## Key Features

- **Deterministic**: Fixed random seeds ensure reproducibility
- **BIDS-compatible**: Follows project conventions
- **MNE-standard**: Uses established MNE functions for spatial operations
- **Robust**: Handles convergence failures and edge cases
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
from reader import load_data
from lmm_model import run_lmm_per_channel
from cluster_test import get_channel_adjacency, spatial_cluster_permutation_test
from plot_results import create_results_report

# Load data
power_data, df_behavioral = load_data('data.pkl')

# Get adjacency
adjacency, ch_names = get_channel_adjacency('montage.bvef', ch_names)

# Run LMM
t_stats, p_values = run_lmm_per_channel(
    power_data, df_behavioral, 
    formula="power ~ 1 + onoff + (1|subject)",
    predictor_of_interest="onoff"
)

# Cluster test
clusters, cluster_stats, cluster_p = spatial_cluster_permutation_test(
    t_stats, power_data, df_behavioral,
    formula="power ~ 1 + onoff + (1|subject)",
    predictor_of_interest="onoff",
    adjacency=adjacency,
    threshold=2.0,
    n_permutations=1000
)

# Visualize
create_results_report(t_stats, clusters, cluster_stats, cluster_p, 
                     info, threshold=2.0, alpha=0.05, output_dir='results')
```

## References

- **MNE-Python**: Gramfort et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.
- **Cluster Permutation**: Maris & Oostenveld (2007). Nonparametric statistical testing of EEG- and MEG-data. Journal of Neuroscience Methods.
- **Linear Mixed Models**: Bates et al. (2015). Fitting Linear Mixed-Effects Models Using lme4. Journal of Statistical Software.

## Support

For questions or issues:
1. Check configuration file format
2. Verify data structure matches expected format
3. Ensure all required columns present in behavioral DataFrame
4. Check convergence of LMM models (review warnings if any)
