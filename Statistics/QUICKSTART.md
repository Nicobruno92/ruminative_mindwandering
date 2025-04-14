# Statistics Pipeline - Quick Reference

## Quick Start

```bash
# 1. Activate environment
conda activate eeg

# 2. Run pipeline
python Statistics/run_pipeline.py

# 3. Check results
ls results/Statistics/
```

## Essential Configuration

Edit `Statistics/config.yaml`:

```yaml
project:
  data_path: "path/to/your/data.pkl"  # CHANGE THIS
  output_path: "results/Statistics"
  montage_path: "Preprocessing_pipeline_new/CACS-64_REF.bvef"

lmm:
  formula: "power ~ 1 + onoff + (1|subject)"  # CHANGE THIS
  predictor_of_interest: "onoff"              # CHANGE THIS
  
clustering:
  threshold: 2.0        # T-statistic threshold
  n_permutations: 1000  # More = better (but slower)
  alpha: 0.05          # Significance level
```

## Data Format

Your data file must contain:

```python
{
    'power_data': np.ndarray,      # Shape: (n_observations, n_channels)
    'behavioral_data': pd.DataFrame # Must have 'subject' column + predictors
}
```

Example:
```python
import pickle
import numpy as np
import pandas as pd

data = {
    'power_data': np.random.randn(1000, 64),  # 1000 obs, 64 channels
    'behavioral_data': pd.DataFrame({
        'subject': np.repeat(range(1, 21), 50),  # Subject IDs
        'onoff': np.random.randint(0, 2, 1000),  # Binary predictor
        'distance': np.random.randn(1000)        # Continuous predictor
    })
}

with open('my_data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

## Common Tasks

### Change Formula
```yaml
# Simple effect
formula: "power ~ 1 + onoff + (1|subject)"

# With continuous covariate
formula: "power ~ 1 + onoff + distance + (1|subject)"

# With interaction
formula: "power ~ 1 + onoff * distance + (1|subject)"
```

### Adjust Sensitivity
```yaml
clustering:
  threshold: 1.5   # Lower = more sensitive (more clusters)
  threshold: 2.5   # Higher = more conservative (fewer clusters)
```

### More Permutations
```yaml
clustering:
  n_permutations: 5000  # More accurate p-values (slower)
```

## Output Files

| File | Description |
|------|-------------|
| `results.pkl` | Complete results (t-stats, clusters, config) |
| `cluster_summary.csv` | Significant clusters table |
| `t_statistics.csv` | T-statistic per channel |
| `*_topomap.png` | Topographic map with clusters |
| `*_details.png` | Cluster statistics plots |
| `*_distribution.png` | T-statistic distribution |

## Testing

Test with synthetic data:
```bash
python Statistics/test_pipeline.py
```

## Troubleshooting

### "Variable not found in behavioral data"
→ Check formula variables match DataFrame columns

### LMM convergence warnings
→ Normal for some channels, handled automatically

### "No clusters found"
→ Try lower threshold or check effect size

### Import errors
→ Run `conda activate eeg` before execution

## Key Parameters Explained

| Parameter | What it does | Typical values |
|-----------|--------------|----------------|
| `threshold` | T-value needed for cluster | 1.5-3.0 |
| `n_permutations` | Accuracy of p-values | 1000-10000 |
| `alpha` | Significance cutoff | 0.01, 0.05 |
| `tail` | Test direction | 0=two-sided, 1=positive, -1=negative |
| `method` | LMM optimizer | 'lbfgs' (recommended) |

## Example Workflows

### Basic Analysis
```bash
# 1. Prepare data (create pickle with power_data + behavioral_data)
# 2. Edit config.yaml (data_path, formula, predictor)
# 3. Run pipeline
python Statistics/run_pipeline.py
# 4. Check results/Statistics/
```

### Custom Analysis
```python
from reader import load_data
from lmm_model import run_lmm_per_channel

# Load your data
power, behavior = load_data('mydata.pkl')

# Run LMM
t_stats, p_vals = run_lmm_per_channel(
    power, behavior,
    formula="power ~ 1 + condition + (1|subject)",
    predictor_of_interest="condition"
)

# Use t_stats for further analysis
```

## References

- **Documentation**: `Statistics/README.md`
- **Implementation**: `Statistics/IMPLEMENTATION_SUMMARY.md`
- **Test**: `Statistics/test_pipeline.py`

## Support Checklist

Before asking for help:
- [ ] Activated `eeg` environment
- [ ] Data file exists and is readable
- [ ] `subject` column exists in behavioral_data
- [ ] Formula variables exist in behavioral_data
- [ ] Config paths are correct (absolute or relative)
- [ ] Montage file exists at specified path

## Performance Notes

Approximate runtime (on typical workstation):
- 64 channels, 1000 observations, 1000 permutations: ~10-20 minutes
- Scales linearly with permutations
- Scales linearly with observations
- Can use `n_jobs=-1` for parallel permutations
