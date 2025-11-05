# Cluster Analysis Issues and Fixes

## Problems Identified

### 1. **TFCE Not Being Used Despite Configuration**
**Issue**: Your `config.yaml` specified `method: "tfce"` but the pipeline was ignoring this and always running threshold-based clustering.

**Impact**: 
- You were getting traditional cluster-based results with arbitrary thresholds
- Giant clusters (24/64 channels) due to liberal threshold (2.0)
- Missing the benefits of threshold-free analysis

**Fix**: Modified `run_pipeline.py` to:
- Check `clustering.method` config parameter
- Call `spatial_cluster_test_tfce()` when method='tfce'
- Call `spatial_cluster_permutation_test()` when method='threshold'

### 2. **Insufficient Permutations (50)**
**Issue**: Only 50 permutations were being run.

**Impact**:
- Unreliable p-values
- Cannot detect effects below p=0.02 (1/50)
- Not suitable for publication
- High variance in null distribution

**Fix**: Changed `n_permutations: 50` → `n_permutations: 5000` in config.yaml

**Why 5000?**
- Minimum 1000 for stable p-values
- 5000+ recommended for publication
- Allows detection of p-values down to 0.0002
- Provides stable null distribution

### 3. **Giant Clusters with Threshold Method**
**Issue**: With threshold=2.0 and low permutations, you get overly large clusters.

**Why This Happens**:
- Threshold of 2.0 is relatively liberal (p≈0.05 uncorrected)
- Many adjacent channels exceed this threshold
- They merge into giant clusters
- Low permutations (50) don't provide enough resolution to reject them

**Solutions**:
1. **Use TFCE** (already configured, now working):
   - No arbitrary threshold
   - Channel-wise inference
   - Better spatial specificity
   
2. **If using threshold method**:
   - Increase threshold to 2.5-3.0
   - Use stat_fun: "max" instead of "sum" (more conservative)
   - Increase permutations to 5000+

## Current Configuration Status

Your config.yaml now has:
```yaml
clustering:
  method: "tfce"  # ✓ Will now be used
  permutation_method: "freedman_lane"
  n_permutations: 5000  # ✓ Fixed from 50
  
  # TFCE parameters
  tfce:
    E: 0.5
    H: 2.0
    n_steps: 100
```

## What Changed in Code

### `Statistics/run_pipeline.py`

1. **Added TFCE import**:
```python
from cluster_test import (
    get_channel_adjacency,
    spatial_cluster_permutation_test,
    spatial_cluster_test_tfce  # NEW
)
```

2. **Read clustering method from config**:
```python
clustering_method = config['clustering'].get('method', 'threshold')
tfce_E = config['clustering'].get('tfce', {}).get('E', 0.5)
tfce_H = config['clustering'].get('tfce', {}).get('H', 2.0)
tfce_n_steps = config['clustering'].get('tfce', {}).get('n_steps', 100)
```

3. **Branch based on method**:
```python
if clustering_method == 'tfce':
    # Call TFCE function
    tfce_map, tfce_p_values, diagnostics = spatial_cluster_test_tfce(...)
    # Convert to cluster format for downstream compatibility
else:
    # Call threshold-based function
    clusters, cluster_stats, cluster_p_values, diagnostics = spatial_cluster_permutation_test(...)
```

## Understanding the Topography Color Issue

The "uniform coloring" you saw is likely due to:

1. **Color scale range**: If most t-values are similar, the color gradient appears flat
2. **Interpolation**: MNE interpolates between electrodes, smoothing the appearance
3. **Actual t-value distribution**: Check `t_statistics.csv` - your values range from -0.6 to 3.6, which is reasonable

The t-statistics themselves look fine:
- Range: [-0.599, 3.636]
- Mean: 1.74
- 25 channels with |t| > 2.0

## Next Steps

### Option 1: Re-run with TFCE (Recommended)
```bash
# The pipeline will now use TFCE automatically
sbatch Statistics/submit_marker_array.sh
```

**Expected output**:
- Channel-wise p-values (not cluster-wise)
- No arbitrary threshold
- Better spatial specificity
- Longer runtime (~100x slower than before due to 5000 permutations)

### Option 2: Use Threshold Method with Better Settings

If you prefer traditional clustering, update config.yaml:
```yaml
clustering:
  method: "threshold"
  threshold: 2.5  # More conservative
  stat_fun: "max"  # Peak-based instead of extent-based
  n_permutations: 5000
```

## Computational Cost

With 5000 permutations:
- **Before**: ~5 minutes per marker (50 permutations)
- **Now**: ~8 hours per marker (5000 permutations)
- **With SLURM array**: All markers run in parallel, so total time = time for 1 marker

## Verification

After re-running, check:
1. Log files should show "Running TFCE-based spatial permutation test..."
2. Results should have channel-wise p-values
3. No more giant clusters (TFCE doesn't produce clusters)
4. More conservative results (higher threshold from proper null distribution)

## References

- **TFCE**: Smith & Nichols (2009). Threshold-free cluster enhancement. NeuroImage.
- **Permutation testing**: Maris & Oostenveld (2007). Nonparametric statistical testing of EEG- and MEG-data.
- **Freedman-Lane**: Winkler et al. (2014). Permutation inference for the general linear model.
