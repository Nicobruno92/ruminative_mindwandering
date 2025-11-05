# Quick Fix Summary

## What Was Wrong

1. **TFCE configured but not running** - Pipeline ignored `method: "tfce"` setting
2. **Only 50 permutations** - Way too low for reliable statistics
3. **Giant clusters** - Result of liberal threshold + insufficient permutations

## What Was Fixed

### Code Changes
- ✅ `Statistics/run_pipeline.py` - Now checks and uses TFCE when configured
- ✅ `Statistics/config.yaml` - Increased permutations from 50 → 5000

### Configuration
Your config now properly uses:
- **Method**: TFCE (threshold-free)
- **Permutations**: 5000 (was 50)
- **Permutation method**: Freedman-Lane (more powerful)

## To Re-run Analysis

```bash
# Submit all markers as array job
cd /Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering
sbatch Statistics/submit_marker_array.sh
```

## What to Expect

### Runtime
- **Before**: ~5 min/marker × 48 markers = 4 hours total (in parallel)
- **Now**: ~8 hours/marker × 48 markers = 8 hours total (in parallel)

### Results
- No more giant clusters
- Channel-wise p-values (not cluster-wise)
- More conservative/reliable statistics
- Proper TFCE output in logs

## Verify It's Working

Check the log files for:
```
Running TFCE-based spatial permutation test...
  TFCE parameters: E=0.5, H=2.0, n_steps=100
```

Instead of:
```
Running spatial cluster permutation test...
  Threshold: 2.0
```

## Alternative: Use Threshold Method

If you prefer traditional clustering, change in `config.yaml`:

```yaml
clustering:
  method: "threshold"  # Change from "tfce"
  threshold: 2.5       # Increase from 2.0
  stat_fun: "max"      # Change from "sum"
  n_permutations: 5000 # Keep this
```

## Questions?

See `CLUSTER_ANALYSIS_FIXES.md` for detailed explanation.
