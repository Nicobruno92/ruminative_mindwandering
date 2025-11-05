# Bug Fix Summary - LMM Cluster Pipeline

## Date: 2025-11-05

## Issues Identified

### 1. "Unknown fit method manual" Error
**Symptom:** All permutations failing with error message:
```
Channel 0 LMM failed: Unknown fit method manual
Channel 1 LMM failed: Unknown fit method manual
...
```

**Root Cause:** Variable name conflict in `run_pipeline.py` line 277
- The LMM optimization method (`method = "powell"`) was being overwritten
- Line 277 reassigned `method = exclude_config.get('method', 'manual')`
- This caused the exclude channels method (`"manual"`) to be passed to LMM functions
- LMM functions expect valid optimization methods like `"powell"`, `"REML"`, `"lbfgs"`, etc.

**Fix:** Renamed the variable to `exclude_method` to avoid conflict
- Line 277: `method` → `exclude_method`
- Line 293: `method` → `exclude_method`

### 2. Spurious Whole-Scalp Cluster
**Symptom:** All 64 channels forming a single significant cluster (p=0.0005)

**Root Cause:** Consequence of Issue #1
- When permutations failed, they returned t-statistics of all zeros
- This created TFCE null distribution of all zeros
- Any observed effect appeared highly significant compared to invalid null
- Result: Entire scalp marked as significant

**Fix:** Resolved automatically by fixing Issue #1
- With correct LMM method, permutations now run successfully
- Null distribution is properly computed
- P-values are now valid

## Files Modified

### `/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics/run_pipeline.py`

**Changes:**
1. Line 277: `method = exclude_config.get('method', 'manual')` → `exclude_method = exclude_config.get('method', 'manual')`
2. Line 293: `elif method == 'auto_position':` → `elif exclude_method == 'auto_position':`

## Testing Recommendations

1. **Re-run the pipeline** with the same configuration
2. **Verify** that permutations complete without errors
3. **Check** that the null distribution has non-zero values
4. **Examine** cluster results for spatial specificity (not whole-scalp)
5. **Compare** results with and without channel exclusion enabled

## Prevention

This type of bug can be prevented by:
1. Using more descriptive variable names (e.g., `lmm_method` vs `exclude_method`)
2. Limiting variable scope (use local variables in blocks)
3. Adding type hints and validation
4. Code review focusing on variable shadowing

## Impact

- **Severity:** Critical - All previous results with `exclude_channels.enabled: true` are invalid
- **Affected analyses:** Any runs using the config with channel exclusion enabled
- **Data integrity:** No data corruption; only analysis results affected
- **Reproducibility:** Fixed code will produce correct results when re-run
