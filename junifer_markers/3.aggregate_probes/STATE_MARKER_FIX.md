# State Marker Aggregation Fix

## Problem Summary

State markers were not being aggregated by probe. The script reported "No trials found for state markers" for all probes, even though state epochs and events were loaded correctly.

## Root Cause Analysis

### Issue 1: Event Description Parsing (FIXED)
**Problem**: The event parser couldn't extract `distance_to_probe` from state epoch events.

**Cause**: Different event description formats:
- **Evoked epochs**: `go/correct/-5/SART_TRIAL/onoff33/probe1`
  - Distance encoded as: `/-5/`
- **State epochs**: `state/preprobe_win/dt=-10.00/THOUGHT_PROBE/onoff1/.../probe13`
  - Distance encoded as: `dt=-10.00`

The parser only looked for the evoked format (`/-(\d+)/`), so state events had `distance_to_probe = NaN`.

**Fix**: Updated `enrich_events_with_parsed_fields()` in `helpers.py` to handle both formats:
```python
# Try evoked format: /-X/
m_dist_evoked = desc.str.extract(r"/-(\d+)/", expand=False)
# Try state format: dt=-X.00
m_dist_state = desc.str.extract(r"dt=-(\d+)\.00", expand=False)
# Use whichever matched
dist_val = pd.to_numeric(m_dist_evoked, errors="coerce").fillna(
    pd.to_numeric(m_dist_state, errors="coerce")
)
```

### Issue 2: Trial Type Filtering (FIXED)
**Problem**: Even with correct distance parsing, state trials were still filtered out.

**Cause**: The `select_trials_for_probe()` function was called with `only_go_correct=True` for both evoked and state markers. This filters for `trial_type == "go"`, but:
- **Evoked events**: Start with `go/` or `nogo/` → `trial_type = "go"` or `"nogo"`
- **State events**: Start with `state/` → `trial_type = "unknown"`

When `only_go_correct=True`, all state events were filtered out because they don't have `trial_type == "go"`.

**Fix**: Updated `aggregate_markers_by_probe.py` to use `only_go_correct=False` for state markers:
```python
# Select trials for state markers using STATE events (independent system)
# NOTE: State markers represent continuous brain states, not trial-locked responses
# Therefore, we don't filter by trial_type (only_go_correct=False)
state_trials = select_trials_for_probe(
    events_df=state_events_df,
    probe_number=probe_num,
    only_go_correct=False,  # State events have trial_type='unknown'
    distance_min=state_dist_min,
    distance_max=state_dist_max,
)
```

## Why This Makes Sense

### Evoked Markers (P1, N1, P2, P3a, P3b)
- **Nature**: Trial-locked event-related potentials
- **Filtering**: Should filter for `go/correct` trials only
- **Reason**: ERPs are sensitive to trial type and accuracy
- **Distance**: -5 to -1 (5 trials closest to probe)

### State Markers (spectral, connectivity, information theory)
- **Nature**: Continuous brain state measurements
- **Filtering**: Should NOT filter by trial type
- **Reason**: State markers reflect sustained mental states, not trial-locked responses
- **Distance**: -999 to -1 (all trials before probe)

## Expected Behavior After Fix

For each probe, the script will now:

1. **Evoked markers** (distance -5 to -1):
   - Load evoked epochs (e.g., 414 epochs)
   - Parse evoked events with `/-X/` format
   - Filter for `go/correct` trials only
   - Find ~4-5 trials per probe
   - Aggregate and save evoked markers

2. **State markers** (distance -999 to -1):
   - Load state epochs (e.g., 119 epochs) 
   - Parse state events with `dt=-X.00` format
   - **No trial-type filtering** (all state events included)
   - Find multiple trials per probe (depending on probe timing)
   - Aggregate and save state markers

## Files Modified

1. **`helpers.py`** (lines 233-242):
   - Updated `enrich_events_with_parsed_fields()` to parse both distance formats

2. **`aggregate_markers_by_probe.py`** (lines 1363-1372):
   - Changed `only_go_correct=False` for state marker selection
   - Added explanatory comments

3. **`README.md`** (lines 19-29):
   - Updated documentation to clarify trial-type filtering differences

## Testing

Created and ran test scripts to verify:
1. ✅ Distance parsing works for both evoked and state formats
2. ✅ Trial selection works correctly with `only_go_correct=False`
3. ✅ Trial selection filters out state events with `only_go_correct=True` (bug reproduced)

## Next Steps

Re-run the aggregation pipeline to generate state marker files:
```bash
sbatch run_aggregate_slurm.sh
```

Expected output files per probe:
- `sub-XX_task-YY_desc-probe-NNN_evoked_aggMarkers.csv` (evoked markers)
- `sub-XX_task-YY_desc-probe-NNN_state_aggMarkers.csv` (state markers - NEW!)
