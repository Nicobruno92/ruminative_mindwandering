# LMM Reader Module Documentation

## Overview

The `reader.py` module has been completely rewritten to handle aggregated probe marker data from the output of `aggregate_markers_by_probe.py`. This module is designed for LMM-based spatial cluster permutation testing across channels for each marker.

## Key Features

- **Loads long-format CSV data**: Reads aggregated probe marker files created by the aggregation pipeline
- **Merges across subjects/tasks**: Combines all probe data into a single dataframe
- **Channel-wise preparation**: Prepares data for cluster permutation testing across channels
- **Flexible filtering**: Supports filtering by subjects, tasks, and marker types
- **LMM-ready format**: Outputs data in the correct format for mixed-effects modeling

## Data Structure

### Input Files
The reader expects CSV files with the following naming pattern:
```
sub-XX_task-YY_desc-probe-NNN_[evoked|state]_aggMarkers.csv
```

Located in:
```
features_root/sub-XX/eeg/junifer/
```

### CSV File Format
Each CSV file contains long-format data with one row per channel-marker combination per probe:

| Column | Description | Example |
|--------|-------------|---------|
| `subject` | Subject identifier | "02" |
| `task` | Task identifier | "Sart1" |
| `probe_number` | Probe number | 1 |
| `probe_label` | Probe classification | "onTask" or "offTask" |
| `marker_type` | Type of marker | "evoked" or "state" |
| `marker` | Marker name | "EEG_psd_bands_spectralpower_alpha" |
| `channel` | EEG channel name | "Fp1" |
| `value` | Marker value | 0.123 |
| Additional metadata columns | Probe metadata | `onoff`, `valence`, etc. |

## Main Functions

### `load_all_probe_data()`

Loads and combines all aggregated probe marker CSV files.

```python
df_all = load_all_probe_data(
    features_root="/path/to/features",
    subjects=["02", "03", "04"],  # Optional: filter subjects
    tasks=["Sart1", "Sart2"],     # Optional: filter tasks  
    marker_types=["evoked"],      # Optional: filter marker types
    verbose=True
)
```

**Returns**: Combined DataFrame with all probe data in long format

### `prepare_data_for_lmm()`

Prepares data for LMM analysis for a specific marker.

```python
power_data, df_behavioral, channels = prepare_data_for_lmm(
    df=df_all,
    marker_name="EEG_psd_bands_spectralpower_alpha",
    formula="power ~ probe_label + (1|subject)",
    include_channels=["Fp1", "Fp2", "F3", "F4"],  # Optional
    exclude_channels=["EOG"]  # Optional
)
```

**Returns**:
- `power_data`: numpy array with shape (n_observations, n_channels)
- `df_behavioral`: DataFrame with behavioral variables for LMM
- `channels`: List of channel names

### `prepare_channel_data()`

Prepares data for a single channel for LMM analysis.

```python
channel_df = prepare_channel_data(
    power_data=power_data,
    df_behavioral=df_behavioral,
    channel_idx=0,
    channels=channels
)
```

**Returns**: DataFrame ready for LMM analysis with 'power' column

### `get_available_markers()`

Gets list of available markers from the data.

```python
markers = get_available_markers(
    features_root="/path/to/features",
    marker_types=["evoked", "state"]  # Optional
)
```

**Returns**: Dictionary mapping marker types to lists of marker names

### `get_channel_names()`

Gets list of available channel names.

```python
channels = get_channel_names(features_root="/path/to/features")
```

**Returns**: Sorted list of channel names

## Usage Workflow

### 1. Load All Data
```python
from reader import load_all_probe_data

df_all = load_all_probe_data(
    features_root="/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/features",
    verbose=True
)
```

### 2. Explore Available Markers
```python
from reader import get_available_markers

markers = get_available_markers(features_root)
for marker_type, marker_list in markers.items():
    print(f"{marker_type}: {marker_list}")
```

### 3. Prepare Data for Analysis
```python
from reader import prepare_data_for_lmm

power_data, df_behavioral, channels = prepare_data_for_lmm(
    df=df_all,
    marker_name="EEG_psd_bands_spectralpower_alpha",
    formula="power ~ probe_label + (1|subject)"
)
```

### 4. Run LMM for Each Channel
```python
from reader import prepare_channel_data

results = []
for channel_idx, channel_name in enumerate(channels):
    # Prepare data for this channel
    channel_df = prepare_channel_data(
        power_data=power_data,
        df_behavioral=df_behavioral,
        channel_idx=channel_idx,
        channels=channels
    )
    
    # Run LMM (using your preferred LMM library)
    # lmm_result = run_lmm(channel_df, formula)
    # results.append(lmm_result)
```

### 5. Cluster Permutation Testing
Use the results from all channels to perform cluster permutation testing across the scalp.

## Data Validation

The module includes several validation checks:

- **File existence**: Checks that features root directory exists
- **Marker availability**: Validates that requested markers exist in data
- **Formula variables**: Ensures all variables in LMM formula exist in behavioral data
- **Data consistency**: Checks for missing values and data integrity

## Error Handling

Common error scenarios and solutions:

1. **FileNotFoundError**: Update `features_root` path
2. **ValueError (marker not found)**: Check available markers with `get_available_markers()`
3. **ValueError (variables not found)**: Check formula variables exist in data
4. **Empty data**: Verify that aggregation pipeline has run successfully

## Performance Considerations

- **Memory usage**: Large datasets may require filtering by subjects/tasks
- **Loading time**: First load may be slow; subsequent operations are faster
- **File I/O**: CSV files are read individually and combined in memory

## Integration with LMM Pipeline

This reader is designed to work with:
- **Mixed-effects models**: Provides data in correct format for LMM libraries
- **Cluster permutation**: Supports spatial cluster testing across channels
- **Multiple comparisons**: Handles correction for multiple channels/markers

## Example Script

See `example_usage.py` for a complete working example that demonstrates:
- Loading data
- Exploring available markers and channels
- Preparing data for LMM analysis
- Setting up for cluster permutation testing

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `pathlib`: File path handling
- Standard library: `typing`, `warnings`

## Migration from Old Reader

The old `load_data()` function is deprecated but maintained for backward compatibility. New code should use:
- `load_all_probe_data()` for initial data loading
- `prepare_data_for_lmm()` for marker-specific preparation
- `prepare_channel_data()` for channel-specific analysis
