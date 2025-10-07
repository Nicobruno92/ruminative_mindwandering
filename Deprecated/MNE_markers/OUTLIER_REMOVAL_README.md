# Subject-Level Outlier Removal in MNE Markers Aggregation

## Overview

The MNE markers aggregation script now includes automatic subject-level outlier removal during the file loading phase. This is much more efficient than doing it after loading all data and helps improve data quality by removing obvious artifacts and problematic channels at the individual subject level.

## How It Works

1. **Per-Subject Processing**: During parallel file loading, each subject's data is processed individually.

2. **Artifact Detection**: For each subject, the script identifies marker-channel combinations that show clear signs of artifacts:
   - **Mean-Median Discrepancy**: When the mean is very different from the median (>200% relative difference), suggesting extreme outliers
   - **High Variability**: When the coefficient of variation is extremely high (>5.0), indicating very noisy channels

3. **Data Removal**: Problematic marker-channel combinations are removed from that subject's data before it's combined with other subjects.

4. **Efficiency**: This approach is much faster because it processes each subject file independently during the parallel loading phase, rather than requiring a separate global outlier detection step.

## Configuration Options

### Command Line Arguments

- `--remove_outliers` (default: True): Enable subject-level outlier removal
- `--no_remove_outliers`: Disable outlier removal
- `--outlier_method` (default: 'iqr'): Currently uses artifact detection regardless of setting

### Programmatic Usage

```python
# Example with outlier removal enabled (default)
output_path = run_aggregation(
    input_dir='./results/mne_markers',
    output_dir='./results/aggregated_mne_markers',
    conditions=['onoff'],
    remove_outliers=True,
    outlier_method='iqr',
    iqr_factor=1.5,
    z_threshold=3.0
)

# Example with outlier removal disabled
output_path = run_aggregation(
    input_dir='./results/mne_markers',
    output_dir='./results/aggregated_mne_markers',
    conditions=['onoff'],
    remove_outliers=False
)
```

## Output Files

### Main Aggregated Data
The main output file will include `_outliers_removed_{method}` in the filename when outlier removal is enabled:
- Example: `aggregated_mne_markers_onoff_5trials_outliers_removed_iqr_task.csv`

### No Additional Documentation Files
Unlike the previous version, this simplified approach does not generate separate outlier documentation files. The outlier removal happens during file loading and removes obvious artifacts at the individual subject level, making detailed cross-subject outlier logs unnecessary.

## Quality Control

### When to Use Outlier Removal
- **Recommended**: For most analyses, especially when combining data across subjects
- **Consider carefully**: When sample sizes are very small (< 10 subjects per condition)
- **May skip**: For exploratory analyses where you want to see the full data distribution

### Artifact Detection Approach
- **Simple and Fast**: Uses basic statistical checks that work well for detecting obvious artifacts
- **Subject-Specific**: Applied independently to each subject, so no assumptions about population distributions
- **Conservative**: Only removes clearly problematic data, not just statistical outliers

### Validation Steps
1. Compare results with and without outlier removal
2. Check for reasonable data ranges in the final aggregated results
3. Monitor for any subjects with unusually sparse data after processing

## Technical Details

### Artifact Detection Methods

**Mean-Median Discrepancy:**
- `relative_diff = |mean - median| / |mean|`
- Threshold: relative_diff > 2.0 (200%)
- Detects: Extreme outliers pulling the mean away from the median

**High Variability Detection:**
- `coefficient_of_variation = std / |mean|`
- Threshold: CV > 5.0
- Detects: Very noisy channels with excessive variability

### Computational Efficiency
- Artifact detection happens during parallel file loading, adding minimal overhead
- No need for a separate outlier removal step that processes the entire combined dataset
- Memory efficient since each subject is processed independently
- Much faster than the previous population-based outlier detection approach

## Example Usage

```bash
# Basic usage with default artifact removal
python aggregate_mne_markers.py \
    --input_dir ./results/mne_markers \
    --output_dir ./results/aggregated_mne_markers \
    --conditions onoff valence \
    --aggregate_level task

# Disable artifact removal
python aggregate_mne_markers.py \
    --input_dir ./results/mne_markers \
    --output_dir ./results/aggregated_mne_markers \
    --conditions onoff \
    --no_remove_outliers
```

## Version Information
- Added in: MNE markers aggregation script v2.0
- Dependencies: scipy.stats (for z-score calculation)
- Compatible with: All existing aggregation levels (probe, task, subject) 