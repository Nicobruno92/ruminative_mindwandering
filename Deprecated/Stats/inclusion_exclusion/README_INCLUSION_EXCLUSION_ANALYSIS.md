# Inclusion/Exclusion Analysis Pipeline

This directory contains scripts for analyzing EEG markers in inclusion and exclusion conditions based on task order and SART task type.

## Overview

The analysis pipeline consists of two main components:

1. **OFF-Only Analysis**: Compares inclusion vs exclusion conditions during OFF-task (mind-wandering) periods only
2. **Interaction Analysis**: Examines the interaction between onoff conditions (ON-task vs OFF-task) and inclusion/exclusion conditions

## Condition Assignment Logic

Based on the order (IE/EI) in `metadata_experiment.csv` and task (Sart2/Sart4):

- **IE order**: Inclusion first (Sart2), then Exclusion (Sart4)
- **EI order**: Exclusion first (Sart2), then Inclusion (Sart4)

This design allows for counterbalanced assignment of inclusion/exclusion conditions across subjects.

## Analysis Scripts

### 1. OFF-Only Inclusion vs Exclusion Analysis

**Script**: `lmm_fdr_analysis_off_inclusion_exclusion.py`  
**Runner**: `run_off_ie_analysis.sh`

#### Purpose
Compares EEG markers between inclusion and exclusion conditions during OFF-task periods only (mind-wandering episodes).

#### Model
```
marker ~ inclusion_exclusion_label + (1|subject_id)
```

#### Features
- Filters data for OFF conditions (`onoff_label == 'low'`) only
- Uses only Sart2 and Sart4 tasks
- Assigns inclusion/exclusion labels based on order and task
- Applies FDR correction for multiple comparisons (α = 0.05)
- Generates comprehensive topoplots showing:
  - Raw values for inclusion and exclusion conditions
  - Difference maps (inclusion minus exclusion)
  - T-statistics and significant channels
  - Summary tables

#### Output Directory
`results/off_inclusion_exclusion_comparison/`

#### Files Generated
For each marker:
- `{marker}_inclusion_exclusion_results_full_results.csv`: Complete results
- `{marker}_inclusion_exclusion_results_significant_results.csv`: Significant channels only
- `{marker}_inclusion_exclusion_plot.png`: Comprehensive visualization

### 2. OnOff × Inclusion/Exclusion Interaction Analysis

**Script**: `lmm_fdr_interaction_analysis_onoff_ie.py`  
**Runner**: `run_onoff_ie_interaction_analysis.sh`

#### Purpose
Examines the interaction between task state (ON-task vs OFF-task) and inclusion/exclusion conditions.

#### Model
```
marker ~ onoff_label * inclusion_exclusion_label + (1|subject_id)
```

This model tests three effects:
1. **Main effect of OnOff**: High (on-task) vs Low (off-task) across all conditions
2. **Main effect of Inclusion/Exclusion**: Inclusion vs Exclusion across all task states
3. **Interaction effect**: Whether the OnOff difference varies between inclusion/exclusion conditions

#### Features
- Uses both ON and OFF conditions
- Uses only Sart2 and Sart4 tasks
- Applies FDR correction separately to each effect (α = 0.05)
- Generates comprehensive plots with separate rows for each effect:
  - T-statistics topoplots for each effect
  - Significant channels for each effect
  - Summary table comparing all effects

#### Output Directory
`results/onoff_inclusion_exclusion_interaction_analysis/`

#### Files Generated
For each marker:
- `{marker}_onoff_ie_interaction_results_full_results.csv`: All effects combined
- `{marker}_onoff_ie_interaction_results_significant_results.csv`: Significant channels only
- `{marker}_onoff_ie_interaction_results_{effect_name}_results.csv`: Individual effect files
- `{marker}_onoff_ie_interaction_plot.png`: Comprehensive visualization

## Usage

### Running OFF-Only Analysis
```bash
cd Stats/inclusion_exclusion
./run_off_ie_analysis.sh
```

### Running Interaction Analysis
```bash
cd Stats/inclusion_exclusion
./run_onoff_ie_interaction_analysis.sh
```

## Data Requirements

### Input Files
1. **Aggregated MNE Markers**: 
   - `results/aggregated_mne_markers/aggregated_mne_markers_onoff_valence_confidence_time_selfother_5trials_go_correct_probe.csv`
   - Probe-level aggregated data with all EEG markers

2. **Metadata**: 
   - `metadata_experiment.csv`
   - Contains subject information including order (IE/EI)

### Subject Distribution
Based on `metadata_experiment.csv`:
- **Total subjects**: 44
- **IE order**: 22 subjects (Sart2=Inclusion, Sart4=Exclusion)
- **EI order**: 22 subjects (Sart2=Exclusion, Sart4=Inclusion)
- **Total inclusion conditions**: 44 (22 from IE-Sart2 + 22 from EI-Sart4)
- **Total exclusion conditions**: 44 (22 from IE-Sart4 + 22 from EI-Sart2)

## Statistical Approach

### Linear Mixed-Effects Models (LMM)
- **Fixed effects**: Condition variables (inclusion/exclusion, onoff)
- **Random effects**: Subject-specific intercepts `(1|subject_id)`
- **Estimation method**: Maximum Likelihood (ML)

### Multiple Comparisons Correction
- **Method**: False Discovery Rate (FDR) correction
- **Alpha level**: 0.05
- **Implementation**: Benjamini-Hochberg procedure
- Applied separately to each effect in interaction analysis

### Channel-wise Analysis
- Each EEG channel analyzed separately
- Results aggregated across channels for each marker
- Topoplots show spatial distribution of effects

## Interpretation Guidelines

### OFF-Only Analysis
- **Positive coefficients**: Higher values in inclusion vs exclusion during mind-wandering
- **Negative coefficients**: Lower values in inclusion vs exclusion during mind-wandering
- **Clinical relevance**: Differences in neural activity patterns during mind-wandering between inclusion/exclusion conditions

### Interaction Analysis

#### Main Effect of OnOff
- **Positive coefficients**: Higher values during on-task vs off-task periods
- **Negative coefficients**: Lower values during on-task vs off-task periods
- **Interpretation**: General task engagement effects

#### Main Effect of Inclusion/Exclusion
- **Positive coefficients**: Higher values in inclusion vs exclusion conditions
- **Negative coefficients**: Lower values in inclusion vs exclusion conditions
- **Interpretation**: General differences between inclusion/exclusion conditions

#### Interaction Effect
- **Positive coefficients**: OnOff difference is larger in inclusion vs exclusion
- **Negative coefficients**: OnOff difference is smaller in inclusion vs exclusion
- **Interpretation**: Inclusion/exclusion conditions modulate task engagement differently

## Output Visualization

### Topoplots
- **Color scale**: Red-Blue diverging colormap
  - Red: Positive values/effects
  - Blue: Negative values/effects
- **Contours**: 6 contour lines for smooth visualization
- **Sensors**: Black dots showing electrode positions
- **Significant channels**: Highlighted in separate plots

### Summary Tables
- Top 10 most significant channels per effect
- Includes T-statistics, p-values, FDR-corrected p-values, and coefficients

## Technical Notes

### Performance Optimization
- Chunked data loading for large files (>500MB)
- Parallel processing disabled to avoid conflicts
- Memory-efficient processing with progress bars

### Error Handling
- Graceful handling of missing channels
- Fallback to bar plots if topoplot creation fails
- Comprehensive logging of analysis progress

### File Organization
- Results organized by analysis type
- Consistent naming conventions
- Both full and significant-only result files

## Dependencies

### Python Packages
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `matplotlib`: Plotting
- `mne`: EEG analysis and topoplots
- `statsmodels`: Linear mixed-effects models
- `scipy`: Statistical functions
- `tqdm`: Progress bars

### System Requirements
- Python 3.7+
- Sufficient RAM for large datasets (recommended: 16GB+)
- Storage space for results (estimated: 1-2GB per analysis)

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce `chunksize` in data loading sections
2. **Convergence warnings**: Normal for some channels/markers, results still valid
3. **Missing topoplots**: Check MNE installation and electrode positioning
4. **No significant results**: Consider adjusting FDR alpha or examining raw p-values

### Log Files
Check console output for detailed progress information and error messages.

## References

### Statistical Methods
- Linear Mixed-Effects Models: Bates et al. (2015)
- FDR Correction: Benjamini & Hochberg (1995)

### EEG Analysis
- MNE-Python: Gramfort et al. (2013)
- Cluster-based permutation testing: Maris & Oostenveld (2007)

---

**Note**: This analysis pipeline is designed for the specific experimental design with inclusion/exclusion conditions based on task order. Modifications may be needed for different experimental paradigms. 