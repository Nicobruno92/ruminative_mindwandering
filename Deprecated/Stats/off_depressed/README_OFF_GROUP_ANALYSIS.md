# OFF Task Group Comparison Analysis

This directory contains scripts for comparing OFF task conditions between Group 1 and Group 2 (depressed patients) using Linear Mixed-Effects Models (LMM) with False Discovery Rate (FDR) correction.

## Overview

The analysis compares EEG markers during OFF-task conditions between two groups of depressed patients:
- **Group 1**: Subjects with `group = 1` in `metadata_experiment.csv`
- **Group 2**: Subjects with `group = 2` in `metadata_experiment.csv`

## Files

### Scripts
- `lmm_fdr_analysis_off_groups.py`: Main Python script for OFF-only group comparison
- `run_off_group_analysis.sh`: Shell script to execute the OFF-only analysis
- `lmm_fdr_interaction_analysis.py`: Python script for OnOff * Group interaction analysis
- `run_interaction_analysis.sh`: Shell script to execute the interaction analysis

### Configuration
The analysis is configured to:
- Filter data for OFF task conditions only (`onoff_label == 'low'`)
- Use group information from `metadata_experiment.csv`
- Apply LMM with formula: `mean ~ group_label`
- Use subject as random effect: `subject_id`
- Apply FDR correction with α = 0.05

## Group Distribution

Based on `metadata_experiment.csv`:
- **Group 1**: 30 subjects (subjects 1-42 with group=1)
- **Group 2**: 12 subjects (subjects 17-38 with group=2)

## Usage

### Running the Analysis

#### OFF-only Group Comparison
```bash
# Navigate to the directory
cd Stats/off_depressed/

# Run the OFF-only group comparison
./run_off_group_analysis.sh

# Or run directly with Python
python3 lmm_fdr_analysis_off_groups.py
```

#### OnOff * Group Interaction Analysis
```bash
# Navigate to the directory
cd Stats/off_depressed/

# Run the interaction analysis
./run_interaction_analysis.sh

# Or run directly with Python
python3 lmm_fdr_interaction_analysis.py
```

### Output

#### OFF-only Group Comparison
Results are saved to: `../../results/off_depressed_group_comparison/`

For each EEG marker, the following files are generated:
- `{marker}_group_comparison_results_full_results.csv`: Complete results for all channels
- `{marker}_group_comparison_results_significant_results.csv`: Only significant channels (if any)
- `{marker}_group_comparison_plot.png`: Visualization of results

#### OnOff * Group Interaction Analysis
Results are saved to: `../../results/onoff_group_interaction_analysis/`

For each EEG marker, the following files are generated:
- `{marker}_interaction_results_full_results.csv`: Complete results for all effects and channels
- `{marker}_interaction_results_significant_results.csv`: Only significant channels across all effects
- `{marker}_interaction_results_{effect_name}_results.csv`: Separate file for each effect
- `{marker}_interaction_plot.png`: Comprehensive visualization showing all three effects

## Analysis Details

### Data Processing
1. Load probe-level aggregated data from MNE markers
2. Filter for OFF task conditions only (`onoff_label == 'low'`)
3. Merge with metadata to assign group labels
4. Remove subjects without group information

### Statistical Analysis

#### OFF-only Group Comparison
1. **Linear Mixed-Effects Model**: `mean ~ group_label` with random effects for `subject_id`
2. **Multiple Comparisons**: FDR correction (Benjamini-Hochberg) at α = 0.05
3. **Channel-wise Analysis**: Separate LMM for each EEG channel
4. **Group Comparison**: Group 2 vs Group 1 (positive coefficients indicate Group 2 > Group 1)

#### OnOff * Group Interaction Analysis
1. **Linear Mixed-Effects Model**: `mean ~ onoff_label * group_label` with random effects for `subject_id`
2. **Three Effects Tested**:
   - **Main effect of OnOff**: High (on-task) vs Low (off-task) across all subjects
   - **Main effect of Group**: Group 2 vs Group 1 across all conditions
   - **Interaction effect**: Whether the OnOff difference varies between groups
3. **Multiple Comparisons**: FDR correction applied separately to each effect at α = 0.05
4. **Channel-wise Analysis**: Separate LMM for each EEG channel

### Visualization

#### OFF-only Group Comparison
Each marker generates a comprehensive plot showing:
- **Top Row**: Raw values for Group 1, Group 2, and their difference (topoplots when possible)
- **Middle Row**: T-statistics and significant channels (topoplots when possible)
- **Bottom Row**: Summary table of significant channels

#### OnOff * Group Interaction Analysis
Each marker generates a comprehensive plot showing all three effects:
- **Row 1**: Main effect of OnOff - T-statistics and significant channels
- **Row 2**: Main effect of Group - T-statistics and significant channels  
- **Row 3**: Interaction effect - T-statistics and significant channels
- **Row 4**: Summary table comparing all three effects

## Interpretation

### OFF-only Group Comparison
#### Coefficients
- **Positive coefficient**: Group 2 shows higher values than Group 1 for that marker
- **Negative coefficient**: Group 1 shows higher values than Group 2 for that marker

#### Significance
- Channels are considered significant after FDR correction at α = 0.05
- Results show which brain regions differ between groups during OFF-task conditions

### OnOff * Group Interaction Analysis
#### Main Effects
- **OnOff Effect**: Brain regions where on-task differs from off-task (averaged across groups)
- **Group Effect**: Brain regions where Group 2 differs from Group 1 (averaged across conditions)

#### Interaction Effect
- **Significant interaction**: The difference between on-task and off-task varies between groups
- **Positive interaction coefficient**: The on-task vs off-task difference is larger in Group 2
- **Negative interaction coefficient**: The on-task vs off-task difference is larger in Group 1

#### Clinical Interpretation
- **Main effects** reveal general patterns of brain activity differences
- **Interaction effects** reveal group-specific responses to task demands, which may indicate different neural mechanisms in depression subtypes

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- mne
- statsmodels
- scipy
- tqdm

## Notes

- The analysis focuses specifically on OFF-task conditions to understand group differences in mind-wandering states
- Random effects account for subject-level variability
- FDR correction controls for multiple comparisons across channels
- Topoplots are generated when channel positioning is available; otherwise, bar plots are used as fallback 