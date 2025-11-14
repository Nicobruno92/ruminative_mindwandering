# Enhanced Linear Mixed Model Analysis Report

**Analysis Date:** 2025-09-30 15:24:00

## Analysis Overview
This enhanced analysis includes:
1. **Time-on-task variable**: Cumulative probe number across SART tasks
2. **Two dataset analyses**:
   - Full dataset: All tasks (Sart1-4) with baseline vs experimental condition
   - IE dataset: Only Sart2/4 with inclusion vs exclusion conditions
3. **BIC-based variable selection**: Recursive forward selection for optimal models
4. **Enhanced visualization**: Diagonal lines indicate variables not selected

## Key Features
- **Model Selection**: BIC-optimized models for each dependent variable
- **Multiple Comparison Correction**: FDR correction applied to selected coefficients
- **Repeated Measures**: Random intercepts by participant
- **Standardized Coefficients**: All variables z-scored for comparability

## Variable Selection Results

### Enhanced Full Dataset

**Summary**: 8 models fitted, average 0.9 predictors selected per model

**OnOff**: 1 predictors selected
  - Selected: Time-on-Task
  - BIC: 648.86
  - Significant: Time-on-Task (β=-0.049, p=0.001); C(group, Treatment(reference='Controls'))[T.Risk of Depression] (β=-0.391, p=0.001); Group Var (β=0.689, p=0.001)

**Valence**: 1 predictors selected
  - Selected: FNE
  - BIC: 1433.92
  - Significant: FNE (β=-0.390, p=0.000); Group Var (β=0.583, p=0.001)

**Time**: 1 predictors selected
  - Selected: Time-on-Task
  - BIC: 1750.60
  - Significant: Time-on-Task (β=-0.111, p=0.001); Group Var (β=0.089, p=0.022)

**Self/Other**: 0 predictors selected
  - No predictors selected (intercept-only model)

**Confidence**: 0 predictors selected
  - No predictors selected (intercept-only model)

**PC1**: 2 predictors selected
  - Selected: FNE, Time-on-Task
  - BIC: 1501.04
  - Significant: FNE (β=-0.352, p=0.004); Time-on-Task (β=-0.079, p=0.003); Group Var (β=0.795, p=0.001)

**PC2**: 2 predictors selected
  - Selected: FNE, Time-on-Task
  - BIC: 1705.48
  - Significant: FNE (β=-0.284, p=0.003); Time-on-Task (β=0.110, p=0.000); Group Var (β=0.276, p=0.003)

**PC3**: 0 predictors selected
  - No predictors selected (intercept-only model)


### Enhanced IE Dataset

**Summary**: 8 models fitted, average 0.4 predictors selected per model

**OnOff**: 0 predictors selected
  - No predictors selected (intercept-only model)

**Valence**: 1 predictors selected
  - Selected: FNE
  - BIC: 802.58
  - Significant: FNE (β=-0.301, p=0.003); C(condition, Treatment(reference='inclusion'))[T.exclusion]:C(group, Treatment(reference='Controls'))[T.Risk of Depression] (β=-0.625, p=0.000); Group Var (β=0.585, p=0.002)

**Time**: 1 predictors selected
  - Selected: Time-on-Task
  - BIC: 1030.89
  - Significant: Time-on-Task (β=-0.117, p=0.011)

**Self/Other**: 0 predictors selected
  - No predictors selected (intercept-only model)

**Confidence**: 1 predictors selected
  - Selected: SRIS
  - BIC: 1067.08
  - Significant: SRIS (β=0.365, p=0.009); C(group, Treatment(reference='Controls'))[T.Risk of Depression] (β=-0.792, p=0.009); C(condition, Treatment(reference='inclusion'))[T.exclusion]:C(group, Treatment(reference='Controls'))[T.Risk of Depression] (β=0.730, p=0.000); Group Var (β=0.644, p=0.002)

**PC1**: 0 predictors selected
  - No predictors selected (intercept-only model)

**PC2**: 0 predictors selected
  - No predictors selected (intercept-only model)

**PC3**: 0 predictors selected
  - No predictors selected (intercept-only model)


## Methodology
### Variable Selection Process
1. **Exhaustive Search**: Evaluates ALL possible combinations of predictors
2. **BIC Optimization**: Selects model with globally minimum BIC
3. **Complete Evaluation**: All 2^n combinations evaluated (no limits)
4. **Guaranteed Optimum**: Unlike greedy algorithms, finds true global optimum

### Statistical Approach
- **Base Model**: Experimental predictors (condition/group) + random intercepts
- **Candidate Predictors**: Age, psychometric scales, time-on-task
- **Selection Criterion**: Bayesian Information Criterion (BIC)
- **Multiple Comparisons**: FDR correction on final selected models

## Interpretation Guide
### Visualization Elements
- **Colored cells**: Variables selected with standardized beta coefficients
- **Diagonal lines (///)**: Variables not selected in optimal model
- **White background**: Non-significant effects (FDR p ≥ 0.05)
- **Color intensity**: Effect size magnitude

### Model Selection Benefits
- **Reduced overfitting**: BIC penalizes model complexity
- **Improved interpretability**: Focus on most important predictors
- **Better generalization**: Optimal bias-variance tradeoff

## Files Generated
- `enhanced_lmm_heatmap_full.png/svg` - Full dataset visualization
- `enhanced_lmm_heatmap_ie.png/svg` - IE dataset visualization
- `model_selection_summary_full.csv` - Full dataset model summaries
- `model_selection_summary_ie.csv` - IE dataset model summaries
- `selected_coefficients_full.csv` - Full dataset selected coefficients
- `selected_coefficients_ie.csv` - IE dataset selected coefficients
- `enhanced_analysis_report.md` - This comprehensive report