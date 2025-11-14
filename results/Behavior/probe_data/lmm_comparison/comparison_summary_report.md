# LMM Analysis Approaches Comparison Report

**Analysis Date:** 2025-09-29 15:07:10

## Comparison Overview
This report compares three analytical approaches:
1. **Original Analysis**: All predictors included in models
2. **Enhanced Full Dataset**: BIC-selected predictors, all SART tasks
3. **Enhanced IE Dataset**: BIC-selected predictors, SART2/4 only

## Key Methodological Differences

### Original Approach
- **Strategy**: Include all available predictors
- **Dataset**: Focus on inclusion/exclusion conditions
- **Selection**: No variable selection, all predictors tested
- **Correction**: FDR correction across all tests

### Enhanced Approach
- **Strategy**: BIC-based recursive variable selection
- **Datasets**: Both full (all tasks) and IE-specific (Sart2/4)
- **Selection**: Forward selection optimizing BIC
- **Correction**: FDR correction on selected models only

## Key Findings

### Significant Effects Comparison
- **Original Analysis**: 1 significant effects (FDR corrected)
- **Enhanced Full Dataset**: 6 selected predictors
- **Enhanced IE Dataset**: 2 selected predictors

### Parsimony Analysis
- **Average predictors per model (Enhanced Full)**: 0.8
- **Average predictors per model (Enhanced IE)**: 0.2
- **Average BIC (Enhanced Full)**: 1554.6
- **Average BIC (Enhanced IE)**: 890.0

### Consistent Findings Across Approaches

**Original ↔ Enhanced Full overlaps:**
- valence ~ fne
**Original ↔ Enhanced IE overlaps:**
- valence ~ fne

**Unique to Enhanced Approaches:**
- PC2 ~ time_on_task
- PC2 ~ fne
- PC1 ~ fne
- PC1 ~ time_on_task
- time ~ time_on_task

## Recommendations

### When to Use Original Approach
- **Exploratory analysis**: When you want to see all possible associations
- **Hypothesis testing**: When testing specific predicted relationships
- **Comprehensive screening**: When model parsimony is not a priority

### When to Use Enhanced Approach
- **Predictive modeling**: When model generalization is important
- **Parsimonious models**: When interpretation simplicity is valued
- **Limited sample sizes**: When overfitting is a concern
- **Model comparison**: When comparing different theoretical frameworks

## Technical Considerations

### Strengths of Enhanced Approach
- **Reduced overfitting**: BIC naturally penalizes complexity
- **Better generalization**: Models likely to replicate better
- **Clearer interpretation**: Fewer variables to interpret
- **Computational efficiency**: Fewer statistical tests

### Limitations of Enhanced Approach
- **Potential missed effects**: Small but real effects might be excluded
- **Selection bias**: Variables selected based on this sample
- **Reduced power**: Fewer tests but also fewer opportunities to detect effects

## Files Generated
- `lmm_approaches_comparison.png/svg` - Side-by-side visualization
- `model_performance_comparison.png/svg` - Performance metrics comparison
- `comparison_summary_report.md` - This comprehensive report