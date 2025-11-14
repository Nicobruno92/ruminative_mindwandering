# Linear Mixed Model Beta Analysis - Summary Report

**Analysis Date:** 2025-09-29 14:57:59
**Total Models Fitted:** 96
**Successful Models:** 96

## Analysis Overview
This analysis recreates correlation matrix patterns using Linear Mixed Models (LMMs)
with standardized beta coefficients. This approach accounts for:
- Repeated measures structure (random intercepts by participant)
- Experimental design (condition and group effects)
- Multiple comparison correction (FDR)

## Variables Analyzed
**Dependent Variables:** onoff, valence, time, selfother, confidence, PC1, PC2, PC3
**Experimental Predictors:** condition, group, condition_x_group
**Continuous Predictors:** age, bdi, rrs_tot, mwq, fne, self_esteem, ctq_tot, a_rsq, sris

## Statistical Method
- **Model Type:** Linear Mixed Models with random intercepts by participant
- **Coefficients:** Standardized betas (all variables z-scored)
- **Multiple Comparisons:** FDR correction across all tests
- **Significance Threshold:** p < 0.05

## Key Findings

### Significant Effects (FDR corrected p < 0.05)

- **Valence ~ FNE:** β = -0.389, p = 0.016 (medium effect)

### Effects Significant at Uncorrected p < 0.05 (18 total)

- Valence ~ Group: β = -0.484, p = 0.046
- Confidence ~ Condition×Group: β = -0.412, p = 0.025
- OnOff ~ Group: β = -0.400, p = 0.001
- Valence ~ FNE: β = -0.389, p = 0.000 (FDR sig.)
- PC1 ~ FNE: β = -0.362, p = 0.003
- PC1 ~ RRS: β = -0.336, p = 0.007
- Valence ~ Self-Est.: β = 0.293, p = 0.011
- Confidence ~ SRIS: β = 0.289, p = 0.037
- Valence ~ MWQ: β = -0.279, p = 0.035
- Valence ~ RRS: β = -0.277, p = 0.016

## Files Generated
- `lmm_beta_heatmap.png/svg` - Visualization of standardized betas
- `lmm_beta_results_complete.csv` - Complete model results
- `beta_matrix.csv` - Beta coefficient matrix
- `pval_matrix.csv` - Uncorrected p-value matrix
- `pval_fdr_matrix.csv` - FDR-corrected p-value matrix
- `se_matrix.csv` - Standard error matrix
- `significant_effects_fdr.csv` - FDR-significant effects only
- `lmm_summary_report.md` - This comprehensive report