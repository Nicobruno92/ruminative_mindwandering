# Linear Mixed Model Analysis of PCA Components - Summary Report

**Analysis Date:** 2025-12-29 19:39:32
**Sample:** 1082 observations from 35 subjects

## PCA Information
**Variables used:** valence, selfother, time
**PC1 explained variance:** 0.426 (42.6%)
**PC2 explained variance:** 0.334 (33.4%)
**PC3 explained variance:** 0.240 (24.0%)
**Total explained variance (PC1-3):** 1.000 (100.0%)

## Component Loadings
### PC1 Loadings:
- Valence: 0.708
- Self-Other: 0.522
- Time: 0.475

### PC2 Loadings:
- Valence: 0.001
- Self-Other: -0.673
- Time: 0.739

### PC3 Loadings:
- Valence: 0.706
- Self-Other: -0.523
- Time: -0.477

## Group Distribution
- **Controls:** 644 (59.5%)
- **Risk of Depression:** 438 (40.5%)

## Inclusion/Exclusion Distribution
- **Inclusion:** 280 (25.9%)
- **Exclusion:** 319 (29.5%)

## Models Tested
1. **Group Effect:** PC ~ Group (Controls vs Risk of Depression)
2. **Inclusion/Exclusion Effect:** PC ~ Inclusion/Exclusion
3. **Interaction Effect:** PC ~ Group × Inclusion/Exclusion

## Statistical Method
- **Model:** Linear Mixed Models (LMM) with random intercepts by subject
- **Clustering:** By subject ID to account for repeated measures
- **Multiple Comparisons:** Bonferroni correction applied within each component

## Key Findings

### Group Effect
**PC1:** 0 significant fixed effects (uncorrected), 0 significant fixed effects (corrected)
**PC2:** 0 significant fixed effects (uncorrected), 0 significant fixed effects (corrected)
**PC3:** 1 significant fixed effects (uncorrected), 1 significant fixed effects (corrected)
  - group[T.Risk of Depression]: β = -0.318, p = 0.0249

### Inclusion/Exclusion Effect
- **No significant effects after correction**

### Group × Inclusion/Exclusion Interaction
**PC1:** 2 significant fixed effects (uncorrected), 2 significant fixed effects (corrected)
  - group[T.Risk of Depression]: β = -0.794, p = 0.0265
  - group[T.Risk of Depression]:inclusion_exclusion[T.inclusion]: β = 0.674, p = 0.0000
**PC2:** 0 significant fixed effects (uncorrected), 0 significant fixed effects (corrected)
**PC3:** 1 significant fixed effects (uncorrected), 0 significant fixed effects (corrected)

## Interpretation
- **PC1:** Captures the primary dimension of mind-wandering variation
- **PC2:** Captures the secondary dimension of mind-wandering variation
- **PC3:** Captures the tertiary dimension of mind-wandering variation

Significant effects indicate that group membership and/or inclusion/exclusion conditions
are associated with different patterns along the principal component dimensions.

## Files Generated
- `*_results.csv` - Complete model results for each component
- `*_significant_results.csv` - Uncorrected significant effects
- `*_significant_corrected.csv` - Bonferroni-corrected significant effects
- `*_summary.txt` - Detailed model output
- `lmm_summary_report.md` - This comprehensive report