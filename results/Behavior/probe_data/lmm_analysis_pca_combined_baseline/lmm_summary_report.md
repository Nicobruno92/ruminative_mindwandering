# Linear Mixed Model Analysis of PCA Components - Summary Report

**Analysis Date:** 2026-03-03 18:39:35
**Sample:** 888 observations from 33 subjects

## PCA Information
**Variables used:** valence, selfother, time
**PC1 explained variance:** 0.421 (42.1%)
**PC2 explained variance:** 0.336 (33.6%)
**PC3 explained variance:** 0.244 (24.4%)
**Total explained variance (PC1-3):** 1.000 (100.0%)

## Component Loadings
### PC1 Loadings:
- Valence: 0.712
- Self-Other: 0.505
- Time: 0.489

### PC2 Loadings:
- Valence: 0.001
- Self-Other: -0.696
- Time: 0.718

### PC3 Loadings:
- Valence: 0.703
- Self-Other: -0.511
- Time: -0.495

## Group Distribution
- **Controls:** 484 (54.5%)
- **Risk of Depression:** 404 (45.5%)

## Inclusion/Exclusion Distribution
- **Inclusion:** 225 (25.3%)
- **Exclusion:** 262 (29.5%)

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
  - group[T.Risk of Depression]: β = -0.351, p = 0.0243

### Inclusion/Exclusion Effect
- **No significant effects after correction**

### Group × Inclusion/Exclusion Interaction
**PC1:** 2 significant fixed effects (uncorrected), 2 significant fixed effects (corrected)
  - group[T.Risk of Depression]: β = -0.765, p = 0.0402
  - group[T.Risk of Depression]:inclusion_exclusion[T.inclusion]: β = 0.680, p = 0.0001
**PC2:** 0 significant fixed effects (uncorrected), 0 significant fixed effects (corrected)
**PC3:** 1 significant fixed effects (uncorrected), 1 significant fixed effects (corrected)
  - group[T.Risk of Depression]: β = -0.472, p = 0.0303

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