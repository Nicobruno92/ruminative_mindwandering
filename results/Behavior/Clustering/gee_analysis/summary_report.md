# Multinomial Analysis of Cluster Levels - Summary Report

**Analysis Date:** 2025-06-26 15:06:13
**Sample:** 510 observations from 29 subjects

## Cluster Distribution
- **Low:** 174 (34.1%)
- **Self-Centered:** 168 (32.9%)
- **Positive-Future-Oriented:** 168 (32.9%)

## Group Distribution
- **Controls:** 316 (62.0%)
- **Risk of Depression:** 194 (38.0%)

## Inclusion/Exclusion Distribution
- **Inclusion:** 237 (46.5%)
- **Exclusion:** 273 (53.5%)

## Models Tested
1. **Group Effect:** Cluster ~ Group (Controls vs Risk of Depression)
2. **Inclusion/Exclusion Effect:** Cluster ~ Inclusion/Exclusion
3. **Interaction Effect:** Cluster ~ Group × Inclusion/Exclusion

## Statistical Method
- **Model:** Multinomial Logistic Regression with cluster-robust standard errors
- **Reference Category:** Low cluster (most common)
- **Clustering:** By subject ID to account for repeated measures
- **Multiple Comparisons:** Bonferroni correction applied

## Key Findings

### Group Effect
- **Total effects tested:** 2
- **Significant (p < 0.05):** 0
- **Significant (Bonferroni corrected):** 0
- **No significant effects after correction**

### Inclusion/Exclusion Effect
- **Total effects tested:** 2
- **Significant (p < 0.05):** 0
- **Significant (Bonferroni corrected):** 0
- **No significant effects after correction**

### Group × Inclusion/Exclusion Interaction
- **Total effects tested:** 6
- **Significant (p < 0.05):** 0
- **Significant (Bonferroni corrected):** 0
- **No significant effects after correction**

## Interpretation
- **Low cluster:** Baseline/reference category representing low mind-wandering
- **Self-Centered cluster:** Mind-wandering focused on self-related thoughts
- **Positive-Future-Oriented cluster:** Mind-wandering with positive, future-focused content

Significant effects indicate that group membership and/or inclusion/exclusion conditions
are associated with different patterns of mind-wandering cluster membership.

## Files Generated
- `*_results.csv` - Complete model results
- `*_significant_results.csv` - Uncorrected significant effects
- `*_significant_corrected.csv` - Bonferroni-corrected significant effects
- `*_summary.txt` - Detailed model output
- `summary_report.md` - This comprehensive report