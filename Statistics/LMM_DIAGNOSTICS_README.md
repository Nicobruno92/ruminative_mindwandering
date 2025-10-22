# LMM Model Quality and Assumptions Diagnostics

## Overview

The LMM pipeline now includes comprehensive model quality metrics and assumption checks that are automatically computed, printed, and saved for each analysis.

## What's New

### Model Quality Metrics (per channel)
1. **AIC (Akaike Information Criterion)**: Lower values indicate better model fit
2. **BIC (Bayesian Information Criterion)**: Lower values indicate better model fit (penalizes complexity more than AIC)
3. **Log-likelihood**: Higher values indicate better fit
4. **Conditional R²**: Proportion of variance explained by fixed + random effects (0-1 scale)

### Model Assumption Checks (per channel)
1. **Residual Normality (Shapiro-Wilk test)**:
   - Tests if model residuals are normally distributed
   - p-value < 0.05 indicates violation of normality assumption
   - Reports: mean p-value and % of channels with violations

2. **Homoscedasticity (Breusch-Pagan approximation)**:
   - Tests if residual variance is constant (not related to fitted values)
   - p-value < 0.05 indicates heteroscedasticity (violation)
   - Reports: mean p-value and % of channels with violations

## Output Files

For each marker analysis, the following files are now saved:

### 1. `model_quality.csv`
Per-channel model quality metrics:
- `channel`: Channel name
- `aic`, `bic`, `log_likelihood`: Model fit metrics
- `conditional_r2`: Variance explained
- `shapiro_p_value`: Normality test p-value
- `breusch_pagan_p`: Homoscedasticity test p-value
- `residual_variance`: Residual variance

### 2. `lmm_diagnostics_summary.csv`
Summary statistics across all channels:
- Convergence information (n_converged, convergence_rate)
- Mean model quality metrics (aic_mean, bic_mean, conditional_r2_mean, etc.)
- Assumption violations summary (pct_normality_violations, pct_heteroscedasticity)

### 3. `pipeline_summary.csv` (enhanced)
Now includes model quality columns:
- `lmm_converged`: Number of converged models
- `lmm_convergence_rate`: Convergence rate (0-1)
- `aic_mean`, `bic_mean`: Average information criteria
- `conditional_r2_mean`, `conditional_r2_median`: Average variance explained
- `shapiro_p_mean`, `pct_normality_violations`: Normality check summary
- `pct_heteroscedasticity`: Homoscedasticity violations

## Console Output

When running the pipeline, you'll now see:

```
LMM Convergence Summary:
  Total channels: 64
  Converged: 64
  Failed: 0
  Insufficient data: 0
  Convergence rate: 100.0%

Model Quality Metrics (averaged across 64 channels):
  AIC (mean): 1234.56
  BIC (mean): 1245.67
  Log-likelihood (mean): -612.34
  Conditional R² (mean): 0.234
  Conditional R² (median): 0.221

Model Assumptions:
  Residual normality (Shapiro-Wilk):
    Mean p-value: 0.342
    Violations (p < 0.05): 8/64 (12.5%)
  Homoscedasticity (Breusch-Pagan approximation):
    Mean p-value: 0.456
    Violations (p < 0.05): 5/64 (7.8%)
```

## Interpreting Results

### Good Model Quality
- **Conditional R² > 0.1**: Model explains meaningful variance
- **Low % violations**: < 20% of channels violate assumptions
- **High convergence rate**: > 95% of models converged

### Warning Signs
- **Conditional R² < 0.05**: Model explains very little variance
- **High % normality violations**: > 30% suggests non-normal residuals
- **High % heteroscedasticity**: > 30% suggests unequal variance
- **Low convergence rate**: < 90% suggests model specification issues

### What to Do if Assumptions are Violated
1. **Normality violations**: LMMs are robust to moderate violations with large samples
2. **Heteroscedasticity**: Consider robust standard errors or data transformation
3. **Low R²**: May indicate weak effect or need for additional predictors
4. **Poor convergence**: Check for multicollinearity, scaling issues, or model complexity

## Technical Notes

- **Conditional R²** is computed as: 1 - (residual variance / total variance)
- **Shapiro-Wilk test** is only computed for channels with 3-5000 observations
- **Breusch-Pagan approximation** uses Spearman correlation between |residuals| and fitted values
- All metrics are stored in the `lmm_diagnostics` dictionary in the results pickle file

## Example Usage

Results are automatically computed when `return_diagnostics=True` in `run_lmm_per_channel()`.
The pipeline automatically sets this to `True`, so no code changes are needed.

To access diagnostics from saved results:
```python
import pickle
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
    
diagnostics = results['lmm_diagnostics']
print(f"Mean R²: {diagnostics['conditional_r2_mean']:.3f}")
print(f"Normality violations: {diagnostics['pct_normality_violations']:.1f}%")
```
