"""
Linear Mixed Model (LMM) module for channel-wise statistical testing.

This module implements LMM-based testing for each EEG channel independently,
extracting t-statistics for spatial cluster permutation testing.
Updated to work with aggregated probe marker data from the Junifer pipeline.

Notes
-----
Some validation and robustness patterns inspired by MNE-Python's linear_regression
implementation (mne.stats.regression), particularly:
- Design matrix validation
- Degenerate case handling (NaN/Inf in statistics)
- Robust statistical computation

References
----------
MNE-Python: https://mne.tools/stable/generated/mne.stats.regression.linear_regression.html
"""

import numpy as np
import pandas as pd
import warnings
import re
from typing import Tuple, Optional
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def is_interaction_term(predictor: str) -> bool:
    """
    Check if a predictor_of_interest is an interaction term.

    Parameters
    ----------
    predictor : str
        Predictor name, e.g. ``"onoff"`` or ``"onoff:valence"``.

    Returns
    -------
    bool
        True if the predictor contains a ``:`` separator (interaction).
    """
    return ':' in predictor


def get_interaction_components(predictor: str) -> list:
    """
    Extract constituent variable names from an interaction term.

    Parameters
    ----------
    predictor : str
        Interaction term, e.g. ``"onoff:valence"``.

    Returns
    -------
    list of str
        Constituent variables, e.g. ``["onoff", "valence"]``.
        For a non-interaction term, returns a single-element list.
    """
    return [v.strip() for v in predictor.split(':')]


def parse_random_effects(formula: str) -> Tuple[str, Optional[str]]:
    """
    Parse random effects from formula to extract fixed effects and random effects structure.
    
    Parameters
    ----------
    formula : str
        Full LMM formula (e.g., "power ~ onoff + (1 + onoff|subject)")
        
    Returns
    -------
    formula_fixed : str
        Fixed effects only (e.g., "power ~ onoff")
    re_formula : str or None
        Random effects formula for statsmodels (e.g., "1 + onoff")
        None if only random intercept (1|subject)
        
    Examples
    --------
    >>> parse_random_effects("power ~ onoff + (1|subject)")
    ('power ~ onoff', None)
    >>> parse_random_effects("power ~ onoff + (1 + onoff|subject)")
    ('power ~ onoff', '1 + onoff')
    """
    # Extract random effects (anything in parentheses with |)
    random_effects_match = re.search(r'\(([^)]*)\|[^)]*\)', formula)
    
    if not random_effects_match:
        # No random effects found
        return formula.strip(), None
    
    # Get the random effects specification (before the |)
    re_spec = random_effects_match.group(1).strip()
    
    # Remove random effects from formula to get fixed effects only
    formula_fixed = re.sub(r'\s*\+?\s*\([^)]*\|[^)]*\)', '', formula).strip()
    
    # Check if it's just random intercept (1|subject) or has slopes
    if re_spec == '1' or re_spec == '':
        # Random intercept only - use groups parameter (default behavior)
        return formula_fixed, None
    else:
        # Random slopes - need re_formula parameter
        # statsmodels expects format like "1 + onoff" for random intercept + slope
        return formula_fixed, re_spec


def run_lmm_per_channel(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = 'powell',
    maxiter: int = 1000,
    random_state: int = 42,
    return_diagnostics: bool = False
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run linear mixed model for each channel independently.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with 'subject' column for grouping
    formula : str
        R-style formula for the mixed model (e.g., "power ~ onoff + (1|subject)")
    predictor_of_interest : str
        Name of the predictor to extract t-statistic from (e.g., "onoff")
    method : str, default='powell'
        Optimization method ('powell', 'lbfgs', 'bfgs', 'cg', 'ncg', 'nm', etc.)
        Note: This is the optimization algorithm, not the estimation method.
        REML estimation is always used (reml=True in model.fit()).
        'powell' is recommended for LMMs as it's robust and doesn't require gradients.
    maxiter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
    return_diagnostics : bool, default=False
        If True, return detailed convergence diagnostics
        
    Returns
    -------
    t_stats : np.ndarray
        T-statistics for predictor_of_interest, shape (n_channels,)
    p_values : np.ndarray
        P-values for predictor_of_interest, shape (n_channels,)
    diagnostics : dict
        Convergence and model diagnostics (only if return_diagnostics=True)
        Contains:
        - 'n_converged': number of successfully converged models
        - 'n_failed': number of failed models
        - 'n_insufficient_data': number of channels with insufficient data
        - 'convergence_rate': proportion of successful convergences
        - 'failed_channels': list of channel indices that failed
        - 'convergence_warnings': list of convergence warning messages
        - 'aic': list of AIC values per channel
        - 'bic': list of BIC values per channel
        - 'log_likelihood': list of log-likelihood values per channel
        - 'conditional_r2': list of conditional R² values per channel
        - 'shapiro_p_values': list of Shapiro-Wilk p-values (residual normality)
        - 'breusch_pagan_p': list of Breusch-Pagan p-values (homoscedasticity)
        - 'aic_mean', 'bic_mean', 'conditional_r2_mean': summary statistics
        - 'shapiro_p_mean', 'pct_normality_violations': normality check summary
        - 'breusch_pagan_p_mean', 'pct_heteroscedasticity': homoscedasticity summary
        
    Notes
    -----
    - Convergence failures are handled gracefully (t=0, p=1)
    - Groups parameter is automatically set to df_behavioral["subject"]
    - Works with aggregated probe marker data (onoff, valence, confidence, etc.)
    - Handles missing data by dropping NaN values per channel
    - Supports both continuous and binary predictors
    """
    n_channels = power_data.shape[1]
    # NaN initialization: failed/excluded channels are properly masked out of cluster
    # formation in both observed and permuted data (via _find_clusters valid_mask).
    # Using 0 was wrong: a channel that fails in observed but not in permuted (or vice
    # versa) creates an asymmetric null distribution that inflates or deflates p-values.
    t_stats = np.full(n_channels, np.nan)
    p_values = np.full(n_channels, np.nan)

    # Initialize diagnostics tracking
    diagnostics = {
        'n_converged': 0,
        'n_warnings': 0,   # converged but with ConvergenceWarning (kept separate)
        'n_failed': 0,
        'n_insufficient_data': 0,
        'convergence_rate': 0.0,
        'failed_channels': [],
        'convergence_warnings': [],
        # Model quality metrics
        'aic': [],
        'bic': [],
        'log_likelihood': [],
        'conditional_r2': [],
        # Assumption checks
        'shapiro_p_values': [],
        'residual_variance': [],
        'breusch_pagan_p': []
    }
    
    np.random.seed(random_state)
    
    # Track convergence warnings separately
    convergence_warning_count = 0
    
    # Validate design matrix structure (MNE-inspired validation)
    is_valid, validation_msg = validate_design_matrix_structure(
        df_behavioral, formula, predictor_of_interest
    )
    if not is_valid:
        raise ValueError(f"Design matrix validation failed: {validation_msg}")
    
    # Validate data structure before processing
    # For interaction terms (e.g. "onoff:valence"), check constituent columns
    if is_interaction_term(predictor_of_interest):
        for comp in get_interaction_components(predictor_of_interest):
            if comp not in df_behavioral.columns:
                raise ValueError(
                    f"Interaction component '{comp}' (from '{predictor_of_interest}') "
                    f"not found in behavioral data. Available columns: {list(df_behavioral.columns)}"
                )
            if df_behavioral[comp].nunique() < 2:
                raise ValueError(
                    f"Interaction component '{comp}' has insufficient variation "
                    f"(only {df_behavioral[comp].nunique()} unique values)"
                )
    else:
        if predictor_of_interest not in df_behavioral.columns:
            raise ValueError(f"Predictor '{predictor_of_interest}' not found in behavioral data. Available columns: {list(df_behavioral.columns)}")
        
        # Check for sufficient variation in predictor across all data
        if df_behavioral[predictor_of_interest].nunique() < 2:
            raise ValueError(f"Predictor '{predictor_of_interest}' has insufficient variation (only {df_behavioral[predictor_of_interest].nunique()} unique values)")
    
    # Check subjects
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 2:
        raise ValueError(f"Insufficient subjects for mixed effects model: {n_subjects}")
    
    print(f"Processing {n_channels} channels with {n_subjects} subjects")
    
    # Extract all formula variables to ensure we drop NaN for ALL of them
    # This prevents index mismatches when statsmodels internally drops NaN
    from helpers import extract_all_formula_variables
    formula_vars = extract_all_formula_variables(formula)
    
    for ch_idx in range(n_channels):
        try:
            # Prepare data for this channel
            df_ch = df_behavioral.copy()
            df_ch['power'] = power_data[:, ch_idx]
            
            # CRITICAL: Remove NaN values for ALL formula variables, not just predictor_of_interest
            # This prevents index mismatch errors when statsmodels internally drops NaN rows
            dropna_cols = ['power', 'subject'] + formula_vars
            df_ch = df_ch.dropna(subset=dropna_cols)
            
            # Ensure subject remains string type after dropna (prevents mixedlm TypeError)
            df_ch['subject'] = df_ch['subject'].astype(str)
            
            # Check minimum observations threshold
            if len(df_ch) < 10:
                # NaN: excluded from clusters in both observed and permuted
                t_stats[ch_idx] = np.nan
                p_values[ch_idx] = np.nan
                diagnostics['n_insufficient_data'] += 1
                continue

            # Check if we have multiple subjects for this channel
            if df_ch['subject'].nunique() < 2:
                t_stats[ch_idx] = np.nan
                p_values[ch_idx] = np.nan
                diagnostics['n_insufficient_data'] += 1
                continue

            # Check for sufficient variation in predictor for this channel
            # For interaction terms, check all constituent variables
            _insufficient_variation = False
            if is_interaction_term(predictor_of_interest):
                for comp in get_interaction_components(predictor_of_interest):
                    if comp in df_ch.columns and df_ch[comp].nunique() < 2:
                        _insufficient_variation = True
                        break
            else:
                if df_ch[predictor_of_interest].nunique() < 2:
                    _insufficient_variation = True
            if _insufficient_variation:
                t_stats[ch_idx] = np.nan
                p_values[ch_idx] = np.nan
                diagnostics['n_insufficient_data'] += 1
                continue
            
            # Parse random effects to check for random slopes
            formula_fixed, re_formula = parse_random_effects(formula)
            
            # Fit mixed model with random slopes support
            if re_formula is None:
                # Random intercept only: (1|subject)
                model = mixedlm(
                    formula=formula_fixed,
                    data=df_ch,
                    groups=df_ch["subject"]
                )
            else:
                # Random slopes: (1 + predictor|subject)
                # Use re_formula parameter for random slopes
                model = mixedlm(
                    formula=formula_fixed,
                    data=df_ch,
                    groups=df_ch["subject"],
                    re_formula=re_formula
                )
            
            # Ensure method is uppercase (statsmodels expects uppercase)
            method_upper = method.upper()
            
            # Catch convergence warnings during model fitting
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                result = model.fit(
                    method=method_upper,
                    maxiter=maxiter,
                    reml=True,  # Use REML instead of ML for unbiased variance estimates
                    disp=False
                )
                
                # Check if convergence warning was raised
                convergence_warnings = [warning for warning in w if issubclass(warning.category, ConvergenceWarning)]
                has_convergence_warning = len(convergence_warnings) > 0
            
            # Extract t-statistic and p-value for predictor of interest
            if predictor_of_interest in result.tvalues.index:
                # Always extract the values that the model returned
                t_val = float(result.tvalues[predictor_of_interest])
                p_val = float(result.pvalues[predictor_of_interest])
                
                # Handle degenerate cases (inspired by MNE's robust approach)
                # Check for NaN or Inf values that could propagate
                if np.isnan(t_val) or np.isinf(t_val):
                    # Degenerate case: NaN so the channel is excluded from clustering
                    t_stats[ch_idx] = np.nan
                    p_values[ch_idx] = np.nan
                    diagnostics['n_failed'] += 1
                    diagnostics['failed_channels'].append(ch_idx)
                    if return_diagnostics:
                        diagnostics['convergence_warnings'].append(
                            f"Channel {ch_idx}: Degenerate t-statistic (NaN/Inf)"
                        )
                    continue
                
                t_stats[ch_idx] = t_val
                p_values[ch_idx] = p_val
                
                # Track convergence status
                if has_convergence_warning:
                    # Model ran but did NOT cleanly converge — kept separate from n_converged
                    # so the convergence_rate reflects only clean fits.
                    warning_msgs = [str(warning.message) for warning in convergence_warnings]
                    convergence_warning_count += 1
                    diagnostics['n_warnings'] += 1
                    if return_diagnostics:
                        diagnostics['convergence_warnings'].append(f"Channel {ch_idx}: {'; '.join(warning_msgs)}")
                else:
                    # Model converged successfully without warnings
                    diagnostics['n_converged'] += 1
                
                # Collect model quality metrics
                # Note: AIC/BIC may be NaN for REML estimation (only valid for ML)
                aic_val = float(result.aic) if hasattr(result, 'aic') else np.nan
                bic_val = float(result.bic) if hasattr(result, 'bic') else np.nan
                llf_val = float(result.llf) if hasattr(result, 'llf') else np.nan
                
                diagnostics['aic'].append(aic_val)
                diagnostics['bic'].append(bic_val)
                diagnostics['log_likelihood'].append(llf_val)
                
                # Compute conditional R² (variance explained by fixed + random effects)
                # R² = 1 - (residual variance / total variance)
                residuals = result.resid
                total_var = np.var(df_ch['power'])
                residual_var = np.var(residuals)
                cond_r2 = 1 - (residual_var / total_var) if total_var > 0 else 0.0
                diagnostics['conditional_r2'].append(float(cond_r2))
                diagnostics['residual_variance'].append(float(residual_var))
                
                # Check residual normality (Shapiro-Wilk test)
                # Only test if we have enough residuals (3 <= n <= 5000)
                if 3 <= len(residuals) <= 5000:
                    _, shapiro_p = stats.shapiro(residuals)
                    diagnostics['shapiro_p_values'].append(float(shapiro_p))
                else:
                    diagnostics['shapiro_p_values'].append(np.nan)

                # Check homoscedasticity (Breusch-Pagan test approximation)
                # Test if residual variance is related to fitted values
                fitted = result.fittedvalues
                if len(fitted) > 10:
                    abs_resid = np.abs(residuals)
                    _, bp_p = stats.spearmanr(fitted, abs_resid)
                    diagnostics['breusch_pagan_p'].append(float(bp_p))
                else:
                    diagnostics['breusch_pagan_p'].append(np.nan)
            else:
                # Predictor not found in model output (e.g. perfect collinearity dropped it)
                t_stats[ch_idx] = np.nan
                p_values[ch_idx] = np.nan
                diagnostics['n_failed'] += 1
                diagnostics['failed_channels'].append(ch_idx)

        except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
            # Specific numerical/model failures: singular matrix, degenerate data,
            # or statsmodels internal errors during REML optimisation.
            # NaN ensures this channel is excluded from cluster formation in both
            # observed and permuted data, keeping the null distribution symmetric.
            t_stats[ch_idx] = np.nan
            p_values[ch_idx] = np.nan
            diagnostics['n_failed'] += 1
            diagnostics['failed_channels'].append(ch_idx)
            error_msg = f"Channel {ch_idx} LMM failed ({type(e).__name__}): {e}"
            diagnostics['convergence_warnings'].append(error_msg)
            print(f"  WARNING: {error_msg}")

    # Abort early if failure rate is too high — results would be unreliable.
    n_attempted = n_channels - diagnostics['n_insufficient_data']
    if n_attempted > 0 and diagnostics['n_failed'] / n_attempted > 0.5:
        raise RuntimeError(
            f"LMM failed on {diagnostics['n_failed']}/{n_attempted} channels "
            f"({100 * diagnostics['n_failed'] / n_attempted:.1f}%). "
            "Investigate convergence issues before proceeding. "
            f"First failures: {diagnostics['failed_channels'][:5]}"
        )

    # Calculate convergence rate (only clean fits, warnings counted separately)
    n_attempted = n_channels - diagnostics['n_insufficient_data']
    if n_attempted > 0:
        diagnostics['convergence_rate'] = diagnostics['n_converged'] / n_attempted
        diagnostics['warning_rate'] = diagnostics['n_warnings'] / n_attempted
    
    # Compute summary statistics for model quality and assumptions
    if diagnostics['n_converged'] > 0:
        # Only compute means if we actually collected diagnostic values
        # Suppress RuntimeWarning for empty slices (happens during permutation testing)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
            if len(diagnostics['aic']) > 0:
                diagnostics['aic_mean'] = float(np.nanmean(diagnostics['aic']))
            if len(diagnostics['bic']) > 0:
                diagnostics['bic_mean'] = float(np.nanmean(diagnostics['bic']))
            if len(diagnostics['log_likelihood']) > 0:
                diagnostics['log_likelihood_mean'] = float(np.nanmean(diagnostics['log_likelihood']))
            if len(diagnostics['conditional_r2']) > 0:
                diagnostics['conditional_r2_mean'] = float(np.nanmean(diagnostics['conditional_r2']))
                diagnostics['conditional_r2_median'] = float(np.nanmedian(diagnostics['conditional_r2']))
        
        # Assumption checks summary
        shapiro_valid = [p for p in diagnostics['shapiro_p_values'] if not np.isnan(p)]
        if shapiro_valid:
            diagnostics['shapiro_p_mean'] = float(np.mean(shapiro_valid))
            diagnostics['n_normality_violations'] = sum(1 for p in shapiro_valid if p < 0.05)
            diagnostics['pct_normality_violations'] = 100 * diagnostics['n_normality_violations'] / len(shapiro_valid)
        
        bp_valid = [p for p in diagnostics['breusch_pagan_p'] if not np.isnan(p)]
        if bp_valid:
            diagnostics['breusch_pagan_p_mean'] = float(np.mean(bp_valid))
            diagnostics['n_heteroscedasticity'] = sum(1 for p in bp_valid if p < 0.05)
            diagnostics['pct_heteroscedasticity'] = 100 * diagnostics['n_heteroscedasticity'] / len(bp_valid)
    
    # Print summary if diagnostics requested
    if return_diagnostics:
        print("\nLMM Convergence Summary:")
        print(f"  Total channels: {n_channels}")
        print(f"  Converged (clean): {diagnostics['n_converged']}")
        print(f"  Converged with warnings: {diagnostics['n_warnings']}")
        print(f"  Failed (NaN t-stat, excluded from clusters): {diagnostics['n_failed']}")
        print(f"  Insufficient data (excluded): {diagnostics['n_insufficient_data']}")
        print(f"  Clean convergence rate: {100*diagnostics['convergence_rate']:.1f}%")
        if diagnostics['n_warnings'] > 0:
            warning_rate = diagnostics.get('warning_rate', 0.0)
            print(f"  Warning rate: {100*warning_rate:.1f}%  "
                  f"(t-stats kept but reliability uncertain)")
        if len(diagnostics['convergence_warnings']) > 0:
            print(f"  Logged warnings/errors: {len(diagnostics['convergence_warnings'])}")
        
        # Print model quality metrics
        if diagnostics['n_converged'] > 0:
            print(f"\nModel Quality Metrics (averaged across {diagnostics['n_converged']} channels):")
            
            # Check if AIC/BIC are available and valid
            aic_mean = diagnostics.get('aic_mean', np.nan)
            if np.isnan(aic_mean):
                print("  AIC (mean): N/A (not available with REML estimation)")
            else:
                print(f"  AIC (mean): {aic_mean:.2f}")
            
            bic_mean = diagnostics.get('bic_mean', np.nan)
            if np.isnan(bic_mean):
                print("  BIC (mean): N/A (not available with REML estimation)")
            else:
                print(f"  BIC (mean): {bic_mean:.2f}")
            
            llf_mean = diagnostics.get('log_likelihood_mean', np.nan)
            if np.isnan(llf_mean):
                print("  Log-likelihood (mean): N/A")
            else:
                print(f"  Log-likelihood (mean): {llf_mean:.2f}")
            
            # Conditional R² should always be available if models converged
            if 'conditional_r2_mean' in diagnostics:
                print(f"  Conditional R² (mean): {diagnostics['conditional_r2_mean']:.3f}")
                print(f"  Conditional R² (median): {diagnostics['conditional_r2_median']:.3f}")
            
            # Print assumption checks
            print("\nModel Assumptions:")
            if 'shapiro_p_mean' in diagnostics:
                print("  Residual normality (Shapiro-Wilk):")
                print(f"    Mean p-value: {diagnostics['shapiro_p_mean']:.3f}")
                print(f"    Violations (p < 0.05): {diagnostics['n_normality_violations']}/{len(shapiro_valid)} ({diagnostics['pct_normality_violations']:.1f}%)")
            
            if 'breusch_pagan_p_mean' in diagnostics:
                print("  Homoscedasticity (Breusch-Pagan approximation):")
                print(f"    Mean p-value: {diagnostics['breusch_pagan_p_mean']:.3f}")
                print(f"    Violations (p < 0.05): {diagnostics['n_heteroscedasticity']}/{len(bp_valid)} ({diagnostics['pct_heteroscedasticity']:.1f}%)")
    
    return t_stats, p_values, diagnostics


def validate_design_matrix_structure(
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str
) -> Tuple[bool, str]:
    """
    Validate design matrix structure before fitting LMM (inspired by MNE's validation).
    
    Parameters
    ----------
    df_behavioral : pd.DataFrame
        Behavioral data
    formula : str
        LMM formula
    predictor_of_interest : str
        Main predictor to test
        
    Returns
    -------
    is_valid : bool
        True if design matrix is valid
    message : str
        Validation message (empty if valid, error description if invalid)
        
    Notes
    -----
    Checks inspired by MNE's _fit_lm function:
    - Sufficient observations
    - No perfect collinearity
    - Sufficient variance in predictors
    """
    n_obs = len(df_behavioral)
    
    # Check 1: Sufficient observations (MNE-inspired)
    if n_obs < 10:
        return False, f"Insufficient observations: {n_obs} < 10"
    
    # Check 2: Predictor exists and has variance
    # For interaction terms (e.g. "onoff:valence"), validate each constituent variable
    if is_interaction_term(predictor_of_interest):
        components = get_interaction_components(predictor_of_interest)
        for comp in components:
            if comp not in df_behavioral.columns:
                return False, f"Interaction component '{comp}' not in data"
            comp_vals = df_behavioral[comp].dropna()
            if len(comp_vals) < 5:
                return False, f"Too few non-NaN values in '{comp}': {len(comp_vals)}"
            if comp_vals.nunique() < 2:
                return False, f"Component '{comp}' has no variance: {comp_vals.nunique()} unique values"
            if np.std(comp_vals) < 1e-10:
                return False, f"Component '{comp}' variance too small: std={np.std(comp_vals)}"
    else:
        if predictor_of_interest not in df_behavioral.columns:
            return False, f"Predictor '{predictor_of_interest}' not in data"
        
        predictor_vals = df_behavioral[predictor_of_interest].dropna()
        if len(predictor_vals) < 5:
            return False, f"Too few non-NaN values in predictor: {len(predictor_vals)}"
        
        if predictor_vals.nunique() < 2:
            return False, f"Predictor has no variance: {predictor_vals.nunique()} unique values"
        
        # Check 3: Sufficient variance (avoid numerical issues)
        if np.std(predictor_vals) < 1e-10:
            return False, f"Predictor variance too small: std={np.std(predictor_vals)}"
    
    return True, ""


def validate_probe_data_structure(df_behavioral: pd.DataFrame, formula: str) -> bool:
    """
    Validate that the behavioral data has the expected structure for probe marker analysis.
    
    Parameters
    ----------
    df_behavioral : pd.DataFrame
        Behavioral data from aggregated probe markers
    formula : str
        LMM formula string
        
    Returns
    -------
    bool
        True if data structure is valid, False otherwise
    """
    import re
    
    # Check required columns
    required_cols = ['subject']
    missing_cols = [col for col in required_cols if col not in df_behavioral.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        return False
    
    # Extract variables from formula
    variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
    variables = [v for v in variables if v not in ['power', 'subject']]
    
    # Check if formula variables exist in data
    missing_vars = [v for v in variables if v not in df_behavioral.columns]
    if missing_vars:
        print(f"Formula variables not found in data: {missing_vars}")
        return False
    
    # Check if we have multiple subjects
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 2:
        print(f"Need at least 2 subjects for mixed effects model, found {n_subjects}")
        return False
    
    # Check for sufficient observations
    if len(df_behavioral) < 20:
        print(f"Need at least 20 observations, found {len(df_behavioral)}")
        return False
    
    # Check for common probe marker columns
    probe_cols = ['onoff', 'valence', 'confidence', 'time']
    available_probe_cols = [col for col in probe_cols if col in df_behavioral.columns]
    if available_probe_cols:
        print(f"Available probe marker columns: {available_probe_cols}")
    
    return True


def compute_effect_sizes(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    predictor: str
) -> np.ndarray:
    """
    Compute Cohen's d effect size for each channel.
    
    This function computes effect sizes for probe marker data, handling both
    binary predictors and continuous predictors (like onoff: mind-wandering levels).
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data from aggregated probe markers
    predictor : str
        Name of the predictor variable (e.g., 'onoff', 'valence')
        
    Returns
    -------
    effect_sizes : np.ndarray
        Effect sizes for each channel, shape (n_channels,)
        For binary predictors: Cohen's d
        For continuous predictors: correlation coefficient
    """
    n_channels = power_data.shape[1]
    effect_sizes = np.zeros(n_channels)
    
    if predictor not in df_behavioral.columns:
        warnings.warn(f"Predictor {predictor} not found in data")
        return effect_sizes
    
    unique_vals = df_behavioral[predictor].unique()
    
    # Handle binary predictors (e.g., task conditions)
    if len(unique_vals) == 2:
        mask_group1 = df_behavioral[predictor] == unique_vals[0]
        mask_group2 = df_behavioral[predictor] == unique_vals[1]
        
        for ch_idx in range(n_channels):
            group1 = power_data[mask_group1, ch_idx]
            group2 = power_data[mask_group2, ch_idx]
            
            # Remove NaN values
            group1 = group1[~np.isnan(group1)]
            group2 = group2[~np.isnan(group2)]
            
            if len(group1) < 2 or len(group2) < 2:
                continue
            
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
            n1, n2 = len(group1), len(group2)
            
            # Pooled standard deviation for Cohen's d
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std > 0:
                effect_sizes[ch_idx] = (mean1 - mean2) / pooled_std
    
    # Handle continuous predictors (e.g., onoff, valence, confidence)
    else:
        predictor_vals = df_behavioral[predictor].values
        
        for ch_idx in range(n_channels):
            channel_vals = power_data[:, ch_idx]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(channel_vals) | np.isnan(predictor_vals))
            if np.sum(valid_mask) < 5:  # Need at least 5 valid observations
                continue
            
            channel_clean = channel_vals[valid_mask]
            predictor_clean = predictor_vals[valid_mask]
            
            # Compute correlation coefficient
            if np.std(channel_clean) > 0 and np.std(predictor_clean) > 0:
                correlation = np.corrcoef(channel_clean, predictor_clean)[0, 1]
                effect_sizes[ch_idx] = correlation if not np.isnan(correlation) else 0.0
    
    return effect_sizes


def fit_reduced_model_per_channel(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = 'powell',
    maxiter: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit reduced LMM (without predictor of interest) and extract residuals and fitted values.
    
    This is the first step of the Freedman-Lane permutation procedure.
    The reduced model includes all covariates EXCEPT the predictor of interest.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with 'subject' column
    formula : str
        R-style formula for the full model (e.g., "power ~ X_POI + covariate + (1|subject)")
    predictor_of_interest : str
        Name of the predictor to exclude from reduced model (e.g., "X_POI")
    method : str, default='powell'
        Optimization method ('powell', 'lbfgs', 'bfgs', etc.)
        Note: This is the optimization algorithm. REML estimation is always used.
    maxiter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    residuals : np.ndarray
        Residuals from reduced model, shape (n_observations, n_channels)
        R = Y - Ŷ_reduced
    fitted_values : np.ndarray
        Fitted values from reduced model, shape (n_observations, n_channels)
        Ŷ_reduced = predicted values from model without X_POI
        
    Notes
    -----
    The reduced model formula is created by removing the predictor_of_interest
    from the full formula. For example:
    - Full: "power ~ onoff + valence + (1|subject)"
    - Reduced: "power ~ valence + (1|subject)" (onoff removed)
    
    If the reduced model has no fixed effects (only intercept + random effects),
    it becomes: "power ~ 1 + (1|subject)"
    """
    n_observations, n_channels = power_data.shape
    # NaN initialization: rows that were not fitted (NaN power or behavioral data)
    # remain NaN so freedman_lane_permutation can identify and skip them, preserving
    # the same observation set as the full model.
    residuals = np.full((n_observations, n_channels), np.nan)
    fitted_values = np.full((n_observations, n_channels), np.nan)

    np.random.seed(random_state)

    # Create reduced formula by removing predictor of interest
    reduced_formula = _create_reduced_formula(formula, predictor_of_interest)

    # Parse random effects once (same for all channels)
    formula_fixed, re_formula = parse_random_effects(reduced_formula)

    # IMPORTANT: Freedman-Lane with random slopes
    # The reduced model must have the SAME random effects structure as the full model
    # to preserve the correlation structure under the null hypothesis
    if re_formula is not None:
        warnings.warn(
            "Using Freedman-Lane permutation with random slopes. "
            "Ensure the reduced model preserves the random effects structure. "
            "Consider using 'simple' permutation method if convergence issues occur.",
            UserWarning
        )

    # Use a context manager so the ConvergenceWarning filter is scoped and does not
    # corrupt the global warnings state for the rest of the pipeline.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)

        for ch_idx in range(n_channels):
            try:
                # Prepare data for this channel
                df_ch = df_behavioral.copy()
                df_ch['power'] = power_data[:, ch_idx]

                # Remove NaN values (same logic as run_lmm_per_channel)
                df_ch = df_ch.dropna(subset=['power', 'subject'])
                df_ch['subject'] = df_ch['subject'].astype(str)

                # Check minimum observations
                if len(df_ch) < 10 or df_ch['subject'].nunique() < 2:
                    # Leave residuals/fitted_values as NaN for this channel.
                    # freedman_lane_permutation will propagate NaN → power_permuted NaN
                    # → channel excluded in permuted LMM, consistent with observed.
                    continue

                # Fit reduced model with same random effects structure and REML
                if re_formula is None:
                    model = mixedlm(
                        formula=formula_fixed,
                        data=df_ch,
                        groups=df_ch["subject"]
                    )
                else:
                    model = mixedlm(
                        formula=formula_fixed,
                        data=df_ch,
                        groups=df_ch["subject"],
                        re_formula=re_formula
                    )

                result = model.fit(
                    method=method.upper(),
                    maxiter=maxiter,
                    reml=True,   # explicit: must match full-model estimation
                    disp=False
                )

                # Map fitted values and residuals back to original row positions.
                # Rows not in df_ch.index (NaN-dropped) stay NaN, which correctly
                # excludes them from permutation in freedman_lane_permutation.
                fitted_ch = np.full(n_observations, np.nan)
                residuals_ch = np.full(n_observations, np.nan)

                fitted_ch[df_ch.index] = result.fittedvalues.values
                residuals_ch[df_ch.index] = result.resid.values

                fitted_values[:, ch_idx] = fitted_ch
                residuals[:, ch_idx] = residuals_ch

            except (np.linalg.LinAlgError, ValueError, RuntimeError) as e:
                # Specific numerical/model failures in the reduced model.
                # Keep residuals/fitted_values as NaN: this channel will be NaN in
                # power_permuted and excluded in the permuted LMM, consistent
                # with the observed LMM (which will also fail for the same reason).
                print(f"  WARNING: reduced model failed at ch {ch_idx} "
                      f"({type(e).__name__}): {e}")

    return residuals, fitted_values


def _create_reduced_formula(formula: str, predictor_to_remove: str) -> str:
    """
    Create reduced formula by removing a specific predictor.

    Handles both simple predictors and interaction terms:

    - Simple: ``"power ~ onoff + valence + (1|subject)"`` minus ``"onoff"``
      → ``"power ~ valence + (1|subject)"``
    - Interaction from ``*``: ``"power ~ onoff * valence + (1|subject)"``
      minus ``"onoff:valence"``
      → ``"power ~ onoff + valence + (1|subject)"`` (main effects kept)

    Parameters
    ----------
    formula : str
        Full formula (e.g., ``"power ~ onoff * valence + (1|subject)"``)
    predictor_to_remove : str
        Predictor to remove (e.g., ``"onoff"``, ``"onoff:valence"``)

    Returns
    -------
    str
        Reduced formula with the predictor removed.

    Examples
    --------
    >>> _create_reduced_formula("power ~ onoff + valence + (1|subject)", "onoff")
    'power ~ valence + (1|subject)'
    >>> _create_reduced_formula("power ~ onoff + (1|subject)", "onoff")
    'power ~ 1 + (1|subject)'
    >>> _create_reduced_formula("power ~ onoff * valence + (1|subject)", "onoff:valence")
    'power ~ onoff + valence + (1|subject)'
    """
    # Split formula into left and right sides
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")

    left_side = parts[0].strip()
    right_side = parts[1].strip()

    # Extract random effects part (e.g., "(1|subject)")
    random_effects = re.findall(r'\([^)]*\|[^)]*\)', right_side)

    # Remove random effects from right side to get fixed effects
    fixed_effects = re.sub(r'\s*\+?\s*\([^)]*\|[^)]*\)', '', right_side).strip()

    # Expand "*" (interaction shorthand) into explicit terms
    # e.g. "onoff * valence" → "onoff + valence + onoff:valence"
    expanded_terms = []
    for raw_term in fixed_effects.split('+'):
        raw_term = raw_term.strip()
        if not raw_term:
            continue
        if '*' in raw_term:
            vars_in_star = [v.strip() for v in raw_term.split('*')]
            # Add main effects
            for v in vars_in_star:
                if v and v not in expanded_terms:
                    expanded_terms.append(v)
            # Add interaction term (two-way; extends naturally for higher-order)
            interaction = ':'.join(vars_in_star)
            if interaction not in expanded_terms:
                expanded_terms.append(interaction)
        else:
            if raw_term not in expanded_terms:
                expanded_terms.append(raw_term)

    # Remove the predictor of interest (exact match on expanded terms)
    reduced_terms = [t for t in expanded_terms if t != predictor_to_remove and t != '']

    # Reconstruct formula
    if len(reduced_terms) == 0:
        reduced_fixed = '1'
    else:
        reduced_fixed = ' + '.join(reduced_terms)

    if random_effects:
        reduced_formula = f"{left_side} ~ {reduced_fixed} + {' + '.join(random_effects)}"
    else:
        reduced_formula = f"{left_side} ~ {reduced_fixed}"

    return reduced_formula


def freedman_lane_permutation(
    residuals: np.ndarray,
    fitted_values: np.ndarray,
    df_behavioral: pd.DataFrame,
    seed: int
) -> np.ndarray:
    """
    Perform Freedman-Lane permutation on residuals.
    
    This function implements step 2 of the Freedman-Lane procedure:
    permute residuals within subjects and reconstruct permuted Y.
    
    Parameters
    ----------
    residuals : np.ndarray
        Residuals from reduced model, shape (n_observations, n_channels)
    fitted_values : np.ndarray
        Fitted values from reduced model, shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with 'subject' column
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    power_permuted : np.ndarray
        Permuted power data, shape (n_observations, n_channels)
        Y* = Ŷ_reduced + R_permuted
        
    Notes
    -----
    The Freedman-Lane procedure permutes residuals within subjects to:
    1. Break the association between X_POI and Y
    2. Preserve the structure of covariates
    3. Maintain within-subject correlation
    
    This is more powerful than simple permutation because it:
    - Reduces residual variance by accounting for covariates
    - Preserves important confound structure
    - Provides better Type I error control with multiple predictors
    """
    n_observations, n_channels = residuals.shape
    
    # Validate dimensions match
    if fitted_values.shape != residuals.shape:
        raise ValueError(
            f"Shape mismatch: residuals {residuals.shape} != "
            f"fitted_values {fitted_values.shape}"
        )
    
    if len(df_behavioral) != n_observations:
        raise ValueError(
            f"Shape mismatch: df_behavioral has {len(df_behavioral)} rows "
            f"but residuals has {n_observations} observations"
        )
    
    # NaN initialization: rows that had no valid reduced model fit (NaN residuals)
    # stay NaN in power_permuted and are consequently excluded by dropna in the
    # permuted full LMM, matching exactly what happens in the observed LMM.
    power_permuted = np.full((n_observations, n_channels), np.nan)

    np.random.seed(seed)

    # Guard: if ALL channels have all-NaN residuals, the reduced model failed entirely.
    if np.all(np.isnan(residuals)):
        return power_permuted

    # Permute residuals within each subject using only valid rows
    for subject in df_behavioral['subject'].unique():
        subject_mask = df_behavioral['subject'] == subject
        subject_indices = df_behavioral.loc[subject_mask].index.to_numpy()

        # Valid rows: rows where at least one channel has a non-NaN residual.
        # Using the UNION (any channel) rather than a single reference channel
        # ensures rows that are valid for some channels but not others (e.g. when
        # the reduced model failed only for specific channels) are still included
        # in the permutation.  Channels with NaN residuals at those rows will
        # produce NaN in power_permuted automatically (NaN + anything = NaN).
        any_valid_mask = ~np.all(np.isnan(residuals[subject_indices, :]), axis=1)
        valid_indices = subject_indices[any_valid_mask]
        # Rows not fitted for ANY channel (behavioral NaN etc.) stay NaN.

        if len(valid_indices) < 2:
            # Single valid observation — cannot permute, reconstruct in place
            if len(valid_indices) == 1:
                power_permuted[valid_indices, :] = (
                    fitted_values[valid_indices, :] + residuals[valid_indices, :]
                )
            continue

        # One permutation order for ALL channels — preserves spatial correlation
        permuted_valid_indices = np.random.permutation(valid_indices)

        # Reconstruct Y* = Ŷ_reduced + permuted residuals
        # Channels with NaN residuals at a given row: NaN + NaN = NaN → excluded
        for ch_idx in range(n_channels):
            power_permuted[valid_indices, ch_idx] = (
                fitted_values[valid_indices, ch_idx]
                + residuals[permuted_valid_indices, ch_idx]
            )

    return power_permuted
