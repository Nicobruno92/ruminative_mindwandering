"""
Linear Mixed Model (LMM) module for channel-wise statistical testing.

This module implements LMM-based testing for each EEG channel independently,
extracting t-statistics for spatial cluster permutation testing.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple
from statsmodels.formula.api import mixedlm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def run_lmm_per_channel(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    method: str = 'lbfgs',
    maxiter: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run linear mixed model for each channel independently.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with 'subject' column for grouping
    formula : str
        R-style formula for the mixed model (e.g., "power ~ 1 + onoff + (1|subject)")
    predictor_of_interest : str
        Name of the predictor to extract t-statistic from
    method : str
        Optimization method ('lbfgs' recommended for reproducibility)
    maxiter : int
        Maximum number of iterations
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    t_stats : np.ndarray
        T-statistics for predictor_of_interest, shape (n_channels,)
    p_values : np.ndarray
        P-values for predictor_of_interest, shape (n_channels,)
        
    Notes
    -----
    - Convergence failures are handled silently (t=0, p=1)
    - Groups parameter is automatically set to df_behavioral["subject"]
    - Uses 'lbfgs' method for deterministic optimization
    """
    n_channels = power_data.shape[1]
    t_stats = np.zeros(n_channels)
    p_values = np.ones(n_channels)
    
    np.random.seed(random_state)
    
    # Suppress convergence warnings
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    for ch_idx in range(n_channels):
        try:
            # Prepare data for this channel
            df_ch = df_behavioral.copy()
            df_ch['power'] = power_data[:, ch_idx]
            
            # Remove NaN values
            df_ch = df_ch.dropna()
            
            if len(df_ch) < 10:  # Minimum observations threshold
                continue
            
            # Fit mixed model
            model = mixedlm(
                formula=formula,
                data=df_ch,
                groups=df_ch["subject"]
            )
            
            result = model.fit(
                method=method,
                maxiter=maxiter,
                disp=False
            )
            
            # Extract t-statistic and p-value for predictor of interest
            if predictor_of_interest in result.tvalues.index:
                t_stats[ch_idx] = result.tvalues[predictor_of_interest]
                p_values[ch_idx] = result.pvalues[predictor_of_interest]
            
        except Exception:
            # Convergence failure or other error: set t=0, p=1
            t_stats[ch_idx] = 0.0
            p_values[ch_idx] = 1.0
    
    warnings.filterwarnings('default', category=ConvergenceWarning)
    
    return t_stats, p_values


def compute_effect_sizes(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    predictor: str
) -> np.ndarray:
    """
    Compute Cohen's d effect size for each channel.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data containing the predictor variable
    predictor : str
        Name of the binary predictor variable
        
    Returns
    -------
    effect_sizes : np.ndarray
        Cohen's d for each channel, shape (n_channels,)
    """
    n_channels = power_data.shape[1]
    effect_sizes = np.zeros(n_channels)
    
    # Ensure predictor is binary
    unique_vals = df_behavioral[predictor].unique()
    if len(unique_vals) != 2:
        warnings.warn(f"Predictor {predictor} is not binary, effect size may not be meaningful")
        return effect_sizes
    
    mask_group1 = df_behavioral[predictor] == unique_vals[0]
    mask_group2 = df_behavioral[predictor] == unique_vals[1]
    
    for ch_idx in range(n_channels):
        group1 = power_data[mask_group1, ch_idx]
        group2 = power_data[mask_group2, ch_idx]
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std > 0:
            effect_sizes[ch_idx] = (mean1 - mean2) / pooled_std
    
    return effect_sizes
