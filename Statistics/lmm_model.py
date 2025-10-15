"""
Linear Mixed Model (LMM) module for channel-wise statistical testing.

This module implements LMM-based testing for each EEG channel independently,
extracting t-statistics for spatial cluster permutation testing.
Updated to work with aggregated probe marker data from the Junifer pipeline.
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
    method: str = 'REML',
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
        R-style formula for the mixed model (e.g., "power ~ onoff + (1|subject)")
    predictor_of_interest : str
        Name of the predictor to extract t-statistic from (e.g., "onoff")
    method : str
        Optimization method ('REML' recommended for mixed models)
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
    - Works with aggregated probe marker data (onoff, valence, confidence, etc.)
    - Handles missing data by dropping NaN values per channel
    - Supports both continuous and binary predictors
    """
    n_channels = power_data.shape[1]
    t_stats = np.zeros(n_channels)
    p_values = np.ones(n_channels)
    
    np.random.seed(random_state)
    
    # Suppress convergence warnings
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    
    # Validate data structure before processing
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
    
    for ch_idx in range(n_channels):
        try:
            # Prepare data for this channel
            df_ch = df_behavioral.copy()
            df_ch['power'] = power_data[:, ch_idx]
            
            # Remove NaN values (deterministic)
            df_ch = df_ch.dropna(subset=['power', 'subject', predictor_of_interest])
            
            # Check minimum observations threshold
            if len(df_ch) < 10:
                t_stats[ch_idx] = 0.0
                p_values[ch_idx] = 1.0
                continue
            
            # Check if we have multiple subjects for this channel
            if df_ch['subject'].nunique() < 2:
                t_stats[ch_idx] = 0.0
                p_values[ch_idx] = 1.0
                continue
            
            # Check for sufficient variation in predictor for this channel
            if df_ch[predictor_of_interest].nunique() < 2:
                t_stats[ch_idx] = 0.0
                p_values[ch_idx] = 1.0
                continue
            
            # Fit mixed model (deterministic with fixed random state)
            model = mixedlm(
                formula=formula,
                data=df_ch,
                groups=df_ch["subject"]
            )
            
            # Ensure method is uppercase (statsmodels expects uppercase)
            method_upper = method.upper()
            result = model.fit(
                method=method_upper,
                maxiter=maxiter,
                disp=False
            )
            
            # Extract t-statistic and p-value for predictor of interest
            if predictor_of_interest in result.tvalues.index:
                t_stats[ch_idx] = float(result.tvalues[predictor_of_interest])
                p_values[ch_idx] = float(result.pvalues[predictor_of_interest])
            else:
                t_stats[ch_idx] = 0.0
                p_values[ch_idx] = 1.0
            
        except Exception as e:
            # Convergence failure or other error: set t=0, p=1
            t_stats[ch_idx] = 0.0
            p_values[ch_idx] = 1.0
            if ch_idx < 5:  # Only print first few errors to avoid spam
                print(f"Channel {ch_idx} LMM failed: {e}")
    
    warnings.filterwarnings('default', category=ConvergenceWarning)
    
    return t_stats, p_values


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
