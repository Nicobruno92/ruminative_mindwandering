#!/usr/bin/env python3
"""
Enhanced LMM Analysis with FDR Correction

This script performs Linear Mixed-Effects Model (LMM) analysis with False Discovery Rate (FDR)
correction instead of cluster-based permutation testing. It loads probe-level data and allows
flexible modification of the LMM formula.

Enhanced Features:
- Model robustness: convergence checking, optimizer selection, parameter selection
- Statistical inference: effect size estimation, sample size checking
- Code robustness: verbosity control, parallelization, NaN handling
- Loads probe-level aggregated data
- Configurable LMM formula
- FDR correction for multiple comparisons
- Same plotting layout as cluster analysis
- Easy to modify for different contrasts
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from scipy import stats
import warnings
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import argparse
from pathlib import Path

# Silence convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# -----------------------------------------------------------------------------
# CONFIGURATION DATACLASS
# -----------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """Configuration class for LMM analysis"""
    # Data configuration
    csv_file: str = os.path.join(
        'results/aggregated_mne_markers/aggregated_mne_markers_onoff_5trials_go_correct_iqr_probe.csv'
    )
    out_dir: str = os.path.join(project_root, 'results/lmm_fdr_analysis')
    
    # LMM Formula configuration
    base_formula: str = 'mean ~ onoff_label'
    random_effects_group: str = 'subject_id'
    target_parameter: Optional[str] = None  # If None, uses first non-intercept parameter
    
    # Model robustness
    optimizer_method: str = 'lbfgs'  # Primary optimizer
    fallback_method: str = 'nm'      # Fallback optimizer
    check_convergence: bool = True
    reml: bool = False
    
    # Statistical inference
    min_obs_per_group: int = 10
    compute_effect_size: bool = True
    
    # Data preprocessing
    z_score_by_participant: bool = False  # Z-score markers within each participant
    
    # FDR configuration
    fdr_alpha: float = 0.05
    fdr_method: str = 'indep'  # 'indep' or 'negcorr'
    
    # Code robustness
    verbosity: int = 1  # 0=silent, 1=summary, 2=full debug
    n_jobs: int = -1    # Number of parallel jobs (-1 = all cores)
    test_markers: Optional[List[str]] = None
    
    # Plotting
    condition_high: str = 'high'
    condition_low: str = 'low'
    figsize: Tuple[int, int] = (15, 15)


# -----------------------------------------------------------------------------
# LOGGING UTILITIES
# -----------------------------------------------------------------------------

class VerboseLogger:
    """Logger with verbosity control"""
    
    def __init__(self, verbosity: int = 1):
        self.verbosity = verbosity
    
    def debug(self, msg: str):
        if self.verbosity >= 2:
            print(f"DEBUG: {msg}")
    
    def info(self, msg: str):
        if self.verbosity >= 1:
            print(msg)
    
    def warning(self, msg: str):
        if self.verbosity >= 1:
            print(f"WARNING: {msg}")
    
    def error(self, msg: str):
        print(f"ERROR: {msg}")


# -----------------------------------------------------------------------------
# ENHANCED LMM ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def check_sample_sizes(df_marker: pd.DataFrame, ch_names: List[str], 
                      group_col: str, min_obs: int, logger: VerboseLogger) -> Dict[str, bool]:
    """
    Check sample sizes per group per channel.
    
    Parameters
    ----------
    df_marker : pd.DataFrame
        Data for the marker
    ch_names : List[str]
        Channel names
    group_col : str
        Grouping column (e.g., 'onoff_label')
    min_obs : int
        Minimum observations per group
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping channel names to whether they have sufficient data
    """
    sufficient_data = {}
    
    for ch in ch_names:
        ch_data = df_marker[df_marker['channel'] == ch]
        if ch_data.empty:
            sufficient_data[ch] = False
            continue
            
        # Check if we have the grouping column
        if group_col not in ch_data.columns:
            logger.warning(f"Group column '{group_col}' not found for channel {ch}")
            sufficient_data[ch] = False
            continue
            
        # Count observations per group
        group_counts = ch_data[group_col].value_counts()
        min_group_size = group_counts.min() if len(group_counts) > 0 else 0
        
        sufficient_data[ch] = min_group_size >= min_obs
        
        if not sufficient_data[ch]:
            logger.debug(f"Channel {ch}: insufficient data (min group size: {min_group_size}, required: {min_obs})")
        
    return sufficient_data


def z_score_by_participant(df_marker: pd.DataFrame, logger: VerboseLogger) -> pd.DataFrame:
    """
    Z-score the marker values within each participant across all channels and conditions.
    
    Parameters
    ----------
    df_marker : pd.DataFrame
        Data for the marker
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with z-scored marker values
    """
    df_zscore = df_marker.copy()
    
    # Check if we have the required columns
    if 'subject_id' not in df_zscore.columns or 'mean' not in df_zscore.columns:
        logger.warning("   Missing required columns for z-scoring (subject_id, mean)")
        return df_zscore
    
    n_participants = df_zscore['subject_id'].nunique()
    logger.debug(f"   Z-scoring marker values within {n_participants} participants")
    
    # Apply z-scoring by participant (more robust approach)
    z_scored_groups = []
    
    for subject_id, group in df_zscore.groupby('subject_id'):
        """Z-score within participant"""
        if len(group) < 2:  # Need at least 2 observations for z-scoring
            z_scored_groups.append(group)
            continue
        
        # Calculate z-scores
        group_copy = group.copy()
        mean_val = group_copy['mean'].mean()
        std_val = group_copy['mean'].std()
        
        if std_val > 0:  # Avoid division by zero
            group_copy['mean'] = (group_copy['mean'] - mean_val) / std_val
        else:
            logger.debug(f"   Warning: Zero variance for participant {subject_id}, keeping original values")
        
        z_scored_groups.append(group_copy)
    
    # Combine all groups
    df_zscore = pd.concat(z_scored_groups, ignore_index=True)
    
    # Report statistics
    original_range = (df_marker['mean'].min(), df_marker['mean'].max())
    zscore_range = (df_zscore['mean'].min(), df_zscore['mean'].max())
    
    logger.debug(f"   Original range: [{original_range[0]:.3e}, {original_range[1]:.3e}]")
    logger.debug(f"   Z-scored range: [{zscore_range[0]:.3f}, {zscore_range[1]:.3f}]")
    
    return df_zscore


def compute_effect_size(coefficient: float, std_err: float, method: str = 'standardized') -> float:
    """
    Compute effect size.
    
    Parameters
    ----------
    coefficient : float
        Model coefficient
    std_err : float
        Standard error of coefficient
    method : str
        Method for effect size ('standardized' or 'cohen')
        
    Returns
    -------
    float
        Effect size
    """
    if np.isnan(coefficient) or np.isnan(std_err) or std_err == 0:
        return np.nan
        
    if method == 'standardized':
        # Standardized coefficient (coefficient / std_err)
        return coefficient / std_err
    elif method == 'cohen':
        # Cohen's d approximation
        return coefficient / std_err * np.sqrt(2)
    else:
        return coefficient / std_err


def fit_single_channel_lmm(ch: str, df_marker: pd.DataFrame, formula: str, 
                          group_col: str, config: AnalysisConfig, 
                          sufficient_data: Dict[str, bool], logger: VerboseLogger) -> Dict[str, Any]:
    """
    Fit LMM for a single channel with enhanced robustness.
    
    Parameters
    ----------
    ch : str
        Channel name
    df_marker : pd.DataFrame
        Data for the marker
    formula : str
        LMM formula
    group_col : str
        Random effects grouping column
    config : AnalysisConfig
        Analysis configuration
    sufficient_data : Dict[str, bool]
        Dictionary indicating which channels have sufficient data
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary for this channel
    """
    result = {
        'channel': ch,
        't_value': np.nan,
        'p_value': np.nan,
        'coefficient': np.nan,
        'std_err': np.nan,
        'effect_size': np.nan,
        'converged': False,
        'method_used': None,
        'n_obs': 0,
        'error': None
    }
    
    # Check if channel has sufficient data
    if not sufficient_data.get(ch, False):
        result['error'] = 'Insufficient data'
        logger.debug(f"Skipping channel {ch}: insufficient data")
        return result
    
    # Get channel data
    d_ch = df_marker[df_marker['channel'] == ch].copy()
    if d_ch.empty:
        result['error'] = 'No data'
        return result
    
    result['n_obs'] = len(d_ch)
    
    # Ensure categorical variables are properly coded
    for col in d_ch.columns:
        if col.endswith('_label') and col in formula:
            d_ch[col] = d_ch[col].astype('category')
    
    # Try fitting with primary optimizer
    methods_to_try = [config.optimizer_method]
    if config.fallback_method and config.fallback_method != config.optimizer_method:
        methods_to_try.append(config.fallback_method)
    
    for method in methods_to_try:
        try:
            logger.debug(f"Fitting LMM for channel {ch} with method {method}")
            
            # Fit mixed-effects model
            model = mixedlm(formula, d_ch, groups=d_ch[group_col], re_formula=f'~1 + onoff_label')
            res = model.fit(method=method, reml=config.reml)
            
            # Check convergence
            converged = True
            if config.check_convergence:
                # Check if optimization converged
                if hasattr(res, 'converged'):
                    converged = res.converged
                elif hasattr(res, 'mle_retvals') and res.mle_retvals is not None:
                    converged = res.mle_retvals.get('converged', True)
                else:
                    # Fallback: check if we have reasonable results
                    converged = not (np.isnan(res.params).any() or np.isnan(res.bse).any())
            
            if not converged and method != methods_to_try[-1]:
                logger.debug(f"Method {method} did not converge for channel {ch}, trying next method")
                continue
            
            # Extract parameter of interest
            if config.target_parameter and config.target_parameter in res.params:
                param_name = config.target_parameter
            else:
                # Find the first non-intercept parameter
                param_names = [k for k in res.params.keys() if k != 'Intercept']
                if param_names:
                    param_name = param_names[0]
                else:
                    result['error'] = 'No non-intercept parameters found'
                    logger.debug(f"No non-intercept parameters found for channel {ch}")
                    return result
            
            # Extract statistics
            result['t_value'] = res.tvalues[param_name]
            result['p_value'] = res.pvalues[param_name]
            result['coefficient'] = res.params[param_name]
            result['std_err'] = res.bse[param_name]
            result['converged'] = converged
            result['method_used'] = method
            
            # Compute effect size
            if config.compute_effect_size:
                result['effect_size'] = compute_effect_size(
                    result['coefficient'], 
                    result['std_err']
                )
            
            logger.debug(f"Successfully fitted LMM for channel {ch} (converged: {converged})")
            break
            
        except Exception as e:
            result['error'] = str(e)
            logger.debug(f"LMM failed for channel {ch} with method {method}: {e}")
            if method == methods_to_try[-1]:
                logger.debug(f"All methods failed for channel {ch}")
    
    return result


def compute_lmm_stats_parallel(df_marker: pd.DataFrame, ch_names: List[str], 
                              formula: str, config: AnalysisConfig, 
                              logger: VerboseLogger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[bool], List[str]]:
    """
    Compute LMM statistics for all channels with parallelization and enhanced robustness.
    
    Returns
    -------
    Tuple containing:
        t_values, p_values, coef_values, std_err_values, effect_sizes, converged_list, methods_used
    """
    # Check sample sizes
    if 'onoff_label' in formula:
        group_col_check = 'onoff_label'
    elif 'valence_label' in formula:
        group_col_check = 'valence_label'
    else:
        group_col_check = 'onoff_label'  # default
    
    sufficient_data = check_sample_sizes(
        df_marker, ch_names, group_col_check, 
        config.min_obs_per_group, logger
    )
    
    n_sufficient = sum(sufficient_data.values())
    logger.info(f"   Channels with sufficient data: {n_sufficient}/{len(ch_names)} (min {config.min_obs_per_group} obs/group)")
    
    # Parallel processing
    logger.debug(f"   Running LMM with formula: {formula}")
    logger.debug(f"   Random effects grouped by: {config.random_effects_group}")
    logger.debug(f"   Using {config.n_jobs} parallel jobs")
    
    if config.n_jobs == 1:
        # Sequential processing
        results = []
        for ch in tqdm(ch_names, desc="Fitting LMMs", disable=config.verbosity == 0):
            result = fit_single_channel_lmm(
                ch, df_marker, formula, config.random_effects_group, 
                config, sufficient_data, logger
            )
            results.append(result)
    else:
        # Parallel processing
        results = Parallel(n_jobs=config.n_jobs, backend='threading')(
            delayed(fit_single_channel_lmm)(
                ch, df_marker, formula, config.random_effects_group, 
                config, sufficient_data, logger
            ) for ch in tqdm(ch_names, desc="Fitting LMMs", disable=config.verbosity == 0)
        )
    
    # Extract results
    t_values = np.array([r['t_value'] for r in results])
    p_values = np.array([r['p_value'] for r in results])
    coef_values = np.array([r['coefficient'] for r in results])
    std_err_values = np.array([r['std_err'] for r in results])
    effect_sizes = np.array([r['effect_size'] for r in results])
    converged_list = [r['converged'] for r in results]
    methods_used = [r['method_used'] for r in results]
    
    # Report convergence issues
    n_converged = sum(converged_list)
    n_failed = sum(1 for r in results if r['error'] is not None)
    
    logger.info(f"   Converged models: {n_converged}/{len(ch_names)}")
    if n_failed > 0:
        logger.warning(f"   Failed models: {n_failed}/{len(ch_names)}")
    
    if config.verbosity >= 2:
        for r in results:
            if r['error'] is not None:
                logger.debug(f"   Channel {r['channel']}: {r['error']}")
    
    return t_values, p_values, coef_values, std_err_values, effect_sizes, converged_list, methods_used


def apply_fdr_correction_robust(p_values: np.ndarray, alpha: float = 0.05, 
                               method: str = 'indep', logger: VerboseLogger = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction to p-values with robust NaN handling.
    
    Parameters
    ----------
    p_values : np.ndarray
        Uncorrected p-values
    alpha : float
        FDR alpha level
    method : str
        FDR method ('indep' or 'negcorr')
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        rejected, p_corrected arrays
    """
    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    n_valid = np.sum(valid_mask)
    
    if logger:
        logger.debug(f"   FDR correction: {n_valid}/{len(p_values)} valid p-values")
    
    if n_valid == 0:
        return np.zeros_like(p_values, dtype=bool), np.full_like(p_values, np.nan)
    
    # Apply FDR correction only to valid p-values
    try:
        rejected_valid, p_corrected_valid = fdrcorrection(
            p_values[valid_mask], 
            alpha=alpha, 
            method=method
        )
    except Exception as e:
        if logger:
            logger.error(f"FDR correction failed: {e}")
        return np.zeros_like(p_values, dtype=bool), np.full_like(p_values, np.nan)
    
    # Create full arrays
    rejected = np.zeros_like(p_values, dtype=bool)
    p_corrected = np.full_like(p_values, np.nan)
    
    rejected[valid_mask] = rejected_valid
    p_corrected[valid_mask] = p_corrected_valid
    
    return rejected, p_corrected


def run_lmm_fdr_analysis_enhanced(df: pd.DataFrame, marker: str, config: AnalysisConfig, 
                                 logger: VerboseLogger) -> Optional[Dict[str, Any]]:
    """
    Run enhanced LMM analysis with FDR correction for a single marker.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data for analysis
    marker : str
        Marker name
    config : AnalysisConfig
        Analysis configuration
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Results dictionary or None if analysis failed
    """
    logger.info(f"\n=== Marker: {marker} ===")
    df_m = df[df['marker'] == marker].copy()
    if df_m.empty:
        logger.info("   No data – skipping.")
        return None

    # Get channels present for this marker
    ch_names = sorted(df_m['channel'].unique())
    logger.info(f"   Channels: {len(ch_names)}  |  Rows: {df_m.shape[0]}")
    
    # Apply z-scoring by participant if requested
    if config.z_score_by_participant:
        logger.info("   Applying z-scoring by participant...")
        df_m = z_score_by_participant(df_m, logger)
    
    # Check if we have the required columns for the formula
    required_cols = []
    if 'onoff_label' in config.base_formula:
        required_cols.append('onoff_label')
    if 'valence_label' in config.base_formula:
        required_cols.append('valence_label')
    if 'confidence_label' in config.base_formula:
        required_cols.append('confidence_label')
    if 'time_label' in config.base_formula:
        required_cols.append('time_label')
    if 'selfother_label' in config.base_formula:
        required_cols.append('selfother_label')
    
    missing_cols = [col for col in required_cols if col not in df_m.columns]
    if missing_cols:
        logger.warning(f"   Missing required columns: {missing_cols} - skipping.")
        return None

    # Run enhanced LMM analysis
    (t_values, p_values, coef_values, std_err_values, 
     effect_sizes, converged_list, methods_used) = compute_lmm_stats_parallel(
        df_m, ch_names, config.base_formula, config, logger
    )
    
    # Apply FDR correction
    rejected, p_corrected = apply_fdr_correction_robust(
        p_values, config.fdr_alpha, config.fdr_method, logger
    )
    
    n_significant = np.sum(rejected)
    logger.info(f"   Significant channels (FDR α={config.fdr_alpha}): {n_significant}/{len(ch_names)}")
    
    if n_significant > 0:
        sig_channels = [ch_names[i] for i in np.where(rejected)[0]]
        logger.info(f"   Significant channels: {', '.join(sig_channels[:10])}" + 
                   ("..." if len(sig_channels) > 10 else ""))

    return {
        'marker': marker,
        'ch_names': ch_names,
        't_values': t_values,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'coef_values': coef_values,
        'std_err_values': std_err_values,
        'effect_sizes': effect_sizes,
        'significant': rejected,
        'converged': converged_list,
        'methods_used': methods_used,
        'n_significant': n_significant,
        'df_marker': df_m,
        'config': config
    }


def save_lmm_results_enhanced(results: Dict[str, Any], save_path: str, logger: VerboseLogger) -> int:
    """
    Save enhanced LMM results to CSV files.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from run_lmm_fdr_analysis_enhanced
    save_path : str
        Base path for saving results
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    int
        Number of significant results saved
    """
    # Create results DataFrame
    results_df = pd.DataFrame({
        'channel': results['ch_names'],
        'coefficient': results['coef_values'],
        'std_err': results['std_err_values'],
        'effect_size': results['effect_sizes'],
        't_statistic': results['t_values'],
        'p_value': results['p_values'],
        'p_corrected': results['p_corrected'],
        'significant': results['significant'],
        'converged': results['converged'],
        'method_used': results['methods_used'],
        'marker': results['marker'],
        'formula': results['config'].base_formula,
        'z_scored': results['config'].z_score_by_participant,
        'fdr_alpha': results['config'].fdr_alpha,
        'fdr_method': results['config'].fdr_method
    })
    
    # Save full results
    full_results_file = save_path.replace('.csv', '_full_results.csv')
    results_df.to_csv(full_results_file, index=False)
    logger.info(f"   Full results saved to: {full_results_file}")
    
    # Save significant results only
    if results['n_significant'] > 0:
        sig_results = results_df[results_df['significant']].copy()
        sig_results_file = save_path.replace('.csv', '_significant_results.csv')
        sig_results.to_csv(sig_results_file, index=False)
        logger.info(f"   Significant results saved to: {sig_results_file}")
    
    return results['n_significant']


def plot_lmm_results_enhanced(results: Dict[str, Any], save_path: Optional[str] = None, 
                             logger: VerboseLogger = None) -> plt.Figure:
    """
    Plot enhanced LMM results with robust NaN handling.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from run_lmm_fdr_analysis_enhanced
    save_path : Optional[str]
        Path to save figure
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    config = results['config']
    marker = results['marker']
    ch_names = results['ch_names']
    t_values = results['t_values']
    p_corrected = results['p_corrected']
    significant = results['significant']
    effect_sizes = results['effect_sizes']
    converged = results['converged']
    
    # Create figure with enhanced layout
    fig = plt.figure(figsize=config.figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    
    # Main title
    fig.suptitle(f'Enhanced LMM Analysis: {marker}\n'
                 f'{config.condition_high} vs {config.condition_low} (FDR α = {config.fdr_alpha})', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 10:  # Only create topoplots if we have enough channels
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Filter data for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            t_values_filtered = t_values[ch_indices]
            effect_sizes_filtered = effect_sizes[ch_indices]
            
            # Handle NaN values for plotting
            t_values_filtered = np.nan_to_num(t_values_filtered, nan=0.0)
            effect_sizes_filtered = np.nan_to_num(effect_sizes_filtered, nan=0.0)
            
            # Calculate condition means for available channels
            df_m = results['df_marker']
            means = df_m.groupby(['onoff_label', 'channel'])['mean'].mean().unstack(level=0)
            
            if config.condition_high in means.columns and config.condition_low in means.columns:
                high_filtered = np.array([means.loc[ch, config.condition_high] 
                                        for ch in available_channels if ch in means.index])
                low_filtered = np.array([means.loc[ch, config.condition_low] 
                                       for ch in available_channels if ch in means.index])
                diff_filtered = high_filtered - low_filtered
                
                # Handle NaN values
                high_filtered = np.nan_to_num(high_filtered, nan=0.0)
                low_filtered = np.nan_to_num(low_filtered, nan=0.0)
                diff_filtered = np.nan_to_num(diff_filtered, nan=0.0)
            else:
                high_filtered = low_filtered = diff_filtered = None
            
            # First row: Raw values and effect sizes
            if high_filtered is not None and low_filtered is not None:
                # Plot high condition
                ax_high = fig.add_subplot(gs[0, 0])
                im_high, _ = mne.viz.plot_topomap(high_filtered, info, show=False, axes=ax_high,
                                             cmap='viridis', contours=6, sensors=True,
                                             names=available_channels, outlines='head')
                ax_high.set_title(f'{config.condition_high} (ON-task)')
                plt.colorbar(im_high, ax=ax_high, shrink=0.8)
                
                # Plot low condition
                ax_low = fig.add_subplot(gs[0, 1])
                im_low, _ = mne.viz.plot_topomap(low_filtered, info, show=False, axes=ax_low,
                                             cmap='viridis', contours=6, sensors=True,
                                             names=available_channels, outlines='head')
                ax_low.set_title(f'{config.condition_low} (OFF-task)')
                plt.colorbar(im_low, ax=ax_low, shrink=0.8)
                
                # Plot effect sizes
                ax_effect = fig.add_subplot(gs[0, 2])
                im_effect, _ = mne.viz.plot_topomap(effect_sizes_filtered, info, show=False, axes=ax_effect,
                                                  cmap='RdBu_r', contours=6, sensors=True,
                                                  names=available_channels, outlines='head')
                ax_effect.set_title('Effect Sizes')
                plt.colorbar(im_effect, ax=ax_effect, shrink=0.8)
            
            # Second row: T-statistics and significant channels
            ax_t = fig.add_subplot(gs[1, 0:2])
            im_t, _ = mne.viz.plot_topomap(t_values_filtered, info, show=False, axes=ax_t,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_t.set_title('T-statistics', fontsize=14, fontweight='bold')
            plt.colorbar(im_t, ax=ax_t, shrink=0.8)
            
            # Plot significant channels with convergence info
            ax_sig = fig.add_subplot(gs[1, 2])
            sig_indices = [ch_indices[i] for i, ch in enumerate(available_channels) 
                          if ch in ch_names and significant[ch_names.index(ch)]]
            
            t_masked = np.zeros_like(t_values_filtered)
            if sig_indices:
                for i, ch in enumerate(available_channels):
                    if ch in ch_names and significant[ch_names.index(ch)]:
                        t_masked[i] = t_values_filtered[i]
            
            im_sig, _ = mne.viz.plot_topomap(t_masked, info, show=False, axes=ax_sig,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_sig.set_title(f'Significant (FDR α = {config.fdr_alpha})')
            plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
    
    except Exception as e:
        if logger:
            logger.warning(f"Could not create topoplots: {e}")
        
        # Fallback to bar plots
        # T-statistics
        ax_t = fig.add_subplot(gs[1, 0:2])
        valid_t = ~np.isnan(t_values)
        bars = ax_t.bar(range(len(t_values)), np.nan_to_num(t_values), 
                       color=['red' if t > 0 else 'blue' for t in np.nan_to_num(t_values)])
        
        # Highlight significant and non-converged channels
        for i, (sig, conv) in enumerate(zip(significant, converged)):
            if sig:
                bars[i].set_edgecolor('yellow')
                bars[i].set_linewidth(3)
            if not conv:
                bars[i].set_alpha(0.5)
        
        ax_t.set_title('T-statistics by Channel')
        ax_t.set_xlabel('Channel Index')
        ax_t.set_ylabel('T-statistic')
        ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Effect sizes
        ax_effect = fig.add_subplot(gs[0, 2])
        ax_effect.bar(range(len(effect_sizes)), np.nan_to_num(effect_sizes),
                     color=['green' if e > 0 else 'red' for e in np.nan_to_num(effect_sizes)])
        ax_effect.set_title('Effect Sizes')
        ax_effect.set_xlabel('Channel Index')
        ax_effect.set_ylabel('Effect Size')
        ax_effect.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Third row: Enhanced summary statistics
    ax_stats = fig.add_subplot(gs[2, :])
    
    # Create summary text
    n_converged = sum(converged)
    n_total = len(ch_names)
    summary_text = [
        f"Marker: {marker}",
        f"Formula: {config.base_formula}",
        f"Z-scored by participant: {config.z_score_by_participant}",
        f"Significant channels: {results['n_significant']}/{n_total}",
        f"Converged models: {n_converged}/{n_total}",
        f"FDR correction: {config.fdr_method} (α = {config.fdr_alpha})",
        f"Effect size range: {np.nanmin(effect_sizes):.3f} to {np.nanmax(effect_sizes):.3f}",
        f"Optimizer: {config.optimizer_method}" + (f" (fallback: {config.fallback_method})" if config.fallback_method else "")
    ]
    
    ax_stats.text(0.05, 0.95, '\n'.join(summary_text), 
                 transform=ax_stats.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add convergence and significance legend
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Significant'),
        Patch(facecolor='gray', alpha=0.5, label='Not converged'),
        Patch(facecolor='red', label='Negative effect'),
        Patch(facecolor='blue', label='Positive effect')
    ]
    ax_stats.legend(handles=legend_elements, loc='center right')
    ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"   Enhanced plot saved to: {save_path}")
    
    return fig


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

def main():
    """Main function with argument parsing and enhanced analysis pipeline."""
    
    # Argument parsing
    parser = argparse.ArgumentParser(description='Enhanced LMM Analysis with FDR Correction')
    parser.add_argument('--config-file', type=str, help='Path to configuration file')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                       help='Verbosity level: 0=silent, 1=summary, 2=debug')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--test-markers', nargs='+', 
                       help='Test only specific markers (e.g., --test-markers d a p1)')
    parser.add_argument('--formula', type=str, default='mean ~ onoff_label',
                       help='LMM formula to use')
    parser.add_argument('--optimizer', type=str, default='lbfgs',
                       help='Primary optimizer method')
    parser.add_argument('--fallback-optimizer', type=str, default='nm',
                       help='Fallback optimizer method')
    parser.add_argument('--min-obs', type=int, default=10,
                       help='Minimum observations per group per channel')
    parser.add_argument('--fdr-alpha', type=float, default=0.05,
                       help='FDR alpha level')
    parser.add_argument('--z-score', action='store_true',
                       help='Z-score marker values within each participant')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig()
    
    # Override with command line arguments
    if args.verbosity is not None:
        config.verbosity = args.verbosity
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs
    if args.test_markers is not None:
        config.test_markers = args.test_markers
    if args.formula is not None:
        config.base_formula = args.formula
    if args.optimizer is not None:
        config.optimizer_method = args.optimizer
    if args.fallback_optimizer is not None:
        config.fallback_method = args.fallback_optimizer
    if args.min_obs is not None:
        config.min_obs_per_group = args.min_obs
    if args.fdr_alpha is not None:
        config.fdr_alpha = args.fdr_alpha
    if args.z_score:
        config.z_score_by_participant = True
    
    # Initialize logger
    logger = VerboseLogger(config.verbosity)
    
    logger.info("🧠 ENHANCED LMM ANALYSIS WITH FDR CORRECTION 🧠")
    
    # Check input file
    if not os.path.exists(config.csv_file):
        logger.error(f"CSV not found: {config.csv_file}")
        sys.exit(1)

    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)

    logger.info("Loading data...")
    
    # Check file size before loading
    file_size_mb = os.path.getsize(config.csv_file) / (1024 * 1024)
    logger.info(f"CSV file size: {file_size_mb:.1f} MB")
    
    # Load data with error handling
    try:
        if file_size_mb > 500:
            logger.info("Large file detected. Loading with chunking...")
            chunks = pd.read_csv(config.csv_file, chunksize=100000)
            df_list = []
            for chunk in tqdm(chunks, desc="Processing chunks", disable=config.verbosity == 0):
                if config.test_markers:
                    filtered_chunk = chunk[chunk['marker'].isin(config.test_markers)]
                    if not filtered_chunk.empty:
                        df_list.append(filtered_chunk)
                else:
                    df_list.append(chunk)
            df = pd.concat(df_list, ignore_index=True)
            logger.info(f"Loaded data with shape: {df.shape}")
        else:
            df = pd.read_csv(config.csv_file)
            logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Validate data
    logger.debug(f"Available columns: {df.columns.tolist()}")
    if 'onoff_label' in df.columns:
        logger.debug(f"Available onoff_labels: {df['onoff_label'].unique()}")
    
    # Ensure expected columns
    required_base_cols = ['subject_id', 'marker', 'channel', 'mean']
    missing_base_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_base_cols:
        logger.error(f"Missing required columns: {missing_base_cols}")
        sys.exit(1)

    markers = sorted(df['marker'].unique())
    logger.info(f"Available markers: {markers}")
    
    # Determine which markers to process
    if config.test_markers:
        markers_to_process = [m for m in config.test_markers if m in markers]
        if not markers_to_process:
            logger.error("None of the test markers found in data!")
            sys.exit(1)
        logger.info(f"Processing test markers: {markers_to_process}")
    else:
        markers_to_process = markers
        logger.info(f"Processing all {len(markers_to_process)} markers")

    # Display configuration
    logger.info(f"\nAnalysis Configuration:")
    logger.info(f"  Formula: {config.base_formula}")
    logger.info(f"  Random effects group: {config.random_effects_group}")
    logger.info(f"  Target parameter: {config.target_parameter or 'auto-detect'}")
    logger.info(f"  Optimizer: {config.optimizer_method} (fallback: {config.fallback_method})")
    logger.info(f"  Min obs/group: {config.min_obs_per_group}")
    logger.info(f"  Z-score by participant: {config.z_score_by_participant}")
    logger.info(f"  FDR: {config.fdr_method} (α = {config.fdr_alpha})")
    logger.info(f"  Parallel jobs: {config.n_jobs}")
    logger.info(f"  Verbosity: {config.verbosity}")

    # Run analysis
    all_results = []
    for marker in markers_to_process:
        try:
            logger.info(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            
            # Run enhanced LMM analysis
            results = run_lmm_fdr_analysis_enhanced(df, marker, config, logger)
            
            if results is None:
                logger.warning(f"No results for marker {marker} - skipping")
                continue
                
            all_results.append(results)
            
            # Save results
            results_file = os.path.join(config.out_dir, f"{marker}_lmm_results.csv")
            n_sig_saved = save_lmm_results_enhanced(results, results_file, logger)
            
            # Create plot
            plot_path = os.path.join(config.out_dir, f"{marker}_lmm_plot.png")
            
            try:
                fig = plot_lmm_results_enhanced(results, plot_path, logger)
                plt.close(fig)
                
            except Exception as e:
                logger.error(f"Failed to create plot for marker {marker}: {e}")

        except Exception as e:
            logger.error(f"Failed to process marker {marker}: {e}")
            if config.verbosity >= 2:
                import traceback
                traceback.print_exc()
            continue

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("ENHANCED ANALYSIS SUMMARY")
    logger.info("="*60)
    
    if all_results:
        sig_total = 0
        converged_total = 0
        total_channels = 0
        
        for results in all_results:
            sig = results['n_significant']
            converged = sum(results['converged'])
            n_channels = len(results['ch_names'])
            
            sig_total += sig
            converged_total += converged
            total_channels += n_channels
            
            logger.info(f"{results['marker']:>15}: {sig:>3} significant, {converged:>3}/{n_channels} converged")
        
        logger.info("-" * 60)
        logger.info(f"{'Total markers analyzed:':>25} {len(all_results)}")
        logger.info(f"{'Total significant channels:':>25} {sig_total}")
        logger.info(f"{'Total converged models:':>25} {converged_total}/{total_channels}")
        logger.info(f"{'Results saved to:':>25} {config.out_dir}")
        logger.info(f"{'Configuration:':>25} Enhanced pipeline")
    else:
        logger.warning("No results generated!")
    
    logger.info("")


if __name__ == '__main__':
    main() 