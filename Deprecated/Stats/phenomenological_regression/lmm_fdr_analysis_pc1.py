#!/usr/bin/env python3
"""
Enhanced LMM Analysis with FDR Correction using continuous PC1

This script performs Linear Mixed-Effects Model (LMM) analysis with False Discovery Rate (FDR)
correction using continuous PC1 as a predictor instead of categorical variables.

Key adaptations for PC1:
1. Uses continuous PC1 as fixed effect predictor
2. Adds random slope for PC1 per participant  
3. Z-scoring PC1 within participant option
4. Skips "minimum observations per group" check for continuous predictors
5. Supports PC1-only or PC1 + onoff_label models
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
class PC1AnalysisConfig:
    """Configuration class for PC1 LMM analysis"""
    # Data configuration
    csv_file: str = os.path.join(
        'results/aggregated_mne_markers/merged_pca_eeg_markers.csv'
    )
    out_dir: str = os.path.join(project_root, 'results/lmm_fdr_analysis_pc1')
    
    # LMM Formula configuration - PC1 variants
    base_formula: str = 'mean ~ PC1'  # Options: 'mean ~ PC1' or 'mean ~ PC1 + onoff_label'
    random_effects_formula: str = '~1 + PC1'  # Random slope for PC1
    random_effects_group: str = 'subject_id'
    target_parameter: Optional[str] = 'PC1'  # Focus on PC1 effect
    
    # PC1-specific configuration
    z_score_pc1: bool = True  # Z-score PC1 within each participant
    require_pc1_data: bool = True  # Only analyze rows with PC1 data
    pc1_interaction: Optional[str] = None  # e.g., 'onoff_label' for PC1:onoff interaction
    
    # Model robustness
    optimizer_method: str = 'lbfgs'
    fallback_method: str = 'nm'
    check_convergence: bool = True
    reml: bool = False
    
    # Statistical inference
    min_pc1_observations: int = 10  # Minimum observations with PC1 data per channel
    compute_effect_size: bool = True
    
    # Data preprocessing  
    z_score_by_participant: bool = True  # Z-score markers within each participant
    
    # FDR configuration
    fdr_alpha: float = 0.05
    fdr_method: str = 'indep'
    
    # Code robustness
    verbosity: int = 1
    n_jobs: int = -1
    test_markers: Optional[List[str]] = None
    
    # Plotting
    figsize: Tuple[int, int] = (15, 15)


# -----------------------------------------------------------------------------
# LOGGING UTILITIES (same as original)
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
# PC1-SPECIFIC HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def z_score_pc1_by_participant(df_marker: pd.DataFrame, logger: VerboseLogger) -> pd.DataFrame:
    """
    Z-score PC1 values within each participant.
    
    Parameters
    ----------
    df_marker : pd.DataFrame
        Data for the marker
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    pd.DataFrame
        DataFrame with z-scored PC1 values
    """
    df_zscore = df_marker.copy()
    
    if 'PC1' not in df_zscore.columns or 'subject_id' not in df_zscore.columns:
        logger.warning("   Missing required columns for PC1 z-scoring (PC1, subject_id)")
        return df_zscore
    
    n_participants = df_zscore['subject_id'].nunique()
    logger.debug(f"   Z-scoring PC1 within {n_participants} participants")
    
    # Apply z-scoring by participant
    z_scored_groups = []
    
    for subject_id, group in df_zscore.groupby('subject_id'):
        group_copy = group.copy()
        
        # Only z-score if we have valid PC1 data and variation
        pc1_valid = group_copy['PC1'].dropna()
        if len(pc1_valid) < 2:
            z_scored_groups.append(group_copy)
            continue
            
        pc1_mean = pc1_valid.mean()
        pc1_std = pc1_valid.std()
        
        if pc1_std > 0:
            group_copy['PC1'] = (group_copy['PC1'] - pc1_mean) / pc1_std
        else:
            logger.debug(f"   Warning: Zero PC1 variance for participant {subject_id}")
        
        z_scored_groups.append(group_copy)
    
    df_zscore = pd.concat(z_scored_groups, ignore_index=True)
    
    # Report statistics
    original_range = (df_marker['PC1'].min(), df_marker['PC1'].max())
    zscore_range = (df_zscore['PC1'].min(), df_zscore['PC1'].max())
    
    logger.debug(f"   Original PC1 range: [{original_range[0]:.3f}, {original_range[1]:.3f}]")
    logger.debug(f"   Z-scored PC1 range: [{zscore_range[0]:.3f}, {zscore_range[1]:.3f}]")
    
    return df_zscore


def check_pc1_data_availability(df_marker: pd.DataFrame, ch_names: List[str], 
                               min_obs: int, logger: VerboseLogger) -> Dict[str, bool]:
    """
    Check PC1 data availability per channel (replaces group size check for continuous).
    
    Parameters
    ----------
    df_marker : pd.DataFrame
        Data for the marker
    ch_names : List[str]
        Channel names
    min_obs : int
        Minimum observations with PC1 data
    logger : VerboseLogger
        Logger instance
        
    Returns
    -------
    Dict[str, bool]
        Dictionary mapping channel names to whether they have sufficient PC1 data
    """
    sufficient_data = {}
    
    for ch in ch_names:
        ch_data = df_marker[df_marker['channel'] == ch]
        if ch_data.empty:
            sufficient_data[ch] = False
            continue
            
        # Count observations with valid PC1 data
        pc1_valid_count = ch_data['PC1'].notna().sum()
        sufficient_data[ch] = pc1_valid_count >= min_obs
        
        if not sufficient_data[ch]:
            logger.debug(f"Channel {ch}: insufficient PC1 data ({pc1_valid_count}, required: {min_obs})")
    
    return sufficient_data


# Use original functions with minor adaptations
def compute_effect_size(coefficient: float, std_err: float, method: str = 'standardized') -> float:
    """Compute effect size (same as original)"""
    if np.isnan(coefficient) or np.isnan(std_err) or std_err == 0:
        return np.nan
        
    if method == 'standardized':
        return coefficient / std_err
    elif method == 'cohen':
        return coefficient / std_err * np.sqrt(2)
    else:
        return coefficient / std_err


def z_score_by_participant(df_marker: pd.DataFrame, logger: VerboseLogger) -> pd.DataFrame:
    """Z-score marker values by participant (same as original)"""
    df_zscore = df_marker.copy()
    
    if 'subject_id' not in df_zscore.columns or 'mean' not in df_zscore.columns:
        logger.warning("   Missing required columns for z-scoring (subject_id, mean)")
        return df_zscore
    
    n_participants = df_zscore['subject_id'].nunique()
    logger.debug(f"   Z-scoring marker values within {n_participants} participants")
    
    z_scored_groups = []
    
    for subject_id, group in df_zscore.groupby('subject_id'):
        if len(group) < 2:
            z_scored_groups.append(group)
            continue
        
        group_copy = group.copy()
        mean_val = group_copy['mean'].mean()
        std_val = group_copy['mean'].std()
        
        if std_val > 0:
            group_copy['mean'] = (group_copy['mean'] - mean_val) / std_val
        else:
            logger.debug(f"   Warning: Zero variance for participant {subject_id}")
        
        z_scored_groups.append(group_copy)
    
    df_zscore = pd.concat(z_scored_groups, ignore_index=True)
    return df_zscore


def fit_single_channel_pc1_lmm(ch: str, df_marker: pd.DataFrame, formula: str, 
                              re_formula: str, group_col: str, config: PC1AnalysisConfig,
                              sufficient_data: Dict[str, bool], logger: VerboseLogger) -> Dict[str, Any]:
    """
    Fit PC1 LMM for a single channel.
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
        'n_pc1_obs': 0,
        'error': None
    }
    
    # Check if channel has sufficient data
    if not sufficient_data.get(ch, False):
        result['error'] = 'Insufficient PC1 data'
        return result
    
    # Get channel data with valid PC1
    d_ch = df_marker[df_marker['channel'] == ch].copy()
    d_ch = d_ch.dropna(subset=['PC1'])  # Only keep rows with PC1 data
    
    if d_ch.empty:
        result['error'] = 'No PC1 data'
        return result
    
    result['n_obs'] = len(d_ch)
    result['n_pc1_obs'] = d_ch['PC1'].notna().sum()
    
    # Try fitting with primary optimizer
    methods_to_try = [config.optimizer_method]
    if config.fallback_method and config.fallback_method != config.optimizer_method:
        methods_to_try.append(config.fallback_method)
    
    for method in methods_to_try:
        try:
            logger.debug(f"Fitting PC1 LMM for channel {ch} with method {method}")
            
            # Fit mixed-effects model with random slope for PC1
            model = mixedlm(formula, d_ch, groups=d_ch[group_col], re_formula=re_formula)
            res = model.fit(method=method, reml=config.reml)
            
            # Check convergence
            converged = True
            if config.check_convergence:
                if hasattr(res, 'converged'):
                    converged = res.converged
                elif hasattr(res, 'mle_retvals') and res.mle_retvals is not None:
                    converged = res.mle_retvals.get('converged', True)
                else:
                    converged = not (np.isnan(res.params).any() or np.isnan(res.bse).any())
            
            if not converged and method != methods_to_try[-1]:
                logger.debug(f"Method {method} did not converge for channel {ch}")
                continue
            
            # Extract PC1 parameter
            param_name = config.target_parameter if config.target_parameter in res.params else 'PC1'
            
            if param_name not in res.params:
                result['error'] = f'Parameter {param_name} not found in model'
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
                    result['coefficient'], result['std_err']
                )
            
            logger.debug(f"Successfully fitted PC1 LMM for channel {ch}")
            break
            
        except Exception as e:
            result['error'] = str(e)
            logger.debug(f"PC1 LMM failed for channel {ch} with method {method}: {e}")
    
    return result


def compute_pc1_lmm_stats_parallel(df_marker: pd.DataFrame, ch_names: List[str], 
                                  formula: str, re_formula: str, config: PC1AnalysisConfig,
                                  logger: VerboseLogger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[bool], List[str]]:
    """
    Compute PC1 LMM statistics for all channels with parallelization.
    """
    # Check PC1 data availability (replaces group size check)
    sufficient_data = check_pc1_data_availability(
        df_marker, ch_names, config.min_pc1_observations, logger
    )
    
    n_sufficient = sum(sufficient_data.values())
    logger.info(f"   Channels with sufficient PC1 data: {n_sufficient}/{len(ch_names)} (min {config.min_pc1_observations} obs)")
    
    logger.debug(f"   Running PC1 LMM with formula: {formula}")
    logger.debug(f"   Random effects formula: {re_formula}")
    logger.debug(f"   Random effects grouped by: {config.random_effects_group}")
    
    if config.n_jobs == 1:
        # Sequential processing
        results = []
        for ch in tqdm(ch_names, desc="Fitting PC1 LMMs", disable=config.verbosity == 0):
            result = fit_single_channel_pc1_lmm(
                ch, df_marker, formula, re_formula, config.random_effects_group,
                config, sufficient_data, logger
            )
            results.append(result)
    else:
        # Parallel processing
        results = Parallel(n_jobs=config.n_jobs, backend='threading')(
            delayed(fit_single_channel_pc1_lmm)(
                ch, df_marker, formula, re_formula, config.random_effects_group,
                config, sufficient_data, logger
            ) for ch in tqdm(ch_names, desc="Fitting PC1 LMMs", disable=config.verbosity == 0)
        )
    
    # Extract results
    t_values = np.array([r['t_value'] for r in results])
    p_values = np.array([r['p_value'] for r in results])
    coef_values = np.array([r['coefficient'] for r in results])
    std_err_values = np.array([r['std_err'] for r in results])
    effect_sizes = np.array([r['effect_size'] for r in results])
    converged_list = [r['converged'] for r in results]
    methods_used = [r['method_used'] for r in results]
    
    # Report convergence
    n_converged = sum(converged_list)
    n_failed = sum(1 for r in results if r['error'] is not None)
    
    logger.info(f"   Converged PC1 models: {n_converged}/{len(ch_names)}")
    if n_failed > 0:
        logger.warning(f"   Failed PC1 models: {n_failed}/{len(ch_names)}")
    
    return t_values, p_values, coef_values, std_err_values, effect_sizes, converged_list, methods_used


def apply_fdr_correction_robust(p_values: np.ndarray, alpha: float = 0.05, 
                               method: str = 'indep', logger: VerboseLogger = None) -> Tuple[np.ndarray, np.ndarray]:
    """Apply FDR correction (same as original)"""
    valid_mask = ~np.isnan(p_values)
    n_valid = np.sum(valid_mask)
    
    if logger:
        logger.debug(f"   FDR correction: {n_valid}/{len(p_values)} valid p-values")
    
    if n_valid == 0:
        return np.zeros_like(p_values, dtype=bool), np.full_like(p_values, np.nan)
    
    try:
        rejected_valid, p_corrected_valid = fdrcorrection(
            p_values[valid_mask], alpha=alpha, method=method
        )
    except Exception as e:
        if logger:
            logger.error(f"FDR correction failed: {e}")
        return np.zeros_like(p_values, dtype=bool), np.full_like(p_values, np.nan)
    
    rejected = np.zeros_like(p_values, dtype=bool)
    p_corrected = np.full_like(p_values, np.nan)
    
    rejected[valid_mask] = rejected_valid
    p_corrected[valid_mask] = p_corrected_valid
    
    return rejected, p_corrected


def run_pc1_lmm_fdr_analysis(df: pd.DataFrame, marker: str, config: PC1AnalysisConfig,
                            logger: VerboseLogger) -> Optional[Dict[str, Any]]:
    """
    Run PC1 LMM analysis with FDR correction for a single marker.
    """
    logger.info(f"\n=== Marker: {marker} (PC1 Analysis) ===")
    df_m = df[df['marker'] == marker].copy()
    
    if df_m.empty:
        logger.info("   No data – skipping.")
        return None
    
    # Filter for rows with PC1 data if required
    if config.require_pc1_data:
        initial_rows = len(df_m)
        df_m = df_m.dropna(subset=['PC1'])
        logger.info(f"   Filtered for PC1 data: {len(df_m)}/{initial_rows} rows")
        
        if df_m.empty:
            logger.info("   No PC1 data – skipping.")
            return None
    
    ch_names = sorted(df_m['channel'].unique())
    logger.info(f"   Channels: {len(ch_names)}  |  Rows: {df_m.shape[0]}")
    
    # Apply z-scoring
    if config.z_score_pc1:
        logger.info("   Applying PC1 z-scoring by participant...")
        df_m = z_score_pc1_by_participant(df_m, logger)
    
    if config.z_score_by_participant:
        logger.info("   Applying marker z-scoring by participant...")
        df_m = z_score_by_participant(df_m, logger)
    
    # Check required columns
    required_cols = ['PC1']
    if 'onoff_label' in config.base_formula:
        required_cols.append('onoff_label')
    
    missing_cols = [col for col in required_cols if col not in df_m.columns]
    if missing_cols:
        logger.warning(f"   Missing required columns: {missing_cols} - skipping.")
        return None
    
    # Build formula with interaction if specified
    formula = config.base_formula
    if config.pc1_interaction:
        if config.pc1_interaction in df_m.columns:
            formula = f"{config.base_formula} + PC1:{config.pc1_interaction}"
        else:
            logger.warning(f"   Interaction variable {config.pc1_interaction} not found")
    
    # Run PC1 LMM analysis
    (t_values, p_values, coef_values, std_err_values,
     effect_sizes, converged_list, methods_used) = compute_pc1_lmm_stats_parallel(
        df_m, ch_names, formula, config.random_effects_formula, config, logger
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
        'config': config,
        'formula_used': formula
    }


def save_pc1_lmm_results(results: Dict[str, Any], save_path: str, logger: VerboseLogger) -> int:
    """Save PC1 LMM results to CSV files."""
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
        'formula': results['formula_used'],
        'random_formula': results['config'].random_effects_formula,
        'z_scored_pc1': results['config'].z_score_pc1,
        'z_scored_markers': results['config'].z_score_by_participant,
        'fdr_alpha': results['config'].fdr_alpha,
        'fdr_method': results['config'].fdr_method
    })
    
    # Save full results
    full_results_file = save_path.replace('.csv', '_full_results.csv')
    results_df.to_csv(full_results_file, index=False)
    logger.info(f"   Full results saved to: {full_results_file}")
    
    # Save significant results
    if results['n_significant'] > 0:
        sig_results = results_df[results_df['significant']].copy()
        sig_results_file = save_path.replace('.csv', '_significant_results.csv')
        sig_results.to_csv(sig_results_file, index=False)
        logger.info(f"   Significant results saved to: {sig_results_file}")
    
    return results['n_significant']


def plot_pc1_lmm_results(results: Dict[str, Any], save_path: Optional[str] = None,
                        logger: VerboseLogger = None) -> plt.Figure:
    """Plot PC1 LMM results."""
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
    
    # Create figure
    fig = plt.figure(figsize=config.figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    
    # Main title
    fig.suptitle(f'PC1 LMM Analysis: {marker}\n'
                 f'Formula: {results["formula_used"]} (FDR α = {config.fdr_alpha})',
                 fontsize=16, fontweight='bold')
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 10:
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Filter data for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            t_values_filtered = t_values[ch_indices]
            effect_sizes_filtered = effect_sizes[ch_indices]
            
            # Handle NaN values
            t_values_filtered = np.nan_to_num(t_values_filtered, nan=0.0)
            effect_sizes_filtered = np.nan_to_num(effect_sizes_filtered, nan=0.0)
            
            # Plot t-statistics
            ax_t = fig.add_subplot(gs[0, 0:2])
            im_t, _ = mne.viz.plot_topomap(t_values_filtered, info, show=False, axes=ax_t,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_t.set_title('PC1 T-statistics', fontsize=14, fontweight='bold')
            plt.colorbar(im_t, ax=ax_t, shrink=0.8)
            
            # Plot effect sizes
            ax_effect = fig.add_subplot(gs[0, 2])
            im_effect, _ = mne.viz.plot_topomap(effect_sizes_filtered, info, show=False, axes=ax_effect,
                                              cmap='RdBu_r', contours=6, sensors=True,
                                              names=available_channels, outlines='head')
            ax_effect.set_title('PC1 Effect Sizes')
            plt.colorbar(im_effect, ax=ax_effect, shrink=0.8)
            
            # Plot significant channels
            ax_sig = fig.add_subplot(gs[1, 0:2])
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
            ax_sig.set_title(f'Significant PC1 Effects (FDR α = {config.fdr_alpha})')
            plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
    
    except Exception as e:
        if logger:
            logger.warning(f"Could not create topoplots: {e}")
        
        # Fallback to bar plots
        ax_t = fig.add_subplot(gs[1, 0:2])
        valid_t = ~np.isnan(t_values)
        bars = ax_t.bar(range(len(t_values)), np.nan_to_num(t_values),
                       color=['red' if t > 0 else 'blue' for t in np.nan_to_num(t_values)])
        
        # Highlight significant channels
        for i, (sig, conv) in enumerate(zip(significant, converged)):
            if sig:
                bars[i].set_edgecolor('yellow')
                bars[i].set_linewidth(3)
            if not conv:
                bars[i].set_alpha(0.5)
        
        ax_t.set_title('PC1 T-statistics by Channel')
        ax_t.set_xlabel('Channel Index')
        ax_t.set_ylabel('T-statistic')
        ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Summary statistics
    ax_stats = fig.add_subplot(gs[2, :])
    
    n_converged = sum(converged)
    n_total = len(ch_names)
    
    summary_text = [
        f"Marker: {marker}",
        f"Formula: {results['formula_used']}",
        f"Random effects: {config.random_effects_formula}",
        f"Z-scored PC1: {config.z_score_pc1}",
        f"Z-scored markers: {config.z_score_by_participant}",
        f"Significant channels: {results['n_significant']}/{n_total}",
        f"Converged models: {n_converged}/{n_total}",
        f"PC1 effect size range: {np.nanmin(effect_sizes):.3f} to {np.nanmax(effect_sizes):.3f}",
        f"FDR correction: {config.fdr_method} (α = {config.fdr_alpha})"
    ]
    
    ax_stats.text(0.05, 0.95, '\n'.join(summary_text),
                 transform=ax_stats.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add legend
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Significant'),
        Patch(facecolor='gray', alpha=0.5, label='Not converged'),
        Patch(facecolor='red', label='Positive PC1 effect'),
        Patch(facecolor='blue', label='Negative PC1 effect')
    ]
    ax_stats.legend(handles=legend_elements, loc='center right')
    ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if logger:
            logger.info(f"   PC1 plot saved to: {save_path}")
    
    return fig


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------

def main():
    """Main function with argument parsing for PC1 analysis."""
    
    parser = argparse.ArgumentParser(description='PC1 LMM Analysis with FDR Correction')
    parser.add_argument('--merged-file', type=str,
                       default='results/aggregated_mne_markers/merged_pca_eeg_markers.csv',
                       help='Path to merged PCA-EEG CSV file')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--test-markers', nargs='+')
    parser.add_argument('--formula', type=str, default='mean ~ PC1',
                       help='LMM formula (e.g., "mean ~ PC1" or "mean ~ PC1 + onoff_label")')
    parser.add_argument('--random-formula', type=str, default='~1 + PC1',
                       help='Random effects formula')
    parser.add_argument('--z-score-pc1', action='store_true',
                       help='Z-score PC1 within each participant')
    parser.add_argument('--z-score-markers', action='store_true',
                       help='Z-score markers within each participant')
    parser.add_argument('--pc1-interaction', type=str,
                       help='Variable to interact with PC1 (e.g., onoff_label)')
    parser.add_argument('--min-pc1-obs', type=int, default=10,
                       help='Minimum PC1 observations per channel')
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Create configuration
    config = PC1AnalysisConfig()
    
    # Override with command line arguments
    if args.merged_file:
        config.csv_file = args.merged_file
    if args.verbosity is not None:
        config.verbosity = args.verbosity
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs
    if args.test_markers is not None:
        config.test_markers = args.test_markers
    if args.formula is not None:
        config.base_formula = args.formula
    if args.random_formula is not None:
        config.random_effects_formula = args.random_formula
    if args.z_score_pc1:
        config.z_score_pc1 = True
    if args.z_score_markers:
        config.z_score_by_participant = True
    if args.pc1_interaction:
        config.pc1_interaction = args.pc1_interaction
    if args.min_pc1_obs is not None:
        config.min_pc1_observations = args.min_pc1_obs
    if args.fdr_alpha is not None:
        config.fdr_alpha = args.fdr_alpha
    
    # Initialize logger
    logger = VerboseLogger(config.verbosity)
    
    logger.info("🧠 PC1 LMM ANALYSIS WITH FDR CORRECTION 🧠")
    
    # Check input file
    if not os.path.exists(config.csv_file):
        logger.error(f"Merged CSV not found: {config.csv_file}")
        logger.info("Please run the merge script first:")
        logger.info("python Stats/onoff/merge_pca_eeg_data.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    logger.info("Loading merged data...")
    
    # Load data
    try:
        df = pd.read_csv(config.csv_file)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Check PC1 availability
    pc1_available = df['PC1'].notna().sum()
    logger.info(f"Rows with PC1 data: {pc1_available}/{len(df)} ({100*pc1_available/len(df):.1f}%)")
    
    if pc1_available == 0:
        logger.error("No PC1 data found in merged dataset!")
        sys.exit(1)
    
    # Get markers
    markers = sorted(df['marker'].unique())
    logger.info(f"Available markers: {markers}")
    
    if config.test_markers:
        markers_to_process = [m for m in config.test_markers if m in markers]
        if not markers_to_process:
            logger.error("None of the test markers found in data!")
            sys.exit(1)
        logger.info(f"Processing test markers: {markers_to_process}")
    else:
        markers_to_process = markers
    
    # Display configuration
    logger.info(f"\nPC1 Analysis Configuration:")
    logger.info(f"  Formula: {config.base_formula}")
    logger.info(f"  Random effects: {config.random_effects_formula}")
    logger.info(f"  Z-score PC1: {config.z_score_pc1}")
    logger.info(f"  Z-score markers: {config.z_score_by_participant}")
    logger.info(f"  PC1 interaction: {config.pc1_interaction or 'None'}")
    logger.info(f"  Min PC1 obs/channel: {config.min_pc1_observations}")
    logger.info(f"  FDR: {config.fdr_method} (α = {config.fdr_alpha})")
    
    # Run analysis
    all_results = []
    for marker in markers_to_process:
        try:
            logger.info(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            
            results = run_pc1_lmm_fdr_analysis(df, marker, config, logger)
            
            if results is None:
                logger.warning(f"No results for marker {marker}")
                continue
            
            all_results.append(results)
            
            # Save results
            results_file = os.path.join(config.out_dir, f"{marker}_pc1_lmm_results.csv")
            n_sig_saved = save_pc1_lmm_results(results, results_file, logger)
            
            # Create plot
            plot_path = os.path.join(config.out_dir, f"{marker}_pc1_lmm_plot.png")
            
            try:
                fig = plot_pc1_lmm_results(results, plot_path, logger)
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
    logger.info("PC1 ANALYSIS SUMMARY")
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
    else:
        logger.warning("No results generated!")


if __name__ == '__main__':
    main() 