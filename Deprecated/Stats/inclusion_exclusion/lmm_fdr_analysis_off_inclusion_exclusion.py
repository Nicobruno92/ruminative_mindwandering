#!/usr/bin/env python3
"""
LMM Analysis with FDR Correction - OFF Task Conditions Inclusion/Exclusion Comparison

This script performs Linear Mixed-Effects Model (LMM) analysis with False Discovery Rate (FDR)
correction to compare OFF task conditions between inclusion and exclusion conditions.

Based on the order (IE/EI) in metadata and task (Sart2/Sart4):
- IE order: Sart2 = Inclusion, Sart4 = Exclusion
- EI order: Sart2 = Exclusion, Sart4 = Inclusion

Features:
- Loads probe-level aggregated data filtered for OFF conditions only
- Filters for Sart2 and Sart4 tasks only
- Assigns inclusion/exclusion labels based on order and task
- Compares inclusion vs exclusion for OFF task conditions
- FDR correction for multiple comparisons
- Comprehensive plotting with topoplots
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

# Silence convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Use the probe-level aggregated file for more detailed analysis
CSV_FILE = os.path.join(
    project_root,
    'results/aggregated_mne_markers/aggregated_mne_markers_onoff_valence_confidence_time_selfother_5trials_go_correct_probe.csv'
)

# Metadata file with order information
METADATA_FILE = os.path.join(project_root, 'metadata_experiment.csv')

OUT_DIR = os.path.join(project_root, 'results/off_inclusion_exclusion_comparison')

# LMM Formula configuration - comparing inclusion vs exclusion for OFF conditions
BASE_FORMULA = 'mean ~ inclusion_exclusion_label'  # Compare inclusion vs exclusion

# Random effects (keep subject as random effect)
RANDOM_EFFECTS_GROUP = 'subject_id'

# FDR configuration
FDR_ALPHA = 0.05      # FDR-corrected alpha level
FDR_METHOD = 'indep'  # 'indep' or 'negcorr' for FDR correction

# Tasks to include (Sart2 and Sart4 only)
TASKS_TO_INCLUDE = ['Sart2', 'Sart4']

# For testing, limit to these markers (set to None to process all markers)
TEST_MARKERS = None  # Process all markers by default
# TEST_MARKERS = ['d', 'a', 'p1']  # Example: test only specific markers

# -----------------------------------------------------------------------------
# DATA LOADING AND PREPROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def load_metadata(metadata_file):
    """
    Load metadata and create inclusion/exclusion mapping based on order and task.
    
    Returns
    -------
    dict
        Dictionary mapping (subject_id, task) to inclusion/exclusion condition
    """
    print(f"Loading metadata from: {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    
    # Create subject to order mapping
    subject_to_order = {}
    for _, row in metadata.iterrows():
        subject_id = row['subj']
        order = row['order (IE/EI)']
        subject_to_order[subject_id] = order
    
    # Create mapping for (subject, task) -> inclusion/exclusion
    subject_task_to_condition = {}
    
    for subject_id, order in subject_to_order.items():
        if order == 'IE':
            # IE: Inclusion first (Sart2), then Exclusion (Sart4)
            subject_task_to_condition[(subject_id, 'Sart2')] = 'inclusion'
            subject_task_to_condition[(subject_id, 'Sart4')] = 'exclusion'
        elif order == 'EI':
            # EI: Exclusion first (Sart2), then Inclusion (Sart4)
            subject_task_to_condition[(subject_id, 'Sart2')] = 'exclusion'
            subject_task_to_condition[(subject_id, 'Sart4')] = 'inclusion'
    
    print(f"Loaded metadata for {len(subject_to_order)} subjects")
    
    # Count conditions
    inclusion_count = sum(1 for cond in subject_task_to_condition.values() if cond == 'inclusion')
    exclusion_count = sum(1 for cond in subject_task_to_condition.values() if cond == 'exclusion')
    
    print(f"Inclusion conditions: {inclusion_count}")
    print(f"Exclusion conditions: {exclusion_count}")
    
    return subject_task_to_condition


def filter_and_merge_data(df, subject_task_to_condition):
    """
    Filter data for OFF conditions and Sart2/Sart4 only, then merge with inclusion/exclusion labels.
    
    Parameters
    ----------
    df : DataFrame
        Original aggregated data
    subject_task_to_condition : dict
        Mapping of (subject_id, task) to inclusion/exclusion condition
        
    Returns
    -------
    DataFrame
        Filtered and merged data
    """
    print("Filtering data for OFF conditions, Sart2/Sart4 tasks, and merging with condition info...")
    
    # Filter for OFF conditions only
    df_off = df[df['onoff_label'] == 'low'].copy()
    print(f"Filtered to OFF conditions: {df_off.shape[0]} rows")
    
    # Filter for Sart2 and Sart4 tasks only
    df_off = df_off[df_off['task'].isin(TASKS_TO_INCLUDE)].copy()
    print(f"Filtered to Sart2/Sart4 tasks: {df_off.shape[0]} rows")
    
    # Add inclusion/exclusion condition information
    df_off['inclusion_exclusion'] = df_off.apply(
        lambda row: subject_task_to_condition.get((row['subject_id'], row['task']), None), 
        axis=1
    )
    
    # Remove rows without condition information
    before_filter = df_off.shape[0]
    df_off = df_off.dropna(subset=['inclusion_exclusion']).copy()
    after_filter = df_off.shape[0]
    
    if before_filter != after_filter:
        print(f"Removed {before_filter - after_filter} rows due to missing condition info")
    
    # Convert condition to categorical for LMM
    df_off['inclusion_exclusion_label'] = df_off['inclusion_exclusion'].astype('category')
    
    print(f"Final dataset: {df_off.shape[0]} rows")
    print(f"Condition distribution:")
    print(df_off['inclusion_exclusion_label'].value_counts().sort_index())
    print(f"Task distribution:")
    print(df_off['task'].value_counts().sort_index())
    
    return df_off


# -----------------------------------------------------------------------------
# LMM ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def compute_lmm_stats(df_marker: pd.DataFrame, ch_names, formula=BASE_FORMULA, group_col=RANDOM_EFFECTS_GROUP):
    """
    Compute LMM statistics for each channel.
    
    Returns
    -------
    t_values : array
        T-statistics for each channel
    p_values : array
        P-values for each channel
    coef_values : array
        Coefficient estimates for each channel
    """
    t_vec = np.zeros(len(ch_names))
    p_vec = np.zeros(len(ch_names))
    coef_vec = np.zeros(len(ch_names))
    
    print(f"   Running LMM with formula: {formula}")
    print(f"   Random effects grouped by: {group_col}")
    
    for i, ch in enumerate(ch_names):
        d_ch = df_marker[df_marker['channel'] == ch].copy()
        if d_ch.empty:
            t_vec[i] = np.nan
            p_vec[i] = np.nan
            coef_vec[i] = np.nan
            continue
            
        # Ensure we have both conditions for this channel
        conditions_present = d_ch['inclusion_exclusion_label'].nunique()
        if conditions_present < 2:
            print(f"    Warning: Only {conditions_present} condition(s) present for channel {ch}")
            t_vec[i] = np.nan
            p_vec[i] = np.nan
            coef_vec[i] = np.nan
            continue
        
        try:
            # Fit mixed-effects model
            model = mixedlm(formula, d_ch, groups=d_ch[group_col])
            res = model.fit(reml=False)
            
            # Get the parameter of interest (inclusion vs exclusion comparison)
            # Find the parameter that's not the intercept
            param_names = [k for k in res.params.keys() if k != 'Intercept']
            if param_names:
                param_name = param_names[0]  # Take the first non-intercept parameter
                t_vec[i] = res.tvalues[param_name]
                p_vec[i] = res.pvalues[param_name]
                coef_vec[i] = res.params[param_name]
            else:
                print(f"    Warning: No non-intercept parameters found for channel {ch}")
                t_vec[i] = np.nan
                p_vec[i] = np.nan
                coef_vec[i] = np.nan
                
        except Exception as e:
            print(f"    LMM failed for channel {ch}: {e}")
            t_vec[i] = np.nan
            p_vec[i] = np.nan
            coef_vec[i] = np.nan
    
    return t_vec, p_vec, coef_vec


def apply_fdr_correction(p_values, alpha=FDR_ALPHA, method=FDR_METHOD):
    """
    Apply FDR correction to p-values.
    
    Parameters
    ----------
    p_values : array
        Uncorrected p-values
    alpha : float
        FDR alpha level
    method : str
        FDR method ('indep' or 'negcorr')
        
    Returns
    -------
    rejected : array
        Boolean array indicating which hypotheses are rejected
    p_corrected : array
        FDR-corrected p-values
    """
    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    
    if not np.any(valid_mask):
        return np.zeros_like(p_values, dtype=bool), p_values
    
    # Apply FDR correction only to valid p-values
    rejected_valid, p_corrected_valid = fdrcorrection(
        p_values[valid_mask], 
        alpha=alpha, 
        method=method
    )
    
    # Create full arrays
    rejected = np.zeros_like(p_values, dtype=bool)
    p_corrected = np.full_like(p_values, np.nan)
    
    rejected[valid_mask] = rejected_valid
    p_corrected[valid_mask] = p_corrected_valid
    
    return rejected, p_corrected


def run_lmm_fdr_analysis(df, marker, formula=BASE_FORMULA, group_col=RANDOM_EFFECTS_GROUP, 
                        fdr_alpha=FDR_ALPHA, fdr_method=FDR_METHOD):
    """
    Run LMM analysis with FDR correction for a single marker.
    
    Parameters
    ----------
    df : DataFrame
        Data for analysis (already filtered for OFF conditions)
    marker : str
        Marker name
    formula : str
        LMM formula
    group_col : str
        Column name for random effects grouping
    fdr_alpha : float
        FDR alpha level
    fdr_method : str
        FDR correction method
        
    Returns
    -------
    dict
        Results dictionary
    """
    print(f"\n=== Marker: {marker} ===")
    df_m = df[df['marker'] == marker].copy()
    if df_m.empty:
        print("   No data – skipping.")
        return None

    # Get channels present for this marker
    ch_names = sorted(df_m['channel'].unique())
    print(f"   Channels: {len(ch_names)}  |  Rows: {df_m.shape[0]}")
    
    # Check condition distribution for this marker
    condition_dist = df_m['inclusion_exclusion_label'].value_counts().sort_index()
    print(f"   Condition distribution:")
    for condition, count in condition_dist.items():
        print(f"     {condition}: {count} observations")
    
    if len(condition_dist) < 2:
        print("   Warning: Less than 2 conditions present - skipping marker")
        return None

    # Run LMM analysis
    t_values, p_values, coef_values = compute_lmm_stats(df_m, ch_names, formula, group_col)
    
    # Apply FDR correction
    rejected, p_corrected = apply_fdr_correction(p_values, fdr_alpha, fdr_method)
    
    n_significant = np.sum(rejected)
    print(f"   Significant channels (FDR α={fdr_alpha}): {n_significant}/{len(ch_names)}")
    
    if n_significant > 0:
        sig_channels = [ch_names[i] for i in np.where(rejected)[0]]
        print(f"   Significant channels: {', '.join(sig_channels[:10])}" + 
              ("..." if len(sig_channels) > 10 else ""))

    return {
        'marker': marker,
        'ch_names': ch_names,
        't_values': t_values,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'coef_values': coef_values,
        'significant': rejected,
        'n_significant': n_significant,
        'df_marker': df_m,
        'formula': formula,
        'fdr_alpha': fdr_alpha,
        'fdr_method': fdr_method
    }


def save_lmm_results(results, save_path):
    """
    Save LMM results to CSV files.
    
    Parameters
    ----------
    results : dict
        Results from run_lmm_fdr_analysis
    save_path : str
        Base path for saving results
    """
    # Create results DataFrame
    results_df = pd.DataFrame({
        'channel': results['ch_names'],
        'coefficient': results['coef_values'],
        't_statistic': results['t_values'],
        'p_value': results['p_values'],
        'p_corrected': results['p_corrected'],
        'significant': results['significant'],
        'marker': results['marker'],
        'formula': results['formula'],
        'fdr_alpha': results['fdr_alpha'],
        'fdr_method': results['fdr_method']
    })
    
    # Save full results
    full_results_file = save_path.replace('.csv', '_full_results.csv')
    results_df.to_csv(full_results_file, index=False)
    print(f"   Full results saved to: {full_results_file}")
    
    # Save significant results only
    if results['n_significant'] > 0:
        sig_results = results_df[results_df['significant']].copy()
        sig_results_file = save_path.replace('.csv', '_significant_results.csv')
        sig_results.to_csv(sig_results_file, index=False)
        print(f"   Significant results saved to: {sig_results_file}")
    
    return results['n_significant']


def plot_lmm_results(results, condition_high='Inclusion', condition_low='Exclusion', 
                    save_path=None, figsize=(15, 15)):
    """
    Plot LMM results with the same layout as cluster analysis plots.
    
    Parameters
    ----------
    results : dict
        Results from run_lmm_fdr_analysis
    condition_high : str
        Inclusion condition label
    condition_low : str
        Exclusion condition label
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    marker = results['marker']
    ch_names = results['ch_names']
    t_values = results['t_values']
    p_corrected = results['p_corrected']
    significant = results['significant']
    fdr_alpha = results['fdr_alpha']
    
    # Create figure with a 3x3 grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    
    # Main title
    fig.suptitle(f'LMM Analysis - OFF Task Inclusion vs Exclusion: {marker}\n'
                 f'{condition_high} vs {condition_low} (FDR α = {fdr_alpha})', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 0:
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Filter data for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            t_values_filtered = t_values[ch_indices]
            
            # Calculate condition means for available channels
            df_m = results['df_marker']
            means = df_m.groupby(['inclusion_exclusion_label', 'channel'])['mean'].mean().unstack(level=0)
            
            if 'inclusion' in means.columns and 'exclusion' in means.columns:
                inclusion_filtered = np.array([means.loc[ch, 'inclusion'] for ch in available_channels if ch in means.index])
                exclusion_filtered = np.array([means.loc[ch, 'exclusion'] for ch in available_channels if ch in means.index])
                diff_filtered = inclusion_filtered - exclusion_filtered
            else:
                inclusion_filtered = exclusion_filtered = diff_filtered = None
            
            # First row: Raw values - inclusion, exclusion, and difference
            if inclusion_filtered is not None and exclusion_filtered is not None:
                # Plot inclusion
                ax_inclusion = fig.add_subplot(gs[0, 0])
                im_inclusion, _ = mne.viz.plot_topomap(inclusion_filtered, info, show=False, axes=ax_inclusion,
                                                   cmap='viridis', contours=6, sensors=True,
                                                   names=available_channels, outlines='head')
                ax_inclusion.set_title(f'{condition_high} (OFF-task) Raw Values')
                cbar_inclusion = plt.colorbar(im_inclusion, ax=ax_inclusion, shrink=0.8)
                cbar_inclusion.set_label(f'{marker} value')
                
                # Plot exclusion
                ax_exclusion = fig.add_subplot(gs[0, 1])
                im_exclusion, _ = mne.viz.plot_topomap(exclusion_filtered, info, show=False, axes=ax_exclusion,
                                                   cmap='viridis', contours=6, sensors=True,
                                                   names=available_channels, outlines='head')
                ax_exclusion.set_title(f'{condition_low} (OFF-task) Raw Values')
                cbar_exclusion = plt.colorbar(im_exclusion, ax=ax_exclusion, shrink=0.8)
                cbar_exclusion.set_label(f'{marker} value')
                
                # Plot difference
                ax_diff = fig.add_subplot(gs[0, 2])
                im_diff, _ = mne.viz.plot_topomap(diff_filtered, info, show=False, axes=ax_diff,
                                                 cmap='RdBu_r', contours=6, sensors=True,
                                                 names=available_channels, outlines='head')
                ax_diff.set_title(f'Difference ({condition_high} minus {condition_low})')
                cbar_diff = plt.colorbar(im_diff, ax=ax_diff, shrink=0.8)
                cbar_diff.set_label(f'Difference in {marker} value')
                
                # Add legend for difference plot
                legend_elements = [
                    Patch(facecolor='red', label='Inclusion > Exclusion'),
                    Patch(facecolor='blue', label='Exclusion > Inclusion')
                ]
                ax_diff.legend(handles=legend_elements, loc='lower right')
            
            # Second row: T-statistics and significant channels
            # Plot T-statistics topomap
            ax_t = fig.add_subplot(gs[1, 0:2])  # Span two columns
            im_t, _ = mne.viz.plot_topomap(t_values_filtered, info, show=False, axes=ax_t,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_t.set_title('T-statistics (LMM) - Inclusion vs Exclusion', fontsize=14, fontweight='bold')
            
            # Add text showing T-statistic range
            t_min, t_max = np.nanmin(t_values_filtered), np.nanmax(t_values_filtered)
            ax_t.text(0.02, 0.98, f'T-range: {t_min:.3f} to {t_max:.3f}',
                     transform=ax_t.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar for T-statistics
            cbar_t = plt.colorbar(im_t, ax=ax_t, shrink=0.8)
            cbar_t.set_label('T-statistic', fontsize=12)
            
            # Add legend for T-statistics
            legend_elements = [
                Patch(facecolor='red', label='Inclusion > Exclusion'),
                Patch(facecolor='blue', label='Exclusion > Inclusion')
            ]
            ax_t.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            # Plot significant channels
            ax_sig = fig.add_subplot(gs[1, 2])
            # Create masked T-statistics for significant channels only
            sig_indices = [ch_indices[i] for i, ch in enumerate(available_channels) 
                          if ch in ch_names and significant[ch_names.index(ch)]]
            
            t_masked = np.zeros_like(t_values_filtered)
            if sig_indices:
                sig_mask = np.zeros(len(available_channels), dtype=bool)
                for i, ch in enumerate(available_channels):
                    if ch in ch_names and significant[ch_names.index(ch)]:
                        sig_mask[i] = True
                        t_masked[i] = t_values_filtered[i]
            
            im_sig, _ = mne.viz.plot_topomap(t_masked, info, show=False, axes=ax_sig,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_sig.set_title(f'Significant Channels (FDR α = {fdr_alpha})')
            
            # Add text showing number of significant channels
            n_sig = results['n_significant']
            ax_sig.text(0.02, 0.98, f'Significant: {n_sig}/{len(ch_names)}',
                     transform=ax_sig.transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar for significant channels
            cbar_sig = plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
            cbar_sig.set_label('T-statistic\n(significant only)')
    
    except Exception as e:
        print(f"Could not create topoplots: {e}")
        # Create simple bar plots instead
        # First row: Raw values
        if 'df_marker' in results:
            df_m = results['df_marker']
            means = df_m.groupby(['inclusion_exclusion_label', 'channel'])['mean'].mean().unstack(level=0)
            
            if 'inclusion' in means.columns and 'exclusion' in means.columns:
                inclusion_values = means['inclusion'].reindex(ch_names).values
                exclusion_values = means['exclusion'].reindex(ch_names).values
                
                ax_inclusion = fig.add_subplot(gs[0, 0])
                ax_inclusion.bar(range(len(inclusion_values)), inclusion_values, color='green')
                ax_inclusion.set_title(f'{condition_high} (OFF-task) Raw Values')
                ax_inclusion.set_xlabel('Channel Index')
                ax_inclusion.set_ylabel(f'{marker} value')
                
                ax_exclusion = fig.add_subplot(gs[0, 1])
                ax_exclusion.bar(range(len(exclusion_values)), exclusion_values, color='orange')
                ax_exclusion.set_title(f'{condition_low} (OFF-task) Raw Values')
                ax_exclusion.set_xlabel('Channel Index')
                ax_exclusion.set_ylabel(f'{marker} value')
                
                ax_diff = fig.add_subplot(gs[0, 2])
                diff_values = inclusion_values - exclusion_values
                ax_diff.bar(range(len(diff_values)), diff_values,
                       color=['green' if d > 0 else 'orange' for d in diff_values])
                ax_diff.set_title(f'Difference ({condition_high} minus {condition_low})')
                ax_diff.set_xlabel('Channel Index')
                ax_diff.set_ylabel(f'Difference in {marker} value')
                ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Second row: T-statistics
        ax_t = fig.add_subplot(gs[1, 0:2])
        bars = ax_t.bar(range(len(t_values)), t_values, 
                       color=['green' if t > 0 else 'orange' for t in t_values])
        ax_t.set_title('T-statistics by Channel', fontsize=14, fontweight='bold')
        ax_t.set_xlabel('Channel Index')
        ax_t.set_ylabel('T-statistic')
        ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Highlight significant channels
        for i, sig in enumerate(significant):
            if sig:
                bars[i].set_edgecolor('yellow')
                bars[i].set_linewidth(3)
        
        ax_sig = fig.add_subplot(gs[1, 2])
        ax_sig.text(0.5, 0.5, 'Topoplot not available\n(channel positioning issue)', 
                ha='center', va='center', transform=ax_sig.transAxes)
        ax_sig.set_title('Significant Channels')
    
    # Third row: Summary statistics
    ax_stats = fig.add_subplot(gs[2, :])
    
    if results['n_significant'] > 0:
        # Create summary table for significant channels
        sig_indices = np.where(significant)[0]
        summary_data = []
        
        for idx in sig_indices[:10]:  # Show top 10 most significant
            ch = ch_names[idx]
            summary_data.append({
                'Channel': ch,
                'T-statistic': f"{t_values[idx]:.3f}",
                'P-value': f"{results['p_values'][idx]:.4f}",
                'P-corrected': f"{p_corrected[idx]:.4f}",
                'Coefficient': f"{results['coef_values'][idx]:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Plot as table
        ax_stats.axis('tight')
        ax_stats.axis('off')
        table = ax_stats.table(cellText=summary_df.values,
                         colLabels=summary_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax_stats.set_title(f'Top Significant Channels - Inclusion vs Exclusion (FDR α = {fdr_alpha})', pad=20)
        
    else:
        ax_stats.text(0.5, 0.5, f'No significant channels found (FDR α = {fdr_alpha})', 
                ha='center', va='center', transform=ax_stats.transAxes, fontsize=14)
        ax_stats.set_title('LMM Results Summary - Inclusion vs Exclusion')
        ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   LMM results plot saved to: {save_path}")
    
    return fig


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("🧠 LMM ANALYSIS - OFF TASK INCLUSION VS EXCLUSION COMPARISON 🧠")
    
    # Check if required files exist
    if not os.path.exists(CSV_FILE):
        print(f"CSV not found: {CSV_FILE}")
        sys.exit(1)
    
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading metadata...")
    subject_task_to_condition = load_metadata(METADATA_FILE)
    
    print("Loading data...")
    
    # Check file size before loading
    file_size_mb = os.path.getsize(CSV_FILE) / (1024 * 1024)
    print(f"CSV file size: {file_size_mb:.1f} MB")
    
    # Load data (with chunking for large files if needed)
    if file_size_mb > 500:
        print("Large file detected. Loading with chunking...")
        chunks = pd.read_csv(CSV_FILE, chunksize=100000)
        df_list = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            # Filter for OFF conditions, Sart2/Sart4 tasks, and subjects with condition info
            chunk_off = chunk[chunk['onoff_label'] == 'low']
            chunk_off = chunk_off[chunk_off['task'].isin(TASKS_TO_INCLUDE)]
            chunk_off = chunk_off[chunk_off.apply(
                lambda row: (row['subject_id'], row['task']) in subject_task_to_condition, axis=1
            )]
            
            if TEST_MARKERS:
                chunk_off = chunk_off[chunk_off['marker'].isin(TEST_MARKERS)]
            
            if not chunk_off.empty:
                df_list.append(chunk_off)
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            print(f"Loaded data with shape: {df.shape}")
        else:
            print("No data found after filtering!")
            sys.exit(1)
    else:
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded data with shape: {df.shape}")
    
    # Filter and merge data
    df_filtered = filter_and_merge_data(df, subject_task_to_condition)
    
    if df_filtered.empty:
        print("No data remaining after filtering and merging!")
        sys.exit(1)
    
    # Print column names and basic info
    print(f"Available columns: {df_filtered.columns.tolist()}")
    print(f"Available conditions: {sorted(df_filtered['inclusion_exclusion_label'].unique())}")
    
    markers = sorted(df_filtered['marker'].unique())
    print(f"Available markers: {markers}")
    
    # Determine which markers to process
    if TEST_MARKERS:
        markers_to_process = [m for m in TEST_MARKERS if m in markers]
        if not markers_to_process:
            print("None of the test markers found in data!")
            sys.exit(1)
        print(f"Processing test markers: {markers_to_process}")
    else:
        markers_to_process = markers
        print(f"Processing all {len(markers_to_process)} markers")

    # Display current configuration
    print(f"\nAnalysis Configuration:")
    print(f"  Comparison: Inclusion vs Exclusion (OFF task conditions only)")
    print(f"  Tasks included: {TASKS_TO_INCLUDE}")
    print(f"  Formula: {BASE_FORMULA}")
    print(f"  Random effects group: {RANDOM_EFFECTS_GROUP}")
    print(f"  FDR alpha: {FDR_ALPHA}")
    print(f"  FDR method: {FDR_METHOD}")

    all_results = []
    for marker in markers_to_process:
        try:
            print(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            
            # Run LMM analysis with FDR correction
            results = run_lmm_fdr_analysis(
                df_filtered, marker, 
                formula=BASE_FORMULA, 
                group_col=RANDOM_EFFECTS_GROUP,
                fdr_alpha=FDR_ALPHA, 
                fdr_method=FDR_METHOD
            )
            
            if results is None:
                print(f"No results for marker {marker} - skipping")
                continue
                
            all_results.append(results)
            
            # Save results
            results_file = os.path.join(OUT_DIR, f"{marker}_inclusion_exclusion_results.csv")
            n_sig_saved = save_lmm_results(results, results_file)
            
            # Create plot
            plot_path = os.path.join(OUT_DIR, f"{marker}_inclusion_exclusion_plot.png")
            
            try:
                fig = plot_lmm_results(
                    results,
                    condition_high='Inclusion',
                    condition_low='Exclusion',
                    save_path=plot_path
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"ERROR creating plot for marker {marker}: {e}")
                # Try a simpler fallback plot
                try:
                    plt.figure(figsize=(12, 8))
                    
                    # T-statistics
                    plt.subplot(2, 2, 1)
                    bars = plt.bar(range(len(results['t_values'])), results['t_values'])
                    for i, sig in enumerate(results['significant']):
                        if sig:
                            bars[i].set_color('red')
                    plt.axhline(y=0, color='k', linestyle='-')
                    plt.title(f"T-statistics for {marker} (Inclusion vs Exclusion)")
                    plt.xlabel("Channel index")
                    plt.ylabel("T-value")
                    
                    # P-values
                    plt.subplot(2, 2, 2)
                    plt.bar(range(len(results['p_corrected'])), -np.log10(results['p_corrected']))
                    plt.axhline(y=-np.log10(FDR_ALPHA), color='r', linestyle='--', label=f'FDR α={FDR_ALPHA}')
                    plt.title(f"FDR-corrected -log10(p) for {marker}")
                    plt.xlabel("Channel index")
                    plt.ylabel("-log10(p-value)")
                    plt.legend()
                    
                    # Coefficients
                    plt.subplot(2, 2, 3)
                    bars = plt.bar(range(len(results['coef_values'])), results['coef_values'])
                    for i, sig in enumerate(results['significant']):
                        if sig:
                            bars[i].set_color('red')
                    plt.axhline(y=0, color='k', linestyle='-')
                    plt.title(f"Coefficients for {marker}")
                    plt.xlabel("Channel index")
                    plt.ylabel("Coefficient")
                    
                    # Summary text
                    plt.subplot(2, 2, 4)
                    plt.text(0.1, 0.7, f"Marker: {marker}", fontsize=12, fontweight='bold')
                    plt.text(0.1, 0.6, f"Comparison: Inclusion vs Exclusion (OFF task)", fontsize=10)
                    plt.text(0.1, 0.5, f"Significant channels: {results['n_significant']}/{len(results['ch_names'])}", fontsize=10)
                    plt.text(0.1, 0.4, f"FDR α: {results['fdr_alpha']}", fontsize=10)
                    plt.text(0.1, 0.3, f"FDR method: {results['fdr_method']}", fontsize=10)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   → fallback plot saved to {plot_path}")
                    
                except Exception as e2:
                    print(f"ERROR creating fallback plot for marker {marker}: {e2}")

        except Exception as e:
            print(f"ERROR processing marker {marker}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY - INCLUSION VS EXCLUSION (OFF TASK)")
    print("="*60)
    
    sig_total = 0
    for results in all_results:
        sig = results['n_significant']
        sig_total += sig
        print(f"{results['marker']:>15}: {sig:>3} significant channel(s) (FDR α={FDR_ALPHA})")
    
    print("-" * 60)
    print(f"{'Total markers analyzed:':>25} {len(all_results)}")
    print(f"{'Total significant channels:':>25} {sig_total}")
    print(f"{'Comparison:':>25} Inclusion vs Exclusion (OFF task only)")
    print(f"{'Tasks included:':>25} {', '.join(TASKS_TO_INCLUDE)}")
    print(f"{'Results saved to:':>25} {OUT_DIR}")
    print(f"{'Formula used:':>25} {BASE_FORMULA}")
    print(f"{'FDR correction:':>25} {FDR_METHOD} (α={FDR_ALPHA})")
    print()


if __name__ == '__main__':
    main()