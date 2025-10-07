#!/usr/bin/env python3
"""
LMM Interaction Analysis with FDR Correction - OnOff * Inclusion/Exclusion

This script performs Linear Mixed-Effects Model (LMM) analysis with False Discovery Rate (FDR)
correction to examine the interaction between onoff conditions (on-task vs off-task) and 
inclusion/exclusion conditions.

The model: marker ~ onoff * inclusion_exclusion
This gives us three main effects:
1. Main effect of onoff (on-task vs off-task)
2. Main effect of inclusion/exclusion
3. Interaction effect (onoff * inclusion/exclusion)

Based on the order (IE/EI) in metadata and task (Sart2/Sart4):
- IE order: Sart2 = Inclusion, Sart4 = Exclusion
- EI order: Sart2 = Exclusion, Sart4 = Inclusion

Features:
- Loads probe-level aggregated data (both ON and OFF conditions)
- Filters for Sart2 and Sart4 tasks only
- Assigns inclusion/exclusion labels based on order and task
- Tests interaction between onoff and inclusion/exclusion
- FDR correction for multiple comparisons
- Comprehensive plotting with separate topoplots for each effect
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

OUT_DIR = os.path.join(project_root, 'results/onoff_inclusion_exclusion_interaction_analysis')

# LMM Formula configuration - interaction model
BASE_FORMULA = 'mean ~ onoff_label * inclusion_exclusion_label'  # Full interaction model

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
    Filter data for Sart2/Sart4 tasks and merge with inclusion/exclusion labels (keep both ON and OFF conditions).
    
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
    print("Filtering data for Sart2/Sart4 tasks and merging with condition info...")
    
    # Keep both ON and OFF conditions for interaction analysis
    print(f"Original data shape: {df.shape[0]} rows")
    
    # Filter for Sart2 and Sart4 tasks only
    df_filtered = df[df['task'].isin(TASKS_TO_INCLUDE)].copy()
    print(f"Filtered to Sart2/Sart4 tasks: {df_filtered.shape[0]} rows")
    
    # Add inclusion/exclusion condition information
    df_filtered['inclusion_exclusion'] = df_filtered.apply(
        lambda row: subject_task_to_condition.get((row['subject_id'], row['task']), None), 
        axis=1
    )
    
    # Remove rows without condition information
    before_filter = df_filtered.shape[0]
    df_filtered = df_filtered.dropna(subset=['inclusion_exclusion']).copy()
    after_filter = df_filtered.shape[0]
    
    if before_filter != after_filter:
        print(f"Removed {before_filter - after_filter} rows due to missing condition info")
    
    # Convert variables to categorical for LMM
    df_filtered['inclusion_exclusion_label'] = df_filtered['inclusion_exclusion'].astype('category')
    df_filtered['onoff_label'] = df_filtered['onoff_label'].astype('category')
    
    print(f"Final dataset: {df_filtered.shape[0]} rows")
    print(f"OnOff distribution:")
    print(df_filtered['onoff_label'].value_counts().sort_index())
    print(f"Inclusion/Exclusion distribution:")
    print(df_filtered['inclusion_exclusion_label'].value_counts().sort_index())
    print(f"OnOff x Inclusion/Exclusion crosstab:")
    print(pd.crosstab(df_filtered['onoff_label'], df_filtered['inclusion_exclusion_label'], margins=True))
    print(f"Task distribution:")
    print(df_filtered['task'].value_counts().sort_index())
    
    return df_filtered


# -----------------------------------------------------------------------------
# LMM ANALYSIS FUNCTIONS
# -----------------------------------------------------------------------------

def compute_lmm_stats(df_marker: pd.DataFrame, ch_names, formula=BASE_FORMULA, group_col=RANDOM_EFFECTS_GROUP):
    """
    Compute LMM statistics for each channel.
    
    Returns
    -------
    dict
        Dictionary containing arrays for each effect:
        - onoff_effect: main effect of onoff
        - inclusion_exclusion_effect: main effect of inclusion/exclusion
        - interaction_effect: onoff * inclusion/exclusion interaction
    """
    n_channels = len(ch_names)
    
    # Initialize result arrays for each effect
    results = {
        'onoff_effect': {
            't_values': np.zeros(n_channels),
            'p_values': np.zeros(n_channels),
            'coef_values': np.zeros(n_channels)
        },
        'inclusion_exclusion_effect': {
            't_values': np.zeros(n_channels),
            'p_values': np.zeros(n_channels),
            'coef_values': np.zeros(n_channels)
        },
        'interaction_effect': {
            't_values': np.zeros(n_channels),
            'p_values': np.zeros(n_channels),
            'coef_values': np.zeros(n_channels)
        }
    }
    
    print(f"   Running LMM with formula: {formula}")
    print(f"   Random effects grouped by: {group_col}")
    
    for i, ch in enumerate(ch_names):
        d_ch = df_marker[df_marker['channel'] == ch].copy()
        if d_ch.empty:
            for effect in results.values():
                effect['t_values'][i] = np.nan
                effect['p_values'][i] = np.nan
                effect['coef_values'][i] = np.nan
            continue
            
        # Check if we have all conditions for this channel
        onoff_levels = d_ch['onoff_label'].nunique()
        ie_levels = d_ch['inclusion_exclusion_label'].nunique()
        
        if onoff_levels < 2 or ie_levels < 2:
            print(f"    Warning: Channel {ch} missing conditions (onoff: {onoff_levels}, IE: {ie_levels})")
            for effect in results.values():
                effect['t_values'][i] = np.nan
                effect['p_values'][i] = np.nan
                effect['coef_values'][i] = np.nan
            continue
        
        try:
            # Fit mixed-effects model
            model = mixedlm(formula, d_ch, groups=d_ch[group_col])
            res = model.fit(reml=False)
            
            # Extract effects based on parameter names
            param_names = list(res.params.keys())
            
            # Main effect of onoff (high vs low)
            onoff_param = [p for p in param_names if 'onoff_label' in p and ':' not in p]
            if onoff_param:
                param_name = onoff_param[0]
                results['onoff_effect']['t_values'][i] = res.tvalues[param_name]
                results['onoff_effect']['p_values'][i] = res.pvalues[param_name]
                results['onoff_effect']['coef_values'][i] = res.params[param_name]
            else:
                results['onoff_effect']['t_values'][i] = np.nan
                results['onoff_effect']['p_values'][i] = np.nan
                results['onoff_effect']['coef_values'][i] = np.nan
            
            # Main effect of inclusion/exclusion
            ie_param = [p for p in param_names if 'inclusion_exclusion_label' in p and ':' not in p]
            if ie_param:
                param_name = ie_param[0]
                results['inclusion_exclusion_effect']['t_values'][i] = res.tvalues[param_name]
                results['inclusion_exclusion_effect']['p_values'][i] = res.pvalues[param_name]
                results['inclusion_exclusion_effect']['coef_values'][i] = res.params[param_name]
            else:
                results['inclusion_exclusion_effect']['t_values'][i] = np.nan
                results['inclusion_exclusion_effect']['p_values'][i] = np.nan
                results['inclusion_exclusion_effect']['coef_values'][i] = np.nan
            
            # Interaction effect
            interaction_param = [p for p in param_names if ':' in p and 'onoff_label' in p and 'inclusion_exclusion_label' in p]
            if interaction_param:
                param_name = interaction_param[0]
                results['interaction_effect']['t_values'][i] = res.tvalues[param_name]
                results['interaction_effect']['p_values'][i] = res.pvalues[param_name]
                results['interaction_effect']['coef_values'][i] = res.params[param_name]
            else:
                results['interaction_effect']['t_values'][i] = np.nan
                results['interaction_effect']['p_values'][i] = np.nan
                results['interaction_effect']['coef_values'][i] = np.nan
                
        except Exception as e:
            print(f"    LMM failed for channel {ch}: {e}")
            for effect in results.values():
                effect['t_values'][i] = np.nan
                effect['p_values'][i] = np.nan
                effect['coef_values'][i] = np.nan
    
    return results


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


def run_interaction_analysis(df, marker, formula=BASE_FORMULA, group_col=RANDOM_EFFECTS_GROUP, 
                           fdr_alpha=FDR_ALPHA, fdr_method=FDR_METHOD):
    """
    Run interaction analysis with FDR correction for a single marker.
    
    Parameters
    ----------
    df : DataFrame
        Data for analysis (both ON and OFF conditions)
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
    condition_dist = pd.crosstab(df_m['onoff_label'], df_m['inclusion_exclusion_label'], margins=True)
    print(f"   Condition distribution:")
    print(condition_dist)
    
    if len(df_m['onoff_label'].unique()) < 2 or len(df_m['inclusion_exclusion_label'].unique()) < 2:
        print("   Warning: Less than 2 levels in onoff or inclusion/exclusion - skipping marker")
        return None

    # Run LMM analysis
    lmm_results = compute_lmm_stats(df_m, ch_names, formula, group_col)
    
    # Apply FDR correction to each effect
    effects_results = {}
    for effect_name, effect_data in lmm_results.items():
        rejected, p_corrected = apply_fdr_correction(effect_data['p_values'], fdr_alpha, fdr_method)
        
        n_significant = np.sum(rejected)
        print(f"   {effect_name} - Significant channels (FDR α={fdr_alpha}): {n_significant}/{len(ch_names)}")
        
        if n_significant > 0:
            sig_channels = [ch_names[i] for i in np.where(rejected)[0]]
            print(f"   {effect_name} - Significant channels: {', '.join(sig_channels[:5])}" + 
                  ("..." if len(sig_channels) > 5 else ""))
        
        effects_results[effect_name] = {
            't_values': effect_data['t_values'],
            'p_values': effect_data['p_values'],
            'p_corrected': p_corrected,
            'coef_values': effect_data['coef_values'],
            'significant': rejected,
            'n_significant': n_significant
        }

    return {
        'marker': marker,
        'ch_names': ch_names,
        'effects': effects_results,
        'df_marker': df_m,
        'formula': formula,
        'fdr_alpha': fdr_alpha,
        'fdr_method': fdr_method
    }


def save_interaction_results(results, save_path):
    """
    Save interaction analysis results to CSV files.
    
    Parameters
    ----------
    results : dict
        Results from run_interaction_analysis
    save_path : str
        Base path for saving results
    """
    # Create comprehensive results DataFrame
    all_results = []
    
    for effect_name, effect_data in results['effects'].items():
        effect_df = pd.DataFrame({
            'channel': results['ch_names'],
            'effect': effect_name,
            'coefficient': effect_data['coef_values'],
            't_statistic': effect_data['t_values'],
            'p_value': effect_data['p_values'],
            'p_corrected': effect_data['p_corrected'],
            'significant': effect_data['significant'],
            'marker': results['marker'],
            'formula': results['formula'],
            'fdr_alpha': results['fdr_alpha'],
            'fdr_method': results['fdr_method']
        })
        all_results.append(effect_df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save full results
    full_results_file = save_path.replace('.csv', '_full_results.csv')
    combined_df.to_csv(full_results_file, index=False)
    print(f"   Full results saved to: {full_results_file}")
    
    # Save significant results only
    sig_results = combined_df[combined_df['significant']].copy()
    if len(sig_results) > 0:
        sig_results_file = save_path.replace('.csv', '_significant_results.csv')
        sig_results.to_csv(sig_results_file, index=False)
        print(f"   Significant results saved to: {sig_results_file}")
    
    # Save separate files for each effect
    for effect_name, effect_data in results['effects'].items():
        effect_file = save_path.replace('.csv', f'_{effect_name}_results.csv')
        effect_df = combined_df[combined_df['effect'] == effect_name].copy()
        effect_df.to_csv(effect_file, index=False)
    
    return len(sig_results)


def plot_interaction_results(results, save_path=None, figsize=(20, 15)):
    """
    Plot interaction analysis results with separate topoplots for each effect.
    
    Parameters
    ----------
    results : dict
        Results from run_interaction_analysis
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
    effects = results['effects']
    fdr_alpha = results['fdr_alpha']
    
    # Create figure with a 4x3 grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, height_ratios=[1, 1, 1, 0.5])
    
    # Main title
    fig.suptitle(f'Interaction Analysis: {marker}\n'
                 f'OnOff * Inclusion/Exclusion Effects (FDR α = {fdr_alpha})', 
                 fontsize=18, fontweight='bold')
    
    effect_names = ['onoff_effect', 'inclusion_exclusion_effect', 'interaction_effect']
    effect_titles = ['Main Effect: OnOff (High vs Low)', 'Main Effect: Inclusion vs Exclusion', 'Interaction: OnOff × I/E']
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 0:
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Get indices for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            
            # Plot each effect in a separate row
            for row, (effect_name, effect_title) in enumerate(zip(effect_names, effect_titles)):
                effect_data = effects[effect_name]
                
                # Filter data for available channels
                t_values_filtered = effect_data['t_values'][ch_indices]
                
                # T-statistics topomap (spans 2 columns)
                ax_t = fig.add_subplot(gs[row, 0:2])
                im_t, _ = mne.viz.plot_topomap(t_values_filtered, info, show=False, axes=ax_t,
                                              cmap='RdBu_r', contours=6, sensors=True,
                                              names=available_channels, outlines='head')
                ax_t.set_title(f'{effect_title}\nT-statistics', fontsize=12, fontweight='bold')
                
                # Add text showing T-statistic range
                t_min, t_max = np.nanmin(t_values_filtered), np.nanmax(t_values_filtered)
                ax_t.text(0.02, 0.98, f'T-range: {t_min:.3f} to {t_max:.3f}',
                         transform=ax_t.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add colorbar
                cbar_t = plt.colorbar(im_t, ax=ax_t, shrink=0.8)
                cbar_t.set_label('T-statistic', fontsize=10)
                
                # Plot significant channels
                ax_sig = fig.add_subplot(gs[row, 2])
                
                # Create masked T-statistics for significant channels only
                t_masked = np.zeros_like(t_values_filtered)
                sig_mask = np.zeros(len(available_channels), dtype=bool)
                
                for i, ch in enumerate(available_channels):
                    if ch in ch_names and effect_data['significant'][ch_names.index(ch)]:
                        sig_mask[i] = True
                        t_masked[i] = t_values_filtered[i]
                
                im_sig, _ = mne.viz.plot_topomap(t_masked, info, show=False, axes=ax_sig,
                                              cmap='RdBu_r', contours=6, sensors=True,
                                              names=available_channels, outlines='head')
                ax_sig.set_title(f'Significant Channels\n(n={effect_data["n_significant"]})', fontsize=10)
                
                # Add text showing number of significant channels
                ax_sig.text(0.02, 0.98, f'{effect_data["n_significant"]}/{len(ch_names)}',
                         transform=ax_sig.transAxes, fontsize=8, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add colorbar for significant channels
                cbar_sig = plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
                cbar_sig.set_label('T-stat (sig)', fontsize=8)
    
    except Exception as e:
        print(f"Could not create topoplots: {e}")
        # Create bar plots as fallback
        for row, (effect_name, effect_title) in enumerate(zip(effect_names, effect_titles)):
            effect_data = effects[effect_name]
            
            # T-statistics bar plot
            ax_t = fig.add_subplot(gs[row, 0:2])
            bars = ax_t.bar(range(len(effect_data['t_values'])), effect_data['t_values'], 
                           color=['red' if t > 0 else 'blue' for t in effect_data['t_values']])
            ax_t.set_title(f'{effect_title} - T-statistics', fontsize=12, fontweight='bold')
            ax_t.set_xlabel('Channel Index')
            ax_t.set_ylabel('T-statistic')
            ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
            
            # Highlight significant channels
            for i, sig in enumerate(effect_data['significant']):
                if sig:
                    bars[i].set_edgecolor('yellow')
                    bars[i].set_linewidth(3)
            
            # Summary for this effect
            ax_sig = fig.add_subplot(gs[row, 2])
            ax_sig.text(0.1, 0.7, f"{effect_title}", fontsize=10, fontweight='bold', 
                       transform=ax_sig.transAxes, wrap=True)
            ax_sig.text(0.1, 0.5, f"Significant: {effect_data['n_significant']}/{len(ch_names)}", 
                       fontsize=9, transform=ax_sig.transAxes)
            ax_sig.text(0.1, 0.3, f"Max |T|: {np.nanmax(np.abs(effect_data['t_values'])):.3f}", 
                       fontsize=9, transform=ax_sig.transAxes)
            ax_sig.axis('off')
    
    # Fourth row: Summary table
    ax_summary = fig.add_subplot(gs[3, :])
    
    # Create summary of all effects
    summary_data = []
    for effect_name, effect_title in zip(effect_names, effect_titles):
        effect_data = effects[effect_name]
        summary_data.append({
            'Effect': effect_title,
            'Significant Channels': f"{effect_data['n_significant']}/{len(ch_names)}",
            'Max |T-statistic|': f"{np.nanmax(np.abs(effect_data['t_values'])):.3f}",
            'Max |Coefficient|': f"{np.nanmax(np.abs(effect_data['coef_values'])):.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Plot as table
    ax_summary.axis('tight')
    ax_summary.axis('off')
    table = ax_summary.table(cellText=summary_df.values,
                            colLabels=summary_df.columns,
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax_summary.set_title(f'Summary of Effects for {marker} (FDR α = {fdr_alpha})', pad=20, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Interaction plot saved to: {save_path}")
    
    return fig


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("🧠 INTERACTION ANALYSIS - ONOFF * INCLUSION/EXCLUSION 🧠")
    
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
            # Filter for Sart2/Sart4 tasks and subjects with condition info
            chunk_filtered = chunk[chunk['task'].isin(TASKS_TO_INCLUDE)]
            chunk_filtered = chunk_filtered[chunk_filtered.apply(
                lambda row: (row['subject_id'], row['task']) in subject_task_to_condition, axis=1
            )]
            
            if TEST_MARKERS:
                chunk_filtered = chunk_filtered[chunk_filtered['marker'].isin(TEST_MARKERS)]
            
            if not chunk_filtered.empty:
                df_list.append(chunk_filtered)
        
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
    print(f"Available onoff labels: {sorted(df_filtered['onoff_label'].unique())}")
    print(f"Available inclusion/exclusion labels: {sorted(df_filtered['inclusion_exclusion_label'].unique())}")
    
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
    print(f"  Model: OnOff * Inclusion/Exclusion interaction")
    print(f"  Tasks included: {TASKS_TO_INCLUDE}")
    print(f"  Formula: {BASE_FORMULA}")
    print(f"  Random effects group: {RANDOM_EFFECTS_GROUP}")
    print(f"  FDR alpha: {FDR_ALPHA}")
    print(f"  FDR method: {FDR_METHOD}")

    all_results = []
    for marker in markers_to_process:
        try:
            print(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            
            # Run interaction analysis
            results = run_interaction_analysis(
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
            results_file = os.path.join(OUT_DIR, f"{marker}_onoff_ie_interaction_results.csv")
            n_sig_saved = save_interaction_results(results, results_file)
            
            # Create plot
            plot_path = os.path.join(OUT_DIR, f"{marker}_onoff_ie_interaction_plot.png")
            
            try:
                fig = plot_interaction_results(
                    results,
                    save_path=plot_path
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"ERROR creating plot for marker {marker}: {e}")

        except Exception as e:
            print(f"ERROR processing marker {marker}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("INTERACTION ANALYSIS SUMMARY - ONOFF * INCLUSION/EXCLUSION")
    print("="*60)
    
    for results in all_results:
        print(f"\nMarker: {results['marker']}")
        for effect_name, effect_data in results['effects'].items():
            sig = effect_data['n_significant']
            print(f"  {effect_name:>25}: {sig:>3} significant channel(s)")
    
    print("-" * 60)
    print(f"{'Total markers analyzed:':>25} {len(all_results)}")
    print(f"{'Model:':>25} OnOff * Inclusion/Exclusion interaction")
    print(f"{'Tasks included:':>25} {', '.join(TASKS_TO_INCLUDE)}")
    print(f"{'Results saved to:':>25} {OUT_DIR}")
    print(f"{'Formula used:':>25} {BASE_FORMULA}")
    print(f"{'FDR correction:':>25} {FDR_METHOD} (α={FDR_ALPHA})")
    print()


if __name__ == '__main__':
    main()