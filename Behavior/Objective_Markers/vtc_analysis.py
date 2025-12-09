#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variance Time Course (VTC) Analysis

Implements the VTC methodology from Esterman et al. (2013) with adaptations
for tasks with thought probes (block-wise smoothing).

VTC Methodology:
1. Z-score RTs within-subject (all valid RTs)
2. Calculate VTC_raw = |RT_z - mean(RT_z)| = |RT_z| (since mean of z-scores is 0)
3. Interpolate missing values (errors, omissions) linearly
4. Smooth with Gaussian kernel (FWHM=3 trials) WITHIN each block (probe)
5. Classify trials as In-the-Zone (VTC < median) or Out-of-the-Zone (VTC > median)

Author: Analysis Assistant
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# ROOT = "/network/iss/"
ROOT = "/Volumes/"
BIDS_RAW_ROOT = ROOT + "cenir/analyse/meeg/CYBERSART/BIDS/raw"
OUTPUT_DIR = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/vtc_analysis"

SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
            "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
            "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]

# VTC parameters (Esterman et al., 2013 adapted for thought probes)
FWHM_TRIALS = 3  # Full-Width at Half-Maximum for Gaussian kernel
SIGMA = FWHM_TRIALS / 2.355  # Convert FWHM to sigma for gaussian_filter1d

# Visualization settings
GENERATE_DIAGNOSTIC_PLOTS = True  # Set to True to generate per-subject/task plots
N_EXAMPLE_PLOTS = 12  # Number of example subject-task combinations to plot


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_trial_data(bids_root: str, subject_id: str, task: str) -> pd.DataFrame:
    """
    Load trial-level behavioral data for a subject/task.
    
    Parameters
    ----------
    bids_root : str
        Path to BIDS root directory
    subject_id : str
        Subject identifier (e.g., '02')
    task : str
        Task name (e.g., 'Sart1')
        
    Returns
    -------
    pd.DataFrame
        Trial-level behavioral data
    """
    beh_file = Path(bids_root) / f"sub-{subject_id}" / "beh" / f"sub-{subject_id}_task-{task}.csv"
    
    if not beh_file.exists():
        return pd.DataFrame()
    
    return pd.read_csv(beh_file)


def zscore_rt_within_subject(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize RTs within each subject using all valid RTs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with 'rt' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'rt_z' column
    """
    df = df.copy()
    
    # Get valid RTs (non-NaN)
    valid_rt_mask = df['rt'].notna()
    
    if valid_rt_mask.sum() < 2:
        df['rt_z'] = np.nan
        return df
    
    # Calculate mean and std from all valid RTs
    rt_mean = df.loc[valid_rt_mask, 'rt'].mean()
    rt_std = df.loc[valid_rt_mask, 'rt'].std()
    
    # Z-score (only for valid RTs, leave NaN for missing)
    df['rt_z'] = np.nan
    if rt_std > 0:
        df.loc[valid_rt_mask, 'rt_z'] = (df.loc[valid_rt_mask, 'rt'] - rt_mean) / rt_std
    
    return df


def calculate_vtc_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate raw VTC as absolute deviation from mean RT.
    
    VTC_raw = |RT_z - mean(RT_z)| = |RT_z| (since mean of z-scores is ~0)
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with 'rt_z' column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'vtc_raw' column
    """
    df = df.copy()
    df['vtc_raw'] = df['rt_z'].abs()
    return df


def interpolate_missing_vtc(vtc_values: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate missing VTC values (NaN).
    
    For trials without valid RTs (omissions, errors), interpolate
    based on adjacent trials.
    
    Parameters
    ----------
    vtc_values : np.ndarray
        Array of VTC values with potential NaNs
        
    Returns
    -------
    np.ndarray
        Array with NaNs replaced by interpolated values
    """
    vtc_interp = vtc_values.copy()
    n = len(vtc_interp)
    
    if n == 0:
        return vtc_interp
    
    # Find indices of valid (non-NaN) values
    valid_mask = ~np.isnan(vtc_interp)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        # All NaN - return as is
        return vtc_interp
    
    if len(valid_indices) == n:
        # No NaNs - return as is
        return vtc_interp
    
    # Use numpy interp for linear interpolation
    # For edge cases (NaN at start/end), extrapolate using nearest valid value
    all_indices = np.arange(n)
    valid_values = vtc_interp[valid_mask]
    
    # Interpolate
    vtc_interp = np.interp(all_indices, valid_indices, valid_values)
    
    return vtc_interp


def smooth_vtc_within_block(vtc_values: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to VTC values.
    
    Parameters
    ----------
    vtc_values : np.ndarray
        Array of VTC values (should be interpolated, no NaNs)
    sigma : float
        Standard deviation for Gaussian kernel
        
    Returns
    -------
    np.ndarray
        Smoothed VTC values
    """
    if len(vtc_values) == 0:
        return vtc_values
    
    # Handle case where all values are NaN
    if np.all(np.isnan(vtc_values)):
        return vtc_values
    
    # Apply Gaussian filter
    # mode='nearest' handles edge effects by padding with nearest values
    vtc_smoothed = gaussian_filter1d(vtc_values, sigma=sigma, mode='nearest')
    
    return vtc_smoothed


def calculate_vtc_and_zones(df: pd.DataFrame, 
                            fwhm: float = FWHM_TRIALS,
                            smooth_within_block: bool = True) -> pd.DataFrame:
    """
    Calculate VTC and classify trials into In-the-Zone vs Out-of-the-Zone.
    
    Implements Esterman et al. (2013) methodology with adaptation for
    thought probes (block-wise smoothing).
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial-level behavioral data with columns:
        - subject: Subject ID
        - sart: Task name
        - probe_number: Block/probe identifier
        - trial_number: Trial order
        - rt: Reaction time (can be NaN for omissions)
        - correct: Whether trial was correct
        - trial_class: 'go' or 'nogo'
        
    fwhm : float
        Full-Width at Half-Maximum for Gaussian kernel (in trials)
        Default: 3 trials
        
    smooth_within_block : bool
        If True, smooth VTC within each probe block (recommended for tasks
        with thought probes). If False, smooth across entire task.
        Default: True
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns:
        - rt_z: Z-scored RT (within-subject)
        - vtc_raw: Raw VTC (absolute deviation)
        - vtc_interpolated: VTC with missing values interpolated
        - vtc_smoothed: Smoothed VTC
        - zone_state: 'in_zone' or 'out_zone'
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Convert FWHM to sigma
    sigma = fwhm / 2.355
    
    # Step 1: Z-score RTs within subject
    df = zscore_rt_within_subject(df)
    
    # Step 2: Calculate raw VTC
    df = calculate_vtc_raw(df)
    
    # Step 3 & 4: Interpolate and smooth (within blocks or across task)
    df['vtc_interpolated'] = np.nan
    df['vtc_smoothed'] = np.nan
    
    if smooth_within_block:
        # Process each probe block separately
        for probe_num in df['probe_number'].dropna().unique():
            block_mask = df['probe_number'] == probe_num
            block_indices = df.index[block_mask]
            
            if len(block_indices) == 0:
                continue
            
            # Get VTC values for this block
            vtc_block = df.loc[block_indices, 'vtc_raw'].values
            
            # Interpolate missing values
            vtc_interp = interpolate_missing_vtc(vtc_block)
            df.loc[block_indices, 'vtc_interpolated'] = vtc_interp
            
            # Smooth within block
            vtc_smooth = smooth_vtc_within_block(vtc_interp, sigma)
            df.loc[block_indices, 'vtc_smoothed'] = vtc_smooth
    else:
        # Smooth across entire task (original Esterman method)
        vtc_all = df['vtc_raw'].values
        vtc_interp = interpolate_missing_vtc(vtc_all)
        df['vtc_interpolated'] = vtc_interp
        vtc_smooth = smooth_vtc_within_block(vtc_interp, sigma)
        df['vtc_smoothed'] = vtc_smooth
    
    # Step 5: Classify into zones based on median split
    vtc_median = df['vtc_smoothed'].median()
    
    df['zone_state'] = np.where(
        df['vtc_smoothed'] < vtc_median,
        'in_zone',
        'out_zone'
    )
    
    # Handle NaN cases
    df.loc[df['vtc_smoothed'].isna(), 'zone_state'] = np.nan
    
    # Store the median for reference
    df['vtc_median'] = vtc_median
    
    return df


def process_subject_all_tasks(bids_root: str, 
                              subject_id: str, 
                              tasks: list,
                              fwhm: float = FWHM_TRIALS) -> pd.DataFrame:
    """
    Process VTC analysis for a single subject across ALL tasks combined.
    
    This merges all 4 SARTs into one continuous session so that:
    1. Z-scoring is done across the entire session (~1800 trials)
    2. A single median is computed for the whole session
    3. Time-on-task effects across the full ~24 min session are preserved
    
    Parameters
    ----------
    bids_root : str
        Path to BIDS root directory
    subject_id : str
        Subject identifier
    tasks : list
        List of task names to process
    fwhm : float
        FWHM for Gaussian smoothing
        
    Returns
    -------
    pd.DataFrame
        Trial-level data with VTC columns added, all tasks combined
    """
    # Load all tasks for this subject
    task_dfs = []
    cumulative_trial_offset = 0
    
    for task in tasks:
        df = load_trial_data(bids_root, subject_id, task)
        if df.empty:
            continue
        
        # Add global trial index across all tasks
        df['global_trial_index'] = df['trial_number'] + cumulative_trial_offset
        cumulative_trial_offset += len(df)
        
        task_dfs.append(df)
    
    if not task_dfs:
        return pd.DataFrame()
    
    # Combine all tasks into one continuous session
    df_combined = pd.concat(task_dfs, ignore_index=True)
    
    # Now apply VTC analysis across the entire session
    df_combined = calculate_vtc_and_zones(df_combined, fwhm=fwhm, smooth_within_block=True)
    
    return df_combined


def plot_vtc_diagnostic_session(df: pd.DataFrame, subject_id: str, 
                                 output_dir: Path) -> None:
    """
    Create diagnostic visualization of VTC analysis for a full session (all 4 SARTs).
    
    Replicates the style from Esterman et al. (2013) Figure 2A showing:
    - Raw VTC (gray)
    - Smoothed VTC with zone coloring (blue=in_zone, orange=out_zone)
    - Error markers (lapses/correct omissions)
    - Median threshold line
    - SART boundaries
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with VTC columns for full session (all tasks)
    subject_id : str
        Subject identifier
    output_dir : Path
        Directory to save the plot
    """
    if df.empty or 'vtc_smoothed' not in df.columns:
        return
    
    # Create figure - wider for full session
    fig, ax = plt.subplots(figsize=(18, 5))
    
    # Trial indices (x-axis)
    trial_idx = np.arange(len(df))
    
    # Convert to time in minutes (assuming ~800ms per trial)
    time_mins = trial_idx * 0.8 / 60  # Approximate time in minutes
    
    # Get VTC values
    vtc_raw = df['vtc_raw'].values
    vtc_smoothed = df['vtc_smoothed'].values
    vtc_median = df['vtc_median'].iloc[0] if 'vtc_median' in df.columns else np.nanmedian(vtc_smoothed)
    
    # Plot raw VTC (gray, thin)
    ax.plot(time_mins, vtc_raw, color='lightgray', linewidth=0.5, alpha=0.7, 
            label='Raw VTC')
    
    # Plot smoothed VTC with zone coloring
    zone_states = df['zone_state'].values
    in_zone_mask = zone_states == 'in_zone'
    out_zone_mask = zone_states == 'out_zone'
    
    # Plot smoothed VTC colored by zone
    ax.plot(time_mins, vtc_smoothed, color='gray', linewidth=1.5, alpha=0.3)
    
    # In-zone segments (blue)
    vtc_in_zone = np.where(in_zone_mask, vtc_smoothed, np.nan)
    ax.plot(time_mins, vtc_in_zone, color='#2E86AB', linewidth=2.5, 
            label='In the zone (smoothed VTC)')
    
    # Out-zone segments (orange)
    vtc_out_zone = np.where(out_zone_mask, vtc_smoothed, np.nan)
    ax.plot(time_mins, vtc_out_zone, color='#F4A261', linewidth=2.5, 
            label='Out of the zone (smoothed VTC)')
    
    # Plot median threshold line
    ax.axhline(y=vtc_median, color='black', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'Median = {vtc_median:.2f}')
    
    # Mark error trials
    y_marker_base = -0.15
    
    if 'trial_class' in df.columns and 'correct' in df.columns:
        # Lapses (commission errors on nogo trials)
        lapse_mask = (df['trial_class'] == 'nogo') & (df['correct'] == False)
        if lapse_mask.any():
            lapse_times = time_mins[lapse_mask]
            ax.scatter(lapse_times, np.full(len(lapse_times), y_marker_base), 
                      marker='s', s=20, color='#2A9D8F', label='Lapse', zorder=5)
        
        # Correct omissions (nogo trials without response)
        correct_omission_mask = (df['trial_class'] == 'nogo') & (df['correct'] == True)
        if correct_omission_mask.any():
            correct_times = time_mins[correct_omission_mask]
            ax.scatter(correct_times, np.full(len(correct_times), y_marker_base - 0.1), 
                      marker='s', s=15, color='#8B4513', label='Correct', zorder=5)
    
    # Add SART boundaries as vertical lines (thicker than probe lines)
    if 'sart' in df.columns:
        sart_changes = df['sart'].ne(df['sart'].shift())
        sart_change_indices = np.where(sart_changes)[0]
        for idx in sart_change_indices[1:]:  # Skip first (start of session)
            t = time_mins[idx]
            ax.axvline(x=t, color='red', linestyle='-', linewidth=2, alpha=0.6)
        
        # Add SART labels at top
        sart_groups = df.groupby('sart').first()
        for sart_name in sart_groups.index:
            sart_df = df[df['sart'] == sart_name]
            start_idx = sart_df.index[0]
            end_idx = sart_df.index[-1]
            mid_time = (time_mins[start_idx] + time_mins[end_idx]) / 2
            ax.text(mid_time, ax.get_ylim()[1] * 0.95, sart_name, 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color='darkred', alpha=0.8)
    
    # Add probe boundaries as vertical lines (lighter)
    if 'probe_number' in df.columns:
        # Create unique probe identifier combining sart and probe_number
        df_temp = df.copy()
        df_temp['probe_id'] = df_temp['sart'] + '_' + df_temp['probe_number'].astype(str)
        probe_changes = df_temp['probe_id'].ne(df_temp['probe_id'].shift())
        probe_change_indices = np.where(probe_changes)[0]
        for idx in probe_change_indices[1:]:
            t = time_mins[idx]
            ax.axvline(x=t, color='gray', linestyle=':', linewidth=0.5, alpha=0.4)
    
    # Formatting
    ax.set_xlabel('Time (mins)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized RT Variability\n[Absolute RT deviance (z-score)]', 
                  fontsize=11, fontweight='bold')
    ax.set_title(f'sub-{subject_id} - Full Session (4 SARTs combined)', 
                 fontsize=14, fontweight='bold')
    
    # Set y-axis limits
    y_max = max(3.5, np.nanmax(vtc_smoothed) * 1.1) if not np.all(np.isnan(vtc_smoothed)) else 3.5
    ax.set_ylim(y_marker_base - 0.2, y_max)
    ax.set_xlim(0, time_mins[-1] if len(time_mins) > 0 else 24)
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    plots_dir = output_dir / "diagnostic_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = plots_dir / f"vtc_sub-{subject_id}_full_session.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def aggregate_vtc_by_probe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VTC metrics at the probe level.
    
    Groups by (sart, probe_number) to handle multiple SARTs per subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data with VTC columns
        
    Returns
    -------
    pd.DataFrame
        Probe-level aggregated VTC metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    probe_data = []
    
    # Group by both sart and probe_number to handle full session data
    for (sart, probe_num), probe_trials in df.groupby(['sart', 'probe_number']):
        if len(probe_trials) == 0:
            continue
        
        # Calculate proportion in each zone
        zone_counts = probe_trials['zone_state'].value_counts()
        n_in_zone = zone_counts.get('in_zone', 0)
        n_out_zone = zone_counts.get('out_zone', 0)
        n_total = n_in_zone + n_out_zone
        
        prop_in_zone = n_in_zone / n_total if n_total > 0 else np.nan
        prop_out_zone = n_out_zone / n_total if n_total > 0 else np.nan
        
        # Mean VTC metrics
        vtc_raw_mean = probe_trials['vtc_raw'].mean()
        vtc_smoothed_mean = probe_trials['vtc_smoothed'].mean()
        
        # Calculate global probe index (1-60 across all 4 SARTs)
        sart_num = int(sart[-1])  # Extract number from 'Sart1', 'Sart2', etc.
        global_probe_index = (sart_num - 1) * 15 + probe_num
        
        probe_data.append({
            'subject': probe_trials['subject'].iloc[0],
            'sart': sart,
            'probe_number': probe_num,
            'global_probe_index': global_probe_index,
            'n_trials': len(probe_trials),
            'n_in_zone': n_in_zone,
            'n_out_zone': n_out_zone,
            'prop_in_zone': prop_in_zone,
            'prop_out_zone': prop_out_zone,
            'vtc_raw_mean': vtc_raw_mean,
            'vtc_smoothed_mean': vtc_smoothed_mean,
            'vtc_median': probe_trials['vtc_median'].iloc[0],
            # Probe ratings (from first trial of block)
            'onoff': probe_trials['onoff'].iloc[0] if 'onoff' in probe_trials.columns else np.nan,
            'valence': probe_trials['valence'].iloc[0] if 'valence' in probe_trials.columns else np.nan,
            'time': probe_trials['time'].iloc[0] if 'time' in probe_trials.columns else np.nan,
            'selfother': probe_trials['selfother'].iloc[0] if 'selfother' in probe_trials.columns else np.nan,
            'confidence': probe_trials['confidence'].iloc[0] if 'confidence' in probe_trials.columns else np.nan,
        })
    
    return pd.DataFrame(probe_data)


def main():
    """
    Main execution function.
    
    Processes VTC analysis for all subjects and tasks, saves both
    trial-level and probe-level results.
    """
    print("=" * 60)
    print("VARIANCE TIME COURSE (VTC) ANALYSIS")
    print("=" * 60)
    print(f"BIDS Root: {BIDS_RAW_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"FWHM: {FWHM_TRIALS} trials (sigma={SIGMA:.3f})")
    print(f"Processing {len(SUBJECTS)} subjects (all 4 SARTs combined per subject)")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_trial_data = []
    all_probe_data = []
    plot_counter = 0
    
    for subject_id in SUBJECTS:
        # Process ALL tasks combined for this subject
        df_session = process_subject_all_tasks(BIDS_RAW_ROOT, subject_id, TASKS, fwhm=FWHM_TRIALS)
        
        if df_session.empty:
            print(f"Warning: No data for sub-{subject_id}")
            continue
        
        all_trial_data.append(df_session)
        
        # Aggregate at probe level (now with unique probe IDs across session)
        df_probe = aggregate_vtc_by_probe(df_session)
        if not df_probe.empty:
            all_probe_data.append(df_probe)
        
        # Generate diagnostic plots for first N examples
        if GENERATE_DIAGNOSTIC_PLOTS and plot_counter < N_EXAMPLE_PLOTS:
            plot_vtc_diagnostic_session(df_session, subject_id, output_path)
            plot_counter += 1
        
        # Count zones for this subject's full session
        zone_counts = df_session['zone_state'].value_counts()
        n_in = zone_counts.get('in_zone', 0)
        n_out = zone_counts.get('out_zone', 0)
        n_total = len(df_session)
        vtc_median = df_session['vtc_median'].iloc[0] if 'vtc_median' in df_session.columns else np.nan
        print(f"Processed sub-{subject_id}: {n_total} trials "
              f"(in_zone={n_in}, out_zone={n_out}, median={vtc_median:.3f})")
    
    if not all_trial_data:
        print("Error: No data was processed!")
        return
    
    # Combine all data
    df_all_trials = pd.concat(all_trial_data, ignore_index=True)
    df_all_probes = pd.concat(all_probe_data, ignore_index=True)
    
    # Save trial-level data
    trial_file = output_path / "vtc_trial_level.csv"
    df_all_trials.to_csv(trial_file, index=False)
    print(f"\nSaved trial-level data: {trial_file}")
    
    # Save probe-level data
    probe_file = output_path / "vtc_probe_level.csv"
    df_all_probes.to_csv(probe_file, index=False)
    print(f"Saved probe-level data: {probe_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total trial observations: {len(df_all_trials)}")
    print(f"Total probe observations: {len(df_all_probes)}")
    print(f"Unique subjects: {df_all_trials['subject'].nunique()}")
    
    # Zone distribution
    zone_dist = df_all_trials['zone_state'].value_counts(normalize=True)
    print(f"\nZone Distribution (trial-level):")
    for zone, prop in zone_dist.items():
        print(f"  {zone}: {prop*100:.1f}%")
    
    # VTC statistics
    print(f"\nVTC Statistics:")
    print(f"  VTC Raw: mean={df_all_trials['vtc_raw'].mean():.3f}, "
          f"std={df_all_trials['vtc_raw'].std():.3f}")
    print(f"  VTC Smoothed: mean={df_all_trials['vtc_smoothed'].mean():.3f}, "
          f"std={df_all_trials['vtc_smoothed'].std():.3f}")
    
    # Probe-level zone proportions
    print(f"\nProbe-level Zone Proportions:")
    print(f"  Prop In-Zone: mean={df_all_probes['prop_in_zone'].mean():.3f}, "
          f"std={df_all_probes['prop_in_zone'].std():.3f}")
    print(f"  Prop Out-Zone: mean={df_all_probes['prop_out_zone'].mean():.3f}, "
          f"std={df_all_probes['prop_out_zone'].std():.3f}")
    
    # Show probe-level variation (this is where the interesting variation is)
    print(f"\nProbe-level Variation (range of prop_out_zone):")
    print(f"  Min: {df_all_probes['prop_out_zone'].min():.3f}")
    print(f"  Max: {df_all_probes['prop_out_zone'].max():.3f}")
    print(f"  Probes with >60% out-zone: {(df_all_probes['prop_out_zone'] > 0.6).sum()}")
    print(f"  Probes with <40% out-zone: {(df_all_probes['prop_out_zone'] < 0.4).sum()}")
    
    # Diagnostic plots info
    if GENERATE_DIAGNOSTIC_PLOTS:
        print(f"\nDiagnostic plots saved to: {output_path / 'diagnostic_plots'}")
        print(f"  Generated {min(plot_counter, N_EXAMPLE_PLOTS)} example plots")
    
    print("\n" + "=" * 60)
    print("VTC ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
