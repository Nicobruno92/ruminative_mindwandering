#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Objective Markers by Probe

Aggregates trial-level SART behavioral data into probe-level objective markers:
- Omission rate: proportion of missed go trials
- Commission rate: proportion of false alarms on nogo trials  
- RTCV: Reaction time coefficient of variation (RT_SD / RT_Mean)

Can optionally filter to only include N trials back from probe onset.

Author: Analysis Assistant
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT = "<PATH_TO_YOUR_CLUSTER_MOUNT_ROOT>/"
# ROOT = "/Volumes/"
BIDS_RAW_ROOT = ROOT + "cenir/analyse/meeg/CYBERSART/BIDS/raw"
OUTPUT_DIR = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/objective_markers_10"

SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
            "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
            "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]

# Number of trials back from probe onset to include
# Set to None to use all trials in the window
N_BACK_TO_PROBE = 10  # e.g., 10 means only last 10 trials before each probe

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


def aggregate_by_probe(df_trials: pd.DataFrame, n_back: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate trial-level data into probe-level objective markers.
    
    Parameters
    ----------
    df_trials : pd.DataFrame
        Trial-level behavioral data
    n_back : int, optional
        Number of trials back from probe onset to include.
        If None, uses all trials in the window.
        
    Returns
    -------
    pd.DataFrame
        Probe-level aggregated objective markers
    """
    if df_trials.empty:
        return pd.DataFrame()
    
    probe_data = []
    
    for probe_num in df_trials['probe_number'].dropna().unique():
        # Get trials for this probe
        probe_trials = df_trials[df_trials['probe_number'] == probe_num].copy()
        
        if len(probe_trials) == 0:
            continue
        
        # Filter to N trials back from probe if specified
        if n_back is not None:
            # distance_to_probe is negative, counting down to probe
            # e.g., -10, -9, -8, ..., -1 (where -1 is closest to probe)
            # We want the last N trials, so distance >= -n_back
            probe_trials = probe_trials[probe_trials['distance_to_probe'] >= -n_back]
        
        if len(probe_trials) == 0:
            continue
        
        # Separate go and nogo trials
        go_trials = probe_trials[probe_trials['trial_class'] == 'go']
        nogo_trials = probe_trials[probe_trials['trial_class'] == 'nogo']
        
        # Calculate omission rate (missed go trials)
        n_go = len(go_trials)
        n_omissions = len(go_trials[go_trials['response'] == False]) if n_go > 0 else 0
        omission_rate = n_omissions / n_go if n_go > 0 else np.nan
        
        # Calculate commission rate (false alarms on nogo trials)
        n_nogo = len(nogo_trials)
        n_commissions = len(nogo_trials[nogo_trials['response'] == True]) if n_nogo > 0 else 0
        commission_rate = n_commissions / n_nogo if n_nogo > 0 else np.nan
        
        # Calculate RTCV (RT coefficient of variation)
        # Only for trials with valid RTs (i.e., responses made)
        valid_rts = probe_trials[probe_trials['rt'].notna()]['rt']
        rt_mean = valid_rts.mean() if len(valid_rts) > 0 else np.nan
        rt_sd = valid_rts.std() if len(valid_rts) > 1 else np.nan
        rt_n = len(valid_rts)
        rtcv = rt_sd / rt_mean if rt_mean > 0 and not np.isnan(rt_sd) else np.nan
        
        probe_data.append({
            'subject': df_trials['subject'].iloc[0],
            'sart': df_trials['sart'].iloc[0],
            'probe_number': probe_num,
            'n_trials_window': len(probe_trials),
            'n_go': n_go,
            'n_nogo': n_nogo,
            'n_omissions': n_omissions,
            'n_commissions': n_commissions,
            'omission_rate': omission_rate,
            'commission_rate': commission_rate,
            'rt_mean': rt_mean,
            'rt_sd': rt_sd,
            'rt_n': rt_n,
            'rtcv': rtcv
        })
    
    return pd.DataFrame(probe_data)


def main():
    """
    Main execution function.
    
    Aggregates trial-level SART data into probe-level objective markers
    for all subjects and tasks.
    """
    print("="*60)
    print("AGGREGATING OBJECTIVE MARKERS BY PROBE")
    print("="*60)
    print(f"BIDS Root: {BIDS_RAW_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"N trials back from probe: {N_BACK_TO_PROBE if N_BACK_TO_PROBE else 'All'}")
    print(f"Processing {len(SUBJECTS)} subjects × {len(TASKS)} tasks")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_probe_data = []
    
    for subject_id in SUBJECTS:
        for task in TASKS:
            # Load trial data
            df_trials = load_trial_data(BIDS_RAW_ROOT, subject_id, task)
            
            if df_trials.empty:
                print(f"Warning: No data for sub-{subject_id}/{task}")
                continue
            
            # Aggregate by probe
            df_probe = aggregate_by_probe(df_trials, n_back=N_BACK_TO_PROBE)
            
            if not df_probe.empty:
                all_probe_data.append(df_probe)
                print(f"Processed sub-{subject_id}/{task}: {len(df_probe)} probes")
    
    if not all_probe_data:
        print("Error: No data was aggregated!")
        return
    
    # Combine all data
    df_all = pd.concat(all_probe_data, ignore_index=True)
    
    # Save output
    suffix = f"_n{N_BACK_TO_PROBE}" if N_BACK_TO_PROBE else ""
    output_file = output_path / f"objective_markers_per_probe{suffix}.csv"
    df_all.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total probe observations: {len(df_all)}")
    print(f"Unique subjects: {df_all['subject'].nunique()}")
    print(f"Unique tasks: {df_all['sart'].nunique()}")
    
    print("\nObjective Markers Summary:")
    for marker in ['omission_rate', 'commission_rate', 'rtcv']:
        valid_data = df_all[marker].dropna()
        print(f"  {marker}: mean={valid_data.mean():.4f}, std={valid_data.std():.4f}, "
              f"range=[{valid_data.min():.4f}, {valid_data.max():.4f}]")
    
    print("\n" + "="*60)
    print("AGGREGATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
