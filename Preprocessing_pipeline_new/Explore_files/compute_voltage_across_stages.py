#!/usr/bin/env python3
"""
Compute average voltage across preprocessing stages for all subjects.

This script loads raw and preprocessed EEG data at different stages and computes
the average absolute voltage to identify where signal amplitude is being reduced.

Stages analyzed:
- Raw: Original BrainVision data
- ICA Clean: After ICA artifact removal
- Evoked Epochs: Event-related epochs
- State Epochs: Pre-probe state windows
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# RAW_ROOT = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/raw"
RAW_ROOT = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/raw"  # Local machine alternative
# DERIVATIVES_ROOT = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"
DERIVATIVES_ROOT = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # Local

OUTPUT_CSV = "voltage_analysis_all_subjects.csv"  # Output table filename

# Subjects and tasks to process
SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", 
            "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", 
            "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", 
            "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]

# Channel types to include in voltage calculation
CHANNEL_TYPES = ["eeg"]  # Only EEG channels

VERBOSE = True  # Print progress information
# =============================================================================

import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import Optional, Dict, Tuple


def compute_avg_voltage(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute voltage statistics from EEG data.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data array (channels x timepoints) or (epochs x channels x timepoints).
    
    Returns
    -------
    mean_abs_voltage : float
        Mean of absolute voltage values (µV).
    std_voltage : float
        Standard deviation of voltage values (µV).
    peak_to_peak : float
        Mean peak-to-peak amplitude across channels (µV).
    """
    # Convert to 2D if epochs (flatten epochs dimension)
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    
    # Compute statistics
    mean_abs_voltage = np.mean(np.abs(data))
    std_voltage = np.std(data)
    
    # Peak-to-peak per channel
    if data.ndim == 2:
        peak_to_peak = np.mean([np.ptp(data[ch, :]) for ch in range(data.shape[0])])
    else:
        peak_to_peak = np.ptp(data)
    
    # Convert from V to µV (MNE uses V by default)
    mean_abs_voltage *= 1e6
    std_voltage *= 1e6
    peak_to_peak *= 1e6
    
    return mean_abs_voltage, std_voltage, peak_to_peak


def load_raw_data(subject: str, task: str) -> Optional[mne.io.BaseRaw]:
    """
    Load raw BrainVision data for a subject and task.
    
    Parameters
    ----------
    subject : str
        Subject ID (e.g., "02").
    task : str
        Task name (e.g., "Sart1").
    
    Returns
    -------
    raw : mne.io.BaseRaw or None
        Loaded raw data, or None if file not found.
    """
    # BIDS naming: sub-XX/eeg/sub-XX_task-TaskName_eeg.vhdr
    vhdr_path = Path(RAW_ROOT) / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_eeg.vhdr"
    
    if not vhdr_path.exists():
        return None
    
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        if VERBOSE:
            print(f"  Error loading raw {vhdr_path}: {e}")
        return None


def load_ica_clean(subject: str, task: str) -> Optional[mne.io.BaseRaw]:
    """
    Load ICA-cleaned raw data.
    
    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    
    Returns
    -------
    raw : mne.io.BaseRaw or None
        ICA-cleaned raw data, or None if not found.
    """
    # BIDS derivatives naming: sub-XX_task-TaskName_desc-icaClean_eeg.fif
    fif_path = Path(DERIVATIVES_ROOT) / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_desc-icaClean_eeg.fif"
    
    if not fif_path.exists():
        return None
    
    try:
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        if VERBOSE:
            print(f"  Error loading ICA clean {fif_path}: {e}")
        return None


def load_evoked_epochs(subject: str, task: str) -> Optional[mne.Epochs]:
    """
    Load evoked epochs.
    
    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    
    Returns
    -------
    epochs : mne.Epochs or None
        Evoked epochs, or None if not found.
    """
    # BIDS derivatives naming: sub-XX_task-TaskName_desc-evoked_epo.fif
    epo_path = Path(DERIVATIVES_ROOT) / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_desc-evoked_epo.fif"
    
    if not epo_path.exists():
        return None
    
    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        return epochs
    except Exception as e:
        if VERBOSE:
            print(f"  Error loading evoked epochs {epo_path}: {e}")
        return None


def load_state_epochs(subject: str, task: str) -> Optional[mne.Epochs]:
    """
    Load state epochs.
    
    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    
    Returns
    -------
    epochs : mne.Epochs or None
        State epochs, or None if not found.
    """
    # BIDS derivatives naming: sub-XX_task-TaskName_desc-state_epo.fif
    epo_path = Path(DERIVATIVES_ROOT) / f"sub-{subject}" / "eeg" / f"sub-{subject}_task-{task}_desc-state_epo.fif"
    
    if not epo_path.exists():
        return None
    
    try:
        epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
        return epochs
    except Exception as e:
        if VERBOSE:
            print(f"  Error loading state epochs {epo_path}: {e}")
        return None


def process_subject_task(subject: str, task: str) -> Dict[str, any]:
    """
    Process a single subject-task combination and compute voltage statistics.
    
    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    
    Returns
    -------
    results : dict
        Dictionary with voltage statistics for each stage.
    """
    results = {
        "subject": subject,
        "task": task,
        "raw_mean_abs_uv": np.nan,
        "raw_std_uv": np.nan,
        "raw_peak_to_peak_uv": np.nan,
        "ica_mean_abs_uv": np.nan,
        "ica_std_uv": np.nan,
        "ica_peak_to_peak_uv": np.nan,
        "evoked_mean_abs_uv": np.nan,
        "evoked_std_uv": np.nan,
        "evoked_peak_to_peak_uv": np.nan,
        "state_mean_abs_uv": np.nan,
        "state_std_uv": np.nan,
        "state_peak_to_peak_uv": np.nan,
        "raw_n_channels": np.nan,
        "ica_n_channels": np.nan,
        "evoked_n_epochs": np.nan,
        "state_n_epochs": np.nan,
    }
    
    if VERBOSE:
        print(f"\nProcessing sub-{subject}, task-{task}...")
    
    # Load and process raw data
    raw = load_raw_data(subject, task)
    if raw is not None:
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) > 0:
            data = raw.get_data(picks=picks)
            mean_abs, std, ptp = compute_avg_voltage(data)
            results["raw_mean_abs_uv"] = mean_abs
            results["raw_std_uv"] = std
            results["raw_peak_to_peak_uv"] = ptp
            results["raw_n_channels"] = len(picks)
            if VERBOSE:
                print(f"  Raw: {mean_abs:.2f} µV (mean abs), {ptp:.2f} µV (p2p)")
    
    # Load and process ICA-cleaned data
    ica_clean = load_ica_clean(subject, task)
    if ica_clean is not None:
        picks = mne.pick_types(ica_clean.info, eeg=True, exclude="bads")
        if len(picks) > 0:
            data = ica_clean.get_data(picks=picks)
            mean_abs, std, ptp = compute_avg_voltage(data)
            results["ica_mean_abs_uv"] = mean_abs
            results["ica_std_uv"] = std
            results["ica_peak_to_peak_uv"] = ptp
            results["ica_n_channels"] = len(picks)
            if VERBOSE:
                print(f"  ICA: {mean_abs:.2f} µV (mean abs), {ptp:.2f} µV (p2p)")
    
    # Load and process evoked epochs
    evoked_epo = load_evoked_epochs(subject, task)
    if evoked_epo is not None:
        picks = mne.pick_types(evoked_epo.info, eeg=True, exclude="bads")
        if len(picks) > 0:
            data = evoked_epo.get_data(picks=picks)
            mean_abs, std, ptp = compute_avg_voltage(data)
            results["evoked_mean_abs_uv"] = mean_abs
            results["evoked_std_uv"] = std
            results["evoked_peak_to_peak_uv"] = ptp
            results["evoked_n_epochs"] = len(evoked_epo)
            if VERBOSE:
                print(f"  Evoked: {mean_abs:.2f} µV (mean abs), {ptp:.2f} µV (p2p), {len(evoked_epo)} epochs")
    
    # Load and process state epochs
    state_epo = load_state_epochs(subject, task)
    if state_epo is not None:
        picks = mne.pick_types(state_epo.info, eeg=True, exclude="bads")
        if len(picks) > 0:
            data = state_epo.get_data(picks=picks)
            mean_abs, std, ptp = compute_avg_voltage(data)
            results["state_mean_abs_uv"] = mean_abs
            results["state_std_uv"] = std
            results["state_peak_to_peak_uv"] = ptp
            results["state_n_epochs"] = len(state_epo)
            if VERBOSE:
                print(f"  State: {mean_abs:.2f} µV (mean abs), {ptp:.2f} µV (p2p), {len(state_epo)} epochs")
    
    return results


def main():
    """
    Main function to process all subjects and tasks.
    
    Computes voltage statistics at each preprocessing stage and saves results
    to a CSV file with one row per subject-task combination.
    """
    print("=" * 80)
    print("VOLTAGE ANALYSIS ACROSS PREPROCESSING STAGES")
    print("=" * 80)
    print(f"Raw data root: {RAW_ROOT}")
    print(f"Derivatives root: {DERIVATIVES_ROOT}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Processing {len(SUBJECTS)} subjects × {len(TASKS)} tasks = {len(SUBJECTS) * len(TASKS)} combinations")
    print("=" * 80)
    
    # Process all subject-task combinations
    all_results = []
    for subject in SUBJECTS:
        for task in TASKS:
            results = process_subject_task(subject, task)
            all_results.append(results)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        "subject", "task",
        "raw_mean_abs_uv", "raw_std_uv", "raw_peak_to_peak_uv", "raw_n_channels",
        "ica_mean_abs_uv", "ica_std_uv", "ica_peak_to_peak_uv", "ica_n_channels",
        "evoked_mean_abs_uv", "evoked_std_uv", "evoked_peak_to_peak_uv", "evoked_n_epochs",
        "state_mean_abs_uv", "state_std_uv", "state_peak_to_peak_uv", "state_n_epochs",
    ]
    df = df[column_order]
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Print summary statistics
    print("\nMean absolute voltage (µV) across all valid recordings:")
    print(f"  Raw:         {df['raw_mean_abs_uv'].mean():.2f} ± {df['raw_mean_abs_uv'].std():.2f}")
    print(f"  ICA Clean:   {df['ica_mean_abs_uv'].mean():.2f} ± {df['ica_mean_abs_uv'].std():.2f}")
    print(f"  Evoked Epo:  {df['evoked_mean_abs_uv'].mean():.2f} ± {df['evoked_mean_abs_uv'].std():.2f}")
    print(f"  State Epo:   {df['state_mean_abs_uv'].mean():.2f} ± {df['state_mean_abs_uv'].std():.2f}")
    
    print("\nPeak-to-peak voltage (µV) across all valid recordings:")
    print(f"  Raw:         {df['raw_peak_to_peak_uv'].mean():.2f} ± {df['raw_peak_to_peak_uv'].std():.2f}")
    print(f"  ICA Clean:   {df['ica_peak_to_peak_uv'].mean():.2f} ± {df['ica_peak_to_peak_uv'].std():.2f}")
    print(f"  Evoked Epo:  {df['evoked_peak_to_peak_uv'].mean():.2f} ± {df['evoked_peak_to_peak_uv'].std():.2f}")
    print(f"  State Epo:   {df['state_peak_to_peak_uv'].mean():.2f} ± {df['state_peak_to_peak_uv'].std():.2f}")
    
    print("\nData availability:")
    print(f"  Raw files found:         {df['raw_mean_abs_uv'].notna().sum()} / {len(df)}")
    print(f"  ICA clean files found:   {df['ica_mean_abs_uv'].notna().sum()} / {len(df)}")
    print(f"  Evoked epochs found:     {df['evoked_mean_abs_uv'].notna().sum()} / {len(df)}")
    print(f"  State epochs found:      {df['state_mean_abs_uv'].notna().sum()} / {len(df)}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
