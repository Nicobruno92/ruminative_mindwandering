#!/usr/bin/env python3
"""
Helper functions for Junifer markers aggregation pipeline.

This module provides utility functions for loading configurations, reading epochs,
parsing events, and performing outlier detection. It's designed to be independent
of the ERP pipeline helpers.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne


def load_yaml_config(path: str) -> Dict:
    """
    Load YAML configuration file using PyYAML-safe loader.
    
    Parameters
    ----------
    path : str
        Path to the YAML configuration file
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_derivative_dir(derivatives_root: str, subject: str, datatype: str = "eeg") -> str:
    """
    Create and return derivatives directory for a subject.
    Mirrors structure used by BIDSCompliance.
    
    Parameters
    ----------
    derivatives_root : str
        Root directory for derivatives
    subject : str
        Subject identifier
    datatype : str
        Data type (default: "eeg")
        
    Returns
    -------
    str
        Path to the derivative directory
    """
    out_dir = os.path.join(os.path.abspath(derivatives_root), f"sub-{subject}", datatype)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def make_derivative_fname(subject: str, task: str, suffix: str, desc: Optional[str], extension: str) -> str:
    """
    Build derivative filename following BIDS naming conventions.
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task identifier
    suffix : str
        File suffix (e.g., "epo", "ave")
    desc : Optional[str]
        Description string (e.g., "autoPreproc")
    extension : str
        File extension (e.g., ".fif")
        
    Returns
    -------
    str
        Formatted filename
    """
    if desc:
        return f"sub-{subject}_task-{task}_desc-{desc}_{suffix}{extension}"
    return f"sub-{subject}_task-{task}_{suffix}{extension}"


def read_derivative_epochs(
    derivatives_root: str,
    subject: str,
    task: str,
    desc: Optional[str] = None,
    preload: bool = False,
    proj: bool | str = True,
    bids_root: Optional[str] = None,
) -> Tuple[mne.Epochs, str, str]:
    """
    Read derivative epochs saved by the preprocessing pipeline.
    
    Parameters
    ----------
    derivatives_root : str
        Root directory for derivatives
    subject : str
        Subject identifier
    task : str
        Task identifier
    desc : Optional[str]
        Description string for the epochs (e.g., "autoPreproc")
    preload : bool
        Whether to preload the data into memory
    proj : bool | str
        Whether to apply projections
    bids_root : Optional[str]
        BIDS root directory (unused but kept for compatibility)
        
    Returns
    -------
    Tuple[mne.Epochs, str, str]
        Tuple containing:
        - epochs: Loaded epochs object
        - events_tsv: Path to the events.tsv file
        - events_json: Path to the events.json file
    """
    # Prefer using BIDSCompliance if available
    epochs = None
    path = None
    try:
        from Preprocessing_pipeline_new.bids_compliance import BIDSCompliance  # type: ignore
        bc = BIDSCompliance(bids_root=bids_root or ".")
        path = bc.build_derivative_epochs_path(
            derivatives_root=derivatives_root, subject=subject, task=task, desc=desc
        )
        epochs = bc.read_derivative_epochs(
            derivatives_root=derivatives_root, subject=subject, task=task, desc=desc, preload=preload, proj=proj
        )
    except Exception:
        out_dir = build_derivative_dir(derivatives_root, subject, datatype="eeg")
        fname = make_derivative_fname(subject, task, suffix="epo", desc=desc, extension=".fif")
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            epochs = mne.read_epochs(path, proj=proj, preload=preload, verbose=False)
        except TypeError:
            epochs = mne.read_epochs(path, preload=preload, verbose=False)

    events_tsv = path.replace(".fif", "_events.tsv")
    events_json = path.replace(".fif", "_events.json")
    return epochs, events_tsv, events_json


def parse_events_tsv(events_tsv_path: str) -> pd.DataFrame:
    """
    Read events.tsv into a DataFrame.
    
    Parameters
    ----------
    events_tsv_path : str
        Path to the events.tsv file
        
    Returns
    -------
    pd.DataFrame
        Events dataframe with required columns: onset, duration, description, event_id
    """
    if not os.path.exists(events_tsv_path):
        raise FileNotFoundError(events_tsv_path)
    df = pd.read_csv(events_tsv_path, sep="\t")
    required = {"onset", "duration", "description", "event_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in events.tsv: {sorted(missing)}")
    # Preserve row order to map directly to epochs order
    df = df.reset_index().rename(columns={"index": "row_index"})
    return df


def enrich_events_with_parsed_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse event description strings to extract fields used for selection.
    
    Extracts the following fields from description strings:
    - trial_type: 'go' or 'nogo'
    - correctness: 'correct' or 'incorrect'
    - onoff: integer 0..100 when present
    - distance_to_probe: negative integer (e.g., -5..-1)
    - probe_number: integer identifier of the probe
    
    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe with 'description' column
        
    Returns
    -------
    pd.DataFrame
        Enriched dataframe with parsed fields
    """
    if df is None or df.empty:
        return df

    result_df = df.copy()
    desc = result_df["description"].fillna("").astype(str)

    # trial type
    result_df["trial_type"] = "unknown"
    result_df.loc[desc.str.startswith("go/"), "trial_type"] = "go"
    result_df.loc[desc.str.startswith("nogo/"), "trial_type"] = "nogo"

    # correctness
    result_df["correctness"] = "unknown"
    result_df.loc[desc.str.contains("/correct/"), "correctness"] = "correct"
    result_df.loc[desc.str.contains("/incorrect/"), "correctness"] = "incorrect"

    # Numeric ratings embedded in description (old format binary splits)
    m_onoff = desc.str.extract(r"onoff(\d+)", expand=False)
    m_selfother = desc.str.extract(r"selfother(\d+)", expand=False)
    m_valence = desc.str.extract(r"valence(\d+)", expand=False)
    m_time = desc.str.extract(r"time(\d+)", expand=False)
    m_confidence = desc.str.extract(r"confidence(\d+)", expand=False)
    m_average = desc.str.extract(r"average(\d+)", expand=False)
    result_df["onoff"] = pd.to_numeric(m_onoff, errors="coerce").astype("float64")
    result_df["selfother"] = pd.to_numeric(m_selfother, errors="coerce").astype("float64")
    result_df["valence"] = pd.to_numeric(m_valence, errors="coerce").astype("float64")
    result_df["time"] = pd.to_numeric(m_time, errors="coerce").astype("float64")
    result_df["confidence"] = pd.to_numeric(m_confidence, errors="coerce").astype("float64")
    result_df["average"] = pd.to_numeric(m_average, errors="coerce").astype("float64")

    # distance to probe encoded as '/-X/'
    m_dist = desc.str.extract(r"/-(\d+)/", expand=False)
    dist_val = pd.to_numeric(m_dist, errors="coerce")
    result_df["distance_to_probe"] = np.where(~dist_val.isna(), -dist_val.astype("Int64"), pd.NA)

    # probe number encoded as '/probeX' (end of string)
    m_probe = desc.str.extract(r"/probe(\d+)", expand=False)
    result_df["probe_number"] = pd.to_numeric(m_probe, errors="coerce").astype("Int64")

    return result_df


def label_probe_onoff(events_df: pd.DataFrame, onoff_threshold: int) -> pd.Series:
    """
    Determine the on-task/off-task label per probe using the onoff rating.
    
    If multiple trials per probe have onoff, use the median.
    Threshold rule: on-task if onoff > threshold; off-task if onoff <= threshold; NaN -> 'unknown'.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with 'onoff' and 'probe_number' columns
    onoff_threshold : int
        Threshold for determining on-task vs off-task
        
    Returns
    -------
    pd.Series
        Series indexed by probe_number with values in {"onTask", "offTask", "unknown"}
    """
    df = events_df.copy()
    # compute probe-level central onoff
    probe_onoff = df.groupby("probe_number")["onoff"].median()

    def to_label(val: float) -> str:
        if pd.isna(val):
            return "unknown"
        if val > onoff_threshold:
            return "onTask"
        # Equal to threshold should be considered off-task
        return "offTask"

    return probe_onoff.map(to_label)


def detect_outlier_epochs(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    probe_number: int,
    baseline_distance_min: int,
    baseline_distance_max: int,
    min_baseline_epochs: int,
    z_threshold: float,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Detect outlier epochs for a specific probe using z-score thresholding.
    
    Uses a broader range of epochs to compute baseline statistics for more reliable
    outlier detection than just using the epochs to be aggregated.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs object containing all data
    events_df : pd.DataFrame
        Events dataframe with distance_to_probe and probe_number columns
    probe_number : int
        The probe number to process
    baseline_distance_min : int
        Minimum distance for baseline statistics computation (e.g., -10)
    baseline_distance_max : int
        Maximum distance for baseline statistics computation (e.g., -1)
    min_baseline_epochs : int
        Minimum number of epochs needed for reliable statistics
    z_threshold : float
        Z-score threshold for outlier detection
        
    Returns
    -------
    Tuple[List[int], Dict[str, float]]
        Tuple containing:
        - valid_indices: List of epoch indices that are not outliers
        - stats: Statistics including n_total, n_outliers, n_valid, mean, std
    """
    # Get epochs for baseline statistics computation (broader range)
    baseline_mask = (
        (events_df["probe_number"] == probe_number) &
        (events_df["distance_to_probe"] >= baseline_distance_min) &
        (events_df["distance_to_probe"] <= baseline_distance_max) &
        (events_df["distance_to_probe"].notna())
    )
    
    baseline_indices = events_df.loc[baseline_mask, "epoch_index"].astype(int).tolist()
    
    if len(baseline_indices) < min_baseline_epochs:
        # Not enough epochs for reliable statistics, return empty
        return [], {
            "n_total": 0,
            "n_baseline_epochs": len(baseline_indices),
            "n_outliers": 0,
            "n_valid": 0,
            "mean": np.nan,
            "std": np.nan,
            "z_threshold": z_threshold
        }
    
    try:
        # Get baseline epochs for computing statistics
        baseline_epochs = epochs.copy()[baseline_indices]
        
        # Compute RMS across channels and time for each epoch
        baseline_data = baseline_epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        baseline_rms = np.sqrt(np.mean(baseline_data**2, axis=(1, 2)))  # RMS per epoch
        
        # Compute baseline statistics
        baseline_mean = np.mean(baseline_rms)
        baseline_std = np.std(baseline_rms, ddof=1)
        
        if baseline_std == 0:
            # No variability, can't detect outliers
            return baseline_indices, {
                "n_total": len(baseline_indices),
                "n_baseline_epochs": len(baseline_indices),
                "n_outliers": 0,
                "n_valid": len(baseline_indices),
                "mean": baseline_mean,
                "std": baseline_std,
                "z_threshold": z_threshold
            }
        
        # Compute z-scores for baseline epochs
        z_scores = np.abs((baseline_rms - baseline_mean) / baseline_std)
        
        # Identify valid (non-outlier) epochs
        valid_mask = z_scores <= z_threshold
        valid_indices = [baseline_indices[i] for i in range(len(baseline_indices)) if valid_mask[i]]
        
        stats = {
            "n_total": len(baseline_indices),
            "n_baseline_epochs": len(baseline_indices),
            "n_outliers": len(baseline_indices) - len(valid_indices),
            "n_valid": len(valid_indices),
            "mean": float(baseline_mean),
            "std": float(baseline_std),
            "z_threshold": z_threshold
        }
        
        return valid_indices, stats
        
    except Exception as exc:
        print(f"[WARN] Outlier detection failed for probe {probe_number}: {exc}")
        return baseline_indices, {
            "n_total": len(baseline_indices),
            "n_baseline_epochs": len(baseline_indices),
            "n_outliers": 0,
            "n_valid": len(baseline_indices),
            "mean": np.nan,
            "std": np.nan,
            "z_threshold": z_threshold
        }
