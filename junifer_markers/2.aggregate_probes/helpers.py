#!/usr/bin/env python3
"""
Helper functions for Junifer marker aggregation pipeline.

This module provides:
    - Configuration loading (YAML)
    - BIDS-style path building
    - Event description parsing for epoch metadata extraction
    - Probe on-task/off-task labeling

The parsing logic supports multiple epoch description formats:
    1. Evoked epochs: trial-locked with distance information
    2. State epochs: temporal bins before probe
    3. Sleep epochs: single epoch per probe
    4. Legacy format: numeric ratings

Author: Wandering Mind EEG Analysis Pipeline
"""

import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pathlib import Path


# =============================================================================
# PROJECT ROOT
# =============================================================================

def get_project_root() -> Path:
    """
    Get project root directory.
    
    Assumes this file is located at:
    Analysis/eeg/junifer_markers/2.aggregate_probes/helpers.py
    
    Returns
    -------
    Path
        Path to project root
    """
    # Go up 5 levels from helpers.py
    return Path(__file__).resolve().parent.parent.parent.parent.parent


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_yaml_config(path: str) -> Dict:
    """
    Load YAML configuration file.
    
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


# =============================================================================
# BIDS PATH HELPERS
# =============================================================================

def build_derivative_dir(
    derivatives_root: str, 
    subject: str, 
    datatype: str = "eeg",
    session: Optional[str] = None,
) -> str:
    """
    Build and create a BIDS-style derivatives directory.
    
    Creates the directory structure:
        {derivatives_root}/sub-{subject}/ses-{session}/{datatype}/
    
    Parameters
    ----------
    derivatives_root : str
        Root directory for derivatives
    subject : str
        Subject identifier (e.g., "03")
    datatype : str
        Data type (default: "eeg")
    session : Optional[str]
        Session identifier (e.g., "a", "b")
        
    Returns
    -------
    str
        Absolute path to the derivative directory (created if needed)
    """
    parts = [os.path.abspath(derivatives_root), f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(datatype)
    
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def make_derivative_fname(
    subject: str, 
    task: str, 
    suffix: str, 
    desc: Optional[str], 
    extension: str,
    session: Optional[str] = None,
) -> str:
    """
    Build a BIDS-style derivative filename.
    
    Generates filenames like:
        sub-03_ses-a_task-sartvisual_desc-evoked_epo.fif
    
    Parameters
    ----------
    subject : str
        Subject identifier
    task : str
        Task identifier
    suffix : str
        File suffix (e.g., "epo", "ave")
    desc : Optional[str]
        Description (e.g., "evoked", "state")
    extension : str
        File extension with dot (e.g., ".fif")
    session : Optional[str]
        Session identifier
        
    Returns
    -------
    str
        Formatted BIDS filename
    """
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append(f"task-{task}")
    if desc:
        parts.append(f"desc-{desc}")
    parts.append(suffix)
    return "_".join(parts) + extension


# =============================================================================
# EVENT DESCRIPTION PARSING
# =============================================================================

# Expected probe response categories (from experiment)
PROBE_RESPONSE_CATEGORIES = [
    "on-task", "about-task", "distracted", 
    "spontaneous", "deliberate", "blank", "asleep"
]

# Confidence level strings
CONFIDENCE_LEVELS = [
    "completely confident", "very confident", "somewhat confident",
    "a little confident", "not at all confident"
]

# Depth/immersion level strings
DEPTH_LEVELS = [
    "completely immersed", "very immersed", "somewhat immersed",
    "a little immersed", "not at all immersed"
]


def enrich_events_with_parsed_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse epoch description strings into structured metadata columns.
    
    This function extracts probe-related metadata from event descriptions
    to enable filtering and grouping of epochs by probe characteristics.
    
    Supported Description Formats
    -----------------------------
    1. **Evoked epochs** (trial-locked):
       ``go/correct/probe-N/trial-M/distance-D/ONTASK/content/confidence/depth``
       
    2. **State epochs** (temporal bins):
       ``state/preprobe_win/dt=-X.00/THOUGHT-PROBE/probe-N/ONTASK/content/...``
       
    3. **Sleep epochs** (single per probe):
       ``sleep/preprobe_win/THOUGHT-PROBE/probe-N/ONTASK/content/...``
       
    4. **Legacy format** (numeric ratings):
       ``go/correct/onoff50/selfother30/.../probe1``
    
    Extracted Fields
    ----------------
    - trial_type : {'go', 'nogo', 'state', 'sleep', 'unknown'}
    - correctness : {'correct', 'incorrect', 'unknown'}
    - probe_number : int - Probe identifier for grouping
    - trial_number : int - Global trial index (evoked only)
    - distance_to_probe : int - Distance to probe, 0=closest (evoked only)
    - temporal_bin : int - Seconds before probe (state only)
    - ontask_label : {'ONTASK', 'OFFTASK', 'unknown'}
    - content : Thought content category
    - confidence_level : Confidence rating
    - depth_level : Immersion/depth rating
    - onoff : float - Legacy 0-100 rating
    
    Parameters
    ----------
    df : pd.DataFrame
        Events dataframe with 'description' column
        
    Returns
    -------
    pd.DataFrame
        Copy of input with additional parsed columns
    """
    if df is None or df.empty:
        return df

    result_df = df.copy()
    desc = result_df["description"].fillna("").astype(str)

    # -------------------------------------------------------------------------
    # TRIAL TYPE AND CORRECTNESS
    # -------------------------------------------------------------------------
    result_df["trial_type"] = "unknown"
    result_df.loc[desc.str.startswith("go/"), "trial_type"] = "go"
    result_df.loc[desc.str.startswith("nogo/"), "trial_type"] = "nogo"
    result_df.loc[desc.str.startswith("state/"), "trial_type"] = "state"
    result_df.loc[desc.str.startswith("sleep/"), "trial_type"] = "sleep"

    result_df["correctness"] = "unknown"
    result_df.loc[desc.str.contains("/correct/"), "correctness"] = "correct"
    result_df.loc[desc.str.contains("/incorrect/"), "correctness"] = "incorrect"

    # -------------------------------------------------------------------------
    # PROBE NUMBER (multiple format support)
    # -------------------------------------------------------------------------
    # Harmonized: /probe-N/, Legacy: /probeN/, Segment: /segment-N/
    probe_val = (
        pd.to_numeric(desc.str.extract(r"/probe-(\d+)", expand=False), errors="coerce")
        .fillna(pd.to_numeric(desc.str.extract(r"/probe(\d+)", expand=False), errors="coerce"))
        .fillna(pd.to_numeric(desc.str.extract(r"/segment-(\d+)", expand=False), errors="coerce"))
    )
    result_df["probe_number"] = probe_val.astype("Int64")

    # -------------------------------------------------------------------------
    # TRIAL NUMBER (evoked epochs only)
    # -------------------------------------------------------------------------
    trial_val = (
        pd.to_numeric(desc.str.extract(r"/trial-(\d+)", expand=False), errors="coerce")
        .fillna(pd.to_numeric(desc.str.extract(r"/segment-\d+/(\d+),\d+/", expand=False), errors="coerce"))
    )
    result_df["trial_number"] = trial_val.astype("Int64")

    # -------------------------------------------------------------------------
    # DISTANCE TO PROBE (evoked epochs only)
    # -------------------------------------------------------------------------
    # Harmonized: /distance-D/, Legacy: /-D/, Segment: /segment-N/M,D/
    dist_val = (
        pd.to_numeric(desc.str.extract(r"/distance-(\d+)", expand=False), errors="coerce")
        .fillna(pd.to_numeric(desc.str.extract(r"/-(\d+)/", expand=False), errors="coerce"))
        .fillna(pd.to_numeric(desc.str.extract(r"/segment-\d+/\d+,(\d+)/", expand=False), errors="coerce"))
    )
    result_df["distance_to_probe"] = dist_val.astype("Int64")

    # -------------------------------------------------------------------------
    # TEMPORAL BIN (state epochs only)
    # -------------------------------------------------------------------------
    # Format: dt=-X.00 (seconds before probe)
    m_temporal = desc.str.extract(r"dt=(-?\d+)\.00", expand=False)
    result_df["temporal_bin"] = pd.to_numeric(m_temporal, errors="coerce").astype("Int64")

    # -------------------------------------------------------------------------
    # ON-TASK / OFF-TASK LABEL
    # -------------------------------------------------------------------------
    result_df["ontask_label"] = "unknown"
    
    # Multiple format variants
    ontask_patterns = ["/ONTASK/", "/ONTASK$", "/ontask/", "/ontask$", "/on-task/", "/on-task$"]
    offtask_patterns = ["/OFFTASK/", "/OFFTASK$", "/offtask/", "/offtask$", "/off-task/", "/off-task$"]
    
    for pattern in ontask_patterns:
        if pattern.endswith("$"):
            result_df.loc[desc.str.contains(pattern, case=False, regex=True), "ontask_label"] = "ONTASK"
        else:
            result_df.loc[desc.str.contains(pattern, case=False), "ontask_label"] = "ONTASK"
            
    for pattern in offtask_patterns:
        if pattern.endswith("$"):
            result_df.loc[desc.str.contains(pattern, case=False, regex=True), "ontask_label"] = "OFFTASK"
        else:
            result_df.loc[desc.str.contains(pattern, case=False), "ontask_label"] = "OFFTASK"

    # -------------------------------------------------------------------------
    # THOUGHT CONTENT CATEGORY
    # -------------------------------------------------------------------------
    content_pattern = r"/(" + "|".join(PROBE_RESPONSE_CATEGORIES) + r")(?:/|$)"
    m_content = desc.str.extract(content_pattern, expand=False, flags=re.IGNORECASE)
    result_df["content"] = (
        m_content.str.lower()
        .str.replace("about task", "about-task")
        .fillna("unknown")
    )

    # -------------------------------------------------------------------------
    # CONFIDENCE AND DEPTH LEVELS
    # -------------------------------------------------------------------------
    conf_pattern = r"/(" + "|".join(CONFIDENCE_LEVELS) + r")(?:/|$)"
    m_conf = desc.str.extract(conf_pattern, expand=False, flags=re.IGNORECASE)
    result_df["confidence_level"] = m_conf.str.lower().fillna("unknown")
    
    depth_pattern = r"/(" + "|".join(DEPTH_LEVELS) + r")(?:/|$)"
    m_depth = desc.str.extract(depth_pattern, expand=False, flags=re.IGNORECASE)
    result_df["depth_level"] = m_depth.str.lower().fillna("unknown")

    # -------------------------------------------------------------------------
    # LEGACY NUMERIC RATINGS
    # -------------------------------------------------------------------------
    result_df["onoff"] = pd.to_numeric(
        desc.str.extract(r"onoff(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    result_df["selfother"] = pd.to_numeric(
        desc.str.extract(r"selfother(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    result_df["valence"] = pd.to_numeric(
        desc.str.extract(r"valence(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    result_df["time"] = pd.to_numeric(
        desc.str.extract(r"time(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    result_df["confidence"] = pd.to_numeric(
        desc.str.extract(r"confidence(\d+)", expand=False), errors="coerce"
    ).astype("float64")
    result_df["average"] = pd.to_numeric(
        desc.str.extract(r"average(\d+)", expand=False), errors="coerce"
    ).astype("float64")

    # -------------------------------------------------------------------------
    # DERIVE MISSING VALUES (backward compatibility)
    # -------------------------------------------------------------------------
    # Derive onoff from ontask_label if missing
    mask_ontask = result_df["onoff"].isna() & (result_df["ontask_label"] == "ONTASK")
    mask_offtask = result_df["onoff"].isna() & (result_df["ontask_label"] == "OFFTASK")
    result_df.loc[mask_ontask, "onoff"] = 100.0
    result_df.loc[mask_offtask, "onoff"] = 0.0
    
    # Derive ontask_label from content if still unknown
    mask_content_on = (result_df["ontask_label"] == "unknown") & (result_df["content"] == "on-task")
    mask_content_off = (
        (result_df["ontask_label"] == "unknown") 
        & (result_df["content"] != "unknown") 
        & (result_df["content"] != "on-task")
    )
    result_df.loc[mask_content_on, "ontask_label"] = "ONTASK"
    result_df.loc[mask_content_off, "ontask_label"] = "OFFTASK"
    
    # Update onoff for newly derived labels
    mask_new_on = result_df["onoff"].isna() & (result_df["ontask_label"] == "ONTASK")
    mask_new_off = result_df["onoff"].isna() & (result_df["ontask_label"] == "OFFTASK")
    result_df.loc[mask_new_on, "onoff"] = 100.0
    result_df.loc[mask_new_off, "onoff"] = 0.0

    return result_df


# =============================================================================
# PROBE LABELING
# =============================================================================

def label_probe_onoff(events_df: pd.DataFrame, onoff_threshold: int = 50) -> pd.Series:
    """
    Assign on-task/off-task labels to each probe.
    
    Uses the most common label across all epochs belonging to each probe.
    Supports both harmonized (ONTASK/OFFTASK) and legacy (0-100 rating) formats.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with 'probe_number' column and either
        'ontask_label' or 'onoff' column
    onoff_threshold : int
        Threshold for legacy format: values > threshold → onTask
        
    Returns
    -------
    pd.Series
        Series indexed by probe_number with labels: "onTask", "offTask", "unknown"
    """
    df = events_df.copy()
    
    # Try harmonized format first (ONTASK/OFFTASK labels)
    if "ontask_label" in df.columns:
        def get_mode_label(group: pd.DataFrame) -> str:
            """Get most common non-unknown label in group."""
            labels = group["ontask_label"].dropna()
            labels = labels[labels != "unknown"]
            if len(labels) == 0:
                return "unknown"
            mode = labels.mode()
            if len(mode) > 0:
                label = mode.iloc[0]
                return "onTask" if label == "ONTASK" else "offTask" if label == "OFFTASK" else "unknown"
            return "unknown"
        
        probe_labels = df.groupby("probe_number").apply(get_mode_label)
        
        # Return if we got meaningful labels
        if not (probe_labels == "unknown").all():
            return probe_labels
    
    # Fallback: legacy numeric format
    probe_onoff = df.groupby("probe_number")["onoff"].median()
    
    def numeric_to_label(val: float) -> str:
        if pd.isna(val):
            return "unknown"
        return "onTask" if val > onoff_threshold else "offTask"

    return probe_onoff.map(numeric_to_label)
