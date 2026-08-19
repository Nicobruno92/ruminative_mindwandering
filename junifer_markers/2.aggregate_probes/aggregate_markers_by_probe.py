#!/usr/bin/env python3
"""
Aggregate Junifer markers by probe for mind-wandering analysis.

This script reads per-epoch marker HDF5 files from the Junifer pipeline and
aggregates them into per-probe marker files for statistical analysis.

Pipeline Overview
-----------------
1. Load HDF5 marker file using JuniferHDF5Reader
2. Load corresponding events.tsv for epoch metadata
3. Parse event descriptions to extract probe information
4. For each probe: filter epochs, aggregate using trimmed mean
5. Save per-probe CSV files with channel-level marker values

Epoch Types
-----------
- **Evoked**: Trial-locked epochs. Aggregates last N go/correct trials
  based on distance_to_probe (1 = closest to probe; positive, 1-based).
  
- **State**: Temporal bins before probe. Aggregates all bins (dt=-3 to dt=-10).

- **Sleep**: Single epoch per probe. No aggregation needed.

Usage
-----
Process all subjects (sequential mode):
    python aggregate_markers_by_probe.py --config config.yaml

Process single subject (parallel mode):
    python aggregate_markers_by_probe.py --subject 03 --session a --task sartvisual

Combine partial results:
    python aggregate_markers_by_probe.py --finalize

Author: Wandering Mind EEG Analysis Pipeline
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# Junifer HDF5 reader, packaged as reader_picnic in the junifer-eeg-2 env.
from reader_picnic import JuniferHDF5Reader

# Local modules
from helpers import (
    load_yaml_config,
    enrich_events_with_parsed_fields,
    label_probe_onoff,
    label_probe_onoff,
    build_derivative_dir,
    make_derivative_fname,
    get_project_root,
)
from qa_report import AggregationStats, generate_html_report


# =============================================================================
# CONSTANTS
# =============================================================================

# Frequency bands for spectral markers
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_NAMES_6 = ["delta", "theta", "alpha", "beta", "jota", "gamma"]

# Channels to exclude from analysis
EOG_CHANNELS = ["VEOG", "HEOG"]


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def trimmean_agg(values: pd.Series, proportion: float = 0.15) -> float:
    """
    Compute trimmed mean, handling NaN values.
    
    Parameters
    ----------
    values : pd.Series
        Values to aggregate
    proportion : float
        Proportion to trim from each tail (0-0.5)
        
    Returns
    -------
    float
        Trimmed mean, or np.nan if no valid values
    """
    valid = values.dropna()
    if len(valid) == 0:
        return np.nan
    if len(valid) <= 2:
        # Not enough values to trim, use simple mean
        return float(np.mean(valid))
    return float(scipy.stats.trim_mean(valid, proportion))


# Aggregation function registry
# Each function takes a pd.Series and returns a scalar
def _agg_mean(values: pd.Series) -> float:
    valid = values.dropna()
    return float(np.mean(valid)) if len(valid) > 0 else np.nan

def _agg_std(values: pd.Series) -> float:
    valid = values.dropna()
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan

def _agg_median(values: pd.Series) -> float:
    valid = values.dropna()
    return float(np.median(valid)) if len(valid) > 0 else np.nan

def _agg_min(values: pd.Series) -> float:
    valid = values.dropna()
    return float(np.min(valid)) if len(valid) > 0 else np.nan

def _agg_max(values: pd.Series) -> float:
    valid = values.dropna()
    return float(np.max(valid)) if len(valid) > 0 else np.nan

def _agg_count(values: pd.Series) -> float:
    return float(values.dropna().count())


AGGREGATION_FUNCTIONS = {
    "mean": _agg_mean,
    "std": _agg_std,
    "median": _agg_median,
    "min": _agg_min,
    "max": _agg_max,
    "count": _agg_count,
    # trimmean is handled separately because it needs the proportion parameter
}


def rename_marker_name(original_name: str, marker_mapping: Dict[str, str]) -> str:
    """
    Rename verbose Junifer marker name to clean short name.
    
    Uses case-insensitive substring matching against the mapping patterns.
    
    Parameters
    ----------
    original_name : str
        Original marker name from H5 file
    marker_mapping : Dict[str, str]
        Pattern → clean name mapping from config
        
    Returns
    -------
    str
        Cleaned marker name, or original if no match found
    """
    for pattern, new_name in marker_mapping.items():
        if pattern.lower() in original_name.lower():
            return new_name
    return original_name


# =============================================================================
# DATA LOADING
# =============================================================================

def load_events_tsv(
    derivatives_root: str,
    subject: str,
    task: str,
    desc: str,
    session: Optional[str] = None,
) -> Optional[str]:
    """
    Build path to events.tsv file for a given epoch type.
    
    Parameters
    ----------
    derivatives_root : str
        Root derivatives directory
    subject : str
        Subject ID
    task : str
        Task name
    desc : str
        Epoch description (evoked, state, sleep)
    session : Optional[str]
        Session ID
        
    Returns
    -------
    Optional[str]
        Path to events.tsv if exists, None otherwise
    """
    out_dir = build_derivative_dir(derivatives_root, subject, datatype="eeg", session=session)
    fname = make_derivative_fname(subject, task, suffix="epo", desc=desc, extension=".fif", session=session)
    events_tsv = os.path.join(out_dir, fname).replace(".fif", "_events.tsv")
    
    return events_tsv if os.path.exists(events_tsv) else None


def _find_real_channel_names(reader: JuniferHDF5Reader, n_channels: int) -> Optional[List[str]]:
    """
    Find real channel names from any marker that has them.
    
    Some markers (like WSMI, ERP) don't store channel headers and return
    generic names (ch_0, ch_1, ...). This function searches for a marker
    with real channel names (Fz, Cz, Pz, etc.) and returns them.
    
    Parameters
    ----------
    reader : JuniferHDF5Reader
        H5 reader instance
    n_channels : int
        Expected number of channels
        
    Returns
    -------
    Optional[List[str]]
        Real channel names if found, None otherwise
    """
    for marker_name in reader.list_markers():
        info = reader.get_marker_info(marker_name)
        col_names = info.channel_names or []
        
        # Check if this marker has real (non-generic) channel names
        if col_names and len(col_names) == n_channels:
            has_real_names = any(
                not (name.startswith("ch_") and name[3:].isdigit())
                for name in col_names
            )
            if has_real_names:
                return col_names
    return None


def _rename_generic_columns(df: pd.DataFrame, real_names: List[str]) -> pd.DataFrame:
    """
    Rename generic column names (ch_0, ch_1, ...) to real channel names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potentially generic column names
    real_names : List[str]
        Real channel names to use
        
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns
    """
    current_cols = list(df.columns)
    n_expected = len(real_names)
    
    # Check if columns are generic (ch_0, ch_1, ...)
    generic_cols = [c for c in current_cols if c.startswith("ch_") and c[3:].isdigit()]
    
    if len(generic_cols) == n_expected:
        # Sort by index to ensure correct order
        sorted_generic = sorted(generic_cols, key=lambda x: int(x[3:]))
        rename_map = dict(zip(sorted_generic, real_names))
        return df.rename(columns=rename_map)
    
    return df


def load_marker_dataframes(
    reader: JuniferHDF5Reader,
    events_tsv_path: str,
    marker_mapping: Dict[str, str],
    stats: Optional["AggregationStats"] = None,
    context: str = "",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load all markers from H5 file into a combined long-format DataFrame.
    
    Creates a DataFrame with rows = epochs × channels and columns for
    each marker. Handles multi-band spectral and multi-feature sleep markers
    by creating separate columns for each band/feature.
    
    Parameters
    ----------
    reader : JuniferHDF5Reader
        Reader with loaded H5 file
    events_tsv_path : str
        Path to events.tsv file
    marker_mapping : Dict[str, str]
        Marker name mapping from config
    stats : Optional[AggregationStats]
        Stats tracker for QA report
    context : str
        Context string for error reporting (e.g., "sub-03 ses-a evoked")
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Combined DataFrame (long format: rows = epochs × channels)
        - List of marker column names
    """
    # Load events into reader
    reader.load_events(events_tsv_path)
    events_df = reader.events.copy()
    n_epochs_events = len(events_df)
    
    # Get channel info from first marker
    marker_names = reader.list_markers()
    first_info = reader.get_marker_info(marker_names[0])
    n_channels_raw = first_info.n_channels or len(first_info.channel_names or [])
    
    # Find a non-multi-band marker for accurate epoch count
    # Multi-band spectral markers have shape (bands, epochs, channels) which confuses n_epochs
    n_epochs_h5 = None
    for mk in marker_names:
        mk_info = reader.get_marker_info(mk)
        # Skip multi-band spectral markers (they have band_names)
        if mk_info.marker_type == "spectral" and mk_info.band_names and len(mk_info.band_names) > 1:
            continue
        n_epochs_h5 = mk_info.n_epochs
        break
    
    # Fallback to first marker if no suitable marker found
    if n_epochs_h5 is None:
        n_epochs_h5 = first_info.n_epochs
    
    # Debug: show epoch counts
    if n_epochs_h5 != n_epochs_events:
        print(f"  [DEBUG] H5 epochs: {n_epochs_h5}, events.tsv epochs: {n_epochs_events}")
        print(f"  [WARN] Epoch count mismatch! Check if epochs file was regenerated.")
        print(f"  [DEBUG] Marker shapes for diagnosis:")
        for mk in marker_names[:5]:
            mk_info = reader.get_marker_info(mk)
            print(f"    - {mk}: type={mk_info.marker_type}, epochs={mk_info.n_epochs}, shape={mk_info.shape}")
        
        # Track in QA stats
        if stats is not None:
            stats.add_epoch_mismatch(
                context=context,
                h5_epochs=n_epochs_h5,
                events_epochs=n_epochs_events,
                marker_type=context.split()[-1] if context else "",
            )
    
    # Use events.tsv epoch count as reference for metadata
    n_epochs = n_epochs_events
    
    # Find real channel names from any marker that has them
    channel_names = first_info.channel_names or []
    has_generic_names = not channel_names or all(
        name.startswith("ch_") and name[3:].isdigit() for name in channel_names
    )
    
    if has_generic_names:
        real_names = _find_real_channel_names(reader, n_channels_raw)
        if real_names:
            channel_names = real_names
            print(f"  [INFO] Using real channel names from another marker")
        else:
            channel_names = [f"ch_{i}" for i in range(n_channels_raw)]
            print(f"  [WARN] No real channel names found, using generic names")
    
    # Remove EOG channels
    channel_names = [ch for ch in channel_names if ch not in EOG_CHANNELS]
    n_channels = len(channel_names)
    
    print(f"  Loaded events: {n_epochs} epochs")
    print(f"  Channels: {n_channels} (filtered EOG)")
    print(f"  Markers to process: {len(marker_names)}")
    
    # Build long-format DataFrame: (epoch × channel) rows
    epoch_indices = np.repeat(np.arange(n_epochs), n_channels)
    channel_repeated = np.tile(channel_names, n_epochs)
    
    combined_df = pd.DataFrame({
        "epoch_idx": epoch_indices,
        "channel": channel_repeated,
    })
    
    # Add event metadata columns
    for col in events_df.columns:
        combined_df[col] = events_df[col].values[epoch_indices]
    
    # Process each marker
    marker_columns = []
    expected_len = len(combined_df)  # n_epochs * n_channels
    
    for marker_name in marker_names:
        info = reader.get_marker_info(marker_name)
        clean_name = rename_marker_name(marker_name, marker_mapping)
        
        # Handle multi-band spectral markers
        if info.marker_type == "spectral" and info.band_names and len(info.band_names) > 1:
            for band in info.band_names:
                col_name = f"{clean_name}_{band}"
                df = _safe_load_marker(reader, marker_name, channel_names, band=band)
                if df is not None:
                    flat_vals = df.values.flatten(order="C")
                    if len(flat_vals) == expected_len:
                        combined_df[col_name] = flat_vals
                        marker_columns.append(col_name)
                    else:
                        print(f"    [SKIP] {col_name}: length {len(flat_vals)} != expected {expected_len} "
                              f"(marker epochs={info.n_epochs}, events epochs={n_epochs})")
                        if stats is not None:
                            stats.add_skipped_marker(
                                context=context, marker_name=col_name,
                                expected_length=expected_len, actual_length=len(flat_vals),
                            )
                    
        # Handle multi-feature sleep markers
        elif info.marker_type == "sleep" and info.feature_names and len(info.feature_names) > 1:
            for feature in info.feature_names:
                col_name = f"{clean_name}_{feature.lower().replace(' ', '_')}"
                df = _safe_load_marker(reader, marker_name, channel_names, feature=feature)
                if df is not None:
                    flat_vals = df.values.flatten(order="C")
                    if len(flat_vals) == expected_len:
                        combined_df[col_name] = flat_vals
                        marker_columns.append(col_name)
                    else:
                        print(f"    [SKIP] {col_name}: length {len(flat_vals)} != expected {expected_len} "
                              f"(marker epochs={info.n_epochs}, events epochs={n_epochs})")
                        if stats is not None:
                            stats.add_skipped_marker(
                                context=context, marker_name=col_name,
                                expected_length=expected_len, actual_length=len(flat_vals),
                            )
                    
        # Handle simple markers
        else:
            col_name = clean_name
            df = _safe_load_marker(reader, marker_name, channel_names)
            if df is not None:
                flat_vals = df.values.flatten(order="C")
                if len(flat_vals) == expected_len:
                    combined_df[col_name] = flat_vals
                    marker_columns.append(col_name)
                else:
                    print(f"    [SKIP] {col_name}: length {len(flat_vals)} != expected {expected_len} "
                          f"(marker epochs={info.n_epochs}, events epochs={n_epochs})")
                    if stats is not None:
                        stats.add_skipped_marker(
                            context=context, marker_name=col_name,
                            expected_length=expected_len, actual_length=len(flat_vals),
                        )
    
    print(f"  Successfully loaded {len(marker_columns)} marker columns")
    return combined_df, marker_columns


def _safe_load_marker(
    reader: JuniferHDF5Reader,
    marker_name: str,
    channel_names: List[str],
    band: Optional[str] = None,
    feature: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Safely load a marker DataFrame, handling generic column names.
    
    If the marker has generic column names (ch_0, ch_1, ...), they are
    renamed to the expected real channel names before filtering.
    
    Parameters
    ----------
    reader : JuniferHDF5Reader
        H5 reader instance
    marker_name : str
        Name of marker to load
    channel_names : List[str]
        Expected channel names (real names like Fz, Cz, Pz)
    band : Optional[str]
        Band name for spectral markers
    feature : Optional[str]
        Feature name for sleep markers
        
    Returns
    -------
    Optional[pd.DataFrame]
        Marker DataFrame with columns matching channel_names, or None if failed
    """
    try:
        # Load marker data as DataFrame
        if band:
            df = reader.to_dataframe(marker_name, band=band)
        elif feature:
            df = reader.to_dataframe(marker_name, feature=feature)
        else:
            df = reader.to_dataframe(marker_name)
        
        # Rename generic columns to real channel names if needed
        df = _rename_generic_columns(df, channel_names)
        
        # Filter to requested channels
        available = [ch for ch in channel_names if ch in df.columns]
        if not available:
            print(f"    [WARN] No matching channels for {marker_name}")
            return None
        
        return df[available]
        
    except Exception as e:
        context = f"band={band}" if band else f"feature={feature}" if feature else ""
        print(f"    [WARN] Failed to load {marker_name} {context}: {e}")
        return None


# =============================================================================
# EPOCH FILTERING
# =============================================================================

def filter_epochs_for_probe(
    df: pd.DataFrame,
    probe_number: int,
    epoch_type: str,
    distance_min: int = 1,
    distance_max: int = 5,
    filter_go: bool = True,
    filter_correct: bool = True,
) -> pd.DataFrame:
    """
    Filter DataFrame to epochs belonging to a specific probe.
    
    For evoked epochs, also filters by trial type, correctness, and distance.
    For state and sleep epochs, only filters by probe number.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with all epochs
    probe_number : int
        Target probe number
    epoch_type : str
        "evoked", "state", or "sleep"
    distance_min : int
        Minimum distance to include (evoked only; 1-based, 1 = closest).
        Inclusive.
    distance_max : int
        Maximum distance to include (evoked only). Inclusive, so
        distance_min=1, distance_max=5 selects the last 5 trials.
    only_go_correct : bool
        Filter for go/correct trials only (evoked only)
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    filtered = df[df["probe_number"] == probe_number].copy()
    
    if epoch_type == "evoked":
        # Filter by trial type and correctness
        if filter_go:
            filtered = filtered[filtered["trial_type"] == "go"]
        
        if filter_correct:
            filtered = filtered[filtered["correctness"] == "correct"]
        # Filter by distance to probe
        if "distance_to_probe" in filtered.columns:
            filtered = filtered[
                (filtered["distance_to_probe"] >= distance_min) &
                (filtered["distance_to_probe"] <= distance_max)
            ]
    
    return filtered


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_probe_markers(
    df: pd.DataFrame,
    marker_columns: List[str],
    aggregation_methods: List[str],
    trimmean_proportion: float = 0.15,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate marker values by channel using multiple methods.
    
    Groups by channel and computes the requested aggregation methods
    across epochs for each marker column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Filtered DataFrame for a single probe (long format)
    marker_columns : List[str]
        Marker column names to aggregate
    aggregation_methods : List[str]
        List of aggregation methods to compute (e.g., ["trimmean", "std"])
    trimmean_proportion : float
        Proportion to trim from each tail (for trimmean method)
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Aggregated DataFrame with rows=channels, columns=marker_method
        - List of aggregated column names
    """
    if df.empty or not marker_columns:
        return pd.DataFrame(), []
    
    # Build aggregation functions for all methods and markers
    agg_funcs = {}
    output_columns = []
    
    for marker in marker_columns:
        for method in aggregation_methods:
            # Create output column name: marker_method
            out_col = f"{marker}_{method}"
            output_columns.append(out_col)
            
            if method == "trimmean":
                # Trimmean needs the proportion parameter
                agg_funcs[out_col] = pd.NamedAgg(
                    column=marker,
                    aggfunc=lambda x, prop=trimmean_proportion: trimmean_agg(x, prop)
                )
            elif method in AGGREGATION_FUNCTIONS:
                agg_funcs[out_col] = pd.NamedAgg(
                    column=marker,
                    aggfunc=AGGREGATION_FUNCTIONS[method]
                )
            else:
                print(f"    [WARN] Unknown aggregation method: {method}")
                output_columns.pop()  # Remove from output list
    
    if not agg_funcs:
        return pd.DataFrame(), []
    
    # Perform aggregation
    result = df.groupby("channel").agg(**agg_funcs).reset_index()
    
    return result, output_columns


def aggregate_across_channels(
    df: pd.DataFrame,
    marker_columns: List[str],
    aggregation_methods: List[str],
    trimmean_proportion: float = 0.15,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate across channels to produce a single row per probe.
    
    Second-level aggregation that collapses per-channel values into
    a scalar per marker. Output columns are named: marker_ch_method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Per-channel aggregated DataFrame (from aggregate_probe_markers)
    marker_columns : List[str]
        Marker column names to aggregate (from level 1 aggregation)
    aggregation_methods : List[str]
        Aggregation methods to apply (e.g., ["mean", "std"])
    trimmean_proportion : float
        Proportion to trim from each tail (for trimmean method)
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Single-row DataFrame with aggregated values
        - List of output column names
    """
    if df.empty or not marker_columns:
        return pd.DataFrame(), []
    
    result_data = {}
    output_columns = []
    
    for marker in marker_columns:
        if marker not in df.columns:
            continue
            
        values = df[marker]
        
        for method in aggregation_methods:
            # Create output column name: marker_ch_method
            out_col = f"{marker}_ch_{method}"
            output_columns.append(out_col)
            
            if method == "trimmean":
                result_data[out_col] = trimmean_agg(values, trimmean_proportion)
            elif method in AGGREGATION_FUNCTIONS:
                result_data[out_col] = AGGREGATION_FUNCTIONS[method](values)
            else:
                print(f"    [WARN] Unknown channel aggregation method: {method}")
                output_columns.pop()
    
    if not result_data:
        return pd.DataFrame(), []
    
    return pd.DataFrame([result_data]), output_columns


def aggregate_by_rois(
    df: pd.DataFrame,
    marker_columns: List[str],
    aggregation_methods: List[str],
    rois: Dict[str, List[str]],
    default_rois: Optional[List[str]],
    marker_rois: Dict[str, List[str]],
    trimmean_proportion: float = 0.15,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate channels by ROI (Region of Interest).
    
    For each marker, aggregates channels within each specified ROI using
    the given aggregation methods. Creates columns named: marker_roi_method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Per-channel DataFrame with 'channel' column
    marker_columns : List[str]
        Marker columns from trial-level aggregation (e.g., power_alpha_trimmean)
    aggregation_methods : List[str]
        Methods to aggregate within each ROI (e.g., ["mean", "std"])
    rois : Dict[str, List[str]]
        ROI definitions mapping ROI name to channel list
    default_rois : Optional[List[str]]
        Default ROIs to use for markers not in marker_rois
        None = aggregate all channels together (no ROI separation)
    marker_rois : Dict[str, List[str]]
        Per-marker ROI overrides. Key is marker name (or prefix), value is ROI list
    trimmean_proportion : float
        Proportion for trimmean method
        
    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        - Single-row DataFrame with ROI-aggregated values
        - List of output column names
    """
    if df.empty or not marker_columns or "channel" not in df.columns:
        return pd.DataFrame(), []
    
    result_data = {}
    output_columns = []
    available_channels = set(df["channel"].unique())
    
    for marker in marker_columns:
        if marker not in df.columns:
            continue
        
        # Determine which ROIs to use for this marker
        # Check if any key in marker_rois is a prefix of the marker name
        marker_roi_list = None
        for marker_key, roi_list in marker_rois.items():
            if marker.startswith(marker_key):
                marker_roi_list = roi_list
                break
        
        if marker_roi_list is None:
            marker_roi_list = default_rois
        
        # If no ROIs specified, aggregate all channels together (no ROI label needed)
        if not marker_roi_list:
            for method in aggregation_methods:
                out_col = f"{marker}_{method}"  # No ROI label for default (all channels)
                output_columns.append(out_col)
                values = df[marker]
                
                if method == "trimmean":
                    result_data[out_col] = trimmean_agg(values, trimmean_proportion)
                elif method in AGGREGATION_FUNCTIONS:
                    result_data[out_col] = AGGREGATION_FUNCTIONS[method](values)
            continue
        
        # Aggregate within each ROI
        for roi_name in marker_roi_list:
            if roi_name not in rois:
                print(f"    [WARN] ROI '{roi_name}' not defined, skipping for {marker}")
                continue
            
            roi_channels = rois[roi_name]
            if roi_channels is None:
                # null means all channels
                roi_mask = df["channel"].isin(available_channels)
            else:
                roi_mask = df["channel"].isin(roi_channels)
            
            roi_df = df[roi_mask]
            
            if roi_df.empty:
                continue
            
            values = roi_df[marker]
            
            for method in aggregation_methods:
                out_col = f"{marker}_{roi_name}_{method}"
                output_columns.append(out_col)
                
                if method == "trimmean":
                    result_data[out_col] = trimmean_agg(values, trimmean_proportion)
                elif method in AGGREGATION_FUNCTIONS:
                    result_data[out_col] = AGGREGATION_FUNCTIONS[method](values)
                else:
                    print(f"    [WARN] Unknown aggregation method: {method}")
                    output_columns.pop()
    
    if not result_data:
        return pd.DataFrame(), []
    
    return pd.DataFrame([result_data]), output_columns


def convert_to_long_format(
    df: pd.DataFrame,
    marker_columns: List[str],
    id_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert wide format to long/tidy format.
    
    Transforms columns like 'power_alpha_trimmean', 'power_alpha_std' into
    separate rows with 'marker', 'method', 'value' columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format DataFrame with marker columns
    marker_columns : List[str]
        List of marker column names (e.g., ['power_alpha_trimmean', 'power_alpha_std'])
    id_columns : Optional[List[str]]
        Columns to keep as identifiers (e.g., ['channel', 'subject'])
        If None, uses all non-marker columns
        
    Returns
    -------
    pd.DataFrame
        Long format with columns: id_columns + [marker, method, value]
    """
    if df.empty or not marker_columns:
        return df
    
    # Determine ID columns (all columns except marker columns)
    if id_columns is None:
        id_columns = [c for c in df.columns if c not in marker_columns]
    
    # Melt to long format
    long_df = df.melt(
        id_vars=id_columns,
        value_vars=[c for c in marker_columns if c in df.columns],
        var_name="marker_method",
        value_name="value",
    )
    
    # Split marker_method into marker and method
    # Pattern: marker_name_method (e.g., power_alpha_trimmean, wsmi_theta_std)
    # The method is the last part after the last underscore
    def split_marker_method(col_name: str) -> Tuple[str, str]:
        """Split column name into marker and method."""
        # Known methods (order matters - check longer names first)
        methods = ["trimmean", "median", "count", "mean", "std", "min", "max"]
        for method in methods:
            if col_name.endswith(f"_{method}"):
                marker = col_name[: -(len(method) + 1)]
                return marker, method
        # Fallback: last underscore split
        parts = col_name.rsplit("_", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (col_name, "value")
    
    # Apply split
    split_results = long_df["marker_method"].apply(split_marker_method)
    long_df["marker"] = split_results.apply(lambda x: x[0])
    long_df["method"] = split_results.apply(lambda x: x[1])
    long_df = long_df.drop(columns=["marker_method"])
    
    # Reorder columns: id_columns, marker, method, value
    final_cols = id_columns + ["marker", "method", "value"]
    final_cols = [c for c in final_cols if c in long_df.columns]
    
    return long_df[final_cols]


def apply_nan_policy(
    df: pd.DataFrame,
    marker_columns: List[str],
    nan_policy: str = "null",
) -> pd.DataFrame:
    """Apply the configured NaN-handling policy with a Density guard-rail.

    Two-layer logic:

    1. **Density guard-rail** (always applied, irrespective of ``nan_policy``):
       any column whose name contains ``density`` (case-insensitive) has its
       NaN cells filled with ``0.0``. Density is a count: a (probe, channel)
       cell with no above-threshold waves means 0 waves, not "missing".

    2. **Configured policy** for non-Density markers:
       - ``"null"`` → keep NaN (recommended: Pinggal-correct, no bias)
       - ``"zero"`` → fill 0 (biases continuous SW features toward 0; kept
         only for ablations)
       - ``"mean"`` / ``"median"`` → fill with column statistic

    Filling PTP/Frequency/Duration/Slope with 0 would treat absent waves as
    zero-amplitude waves and bias downstream means/LMMs/CBPT — that is why
    ``"null"`` is the recommended setting.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame.
    marker_columns : list of str
        Marker columns to process.
    nan_policy : str
        ``"null"`` (keep NaN), ``"zero"``, ``"mean"``, or ``"median"``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with Density NaNs always set to 0 and the configured
        policy applied to the rest.
    """
    if df.empty:
        return df

    result = df.copy()

    # Layer 1 — Density columns are counts, always 0-fill.
    density_cols = [c for c in marker_columns if "density" in c.lower()]
    for col in density_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)

    # Layer 2 — Configured policy for the remaining markers.
    if nan_policy == "null":
        return result

    for col in marker_columns:
        if col in density_cols or col not in result.columns:
            continue

        values = result[col].values
        nan_mask = np.isnan(values)
        if not nan_mask.any():
            continue

        valid = values[~nan_mask]
        if len(valid) == 0:
            continue

        fill_value = {
            "zero": 0.0,
            "mean": float(np.mean(valid)),
            "median": float(np.median(valid)),
        }.get(nan_policy)

        if fill_value is not None:
            values[nan_mask] = fill_value
            result[col] = values

    return result


# =============================================================================
# OUTPUT
# =============================================================================

def save_probe_csv(
    df: pd.DataFrame,
    features_root: str,
    subject: str,
    task: str,
    probe_number: int,
    label: str,
    epoch_type: str,
    probe_metadata: Dict,
    n_epochs_aggregated: int,
    session: Optional[str] = None,
    overwrite: bool = True,
) -> str:
    """
    Save aggregated probe markers to CSV file.
    
    Output filename format:
        sub-{subject}_ses-{session}_task-{task}_probe-{NNN}_{label}_{epoch_type}.csv
    
    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame (rows=channels, columns=markers)
    features_root : str
        Root directory for output
    subject : str
        Subject ID
    task : str
        Task name
    probe_number : int
        Probe number
    label : str
        On-task/off-task label
    epoch_type : str
        "evoked", "state", or "sleep"
    probe_metadata : Dict
        Additional probe metadata columns
    n_epochs_aggregated : int
        Number of epochs that were aggregated
    session : Optional[str]
        Session ID
    overwrite : bool
        Overwrite existing files
        
    Returns
    -------
    str
        Path to saved file
    """
    # Build output path
    if session:
        out_dir = Path(features_root) / f"sub-{subject}" / f"ses-{session}" / "eeg" / "junifer_aggregated"
        fname = f"sub-{subject}_ses-{session}_task-{task}_probe-{probe_number:03d}_{label}_{epoch_type}.csv"
    else:
        out_dir = Path(features_root) / f"sub-{subject}" / "eeg" / "junifer_aggregated"
        fname = f"sub-{subject}_task-{task}_probe-{probe_number:03d}_{label}_{epoch_type}.csv"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname
    
    if out_path.exists() and not overwrite:
        print(f"    [SKIP] {out_path} exists")
        return str(out_path)
    
    # Add metadata columns
    result = df.copy()
    metadata_cols = ["subject", "task", "probe_number", "label", "n_epochs"]
    if session:
        metadata_cols.insert(1, "session")
    
    result.insert(0, "subject", subject)
    col_idx = 1
    if session:
        result.insert(col_idx, "session", session)
        col_idx += 1
    result.insert(col_idx, "task", task)
    result.insert(col_idx + 1, "probe_number", probe_number)
    result.insert(col_idx + 2, "label", label)
    result.insert(col_idx + 3, "n_epochs", n_epochs_aggregated)
    
    # Add probe metadata
    for key, value in probe_metadata.items():
        if key not in result.columns:
            result[key] = value
    
    result.to_csv(out_path, index=False)
    
    file_size = out_path.stat().st_size
    n_markers = len([c for c in result.columns if c not in metadata_cols + ["channel"] + list(probe_metadata.keys())])
    print(f"    [SAVE] {out_path.name} ({file_size:,} bytes, {n_markers} markers)")
    
    return str(out_path)


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_epoch_type(
    h5_path: Path,
    events_tsv_path: str,
    epoch_type: str,
    cfg: Dict,
    subject: str,
    task: str,
    session: Optional[str],
    probe_labels: pd.Series,
    stats: Optional[AggregationStats] = None,
) -> List[Dict]:
    """
    Process all probes for a single epoch type.
    
    Parameters
    ----------
    h5_path : Path
        Path to H5 marker file
    events_tsv_path : str
        Path to events.tsv file
    epoch_type : str
        "evoked", "state", or "sleep"
    cfg : Dict
        Configuration dictionary
    subject : str
        Subject ID
    task : str
        Task name
    session : Optional[str]
        Session ID
    probe_labels : pd.Series
        Pre-computed probe labels
    stats : Optional[AggregationStats]
        Statistics tracker
        
    Returns
    -------
    List[Dict]
        List of output records for summary
    """
    print(f"\n  Processing {epoch_type.upper()} markers")
    
    # Extract configuration
    trial_cfg = cfg.get("trial_selection", {})
    # Legacy support: if only_go_correct is present, use it to set both defaults
    if "only_go_correct" in trial_cfg:
        default_filter = bool(trial_cfg["only_go_correct"])
        filter_go = trial_cfg.get("filter_go", default_filter)
        filter_correct = trial_cfg.get("filter_correct", default_filter)
    else:
        filter_go = bool(trial_cfg.get("filter_go", True))
        filter_correct = bool(trial_cfg.get("filter_correct", True))
    evoked_dist_min = int(trial_cfg.get("evoked_distance_min", 0))
    evoked_dist_max = int(trial_cfg.get("evoked_distance_max", 4))
    min_epochs_required = int(trial_cfg.get("min_required_distances", 3))
    
    agg_cfg = cfg.get("aggregation", {})
    # Level 1: epochs → channels
    trial_methods = agg_cfg.get("trial_methods", agg_cfg.get("methods", ["trimmean"]))
    # Level 2: channels → ROI (optional, null/empty = skip)
    channel_methods = agg_cfg.get("channel_methods", None) or []
    # ROI configuration
    rois = agg_cfg.get("rois", {})
    default_rois = agg_cfg.get("default_rois", None)
    marker_rois = agg_cfg.get("marker_rois", {})
    trimmean_percent = float(agg_cfg.get("trimmean_percent", 30.0))
    trimmean_proportion = trimmean_percent / 200  # Convert to scipy proportion
    
    out_cfg = cfg.get("output", {})
    overwrite = bool(out_cfg.get("overwrite", True))
    output_format = str(out_cfg.get("output_format", "wide")).lower()
    
    project = cfg.get("project", {})
    features_root = project.get("features_root")
    # Resolve paths relative to project root
    project_root = get_project_root()
    if features_root and not Path(features_root).is_absolute():
        features_root = str(project_root / features_root)
        
    marker_mapping = cfg.get("marker_name_mapping", {})
    
    sleep_cfg = cfg.get("sleep_markers", {})
    sleep_nan_policy = str(sleep_cfg.get("nan_policy", "null")).lower()
    
    outputs = []
    
    # Load H5 file
    try:
        reader = JuniferHDF5Reader(h5_path)
        print(f"    Loaded H5: {h5_path.name}")
        print(f"    Available markers: {reader.list_markers()}")
    except Exception as e:
        print(f"    [ERROR] Failed to load H5: {e}")
        if stats:
            stats.add_error(
                context=f"sub-{subject} ses-{session} task-{task} {epoch_type}",
                message=f"Failed to load H5: {e}"
            )
        return outputs
    
    # Load all markers into combined DataFrame
    context_str = f"sub-{subject} ses-{session} task-{task} {epoch_type}"
    try:
        combined_df, marker_columns = load_marker_dataframes(
            reader=reader,
            events_tsv_path=events_tsv_path,
            marker_mapping=marker_mapping,
            stats=stats,
            context=context_str,
        )
    except Exception as e:
        print(f"    [ERROR] Failed to load marker DataFrames: {e}")
        import traceback
        traceback.print_exc()
        if stats:
            stats.add_error(
                context=f"sub-{subject} ses-{session} task-{task} {epoch_type}",
                message=f"Failed to load marker DataFrames: {e}"
            )
        return outputs
    
    # Enrich with parsed event fields
    combined_df = enrich_events_with_parsed_fields(combined_df)
    
    # Get unique probes
    probes = combined_df["probe_number"].dropna().unique()
    print(f"    Found {len(probes)} probes")
    
    # Process each probe
    for probe_num in sorted(probes):
        probe_num = int(probe_num)
        
        # Get probe label
        if hasattr(probe_labels, "get"):
            label = probe_labels.get(probe_num, "unknown")
        elif probe_num in probe_labels.index:
            label = probe_labels.loc[probe_num]
        else:
            label = "unknown"
        label_str = str(label)
        
        # Filter epochs for this probe
        filtered_df = filter_epochs_for_probe(
            df=combined_df,
            probe_number=probe_num,
            epoch_type=epoch_type,
            distance_min=evoked_dist_min,
            distance_max=evoked_dist_max,
            filter_go=filter_go,
            filter_correct=filter_correct,
        )
        
        n_epochs = filtered_df["epoch_idx"].nunique()
        
        # Validate epoch count
        if epoch_type == "evoked" and n_epochs < min_epochs_required:
            print(f"    Probe {probe_num}: insufficient epochs ({n_epochs} < {min_epochs_required})")
            if stats:
                stats.add_discarded_probe(
                    subject=subject, session=session, task=task,
                    probe_number=probe_num, epoch_type=epoch_type,
                    n_epochs=n_epochs, reason=f"Insufficient epochs ({n_epochs} < {min_epochs_required})"
                )
            continue
        
        if n_epochs == 0:
            print(f"    Probe {probe_num}: no epochs found")
            if stats:
                stats.add_discarded_probe(
                    subject=subject, session=session, task=task,
                    probe_number=probe_num, epoch_type=epoch_type,
                    n_epochs=0, reason="No epochs found"
                )
            continue
        
        # Level 1: Aggregate markers by channel (epochs → channels)
        aggregated_df, aggregated_columns = aggregate_probe_markers(
            df=filtered_df,
            marker_columns=marker_columns,
            aggregation_methods=trial_methods,
            trimmean_proportion=trimmean_proportion,
        )
        
        if aggregated_df.empty:
            print(f"    Probe {probe_num}: aggregation failed")
            if stats:
                stats.add_discarded_probe(
                    subject=subject, session=session, task=task,
                    probe_number=probe_num, epoch_type=epoch_type,
                    n_epochs=n_epochs, reason="Aggregation returned empty DataFrame"
                )
            continue
        
        # Apply NaN policy for sleep markers (before channel aggregation)
        if epoch_type == "sleep":
            aggregated_df = apply_nan_policy(
                df=aggregated_df,
                marker_columns=aggregated_columns,
                nan_policy=sleep_nan_policy,
            )
        
        # Level 2: Aggregate across channels by ROI if configured
        if channel_methods:
            # Keep per-channel data for QA topoplots
            perchannel_df = aggregated_df.copy()
            perchannel_columns = aggregated_columns
            
            # Create ROI-aggregated version for analysis output
            final_df, final_columns = aggregate_by_rois(
                df=aggregated_df,
                marker_columns=aggregated_columns,
                aggregation_methods=channel_methods,
                rois=rois,
                default_rois=default_rois,
                marker_rois=marker_rois,
                trimmean_proportion=trimmean_proportion,
            )
        else:
            perchannel_df = aggregated_df
            perchannel_columns = aggregated_columns
            final_df = aggregated_df
            final_columns = aggregated_columns
        
        # Extract probe metadata
        probe_row = filtered_df.iloc[0]
        probe_metadata = {
            "ontask_label": probe_row.get("ontask_label", ""),
            "content": probe_row.get("content", ""),
            "confidence_level": probe_row.get("confidence_level", ""),
            "depth_level": probe_row.get("depth_level", ""),
            "onoff": probe_row.get("onoff", None),
            "valence": probe_row.get("valence", None),
            "selfother": probe_row.get("selfother", None),
            "time": probe_row.get("time", None),
            "confidence": probe_row.get("confidence", None),
        }
        
        # Convert to long format if requested
        if output_format == "long":
            perchannel_df = convert_to_long_format(
                df=perchannel_df,
                marker_columns=perchannel_columns,
            )
        
        # Save per-channel data (always - used for QA topoplots)
        perchannel_path = save_probe_csv(
            df=perchannel_df,
            features_root=features_root,
            subject=subject,
            task=task,
            probe_number=probe_num,
            label=label_str,
            epoch_type=epoch_type,
            probe_metadata=probe_metadata,
            n_epochs_aggregated=n_epochs,
            session=session,
            overwrite=overwrite,
        )
        
        # If channel aggregation is enabled, also save the aggregated version
        out_path = perchannel_path  # Default output path for summary
        if channel_methods and not final_df.empty:
            # Save aggregated data with "_agg" suffix
            agg_path_str = perchannel_path.replace(".csv", "_agg.csv")
            
            # Add metadata columns to aggregated data (same as per-channel)
            agg_df = final_df.copy()
            agg_df.insert(0, "subject", subject)
            col_idx = 1
            if session:
                agg_df.insert(col_idx, "session", session)
                col_idx += 1
            agg_df.insert(col_idx, "task", task)
            agg_df.insert(col_idx + 1, "probe_number", probe_num)
            agg_df.insert(col_idx + 2, "label", label_str)
            agg_df.insert(col_idx + 3, "n_epochs", n_epochs)
            
            # Add probe metadata
            for key, value in probe_metadata.items():
                if key not in agg_df.columns:
                    agg_df[key] = value
            
            agg_df.to_csv(agg_path_str, index=False)
        
        outputs.append({
            "subject": subject,
            "session": session or "",
            "task": task,
            "probe_number": probe_num,
            "label": label_str,
            "marker_type": epoch_type,
            "n_markers": len(marker_columns),
            "n_epochs": n_epochs,
            "output_path": out_path,  # Points to per-channel data for QA topoplots
        })
    
    print(f"    Processed {len(outputs)} probes for {epoch_type}")
    return outputs


def process_subject_task(
    cfg: Dict,
    subject: str,
    task: str,
    session: Optional[str] = None,
    stats: Optional[AggregationStats] = None,
) -> pd.DataFrame:
    """
    Process all epoch types for a single subject-task-session combination.
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary
    subject : str
        Subject ID
    task : str
        Task name
    session : Optional[str]
        Session ID
    stats : Optional[AggregationStats]
        Statistics tracker
        
    Returns
    -------
    pd.DataFrame
        Summary of processed probes
    """
    project = cfg.get("project", {})
    
    # Resolve paths relative to project root
    project_root = get_project_root()
    
    derivatives_root = project.get("derivatives_root")
    if derivatives_root and not Path(derivatives_root).is_absolute():
        derivatives_root = str(project_root / derivatives_root)
        
    features_root = project.get("features_root")
    if features_root and not Path(features_root).is_absolute():
        features_root = str(project_root / features_root)
    
    epoch_types = [
        ("evoked", project.get("input_evoked_desc", "evoked")),
        ("state", project.get("input_state_desc", "state")),
        ("sleep", project.get("input_sleep_desc", "sleep")),
    ]
    
    labeling = cfg.get("labeling", {})
    onoff_threshold = int(labeling.get("onoff_threshold", 50))
    
    session_str = f" ses-{session}" if session else ""
    
    print(f"\n{'='*80}")
    print(f"Processing sub-{subject}{session_str} task-{task}")
    print(f"{'='*80}")
    
    # Find first available events.tsv for probe label extraction
    events_for_labels = None
    for _, desc in epoch_types:
        events_tsv = load_events_tsv(derivatives_root, subject, task, desc, session)
        if events_tsv:
            events_for_labels = events_tsv
            break
    
    if not events_for_labels:
        print(f"[ERROR] No events TSV found for sub-{subject}{session_str} task-{task}")
        if stats:
            stats.add_error(
                context=f"sub-{subject}{session_str} task-{task}",
                message="No events TSV found for any epoch type"
            )
        return pd.DataFrame()
    
    # Load events and compute probe labels
    events_df = pd.read_csv(events_for_labels, sep="\t")
    events_df = enrich_events_with_parsed_fields(events_df)
    probe_labels = label_probe_onoff(events_df, onoff_threshold)
    
    # Process each epoch type
    all_outputs = []
    junifer_dir = Path(features_root) / "junifer"
    
    for epoch_type, desc in epoch_types:
        # Build H5 path
        if session:
            h5_fname = f"element_sub-{subject}_ses-{session}_{task}_{desc}_markers.h5"
        else:
            h5_fname = f"element_sub-{subject}_{task}_{desc}_markers.h5"
        
        h5_path = junifer_dir / h5_fname
        
        if not h5_path.exists():
            print(f"\n  [WARN] {epoch_type} H5 not found: {h5_path}")
            if stats:
                stats.add_missing_epoch_type(
                    subject=subject,
                    session=session,
                    task=task,
                    epoch_type=epoch_type,
                    reason=f"H5 file not found: {h5_fname}"
                )
            continue
        
        # Get events TSV for this epoch type
        events_tsv = load_events_tsv(derivatives_root, subject, task, desc, session)
        if not events_tsv:
            print(f"\n  [WARN] {epoch_type} events TSV not found")
            if stats:
                stats.add_missing_epoch_type(
                    subject=subject,
                    session=session,
                    task=task,
                    epoch_type=epoch_type,
                    reason=f"Events TSV not found for {desc} epochs"
                )
            continue
        
        # Process this epoch type
        outputs = process_epoch_type(
            h5_path=h5_path,
            events_tsv_path=events_tsv,
            epoch_type=epoch_type,
            cfg=cfg,
            subject=subject,
            task=task,
            session=session,
            probe_labels=probe_labels,
            stats=stats,
        )
        all_outputs.extend(outputs)
    
    return pd.DataFrame(all_outputs)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for aggregation pipeline."""
    parser = argparse.ArgumentParser(
        description="Aggregate Junifer markers by probe for mind-wandering analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all subjects:
    python aggregate_markers_by_probe.py --config config.yaml
    
  Process single subject (for parallel jobs):
    python aggregate_markers_by_probe.py --subject 03 --session a --task sartvisual
    
  Combine partial results after parallel processing:
    python aggregate_markers_by_probe.py --finalize
        """
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config file")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--task", help="Process only this task")
    parser.add_argument("--session", help="Process only this session (e.g., 'a' or 'b')")
    parser.add_argument("--no-report", action="store_true", help="Skip QA report generation")
    parser.add_argument("--finalize", action="store_true", help="Combine partial summaries and generate final report")
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    project = cfg.get("project", {})
    features_root = project.get("features_root")
    
    # Resolve paths relative to project root
    project_root = get_project_root()
    if features_root and not Path(features_root).is_absolute():
        features_root = str(project_root / features_root)
    
    # =========================================================================
    # FINALIZE MODE: Combine partial summaries
    # =========================================================================
    if args.finalize:
        _run_finalize_mode(cfg, features_root)
        return
    
    # =========================================================================
    # NORMAL MODE: Process subjects
    # =========================================================================
    _run_processing_mode(args, cfg, features_root)


def _run_finalize_mode(cfg: Dict, features_root: str) -> None:
    """Combine partial summaries and generate QA report."""
    print("=" * 80)
    print("FINALIZE MODE: Combining partial summaries")
    print("=" * 80)
    
    # Search for partial summaries in all subject folders
    # Pattern: features_root/sub-*/ses-*/eeg/junifer_aggregated/summary_*.csv
    features_path = Path(features_root)
    partial_files = list(features_path.glob("sub-*/ses-*/eeg/junifer_aggregated/summary_*.csv"))
    
    # Also check for files without session (sub-*/eeg/junifer_aggregated/)
    partial_files.extend(features_path.glob("sub-*/eeg/junifer_aggregated/summary_*.csv"))
    
    print(f"Found {len(partial_files)} partial summary files")
    
    if not partial_files:
        print("[ERROR] No partial summary files found in subject folders")
        print(f"  Searched: {features_path}/sub-*/ses-*/eeg/junifer_aggregated/summary_*.csv")
        return
    
    # Combine all partial summaries
    all_dfs = []
    for pf in partial_files:
        try:
            all_dfs.append(pd.read_csv(pf))
        except Exception as e:
            print(f"[WARN] Could not read {pf}: {e}")
    
    if not all_dfs:
        print("[ERROR] No valid partial summaries found")
        return
    
    summary_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (in case of reruns)
    key_cols = ["subject", "session", "task", "probe_number", "marker_type"]
    summary_df = summary_df.drop_duplicates(subset=key_cols, keep="last")
    
    # Save combined summary
    summary_path = Path(features_root) / "junifer_probe_aggregation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n[SUMMARY] Saved combined summary to: {summary_path}")
    print(f"Total probes: {len(summary_df)}")
    print(f"Unique subjects: {summary_df['subject'].nunique()}")
    print(f"Unique sessions: {summary_df['session'].nunique()}")
    print(f"Unique tasks: {summary_df['task'].nunique()}")
    
    # Generate QA report
    stats = AggregationStats()
    report_path = Path(features_root) / "junifer_aggregation_qa_report.html"
    generate_html_report(
        summary_df=summary_df,
        stats=stats.to_dict(),
        config=cfg,
        output_path=str(report_path),
    )
    
    print(f"\n{'='*80}")
    print("FINALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Combined summary: {summary_path}")
    print(f"  QA Report: {report_path}")
    print(f"{'='*80}")


def _run_processing_mode(args: argparse.Namespace, cfg: Dict, features_root: str) -> None:
    """Process subjects and save summaries."""
    stats = AggregationStats()
    
    # Determine what to process
    subjects = [args.subject] if args.subject else cfg.get("subjects", [])
    tasks = [args.task] if args.task else cfg.get("tasks", [])
    sessions = [args.session] if args.session else cfg.get("sessions", []) or [None]
    
    all_results = []
    
    for subject in subjects:
        for session in sessions:
            for task in tasks:
                try:
                    session_str = f" ses-{session}" if session else ""
                    print(f"\n>>> Processing sub-{subject}{session_str} task-{task}")
                    df = process_subject_task(cfg, subject, task, session=session, stats=stats)
                    if not df.empty:
                        all_results.append(df)
                except Exception as exc:
                    session_str = f" ses-{session}" if session else ""
                    print(f"[ERROR] Failed to process sub-{subject}{session_str} task-{task}: {exc}")
                    import traceback
                    traceback.print_exc()
                    stats.add_error(context=f"sub-{subject}{session_str} task-{task}", message=str(exc))
    
    if not all_results:
        print("\n[WARN] No results to save")
        return
    
    summary_df = pd.concat(all_results, ignore_index=True)
    
    # Determine output mode (parallel vs sequential)
    job_id_parts = []
    if args.subject:
        job_id_parts.append(f"sub-{args.subject}")
    if args.session:
        job_id_parts.append(f"ses-{args.session}")
    if args.task:
        job_id_parts.append(f"task-{args.task}")
    
    if job_id_parts:
        # Parallel mode: save partial summary in subject folder
        # Build subject folder path: features_root/sub-XX/ses-X/eeg/junifer_aggregated/
        sub_folder = Path(features_root) / f"sub-{args.subject}"
        if args.session:
            sub_folder = sub_folder / f"ses-{args.session}" / "eeg" / "junifer_aggregated"
        else:
            sub_folder = sub_folder / "eeg" / "junifer_aggregated"
        sub_folder.mkdir(parents=True, exist_ok=True)
        
        partial_path = sub_folder / f"summary_{'_'.join(job_id_parts)}.csv"
        summary_df.to_csv(partial_path, index=False)
        print(f"\n[SUMMARY] Saved summary to: {partial_path}")
        print(f"Processed {len(summary_df)} probes for {' '.join(job_id_parts)}")
        
        # Generate per-subject QA report automatically
        if not args.no_report:
            report_path = sub_folder / f"qa_report_{'_'.join(job_id_parts)}.html"
            generate_html_report(
                summary_df=summary_df,
                stats=stats.to_dict(),
                config=cfg,
                output_path=str(report_path),
            )
            print(f"[REPORT] Saved QA report to: {report_path}")
    else:
        # Sequential mode: save final summary and generate report
        summary_path = Path(features_root) / "junifer_probe_aggregation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SUMMARY] Saved summary to: {summary_path}")
        print(f"Processed {len(summary_df)} probes")
        
        if not args.no_report:
            report_path = Path(features_root) / "junifer_aggregation_qa_report.html"
            generate_html_report(
                summary_df=summary_df,
                stats=stats.to_dict(),
                config=cfg,
                output_path=str(report_path),
            )
            
            print(f"\n{'='*80}")
            print("QA SUMMARY")
            print(f"{'='*80}")
            print(f"  Probes processed: {len(summary_df)}")
            print(f"  Probes discarded: {len(stats.discarded_probes)}")
            print(f"  Errors: {len(stats.errors)}")
            print(f"  Warnings: {len(stats.warnings)}")
            print(f"  Report: {report_path}")
            print(f"{'='*80}")


if __name__ == "__main__":
    main()
