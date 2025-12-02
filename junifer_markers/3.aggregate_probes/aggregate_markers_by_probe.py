#!/usr/bin/env python3
"""
Aggregate Junifer markers by probe for mind-wandering analysis.

This script reads per-epoch marker PKL files and aggregates them into per-probe
marker files, similar to the ERP probe aggregation pipeline. It handles two types
of markers differently:

1. Evoked markers (time-locked topographies): Aggregate only the 5 trials closest 
   to the probe (-5 to -1) with outlier detection (like ERPs)
2. State markers (spectral, connectivity, information theory): Aggregate all trials 
   before the probe to capture sustained mental state

The script follows the same structure as ERPs_new/make_probe_evokeds.py but adapted
for marker data stored in PKL format.

Usage:
    python aggregate_markers_by_probe.py --config config.yaml [--subject XX] [--task TaskName]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# Import local helpers (independent of ERP pipeline)
from helpers import (
    load_yaml_config,
    read_derivative_epochs,
    parse_events_tsv,
    enrich_events_with_parsed_fields,
    label_probe_onoff,
    detect_outlier_epochs,
    rename_markers,
)

# Define minimal versions of h5_to_pkl converter classes for unpickling
# These must match the class definitions in h5_to_pkl_converter.py
# We only need the structure, not the full functionality

class EpochMetadata:
    """Minimal EpochMetadata class for unpickling."""
    def __init__(self, epoch_idx, event, annotation=None, behavioral_data=None):
        self.epoch_idx = epoch_idx
        self.event = event
        self.event_id = event[2] if len(event) > 2 else None
        self.onset_sample = event[0] if len(event) > 0 else None
        self.annotation = annotation
        self.behavioral_data = behavioral_data or {}


class ChannelData:
    """Minimal ChannelData class for unpickling."""
    def __init__(self, channel_name, channel_idx, data, metadata):
        self.channel_name = channel_name
        self.channel_idx = channel_idx
        self.data = data
        self.metadata = metadata


class EpochData:
    """Minimal EpochData class for unpickling."""
    def __init__(self, epoch_idx, metadata, channel_names):
        self.epoch_idx = epoch_idx
        self.metadata = metadata
        self.channel_names = channel_names
        self._channel_data = {}
        self.annotations = getattr(metadata, 'annotation', None)


class MarkerData:
    """Minimal MarkerData class for unpickling."""
    def __init__(self, marker_name, marker_type, channel_names, n_epochs):
        self.marker_name = marker_name
        self.marker_type = marker_type
        self.channel_names = channel_names
        self.n_epochs = n_epochs
        self._epoch_data = {}
        self.metadata = {}


# Register classes in h5_to_pkl_converter namespace for pickle
# This allows pickle to find them when unpickling
import types
h5_to_pkl_converter = types.ModuleType('h5_to_pkl_converter')
h5_to_pkl_converter.MarkerData = MarkerData
h5_to_pkl_converter.EpochData = EpochData
h5_to_pkl_converter.ChannelData = ChannelData
h5_to_pkl_converter.EpochMetadata = EpochMetadata
sys.modules['h5_to_pkl_converter'] = h5_to_pkl_converter

# Also create mock mne module for unpickling annotations
# PKL files may contain mne.Annotations objects
if 'mne' not in sys.modules:
    class MockAnnotations:
        def __init__(self, onset=None, duration=None, description=None):
            self.onset = onset or []
            self.duration = duration or []
            self.description = description or []
    
    mne_module = types.ModuleType('mne')
    mne_module.Annotations = MockAnnotations
    sys.modules['mne'] = mne_module
    
    # Also create mne.annotations submodule
    mne_annotations = types.ModuleType('mne.annotations')
    mne_annotations.Annotations = MockAnnotations
    mne_annotations._AnnotationsExtrasList = list
    mne_annotations._AnnotationsExtrasDict = dict
    sys.modules['mne.annotations'] = mne_annotations


# REMOVED: find_marker_by_partial_name() - NO FUZZY MATCHING ALLOWED
# Only exact name matching via marker_name_mapping from config


def load_marker_pkl(pkl_path: str) -> Dict:
    """
    Load marker PKL file.
    
    Parameters
    ----------
    pkl_path : str
        Path to the PKL file
        
    Returns
    -------
    Dict
        Dictionary with markers, metadata, annotations, and info
    """
    print(f"Loading markers from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    markers = data.get("markers", {})
    n_markers = len(markers)
    
    # Try to get n_epochs from different possible locations
    n_epochs = data.get("info", {}).get("n_epochs", 0)
    if n_epochs == 0:
        # Try metadata.fif_info.n_epochs (h5_to_pkl converter format)
        metadata = data.get("metadata", {})
        fif_info = metadata.get("fif_info", {})
        n_epochs = fif_info.get("n_epochs", 0)
    
    # If still no epochs info, try to get from a marker object
    if n_epochs == 0 and markers:
        first_marker = next(iter(markers.values()))
        if hasattr(first_marker, 'n_epochs'):
            n_epochs = first_marker.n_epochs
    
    print(f"  Loaded {n_markers} markers with {n_epochs} epochs")
    
    # Debug: show marker names if any
    if n_markers > 0:
        marker_names = list(markers.keys())
        print(f"  Marker names: {', '.join(marker_names[:5])}")
        if len(marker_names) > 5:
            print(f"                ... and {len(marker_names) - 5} more")
        
        # Show marker types if they're MarkerData objects
        first_marker = next(iter(markers.values()))
        if hasattr(first_marker, 'marker_type'):
            print(f"  Marker type: {first_marker.marker_type} (MarkerData object)")
            
            # Quick data quality check for first marker
            if hasattr(first_marker, '_epoch_data') and first_marker._epoch_data:
                first_epoch_idx = list(first_marker._epoch_data.keys())[0]
                first_epoch = first_marker._epoch_data[first_epoch_idx]
                if hasattr(first_epoch, '_channel_data') and first_epoch._channel_data:
                    first_channel = list(first_epoch._channel_data.keys())[0]
                    ch_data = first_epoch._channel_data[first_channel]
                    if hasattr(ch_data, 'data') and isinstance(ch_data.data, np.ndarray):
                        n_nans = np.isnan(ch_data.data).sum()
                        n_total = len(ch_data.data)
                        print(f"  Sample data quality: {n_nans}/{n_total} NaNs in first epoch/channel")
                    else:
                        print(f"  Sample data type: {type(ch_data.data) if hasattr(ch_data, 'data') else 'no data attr'}")
                else:
                    print(f"  No channel data found in first epoch")
            else:
                print(f"  No epoch data found in first marker")
    else:
        print(f"  WARNING: No markers found in PKL file!")
        # Debug: show what keys are in the data
        print(f"  Available top-level keys: {list(data.keys())}")
    
    return data


def extract_epoch_value_from_marker(
    marker_data, 
    channel: str, 
    epoch_idx: int,
    marker_name: str,
    band_idx: Optional[int] = None,
    subject_id: str = "unknown",
    task: str = "unknown",
) -> Optional[float]:
    """
    Extract a single epoch value from marker data with deterministic band processing.
    
    Parameters
    ----------
    marker_data : MarkerData or Dict
        Marker data object (MarkerData from h5_to_pkl converter) or dictionary
    channel : str
        Channel name
    epoch_idx : int
        Epoch index
    marker_name : str
        Name of the marker (for debugging)
    band_idx : Optional[int]
        For spectral markers with multiple bands, specific band index to extract.
        REQUIRED for spectral markers - no fallback averaging allowed.
        Band order is: [delta=0, theta=1, alpha=2, beta=3, gamma=4]
    subject_id : str
        Subject identifier for error messages
    task : str
        Task identifier for error messages
        
    Returns
    -------
    Optional[float]
        Extracted value or None if not found
        
    Raises
    ------
    ValueError
        If band_idx is None for spectral markers or data structure is unexpected
    """
    try:
        # Check if this is a MarkerData object from h5_to_pkl converter
        if hasattr(marker_data, '_epoch_data'):
            # This is a MarkerData object with hierarchical structure:
            # marker_data._epoch_data[epoch_idx]._channel_data[channel].data
            
            if epoch_idx not in marker_data._epoch_data:
                return None
            
            epoch_data = marker_data._epoch_data[epoch_idx]
            
            if not hasattr(epoch_data, '_channel_data'):
                return None
            
            channel_data_dict = epoch_data._channel_data
            
            # Check for exact channel match
            if channel in channel_data_dict:
                ch_data = channel_data_dict[channel]
                if hasattr(ch_data, 'data'):
                    data_val = ch_data.data
                    if isinstance(data_val, np.ndarray):
                        # Array of band values for spectral markers
                        if band_idx is not None:
                            if band_idx >= len(data_val):
                                raise ValueError(
                                    f"[{subject_id} {task}] Band index {band_idx} out of range for {marker_name} "
                                    f"channel {channel}: data has {len(data_val)} bands"
                                )
                            return float(data_val[band_idx])
                        else:
                            # For non-spectral markers without band_idx
                            if len(data_val) == 1:
                                # Single value - return it
                                return float(data_val[0])
                            else:
                                # Multi-element array from new h5_to_pkl format
                                # For non-spectral markers, take the mean across all values
                                # This handles cases where the converter stored multiple features
                                return float(np.mean(data_val))
                    else:
                        # Scalar data
                        if band_idx is not None:
                            raise ValueError(
                                f"[{subject_id} {task}] Band index {band_idx} specified for non-array data "
                                f"in {marker_name} channel {channel}"
                            )
                        return float(data_val)
                else:
                    raise ValueError(
                        f"[{subject_id} {task}] No data attribute in {marker_name} "
                        f"channel {channel} epoch {epoch_idx}"
                    )
            else:
                # For connectivity markers: look for channel pairs involving this channel
                if hasattr(marker_data, 'marker_type') and marker_data.marker_type in ['connectivity', 'wsmi', 'smi']:
                    values = []
                    for ch_key in channel_data_dict.keys():
                        if '-' in ch_key and channel in ch_key.split('-'):
                            ch_data = channel_data_dict[ch_key]
                            if hasattr(ch_data, 'data'):
                                data_val = ch_data.data
                                if isinstance(data_val, np.ndarray):
                                    if band_idx is not None and band_idx < len(data_val):
                                        values.append(float(data_val[band_idx]))
                                    elif band_idx is None:
                                        values.append(float(np.mean(data_val)))
                                    else:
                                        raise ValueError(
                                            f"[{subject_id} {task}] Band index {band_idx} out of range for connectivity "
                                            f"marker {marker_name} pair {ch_key}: data has {len(data_val)} elements"
                                        )
                                else:
                                    if band_idx is not None:
                                        raise ValueError(
                                            f"[{subject_id} {task}] Band index {band_idx} specified for non-array "
                                            f"connectivity data in {marker_name} pair {ch_key}"
                                        )
                                    values.append(float(data_val))
                    
                    if values:
                        return float(np.mean(values))
                    else:
                        return None
                else:
                    # Channel not found - this is expected for some markers
                    return None
        
        # Legacy format: dictionary with access_pattern and data keys
        elif isinstance(marker_data, dict):
            access_pattern = marker_data.get("access_pattern", "")
            data = marker_data.get("data", {})
            
            if access_pattern == "channel_epoch":
                if channel not in data:
                    return None
                
                channel_data = data[channel]
                epoch_key = f"epoch_{epoch_idx:04d}"
                
                if epoch_key in channel_data:
                    val = channel_data[epoch_key]
                    if band_idx is not None:
                        if isinstance(val, (list, np.ndarray)) and len(val) > band_idx:
                            return float(val[band_idx])
                        else:
                            raise ValueError(
                                f"[{subject_id} {task}] Band index {band_idx} out of range for {marker_name} "
                                f"channel {channel}: data has {len(val) if hasattr(val, '__len__') else 1} elements"
                            )
                    else:
                        if isinstance(val, (list, np.ndarray)):
                            if len(val) == 1:
                                return float(val[0])
                            else:
                                # Multi-element array - take mean
                                return float(np.mean(val))
                        return float(val)
                return None
                
            elif access_pattern == "epoch_channel":
                epoch_key = f"epoch_{epoch_idx:04d}"
                if epoch_key in data and channel in data[epoch_key]:
                    val = data[epoch_key][channel]
                    if band_idx is not None:
                        if isinstance(val, (list, np.ndarray)) and len(val) > band_idx:
                            return float(val[band_idx])
                        else:
                            raise ValueError(
                                f"[{subject_id} {task}] Band index {band_idx} out of range for {marker_name} "
                                f"channel {channel}: data has {len(val) if hasattr(val, '__len__') else 1} elements"
                            )
                    else:
                        if isinstance(val, (list, np.ndarray)):
                            if len(val) == 1:
                                return float(val[0])
                            else:
                                # Multi-element array - take mean
                                return float(np.mean(val))
                        return float(val)
                return None
                
            elif access_pattern == "channel_pair_epoch":
                epoch_key = f"epoch_{epoch_idx:04d}"
                values = []
                for pair_key, pair_data in data.items():
                    if channel in pair_key.split("-") and epoch_key in pair_data:
                        val = pair_data[epoch_key]
                        if band_idx is not None:
                            if isinstance(val, (list, np.ndarray)) and len(val) > band_idx:
                                values.append(val[band_idx])
                            else:
                                raise ValueError(
                                    f"[{subject_id} {task}] Band index {band_idx} out of range for {marker_name} "
                                    f"pair {pair_key}: data has {len(val) if hasattr(val, '__len__') else 1} elements"
                                )
                        else:
                            if isinstance(val, (list, np.ndarray)):
                                if len(val) == 1:
                                    values.append(val[0])
                                else:
                                    # Multi-element array - take mean
                                    values.append(np.mean(val))
                            else:
                                values.append(val)
                
                if values:
                    return float(np.mean(values))
                return None
            else:
                raise ValueError(
                    f"[{subject_id} {task}] Unknown access pattern: {access_pattern} for {marker_name}"
                )
        else:
            raise ValueError(
                f"[{subject_id} {task}] Unknown marker_data type: {type(marker_data)} for {marker_name}"
            )
            
    except Exception as e:
        if isinstance(e, ValueError):
            raise  # Re-raise ValueError with context
        else:
            raise ValueError(
                f"[{subject_id} {task}] Error extracting epoch {epoch_idx} for {marker_name}, "
                f"channel {channel}: {e}"
            )


def debug_pkl_comprehensive(pkl_path: str):
    """Comprehensive debug of entire PKL file - markers, channels, epochs, stats, events."""
    print(f"\n{'='*80}")
    print(f"[COMPREHENSIVE PKL DEBUG] {pkl_path}")
    print(f"{'='*80}")
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Top-level structure
        print(f"\n📁 TOP-LEVEL STRUCTURE:")
        print(f"  Keys: {list(data.keys())}")
        print(f"  File size: {os.path.getsize(pkl_path):,} bytes")
        
        # Info section
        if 'info' in data:
            info = data['info']
            print(f"\n📊 INFO SECTION:")
            print(f"  Keys: {list(info.keys())}")
            print(f"  n_epochs: {info.get('n_epochs', 'N/A')}")
            print(f"  n_channels: {info.get('n_channels', 'N/A')}")
            if 'channel_names' in info:
                ch_names = info['channel_names']
                print(f"  channel_names: {len(ch_names)} channels")
                print(f"    First 10: {ch_names[:10]}")
                print(f"    Last 10: {ch_names[-10:]}")
        
        # Metadata section
        if 'metadata' in data:
            metadata = data['metadata']
            print(f"\n📋 METADATA SECTION:")
            print(f"  Keys: {list(metadata.keys())}")
            
            if 'fif_info' in metadata:
                fif_info = metadata['fif_info']
                print(f"  FIF Info:")
                print(f"    Keys: {list(fif_info.keys())}")
                print(f"    n_epochs: {fif_info.get('n_epochs', 'N/A')}")
                print(f"    n_channels: {fif_info.get('n_channels', 'N/A')}")
                if 'channel_names' in fif_info:
                    fif_ch = fif_info['channel_names']
                    print(f"    channel_names: {len(fif_ch)} channels")
        
        # Markers section - COMPREHENSIVE
        if 'markers' in data:
            markers = data['markers']
            print(f"\n🔬 MARKERS SECTION (COMPREHENSIVE):")
            print(f"  Total markers: {len(markers)}")
            
            for i, (marker_name, marker_data) in enumerate(markers.items()):
                print(f"\n  📍 Marker {i+1}: {marker_name}")
                
                # Basic marker info
                if hasattr(marker_data, 'marker_type'):
                    print(f"    Type: {marker_data.marker_type}")
                if hasattr(marker_data, 'n_epochs'):
                    print(f"    n_epochs: {marker_data.n_epochs}")
                if hasattr(marker_data, 'channel_names'):
                    marker_ch = marker_data.channel_names
                    print(f"    channel_names: {len(marker_ch)} channels")
                    print(f"      First 5: {marker_ch[:5]}")
                    print(f"      Last 5: {marker_ch[-5:]}")
                
                # Epoch data analysis
                if hasattr(marker_data, '_epoch_data') and marker_data._epoch_data:
                    all_epochs = list(marker_data._epoch_data.keys())
                    all_epochs.sort()
                    
                    print(f"    Epochs: {len(all_epochs)} total")
                    print(f"      Range: {min(all_epochs)} to {max(all_epochs)}")
                    print(f"      First 10: {all_epochs[:10]}")
                    print(f"      Last 10: {all_epochs[-10:]}")
                    
                    # Analyze first epoch in detail
                    if all_epochs:
                        first_epoch_idx = all_epochs[0]
                        first_epoch = marker_data._epoch_data[first_epoch_idx]
                        
                        print(f"    📊 First Epoch Analysis (epoch {first_epoch_idx}):")
                        
                        if hasattr(first_epoch, '_channel_data') and first_epoch._channel_data:
                            epoch_channels = list(first_epoch._channel_data.keys())
                            print(f"      Channels: {len(epoch_channels)}")
                            print(f"      First 5 channels: {epoch_channels[:5]}")
                            
                            # Analyze first few channels in detail
                            for j, ch_name in enumerate(epoch_channels[:3]):
                                ch_data = first_epoch._channel_data[ch_name]
                                print(f"      🔍 Channel {ch_name}:")
                                
                                if hasattr(ch_data, 'data'):
                                    if isinstance(ch_data.data, np.ndarray):
                                        data_array = ch_data.data
                                        print(f"        Data shape: {data_array.shape}")
                                        print(f"        Data type: {data_array.dtype}")
                                        print(f"        NaN count: {np.isnan(data_array).sum()}")
                                        print(f"        Inf count: {np.isinf(data_array).sum()}")
                                        if data_array.size > 0:
                                            valid_data = data_array[~np.isnan(data_array)]
                                            if len(valid_data) > 0:
                                                print(f"        Valid values: {len(valid_data)}")
                                                print(f"        Mean: {np.mean(valid_data):.6f}")
                                                print(f"        Min: {np.min(valid_data):.6f}")
                                                print(f"        Max: {np.max(valid_data):.6f}")
                                                print(f"        Std: {np.std(valid_data):.6f}")
                                            else:
                                                print(f"        All values are NaN!")
                                    else:
                                        print(f"        Data type: {type(ch_data.data)}")
                                        print(f"        Data value: {ch_data.data}")
                                else:
                                    print(f"        No data attribute")
                                    
                                if j >= 2:  # Limit to first 3 channels
                                    break
                        else:
                            print(f"      No channel data in first epoch")
                    
                    # Sample a few more epochs for consistency check
                    if len(all_epochs) > 1:
                        print(f"    🔍 Epoch Consistency Check:")
                        sample_epochs = all_epochs[::max(1, len(all_epochs)//5)][:3]  # Sample 3 epochs
                        for epoch_idx in sample_epochs:
                            epoch = marker_data._epoch_data[epoch_idx]
                            if hasattr(epoch, '_channel_data'):
                                n_ch = len(epoch._channel_data)
                                print(f"      Epoch {epoch_idx}: {n_ch} channels")
                            else:
                                print(f"      Epoch {epoch_idx}: No channel data")
                else:
                    print(f"    No epoch data available")
        
        # Events section (if available)
        if 'events' in data:
            events = data['events']
            print(f"\n🎯 EVENTS SECTION:")
            if isinstance(events, np.ndarray):
                print(f"  Events array shape: {events.shape}")
                print(f"  Event IDs: {np.unique(events[:, 2])}")
                print(f"  Sample range: {events[0, 0]} to {events[-1, 0]}")
            else:
                print(f"  Events type: {type(events)}")
                print(f"  Events keys: {list(events.keys()) if isinstance(events, dict) else 'N/A'}")
        
        print(f"\n{'='*80}")
        
    except Exception as e:
        print(f"❌ ERROR loading PKL: {e}")
        import traceback
        traceback.print_exc()


def debug_probe_comprehensive(
    probe_num: int,
    subject: str,
    task: str,
    evoked_data,
    state_data,
    epochs_data,
    events_data,
    channels: List[str],
):
    """Comprehensive debug for a specific probe."""
    print(f"\n{'='*60}")
    print(f"[COMPREHENSIVE PROBE DEBUG] Probe {probe_num} (sub-{subject}, {task})")
    print(f"{'='*60}")
    
    # Probe events analysis
    if events_data is not None:
        print(f"\n🎯 PROBE EVENTS:")
        print(f"  Events dataframe columns: {list(events_data.columns)}")
        
        # Check if probe_number column exists and get probe events
        if 'probe_number' in events_data.columns:
            probe_events = events_data[events_data['probe_number'] == probe_num]
            print(f"  Total events for probe {probe_num}: {len(probe_events)}")
            
            if len(probe_events) > 0:
                print(f"  Sample events for probe {probe_num}:")
                for i, (_, event) in enumerate(probe_events.head(3).iterrows()):
                    print(f"    Event {i+1}: onset={event.get('onset', 'N/A')}, distance={event.get('distance_to_probe', 'N/A')}")
        else:
            print(f"  ❌ No 'probe_number' column found in events data")
    
    # Epochs data analysis
    if epochs_data is not None:
        print(f"\n📊 EPOCHS DATA:")
        if hasattr(epochs_data, 'events'):
            events = epochs_data.events
            print(f"  Total epochs: {len(events)}")
            print(f"  Event IDs: {np.unique(events[:, 2])}")
            print(f"  Sample range: {events[0, 0]} to {events[-1, 0]}")
        
        if hasattr(epochs_data, 'info'):
            info = epochs_data.info
            print(f"  Info:")
            print(f"    n_epochs: {info.get('n_epochs', 'N/A')}")
            print(f"    sfreq: {info.get('sfreq', 'N/A')}")
            print(f"    ch_names: {len(info.get('ch_names', []))}")
    
    # Channels analysis
    print(f"\n🔌 CHANNELS:")
    print(f"  Total channels: {len(channels)}")
    print(f"  First 10: {channels[:10]}")
    print(f"  Last 10: {channels[-10:]}")
    
    # Evoked data analysis
    if evoked_data:
        print(f"\n⚡ EVOKED DATA:")
        evoked_markers = evoked_data.get("markers", {})
        print(f"  Total evoked markers: {len(evoked_markers)}")
        
        if evoked_markers:
            first_marker = next(iter(evoked_markers.values()))
            if hasattr(first_marker, '_epoch_data'):
                evoked_epochs = list(first_marker._epoch_data.keys())
                evoked_epochs.sort()
                print(f"  Evoked epochs range: {min(evoked_epochs)} to {max(evoked_epochs)} ({len(evoked_epochs)} total)")
                print(f"    First 10: {evoked_epochs[:10]}")
                print(f"    Last 10: {evoked_epochs[-10:]}")
    
    # State data analysis
    if state_data:
        print(f"\n🧠 STATE DATA:")
        state_markers = state_data.get("markers", {})
        print(f"  Total state markers: {len(state_markers)}")
        
        if state_markers:
            first_marker = next(iter(state_markers.values()))
            if hasattr(first_marker, '_epoch_data'):
                state_epochs = list(first_marker._epoch_data.keys())
                state_epochs.sort()
                print(f"  State epochs range: {min(state_epochs)} to {max(state_epochs)} ({len(state_epochs)} total)")
                print(f"    First 10: {state_epochs[:10]}")
                print(f"    Last 10: {state_epochs[-10:]}")
    
    print(f"\n{'='*60}")


def debug_marker_data_before_aggregation(
    marker_data,
    marker_name: str,
    epoch_indices: List[int],
    channels: List[str],
    probe_num: int,
):
    """Debug function to check marker data state before aggregation."""
    print(f"    [DEBUG] Pre-aggregation check for {marker_name} (Probe {probe_num}):")
    
    if hasattr(marker_data, 'marker_type'):
        print(f"      Marker type: {marker_data.marker_type}")
    
    if hasattr(marker_data, '_epoch_data'):
        print(f"      Available epochs in marker: {len(marker_data._epoch_data)}")
        print(f"      Requested epoch indices: {epoch_indices}")
        
        # Check how many requested epochs are available
        available_epochs = [idx for idx in epoch_indices if idx in marker_data._epoch_data]
        missing_epochs = [idx for idx in epoch_indices if idx not in marker_data._epoch_data]
        
        print(f"      Available requested epochs: {len(available_epochs)}")
        if missing_epochs:
            print(f"      Missing epochs: {missing_epochs[:5]}{'...' if len(missing_epochs) > 5 else ''}")
        
        # Check data quality for first available epoch
        if available_epochs and channels:
            first_epoch_idx = available_epochs[0]
            first_epoch = marker_data._epoch_data[first_epoch_idx]
            
            if hasattr(first_epoch, '_channel_data'):
                print(f"      Channels in epoch {first_epoch_idx}: {len(first_epoch._channel_data)}")
                
                # Check first few channels for NaN values
                nan_counts = {}
                total_values = 0
                for i, channel in enumerate(channels[:3]):  # Check first 3 channels
                    if channel in first_epoch._channel_data:
                        ch_data = first_epoch._channel_data[channel]
                        if hasattr(ch_data, 'data'):
                            if isinstance(ch_data.data, np.ndarray):
                                n_nans = np.isnan(ch_data.data).sum()
                                n_total = len(ch_data.data)
                                nan_counts[channel] = (n_nans, n_total)
                                total_values += n_total
                                print(f"        {channel}: {n_nans}/{n_total} NaNs")
                            else:
                                print(f"        {channel}: data is not numpy array ({type(ch_data.data)})")
                        else:
                            print(f"        {channel}: no data attribute")
                    else:
                        print(f"        {channel}: channel not found in epoch")
                
                # Summary
                total_nans = sum(counts[0] for counts in nan_counts.values())
                print(f"      Sample summary: {total_nans}/{total_values} NaNs in first epoch")
            else:
                print(f"      No channel data in epoch {first_epoch_idx}")
        else:
            print(f"      No available epochs or channels to check")
    else:
        print(f"      No epoch data available")


def aggregate_marker_epochs(
    marker_data: Dict,
    marker_name: str,
    epoch_indices: List[int],
    channels: List[str],
    use_trimmean: bool = False,
    trimmean_percent: float = 20.0,
    subject_id: str = "unknown",
    task: str = "unknown",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate marker values across selected epochs with deterministic band processing.
    
    This function is 100% deterministic - no fallbacks, no silent averaging.
    Spectral markers MUST have exactly 5 bands and be processed explicitly.
    
    Parameters
    ----------
    marker_data : Dict
        Marker data dictionary or MarkerData object
    marker_name : str
        Name of the marker
    epoch_indices : List[int]
        List of epoch indices to aggregate
    channels : List[str]
        List of channel names
    use_trimmean : bool
        Whether to use trimmean instead of regular mean
    trimmean_percent : float
        Percentage to trim from each side (e.g., 10.0 = trim 10% from each side, 20% total)
    subject_id : str
        Subject identifier for error messages
    task : str
        Task identifier for error messages
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        For spectral markers: {"delta": {ch: val, ...}, "theta": {ch: val, ...}, ...}
        For other markers: {"overall": {ch: val, ...}}
        
    Raises
    ------
    ValueError
        If marker classification is ambiguous or data structure is unexpected
    """
    # DETERMINISTIC MARKER TYPE CLASSIFICATION
    # Only these markers are considered spectral - no automatic detection
    spectral_whitelist = ["power", "power_normalized"]
    sleep_marker_types = ["sleep_spindles", "sleep_slow_waves"]
    
    is_spectral = False
    is_sleep = False
    marker_type = "unknown"
    
    # Check marker type from object attribute
    if hasattr(marker_data, 'marker_type'):
        marker_type = marker_data.marker_type
        if marker_type == "spectral":
            is_spectral = True
        elif marker_type in sleep_marker_types:
            is_sleep = True
    else:
        # Check marker name against whitelist
        if any(spectral_name in marker_name.lower() for spectral_name in spectral_whitelist):
            is_spectral = True
            marker_type = "spectral"
        else:
            marker_type = "non-spectral"
    
    if is_sleep:
        print(f"→ Aggregating marker: {marker_name} (type={marker_type}, multi-feature sleep)")
        if not hasattr(marker_data, "_epoch_data") or not marker_data._epoch_data:
            raise ValueError(
                f"[{subject_id} {task}] Sleep marker {marker_name} has no epoch data"
            )
        first_epoch = None
        first_epoch_idx = None
        for epoch_idx in epoch_indices:
            if epoch_idx in marker_data._epoch_data:
                first_epoch = marker_data._epoch_data[epoch_idx]
                first_epoch_idx = epoch_idx
                break
        if first_epoch is None:
            raise ValueError(
                f"[{subject_id} {task}] No epochs found for sleep marker {marker_name} "
                f"in requested indices {epoch_indices}"
            )
        if not hasattr(first_epoch, "_channel_data") or not first_epoch._channel_data:
            raise ValueError(
                f"[{subject_id} {task}] Sleep marker {marker_name} has no channel data in epoch {first_epoch_idx}"
            )
        first_channel = channels[0] if channels else None
        if first_channel and first_channel in first_epoch._channel_data:
            ch_data = first_epoch._channel_data[first_channel]
            if hasattr(ch_data, "data") and isinstance(ch_data.data, np.ndarray):
                data_shape = ch_data.data.shape
                if len(data_shape) != 1:
                    raise ValueError(
                        f"[{subject_id} {task}] Sleep marker {marker_name} has unexpected data shape "
                        f"{data_shape} for channel {first_channel}. Expected 1D array of features."
                    )
                n_features = data_shape[0]
            else:
                raise ValueError(
                    f"[{subject_id} {task}] Sleep marker {marker_name} has non-array data "
                    f"for channel {first_channel}. Expected 1D numpy array of features."
                )
        else:
            raise ValueError(
                f"[{subject_id} {task}] Sleep marker {marker_name} missing channel {first_channel} "
                f"in epoch {first_epoch_idx}"
            )
        raw_feature_names = []
        if hasattr(marker_data, "metadata") and isinstance(marker_data.metadata, dict):
            raw_feature_names = marker_data.metadata.get("feature_names", []) or []
        if raw_feature_names and len(raw_feature_names) != n_features:
            raw_feature_names = raw_feature_names[:n_features]
        if not raw_feature_names:
            raw_feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_names = [str(name).strip().lower().replace(" ", "_") for name in raw_feature_names]
        print(f"  Detected {n_features} sleep features: {', '.join(feature_names)}")
        aggregated = {}
        for feat_idx, feat_name in enumerate(feature_names):
            feat_aggregated = {}
            for channel in channels:
                values = []
                for epoch_idx in epoch_indices:
                    val = extract_epoch_value_from_marker(
                        marker_data,
                        channel,
                        epoch_idx,
                        marker_name,
                        band_idx=feat_idx,
                        subject_id=subject_id,
                        task=task,
                    )
                    if val is not None:
                        values.append(val)
                if values:
                    if use_trimmean and len(values) > 2:
                        feat_aggregated[channel] = float(
                            scipy.stats.trim_mean(values, trimmean_percent / 200)
                        )
                    else:
                        feat_aggregated[channel] = float(np.mean(values))
                else:
                    feat_aggregated[channel] = np.nan
                    print(
                        f"        [WARN] No values found for {marker_name} feature {feat_name} channel {channel}"
                    )
            aggregated[feat_name] = feat_aggregated
        print(f"✓ Aggregated {marker_name} sleep features successfully")
    
    # VALIDATE SPECTRAL MARKER STRUCTURE
    elif is_spectral:
        print(f"→ Aggregating marker: {marker_name} (type={marker_type}, 5 bands)")
        
        # Check data structure for spectral markers
        if not hasattr(marker_data, '_epoch_data') or not marker_data._epoch_data:
            raise ValueError(
                f"[{subject_id} {task}] Spectral marker {marker_name} has no epoch data"
            )
        
        # Find first available epoch to validate structure
        first_epoch = None
        first_epoch_idx = None
        for epoch_idx in epoch_indices:
            if epoch_idx in marker_data._epoch_data:
                first_epoch = marker_data._epoch_data[epoch_idx]
                first_epoch_idx = epoch_idx
                break
        
        if first_epoch is None:
            raise ValueError(
                f"[{subject_id} {task}] No epochs found for spectral marker {marker_name} "
                f"in requested indices {epoch_indices}"
            )
        
        # Validate channel data structure
        if not hasattr(first_epoch, '_channel_data') or not first_epoch._channel_data:
            raise ValueError(
                f"[{subject_id} {task}] Spectral marker {marker_name} has no channel data in epoch {first_epoch_idx}"
            )
        
        # Check first channel for band structure
        first_channel = channels[0] if channels else None
        if first_channel and first_channel in first_epoch._channel_data:
            ch_data = first_epoch._channel_data[first_channel]
            if hasattr(ch_data, 'data') and isinstance(ch_data.data, np.ndarray):
                data_shape = ch_data.data.shape
                if len(data_shape) != 1:
                    raise ValueError(
                        f"[{subject_id} {task}] Spectral marker {marker_name} has unexpected data shape "
                        f"{data_shape} for channel {first_channel}. Expected 1D array with 5 or 6 bands."
                    )
                # Detect number of bands: 5 (standard) or 6 (with iota)
                n_bands = data_shape[0]
                if n_bands not in [5, 6]:
                    raise ValueError(
                        f"[{subject_id} {task}] Spectral marker {marker_name} has {n_bands} bands "
                        f"for channel {first_channel}. Expected 5 bands (delta, theta, alpha, beta, gamma) "
                        f"or 6 bands (delta, theta, alpha, beta, gamma, iota)."
                    )
            else:
                raise ValueError(
                    f"[{subject_id} {task}] Spectral marker {marker_name} has non-array data "
                    f"for channel {first_channel}. Expected numpy array with 5 or 6 bands."
                )
        else:
            raise ValueError(
                f"[{subject_id} {task}] Spectral marker {marker_name} missing channel {first_channel} "
                f"in epoch {first_epoch_idx}"
            )
        
        # Process each band explicitly
        # Determine band names based on detected number of bands
        if n_bands == 5:
            band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        else:  # n_bands == 6
            band_names = ["delta", "theta", "alpha", "beta", "gamma", "iota"]
        
        print(f"  Detected {n_bands} spectral bands: {', '.join(band_names)}")
        aggregated = {}
        
        for band_idx, band_name in enumerate(band_names):
            band_aggregated = {}
            for channel in channels:
                values = []
                for epoch_idx in epoch_indices:
                    val = extract_epoch_value_from_marker(
                        marker_data, channel, epoch_idx, marker_name, 
                        band_idx=band_idx, subject_id=subject_id, task=task
                    )
                    if val is not None:
                        values.append(val)
                
                if values:
                    # Use trimmean or regular mean
                    if use_trimmean and len(values) > 2:
                        # trimmean_percent is per side, so divide by 2 for scipy.stats.trim_mean
                        band_aggregated[channel] = float(scipy.stats.trim_mean(values, trimmean_percent/200))
                    else:
                        band_aggregated[channel] = float(np.mean(values))
                else:
                    band_aggregated[channel] = np.nan
                    print(f"        [WARN] No values found for {marker_name} band {band_name} channel {channel}")
            
            aggregated[band_name] = band_aggregated
        
        last_band = band_names[-1]
        print(f"✓ Aggregated {marker_name}_delta … {marker_name}_{last_band} successfully")
        
    else:
        # NON-SPECTRAL MARKER: single aggregation
        print(f"→ Aggregating marker: {marker_name} (type={marker_type}, scalar)")
        
        single_aggregated = {}
        for channel in channels:
            values = []
            for epoch_idx in epoch_indices:
                val = extract_epoch_value_from_marker(
                    marker_data, channel, epoch_idx, marker_name,
                    band_idx=None, subject_id=subject_id, task=task
                )
                if val is not None:
                    values.append(val)
            
            if values:
                # Use trimmean or regular mean
                if use_trimmean and len(values) > 2:
                    # trimmean_percent is per side, so divide by 2 for scipy.stats.trim_mean
                    single_aggregated[channel] = float(scipy.stats.trim_mean(values, trimmean_percent/200))
                else:
                    single_aggregated[channel] = float(np.mean(values))
            else:
                single_aggregated[channel] = np.nan
                print(f"        [WARN] No values found for {marker_name} channel {channel}")
        
        aggregated = {"overall": single_aggregated}
        print(f"✓ Aggregated {marker_name} successfully")
    
    # FINAL VALIDATION
    if is_spectral:
        # Check that all expected bands exist
        expected_bands = ["delta", "theta", "alpha", "beta", "gamma"]
        missing_bands = [band for band in expected_bands if band not in aggregated]
        if missing_bands:
            raise ValueError(
                f"[{subject_id} {task}] Spectral marker {marker_name} missing bands: {missing_bands}"
            )
        
        # Check for NaN values or identical values across bands
        for band_name in expected_bands:
            band_data = aggregated[band_name]
            for channel, value in band_data.items():
                if np.isnan(value):
                    print(f"        [WARN] NaN value for {marker_name}_{band_name} channel {channel}")
    
    return aggregated


def apply_nan_policy_to_sleep_markers(
    aggregated_sleep: Dict[str, Dict[str, float]],
    nan_policy: str,
) -> Dict[str, Dict[str, float]]:
    """Apply NaN handling policy to aggregated sleep markers.

    Parameters
    ----------
    aggregated_sleep : Dict[str, Dict[str, float]]
        Mapping from sleep marker names (possibly feature-expanded) to
        channel -> value dictionaries.
    nan_policy : str
        NaN handling policy. Supported values are:
        - "null": keep NaN values as-is
        - "zero": replace NaN by 0.0
        - "mean": replace NaN by the mean of non-NaN values for the same
          marker across channels (per probe)
        - "median": replace NaN by the median of non-NaN values for the same
          marker across channels (per probe)

    Returns
    -------
    Dict[str, Dict[str, float]]
        Processed mapping with NaNs handled according to the requested policy.
    """
    if not aggregated_sleep:
        return aggregated_sleep

    policy = (nan_policy or "null").lower()
    if policy not in {"null", "zero", "mean", "median"}:
        policy = "null"

    if policy == "null":
        return aggregated_sleep

    processed: Dict[str, Dict[str, float]] = {}

    for marker_name, channel_values in aggregated_sleep.items():
        if not channel_values:
            processed[marker_name] = channel_values
            continue

        channels = list(channel_values.keys())
        values = np.array([channel_values[ch] for ch in channels], dtype=float)
        nan_mask = np.isnan(values)

        if not nan_mask.any():
            processed[marker_name] = channel_values
            continue

        if policy == "zero":
            fill_value = 0.0
        else:
            valid_values = values[~nan_mask]
            if valid_values.size == 0:
                processed[marker_name] = channel_values
                continue
            if policy == "mean":
                fill_value = float(np.mean(valid_values))
            else:
                fill_value = float(np.median(valid_values))

        new_channel_values: Dict[str, float] = {}
        for ch, val, is_nan in zip(channels, values, nan_mask):
            if is_nan:
                new_channel_values[ch] = fill_value
            else:
                new_channel_values[ch] = float(val)

        processed[marker_name] = new_channel_values

    return processed


def select_trials_for_probe(
    events_df: pd.DataFrame,
    probe_number: int,
    only_go_correct: bool,
    distance_min: int,
    distance_max: int,
) -> pd.DataFrame:
    """
    Filter events to keep only the desired trials for a specific probe.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with all trials
    probe_number : int
        Probe number to filter for
    only_go_correct : bool
        If True, keep only go/correct trials
    distance_min : int
        Minimum distance to probe (inclusive)
    distance_max : int
        Maximum distance to probe (inclusive)
        
    Returns
    -------
    pd.DataFrame
        Filtered events dataframe
    """
    df = events_df.copy()
    
    # Filter for this probe
    df = df[df["probe_number"] == probe_number]
    
    if only_go_correct:
        df = df[(df["trial_type"] == "go") & (df["correctness"] == "correct")]
    
    # For sleep/preprobe windows and other events without distance coding,
    # distance_to_probe may be entirely missing or NaN. In that case, keep
    # all epochs for this probe instead of dropping everything.
    if "distance_to_probe" not in df.columns or df["distance_to_probe"].isna().all():
        return df

    # Keep distance in [distance_min, distance_max]
    df = df[df["distance_to_probe"].notna()]
    df = df[(df["distance_to_probe"] >= distance_min) & (df["distance_to_probe"] <= distance_max)]
    
    return df


def save_aggregated_markers(
    aggregated_markers: Dict[str, Dict[str, float]],
    features_root: str,
    subject: str,
    task: str,
    probe_number: int,
    label: str,
    extra_meta: Dict,
    overwrite: bool = True,
    marker_type: str = "mixed",  # "evoked", "state", or "mixed"
    save_format: str = "csv",  # "csv" or "pkl"
    probe_metadata: Dict = None,  # Additional probe metadata
) -> str:
    """
    Save aggregated markers to CSV or PKL file.
    
    Parameters
    ----------
    aggregated_markers : Dict[str, Dict[str, float]]
        Dictionary mapping marker names to channel-value dictionaries
    features_root : str
        Root directory for features
    subject : str
        Subject identifier
    task : str
        Task identifier
    probe_number : int
        Probe number
    label : str
        Probe label (onTask/offTask)
    extra_meta : Dict
        Additional metadata to save
    overwrite : bool
        Whether to overwrite existing files
    marker_type : str
        Type of markers ("evoked", "state", or "mixed")
    save_format : str
        Output format ("csv" or "pkl")
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create output directory: features/sub-XX/eeg/junifer/
    out_dir = Path(features_root) / f"sub-{subject}" / "eeg" / "junifer"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename: features/sub-XX/eeg/junifer/
    desc = f"probe-{probe_number:03d}"
    extension = save_format.lower()
    if marker_type in ["evoked", "state", "sleep"]:
        fname = f"sub-{subject}_task-{task}_desc-{desc}_{marker_type}_aggMarkers.{extension}"
    else:
        fname = f"sub-{subject}_task-{task}_desc-{desc}_aggMarkers.{extension}"
    out_path = out_dir / fname
    
    if out_path.exists() and not overwrite:
        print(f"[SKIP] File exists and overwrite=False: {out_path}")
        return str(out_path)
    
    if save_format.lower() == "csv":
        # Convert to DataFrame for CSV saving with metadata
        df_data = []
        for marker_name, channel_values in aggregated_markers.items():
            for channel, value in channel_values.items():
                row_data = {
                    "subject": subject,
                    "task": task,
                    "probe_number": probe_number,
                    "probe_label": label,
                    "marker_type": marker_type,
                    "marker": marker_name,
                    "channel": channel,
                    "value": value
                }
                
                # Add probe metadata if available
                if probe_metadata:
                    row_data.update({
                        "onoff": probe_metadata.get("onoff", ""),
                        "valence": probe_metadata.get("valence", ""),
                        "time": probe_metadata.get("time", ""),
                        "confidence": probe_metadata.get("confidence", ""),
                        "average": probe_metadata.get("average", ""),
                        "selfother": probe_metadata.get("selfother", ""),
                        "trial_type": probe_metadata.get("trial_type", ""),
                        "correctness": probe_metadata.get("correctness", ""),
                    })
                
                df_data.append(row_data)
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        df.to_csv(out_path, index=False)
        
    else:
        # Prepare output structure for PKL
        pkl_data = {
            "markers": aggregated_markers,
            "metadata": {
                "subject": subject,
                "task": task,
                "probe_number": probe_number,
                "label": label,
                **extra_meta,
            },
            "info": {
                "created_at": str(pd.Timestamp.now()),
                "storage_type": "aggregated_probe_markers",
                "n_markers": len(aggregated_markers),
                "marker_names": list(aggregated_markers.keys()),
            },
        }
        
        # Save PKL file
        with open(out_path, "wb") as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Report file size
    file_size = out_path.stat().st_size
    print(f"[SAVE] {out_path} ({file_size:,} bytes, {file_size/1024:.1f} KB)")
    
    # Debug: Show marker count breakdown
    spectral_count = sum(1 for name in aggregated_markers.keys() if any(band in name for band in ['_delta', '_theta', '_alpha', '_beta', '_gamma']))
    non_spectral_count = len(aggregated_markers) - spectral_count
    print(f"       {len(aggregated_markers)} total markers ({spectral_count} spectral, {non_spectral_count} non-spectral)")
    print(f"       Format: {save_format.upper()}")
    
    return str(out_path)


def process_subject_task(cfg: Dict, subject: str, task: str) -> pd.DataFrame:
    """
    Process a single subject-task combination to aggregate markers by probe.
    
    This function:
    1. Loads per-epoch marker PKL files (evoked and state)
    2. Loads preprocessed epochs for metadata and outlier detection
    3. Parses events to identify probes
    4. For each probe:
       - Labels as onTask/offTask
       - Selects appropriate trials (different for evoked vs state)
       - Applies outlier detection
       - Aggregates markers across epochs
       - Saves aggregated markers
    
    Parameters
    ----------
    cfg : Dict
        Configuration dictionary
    subject : str
        Subject identifier
    task : str
        Task identifier
        
    Returns
    -------
    pd.DataFrame
        Summary dataframe with probe information
    """
    project = cfg.get("project", {})
    derivatives_root = project.get("derivatives_root")
    features_root = project.get("features_root")
    bids_root = project.get("bids_root")
    input_evoked_desc = project.get("input_evoked_desc", "evoked")
    input_state_desc = project.get("input_state_desc", "state")
    input_sleep_desc = project.get("input_sleep_desc", "sleep")
    
    if not derivatives_root or not features_root:
        raise ValueError("'project.derivatives_root' and 'project.features_root' must be set in config")
    
    # Configuration parameters
    trial_sel = cfg.get("trial_selection", {})
    only_go_correct = bool(trial_sel.get("only_go_correct", True))
    evoked_dist_min = int(trial_sel.get("evoked_distance_min", -5))
    evoked_dist_max = int(trial_sel.get("evoked_distance_max", -1))
    state_dist_min = int(trial_sel.get("state_distance_min", -999))
    state_dist_max = int(trial_sel.get("state_distance_max", -1))
    min_req = int(trial_sel.get("min_required_distances", 3))
    
    labeling = cfg.get("labeling", {})
    onoff_thr = int(labeling.get("onoff_threshold", 50))
    
    outlier_cfg = cfg.get("outlier_detection", {})
    enable_outlier_detection = bool(outlier_cfg.get("enable_outlier_detection", False))
    epoch_z_threshold = float(outlier_cfg.get("epoch_z_threshold", 3.0))
    baseline_distance_min = int(outlier_cfg.get("baseline_distance_min", -10))
    baseline_distance_max = int(outlier_cfg.get("baseline_distance_max", -1))
    min_baseline_epochs = int(outlier_cfg.get("min_baseline_epochs", 5))
    trimmean_percent = float(outlier_cfg.get("trimmean_percent", 20.0))
    
    out_cfg = cfg.get("output", {})
    overwrite = bool(out_cfg.get("overwrite", True))

    sleep_cfg = cfg.get("sleep_markers", {})
    sleep_nan_policy = str(sleep_cfg.get("nan_policy", "null")).lower()
    
    # Get marker name mapping for renaming
    marker_name_mapping = cfg.get("marker_name_mapping", {})
    
    # STRICT CONNECTIVITY POLICY VALIDATION
    connectivity_cfg = cfg.get("connectivity", {})
    channel_reduction = connectivity_cfg.get("channel_reduction", "")
    if channel_reduction != "mean_pairs":
        raise ValueError("Connectivity reduction policy must be 'mean_pairs'")
    
    print(f"\n{'='*80}")
    print(f"Processing sub-{subject} task-{task}")
    print(f"{'='*80}")
    
    # ========================================================================
    # LOAD EVOKED EPOCHS AND EVENTS (for evoked markers and outlier detection)
    # ========================================================================
    evoked_epochs = None
    evoked_events_df = None
    try:
        evoked_epochs, evoked_events_tsv, _ = read_derivative_epochs(
            derivatives_root=derivatives_root,
            subject=subject,
            task=task,
            desc=input_evoked_desc,
            preload=True,
            bids_root=bids_root,
        )
        # Parse evoked events
        evoked_events_df = parse_events_tsv(evoked_events_tsv)
        evoked_events_df = enrich_events_with_parsed_fields(evoked_events_df)
        evoked_events_df['epoch_index'] = np.arange(len(evoked_events_df))
        evoked_events_df = evoked_events_df.rename(columns={"row_index": "tsv_row_index"})
        print(f"Loaded evoked epochs: {len(evoked_events_df)} epochs")
    except FileNotFoundError as exc:
        print(f"[WARN] Evoked epochs not found: {exc}")
    
    state_epochs = None
    state_events_df = None
    try:
        state_epochs, state_events_tsv, _ = read_derivative_epochs(
            derivatives_root=derivatives_root,
            subject=subject,
            task=task,
            desc=input_state_desc,
            preload=True,
            bids_root=bids_root,
        )
        # Parse state events - completely independent from evoked
        state_events_df = parse_events_tsv(state_events_tsv)
        state_events_df = enrich_events_with_parsed_fields(state_events_df)
        state_events_df['epoch_index'] = np.arange(len(state_events_df))
        state_events_df = state_events_df.rename(columns={"row_index": "tsv_row_index"})
        print(f"Loaded state epochs: {len(state_events_df)} epochs")
    except FileNotFoundError as exc:
        print(f"[WARN] State epochs not found: {exc}")
    sleep_epochs = None
    sleep_events_df = None
    try:
        sleep_epochs, sleep_events_tsv, _ = read_derivative_epochs(
            derivatives_root=derivatives_root,
            subject=subject,
            task=task,
            desc=input_sleep_desc,
            preload=True,
            bids_root=bids_root,
        )
        sleep_events_df = parse_events_tsv(sleep_events_tsv)
        sleep_events_df = enrich_events_with_parsed_fields(sleep_events_df)
        sleep_events_df["epoch_index"] = np.arange(len(sleep_events_df))
        sleep_events_df = sleep_events_df.rename(columns={"row_index": "tsv_row_index"})
        print(f"Loaded sleep epochs: {len(sleep_events_df)} epochs")
    except FileNotFoundError as exc:
        print(f"[WARN] Sleep epochs not found: {exc}")
    
    if evoked_events_df is None and state_events_df is None and sleep_events_df is None:
        print(f"[ERROR] No epochs found for sub-{subject} task-{task}")
        return pd.DataFrame()
    
    events_df = evoked_events_df if evoked_events_df is not None else (
        state_events_df if state_events_df is not None else sleep_events_df
    )
    
    # Get unique probes
    probes = events_df["probe_number"].dropna().unique()
    if len(probes) == 0:
        print(f"[WARN] No probes found for sub-{subject} task-{task}")
        return pd.DataFrame()
    
    print(f"Found {len(probes)} probes")
    
    # Label probes
    probe_labels = label_probe_onoff(events_df=events_df, onoff_threshold=onoff_thr)
    
    # Load marker PKL files
    junifer_dir = Path(features_root) / f"sub-{subject}" / "eeg" / "junifer"
    
    # Try to load evoked markers
    evoked_pkl_path = junifer_dir / f"sub-{subject}_task-{task}_desc-evoked_markers.pkl"
    evoked_data = None
    if evoked_pkl_path.exists():
        # Comprehensive PKL debug
        debug_pkl_comprehensive(str(evoked_pkl_path))
        evoked_data = load_marker_pkl(str(evoked_pkl_path))
        # Rename markers using mapping from config
        if evoked_data and "markers" in evoked_data and marker_name_mapping:
            evoked_data["markers"] = rename_markers(evoked_data["markers"], marker_name_mapping)
            print(f"  Renamed evoked markers: {list(evoked_data['markers'].keys())}")
    else:
        print(f"[WARN] Evoked markers not found: {evoked_pkl_path}")
    
    # Try to load state markers
    state_pkl_path = junifer_dir / f"sub-{subject}_task-{task}_desc-state_markers.pkl"
    state_data = None
    if state_pkl_path.exists():
        # Comprehensive PKL debug
        debug_pkl_comprehensive(str(state_pkl_path))
        state_data = load_marker_pkl(str(state_pkl_path))
        # Rename markers using mapping from config
        if state_data and "markers" in state_data and marker_name_mapping:
            state_data["markers"] = rename_markers(state_data["markers"], marker_name_mapping)
            print(f"  Renamed state markers: {list(state_data['markers'].keys())}")
    else:
        print(f"[WARN] State markers not found: {state_pkl_path}")
    sleep_pkl_path = junifer_dir / f"sub-{subject}_task-{task}_desc-sleep_markers.pkl"
    sleep_data = None
    if sleep_pkl_path.exists():
        debug_pkl_comprehensive(str(sleep_pkl_path))
        sleep_data = load_marker_pkl(str(sleep_pkl_path))
        if sleep_data and "markers" in sleep_data and marker_name_mapping:
            sleep_data["markers"] = rename_markers(sleep_data["markers"], marker_name_mapping)
            print(f"  Renamed sleep markers: {list(sleep_data['markers'].keys())}")
    else:
        print(f"[WARN] Sleep markers not found: {sleep_pkl_path}")
    
    if evoked_data is None and state_data is None and sleep_data is None:
        print(f"[ERROR] No marker PKL files found for sub-{subject} task-{task}")
        return pd.DataFrame()
    
    # STRICT CHANNEL EXTRACTION - ONLY FROM metadata.fif_info.channel_names
    channels = []
    if evoked_data is not None:
        metadata = evoked_data.get("metadata", {})
        fif_info = metadata.get("fif_info", {})
        channels = fif_info.get("channel_names", [])
    
    if not channels and state_data is not None:
        metadata = state_data.get("metadata", {})
        fif_info = metadata.get("fif_info", {})
        channels = fif_info.get("channel_names", [])
    
    if not channels and sleep_data is not None:
        metadata = sleep_data.get("metadata", {})
        fif_info = metadata.get("fif_info", {})
        channels = fif_info.get("channel_names", [])
    
    if not channels:
        raise ValueError("Missing metadata.fif_info.channel_names in PKL")
    
    # FILTER OUT EOG CHANNELS - they should not be processed for any markers
    eog_channels = ['VEOG', 'HEOG']
    channels = [ch for ch in channels if ch not in eog_channels]
    print(f"Filtered out EOG channels: {eog_channels}")
    print(f"Processing {len(channels)} EEG channels")
    
    # Process each probe
    outputs: List[Dict] = []
    
    for probe_num in sorted(probes):
        probe_num = int(probe_num)
        print(f"\n--- Probe {probe_num} ---")
        
        # Comprehensive debug for first probe
        if probe_num == 1:
            debug_probe_comprehensive(
                probe_num=probe_num,
                subject=subject,
                task=task,
                evoked_data=evoked_data,
                state_data=state_data,
                epochs_data=evoked_epochs if evoked_epochs is not None else state_epochs,
                events_data=events_df,
                channels=channels,
            )
        
        # Get probe label
        label = probe_labels.get(probe_num, "unknown") if hasattr(probe_labels, "get") else (
            probe_labels.loc[probe_num] if probe_num in probe_labels.index else "unknown"
        )
        label_str = str(label)
        
        # Get probe metadata from events_df
        probe_metadata = {}
        if events_df is not None:
            probe_events = events_df[events_df['probe_number'] == probe_num]
            if len(probe_events) > 0:
                probe_row = probe_events.iloc[0]  # Get first row for this probe
                probe_metadata = {
                    "onoff": probe_row.get('onoff', ''),
                    "valence": probe_row.get('valence', ''),
                    "time": probe_row.get('time', ''),
                    "confidence": probe_row.get('confidence', ''),
                    "average": probe_row.get('average', ''),
                    "selfother": probe_row.get('selfother', ''),
                    "trial_type": probe_row.get('trial_type', ''),
                    "correctness": probe_row.get('correctness', ''),
                }
        
        # ========================================================================
        # PROCESS EVOKED MARKERS (ALL MARKERS with evoked trials)
        # ========================================================================
        if evoked_data is not None and evoked_events_df is not None:
            print(f"  Processing EVOKED markers (distance {evoked_dist_min} to {evoked_dist_max})")
            
            # Select trials for evoked markers using EVOKED events
            evoked_trials = select_trials_for_probe(
                events_df=evoked_events_df,
                probe_number=probe_num,
                only_go_correct=only_go_correct,
                distance_min=evoked_dist_min,
                distance_max=evoked_dist_max,
            )
            
            if len(evoked_trials) >= min_req:
                # Get epoch indices from EVOKED events
                evoked_epoch_indices = evoked_trials["epoch_index"].astype(int).tolist()
                
                # Apply outlier detection if enabled (using EVOKED epochs)
                if enable_outlier_detection:
                    valid_indices, outlier_stats = detect_outlier_epochs(
                        epochs=evoked_epochs,
                        events_df=evoked_events_df,
                        probe_number=probe_num,
                        baseline_distance_min=baseline_distance_min,
                        baseline_distance_max=baseline_distance_max,
                        min_baseline_epochs=min_baseline_epochs,
                        z_threshold=epoch_z_threshold,
                    )
                    final_evoked_indices = [idx for idx in evoked_epoch_indices if idx in valid_indices]
                    print(f"    {len(final_evoked_indices)} valid epochs after outlier removal")
                else:
                    final_evoked_indices = evoked_epoch_indices
                    print(f"    {len(final_evoked_indices)} epochs (outlier detection disabled)")
                
                if len(final_evoked_indices) > 0:
                    # Aggregate ALL markers from evoked data using evoked trials
                    evoked_markers_dict = evoked_data.get("markers", {})
                    aggregated_evoked = {}
                    
                    for marker_name, marker_data in evoked_markers_dict.items():
                        print(f"      Processing evoked marker: {marker_name}")
                        
                        aggregated = aggregate_marker_epochs(
                            marker_data=marker_data,
                            marker_name=marker_name,
                            epoch_indices=final_evoked_indices,
                            channels=channels,
                            use_trimmean=not enable_outlier_detection,
                            trimmean_percent=trimmean_percent,
                            subject_id=subject,
                            task=task,
                        )
                        
                        # Handle spectral vs non-spectral markers
                        if "overall" in aggregated:
                            # Non-spectral marker
                            aggregated_evoked[marker_name] = aggregated["overall"]
                        else:
                            # Spectral marker with bands
                            for band_name, band_data in aggregated.items():
                                band_marker_name = f"{marker_name}_{band_name}"
                                aggregated_evoked[band_marker_name] = band_data
                    
                    # Save evoked markers
                    if len(aggregated_evoked) > 0:
                        extra_meta = {
                            "n_evoked_trials": len(final_evoked_indices),
                            "aggregation_method": "trimmean" if not enable_outlier_detection else "outlier_detection",
                            "trimmean_percent": trimmean_percent if not enable_outlier_detection else None,
                        }
                        
                        out_path = save_aggregated_markers(
                            aggregated_markers=aggregated_evoked,
                            features_root=features_root,
                            subject=subject,
                            task=task,
                            probe_number=probe_num,
                            label=label_str,
                            extra_meta=extra_meta,
                            overwrite=overwrite,
                            marker_type="evoked",
                            save_format="csv",
                            probe_metadata=probe_metadata,
                        )
                        
                        outputs.append({
                            "subject": subject,
                            "task": task,
                            "probe_number": probe_num,
                            "label": label_str,
                            "marker_type": "evoked",
                            "n_markers": len(aggregated_evoked),
                            "n_trials": len(final_evoked_indices),
                            "output_path": out_path,
                        })
                else:
                    print(f"    No valid epochs for evoked markers")
            else:
                print(f"    Insufficient evoked trials: {len(evoked_trials)} < {min_req}")
        
        # ========================================================================
        # PROCESS STATE MARKERS (ALL MARKERS with state trials - INDEPENDENT)
        # ========================================================================
        if state_data is not None and state_events_df is not None:
            print(f"  Processing STATE markers (distance {state_dist_min} to {state_dist_max})")
            
            # Select trials for state markers using STATE events (independent system)
            # NOTE: State markers represent continuous brain states, not trial-locked responses
            # Therefore, we don't filter by trial_type (only_go_correct=False)
            state_trials = select_trials_for_probe(
                events_df=state_events_df,
                probe_number=probe_num,
                only_go_correct=False,  # State events have trial_type='unknown'
                distance_min=state_dist_min,
                distance_max=state_dist_max,
            )
            
            if len(state_trials) > 0:
                # Get epoch indices from STATE events (0, 1, 2... up to 120)
                state_epoch_indices = state_trials["epoch_index"].astype(int).tolist()
                
                # Apply outlier detection if enabled (using STATE epochs)
                if enable_outlier_detection:
                    valid_indices, outlier_stats = detect_outlier_epochs(
                        epochs=state_epochs,
                        events_df=state_events_df,
                        probe_number=probe_num,
                        baseline_distance_min=baseline_distance_min,
                        baseline_distance_max=baseline_distance_max,
                        min_baseline_epochs=min_baseline_epochs,
                        z_threshold=epoch_z_threshold,
                    )
                    final_state_indices = [idx for idx in state_epoch_indices if idx in valid_indices]
                    print(f"    {len(final_state_indices)} valid epochs after outlier removal")
                else:
                    final_state_indices = state_epoch_indices
                    print(f"    {len(final_state_indices)} epochs (outlier detection disabled)")
                
                if len(final_state_indices) > 0:
                    # Aggregate ALL markers from state data using state trials
                    state_markers_dict = state_data.get("markers", {})
                    aggregated_state = {}
                    
                    for marker_name, marker_data in state_markers_dict.items():
                        print(f"      Processing state marker: {marker_name}")
                        
                        aggregated = aggregate_marker_epochs(
                            marker_data=marker_data,
                            marker_name=marker_name,
                            epoch_indices=final_state_indices,
                            channels=channels,
                            use_trimmean=not enable_outlier_detection,
                            trimmean_percent=trimmean_percent,
                            subject_id=subject,
                            task=task,
                        )
                        
                        # Handle spectral vs non-spectral markers
                        if "overall" in aggregated:
                            # Non-spectral marker
                            aggregated_state[marker_name] = aggregated["overall"]
                        else:
                            # Spectral marker with bands
                            for band_name, band_data in aggregated.items():
                                band_marker_name = f"{marker_name}_{band_name}"
                                aggregated_state[band_marker_name] = band_data
                    
                    # Save state markers
                    if len(aggregated_state) > 0:
                        extra_meta = {
                            "n_state_trials": len(final_state_indices),
                            "aggregation_method": "trimmean" if not enable_outlier_detection else "outlier_detection",
                            "trimmean_percent": trimmean_percent if not enable_outlier_detection else None,
                        }
                        
                        out_path = save_aggregated_markers(
                            aggregated_markers=aggregated_state,
                            features_root=features_root,
                            subject=subject,
                            task=task,
                            probe_number=probe_num,
                            label=label_str,
                            extra_meta=extra_meta,
                            overwrite=overwrite,
                            marker_type="state",
                            save_format="csv",
                            probe_metadata=probe_metadata,
                        )
                        
                        outputs.append({
                            "subject": subject,
                            "task": task,
                            "probe_number": probe_num,
                            "label": label_str,
                            "marker_type": "state",
                            "n_markers": len(aggregated_state),
                            "n_trials": len(final_state_indices),
                            "output_path": out_path,
                        })
                else:
                    print(f"    No valid epochs for state markers after aggregation")
            else:
                print(f"    No trials found for state markers")
        if sleep_data is not None and sleep_events_df is not None:
            print(f"  Processing SLEEP markers (distance {state_dist_min} to {state_dist_max})")
            sleep_trials = select_trials_for_probe(
                events_df=sleep_events_df,
                probe_number=probe_num,
                only_go_correct=False,
                distance_min=state_dist_min,
                distance_max=state_dist_max,
            )
            if len(sleep_trials) > 0:
                sleep_epoch_indices = sleep_trials["epoch_index"].astype(int).tolist()
                if enable_outlier_detection:
                    valid_indices, outlier_stats = detect_outlier_epochs(
                        epochs=sleep_epochs,
                        events_df=sleep_events_df,
                        probe_number=probe_num,
                        baseline_distance_min=baseline_distance_min,
                        baseline_distance_max=baseline_distance_max,
                        min_baseline_epochs=min_baseline_epochs,
                        z_threshold=epoch_z_threshold,
                    )
                    final_sleep_indices = [idx for idx in sleep_epoch_indices if idx in valid_indices]
                    print(f"    {len(final_sleep_indices)} valid epochs after outlier removal")
                else:
                    final_sleep_indices = sleep_epoch_indices
                    print(f"    {len(final_sleep_indices)} epochs (outlier detection disabled)")
                if len(final_sleep_indices) > 0:
                    sleep_markers_dict = sleep_data.get("markers", {})
                    aggregated_sleep = {}
                    for marker_name, marker_data in sleep_markers_dict.items():
                        print(f"      Processing sleep marker: {marker_name}")
                        aggregated = aggregate_marker_epochs(
                            marker_data=marker_data,
                            marker_name=marker_name,
                            epoch_indices=final_sleep_indices,
                            channels=channels,
                            use_trimmean=not enable_outlier_detection,
                            trimmean_percent=trimmean_percent,
                            subject_id=subject,
                            task=task,
                        )
                        if "overall" in aggregated:
                            aggregated_sleep[marker_name] = aggregated["overall"]
                        else:
                            for feature_name, feature_data in aggregated.items():
                                feature_marker_name = f"{marker_name}_{feature_name}"
                                aggregated_sleep[feature_marker_name] = feature_data
                    if len(aggregated_sleep) > 0:
                        aggregated_sleep = apply_nan_policy_to_sleep_markers(
                            aggregated_sleep=aggregated_sleep,
                            nan_policy=sleep_nan_policy,
                        )
                        extra_meta = {
                            "n_sleep_trials": len(final_sleep_indices),
                            "aggregation_method": "trimmean" if not enable_outlier_detection else "outlier_detection",
                            "trimmean_percent": trimmean_percent if not enable_outlier_detection else None,
                        }
                        out_path = save_aggregated_markers(
                            aggregated_markers=aggregated_sleep,
                            features_root=features_root,
                            subject=subject,
                            task=task,
                            probe_number=probe_num,
                            label=label_str,
                            extra_meta=extra_meta,
                            overwrite=overwrite,
                            marker_type="sleep",
                            save_format="csv",
                            probe_metadata=probe_metadata,
                        )
                        outputs.append({
                            "subject": subject,
                            "task": task,
                            "probe_number": probe_num,
                            "label": label_str,
                            "marker_type": "sleep",
                            "n_markers": len(aggregated_sleep),
                            "n_trials": len(final_sleep_indices),
                            "output_path": out_path,
                        })
                else:
                    print(f"    No valid epochs for sleep markers after aggregation")
            else:
                print(f"    No trials found for sleep markers")
    
    return pd.DataFrame(outputs)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Junifer markers by probe for mind-wandering analysis"
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--task", help="Process only this task")
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    
    subjects_cfg = cfg.get("subjects", [])
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = subjects_cfg
    
    tasks_cfg = cfg.get("tasks", [])
    if args.task:
        tasks = [args.task]
    else:
        tasks = tasks_cfg
    
    all_results = []
    
    for sub in subjects:
        for tsk in tasks:
            try:
                df = process_subject_task(cfg, sub, tsk)
                if not df.empty:
                    all_results.append(df)
            except Exception as exc:
                print(f"[ERROR] Failed to process sub-{sub} task-{tsk}: {exc}")
                import traceback
                traceback.print_exc()
    
    # Save summary CSV
    if all_results:
        summary_df = pd.concat(all_results, ignore_index=True)
        features_root = cfg.get("project", {}).get("features_root")
        summary_path = Path(features_root) / "junifer_probe_aggregation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SUMMARY] Saved summary to: {summary_path}")
        print(f"Processed {len(summary_df)} probes across {len(subjects)} subjects and {len(tasks)} tasks")
    else:
        print("\n[WARN] No results to save")


if __name__ == "__main__":
    main()
