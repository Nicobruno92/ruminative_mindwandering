#!/usr/bin/env python3
"""
Explore Spectral Topography from PKL Markers

This script reads pre-computed spectral markers from PKL files (created by the
h5_to_pkl converter) and creates topography plots by averaging across epochs.

Unlike explore_spectral_topography.py which computes markers on-the-fly, this
script uses the already computed markers stored in PKL format, making it useful
for debugging the pipeline and comparing results.

Usage:
    python explore_spectral_topography_from_pkl.py
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# Path to the PKL file containing markers
PKL_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/features/sub-07/eeg/junifer/sub-07_task-Sart1_desc-evoked_markers.pkl"

# Subject and session info (for plot titles)
SUBJECT_ID = "07"
SESSION = "Sart1"
DATA_TYPE = "evoked"  # 'state' or 'evoked'

# Output directory for plots
OUTPUT_DIR = "./spectral_topography_plots_from_pkl"

# Frequency bands to plot (must match the bands in the PKL file)
# Standard order: [delta=0, theta=1, alpha=2, beta=3, gamma=4]
FREQ_BANDS = {
    'delta': 0,
    'theta': 1,
    'alpha': 2,
    'beta': 3,
    'gamma': 4
}

# Spectral marker names to look for in PKL
# These are the exact names from the h5_to_pkl converter
SPECTRAL_MARKERS = {
    'absolute': 'EEG_psd_bands_spectralpower',
    'relative': 'EEG_psd_relative_spectralpower'
}

# Which spectral marker to use for plotting
MARKER_TO_PLOT = 'absolute'  # 'absolute' or 'relative'

# Aggregation method for epochs
AGGREGATION = 'mean'  # 'mean', 'median', 'trim_mean'
TRIM_PERCENT = 10.0  # For trim_mean: percentage to trim from each end

# Color scale for topoplots
USE_FIXED_SCALE = True  # If True, use VMIN/VMAX; if False, use auto scale (0 to 95th percentile)
VMIN = -120  # Minimum value for color scale (in dB for spectral power)
VMAX = -97   # Maximum value for color scale (in dB for spectral power)

VERBOSE = True
# =============================================================================

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import mne

# Minimal classes for unpickling (must match h5_to_pkl_converter.py)
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


# Register classes for pickle
import types
h5_to_pkl_converter = types.ModuleType('h5_to_pkl_converter')
h5_to_pkl_converter.MarkerData = MarkerData
h5_to_pkl_converter.EpochData = EpochData
h5_to_pkl_converter.ChannelData = ChannelData
h5_to_pkl_converter.EpochMetadata = EpochMetadata
sys.modules['h5_to_pkl_converter'] = h5_to_pkl_converter

# Mock MNE for unpickling annotations
if 'mne' not in sys.modules or not hasattr(sys.modules['mne'], 'Annotations'):
    class MockAnnotations:
        def __init__(self, onset=None, duration=None, description=None):
            self.onset = onset or []
            self.duration = duration or []
            self.description = description or []
    
    if 'mne' not in sys.modules:
        mne_module = types.ModuleType('mne')
        sys.modules['mne'] = mne_module
    else:
        mne_module = sys.modules['mne']
    
    mne_module.Annotations = MockAnnotations
    
    # Also create mne.annotations submodule
    mne_annotations = types.ModuleType('mne.annotations')
    mne_annotations.Annotations = MockAnnotations
    mne_annotations._AnnotationsExtrasList = list
    mne_annotations._AnnotationsExtrasDict = dict
    sys.modules['mne.annotations'] = mne_annotations


def load_pkl_file(pkl_path: str) -> Dict[str, Any]:
    """
    Load PKL file containing markers.
    
    Parameters
    ----------
    pkl_path : str
        Path to PKL file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with markers, metadata, and epoch_metadata
    """
    if VERBOSE:
        print(f"Loading PKL file: {pkl_path}")
    
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    if VERBOSE:
        markers = data.get('markers', {})
        metadata = data.get('metadata', {})
        print(f"  Loaded {len(markers)} markers")
        
        if metadata:
            fif_info = metadata.get('fif_info', {})
            print(f"  Epochs: {fif_info.get('n_epochs', 'N/A')}")
            print(f"  Channels: {fif_info.get('n_channels', 'N/A')}")
            print(f"  Sampling rate: {fif_info.get('sfreq', 'N/A')} Hz")
        
        if markers:
            print(f"  Available markers:")
            for marker_name in markers.keys():
                print(f"    - {marker_name}")
    
    return data


def extract_spectral_power_by_band(
    marker_data: MarkerData,
    band_idx: int,
    aggregation: str = 'trim_mean',
    trim_percent: float = 10.0
) -> np.ndarray:
    """
    Extract spectral power for a specific frequency band across all epochs.
    
    Parameters
    ----------
    marker_data : MarkerData
        Marker data object containing spectral power
    band_idx : int
        Band index (0=delta, 1=theta, 2=alpha, 3=beta, 4=gamma)
    aggregation : str
        Aggregation method: 'mean', 'median', 'trim_mean'
    trim_percent : float
        Percentage to trim from each end for trim_mean
        
    Returns
    -------
    np.ndarray
        Array of power values per channel (n_channels,)
    """
    if not hasattr(marker_data, '_epoch_data'):
        raise ValueError("Marker data does not have epoch data")
    
    # Get channel names (spectral markers use 64 EEG channels)
    channel_names = marker_data.channel_names
    n_channels = len(channel_names)
    
    # Collect data for all epochs
    epoch_indices = sorted(marker_data._epoch_data.keys())
    n_epochs = len(epoch_indices)
    
    if VERBOSE:
        print(f"    Extracting band {band_idx} from {n_epochs} epochs, {n_channels} channels")
    
    # Initialize array: (n_epochs, n_channels)
    epoch_channel_data = np.full((n_epochs, n_channels), np.nan)
    
    # Extract data for each epoch and channel
    for ep_idx, epoch_idx in enumerate(epoch_indices):
        epoch_data = marker_data._epoch_data[epoch_idx]
        
        if not hasattr(epoch_data, '_channel_data'):
            continue
        
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name in epoch_data._channel_data:
                ch_data = epoch_data._channel_data[ch_name]
                
                if hasattr(ch_data, 'data'):
                    data_val = ch_data.data
                    
                    if isinstance(data_val, np.ndarray):
                        # Spectral markers have shape (5,) for 5 bands
                        if band_idx < len(data_val):
                            epoch_channel_data[ep_idx, ch_idx] = data_val[band_idx]
                    else:
                        # Scalar value (shouldn't happen for spectral markers)
                        epoch_channel_data[ep_idx, ch_idx] = float(data_val)
    
    # Aggregate across epochs for each channel
    channel_power = np.zeros(n_channels)
    
    for ch_idx in range(n_channels):
        ch_values = epoch_channel_data[:, ch_idx]
        
        # Remove NaN values
        valid_values = ch_values[~np.isnan(ch_values)]
        
        if len(valid_values) == 0:
            channel_power[ch_idx] = np.nan
            continue
        
        # Apply aggregation method
        if aggregation == 'mean':
            channel_power[ch_idx] = np.mean(valid_values)
        elif aggregation == 'median':
            channel_power[ch_idx] = np.median(valid_values)
        elif aggregation == 'trim_mean':
            # Trim mean: remove top and bottom percentiles
            n_trim = int(len(valid_values) * (trim_percent / 100.0))
            if n_trim > 0 and len(valid_values) > 2 * n_trim:
                sorted_values = np.sort(valid_values)
                trimmed_values = sorted_values[n_trim:-n_trim]
                channel_power[ch_idx] = np.mean(trimmed_values)
            else:
                channel_power[ch_idx] = np.mean(valid_values)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    return channel_power


def create_mne_info_from_metadata(metadata: Dict[str, Any]) -> mne.Info:
    """
    Create MNE Info object from PKL metadata for plotting.
    
    Parameters
    ----------
    metadata : Dict[str, Any]
        Metadata dictionary from PKL file
        
    Returns
    -------
    mne.Info
        MNE Info object for topography plotting
    """
    fif_info = metadata.get('fif_info', {})
    
    # Get channel names (for spectral markers, use first 64 EEG channels)
    all_ch_names = fif_info.get('channel_names', [])
    
    # Filter to EEG channels only (remove EOG)
    eeg_ch_names = [ch for ch in all_ch_names 
                    if not ch.startswith('EOG') and ch not in ['VEOG', 'HEOG']]
    
    # Take first 64 EEG channels (spectral markers use 64 channels)
    ch_names = eeg_ch_names[:64]
    
    sfreq = fif_info.get('sfreq', 250.0)
    
    # Create Info object
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Try to read custom montage from BVEF file
    bvef_path = '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Preprocessing_pipeline_new/CACS-64_REF.bvef'
    
    try:
        if os.path.exists(bvef_path):
            # Read custom montage from BVEF file
            montage = mne.channels.read_custom_montage(bvef_path)
            info.set_montage(montage, on_missing='ignore')
            if VERBOSE:
                print(f"  Using custom montage from: {bvef_path}")
        else:
            # Fallback to standard 10-20 system
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, on_missing='ignore')
            if VERBOSE:
                print(f"  Using standard 10-20 montage (BVEF not found)")
    except Exception as e:
        # If custom montage fails, use standard 10-20
        if VERBOSE:
            print(f"  Warning: Could not load custom montage: {e}")
            print(f"  Falling back to standard 10-20 montage")
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')
    
    return info


def create_topography_plots(
    band_powers: Dict[str, np.ndarray],
    info: mne.Info,
    output_dir: str,
    subject_id: str,
    session: str,
    data_type: str,
    marker_type: str
) -> None:
    """
    Create and save topography plots for each frequency band.
    
    Parameters
    ----------
    band_powers : Dict[str, np.ndarray]
        Dictionary mapping band names to power values per channel
    info : mne.Info
        MNE Info object for plotting
    output_dir : str
        Directory to save plots
    subject_id : str
        Subject identifier
    session : str
        Session identifier
    data_type : str
        Type of data being plotted
    marker_type : str
        Type of marker ('absolute' or 'relative')
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for all bands
    n_bands = len(band_powers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (band_name, power_values) in enumerate(band_powers.items()):
        ax = axes[idx]
        
        # Set color scale limits
        valid_values = power_values[~np.isnan(power_values)]
        if len(valid_values) == 0:
            print(f"Warning: No valid values for {band_name} band")
            continue
        
        if USE_FIXED_SCALE:
            vmin = VMIN
            vmax = VMAX
        else:
            vmin = 0
            vmax = np.percentile(valid_values, 95)
        
        # Create topography plot with sequential scale
        im, _ = mne.viz.plot_topomap(
            power_values,
            info,
            axes=ax,
            show=False,
            cmap='viridis',
            contours=6,
            vlim=(vmin, vmax)
        )
        
        ax.set_title(f'{band_name.capitalize()} Band')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    title = f'Spectral Topography from PKL ({marker_type.capitalize()})\nSubject {subject_id}, {session}, {data_type.capitalize()} Data'
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the combined plot
    output_filename = f"sub-{subject_id}_task-{session}_desc-{data_type}_{marker_type}_spectral_topography_from_pkl.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined topography plot: {output_path}")
    
    # Also create individual plots for each band
    for band_name, power_values in band_powers.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Set color scale limits
        valid_values = power_values[~np.isnan(power_values)]
        if len(valid_values) == 0:
            plt.close()
            continue
        
        if USE_FIXED_SCALE:
            vmin = VMIN
            vmax = VMAX
        else:
            vmin = 0
            vmax = np.percentile(valid_values, 95)
        
        im, _ = mne.viz.plot_topomap(
            power_values,
            info,
            axes=ax,
            show=False,
            cmap='viridis',
            contours=6,
            vlim=(vmin, vmax)
        )
        
        title_text = f'{band_name.capitalize()} Band\nSubject {subject_id}, {session}, {data_type.capitalize()} Data\n({marker_type.capitalize()} Power from PKL)'
        ax.set_title(title_text)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Save individual plot
        individual_filename = f"sub-{subject_id}_task-{session}_desc-{data_type}_{marker_type}_{band_name}_topography_from_pkl.png"
        individual_path = os.path.join(output_dir, individual_filename)
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {band_name} band plot: {individual_path}")


def main() -> None:
    """
    Main function to create spectral topography plots from PKL markers.
    """
    if VERBOSE:
        print(f"Creating spectral topography plots from PKL markers")
        print(f"PKL file: {PKL_PATH}")
        print(f"Subject: {SUBJECT_ID}, Session: {SESSION}, Data type: {DATA_TYPE}")
        print(f"Marker to plot: {MARKER_TO_PLOT}")
        print(f"Aggregation: {AGGREGATION}")
        if AGGREGATION == 'trim_mean':
            print(f"Trim percent: {TRIM_PERCENT}%")
    
    # Load PKL file
    pkl_data = load_pkl_file(PKL_PATH)
    
    # Get markers and metadata
    markers = pkl_data.get('markers', {})
    metadata = pkl_data.get('metadata', {})
    
    if not markers:
        print("Error: No markers found in PKL file")
        sys.exit(1)
    
    # Find the spectral marker to plot
    marker_name = SPECTRAL_MARKERS.get(MARKER_TO_PLOT)
    if not marker_name:
        print(f"Error: Unknown marker type: {MARKER_TO_PLOT}")
        print(f"Available types: {list(SPECTRAL_MARKERS.keys())}")
        sys.exit(1)
    
    if marker_name not in markers:
        print(f"Error: Marker '{marker_name}' not found in PKL file")
        print(f"Available markers: {list(markers.keys())}")
        sys.exit(1)
    
    marker_data = markers[marker_name]
    
    if VERBOSE:
        print(f"\nProcessing marker: {marker_name}")
        if hasattr(marker_data, 'marker_type'):
            print(f"  Marker type: {marker_data.marker_type}")
        if hasattr(marker_data, 'n_epochs'):
            print(f"  Number of epochs: {marker_data.n_epochs}")
        if hasattr(marker_data, 'channel_names'):
            print(f"  Number of channels: {len(marker_data.channel_names)}")
    
    # Extract power for each frequency band
    band_powers = {}
    
    if VERBOSE:
        print(f"\nExtracting spectral power for frequency bands...")
    
    for band_name, band_idx in FREQ_BANDS.items():
        if VERBOSE:
            print(f"  Processing {band_name} band (index {band_idx})...")
        
        try:
            power_values = extract_spectral_power_by_band(
                marker_data,
                band_idx,
                aggregation=AGGREGATION,
                trim_percent=TRIM_PERCENT
            )
            
            band_powers[band_name] = power_values
            
            # Print statistics
            valid_values = power_values[~np.isnan(power_values)]
            if len(valid_values) > 0:
                print(f"    {band_name.capitalize()}: mean={np.mean(valid_values):.2e}, "
                      f"std={np.std(valid_values):.2e}, "
                      f"min={np.min(valid_values):.2e}, "
                      f"max={np.max(valid_values):.2e}")
            else:
                print(f"    {band_name.capitalize()}: No valid values")
        
        except Exception as e:
            print(f"    Error processing {band_name} band: {e}")
            continue
    
    if not band_powers:
        print("Error: No band powers extracted")
        sys.exit(1)
    
    # Create MNE Info object for plotting
    if VERBOSE:
        print(f"\nCreating MNE Info object for plotting...")
    
    info = create_mne_info_from_metadata(metadata)
    
    # Create and save topography plots
    if VERBOSE:
        print(f"\nCreating topography plots...")
        print(f"Output directory: {OUTPUT_DIR}")
    
    create_topography_plots(
        band_powers,
        info,
        OUTPUT_DIR,
        SUBJECT_ID,
        SESSION,
        DATA_TYPE,
        MARKER_TO_PLOT
    )
    
    print(f"\n✓ Spectral topography analysis from PKL completed successfully!")
    print(f"✓ Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
