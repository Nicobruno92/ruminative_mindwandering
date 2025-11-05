#!/usr/bin/env python3
"""
Explore Spectral Topography from PKL using Junifer Structure

This script reads pre-computed spectral markers from PKL files (created by the
h5_to_pkl converter) and creates topography plots using junifer-style utilities.
Based on the junifer plotting_from_pkl examples but adapted for CACS data.

Usage:
    python explore_spectral_topography_from_pkl_junifer.py
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# Path to the PKL file containing markers
# PKL_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/features/sub-07/eeg/junifer/sub-07_task-Sart1_desc-evoked_markers.pkl"
PKL_PATH = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/features/sub-07/eeg/junifer/sub-07_task-Sart1_desc-evoked_markers.pkl"

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
SPECTRAL_MARKERS = [
    "EEG_psd_bands_spectralpower",      # Absolute power in dB
    "EEG_psd_relative_spectralpower"    # Relative power (linear)
]

# Aggregation method for epochs
AGGREGATION = 'trim_mean'  # 'mean', 'trim_mean'
TRIM_PERCENT = 10.0  # For trim_mean: percentage to trim from each end

VERBOSE = True
# =============================================================================

import os
import sys
import pickle
import types
import numpy as np
import matplotlib.pyplot as plt
import mne

# Minimal classes for unpickling
class EpochMetadata:
    def __init__(self, epoch_idx, event, annotation=None, behavioral_data=None):
        self.epoch_idx = epoch_idx
        self.event = event

class ChannelData:
    def __init__(self, channel_name, channel_idx, data, metadata):
        self.channel_name = channel_name
        self.channel_idx = channel_idx
        self.data = data
        self.metadata = metadata

class EpochData:
    def __init__(self, epoch_idx, metadata, channel_names):
        self.epoch_idx = epoch_idx
        self.metadata = metadata
        self.channel_names = channel_names
        self._channel_data = {}

class MarkerData:
    def __init__(self, marker_name, marker_type, channel_names, n_epochs):
        self.marker_name = marker_name
        self.marker_type = marker_type
        self.channel_names = channel_names
        self.n_epochs = n_epochs
        self._epoch_data = {}
        self.metadata = {}

# Register classes for pickle
h5_to_pkl_converter = types.ModuleType('h5_to_pkl_converter')
h5_to_pkl_converter.MarkerData = MarkerData
h5_to_pkl_converter.EpochData = EpochData
h5_to_pkl_converter.ChannelData = ChannelData
h5_to_pkl_converter.EpochMetadata = EpochMetadata
sys.modules['h5_to_pkl_converter'] = h5_to_pkl_converter


def load_pkl_file(pkl_path):
    """Load PKL file."""
    if VERBOSE:
        print(f"Loading: {pkl_path}")
    
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
    
    return data


def extract_spectral_power_by_band(marker_data, band_idx, aggregation='trim_mean', trim_percent=10.0):
    """Extract spectral power for a specific frequency band."""
    channel_names = marker_data.channel_names
    n_channels = len(channel_names)
    epoch_indices = sorted(marker_data._epoch_data.keys())
    n_epochs = len(epoch_indices)
    
    if VERBOSE:
        print(f"    Extracting band {band_idx}: {n_epochs} epochs × {n_channels} channels")
    
    epoch_channel_data = np.full((n_epochs, n_channels), np.nan)
    
    for ep_idx, epoch_idx in enumerate(epoch_indices):
        epoch_data = marker_data._epoch_data[epoch_idx]
        
        if not hasattr(epoch_data, '_channel_data'):
            continue
        
        for ch_idx, ch_name in enumerate(channel_names):
            if ch_name in epoch_data._channel_data:
                ch_data = epoch_data._channel_data[ch_name]
                
                if hasattr(ch_data, 'data'):
                    data_val = ch_data.data
                    
                    if isinstance(data_val, np.ndarray) and band_idx < len(data_val):
                        epoch_channel_data[ep_idx, ch_idx] = data_val[band_idx]
    
    # Aggregate across epochs
    channel_power = np.zeros(n_channels)
    
    for ch_idx in range(n_channels):
        ch_values = epoch_channel_data[:, ch_idx]
        valid_values = ch_values[~np.isnan(ch_values)]
        
        if len(valid_values) == 0:
            channel_power[ch_idx] = np.nan
            continue
        
        if aggregation == 'mean':
            channel_power[ch_idx] = np.mean(valid_values)
        elif aggregation == 'trim_mean':
            n_trim = int(len(valid_values) * (trim_percent / 100.0))
            if n_trim > 0 and len(valid_values) > 2 * n_trim:
                sorted_values = np.sort(valid_values)
                trimmed_values = sorted_values[n_trim:-n_trim]
                channel_power[ch_idx] = np.mean(trimmed_values)
            else:
                channel_power[ch_idx] = np.mean(valid_values)
    
    return channel_power


def create_mne_info_from_metadata(metadata):
    """Create MNE Info object from metadata."""
    fif_info = metadata.get('fif_info', {})
    all_ch_names = fif_info.get('channel_names', [])
    
    eeg_ch_names = [ch for ch in all_ch_names 
                    if not ch.startswith('EOG') and ch not in ['VEOG', 'HEOG']]
    ch_names = eeg_ch_names[:64]
    sfreq = fif_info.get('sfreq', 250.0)
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Try custom montage, fallback to standard
    bvef_path = '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Preprocessing_pipeline_new/CACS-64_REF.bvef'
    
    try:
        if os.path.exists(bvef_path):
            montage = mne.channels.read_custom_montage(bvef_path)
            info.set_montage(montage, on_missing='ignore')
            if VERBOSE:
                print(f"  Using custom montage from: {bvef_path}")
        else:
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, on_missing='ignore')
            if VERBOSE:
                print(f"  Using standard 10-20 montage (BVEF not found)")
    except Exception as e:
        if VERBOSE:
            print(f"  Warning: Could not load custom montage: {e}")
            print(f"  Falling back to standard 10-20 montage")
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')
    
    return info


def create_topography_plots(band_powers, info, output_dir, subject_id, session, data_type, marker_name):
    """Create and save topography plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    use_db = "psd_bands" in marker_name
    marker_type = "absolute" if use_db else "relative"
    
    # Combined figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (band_name, power_values) in enumerate(band_powers.items()):
        ax = axes[idx]
        
        valid_values = power_values[~np.isnan(power_values)]
        if len(valid_values) == 0:
            continue
        
        if use_db:
            vmin = np.percentile(valid_values, 5)
            vmax = np.percentile(valid_values, 95)
        else:
            vmin = 0
            vmax = np.percentile(valid_values, 95)
        
        im, _ = mne.viz.plot_topomap(
            power_values, info, axes=ax, show=False,
            cmap='viridis', contours=6, vlim=(vmin, vmax)
        )
        
        ax.set_title(f'{band_name.capitalize()} Band')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    for idx in range(len(band_powers), len(axes)):
        axes[idx].set_visible(False)
    
    title = f'Spectral Topography ({marker_type.capitalize()}) - from PKL\nSubject {subject_id}, {session}, {data_type.capitalize()} Data'
    fig.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the combined plot
    output_filename = f"sub-{subject_id}_task-{session}_desc-{data_type}_{marker_type}_spectral_topography_from_pkl.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    
    # Also create individual plots for each band
    for band_name, power_values in band_powers.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        valid_values = power_values[~np.isnan(power_values)]
        if len(valid_values) == 0:
            plt.close()
            continue
        
        if use_db:
            vmin = np.percentile(valid_values, 5)
            vmax = np.percentile(valid_values, 95)
        else:
            vmin = 0
            vmax = np.percentile(valid_values, 95)
        
        im, _ = mne.viz.plot_topomap(
            power_values, info, axes=ax, show=False,
            cmap='viridis', contours=6, vlim=(vmin, vmax)
        )
        
        title_text = f'{band_name.capitalize()} Band\nSubject {subject_id}, {session}, {data_type.capitalize()} Data\n({marker_type.capitalize()} Power from PKL)'
        ax.set_title(title_text)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Save individual plot
        individual_filename = f"sub-{subject_id}_task-{session}_desc-{data_type}_{marker_type}_{band_name}_topography_from_pkl.png"
        individual_path = os.path.join(output_dir, individual_filename)
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {band_name} band plot: {individual_path}")


def main():
    """Main function."""
    print("=" * 70)
    print("Spectral Power Topography Visualization from PKL (Junifer Style)")
    print("=" * 70)
    
    pkl_data = load_pkl_file(PKL_PATH)
    markers = pkl_data.get('markers', {})
    metadata = pkl_data.get('metadata', {})
    
    info = create_mne_info_from_metadata(metadata)
    
    for marker_name in SPECTRAL_MARKERS:
        print(f"\n{'='*70}")
        print(f"Processing: {marker_name}")
        print(f"{'='*70}")
        
        if marker_name not in markers:
            print(f"✗ Marker not found: {marker_name}")
            continue
        
        marker_data = markers[marker_name]
        
        try:
            band_powers = {}
            for band_name, band_idx in FREQ_BANDS.items():
                if VERBOSE:
                    print(f"  Extracting {band_name}...")
                power_values = extract_spectral_power_by_band(
                    marker_data, band_idx, AGGREGATION, TRIM_PERCENT
                )
                band_powers[band_name] = power_values
                
                # Print statistics
                valid_values = power_values[~np.isnan(power_values)]
                if len(valid_values) > 0 and VERBOSE:
                    print(f"    {band_name.capitalize()}: mean={np.mean(valid_values):.2e}, "
                          f"std={np.std(valid_values):.2e}, "
                          f"min={np.min(valid_values):.2e}, "
                          f"max={np.max(valid_values):.2e}")
            
            create_topography_plots(band_powers, info, OUTPUT_DIR, SUBJECT_ID, SESSION, DATA_TYPE, marker_name)
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ All spectral plots saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

