#!/usr/bin/env python3

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# DATA_PATH = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # Base directory to search
DATA_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # Base directory to search

SUBJECT_ID = "07"  # Participant ID (e.g., "02", "03", etc.)
SART_SESSION = "ALL"  # SART session: "Sart1", "Sart2", "Sart3", "Sart4", or "ALL" for average across all sessions
DATA_TYPE = "evoked"  # options: 'raw', 'evoked', 'state'
OUTPUT_DIR = "./spectral_topography_plots"  # Directory to save plots
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)  # Adjusted for evoked data (h_freq=45 Hz)
}
WINDOW_SIZE = 2.0  # seconds for spectral analysis
OVERLAP = 0.5  # overlap fraction for spectral analysis
VERBOSE = True  # Print extra information
# =============================================================================

import os
import sys
import glob
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import psd_array_welch


def find_data_files(path: str, subject_id: str, sart_session: str, data_type: str) -> List[str]:
    """
    Find data files for a specific subject, session, and data type.
    
    Parameters
    ----------
    path : str
        Base directory to search
    subject_id : str
        Subject identifier
    sart_session : str
        SART session identifier or "ALL" for all sessions
    data_type : str
        Type of data to find ('raw', 'evoked', 'state')
        
    Returns
    -------
    List[str]
        List of matching file paths
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Define file patterns based on data type and naming conventions from pipeline
    patterns = []
    
    if data_type == "raw":
        # Clean raw data after ICA
        patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-icaClean_eeg.fif",
            f"**/sub-{subject_id}_task-{sart_session}_desc-csd_eeg.fif",  # CSD version
        ]
    elif data_type == "evoked":
        # Evoked epochs
        patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-evoked_epo.fif",
        ]
    elif data_type == "state":
        # State epochs
        patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-state_epo.fif",
        ]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    files: List[str] = []
    
    # Handle "ALL" sessions case
    if sart_session == "ALL":
        all_sessions = ["Sart1", "Sart2", "Sart3", "Sart4"]
        for session in all_sessions:
            for pattern in patterns:
                # Replace the session placeholder in the pattern
                session_pattern = pattern.replace(sart_session, session)
                files.extend(glob.glob(os.path.join(path, session_pattern), recursive=True))
    else:
        # Single session case
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(path, pattern), recursive=True))
    
    # Deduplicate while preserving order
    seen = set()
    unique_files: List[str] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    
    return sorted(unique_files)


def load_and_average_data(filepaths: List[str], data_type: str) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """
    Load and average multiple data files of the same type.
    
    Parameters
    ----------
    filepaths : List[str]
        List of file paths to load and average
    data_type : str
        Type of data ('raw', 'evoked', 'state')
        
    Returns
    -------
    Union[mne.io.BaseRaw, mne.Epochs]
        Averaged MNE object
    """
    if not filepaths:
        raise ValueError("No file paths provided")
    
    if len(filepaths) == 1:
        return load_data(filepaths[0], data_type)
    
    if VERBOSE:
        print(f"Loading and averaging {len(filepaths)} files...")
    
    # Load all files
    data_objects = []
    for filepath in filepaths:
        if VERBOSE:
            print(f"  Loading: {os.path.basename(filepath)}")
        data_objects.append(load_data(filepath, data_type))
    
    # Average the data
    if data_type == "raw":
        # For raw data, concatenate and then average
        raw_concatenated = mne.concatenate_raws(data_objects, preload=True)
        # Create a copy for averaging
        raw_averaged = raw_concatenated.copy()
        # Average across time (this is a simplified approach)
        # In practice, you might want to segment and average epochs
        return raw_averaged
    else:
        # For epochs, concatenate and average
        epochs_concatenated = mne.concatenate_epochs(data_objects)
        return epochs_concatenated


def load_data(filepath: str, data_type: str) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """
    Load data from file based on type.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    data_type : str
        Type of data ('raw', 'evoked', 'state')
        
    Returns
    -------
    Union[mne.io.BaseRaw, mne.Epochs]
        Loaded MNE object
    """
    if data_type == "raw":
        return mne.io.read_raw_fif(filepath, preload=True, verbose=False)
    elif data_type in ["evoked", "state"]:
        return mne.read_epochs(filepath, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def compute_spectral_power(data: Union[mne.io.BaseRaw, mne.Epochs], 
                          freq_bands: Dict[str, Tuple[float, float]], 
                          window_size: float = 2.0,
                          overlap: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Compute spectral power for different frequency bands using proper epoch-wise processing.
    
    Parameters
    ----------
    data : Union[mne.io.BaseRaw, mne.Epochs]
        MNE data object
    freq_bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to frequency ranges
    window_size : float
        Window size in seconds for spectral analysis
    overlap : float
        Overlap fraction for spectral analysis
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping band names to power values per channel
    """
    sfreq = data.info['sfreq']
    
    # Set parameters for Welch method (acordes al muestreo de 250 Hz)
    n_per_seg = int(sfreq)  # ~1 second for 250 Hz
    n_overlap = int(n_per_seg * overlap)  # 50% overlap
    
    print(f"PSD parameters: n_per_seg={n_per_seg}, n_overlap={n_overlap}, sfreq={sfreq}")
    
    if isinstance(data, mne.io.BaseRaw):
        # For raw data, use psd_array_welch
        data_array = data.get_data(picks='eeg')
        
        # Ensure n_per_seg doesn't exceed signal length
        if n_per_seg > data_array.shape[-1]:
            n_per_seg = data_array.shape[-1] // 2
            n_overlap = int(n_per_seg * overlap)
            print(f"Adjusted window size: {n_per_seg / sfreq:.3f} (s)")
        
        psds, freqs = psd_array_welch(
            data_array,
            sfreq=sfreq,
            fmin=1,
            fmax=45,  # Limit to 45 Hz for evoked data
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            n_jobs=1
        )
        
        # For raw data, we have (n_channels, n_freqs)
        # Sum power in each band (linear, not average)
        band_powers = {}
        print("\nSpectral power statistics (Raw data):")
        for band_name, (fmin, fmax) in freq_bands.items():
            # Adjust gamma band for evoked data
            if band_name == 'gamma' and fmax > 45:
                fmax = 45
                print(f"Adjusted {band_name} band to {fmin}-{fmax} Hz for evoked data")
            
            # Find frequency indices
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            # Sum power across frequencies (linear)
            band_power = np.sum(psds[:, freq_mask], axis=1)
            band_powers[band_name] = band_power
            
            print(f"- {band_name.capitalize()}: mean={np.mean(band_power):.2e}, "
                  f"std={np.std(band_power):.2e}, "
                  f"min={np.min(band_power):.2e}, "
                  f"max={np.max(band_power):.2e}")
    
    else:  # Epochs - use native MNE PSD computation
        print("Computing PSD using MNE native method for epochs...")
        
        # Use MNE's native PSD computation for epochs
        psds = data.compute_psd(
            method='welch',
            fmin=1,
            fmax=45,  # Limit to 45 Hz for evoked data
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            n_jobs=1
        )
        
        # Get frequency array
        freqs = psds.freqs
        
        # Get power data: (n_epochs, n_channels, n_freqs)
        power_data = psds.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
        
        print(f"PSD shape: {power_data.shape}")
        
        # Process each frequency band
        band_powers = {}
        print("\nSpectral power statistics (Epochs data):")
        
        for band_name, (fmin, fmax) in freq_bands.items():
            # Adjust gamma band for evoked data
            if band_name == 'gamma' and fmax > 45:
                fmax = 45
                print(f"Adjusted {band_name} band to {fmin}-{fmax} Hz for evoked data")
            
            # Find frequency indices
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            
            # Sum power across frequencies for each epoch and channel
            # Shape: (n_epochs, n_channels)
            epoch_band_power = np.sum(power_data[:, :, freq_mask], axis=2)
            
            # Robust aggregation across epochs (trimmed mean 80%)
            n_epochs = epoch_band_power.shape[0]
            trim_percent = 0.1  # 10% from each end = 80% trimmed mean
            n_trim = int(n_epochs * trim_percent)
            
            # Sort each channel independently and trim
            band_power = np.zeros(epoch_band_power.shape[1])
            for ch_idx in range(epoch_band_power.shape[1]):
                ch_power = epoch_band_power[:, ch_idx]
                sorted_power = np.sort(ch_power)
                trimmed_power = sorted_power[n_trim:-n_trim] if n_trim > 0 else sorted_power
                band_power[ch_idx] = np.mean(trimmed_power)
            
            band_powers[band_name] = band_power
            
            print(f"- {band_name.capitalize()}: mean={np.mean(band_power):.2e}, "
                  f"std={np.std(band_power):.2e}, "
                  f"min={np.min(band_power):.2e}, "
                  f"max={np.max(band_power):.2e}")
    
    return band_powers


def create_topography_plots(data: Union[mne.io.BaseRaw, mne.Epochs],
                           band_powers: Dict[str, np.ndarray],
                           output_dir: str,
                           subject_id: str,
                           sart_session: str,
                           data_type: str) -> None:
    """
    Create and save topography plots for each frequency band.
    
    Parameters
    ----------
    data : Union[mne.io.BaseRaw, mne.Epochs]
        MNE data object
    band_powers : Dict[str, np.ndarray]
        Dictionary mapping band names to power values per channel
    output_dir : str
        Directory to save plots
    subject_id : str
        Subject identifier
    sart_session : str
        SART session identifier (or "ALL" for averaged sessions)
    data_type : str
        Type of data being plotted
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with subplots for all bands
    n_bands = len(band_powers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (band_name, power_values) in enumerate(band_powers.items()):
        ax = axes[idx]
        
        # Calculate vmin=0 and vmax=95th percentile for sequential scale
        vmin = 0
        vmax = np.percentile(power_values, 95)
        
        # Create topography plot with sequential scale
        im, _ = mne.viz.plot_topomap(
            power_values,
            data.info,
            axes=ax,
            show=False,
            cmap='viridis',
            contours=6,
            vlim=(vmin, vmax)
        )
        
        ax.set_title(f'{band_name.capitalize()} Band\n({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplots
    for idx in range(n_bands, len(axes)):
        axes[idx].set_visible(False)
    
    # Create appropriate title based on session type
    if sart_session == "ALL":
        title = f'Spectral Topography - Subject {subject_id}, All Sessions Average, {data_type.capitalize()} Data'
        filename_suffix = "all_sessions"
    else:
        title = f'Spectral Topography - Subject {subject_id}, {sart_session}, {data_type.capitalize()} Data'
        filename_suffix = sart_session.lower()
    
    # Add overall title
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the combined plot
    output_filename = f"sub-{subject_id}_task-{filename_suffix}_desc-{data_type}_spectral_topography.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined topography plot: {output_path}")
    
    # Also create individual plots for each band
    for band_name, power_values in band_powers.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Calculate vmin=0 and vmax=95th percentile for sequential scale
        vmin = 0
        vmax = np.percentile(power_values, 95)
        
        im, _ = mne.viz.plot_topomap(
            power_values,
            data.info,
            axes=ax,
            show=False,
            cmap='viridis',
            contours=6,
            vlim=(vmin, vmax)
        )
        
        if sart_session == "ALL":
            title_text = f'{band_name.capitalize()} Band ({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)\nSubject {subject_id}, All Sessions Average, {data_type.capitalize()} Data'
        else:
            title_text = f'{band_name.capitalize()} Band ({FREQ_BANDS[band_name][0]}-{FREQ_BANDS[band_name][1]} Hz)\nSubject {subject_id}, {sart_session}, {data_type.capitalize()} Data'
        
        ax.set_title(title_text)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Save individual plot
        individual_filename = f"sub-{subject_id}_task-{filename_suffix}_desc-{data_type}_{band_name}_topography.png"
        individual_path = os.path.join(output_dir, individual_filename)
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {band_name} band plot: {individual_path}")


def print_data_summary(data: Union[mne.io.BaseRaw, mne.Epochs], data_type: str, sart_session: str) -> None:
    """
    Print summary information about the loaded data.
    
    Parameters
    ----------
    data : Union[mne.io.BaseRaw, mne.Epochs]
        MNE data object
    data_type : str
        Type of data
    sart_session : str
        Session identifier (or "ALL" for averaged sessions)
    """
    if sart_session == "ALL":
        print(f"\nLoaded {data_type} data (averaged across all sessions):")
    else:
        print(f"\nLoaded {data_type} data:")
    
    if isinstance(data, mne.io.BaseRaw):
        print(f"- Duration: {data.n_times / data.info['sfreq']:.2f} seconds")
        print(f"- Sampling rate: {data.info['sfreq']:.1f} Hz")
        print(f"- Channels: {data.info['nchan']}")
        ch_types = mne.channel_indices_by_type(data.info)
        ch_type_counts = {k: len(v) for k, v in ch_types.items()}
        print(f"- Channel types: {ch_type_counts}")
        
    else:  # Epochs
        print(f"- Number of epochs: {len(data)}")
        print(f"- Epoch duration: {data.tmax - data.tmin:.3f} seconds")
        print(f"- Time range: {data.tmin:.3f} to {data.tmax:.3f} seconds")
        print(f"- Sampling rate: {data.info['sfreq']:.1f} Hz")
        print(f"- Channels: {data.info['nchan']}")
        ch_types = mne.channel_indices_by_type(data.info)
        ch_type_counts = {k: len(v) for k, v in ch_types.items()}
        print(f"- Channel types: {ch_type_counts}")
        
        if data.event_id:
            print(f"- Event types: {list(data.event_id.keys())}")


def main() -> None:
    """
    Main function to explore spectral topography of EEG data.
    
    This script loads EEG data (raw or epoched), computes spectral power
    for different frequency bands, and creates topography plots showing
    the spatial distribution of power for each band.
    """
    if VERBOSE:
        if SART_SESSION == "ALL":
            print(f"Searching for {DATA_TYPE} data for subject {SUBJECT_ID}, all sessions")
        else:
            print(f"Searching for {DATA_TYPE} data for subject {SUBJECT_ID}, session {SART_SESSION}")
        print(f"Search path: {DATA_PATH}")
    
    # Find data files
    files = find_data_files(DATA_PATH, SUBJECT_ID, SART_SESSION, DATA_TYPE)
    
    if not files:
        if SART_SESSION == "ALL":
            print(f"No {DATA_TYPE} files found for subject {SUBJECT_ID} across all sessions.")
        else:
            print(f"No {DATA_TYPE} files found for subject {SUBJECT_ID}, session {SART_SESSION}.")
        print("Available subjects: 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43")
        print("Available sessions: Sart1, Sart2, Sart3, Sart4, or ALL for average across all sessions")
        print("Available data types: raw, evoked, state")
        sys.exit(1)
    
    if VERBOSE:
        if SART_SESSION == "ALL":
            print(f"Found {len(files)} file(s) for subject {SUBJECT_ID} across all sessions:")
        else:
            print(f"Found {len(files)} file(s) for subject {SUBJECT_ID}, session {SART_SESSION}:")
        for idx, f in enumerate(files):
            print(f"  [{idx}] {f}")
    
    # Load data (single file or multiple files for averaging)
    if len(files) == 1:
        if VERBOSE:
            print(f"\nLoading: {files[0]}")
        data = load_data(files[0], DATA_TYPE)
    else:
        if VERBOSE:
            print(f"\nLoading and averaging {len(files)} files...")
        data = load_and_average_data(files, DATA_TYPE)
    
    # Print data summary
    print_data_summary(data, DATA_TYPE, SART_SESSION)
    
    # Compute spectral power for each frequency band
    if VERBOSE:
        print(f"\nComputing spectral power for frequency bands...")
        print(f"Window size: {WINDOW_SIZE} seconds")
        print(f"Overlap: {OVERLAP}")
    
    band_powers = compute_spectral_power(data, FREQ_BANDS, WINDOW_SIZE, OVERLAP)
    
    # Print power statistics for each band
    if VERBOSE:
        print(f"\nSpectral power statistics:")
        for band_name, power_values in band_powers.items():
            print(f"- {band_name.capitalize()}: mean={np.mean(power_values):.2e}, "
                  f"std={np.std(power_values):.2e}, "
                  f"min={np.min(power_values):.2e}, "
                  f"max={np.max(power_values):.2e}")
    
    # Create and save topography plots
    if VERBOSE:
        print(f"\nCreating topography plots...")
        print(f"Output directory: {OUTPUT_DIR}")
    
    create_topography_plots(
        data, 
        band_powers, 
        OUTPUT_DIR, 
        SUBJECT_ID, 
        SART_SESSION, 
        DATA_TYPE
    )
    
    print(f"\n✓ Spectral topography analysis completed successfully!")
    print(f"✓ Plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
