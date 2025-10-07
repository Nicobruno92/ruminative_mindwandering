import sys
import os
import argparse
import pandas as pd
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None):
        print(f"Processing {desc or 'items'}...")
        return iterable
from scipy.stats import kurtosis, skew, trim_mean, median_abs_deviation
from scipy.signal import find_peaks
from git import Repo
from typing import Dict, Tuple, Optional

# Get the current working directory and repository root
try:
    repo = Repo(os.getcwd(), search_parent_directories=True)
    repo_root = repo.git.rev_parse("--show-toplevel")
except:
    # Fallback if not in a git repository
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (parent of script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add parent directory to path for utils import
sys.path.insert(0, project_root)

from utils.bids_compliance import read_epochs
try:
    from utils.bids_compliance import save_eeg_features_bids, load_eeg_features_bids
except ImportError:
    # Fallback functions if not available
    def save_eeg_features_bids(dataframe, subject, session, task, features_folder, desc='mne_markers'):
        os.makedirs(features_folder, exist_ok=True)
        outname = f"sub-{subject}_ses-{session}_task-{task}_{desc}.csv"
        outpath = os.path.join(features_folder, outname)
        dataframe.to_csv(outpath, index=False)
        return outpath
    
    def load_eeg_features_bids(subject, session, task, derivatives_folder, desc='mne_markers'):
        filename = f"sub-{subject}_ses-{session}_task-{task}_{desc}.csv"
        filepath = os.path.join(derivatives_folder, filename)
        return pd.read_csv(filepath)

try:
    from mne.time_frequency import psd_welch
except ImportError:
    psd_welch = None


def compute_spectral_markers(epochs, tmin=0, tmax=2, fmin=1, fmax=45):
    """
    Compute spectral markers using MNE's built-in functions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epoched data
    tmin : float
        Start time for analysis
    tmax : float
        End time for analysis
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency
        
    Returns
    -------
    dict
        Dictionary with marker names as keys and arrays as values
        Shape: (n_epochs, n_channels) for each marker
        Absolute powers in µV², relative powers as ratios (0-1),
        frequencies in Hz, spectral entropy normalized (0-1)
    """
    # Calculate actual signal length for the analysis window
    sfreq = epochs.info['sfreq']
    signal_length = int((tmax - tmin) * sfreq)
    
    print(f"[DEBUG] Signal length: {signal_length} samples, "
          f"sampling frequency: {sfreq} Hz, "
          f"analysis window: {tmax - tmin:.3f}s")
    
    # Dynamic n_fft: aim for 2 seconds of data for stable 0.5 Hz resolution
    # But ensure it doesn't exceed the actual signal length
    desired_n_fft = int(sfreq * 2)
    n_fft = min(desired_n_fft, signal_length)
    
    # Ensure minimum n_fft for meaningful spectral analysis (at least 256 samples)
    min_n_fft = 256
    if n_fft < min_n_fft:
        print(f"[WARNING] n_fft ({n_fft}) is very small, using minimum value {min_n_fft}")
        n_fft = min(min_n_fft, signal_length)
    
    # If signal is still too short, use the full signal length
    if n_fft > signal_length:
        n_fft = signal_length
        print(f"[WARNING] Using full signal length ({n_fft}) for n_fft due to short data")
    
    n_overlap = n_fft // 2
    
    print(f"[DEBUG] Using n_fft={n_fft}, n_overlap={n_overlap}")
    
    # Compute PSDs using MNE's compute_psd method (newer MNE versions)
    try:
        spectrum = epochs.compute_psd(
            method='welch',
            fmin=fmin, 
            fmax=fmax, 
            tmin=tmin, 
            tmax=tmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            picks='eeg',
            verbose=False
        )
        psds = spectrum.get_data()  # Shape: (n_epochs, n_channels, n_freqs)
        freqs = spectrum.freqs
    except AttributeError:
        # Fallback to older MNE versions
        if psd_welch is None:
            raise ImportError("mne.time_frequency.psd_welch not available")
        psds, freqs = psd_welch(
            epochs, 
            fmin=fmin, 
            fmax=fmax, 
            tmin=tmin, 
            tmax=tmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            picks='eeg',
            verbose=False
        )
    
    # Check scaling and convert units properly
    # Check if MNE stored data in Volts (look for scaling info)
    if hasattr(epochs, '_undo_scaling_dict') and 'eeg' in epochs._undo_scaling_dict:
        # Data was stored in Volts, convert PSD from V²/Hz to µV²/Hz
        psds *= 1e12  # (1e6)² to convert V² to µV²
    # If no scaling dict or data already in µV, PSDs should be in µV²/Hz already
    
    # Define frequency bands (matching NICE pipeline)
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    markers = {}
    
    # Compute total power first (sum across all frequencies)
    total_power = np.sum(psds, axis=2) + 1e-10  # Small epsilon to avoid division by zero
    
    # Compute band powers (use sum, not mean - represents area under curve)
    band_powers = {}
    for band_name, (f_low, f_high) in bands.items():
        # Find frequency indices
        freq_mask = (freqs >= f_low) & (freqs <= f_high)
        
        # Absolute power (sum gives proper power, not mean)
        band_power = np.sum(psds[:, :, freq_mask], axis=2)  # µV²
        band_powers[band_name[0]] = band_power
        markers[band_name[0]] = band_power  # e.g., 'd' for delta
        
        # Normalized power (relative to total power) - true percentage
        normalized_power = band_power / total_power  # 0-1 ratio
        markers[f"{band_name[0]}_n"] = normalized_power
    
    # Add useful band ratios for cognitive markers
    markers['theta_alpha_ratio'] = band_powers['t'] / (band_powers['a'] + 1e-10)
    markers['alpha_beta_ratio'] = band_powers['a'] / (band_powers['b'] + 1e-10)
    markers['t_b_ratio'] = band_powers['t'] / (band_powers['b'] + 1e-10)
    
    # Spectral entropy (normalized to 0-1 range)
    # Normalize PSDs to probabilities
    psd_norm = psds / (np.sum(psds, axis=2, keepdims=True) + 1e-10)
    # Compute entropy
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=2)
    # Normalize by maximum possible entropy (log of number of frequency bins)
    max_entropy = np.log(psds.shape[2])
    markers['se'] = spectral_entropy / max_entropy  # Now 0-1 range
    
    # Vectorized computation of MSF and SEF
    cumsum_psd = np.cumsum(psds, axis=2)
    
    # Median spectral frequency (MSF) - vectorized
    half_power = total_power / 2
    # Find indices where cumulative power exceeds half power
    idx_msf = (cumsum_psd >= half_power[..., None]).argmax(axis=2)
    # Clamp indices to valid range
    idx_msf = np.clip(idx_msf, 0, len(freqs) - 1)
    markers['msf'] = freqs[idx_msf]
    
    # Spectral edge frequencies - vectorized
    for percentile, name in [(0.9, 'sef90'), (0.95, 'sef95')]:
        edge_power = total_power * percentile
        idx_sef = (cumsum_psd >= edge_power[..., None]).argmax(axis=2)
        idx_sef = np.clip(idx_sef, 0, len(freqs) - 1)
        markers[name] = freqs[idx_sef]
    
    return markers


def compute_erp_components(
    epochs, 
    tmin=0, 
    tmax=2,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    peak_windows: Optional[Dict[str, Tuple[float, float, str]]] = None,
    baseline: Tuple[float, float] = (-0.2, 0.0),
    agg: str = "trim",           # 'mean', 'median', or 'trim' (trimmed mean)
    trim_prop: float = 0.1,       # proportion to cut for trimmed mean
    picks: str = "eeg",
    min_prom: float = 0.0         # minimum peak prominence in µV
):
    """
    Compute robust single‑trial ERP features and peak metrics.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epoched EEG data.
    tmin : float
        Start time for analysis
    tmax : float
        End time for analysis
    windows : dict | None
        Mapping of component names to (t0, t1) in seconds for mean‑amplitude
        features. If *None*, a sensible default set (CNV, P1, N1, P2, P3a, P3b)
        is used.
    peak_windows : dict | None
        Mapping of names to (t0, t1, polarity) where polarity is 'pos' for a
        positive peak or 'neg' for a negative one. Two features will be
        returned per entry: ``'<name>_amp'`` and ``'<name>_lat'``.
        If *None*, default peak windows for P1, N1, P2, P3a, P3b are used.
    baseline : tuple
        (t_min, t_max) for baseline correction (applied per epoch).
    agg : {'mean', 'median', 'trim'}
        Aggregation across time inside each window.
    trim_prop : float
        Fraction of samples trimmed from each tail when ``agg='trim'``.
    picks : str | list | slice
        Channels to include (passed to :py:meth:`mne.Epochs.get_data`).
    min_prom : float
        Minimum prominence (in µV) for a peak to be considered.

    Returns
    -------
    dict
        Dictionary with marker names as keys and arrays as values
        Shape: (n_epochs, n_channels) for each marker
    """
    
    # ---------------- Pre‑processing ----------------
    epochs = epochs.copy().apply_baseline(baseline)
    data = epochs.get_data(picks=picks, tmin=tmin, tmax=tmax) * 1e6  # convert to µV
    
    # Get the times corresponding to the cropped data
    # epochs.times gives times for the full epoch, but data is cropped
    full_times = epochs.times
    time_mask = (full_times >= tmin) & (full_times <= tmax)
    analysis_times = full_times[time_mask]
    
    # Ensure the analysis_times matches the data dimensions
    if len(analysis_times) != data.shape[2]:
        print(f"[DEBUG] Time mismatch: analysis_times={len(analysis_times)}, data.shape[2]={data.shape[2]}")
        # Create new time array that matches the data
        analysis_times = np.linspace(tmin, tmax, data.shape[2])
    
    n_ep, n_ch, _ = data.shape

    # Default ERP windows if none supplied
    if windows is None:
        windows = {
            'cnv': (-0.004, 0.596),   # Contingent Negative Variation
            'p1': (0.100, 0.150),     # P1 component
            'n1': (0.150, 0.200),     # N1 component  
            'p2': (0.200, 0.275),     # P2 component
            'p3a': (0.275, 0.375),    # P3a component
            'p3b': (0.375, 0.600),    # P3b component
        }

    # Default peak windows if none supplied
    if peak_windows is None:
        peak_windows = {
            'p1': (0.100, 0.150, 'pos'),   # P1 component - positive peak
            'n1': (0.150, 0.200, 'neg'),   # N1 component - negative peak
            'p2': (0.200, 0.275, 'pos'),   # P2 component - positive peak
            'p3a': (0.275, 0.375, 'pos'),  # P3a component - positive peak
            'p3b': (0.375, 0.600, 'pos'),  # P3b component - positive peak
        }

    # Aggregators ----------------------------------------------------------
    if agg == "mean":
        _agg = lambda x: x.mean(axis=2)
    elif agg == "median":
        _agg = lambda x: np.median(x, axis=2)
    elif agg == "trim":
        _agg = lambda x: trim_mean(x, proportiontocut=trim_prop, axis=2)
    else:
        raise ValueError("agg must be 'mean', 'median', or 'trim'")

    markers: Dict[str, np.ndarray] = {}

    # Mean‑amplitude features ---------------------------------------------
    for name, (t0, t1) in windows.items():
        # Find time indices for this component within the analysis window
        comp_mask = (analysis_times >= t0) & (analysis_times <= t1)
        if not comp_mask.any():
            markers[name] = np.full((n_ep, n_ch), np.nan)
            continue
        seg = data[:, :, comp_mask]
        markers[name] = _agg(seg)

    # Global time‑domain statistics ---------------------------------------
    markers.update({
        "mean_amp": _agg(data),
        "mad":      median_abs_deviation(data, axis=2),
        "p2p_amp":  np.ptp(data, axis=2),
        "rms":      np.sqrt((data ** 2).mean(axis=2)),
        "std":      np.std(data, axis=2),
        "var":      np.var(data, axis=2),
        "skew":     skew(data, axis=2, nan_policy="omit"),
        "kurtosis": kurtosis(data, axis=2, nan_policy="omit"),
    })

    # Peak detection -------------------------------------------------------
    for name, (t0, t1, polarity) in peak_windows.items():
        mask = (analysis_times >= t0) & (analysis_times <= t1)
        t_seg = analysis_times[mask]

        amp_arr = np.full((n_ep, n_ch), np.nan)
        lat_arr = np.full((n_ep, n_ch), np.nan)

        for ep in range(n_ep):
            for ch in range(n_ch):
                sig = data[ep, ch, mask]
                sig_proc = -sig if polarity == "neg" else sig

                try:
                    idx, props = find_peaks(sig_proc, prominence=min_prom)
                    if idx.size:
                        best = idx[np.argmax(props["prominences"])]
                        amp_arr[ep, ch] = sig[best]    # original sign
                        lat_arr[ep, ch] = t_seg[best]
                except:
                    # Handle cases where peak detection fails
                    continue

        markers[f"{name}_amp"] = amp_arr
        markers[f"{name}_lat"] = lat_arr

    return markers


def get_marker_unit(marker_name):
    """
    Return the unit for each marker type.
    
    Parameters
    ----------
    marker_name : str
        Name of the marker
        
    Returns
    -------
    str
        Unit string for the marker
    """
    # Spectral power markers (absolute power)
    spectral_power_markers = ['d', 't', 'a', 'b', 'g']
    
    # Normalized spectral markers (ratios)
    normalized_markers = ['d_n', 't_n', 'a_n', 'b_n', 'g_n']
    
    # Band ratio markers (cognitive markers)
    band_ratio_markers = ['theta_alpha_ratio', 't_b_ratio','alpha_beta_ratio']
    
    # Frequency markers
    frequency_markers = ['msf', 'sef90', 'sef95']
    
    # Amplitude markers (time domain)
    amplitude_markers = ['cnv', 'p1', 'n1', 'p2', 'p3a', 'p3b', 
                        'mean_amp', 'p2p_amp', 'rms', 'std', 'mad']
    
    # Variance markers
    variance_markers = ['var']
    
    # Dimensionless markers (0-1 range)
    dimensionless_markers = ['se', 'kurtosis', 'skew']
    
    # Peak amplitude markers (any marker ending with '_amp')
    if marker_name.endswith('_amp'):
        return 'µV'
    
    # Latency markers (any marker ending with '_lat')
    elif marker_name.endswith('_lat'):
        return 's'
    
    elif marker_name in spectral_power_markers:
        return 'µV²'
    elif marker_name in normalized_markers:
        return 'ratio'
    elif marker_name in band_ratio_markers:
        return 'ratio'
    elif marker_name in frequency_markers:
        return 'Hz'
    elif marker_name in amplitude_markers:
        return 'µV'
    elif marker_name in variance_markers:
        return 'µV²'
    elif marker_name in dimensionless_markers:
        return 'dimensionless'
    else:
        return 'unknown'


def load_eeg_markers(derivatives_folder, subject, session, task, desc='mne_markers'):
    """
    Load EEG markers from BIDS-compliant structure.
    
    Parameters
    ----------
    derivatives_folder : str
        Path to derivatives directory
    subject : str
        Subject ID
    session : str
        Session ID
    task : str
        Task name
    desc : str
        Description for the markers
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the EEG markers
    """
    try:
        df = load_eeg_features_bids(
            subject=subject,
            session=session,
            task=task,
            derivatives_folder=derivatives_folder,
            desc=desc
        )
        print(f"Loaded EEG markers for {subject} {session} {task}")
        return df
    except FileNotFoundError as e:
        print(f"EEG markers not found for {subject} {session} {task}: {e}")
        return None


def process_one_subject(root, subject, task, data_type, desc, output_dir, 
                        tmin=0, tmax=2, session=None):
    """
    Process one subject/task and compute all markers.
    
    Parameters
    ----------
    root : str
        Root directory path
    subject : str
        Subject ID
    task : str
        Task name
    data_type : str
        Data type (e.g., 'eeg')
    desc : str
        Description (e.g., 'autoPreproc')
    output_dir : str
        Output directory
    tmin : float
        Start time for marker computation
    tmax : float
        End time for marker computation
    session : str, optional
        Session ID (if applicable)
    """
    # Handle output directory path to be relative to project root
    if output_dir.startswith('./') or not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, output_dir.lstrip('./')))
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    try:
        # Read epochs - handle both session-based and non-session-based data
        if session:
            epochs, events = read_epochs(
                os.path.join(root, 'derivatives_nico'), 
                subject, session, task, data_type, desc=desc
            )
        else:
            epochs, events = read_epochs(
                os.path.join(root, 'derivatives_nico'), 
                subject, task, data_type, desc=desc
            )
        
        # Remove EOG channels if present
        eog_channels = {'VEOG', 'HEOG', 'EOG', 'EOG1', 'EOG2'}
        eeg_ch_names = [ch for ch in epochs.ch_names 
                        if ch not in eog_channels]
        if len(eeg_ch_names) < len(epochs.ch_names):
            session_str = f" {session}" if session else ""
            print(f"[INFO] Removing EOG channels for {subject}{session_str} {task}")
            epochs = epochs.pick_channels(eeg_ch_names)
        
        session_str = f" {session}" if session else ""
        print(f"Processing {subject}{session_str} {task}: {len(epochs)} epochs, "
              f"{len(epochs.ch_names)} channels")
        
        # Print epoch timing information for debugging
        print(f"[DEBUG] Epoch time range: {epochs.tmin:.3f} to {epochs.tmax:.3f}s, "
              f"duration: {epochs.tmax - epochs.tmin:.3f}s")
        print(f"[DEBUG] Analysis window: {tmin:.3f} to {tmax:.3f}s, "
              f"requested duration: {tmax - tmin:.3f}s")
        
        # Adjust tmax if it exceeds the available data
        actual_tmax = min(tmax, epochs.tmax)
        if actual_tmax != tmax:
            print(f"[WARNING] Adjusting tmax from {tmax:.3f}s to {actual_tmax:.3f}s "
                  f"(limited by available data)")
        
        # Compute spectral markers (with CAR reference)
        session_str = f" {session}" if session else ""
        print(f"Computing spectral markers for {subject}{session_str} {task}...")
        spectral_markers = compute_spectral_markers(epochs, tmin, actual_tmax)
        
        # Re-reference to mastoids before computing ERP markers
        ref_channels = ['TP9', 'TP10']
        available_ref_channels = [ch for ch in ref_channels if ch in epochs.ch_names]
        
        if len(available_ref_channels) == 2:
            print(f"[INFO] Re-referencing to mastoids {ref_channels} for ERP markers for {subject} {task}")
            epochs_mastoid = epochs.copy().set_eeg_reference(ref_channels=ref_channels)
        elif len(available_ref_channels) == 1:
            print(f"[WARNING] Only {available_ref_channels[0]} mastoid available for {subject} {task}, re-referencing to single mastoid for ERP markers")
            epochs_mastoid = epochs.copy().set_eeg_reference(ref_channels=available_ref_channels)
        else:
            print(f"[WARNING] No mastoid channels (TP9, TP10) available for {subject} {task}, using CAR reference for ERP markers")
            epochs_mastoid = epochs.copy()
        
        # Compute ERP markers (with mastoid reference)
        print(f"Computing ERP markers for {subject} {task}...")
        erp_markers = compute_erp_components(epochs_mastoid, tmin, actual_tmax)
        
        # Combine all markers
        all_markers = {**spectral_markers, **erp_markers}
        
        # Get event information
        event_ids = (epochs.events[:, 2] if hasattr(epochs, 'events') 
                     else [None]*len(epochs))
        event_names = []
        if hasattr(epochs, 'event_id') and hasattr(epochs, 'events'):
            inv_event_id = {v: k for k, v in epochs.event_id.items()}
            for eid in event_ids:
                event_names.append(inv_event_id.get(eid, 'unknown'))
        else:
            event_names = ['unknown']*len(event_ids)
        
        # Convert to long format DataFrame
        for marker_name, marker_data in all_markers.items():
            # Get unit for this marker
            unit = get_marker_unit(marker_name)
            
            # Ensure we don't exceed the marker data dimensions
            n_epochs, n_channels = marker_data.shape
            actual_ch_names = epochs.ch_names[:n_channels]  # Use only the channels that exist in data
            
            for ch_idx, ch_name in enumerate(actual_ch_names):
                for ep_idx in range(min(len(epochs), n_epochs)):  # Don't exceed epochs in marker data
                    event_id = (event_ids[ep_idx] if ep_idx < len(event_ids) 
                                else None)
                    event_name = (event_names[ep_idx] 
                                  if ep_idx < len(event_names) else None)
                    results.append({
                        'marker': marker_name,
                        'channel': ch_name,
                        'value': marker_data[ep_idx, ch_idx],
                        'unit': unit,
                        'subject_id': subject,
                        'session_id': session if session else '',
                        'task': task,
                        'epoch': ep_idx,
                        'event_id': event_id,
                        'event_name': event_name
                    })
        
        # Save to CSV or BIDS-compliant structure
        df = pd.DataFrame(results)
        # Reorder columns to put unit after value
        column_order = ['marker', 'channel', 'value', 'unit', 'subject_id', 
                       'session_id', 'task', 'epoch', 'event_id', 'event_name']
        df = df[column_order]
        
        # Try BIDS-compliant saving first, fallback to CSV
        try:
            if session:
                filepath = save_eeg_features_bids(
                    dataframe=df,
                    subject=subject,
                    session=session,
                    task=task,
                    features_folder=output_dir,
                    desc='mne_markers'
                )
            else:
                # Fallback to regular CSV for non-session data
                outname = f"sub-{subject}_task-{task}_mne_markers.csv"
                filepath = os.path.join(output_dir, outname)
                df.to_csv(filepath, index=False)
        except:
            # Fallback to regular CSV saving
            session_str = f"_ses-{session}" if session else ""
            outname = f"sub-{subject}{session_str}_task-{task}_mne_markers.csv"
            filepath = os.path.join(output_dir, outname)
            df.to_csv(filepath, index=False)
        
        print(f"Saved markers to {filepath}")
        
    except Exception as e:
        session_str = f" {session}" if session else ""
        print(f"Error processing {subject}{session_str} {task}: {e}")


def main():
    """Main function to run the marker computation."""
    parser = argparse.ArgumentParser(
        description='Compute spectral and ERP markers using MNE functions.'
    )
    parser.add_argument('--subject', type=str, default=None, 
                        help='Subject ID (e.g., 02)')
    parser.add_argument('--session', type=str, default=None, 
                        help='Session ID (e.g., a or b) - optional')
    parser.add_argument('--task', type=str, default=None, 
                        help='Task name (e.g., Sart1)')
    parser.add_argument('--output-dir', type=str, 
                        default='./results/mne_markers', 
                        help='Output directory')
    parser.add_argument('--root', type=str, 
                        default='/network/iss/cenir/analyse/meeg/CYBERSART/', 
                        help='Root data directory')
    parser.add_argument('--tmin', type=float, default=0, 
                        help='Start time for marker computation')
    parser.add_argument('--tmax', type=float, default=2, 
                        help='End time for marker computation')
    parser.add_argument('--desc', type=str, default='autoPreproc', 
                        help='Description for the data')
    
    args = parser.parse_args()
    
    # Handle output directory path
    if args.output_dir.startswith('./') or not os.path.isabs(args.output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, args.output_dir.lstrip('./')))
    else:
        output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if args.subject and args.task:
        # Process single subject/task
        process_one_subject(
            args.root, args.subject, args.task, 'eeg', args.desc, 
            output_dir, args.tmin, args.tmax, args.session
        )
    else:
        # Process all subjects and tasks
        subjects = [f"{i:02}" for i in range(2, 43)]
        tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        
        print(f"Processing {len(subjects)} subjects and {len(tasks)} tasks...")
        print(f"Output will be saved to: {output_dir}")
        
        for subject in tqdm(subjects, desc='Subjects'):
            for task in tasks:
                process_one_subject(
                    args.root, subject, task, 'eeg', args.desc, 
                    output_dir, args.tmin, args.tmax, args.session
                )


# -------------------------------------------------------------------------
# Example usage of enhanced ERP computation
# -------------------------------------------------------------------------
def example_enhanced_erp_usage():
    """
    Example demonstrating how to use the enhanced compute_erp_components function
    with peak detection and robust aggregation methods.
    """
    
    # Example peak detection configuration
    peak_windows_config = {
        "p3b": (0.375, 0.600, "pos"),  # P3b positive peak
        "n1":  (0.150, 0.200, "neg"),  # N1 negative peak
        "p2":  (0.200, 0.275, "pos"),  # P2 positive peak
        "n2":  (0.275, 0.375, "neg")   # N2 negative peak (optional)
    }
    
    # Example custom time windows (optional - defaults will be used if None)
    custom_windows = {
        'cnv': (-0.004, 0.596),   # Contingent Negative Variation
        'p1': (0.100, 0.150),     # P1 component
        'n1': (0.150, 0.200),     # N1 component  
        'p2': (0.200, 0.275),     # P2 component
        'p3a': (0.275, 0.375),    # P3a component
        'p3b': (0.375, 0.600),    # P3b component
        'late_window': (0.600, 1.0)  # Custom late window
    }
    
    print("Enhanced ERP computation example:")
    print("=" * 50)
    print("Basic usage (backward compatible):")
    print("erp_markers = compute_erp_components(epochs, tmin=0, tmax=2)")
    print()
    print("Enhanced usage with peak detection:")
    print("erp_markers = compute_erp_components(")
    print("    epochs, ")
    print("    tmin=0, tmax=2,")
    print("    peak_windows=peak_windows_config,")
    print("    baseline=(-0.2, 0.0),")
    print("    agg='trim',  # or 'mean', 'median'")
    print("    trim_prop=0.1,")
    print("    min_prom=2.0  # minimum peak prominence in µV")
    print(")")
    print()
    print("Peak windows configuration:")
    for name, (t0, t1, polarity) in peak_windows_config.items():
        print(f"  {name}: {t0:.3f}-{t1:.3f}s, {polarity} peak")
    print()
    print("This will return amplitude and latency for each peak:")
    for name in peak_windows_config.keys():
        print(f"  - {name}_amp: peak amplitude in µV")
        print(f"  - {name}_lat: peak latency in seconds")
    print()
    print("Aggregation methods:")
    print("  - 'mean': standard mean")
    print("  - 'median': median (robust to outliers)")
    print("  - 'trim': trimmed mean (exclude extreme values)")
    print()
    print("Baseline correction is applied automatically with baseline=(-0.2, 0.0)")
    print("All amplitudes are returned in µV, latencies in seconds")


if __name__ == '__main__':
    main() 