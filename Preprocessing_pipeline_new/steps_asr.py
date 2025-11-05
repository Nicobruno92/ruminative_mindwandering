"""Artifact Subspace Reconstruction (ASR) utilities.

Provides optional ASR artifact removal using meegkit.asr.ASR before PyPREP
bad channel detection. This helps reduce artifactual variance in the data.
"""

from typing import Optional, Tuple
import gc

import numpy as np
import mne


def apply_asr_if_configured(
    raw: mne.io.BaseRaw,
    asr_config: Optional[dict],
    sfreq: float,
) -> Tuple[mne.io.BaseRaw, bool]:
    """Apply ASR to Raw if configured.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data to process.
    asr_config : dict or None
        ASR configuration from YAML. Should contain:
        - apply: bool, whether to apply ASR
        - cutoff: float, standard deviation cutoff for ASR
        - blocksize: int, block size in samples for ASR
        - win_len: float, window length in seconds for calibration
        - win_overlap: float, window overlap fraction
        - max_dropout_fraction: float, maximum fraction of channels to drop
        - method: str, ASR method ('euclid' or 'riemann')
    sfreq : float
        Sampling frequency (used for blocksize calculation if needed).

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Cleaned raw data (original if ASR not applied).
    asr_applied : bool
        Whether ASR was successfully applied.

    Notes
    -----
    If meegkit is not available, returns original raw without error.
    ASR operates on a copy of the raw data to preserve original.
    """
    # Check if ASR should be applied
    if asr_config is None or not asr_config.get("apply", False):
        return raw, False

    # Try importing meegkit
    try:
        from meegkit.asr import ASR
    except ImportError:
        print("Warning: meegkit not available, skipping ASR step")
        return raw, False

    # Extract ASR parameters from config
    cutoff = float(asr_config.get("cutoff", 5.0))
    blocksize = asr_config.get("blocksize", None)
    win_len = float(asr_config.get("win_len", 0.5))
    win_overlap = float(asr_config.get("win_overlap", 0.66))
    max_dropout_fraction = float(asr_config.get("max_dropout_fraction", 0.1))
    method = asr_config.get("method", "euclid")

    # Calculate blocksize if not provided (default to 0.5s worth of samples)
    if blocksize is None:
        blocksize = int(0.5 * sfreq)

    print(
        f"Applying ASR (cutoff={cutoff}, blocksize={blocksize}, "
        f"method={method})..."
    )

    # Work on a copy to preserve original
    raw_asr = raw.copy()
    raw_asr.load_data()

    # Get EEG data
    # CRITICAL: Use reject_by_annotation=None to preserve full data length
    # This ensures 1:1 correspondence with raw._data shape
    # Exclude reference and auxiliary channels (consistent with pipeline)
    ref_like = ["REF", "EXG1", "EXG2", "HEOG", "VEOG", "M1", "M2", "GND"]
    picks_eeg = mne.pick_types(raw_asr.info, eeg=True, exclude=ref_like)
    
    # Check if we have valid EEG channels
    if len(picks_eeg) == 0:
        print(f"Warning: No valid EEG channels found for ASR (excluded: {ref_like})")
        print("Skipping ASR application.")
        return raw, False
    
    data = raw_asr.get_data(
        picks=picks_eeg,
        reject_by_annotation=None
    )  # Shape: (n_channels, n_times) - ASR expects this format!

    # Additional check: ensure we have actual data
    if data.size == 0:
        print("Warning: No valid data for ASR after channel selection")
        print("Skipping ASR application.")
        return raw, False

    # Initialize and fit ASR
    asr = ASR(
        sfreq=sfreq,
        cutoff=cutoff,
        blocksize=blocksize,
        win_len=win_len,
        win_overlap=win_overlap,
        max_dropout_fraction=max_dropout_fraction,
        method=method,
    )

    # Fit on clean data (first part of recording assumed cleaner)
    # ASR expects shape (n_channels, n_samples), data is (n_channels, n_times)
    # Use first 60 seconds or 30% of data, whichever is smaller
    # Increased from 30s/20% to provide more training data for better calibration
    n_samples = data.shape[1]  # data is (n_channels, n_times), so samples are on axis 1
    train_samples = min(int(60 * sfreq), int(0.3 * n_samples))
    
    # Ensure we have enough training samples (ASR needs at least 1 sample)
    if train_samples == 0:
        train_samples = min(500, n_samples)  # Use at least 500 samples or all if less
    
    if train_samples == 0 or n_samples == 0:
        print("Warning: Insufficient training data for ASR")
        print("Skipping ASR application.")
        return raw, False
    
    # Ensure minimum training window (at least 10 seconds of data)
    min_train_samples = int(10 * sfreq)
    if train_samples < min_train_samples:
        if n_samples >= min_train_samples:
            train_samples = min_train_samples
        else:
            print(f"Warning: Dataset too short for ASR calibration (need {min_train_samples}, have {n_samples})")
            print("Skipping ASR application.")
            return raw, False
    
    # Extract training data - data is (n_channels, n_samples), slice along time axis (axis 1)
    train_data = data[:, :train_samples]
    print(f"Training ASR on {train_samples} samples (out of {n_samples} total)")
    print(f"Data shape: {data.shape}, Training slice shape: {train_data.shape}")
    
    # Final validation: check training data is not empty
    if train_data.shape[0] == 0 or train_data.shape[1] == 0:
        print(f"Warning: Training data has invalid shape {train_data.shape}")
        print("Skipping ASR application.")
        return raw, False
    
    # Check if training data has any NaNs or Infs
    if np.any(np.isnan(train_data)) or np.any(np.isinf(train_data)):
        print("Warning: Training data contains NaN or Inf values")
        print("Skipping ASR application.")
        return raw, False
    
    # Additional debugging info
    print(f"Training data stats: min={np.nanmin(train_data):.4f}, max={np.nanmax(train_data):.4f}, "
          f"mean={np.nanmean(train_data):.4f}, std={np.nanstd(train_data):.4f}")
    
    # Wrap fit in try-except to catch any remaining ASR errors
    try:
        asr.fit(train_data)
        print("ASR fit completed successfully")
    except (IndexError, ValueError, RuntimeError) as e:
        print(f"Warning: ASR fit failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Skipping ASR application.")
        return raw, False

    # Transform entire dataset
    try:
        data_clean = asr.transform(data)  # data is already in correct shape (n_channels, n_samples)
    except (IndexError, ValueError, RuntimeError) as e:
        print(f"Warning: ASR transform failed with error: {e}")
        print("Skipping ASR application.")
        return raw, False

    # Update raw - data_clean is already in shape (n_channels, n_samples)
    raw_asr._data[picks_eeg, :] = data_clean

    # Clean up
    del data, data_clean, train_data
    gc.collect()

    print("ASR applied successfully")
    return raw_asr, True
