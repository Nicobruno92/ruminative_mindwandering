"""Artifact Subspace Reconstruction (ASR) utilities.

Provides optional ASR artifact removal using meegkit.asr.ASR before PyPREP
bad channel detection. This helps reduce artifactual variance in the data.
"""

from typing import Optional, Tuple
import gc

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
    picks_eeg = mne.pick_types(raw_asr.info, eeg=True, exclude=[])
    data = raw_asr.get_data(
        picks=picks_eeg,
        reject_by_annotation=None
    )  # Shape: (n_channels, n_times)

    # Transpose for ASR (expects n_times x n_channels)
    data_t = data.T

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
    # Use first 30 seconds or 20% of data, whichever is smaller
    train_samples = min(int(30 * sfreq), int(0.2 * data_t.shape[0]))
    asr.fit(data_t[:train_samples, :])

    # Transform entire dataset
    data_clean = asr.transform(data_t)

    # Transpose back and update raw
    data_clean_t = data_clean.T
    raw_asr._data[picks_eeg, :] = data_clean_t

    # Clean up
    del data, data_t, data_clean, data_clean_t
    gc.collect()

    print("ASR applied successfully")
    return raw_asr, True
