#!/usr/bin/env python3
"""
View Preprocessed EEG Data - All Pipeline Stages
=================================================
Comprehensive viewer for all preprocessing stages from raw to final epochs.

Supported file types from preprocessing pipeline:
1. RAW FILES:
   - BrainVision raw: sub-{ID}_task-{session}_eeg.vhdr (BIDS/raw)
   - ICA-cleaned raw: sub-{ID}_task-{session}_desc-icaClean_eeg.fif (derivatives)
   - CSD-transformed: sub-{ID}_task-{session}_desc-csd_eeg.fif (derivatives)

2. ICA FILES:
   - ICA decomposition: sub-{ID}_task-{session}_desc-ica_ica.fif (derivatives)

3. EPOCH FILES:
   - Evoked epochs: sub-{ID}_task-{session}_desc-evoked_epo.fif (derivatives)
   - State epochs: sub-{ID}_task-{session}_desc-state_epo.fif (derivatives)

Usage:
------
Modify the configuration variables below and run the script.
Set FILE_TYPE to filter by stage: "raw", "ica", "epochs", or None (all)
Set EPOCH_TYPE to filter epochs: "evoked", "state", or None (all)
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# DATA_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/raw"  # derivatives for preprocessed
DATA_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # raw for BrainVision files
SUBJECT_ID = "03"  # Participant ID (e.g., "02", "03", etc.)
SART_SESSION = "Sart1"  # SART session: "Sart1", "Sart2", "Sart3", or "Sart4"
FILE_INDEX = None  # Integer index if multiple files found; None picks first
PLOT_KIND = "auto"  # 'auto', 'epochs', 'raw', 'ica', 'evoked'
PICKS = None  # e.g., ['eeg'] or list of channel names, or None
BLOCK = True  # Whether plots should block execution
VERBOSE = True  # Print extra information

# Filters
FILE_TYPE = "epochs"  # "raw", "ica", "epochs", or None (all types)
EPOCH_TYPE = "evoked"  # "evoked", "state", or None (all epochs)
# =============================================================================

import os
import sys
import glob
from typing import List, Optional, Tuple, Union

import mne


def find_mne_files(
    path: str,
    subject_id: str,
    sart_session: str,
    epoch_type: Optional[str] = None,
    file_type: Optional[str] = None,
) -> List[str]:
    """
    Find MNE files for a subject/session across all preprocessing stages.

    Parameters
    ----------
    path : str
        Base directory to search
    subject_id : str
        Subject ID (e.g., "03")
    sart_session : str
        Task/session name (e.g., "Sart1")
    epoch_type : str, optional
        Filter epochs: "evoked", "state", or None
    file_type : str, optional
        Filter by stage: "raw", "ica", "epochs", or None

    Returns
    -------
    List[str]
        Sorted list of matching file paths
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    files: List[str] = []

    # EPOCHS: desc-evoked_epo.fif and desc-state_epo.fif
    if file_type is None or file_type == "epochs":
        epoch_patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-evoked_epo.fif",
            f"**/sub-{subject_id}_task-{sart_session}_desc-state_epo.fif",
            f"**/sub-{subject_id}_task-{sart_session}_desc-*_epo.fif",
        ]
        for pat in epoch_patterns:
            files.extend(glob.glob(os.path.join(path, pat), recursive=True))

    # RAW: desc-icaClean_eeg.fif, desc-csd_eeg.fif, and BrainVision
    if file_type is None or file_type == "raw":
        raw_patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-icaClean_eeg.fif",
            f"**/sub-{subject_id}_task-{sart_session}_desc-csd_eeg.fif",
            f"**/sub-{subject_id}_task-{sart_session}_desc-*_eeg.fif",
            f"**/sub-{subject_id}_task-{sart_session}_eeg.vhdr",  # BrainVision
        ]
        for pat in raw_patterns:
            files.extend(glob.glob(os.path.join(path, pat), recursive=True))

    # ICA: desc-ica_ica.fif
    if file_type is None or file_type == "ica":
        ica_patterns = [
            f"**/sub-{subject_id}_task-{sart_session}_desc-ica_ica.fif",
            f"**/sub-{subject_id}_task-{sart_session}_*_ica.fif",
        ]
        for pat in ica_patterns:
            files.extend(glob.glob(os.path.join(path, pat), recursive=True))

    # Deduplicate
    seen = set()
    unique_files: List[str] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    # Filter by epoch type (only for epoch files)
    if epoch_type:
        unique_files = [
            f
            for f in unique_files
            if "_epo.fif" not in f or f"desc-{epoch_type}_epo.fif" in f
        ]

    return sorted(unique_files)


def classify_filetype(filepath: str) -> str:
    """Classify file type from filename."""
    name = os.path.basename(filepath).lower()
    if name.endswith(".vhdr"):
        return "brainvision"
    if "_ica.fif" in name:
        return "ica"
    if "_epo.fif" in name:
        return "epochs"
    if "_eeg.fif" in name:
        return "raw"
    return "raw"


def load_mne_object(
    filepath: str, kind: Optional[str] = None
) -> Tuple[
    str,
    Union[
        mne.Epochs,
        mne.io.BaseRaw,
        mne.evoked.Evoked,
        List[mne.evoked.Evoked],
        mne.preprocessing.ICA,
    ],
]:
    """Load MNE object from file."""
    if filepath.lower().endswith(".vhdr"):
        obj = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        return "brainvision", obj

    load_kind = kind or classify_filetype(filepath)

    if load_kind == "ica":
        obj = mne.preprocessing.read_ica(filepath, verbose=False)
        return "ica", obj
    if load_kind == "epochs":
        obj = mne.read_epochs(filepath, preload=True, verbose=False)
        return "epochs", obj
    if load_kind == "raw":
        obj = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        return "raw", obj
    if load_kind == "evoked":
        evokeds = mne.read_evokeds(filepath, verbose=False)
        return "evoked", evokeds

    raise ValueError(f"Unsupported kind: {load_kind}")


def print_info_summary(
    kind: str,
    obj: Union[
        mne.Epochs,
        mne.io.BaseRaw,
        mne.evoked.Evoked,
        List[mne.evoked.Evoked],
        mne.preprocessing.ICA,
    ],
):
    """Print summary of loaded object."""

    def _summarize_info(info: mne.Info) -> None:
        ch_types = mne.channel_indices_by_type(info)
        sfreq = info["sfreq"]
        n_ch = info["nchan"]
        ch_type_counts = {k: len(v) for k, v in ch_types.items()}
        ch_type_str = ", ".join([f"{k}: {v}" for k, v in ch_type_counts.items()])
        print(f"- Channels: {n_ch} ({ch_type_str}) | sfreq: {sfreq:.2f} Hz")
        bads = info.get("bads", [])
        if bads:
            print(f"- Bad channels: {bads}")

    print(f"Loaded kind: {kind}")
    if kind == "ica":
        ica: mne.preprocessing.ICA = obj  # type: ignore
        print(f"- ICA components: {ica.n_components_}")
        print(f"- Method: {ica.method}")
        if hasattr(ica, "exclude") and ica.exclude:
            print(f"- Excluded components: {ica.exclude}")
        if hasattr(ica, "info"):
            _summarize_info(ica.info)
    elif kind == "epochs":
        epochs: mne.Epochs = obj  # type: ignore
        print(f"- Epochs: {len(epochs)} | tmin={epochs.tmin:.3f}s | tmax={epochs.tmax:.3f}s")
        _summarize_info(epochs.info)
        if epochs.event_id:
            print(f"- Event IDs: {epochs.event_id}")
    elif kind == "raw" or kind == "brainvision":
        raw: mne.io.BaseRaw = obj  # type: ignore
        print(f"- Raw duration: {raw.n_times / raw.info['sfreq']:.2f} s")
        _summarize_info(raw.info)
    elif kind == "evoked":
        if isinstance(obj, list):
            print(f"- Evoked list with {len(obj)} conditions: {[e.comment for e in obj]}")
            if len(obj) > 0:
                _summarize_info(obj[0].info)
        else:
            evk: mne.evoked.Evoked = obj  # type: ignore
            print(f"- Evoked: {evk.comment if evk.comment else 'Unnamed'}")
            _summarize_info(evk.info)


def plot_object(
    kind: str,
    obj: Union[
        mne.Epochs,
        mne.io.BaseRaw,
        mne.evoked.Evoked,
        List[mne.evoked.Evoked],
        mne.preprocessing.ICA,
    ],
    picks=None,
    block: bool = True,
) -> None:
    """Plot MNE object."""
    if kind == "ica":
        ica: mne.preprocessing.ICA = obj  # type: ignore
        print("\nNote: ICA plotting requires raw data. Showing component topographies only.")
        ica.plot_components(picks=None)
    elif kind == "epochs":
        epochs: mne.Epochs = obj  # type: ignore
        epochs.plot(picks=picks, block=block)
    elif kind == "raw" or kind == "brainvision":
        raw: mne.io.BaseRaw = obj  # type: ignore
        raw.plot(picks=picks, block=block)
    elif kind == "evoked":
        if isinstance(obj, list):
            if len(obj) == 0:
                print("No evoked objects to plot.")
                return
            obj[0].plot(picks=picks, selectable=True, time_unit="s")
        else:
            evk: mne.evoked.Evoked = obj  # type: ignore
            evk.plot(picks=picks, selectable=True, time_unit="s")
    else:
        raise ValueError(f"Unsupported kind for plotting: {kind}")


def main() -> None:
    """Main function."""
    if VERBOSE:
        print(f"Searching for subject {SUBJECT_ID}, session {SART_SESSION}")
        print(f"Search path: {DATA_PATH}")
        if FILE_TYPE:
            print(f"Filtering for file type: {FILE_TYPE}")
        if EPOCH_TYPE:
            print(f"Filtering for epoch type: {EPOCH_TYPE}")

    files = find_mne_files(
        DATA_PATH, SUBJECT_ID, SART_SESSION, epoch_type=EPOCH_TYPE, file_type=FILE_TYPE
    )

    if not files:
        print(f"\nNo files found for subject {SUBJECT_ID}, session {SART_SESSION}.")
        print("\nTroubleshooting:")
        print("1. Check DATA_PATH points to correct location:")
        print("   - derivatives: for preprocessed files (icaClean, epochs, ICA)")
        print("   - raw: for original BrainVision files")
        print("2. Verify subject/session exist in the dataset")
        print("3. Try FILE_TYPE=None to search all file types")
        print(f"\nCurrent settings:")
        print(f"  DATA_PATH: {DATA_PATH}")
        print(f"  FILE_TYPE: {FILE_TYPE}")
        print(f"  EPOCH_TYPE: {EPOCH_TYPE}")
        sys.exit(1)

    if VERBOSE:
        print(f"\nFound {len(files)} file(s):")
        for idx, f in enumerate(files):
            ftype = classify_filetype(f)
            fname = os.path.basename(f)
            print(f"  [{idx}] {ftype:12s} {fname}")

    # Select file
    if FILE_INDEX is not None:
        if not (0 <= FILE_INDEX < len(files)):
            raise IndexError(f"FILE_INDEX {FILE_INDEX} out of range [0, {len(files)-1}]")
        filepath = files[FILE_INDEX]
    else:
        filepath = files[0]

    kind = None if PLOT_KIND == "auto" else PLOT_KIND

    if VERBOSE:
        print(f"\nLoading: {os.path.basename(filepath)}")

    kind, obj = load_mne_object(filepath, kind=kind)
    print_info_summary(kind, obj)

    if VERBOSE:
        print("\nOpening plot window...")
    plot_object(kind, obj, picks=PICKS, block=BLOCK)


if __name__ == "__main__":
    main()
