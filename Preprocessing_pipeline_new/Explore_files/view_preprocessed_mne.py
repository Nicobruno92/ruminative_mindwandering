#!/usr/bin/env python3

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# DATA_PATH = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # Base directory to search (use derivatives, not raw!)
DATA_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/derivatives"  # Base directory to search (use derivatives, not raw!)
# DATA_PATH = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/raw"  # For raw BrainVision files
SUBJECT_ID = "05"  # Participant ID (e.g., "02", "03", etc.)
SART_SESSION = "Sart1"  # SART session: "Sart1", "Sart2", "Sart3", or "Sart4"
FILE_INDEX = None  # Integer index if multiple files found for this subject/session; None picks the first suitable file
PLOT_KIND = "auto"  # options: 'auto', 'epochs', 'raw', 'evoked', 'ica'
PICKS = None  # e.g., ['eeg'] or list of channel names, or None
BLOCK = True  # Whether plots should block execution
VERBOSE = True  # Print extra information
EPOCH_TYPE = None  # "evoked" or "state" or None (auto-select first). Set to "evoked" or "state" to pick specific epoch type
FILE_TYPE = None  # "raw", "epochs", "ica", or None (auto-detect all). Filter by preprocessing stage
# =============================================================================

import os
import sys
import glob
from typing import List, Optional, Tuple, Union

import mne


def find_mne_files(path: str, subject_id: str, sart_session: str, epoch_type: Optional[str] = None) -> List[str]:
    """
    Find candidate MNE or BrainVision files for a specific subject and SART session.

    Returns a sorted list of candidate files matching the subject and session.
    
    Parameters
    ----------
    path : str
        Base directory to search
    subject_id : str
        Subject ID (e.g., "03")
    sart_session : str
        Task/session name (e.g., "Sart1")
    epoch_type : str, optional
        Filter by epoch type: "evoked", "state", or None (all)
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Path not found: {path}")

    # Patterns for FIF files (epochs, raw, evoked)
    # Prioritize BIDS-compliant naming from preprocessing pipeline
    fif_patterns = [
        # BIDS-style epochs from preprocessing (exact naming)
        f"**/sub-{subject_id}_task-{sart_session}_desc-*_epo.fif",
        # Generic patterns for backwards compatibility
        f"**/sub-{subject_id}_task-{sart_session}_*-epo.fif",
        f"**/sub-{subject_id}_task-{sart_session}_*epo*.fif",
        f"**/sub-{subject_id}_{sart_session}_*-epo.fif",
        f"**/sub-{subject_id}_{sart_session}_*epo*.fif",
        f"**/*sub-{subject_id}*{sart_session}*-epo.fif",
        f"**/*sub-{subject_id}*{sart_session}*epo*.fif",
        # Raw data patterns
        f"**/sub-{subject_id}_task-{sart_session}_*-raw.fif",
        f"**/sub-{subject_id}_task-{sart_session}_*raw*.fif",
        f"**/sub-{subject_id}_{sart_session}_*-raw.fif",
        f"**/sub-{subject_id}_{sart_session}_*raw*.fif",
        f"**/*sub-{subject_id}*{sart_session}*-raw.fif",
        f"**/*sub-{subject_id}*{sart_session}*raw*.fif",
        # Evoked patterns
        f"**/sub-{subject_id}_task-{sart_session}_*-ave.fif",
        f"**/sub-{subject_id}_task-{sart_session}_*evok*.fif",
        f"**/sub-{subject_id}_{sart_session}_*-ave.fif",
        f"**/sub-{subject_id}_{sart_session}_*evok*.fif",
        f"**/*sub-{subject_id}*{sart_session}*-ave.fif",
        f"**/*sub-{subject_id}*{sart_session}*evok*.fif",
    ]

    files: List[str] = []
    for pat in fif_patterns:
        files.extend(glob.glob(os.path.join(path, pat), recursive=True))

    # If no FIF files found, look for BrainVision files (vhdr is the header, but need all three for reading)
    if not files:
        # BrainVision BIDS: sub-08_task-Sart1_eeg.vhdr (and .eeg, .vmrk)
        bv_pattern = f"**/sub-{subject_id}_task-{sart_session}_eeg.vhdr"
        files = glob.glob(os.path.join(path, bv_pattern), recursive=True)

    # Deduplicate while preserving order
    seen = set()
    unique_files: List[str] = []
    for f in files:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)

    # Filter by epoch type if specified
    if epoch_type:
        unique_files = [
            f for f in unique_files 
            if f"desc-{epoch_type}_epo.fif" in f.lower() or epoch_type in os.path.basename(f).lower()
        ]
    
    return sorted(unique_files)


def classify_filetype(filepath: str) -> str:
    """
    Classify a file as 'epochs', 'raw', 'evoked', or 'brainvision' based on filename/extension.
    """
    name = os.path.basename(filepath).lower()
    if name.endswith('.vhdr'):
        return 'brainvision'
    if "-epo" in name or "epo" in name:
        return "epochs"
    if "-raw" in name or "raw" in name:
        return "raw"
    if "-ave" in name or "evok" in name or "evoked" in name:
        return "evoked"
    return "raw"


def load_mne_object(filepath: str, kind: Optional[str] = None) -> Tuple[str, Union[mne.Epochs, mne.io.BaseRaw, mne.evoked.Evoked, List[mne.evoked.Evoked]]]:
    """
    Load an MNE object from a FIF or BrainVision file.
    """
    # Always use BrainVision loader for .vhdr files
    if filepath.lower().endswith('.vhdr'):
        obj = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        return "brainvision", obj

    load_kind = kind or classify_filetype(filepath)

    if load_kind == "epochs":
        obj = mne.read_epochs(filepath, preload=True, verbose=False)
        return "epochs", obj
    if load_kind == "raw":
        obj = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        return "raw", obj
    if load_kind == "evoked":
        evokeds = mne.read_evokeds(filepath, verbose=False)
        if isinstance(evokeds, list):
            return "evoked", evokeds
        return "evoked", evokeds

    raise ValueError(f"Unsupported kind: {load_kind}")


def print_info_summary(kind: str, obj: Union[mne.Epochs, mne.io.BaseRaw, mne.evoked.Evoked, List[mne.evoked.Evoked]]):
    """
    Print a concise summary of the loaded MNE object's info.

    Parameters
    ----------
    kind : str
        One of 'epochs', 'raw', or 'evoked'.
    obj : Union[mne.Epochs, mne.io.BaseRaw, mne.evoked.Evoked, List[mne.evoked.Evoked]]
        Loaded object.
    """
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
    if kind == "epochs":
        epochs: mne.Epochs = obj  # type: ignore[assignment]
        print(f"- Epochs: {len(epochs)} | tmin={epochs.tmin:.3f}s | tmax={epochs.tmax:.3f}s")
        _summarize_info(epochs.info)
        if epochs.event_id:
            print(f"- Event IDs: {epochs.event_id}")
    elif kind == "raw" or kind == "brainvision":
        raw: mne.io.BaseRaw = obj  # type: ignore[assignment]
        print(f"- Raw duration: {raw.n_times / raw.info['sfreq']:.2f} s")
        _summarize_info(raw.info)
    elif kind == "evoked":
        if isinstance(obj, list):
            print(f"- Evoked list with {len(obj)} conditions: {[e.comment for e in obj]}")
            if len(obj) > 0:
                _summarize_info(obj[0].info)
        else:
            evk: mne.evoked.Evoked = obj  # type: ignore[assignment]
            print(f"- Evoked: {evk.comment if evk.comment else 'Unnamed'}")
            _summarize_info(evk.info)


def plot_object(kind: str, obj: Union[mne.Epochs, mne.io.BaseRaw, mne.evoked.Evoked, List[mne.evoked.Evoked]], picks=None, block: bool = True) -> None:
    """
    Plot the loaded MNE object.

    Parameters
    ----------
    kind : str
        One of 'epochs', 'raw', or 'evoked'.
    obj : Union[mne.Epochs, mne.io.BaseRaw, mne.evoked.Evoked, List[mne.evoked.Evoked]]
        Loaded object.
    picks : Any
        Channel picks passed to MNE plotting functions.
    block : bool
        Whether plots should block execution.
    """
    if kind == "epochs":
        epochs: mne.Epochs = obj  # type: ignore[assignment]

        epochs.plot(picks=picks, block=block)
    elif kind == "raw" or kind == "brainvision":
        raw: mne.io.BaseRaw = obj  # type: ignore[assignment]
        raw.plot(picks=picks, block=block)
    elif kind == "evoked":
        if isinstance(obj, list):
            # Plot the first condition by default
            if len(obj) == 0:
                print("No evoked objects to plot.")
                return
            obj[0].plot(picks=picks, selectable=True, time_unit="s")
        else:
            evk: mne.evoked.Evoked = obj  # type: ignore[assignment]
            evk.plot(picks=picks, selectable=True, time_unit="s")
    else:
        raise ValueError(f"Unsupported kind for plotting: {kind}")


def choose_file(files: List[str], file_index: Optional[int]) -> str:
    """
    Choose a file from a list, honoring an optional index.

    Parameters
    ----------
    files : List[str]
        Candidate files.
    file_index : Optional[int]
        If provided, the file at this index is chosen. Otherwise, the first is used.

    Returns
    -------
    str
        Selected file path.
    """
    if not files:
        raise FileNotFoundError("No MNE-compatible FIF files found under DATA_PATH.")

    if file_index is not None:
        if not (0 <= file_index < len(files)):
            raise IndexError(f"FILE_INDEX {file_index} out of range [0, {len(files)-1}]")
        return files[file_index]

    return files[0]


def main() -> None:
    """
    Discover, load, summarize, and plot MNE data for a specific participant and SART session.

    Notes
    -----
    - Searches recursively under `DATA_PATH` for files matching `SUBJECT_ID` and `SART_SESSION`.
    - The preprocessing pipeline generates two types of epochs:
      * evoked epochs (desc-evoked)
      * state epochs (desc-state)
    - Set `EPOCH_TYPE` to 'evoked' or 'state' to select a specific epoch type.
    - Preference is given to `*-epo.fif` for epochs viewing.
    - Set `PLOT_KIND` to force a loader, or leave as 'auto' to infer from filename.
    - Use `PICKS` to limit channels (e.g., ['eeg']).
    - Plot windows may block execution depending on `BLOCK`.
    """
    if VERBOSE:
        print(f"Searching for subject {SUBJECT_ID}, session {SART_SESSION} under: {DATA_PATH}")
        if EPOCH_TYPE:
            print(f"Filtering for epoch type: {EPOCH_TYPE}")
    
    files = find_mne_files(DATA_PATH, SUBJECT_ID, SART_SESSION, epoch_type=EPOCH_TYPE)

    if not files:
        print(f"No files found for subject {SUBJECT_ID}, session {SART_SESSION}.")
        print("Available subjects from config: 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43")
        print("Available sessions: Sart1, Sart2, Sart3, Sart4")
        print(f"\nTip: Set EPOCH_TYPE to 'evoked' or 'state' to filter by epoch type")
        print(f"Current search path: {DATA_PATH}")
        sys.exit(1)

    if VERBOSE:
        print(f"Found {len(files)} file(s) for subject {SUBJECT_ID}, session {SART_SESSION}:")
        for idx, f in enumerate(files):
            print(f"  [{idx}] {f}")
        
        # If multiple epochs found, suggest filtering
        epoch_files = [f for f in files if "-epo.fif" in f]
        if len(epoch_files) > 1:
            print(f"\nMultiple epoch files found ({len(epoch_files)}).")
            print("Tip: Set EPOCH_TYPE to 'evoked' or 'state' to view a specific type.")

    filepath = choose_file(files, FILE_INDEX)
    kind = None if PLOT_KIND == "auto" else PLOT_KIND

    if VERBOSE:
        print(f"\nLoading: {filepath}")

    kind, obj = load_mne_object(filepath, kind=kind)

    print_info_summary(kind, obj)

    if VERBOSE:
        print("\nOpening plot window...")
    plot_object(kind, obj, picks=PICKS, block=BLOCK)


if __name__ == "__main__":
    main()
