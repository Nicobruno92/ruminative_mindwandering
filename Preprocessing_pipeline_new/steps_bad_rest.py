"""
BAD_rest segment handling for EEG preprocessing.

This module provides functions to drop BAD_rest annotated segments from Raw data.
BAD_rest segments are periods outside task blocks (between BLOCK_START/BLOCK_END)
that were annotated during data harmonization.

Dropping these segments early prevents rest periods from contaminating:
- ASR artifact subspace reconstruction
- PyPREP bad channel detection
- ICA fitting
"""

from typing import List, Tuple

import mne


def drop_bad_rest_segments(
    raw: mne.io.BaseRaw,
    bad_rest_label: str = "BAD_rest",
    verbose: bool = True
) -> mne.io.BaseRaw:
    """
    Drop segments annotated as BAD_rest from the Raw object.
    
    This function creates a copy of the Raw data with BAD_rest segments
    removed. Use this early in preprocessing to prevent rest periods from
    contaminating ASR, PyPREP bad channel detection, or ICA fitting.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Raw object with BAD_rest annotations.
    bad_rest_label : str
        Label identifying bad rest annotations (default: "BAD_rest").
    verbose : bool
        Print summary of dropped segments.
        
    Returns
    -------
    mne.io.BaseRaw
        Raw object with BAD_rest segments removed.
        
    Notes
    -----
    This function uses MNE's crop functionality to extract only the good
    segments. The returned Raw object will have discontinuous time points
    but annotations are preserved and adjusted.
    """
    annotations = raw.annotations
    descriptions = list(annotations.description)
    onsets = list(annotations.onset)
    durations = list(annotations.duration)
    
    # Find BAD_rest segments
    bad_segments: List[Tuple[float, float]] = []
    for i, desc in enumerate(descriptions):
        if desc == bad_rest_label:
            start = float(onsets[i])
            end = start + float(durations[i])
            bad_segments.append((start, end))
    
    if not bad_segments:
        if verbose:
            print("[DROP_BAD_REST] No BAD_rest segments found")
        return raw
    
    # Sort by start time
    bad_segments.sort(key=lambda x: x[0])
    
    # Calculate good segments (inverse of bad segments)
    recording_start = 0.0
    recording_end = raw.times[-1]
    
    good_segments: List[Tuple[float, float]] = []
    current_pos = recording_start
    
    for bad_start, bad_end in bad_segments:
        if bad_start > current_pos:
            good_segments.append((current_pos, bad_start))
        current_pos = max(current_pos, bad_end)
    
    if current_pos < recording_end:
        good_segments.append((current_pos, recording_end))
    
    if not good_segments:
        if verbose:
            print("[DROP_BAD_REST] Warning: No good segments remain after dropping BAD_rest")
        return raw
    
    # Concatenate good segments
    raw_segments = []
    for start, end in good_segments:
        segment = raw.copy().crop(tmin=start, tmax=end)
        raw_segments.append(segment)
    
    if len(raw_segments) == 1:
        raw_clean = raw_segments[0]
    else:
        raw_clean = mne.concatenate_raws(raw_segments)
    
    total_dropped = sum(end - start for start, end in bad_segments)
    total_kept = sum(end - start for start, end in good_segments)
    
    if verbose:
        print(f"[DROP_BAD_REST] Dropped {len(bad_segments)} BAD_rest segments")
        print(f"[DROP_BAD_REST] Total dropped: {total_dropped:.1f}s")
        print(f"[DROP_BAD_REST] Total kept: {total_kept:.1f}s")
        print(f"[DROP_BAD_REST] Original duration: {recording_end:.1f}s")
    
    return raw_clean
