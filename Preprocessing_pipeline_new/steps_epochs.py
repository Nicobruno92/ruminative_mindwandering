"""Epoch construction utilities used by the preprocessing pipeline.

This module isolates the logic that builds:
- Evoked (ERP) epochs around task events
- Pre-probe state epochs (windowed segments before THOUGHT_PROBE)

Keeping epoch creation here improves readability and testability.
"""

from typing import Dict, List, Optional

import mne
import numpy as np


def make_evoked_epochs(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float,
    tmax: float,
    baseline: Optional[tuple],
    reject_by_annotation: bool,
) -> mne.Epochs:
    """Create evoked (ERP) epochs.

    Parameters
    - raw: MNE Raw object (already cleaned, projector applied if desired)
    - events: events array as returned by mne.events_from_annotations
    - event_id: mapping of event labels to numerical codes
    - tmin, tmax: epoch window
    - baseline: baseline tuple or None
    - reject_by_annotation: honor BAD_* annotations

    Returns
    - Epochs instance (preloaded for AutoReject compatibility)
    """
    return mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject_by_annotation=reject_by_annotation,
        event_repeated="drop",
        preload=True,  # Required for AutoReject
        verbose=False,
    )


def make_state_preprobe_epochs(
    raw: mne.io.BaseRaw,
    pre_probe_s: float,
    mini_epoch_s: float,
    overlap_s: float,
) -> Optional[mne.Epochs]:
    """Create pre-probe state epochs anchored to THOUGHT_PROBE events.

    Behavior
    - For each annotation whose description starts with "THOUGHT_PROBE",
      define a 10 s (config-driven via pre_probe_s) big pre-probe window
      [probe - pre_probe_s, probe).
    - Split that big window into overlapping mini-epochs of length
      mini_epoch_s with step (mini_epoch_s - overlap_s).
    - For each mini-epoch, add an annotation to the Raw indicating the
      offset from the THOUGHT_PROBE onset (negative seconds) and retain the
            full probe identification string from the original annotation
            (e.g.,
            "/onoff91/selfother51/valence72/time68/confidence76/average71/probe15").
        - Additionally, infer and append '/ontask' or '/offtask' based on the
            probe's onoff value (>=50 -> ontask, <50 -> offtask).
            If bin tokens for thought dimensions are present on the
            THOUGHT_PROBE annotation, they are preserved; if missing, they are
            computed with a 50 threshold.

    The returned mne.Epochs contains the mini-epochs with a single
    event_id ("state/preprobe": 1). The auxiliary annotations are added
    to the provided Raw in-place for provenance and downstream inspection.
    """

    ann = raw.annotations
    if ann is None or len(ann) == 0:
        return None

    # Identify probes and carry their full identification suffix
    tp_records = []  # (onset, probe_tail)

    def _ensure_bins_and_label(tail: str) -> str:
        """Ensure probe tail includes bin tokens and an ontask/offtask label.

        Parameters
        ----------
        tail : str
            Portion after 'THOUGHT_PROBE/' such as
            'onoff41/selfother38/valence46/time51/confidence62/average44/probe1'

        Returns
        -------
        str
            Tail augmented to include '/onoffBinX', '/selfotherBinX', ...
            tokens if missing, and '/ontask' or '/offtask' inferred from
            onoff value.
        """
        if not isinstance(tail, str) or not tail:
            return tail

        parts = [
            p for p in tail.split('/') if isinstance(p, str) and len(p) > 0
        ]

        # Parse onoff numeric value if present
        onoff_val = None
        dim_vals = {k: None for k in [
            'onoff', 'selfother', 'valence', 'time', 'confidence', 'average']}
        for p in parts:
            for key in dim_vals.keys():
                if p.startswith(key):
                    digits = ''.join(ch for ch in p if ch.isdigit())
                    if digits:
                        try:
                            dim_vals[key] = int(digits)
                        except Exception:
                            dim_vals[key] = None
        onoff_val = dim_vals['onoff']

        # Ensure bin tokens present; threshold at 50
        existing_bin_prefixes = {
            q.split('Bin')[0] + 'Bin' for q in parts if 'Bin' in q
        }
        for key, val in dim_vals.items():
            if val is None:
                continue
            prefix = f"{key}Bin"
            if prefix in existing_bin_prefixes:
                continue
            bin_val = 1 if val >= 50 else 0
            parts.append(f"{prefix}{bin_val}")

        # Append ontask/offtask label if missing and onoff present
        has_label = any(q in ['ontask', 'offtask'] for q in parts)
        if (not has_label) and (onoff_val is not None):
            parts.append('ontask' if onoff_val >= 50 else 'offtask')

        return '/'.join(parts)
    for onset, desc in zip(ann.onset, ann.description):
        if isinstance(desc, str) and desc.startswith("THOUGHT_PROBE"):
            # Keep everything after 'THOUGHT_PROBE' to retain identification
            if "/" in desc:
                probe_tail = _ensure_bins_and_label(desc.split("/", 1)[1])
            else:
                probe_tail = ""
            tp_records.append((float(onset), probe_tail))

    if not tp_records:
        return None

    # Remove previously created state/preprobe annotations to avoid duplication
    try:
        keep_idx = [
            i
            for i, d in enumerate(ann.description)
            if not (isinstance(d, str) and d.startswith("state/preprobe"))
        ]
        if len(keep_idx) != len(ann):
            raw.set_annotations(
                mne.Annotations(
                    onset=[ann.onset[i] for i in keep_idx],
                    duration=[ann.duration[i] for i in keep_idx],
                    description=[ann.description[i] for i in keep_idx],
                    orig_time=ann.orig_time,
                )
            )
            ann = raw.annotations
    except Exception:
        pass

    step = max(0.01, float(mini_epoch_s) - float(overlap_s))

    # Build annotations to append for big window and mini-epochs
    new_onsets: List[float] = []
    new_durs: List[float] = []
    new_desc: List[str] = []

    for tp_onset, probe_tail in tp_records:
        start = max(0.0, tp_onset - float(pre_probe_s))
        stop = tp_onset
        big_dur = max(0.0, stop - start)
        if big_dur <= 0.0:
            continue

        # Big preprobe window annotation
        new_onsets.append(start)
        new_durs.append(big_dur)
        big_label = "state/preprobe_big/" + (
            f"THOUGHT_PROBE/{probe_tail}" if probe_tail else "THOUGHT_PROBE"
        )
        new_desc.append(big_label)

        # Sliding mini-epochs within the big window
        t = start
        while t + float(mini_epoch_s) <= stop + 1e-6:
            # Annotation for this mini-epoch with delta time to probe
            delta_sec = t - tp_onset  # negative value
            win_label = (
                f"state/preprobe_win/dt={delta_sec:.2f}/"
                + (
                    f"THOUGHT_PROBE/{probe_tail}"
                    if probe_tail
                    else "THOUGHT_PROBE"
                )
            )
            new_onsets.append(float(t))
            new_durs.append(float(mini_epoch_s))
            new_desc.append(win_label)

            t += step

    # If we didn't add any mini-window annotations, nothing to epoch
    if not new_onsets:
        return None

    # Append new annotations in a single set_annotations call to keep orig_time
    try:
        ann = raw.annotations
        extra = mne.Annotations(
            onset=new_onsets,
            duration=new_durs,
            description=new_desc,
            orig_time=ann.orig_time if ann is not None else None,
        )
        raw.set_annotations(ann + extra if ann is not None else extra)
    except Exception:
        pass

    # Build events from annotations so each mini-window keeps full description
    events_all, id_map_all = mne.events_from_annotations(raw, event_id='auto')
    # Only keep our mini-window labels
    filtered_map = {
        k: v
        for k, v in id_map_all.items()
        if isinstance(k, str) and k.startswith('state/preprobe_win/')
    }
    if not filtered_map:
        return None
    events, event_id = mne.events_from_annotations(raw, event_id=filtered_map)

    return mne.Epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=float(mini_epoch_s),
        baseline=None,
        reject_by_annotation=True,
        event_repeated="drop",
        preload=True,  # Required for AutoReject
        verbose=False,
    )


