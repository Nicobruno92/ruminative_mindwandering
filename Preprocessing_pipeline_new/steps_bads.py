"""Bad channel detection utilities.

Provides two complementary detectors:
- Flat/saturated channels based on variance and constant segments
- pyprep.NoisyChannels wrapper (RANSAC/channel-wise)
"""

from typing import List, Optional, Dict, Any
import gc

import mne
import numpy as np


def detect_flat_or_saturated_channels(
    raw: mne.io.BaseRaw,
    window_s: float,
    flat_min_var: float,
) -> List[str]:
    """Memory-efficient bad-channel detection.

    Avoid loading the full data matrix into memory by iterating channel-wise.
    """
    picks = mne.pick_types(raw.info, eeg=True)
    ch_names = [raw.ch_names[p] for p in picks]
    sfreq = float(raw.info["sfreq"])
    flat_chs: List[str] = []
    for pick_index, name in zip(picks, ch_names):
        # Load one channel at a time into RAM
        x = raw.get_data(picks=[pick_index], verbose=False)[0]
        if np.nanvar(x) < flat_min_var:
            flat_chs.append(name)
            continue
        diffs = np.diff(x)
        zero_mask = np.isclose(diffs, 0.0)
        if zero_mask.size:
            max_run = 0
            cur = 0
            for is_zero in zero_mask:
                if is_zero:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
            if (max_run / sfreq) >= window_s:
                flat_chs.append(name)
    return flat_chs


def run_pyprep_noisychannels(
    raw: mne.io.BaseRaw,
    do_detrend: bool,
    ransac: bool,
    channel_wise: bool,
    random_state: int,
    prefilter: Optional[Dict[str, Any]] = None,
    pre_ica_opts: Optional[Dict[str, Any]] = None,
) -> List[str]:
    try:
        from pyprep import NoisyChannels
    except Exception:
        return []
    # Optionally run PyPREP on a lightly filtered COPY to avoid over-flagging
    # frontals due to slow drifts and strong blinks. This does NOT modify the
    # original Raw used downstream.
    raw_for_detect = raw

    if prefilter:
        l_freq = prefilter.get("l_freq", None)
        h_freq = prefilter.get("h_freq", None)
        # Only create a copy if any bound is provided
        if l_freq is not None or h_freq is not None:
            raw_for_detect = raw.copy().load_data()
            raw_for_detect.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                phase="zero",
                n_jobs=-1,
            )
    # Optional quick ICA cleaning before detection (on a COPY)

    if pre_ica_opts and bool(pre_ica_opts.get("apply", False)):
        # Absolute import to work when running as a script
        from steps_ica import fit_ica_deterministic

        # Prepare copy for ICA
        raw_for_ica = raw_for_detect.copy().load_data()
        # Light filter for ICA if provided (faster, conservative)
        ica_filt = pre_ica_opts.get("filters", {})
        lf = ica_filt.get("l_freq", None)
        hf = ica_filt.get("h_freq", None)
        if lf is not None or hf is not None:
            raw_for_ica.filter(
                l_freq=lf,
                h_freq=hf,
                phase="zero",
                n_jobs=-1,
            )

        # Fit a small/fast ICA
        method = pre_ica_opts.get("method", "fastica")
        n_comp = pre_ica_opts.get("n_components", 0.95)
        max_iter = int(pre_ica_opts.get("max_iter", 300))
        rs = int(pre_ica_opts.get("random_state", random_state))
        ica = fit_ica_deterministic(
            raw_for_ica,
            method=method,
            n_components=(
                float(n_comp) if isinstance(n_comp, (int, float)) else 0.95
            ),
            max_iter=max_iter,
            random_state=rs,
            report=None,
        )

        # Quick component selection: pattern-based only (no ICLabel)
        eog_chs = pre_ica_opts.get("eog_channels", None)
        if eog_chs is not None:
            # Only keep those that exist
            eog_chs = [c for c in eog_chs if c in raw_for_ica.ch_names]
            if not eog_chs:
                eog_chs = None
        eog_comps, _ = ica.find_bads_eog(inst=raw_for_ica, ch_name=eog_chs)
        thr = float(pre_ica_opts.get("muscle_threshold", 0.7))
        muscle_comps, _ = ica.find_bads_muscle(raw_for_ica, threshold=thr)
        to_exclude = list(set(eog_comps + muscle_comps))
        if to_exclude:
            ica.exclude = to_exclude
            ica.apply(raw_for_ica)

        # Use the ICA-cleaned copy for PyPREP detection
        raw_for_detect = raw_for_ica
        # Free ICA object memory
        del ica
        gc.collect()

    nd = NoisyChannels(
        raw_for_detect,
        do_detrend=do_detrend,
        random_state=random_state,
    )
    nd.find_all_bads(ransac=ransac, channel_wise=channel_wise)
    bads = nd.get_bads() or []

    # Ensure temporary copies are eligible for GC
    if raw_for_detect is not raw:
        try:
            del raw_for_detect
        except Exception:
            pass
        gc.collect()

    return bads


