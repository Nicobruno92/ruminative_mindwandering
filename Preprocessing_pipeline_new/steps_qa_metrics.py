"""
Quality Assurance metrics computation for EEG preprocessing pipeline.

This module handles all QA-related calculations, logging, and reporting
that were previously embedded in the main preprocessing pipeline.
"""
import os
import tempfile
import fcntl
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mne

from utils.log_preprocessing import LogPreprocessingDetails


def compute_band_power_db(
    raw: mne.io.BaseRaw,
    freq_low: float,
    freq_high: float,
    picks: Optional[Union[List[int], np.ndarray, List[str]]] = None,
) -> float:
    """
    Compute average band power in decibels for a specific frequency band.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    freq_low : float
        Lower frequency bound of the band in Hz
    freq_high : float
        Higher frequency bound of the band in Hz
    picks : list of int | list of str | np.ndarray | None, optional
        Channel selection to use when computing PSD. If None, EEG channels
        (excluding bads) are used.
        
    Returns
    -------
    float
        Average band power across selected channels in decibels
    """
    # Determine channel picks
    if picks is None:
        # Default to EEG channels excluding bads
        picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    
    # Guard against insufficient data length for PSD computation
    n_samples = raw.n_times
    n_fft = 2048
    if n_samples < n_fft:
        # Not enough samples for reliable PSD - return NaN
        return float('nan')
    
    # Compute PSD using the modern Raw.compute_psd method
    spectrum = raw.compute_psd(
        method='welch',
        fmin=freq_low,
        fmax=freq_high,
        tmin=None,
        tmax=None,
        n_fft=2048,
        n_overlap=512,
        picks=picks,
        verbose=False
    )
    
    # Get PSDs and frequencies. Use picks='all' to avoid MNE default
    # behavior ("data_or_ica") which can exclude non-EEG channels
    # like EOG in Spectrum objects with restricted info.
    psds = spectrum.get_data(picks='all')  # Shape: (n_channels, n_freqs)
    freqs = spectrum.freqs
    
    # Find frequency indices for the band
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    
    # Compute band power (sum over frequencies, mean over channels)
    band_power = np.mean(np.sum(psds[:, freq_mask], axis=1))
    
    # Convert to decibels (10 * log10(power))
    # Add small epsilon to avoid log(0)
    band_power_db = 10 * np.log10(band_power + 1e-12)
    
    return float(band_power_db)


def compute_evoked_epoch_metrics(
    epochs: "mne.Epochs",
    ar_epochs: "mne.Epochs",
    ar_reject_epochs_idx: List[int],
    epochs_state: Optional["mne.Epochs"] = None,
    state_rejected_by_ar: Optional[int] = None,
    state_input_total: Optional[int] = None,
    state_after_construct: Optional[int] = None,
    state_dropped_post_ar: Optional[int] = None,
    epochs_sleep: Optional["mne.Epochs"] = None,
    sleep_input_total: Optional[int] = None,
    sleep_after_construct: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive epoch metrics for QA assessment.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Original epochs before AutoReject
    ar_epochs : mne.Epochs
        Epochs after AutoReject and drop_bad
    ar_reject_epochs_idx : List[int]
        Indices of epochs rejected by AutoReject
    epochs_state : mne.Epochs, optional
        State epochs if available
    state_rejected_by_ar : int, optional
        Number of state epochs rejected by AutoReject
    state_input_total : int, optional
        Total state epochs input
    state_after_construct : int, optional
        State epochs after construction
    state_dropped_post_ar : int, optional
        State epochs dropped after AutoReject
    epochs_sleep : mne.Epochs, optional
        Sleep epochs if available
    sleep_input_total : int, optional
        Total sleep epochs input
    sleep_after_construct : int, optional
        Sleep epochs after construction
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive epoch metrics
    """
    # Track epoch counts - BAD_rest is not a "rejection"
    n_epochs_input_total = len(getattr(epochs, "drop_log", []))
    n_epochs_after_construct = int(len(epochs.events))
    
    # BAD_rest exclusions are not "epoch rejections" - they prevent creation
    # Only count true epoch rejections (non-annotation, non-boundary)
    drop_log = getattr(epochs, "drop_log", [])
    annotation_exclusions = {'IGNORED', 'EDGE'}
    n_epochs_dropped_pre_ar = sum(
        1 for log in drop_log
        if len(log) > 0 and not any(
            str(reason).upper().startswith('BAD') or
            str(reason) in annotation_exclusions
            for reason in log
        )
    )
    
    # Calculate AutoReject counts
    n_rejected_by_ar = len(ar_reject_epochs_idx) if ar_reject_epochs_idx else 0
    n_evoked_epochs = len(ar_epochs)
    
    # Calculate rejection percentage
    evoked_rejection_pct = (
        n_rejected_by_ar / max(n_epochs_after_construct or 1, 1)
        if n_epochs_after_construct else 0.0
    )
    
    # Calculate annotation drop ratio
    annot_drop_ratio = (
        float(n_epochs_dropped_pre_ar) / float(n_epochs_input_total)
        if (n_epochs_input_total and n_epochs_dropped_pre_ar is not None and
            n_epochs_input_total > 0)
        else None
    )
    
    # Base evoked metrics
    epoch_metrics = {
        'evoked_rejection_pct': evoked_rejection_pct,
        'n_epochs_input_total': n_epochs_input_total,
        'n_epochs_after_construct': n_epochs_after_construct,
        'n_epochs_dropped_pre_ar': n_epochs_dropped_pre_ar,
        'n_rejected_by_ar': n_rejected_by_ar,
        'n_evoked_epochs': n_evoked_epochs,
        'annot_drop_ratio': annot_drop_ratio,
        'has_state_epochs': epochs_state is not None,
    }
    
    # Add state epoch metrics if available
    if epochs_state is not None:
        epoch_metrics.update({
            'state_input_total': state_input_total,
            'state_after_construct': state_after_construct,
            'state_rejected_by_ar': state_rejected_by_ar or 0,
            'n_state_epochs': len(epochs_state),
            'state_dropped_post_ar': state_dropped_post_ar or 0,
        })
    
    # Add sleep epoch metrics if available
    if epochs_sleep is not None:
        epoch_metrics.update({
            'has_sleep_epochs': True,
            'sleep_input_total': sleep_input_total,
            'sleep_after_construct': sleep_after_construct,
            'n_sleep_epochs': len(epochs_sleep),
        })
    
    return epoch_metrics


def compute_epoch_counts_around_autoreject(
    epochs_before: "mne.Epochs",
    epochs_after_ar: "mne.Epochs",
    epochs_final: "mne.Epochs"
) -> Tuple[int, int, int]:
    """
    Compute epoch counts before/after AutoReject and after drop_bad.
    
    Parameters
    ----------
    epochs_before : mne.Epochs
        Epochs before AutoReject
    epochs_after_ar : mne.Epochs
        Epochs after AutoReject but before drop_bad
    epochs_final : mne.Epochs
        Final epochs after drop_bad
        
    Returns
    -------
    Tuple[int, int, int]
        (n_before_ar, n_after_ar, n_dropped_post_ar)
    """
    n_before_ar = int(len(epochs_before.events))
    pre_len_after_ar = int(len(epochs_after_ar.events))
    n_after_drop_bad = int(len(epochs_final))
    n_dropped_post_ar = max(0, pre_len_after_ar - n_after_drop_bad)
    
    return n_before_ar, pre_len_after_ar, n_dropped_post_ar


def prepare_epoch_metrics_for_reports(
    epoch_metrics: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Prepare epoch metrics for HTML report display.
    
    Parameters
    ----------
    epoch_metrics : Dict[str, Any]
        Raw epoch metrics from compute_evoked_epoch_metrics
        
    Returns
    -------
    Tuple[Dict[str, Any], Optional[Dict[str, Any]]]
        (evoked_metrics, state_metrics) formatted for reports
    """
    # Prepare evoked metrics
    evoked_rejection_pct = epoch_metrics.get('evoked_rejection_pct', 0.0) * 100.0
    thr_bad = epoch_metrics.get('qa_thr_bad_epochs_ratio', None)
    evoked_flag = None
    if isinstance(thr_bad, (int, float)):
        try:
            evoked_flag = (evoked_rejection_pct / 100.0) > float(thr_bad)
        except Exception:
            evoked_flag = None
    evoked_metrics = {
        'n_after_construct': epoch_metrics.get('n_epochs_after_construct', 0),
        'n_rejected_by_ar': epoch_metrics.get('n_rejected_by_ar', 0),
        'rejection_pct': evoked_rejection_pct,
        'flag_too_many_bad_epochs': evoked_flag,
    }
    
    # Prepare state metrics (if available)
    state_metrics = None
    if epoch_metrics.get('has_state_epochs', False):
        state_rejection_pct = 0.0
        if (epoch_metrics.get('state_rejected_by_ar') and
                epoch_metrics.get('state_after_construct')):
            state_rejection_pct = (
                epoch_metrics['state_rejected_by_ar'] /
                max(epoch_metrics['state_after_construct'] or 1, 1)
            )
        state_flag = None
        if isinstance(thr_bad, (int, float)):
            try:
                state_flag = state_rejection_pct > float(thr_bad)
            except Exception:
                state_flag = None
        state_metrics = {
            'n_after_construct': epoch_metrics.get('state_after_construct', 0),
            'n_rejected_by_ar': epoch_metrics.get('state_rejected_by_ar', 0),
            'rejection_pct': state_rejection_pct * 100.0,  # Convert to %
            'flag_too_many_bad_epochs': state_flag,
        }
    
    return evoked_metrics, state_metrics


def compute_qa_metrics(
    raw_before_ica: mne.io.Raw,
    raw_after_ica: mne.io.Raw,
    ica_exclude: List[int],
    ica_n_components: int,
    bads_pre_interp: List[str],
    cfg: Dict[str, Any],
    ica_selection_log: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute all QA metrics for preprocessing assessment.
    
    Parameters
    ----------
    raw_before_ica : mne.io.Raw
        Raw data before ICA application
    raw_after_ica : mne.io.Raw
        Raw data after ICA application
    ica_exclude : List[int]
        List of excluded ICA component indices
    ica_n_components : int
        Total number of ICA components
    bads_pre_interp : List[str]
        List of bad channels before interpolation
    cfg : Dict[str, Any]
        Configuration dictionary
    ica_selection_log : Dict[str, Any], optional
        Detailed log from variance-weighted ICA selection
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all QA metrics
    """
    # Get QA band definitions from config
    qa_line = cfg["qa_bands"]["line"]
    qa_emg = cfg["qa_bands"]["emg"]
    qa_blink_low = cfg["qa_bands"]["blink_low"]
    
    # Band power calculations pre-ICA
    line_db_pre = compute_band_power_db(raw_before_ica, qa_line[0], qa_line[1])
    emg_db_pre = compute_band_power_db(raw_before_ica, qa_emg[0], qa_emg[1])
    
    # Band power calculations post-ICA
    line_db_post = compute_band_power_db(raw_after_ica, qa_line[0], qa_line[1])
    emg_db_post = compute_band_power_db(raw_after_ica, qa_emg[0], qa_emg[1])
    
    # Calculate improvements
    delta_50_db = line_db_pre - line_db_post
    delta_emg_db = emg_db_pre - emg_db_post
    
    # Blink residual metric: frontal 1–4 Hz pre vs post
    frontal_picks = mne.pick_channels_regexp(
        raw_after_ica.info, regexp=r"^(Fp|AF|F).*"
    )
    blink_pre_db = compute_band_power_db(
        raw_before_ica, qa_blink_low[0], qa_blink_low[1],
        picks=frontal_picks if len(frontal_picks) else None
    )
    blink_post_db = compute_band_power_db(
        raw_after_ica, qa_blink_low[0], qa_blink_low[1],
        picks=frontal_picks if len(frontal_picks) else None
    )
    delta_blink_db = blink_pre_db - blink_post_db
    
    # ICA exclusion ratio
    ica_excl_ratio = len(ica_exclude) / float(ica_n_components)
    
    # Variance metrics from ICA selection log (if available)
    ica_excluded_variance_ratio = None
    ica_excluded_variance_pct = None
    variance_budget_exceeded = None
    component_count_exceeded = None
    if ica_selection_log:
        ica_excluded_variance_ratio = ica_selection_log.get(
            'final_excluded_variance_ratio'
        )
        ica_excluded_variance_pct = ica_selection_log.get(
            'final_excluded_variance_pct'
        )
        variance_budget_exceeded = ica_selection_log.get(
            'variance_budget_exceeded'
        )
        component_count_exceeded = ica_selection_log.get(
            'component_count_exceeded'
        )
        excluded_components_summary = ica_selection_log.get(
            'excluded_components_summary', []
        )
    
    # EOG channel analysis
    eog_picks = mne.pick_types(raw_after_ica.info, eog=True)
    eog_db_1_4 = None
    if len(eog_picks):
        eog_db_1_4 = compute_band_power_db(
            raw_after_ica, qa_blink_low[0], qa_blink_low[1], picks=eog_picks
        )
    
    metrics = {
        'line_db_pre': float(line_db_pre),
        'line_db_post': float(line_db_post),
        'emg_db_pre': float(emg_db_pre),
        'emg_db_post': float(emg_db_post),
        'blink_pre_db': float(blink_pre_db),
        'blink_post_db': float(blink_post_db),
        'delta_50_db': float(delta_50_db),
        'delta_emg_db': float(delta_emg_db),
        'delta_blink_db': float(delta_blink_db),
        'ica_excl_ratio': float(ica_excl_ratio),
        'eog_db_1_4': float(eog_db_1_4) if eog_db_1_4 is not None else None,
        'n_bad_channels': len(bads_pre_interp),
        'bad_channels_list': bads_pre_interp,
        'n_ica_excluded': len(ica_exclude),
        'ica_excluded_indices': ica_exclude,
        'excluded_components_summary': excluded_components_summary if ica_selection_log else []
    }
    
    # Add variance metrics if available
    if ica_excluded_variance_ratio is not None:
        metrics['ica_excluded_variance_ratio'] = float(
            ica_excluded_variance_ratio
        )
    if ica_excluded_variance_pct is not None:
        metrics['ica_excluded_variance_pct'] = float(
            ica_excluded_variance_pct
        )
    # Ensure variance budget exceeded flag is present and consistent with config
    if variance_budget_exceeded is not None:
        metrics['variance_budget_exceeded'] = bool(variance_budget_exceeded)
    else:
        try:
            thr = float(cfg["ica_selection"]["max_excluded_variance_ratio"])  # absolute of total
            v_pct = metrics.get('ica_excluded_variance_pct')
            if v_pct is not None:
                metrics['variance_budget_exceeded'] = (float(v_pct) / 100.0) > thr
        except Exception:
            pass
    if component_count_exceeded is not None:
        metrics['component_count_exceeded'] = bool(component_count_exceeded)
    
    return metrics


def assess_qa_flags(
    qa_metrics: Dict[str, Any],
    epoch_metrics: Dict[str, Any],
    raw_after_ica: mne.io.Raw,
    ica_exclude: List[int],
    cfg: Dict[str, Any]
) -> Tuple[List[str], bool, List[str]]:
    """
    Assess QA flags and determine pass/fail status.
    
    Parameters
    ----------
    qa_metrics : Dict[str, Any]
        QA metrics from compute_qa_metrics
    epoch_metrics : Dict[str, Any]
        Epoch-related metrics (rejection rates, etc.)
    raw_after_ica : mne.io.Raw
        Raw data after processing
    ica_exclude : List[int]
        Excluded ICA components
    cfg : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Tuple[List[str], bool, List[str]]
        (flags, passed, fail_reasons)
    """
    flags = []
    
    # Get thresholds from config
    max_bad_channels_ratio = float(
        cfg["qa_thresholds"]["max_bad_channels_ratio"]
    )
    thr_bad_epochs = float(cfg["qa_thresholds"]["max_bad_epochs_ratio"])
    thr_min_line_db = float(
        cfg["qa_thresholds"]["min_line_noise_db_improvement"]
    )
    
    # Check bad channels ratio
    eeg_picks = mne.pick_types(raw_after_ica.info, eeg=True)
    bad_channels_ratio = qa_metrics['n_bad_channels'] / len(eeg_picks)
    if bad_channels_ratio > max_bad_channels_ratio:
        flags.append("max_bad_channels_ratio")
    
    # Check epoch rejection rate
    pct_epochs_rejected = epoch_metrics.get('evoked_rejection_pct', 0.0)
    if pct_epochs_rejected > thr_bad_epochs:
        flags.append("max_bad_epochs_ratio")

    # Check state epoch rejection rate independently (if available)
    try:
        if epoch_metrics.get('has_state_epochs', False):
            state_after = epoch_metrics.get('state_after_construct') or 0
            state_rej = epoch_metrics.get('state_rejected_by_ar') or 0
            state_rate = (state_rej / float(max(state_after, 1))) if state_after else 0.0
            if state_rate > thr_bad_epochs:
                flags.append("max_bad_epochs_ratio_state")
    except Exception:
        pass
    
    # Check annotation/bounds drop ratio
    annot_drop_ratio = epoch_metrics.get('annot_drop_ratio')
    try:
        thr_annot_drop = float(cfg["qa_thresholds"]["max_annot_dropped_epochs_ratio"])
        if annot_drop_ratio is not None and annot_drop_ratio > thr_annot_drop:
            flags.append("max_annot_dropped_epochs_ratio")
    except KeyError:
        pass
    
    # Check variance budget exceeded (from ICA selection log)
    if qa_metrics.get('variance_budget_exceeded', False):
        flags.append("variance_budget_exceeded")
    
    # Check line noise improvement
    if qa_metrics['delta_50_db'] < thr_min_line_db:
        flags.append("min_line_noise_db_improvement")
    
    # Check ocular component detection
    eog_present = any(
        ch.upper().startswith("VEOG") or ch.upper().startswith("HEOG") or ch.upper().startswith("EOG") 
        for ch in raw_after_ica.ch_names
    )
    if eog_present and not ica_exclude:  # Simplified check - in real case you'd check if ocular ICs were detected
        flags.append("ocular_ic_required")
    
    # Pass/Fail decision
    pass_cfg = cfg.get("pass_criteria", {}) or {}
    fail_on_flags = list(pass_cfg.get("fail_on_flags", [
        "max_bad_channels_ratio",
        "max_bad_epochs_ratio",
        "variance_budget_exceeded",
        "min_line_noise_db_improvement",
        "ocular_ic_required",
    ]))
    require_all_absent = bool(pass_cfg.get("require_all_flags_absent", False))
    fail_reasons = sorted([f for f in flags if f in fail_on_flags])
    
    if require_all_absent:
        passed = len(flags) == 0
    else:
        passed = len(fail_reasons) == 0
    
    return flags, passed, fail_reasons


def log_qa_details(
    logger: LogPreprocessingDetails,
    qa_metrics: Dict[str, Any],
    epoch_metrics: Dict[str, Any],
    flags: List[str],
    passed: bool,
    fail_reasons: List[str],
    prelim_bads: List[str] = None,
    prep_bads: List[str] = None
) -> None:
    """
    Log all QA details to the preprocessing logger.
    
    Parameters
    ----------
    logger : LogPreprocessingDetails
        Preprocessing logger instance
    qa_metrics : Dict[str, Any]
        QA metrics
    epoch_metrics : Dict[str, Any]
        Epoch metrics
    flags : List[str]
        QA flags
    passed : bool
        Overall pass status
    fail_reasons : List[str]
        Reasons for failure
    """
    logger.log_detail("prelim_bad_channels", prelim_bads or [])
    logger.log_detail("pyprep_bad_channels", prep_bads or [])
    logger.log_detail("final_bad_channels", qa_metrics['bad_channels_list'])
    logger.log_detail("ica_excluded", qa_metrics['ica_excluded_indices'])
    logger.log_detail("ica_excluded_ratio", qa_metrics['ica_excl_ratio'])
    logger.log_detail("delta_50hz_db", qa_metrics['delta_50_db'])
    logger.log_detail("delta_emg_30_80_db", qa_metrics['delta_emg_db'])
    logger.log_detail("delta_blink_1_4hz_frontal_db", qa_metrics['delta_blink_db'])
    logger.log_detail("pct_epochs_rejected", epoch_metrics.get('evoked_rejection_pct', 0.0))
    logger.log_detail("annot_drop_ratio", epoch_metrics.get('annot_drop_ratio'))
    logger.log_detail("n_epochs_input_total", epoch_metrics.get('n_epochs_input_total'))
    logger.log_detail("n_epochs_after_construct", epoch_metrics.get('n_epochs_after_construct'))
    logger.log_detail("n_epochs_dropped_pre_ar", epoch_metrics.get('n_epochs_dropped_pre_ar'))
    logger.log_detail("flags", flags)
    logger.log_detail("frontal_1_4hz_db", qa_metrics['blink_post_db'])
    logger.log_detail("eog_1_4hz_db", qa_metrics['eog_db_1_4'])
    logger.log_detail("passed", passed)
    logger.log_detail("fail_reasons", fail_reasons)
    logger.save_preprocessing_details()


def save_qa_csv(
    derivatives_root: str,
    subject: str,
    task: str,
    qa_metrics: Dict[str, Any],
    epoch_metrics: Dict[str, Any],
    flags: List[str],
    passed: bool,
    fail_reasons: List[str],
    session: str = "default"
) -> None:
    """
    Save QA metrics to CSV files (per-subject and summary).
    
    Parameters
    ----------
    derivatives_root : str
        Path to derivatives directory
    subject : str
        Subject identifier
    task : str
        Task identifier
    qa_metrics : Dict[str, Any]
        QA metrics
    epoch_metrics : Dict[str, Any]
        Epoch metrics
    flags : List[str]
        QA flags
    passed : bool
        Overall pass status
    fail_reasons : List[str]
        Reasons for failure
    session : str
        Session identifier (default: "default")
    """
    def _safe_int(val, default=0):
        try:
            return int(val)
        except Exception:
            return int(default)

    def _safe_float(val, default=0.0):
        try:
            return float(val)
        except Exception:
            return float(default)

    # Common QA metrics
    common_metrics = {
        "subject": subject,
        "session": session,
        "task": task,
        "delta_50hz_db": qa_metrics['delta_50_db'],
        "delta_emg_30_80_db": qa_metrics['delta_emg_db'],
        "delta_blink_1_4hz_frontal_db": qa_metrics['delta_blink_db'],
        "ica_excluded_ratio": qa_metrics['ica_excl_ratio'],
        "ica_excluded_variance_pct": qa_metrics.get('ica_excluded_variance_pct'),
        "n_bad_channels": qa_metrics['n_bad_channels'],
        "bad_channels": ",".join(qa_metrics['bad_channels_list']) if qa_metrics['bad_channels_list'] else "",
        "n_ica_excluded": qa_metrics['n_ica_excluded'],
        "ica_excluded_indices": ",".join(map(str, qa_metrics['ica_excluded_indices'])) if qa_metrics['ica_excluded_indices'] else "",
        "too_many_bad_channels": bool("max_bad_channels_ratio" in flags),
        "variance_budget_exceeded": bool(qa_metrics.get('variance_budget_exceeded', False)),
        "excluded_components_summary": ",".join(qa_metrics.get('excluded_components_summary') or []),
        "too_many_ica_excluded": bool("max_ica_excluded_ratio" in flags),
        "low_line_noise_suppression": bool("min_line_noise_db_improvement" in flags),
        "no_ocular_ic_detected_with_eog": bool("ocular_ic_required" in flags),
        "passed": bool(passed),
        "fail_reasons": ",".join(fail_reasons),
    }

    # Create separate QA entries for evoked and state epochs
    qa_entries = []
    
    # EVOKED EPOCHS ENTRY
    evoked_qa = {
        **common_metrics,
        "epoch_type": "evoked",
        "too_many_bad_epochs": bool("max_bad_epochs_ratio" in flags),
        "n_triggers": _safe_int(epoch_metrics.get('n_epochs_input_total', 0)),
        "n_epochs_after_construct": _safe_int(epoch_metrics.get('n_epochs_after_construct', 0)),
        "n_rejected_by_ar": _safe_int(epoch_metrics.get('n_rejected_by_ar', 0)),
        "n_epochs": _safe_int(epoch_metrics.get('n_evoked_epochs', 0)),
        "n_epochs_dropped_pre_ar": _safe_int(epoch_metrics.get('n_epochs_dropped_pre_ar', 0)),
        "n_dropped_post_ar": _safe_int(epoch_metrics.get('n_dropped_post_ar', 0)),
        "annot_drop_ratio": _safe_float(epoch_metrics.get('annot_drop_ratio', 0.0)),
        "used_for_qa": True,
    }
    qa_entries.append(evoked_qa)

    # STATE EPOCHS ENTRY (if available)
    if epoch_metrics.get('has_state_epochs', False):
        # Compute state-specific rejection flag (independent from global flags)
        try:
            state_after_construct = _safe_int(epoch_metrics.get('state_after_construct', 0))
            state_rejected_by_ar = _safe_int(epoch_metrics.get('state_rejected_by_ar', 0))
            state_rejection_ratio = (
                (state_rejected_by_ar / float(max(state_after_construct, 1)))
                if state_after_construct else 0.0
            )
            thr_bad_epochs = float(epoch_metrics.get('qa_thr_bad_epochs_ratio', 1.0))
            state_too_many_bad_epochs = state_rejection_ratio > thr_bad_epochs
        except Exception:
            state_too_many_bad_epochs = False

        state_qa = {
            **common_metrics,
            "epoch_type": "state",
            "too_many_bad_epochs": bool(state_too_many_bad_epochs),
            "n_triggers": _safe_int(epoch_metrics.get('state_input_total', 0)),
            "n_epochs_after_construct": _safe_int(epoch_metrics.get('state_after_construct', 0)),
            "n_rejected_by_ar": _safe_int(epoch_metrics.get('state_rejected_by_ar', 0)),
            "n_epochs": _safe_int(epoch_metrics.get('n_state_epochs', 0)),
            "n_epochs_dropped_pre_ar": 0,
            "n_dropped_post_ar": _safe_int(epoch_metrics.get('state_dropped_post_ar', 0)),
            "annot_drop_ratio": 0.0,
            "used_for_qa": False,
        }
        qa_entries.append(state_qa)

    # SLEEP EPOCHS ENTRY (if available)
    if epoch_metrics.get('has_sleep_epochs', False):
        # Sleep epochs don't go through AutoReject, so no rejection metrics
        sleep_qa = {
            **common_metrics,
            "epoch_type": "sleep",
            "too_many_bad_epochs": False,  # No AutoReject for sleep
            "n_triggers": _safe_int(epoch_metrics.get('sleep_input_total', 0)),
            "n_epochs_after_construct": _safe_int(epoch_metrics.get('sleep_after_construct', 0)),
            "n_rejected_by_ar": 0,  # No AutoReject for sleep
            "n_epochs": _safe_int(epoch_metrics.get('n_sleep_epochs', 0)),
            "n_epochs_dropped_pre_ar": 0,
            "n_dropped_post_ar": 0,
            "annot_drop_ratio": 0.0,
            "used_for_qa": False,
        }
        qa_entries.append(sleep_qa)

    # Save per-subject CSV
    try:
        if session and session != "default":
            qa_dir = os.path.join(derivatives_root, f"sub-{subject}", f"ses-{session}")
            qa_csv_name = f"sub-{subject}_ses-{session}_task-{task}_qa_metrics.csv"
        else:
            qa_dir = os.path.join(derivatives_root, f"sub-{subject}")
            qa_csv_name = f"sub-{subject}_task-{task}_qa_metrics.csv"
        os.makedirs(qa_dir, exist_ok=True)
        qa_csv = os.path.join(qa_dir, qa_csv_name)

        col_order = [
            'subject', 'session', 'task', 'epoch_type', 'passed', 'fail_reasons',
            'delta_50hz_db', 'delta_emg_30_80_db', 'delta_blink_1_4hz_frontal_db',
            'ica_excluded_ratio', 'ica_excluded_variance_pct', 'n_bad_channels', 'bad_channels', 'n_ica_excluded',
            'ica_excluded_indices', 'excluded_components_summary',
            'n_triggers', 'n_epochs_after_construct', 'n_rejected_by_ar', 'n_epochs',
            'n_epochs_dropped_pre_ar', 'n_dropped_post_ar', 'annot_drop_ratio', 'used_for_qa',
            'too_many_bad_channels', 'variance_budget_exceeded', 'too_many_ica_excluded', 'low_line_noise_suppression',
            'no_ocular_ic_detected_with_eog', 'too_many_bad_epochs'
        ]
        df_qa = pd.DataFrame(qa_entries)
        col_order = [c for c in col_order if c in df_qa.columns]
        df_qa = df_qa[col_order]
        df_qa.to_csv(qa_csv, index=False)
    except Exception:
        pass

    # Update cumulative summary CSV
    try:
        summary_path = os.path.join(derivatives_root, "qa_summary.csv")
        lock_path = summary_path + ".lock"

        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        # Acquire exclusive lock
        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            
            # Read existing summary if present
            if os.path.exists(summary_path):
                try:
                    df_sum = pd.read_csv(summary_path)
                except Exception:
                    df_sum = pd.DataFrame()
            else:
                df_sum = pd.DataFrame()

            # Update existing entries or add new ones
            for row in qa_entries:
                if not df_sum.empty and {"subject", "session", "task", "epoch_type"}.issubset(df_sum.columns):
                    mask = (
                        (df_sum["subject"].astype(str) == str(subject)) & 
                        (df_sum["session"].astype(str) == str(session)) &
                        (df_sum["task"].astype(str) == str(task)) &
                        (df_sum["epoch_type"].astype(str) == str(row["epoch_type"]))
                    )
                    if mask.any():
                        for k, v in row.items():
                            df_sum.loc[mask, k] = v
                    else:
                        df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
                else:
                    df_sum = pd.DataFrame([row]) if df_sum.empty else pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)

            # Write atomically
            with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(summary_path), suffix=".tmp") as tf:
                df_sum.to_csv(tf.name, index=False)
                tmp_name = tf.name
            os.replace(tmp_name, summary_path)
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            except Exception:
                pass
    except Exception:
        pass


def run_complete_qa_assessment(
    raw_before_ica: mne.io.Raw,
    raw_after_ica: mne.io.Raw,
    ica_exclude: List[int],
    ica_n_components: int,
    bads_pre_interp: List[str],
    epoch_metrics: Dict[str, Any],
    derivatives_root: str,
    subject: str,
    task: str,
    cfg: Dict[str, Any],
    logger: Optional[LogPreprocessingDetails] = None,
    prelim_bads: List[str] = None,
    prep_bads: List[str] = None,
    ica_selection_log: Optional[Dict[str, Any]] = None,
    session: str = "default"
) -> Dict[str, Any]:
    """
    Run complete QA assessment pipeline.
    
    Parameters
    ----------
    raw_before_ica : mne.io.Raw
        Raw data before ICA
    raw_after_ica : mne.io.Raw
        Raw data after ICA
    ica_exclude : List[int]
        Excluded ICA components
    ica_n_components : int
        Total ICA components
    bads_pre_interp : List[str]
        Bad channels before interpolation
    epoch_metrics : Dict[str, Any]
        Epoch-related metrics
    derivatives_root : str
        Derivatives directory path
    subject : str
        Subject identifier
    task : str
        Task identifier
    cfg : Dict[str, Any]
        Configuration dictionary
    logger : Optional[LogPreprocessingDetails]
        Logger instance
    prelim_bads : List[str], optional
        Preliminary bad channels
    prep_bads : List[str], optional
        PyPREP detected bad channels
    ica_selection_log : Dict[str, Any], optional
        Detailed log from variance-weighted ICA selection
        
    Returns
    -------
    Dict[str, Any]
        QA assessment results
    """
    # Compute QA metrics (include ICA selection log for variance metrics)
    qa_metrics = compute_qa_metrics(
        raw_before_ica,
        raw_after_ica,
        ica_exclude,
        ica_n_components,
        bads_pre_interp,
        cfg,
        ica_selection_log
    )
    
    # Assess flags and pass/fail
    flags, passed, fail_reasons = assess_qa_flags(
        qa_metrics, epoch_metrics, raw_after_ica, ica_exclude, cfg
    )
    
    # Log details if logger provided
    if logger is not None:
        log_qa_details(
            logger, qa_metrics, epoch_metrics, flags, passed, fail_reasons,
            prelim_bads, prep_bads
        )
    
    # Save CSV files
    save_qa_csv(
        derivatives_root, subject, task, qa_metrics, epoch_metrics, flags, passed, fail_reasons,
        session=session
    )
    
    return {
        'qa_metrics': qa_metrics,
        'flags': flags,
        'passed': passed,
        'fail_reasons': fail_reasons
    }


def generate_preprocessing_qa_html_report(
    derivatives_root: str,
    output_filename: str = "preprocessing_qa_summary.html",
    subject_filter: str = None
) -> str:
    """
    Generate a visual HTML summary report from the QA summary CSV.
    
    Parameters
    ----------
    derivatives_root : str
        Path to derivatives directory containing qa_summary.csv
    output_filename : str
        Name of the output HTML file
    subject_filter : str, optional
        If provided, only include data for this subject
        
    Returns
    -------
    str
        Path to the generated HTML report
    """
    from datetime import datetime
    
    summary_csv = os.path.join(derivatives_root, "qa_summary.csv")
    
    # Determine output path
    if subject_filter:
        subject_dir = os.path.join(derivatives_root, f"sub-{subject_filter}")
        os.makedirs(subject_dir, exist_ok=True)
        output_path = os.path.join(subject_dir, output_filename)
    else:
        output_path = os.path.join(derivatives_root, output_filename)
    
    # Load data
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
    else:
        df = pd.DataFrame()
    
    # Filter by subject if requested
    if subject_filter and not df.empty and 'subject' in df.columns:
        df = df[df['subject'].astype(str) == str(subject_filter)]
    
    # Compute summary statistics
    if not df.empty:
        total = len(df)
        # Group by subject-session-task for unique recordings
        unique_recordings = df.groupby(['subject', 'session', 'task']).first().reset_index()
        n_recordings = len(unique_recordings)
        n_passed = unique_recordings['passed'].sum() if 'passed' in unique_recordings.columns else 0
        n_failed = n_recordings - n_passed
        pass_rate = (n_passed / n_recordings * 100) if n_recordings > 0 else 0
        
        # Count by epoch type
        evoked_df = df[df['epoch_type'] == 'evoked'] if 'epoch_type' in df.columns else df
        state_df = df[df['epoch_type'] == 'state'] if 'epoch_type' in df.columns else pd.DataFrame()
        sleep_df = df[df['epoch_type'] == 'sleep'] if 'epoch_type' in df.columns else pd.DataFrame()
        
        # Flag counts
        flag_cols = [
            'too_many_bad_channels', 'variance_budget_exceeded', 
            'too_many_bad_epochs', 'low_line_noise_suppression',
            'no_ocular_ic_detected_with_eog'
        ]
        flag_counts = {}
        for col in flag_cols:
            if col in df.columns:
                flag_counts[col] = df[col].sum()
    else:
        n_recordings = 0
        n_passed = 0
        n_failed = 0
        pass_rate = 0
        evoked_df = pd.DataFrame()
        state_df = pd.DataFrame()
        sleep_df = pd.DataFrame()
        flag_counts = {}
    
    # Build HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>EEG Preprocessing QA Summary</title>",
        "<style>",
        "body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }",
        ".container { max-width: 1400px; margin: 0 auto; }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        ".summary-cards { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }",
        ".card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); min-width: 150px; text-align: center; }",
        ".card-value { font-size: 36px; font-weight: bold; }",
        ".card-label { color: #7f8c8d; font-size: 14px; margin-top: 5px; }",
        ".pass { color: #27ae60; }",
        ".fail { color: #e74c3c; }",
        ".warn { color: #f39c12; }",
        ".neutral { color: #3498db; }",
        ".progress-bar { background: #ecf0f1; border-radius: 10px; height: 30px; overflow: hidden; margin: 10px 0; }",
        ".progress-fill { height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }",
        ".progress-pass { background: linear-gradient(90deg, #27ae60, #2ecc71); }",
        ".progress-fail { background: linear-gradient(90deg, #e74c3c, #c0392b); }",
        ".flags-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }",
        ".flag-item { background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #e74c3c; }",
        ".flag-item.ok { border-left-color: #27ae60; }",
        ".flag-count { font-size: 24px; font-weight: bold; }",
        ".flag-name { color: #7f8c8d; font-size: 12px; }",
        "table { width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 20px 0; }",
        "th { background: #3498db; color: white; padding: 12px 8px; text-align: left; font-size: 12px; }",
        "td { padding: 10px 8px; border-bottom: 1px solid #ecf0f1; font-size: 12px; }",
        "tr:hover { background: #f8f9fa; }",
        ".status-pass { background: #d5f4e6; color: #27ae60; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".status-fail { background: #fde8e8; color: #e74c3c; padding: 4px 8px; border-radius: 4px; font-weight: bold; }",
        ".epoch-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }",
        ".epoch-evoked { background: #e8f4fd; color: #3498db; }",
        ".epoch-state { background: #fef3e2; color: #f39c12; }",
        ".epoch-sleep { background: #f3e8fd; color: #9b59b6; }",
        ".timestamp { color: #95a5a6; font-size: 12px; margin-top: 30px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        f"<h1>EEG Preprocessing QA Report{' - Subject ' + subject_filter if subject_filter else ' - Global Summary'}</h1>",
        f"<p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "",
        "<div class='summary-cards'>",
        f"<div class='card'><div class='card-value neutral'>{n_recordings}</div><div class='card-label'>Total Recordings</div></div>",
        f"<div class='card'><div class='card-value pass'>{n_passed}</div><div class='card-label'>Passed QA</div></div>",
        f"<div class='card'><div class='card-value fail'>{n_failed}</div><div class='card-label'>Failed QA</div></div>",
        f"<div class='card'><div class='card-value'>{pass_rate:.1f}%</div><div class='card-label'>Pass Rate</div></div>",
        "</div>",
        "",
        "<h2>Pass Rate</h2>",
        "<div class='progress-bar'>",
        f"<div class='progress-fill progress-pass' style='width: {pass_rate}%'>{n_passed} passed</div>" if pass_rate > 0 else "",
        f"<div class='progress-fill progress-fail' style='width: {100-pass_rate}%'>{n_failed} failed</div>" if (100-pass_rate) > 0 else "",
        "</div>",
    ]
    
    # QA Flags section
    if flag_counts:
        html_parts.append("<h2>QA Flag Summary</h2>")
        html_parts.append("<div class='flags-grid'>")
        flag_labels = {
            'too_many_bad_channels': 'Too Many Bad Channels',
            'variance_budget_exceeded': 'ICA Variance Budget Exceeded',
            'too_many_bad_epochs': 'Too Many Bad Epochs',
            'low_line_noise_suppression': 'Low Line Noise Suppression',
            'no_ocular_ic_detected_with_eog': 'No Ocular IC Detected'
        }
        for flag, count in flag_counts.items():
            label = flag_labels.get(flag, flag.replace('_', ' ').title())
            ok_class = "ok" if count == 0 else ""
            html_parts.append(
                f"<div class='flag-item {ok_class}'>"
                f"<div class='flag-count'>{int(count)}</div>"
                f"<div class='flag-name'>{label}</div>"
                f"</div>"
            )
        html_parts.append("</div>")
    
    # Epoch type summary
    html_parts.append("<h2>Epoch Type Summary</h2>")
    html_parts.append("<div class='summary-cards'>")
    html_parts.append(f"<div class='card'><div class='card-value'>{len(evoked_df)}</div><div class='card-label'>Evoked Epochs Records</div></div>")
    html_parts.append(f"<div class='card'><div class='card-value'>{len(state_df)}</div><div class='card-label'>State Epochs Records</div></div>")
    html_parts.append(f"<div class='card'><div class='card-value'>{len(sleep_df)}</div><div class='card-label'>Sleep Epochs Records</div></div>")
    html_parts.append("</div>")

    # Subject-level summary (subjects x sessions/tasks, PASS/FAIL)
    html_parts.append("<h2>Subject Summary</h2>")
    if (not df.empty and
        all(col in df.columns for col in ['subject', 'session', 'task', 'passed'])):
        try:
            # One row per unique recording
            rec_df = df.drop_duplicates(subset=['subject', 'session', 'task'])

            # Classify subjects: completely good / partially bad / completely bad
            subj_pass = rec_df.groupby('subject')['passed'].apply(list)
            n_complete_good = 0
            n_partial_bad = 0
            n_complete_bad = 0
            for _subj, vals in subj_pass.items():
                has_pass = any(bool(v) for v in vals)
                has_fail = any(not bool(v) for v in vals)
                if has_pass and not has_fail:
                    n_complete_good += 1
                elif not has_pass and has_fail:
                    n_complete_bad += 1
                elif has_pass and has_fail:
                    n_partial_bad += 1

            html_parts.append("<div class='summary-cards'>")
            html_parts.append(
                f"<div class='card'><div class='card-value pass'>{n_complete_good}</div><div class='card-label'>Subjects Completely OK</div></div>"
            )
            html_parts.append(
                f"<div class='card'><div class='card-value neutral'>{n_partial_bad}</div><div class='card-label'>Subjects Partially Problematic</div></div>"
            )
            html_parts.append(
                f"<div class='card'><div class='card-value fail'>{n_complete_bad}</div><div class='card-label'>Subjects Completely Problematic</div></div>"
            )
            html_parts.append("</div>")

            summary = rec_df.pivot_table(
                index='subject',
                columns=['session', 'task'],
                values='passed',
                aggfunc='first'
            )
            summary = summary.sort_index(axis=0)

            sessions = []
            tasks_by_session = {}
            if summary.columns.nlevels == 2:
                for sess, task in summary.columns:
                    if sess not in tasks_by_session:
                        tasks_by_session[sess] = []
                        sessions.append(sess)
                    if task not in tasks_by_session[sess]:
                        tasks_by_session[sess].append(task)

            html_parts.append("<table>")
            # Header row: sessions
            html_parts.append("<tr>")
            html_parts.append("<th rowspan='2'>Subject</th>")
            for sess in sessions:
                n_tasks = len(tasks_by_session.get(sess, []))
                html_parts.append(
                    f"<th colspan='{n_tasks}'>ses-{sess}</th>"
                )
            html_parts.append("</tr>")

            # Header row: tasks
            html_parts.append("<tr>")
            for sess in sessions:
                for task in tasks_by_session.get(sess, []):
                    html_parts.append(f"<th>{task}</th>")
            html_parts.append("</tr>")

            # Rows per subject
            for subj in summary.index:
                html_parts.append("<tr>")
                html_parts.append(f"<td>sub-{subj}</td>")
                for sess in sessions:
                    for task in tasks_by_session.get(sess, []):
                        val = None
                        if (sess, task) in summary.columns:
                            try:
                                val = summary.loc[subj, (sess, task)]
                            except Exception:
                                val = None
                        if pd.isna(val):
                            val = None

                        cell_html = ""
                        if isinstance(val, (bool, int)):
                            passed_flag = bool(val)
                            status_class = 'status-pass' if passed_flag else 'status-fail'
                            status_text = 'PASS' if passed_flag else 'FAIL'
                            cell_html = f"<span class='{status_class}'>{status_text}</span>"
                        html_parts.append(f"<td>{cell_html}</td>")
                html_parts.append("</tr>")

            html_parts.append("</table>")
        except Exception:
            html_parts.append("<p>Could not compute subject summary.</p>")
    else:
        html_parts.append("<p>No subject summary available.</p>")

    # Detailed table
    html_parts.append("<h2>Detailed Results</h2>")
    if not df.empty:
        # Select columns to display
        display_cols = [
            'subject', 'session', 'task', 'epoch_type', 'passed', 'fail_reasons',
            'n_epochs', 'n_rejected_by_ar', 'n_bad_channels', 'n_ica_excluded',
            'delta_50hz_db', 'ica_excluded_variance_pct'
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        
        html_parts.append("<table>")
        html_parts.append("<tr>")
        for col in display_cols:
            html_parts.append(f"<th>{col.replace('_', ' ').title()}</th>")
        html_parts.append("</tr>")
        
        for _, row in df.iterrows():
            html_parts.append("<tr>")
            for col in display_cols:
                val = row.get(col, '')
                if pd.isna(val):
                    val = ''
                
                # Format specific columns
                if col == 'passed':
                    status_class = 'status-pass' if val else 'status-fail'
                    status_text = 'PASS' if val else 'FAIL'
                    val = f"<span class='{status_class}'>{status_text}</span>"
                elif col == 'epoch_type':
                    badge_class = f"epoch-{val}" if val in ['evoked', 'state', 'sleep'] else ''
                    val = f"<span class='epoch-badge {badge_class}'>{val}</span>"
                elif col in ['delta_50hz_db', 'ica_excluded_variance_pct']:
                    try:
                        val = f"{float(val):.2f}"
                    except:
                        pass
                
                html_parts.append(f"<td>{val}</td>")
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
    else:
        html_parts.append("<p>No QA data available yet.</p>")
    
    # Failed recordings section
    if not df.empty and 'passed' in df.columns:
        failed_df = df[df['passed'] == False]
        if not failed_df.empty:
            html_parts.append("<h2>Failed Recordings Details</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Subject</th><th>Session</th><th>Task</th><th>Epoch Type</th><th>Fail Reasons</th></tr>")
            for _, row in failed_df.iterrows():
                html_parts.append(
                    f"<tr><td>sub-{row.get('subject', '?')}</td>"
                    f"<td>ses-{row.get('session', '?')}</td>"
                    f"<td>{row.get('task', '?')}</td>"
                    f"<td>{row.get('epoch_type', '?')}</td>"
                    f"<td>{row.get('fail_reasons', '')}</td></tr>"
                )
            html_parts.append("</table>")
    
    html_parts.extend([
        "</div>",
        "</body>",
        "</html>",
    ])
    
    # Write HTML
    with open(output_path, 'w') as f:
        f.write("\n".join(html_parts))
    
    return output_path
