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
    state_dropped_post_ar: Optional[int] = None
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
    fail_reasons: List[str]
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

    # Save per-subject CSV
    try:
        qa_dir = os.path.join(derivatives_root, f"sub-{subject}")
        os.makedirs(qa_dir, exist_ok=True)
        qa_csv = os.path.join(qa_dir, f"sub-{subject}_task-{task}_qa_metrics.csv")

        col_order = [
            'subject', 'task', 'epoch_type', 'passed', 'fail_reasons',
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
                if not df_sum.empty and {"subject", "task", "epoch_type"}.issubset(df_sum.columns):
                    mask = (
                        (df_sum["subject"].astype(str) == str(subject)) & 
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
    ica_selection_log: Optional[Dict[str, Any]] = None
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
        derivatives_root, subject, task, qa_metrics, epoch_metrics, flags, passed, fail_reasons
    )
    
    return {
        'qa_metrics': qa_metrics,
        'flags': flags,
        'passed': passed,
        'fail_reasons': fail_reasons
    }
