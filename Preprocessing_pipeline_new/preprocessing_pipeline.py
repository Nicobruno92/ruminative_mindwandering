# =============================================================================
# DETERMINISTIC EEG PREPROCESSING PIPELINE
# =============================================================================
"""
Automated EEG preprocessing pipeline for mind-wandering analysis.

This pipeline processes raw BrainVision EEG data through a standardized
preprocessing workflow:
1. Data loading and initial filtering
2. Bad channel detection
3. ICA artifact removal
4. Epoch extraction (evoked and state)
5. Artifact rejection
6. Quality assessment

The pipeline is designed for scientific reproducibility with deterministic
methods and fixed random seeds.
"""
# =============================================================================

import os
import sys
import gc
from typing import Any, Dict, List

# =============================================================================
# THREADING CONFIGURATION
# =============================================================================
# Set threading environment variables early to prevent onnxruntime affinity errors
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1") 
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
os.environ.setdefault("ONNX_DISABLE_THREADPOOL_AFFINITY", "1")
os.environ.setdefault("ORT_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "sequential")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

# =============================================================================
# CORE IMPORTS
# =============================================================================
import numpy as np
import mne

# =============================================================================
# LOCAL MODULE IMPORTS
# =============================================================================
# Local imports (support running as a script)
try:
    from bids_compliance import BIDSCompliance
    from data_harmonization import load_config
    from steps_asr import apply_asr_if_configured
    from steps_bads import run_pyprep_noisychannels
    from steps_ica import fit_ica_deterministic, auto_select_ica_components
    from steps_epochs import (
        make_evoked_epochs,
        make_state_preprobe_epochs,
        make_sleep_preprobe_epochs,
    )
    from steps_reject import run_autoreject
    from steps_report import (
        compute_psd_figure, add_psd_figures, add_excluded_ics,
        add_ica_selection_table, add_drop_log,
        add_epoch_rejection_summary, add_qa_metrics_summary
    )
    from steps_qa_metrics import (
        run_complete_qa_assessment, compute_evoked_epoch_metrics,
        compute_epoch_counts_around_autoreject, 
        prepare_epoch_metrics_for_reports
    )
except ImportError:
    # Fallback for running as script from within directory
    _this_dir = os.path.dirname(__file__)
    if _this_dir not in sys.path:
        sys.path.append(_this_dir)
    from bids_compliance import BIDSCompliance
    from data_harmonization import load_config
    from steps_report import compute_psd_figure
    from steps_asr import apply_asr_if_configured
    from steps_bads import run_pyprep_noisychannels
    from steps_ica import fit_ica_deterministic, auto_select_ica_components
    from steps_epochs import (
        make_evoked_epochs,
        make_state_preprobe_epochs,
        make_sleep_preprobe_epochs,
    )
    from steps_reject import run_autoreject
    from steps_report import (
        add_psd_figures, add_excluded_ics, add_ica_selection_table,
        add_drop_log, add_epoch_rejection_summary, add_qa_metrics_summary
    )
    from steps_qa_metrics import (
        run_complete_qa_assessment, compute_evoked_epoch_metrics,
        compute_epoch_counts_around_autoreject, 
        prepare_epoch_metrics_for_reports
    )

from utils.log_preprocessing import LogPreprocessingDetails

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _compute_voltage_stats(raw: mne.io.BaseRaw, label: str = "") -> Dict[str, float]:
    """
    Compute voltage statistics for debugging.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data object
    label : str
        Label for this stage
        
    Returns
    -------
    Dict[str, float]
        Dictionary with voltage statistics
    """
    data = raw.get_data(picks='eeg')
    mean_abs = np.mean(np.abs(data))
    p2p = np.ptp(data)
    std = np.std(data)
    
    stats = {
        'mean_abs_uv': mean_abs * 1e6,
        'p2p_uv': p2p * 1e6,
        'std_uv': std * 1e6,
        'min_uv': np.min(data) * 1e6,
        'max_uv': np.max(data) * 1e6
    }
    
    if label:
        print(f"  [VOLTAGE DEBUG] {label}:")
        print(f"    Mean abs: {stats['mean_abs_uv']:.2f} µV")
        print(f"    Peak-to-peak: {stats['p2p_uv']:.2f} µV")
        print(f"    Std dev: {stats['std_uv']:.2f} µV")
    
    return stats


def _get_cfg_required(cfg: Dict[str, Any], key_path: List[str]):
    """
    Extract required configuration parameter from nested dictionary.
    
    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary
    key_path : List[str]
        Path to the required key (e.g., ["filters", "ica"])
        
    Returns
    -------
    Any
        The value at the specified key path
        
    Raises
    ------
    KeyError
        If the key path is not found in the configuration
    """
    cur = cfg
    for key in key_path:
        if not isinstance(cur, dict) or key not in cur:
            joined = "/".join(key_path)
            raise KeyError(f"Missing required config key: {joined}")
        cur = cur[key]
    return cur


def _ensure_report_dir(derivatives_root: str, subject: str) -> str:
    """
    Create and return the per-subject reports directory under derivatives.
    
    Parameters
    ----------
    derivatives_root : str
        Path to BIDS derivatives root
    subject : str
        Subject identifier
        
    Returns
    -------
    str
        Path to the created reports directory
    """
    out_dir = os.path.join(derivatives_root, f"sub-{subject}", "reports")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================
def process_subject_task(cfg: Dict[str, Any], subject: str, task: str) -> None:
    """
    Process a single subject-task combination through the EEG preprocessing 
    pipeline.
    
    This function implements the complete preprocessing workflow:
    1. Configuration and setup
    2. Data loading and initial filtering
    3. Bad channel detection
    4. ICA artifact removal
    5. Epoch extraction (evoked and state)
    6. Artifact rejection
    7. Quality assessment and reporting
    
    Parameters
    ----------
    cfg : Dict[str, Any]
        Configuration dictionary loaded from YAML
    subject : str
        Subject identifier (e.g., "04")
    task : str
        Task identifier (e.g., "Sart1")
    """
    # =========================================================================
    # 1. CONFIGURATION AND SETUP
    # =========================================================================
    print(f"Processing subject {subject}, task {task}")
    
    # Extract project configuration
    project = cfg["project"]
    bids_root = project.get("raw_root") or project.get("bids_root")
    derivatives_root = project.get("derivatives_root")
    overwrite = bool(project.get("overwrite", True))

    # Initialize BIDS I/O handler
    bidsio = BIDSCompliance(
        bids_root=bids_root, 
        dataset_name=project.get("dataset_name")
    )

    # Setup reporting and logging
    reports_dir = _ensure_report_dir(derivatives_root, subject)
    report = mne.Report(
        title=f"Preprocessing sub-{subject} task-{task}", 
        verbose=False
    )
    log_json = os.path.join(
        derivatives_root, 
        "logs_preprocessing_details_all_subjects_eeg.json"
    )
    logger = LogPreprocessingDetails(
        json_path=log_json, 
        subject=subject, 
        task=task
    )

    
    # =========================================================================
    # 2. EXTRACT CONFIGURATION PARAMETERS
    # =========================================================================
    # Basic processing parameters
    sr_target = int(_get_cfg_required(cfg, ["sr_target"]))
    notch_freqs = _get_cfg_required(cfg, ["notch_freqs"])
    avg_ref_projector = bool(_get_cfg_required(cfg, ["avg_ref_projector"]))
    use_csd = bool(_get_cfg_required(cfg, ["use_csd"]))
    
    # Filter parameters
    filters_ica = _get_cfg_required(cfg, ["filters", "ica"])
    filters_analysis= _get_cfg_required(cfg, ["filters", "analysis"])
    # filters_state = _get_cfg_required(cfg, ["filters", "state"])
    
    # ASR parameters (optional)
    asr_cfg = cfg.get("asr", None)
    
    # PyPREP bad channel detection parameters
    do_detrend = bool(_get_cfg_required(cfg, ["pyprep", "do_detrend"]))
    use_ransac = bool(_get_cfg_required(cfg, ["pyprep", "ransac"]))
    channel_wise = bool(_get_cfg_required(cfg, ["pyprep", "channel_wise"]))
    # Optional prefilter for bad-channel detection
    # (helps reduce frontal over-flagging from drifts/blinks)
    pyprep_prefilter_cfg = cfg.get("pyprep_prefilter", {})
    prefilter_for_pyprep = None
    if (
        isinstance(pyprep_prefilter_cfg, dict)
        and bool(pyprep_prefilter_cfg.get("apply", False))
    ):
        prefilter_for_pyprep = {
            "l_freq": pyprep_prefilter_cfg.get("l_freq"),
            "h_freq": pyprep_prefilter_cfg.get("h_freq"),
        }

    # Optional quick ICA before bad detection (on a copy)
    pre_ica_cfg = cfg.get("pyprep_pre_ica", {})
    pre_ica_opts = None
    if isinstance(pre_ica_cfg, dict) and bool(pre_ica_cfg.get("apply", False)):
        rs_default = int(cfg.get("ica", {}).get("random_state", 42))
        pre_ica_opts = {
            "apply": True,
            "method": pre_ica_cfg.get("method", "fastica"),
            "n_components": pre_ica_cfg.get("n_components", 0.95),
            "max_iter": pre_ica_cfg.get("max_iter", 300),
            "random_state": pre_ica_cfg.get("random_state", rs_default),
            "eog_channels": pre_ica_cfg.get("eog_channels", None),
            "muscle_threshold": pre_ica_cfg.get("muscle_threshold", 0.7),
            "filters": pre_ica_cfg.get(
                "filters",
                {"l_freq": 1.0, "h_freq": 40},
            ),
        }
    
    # Output file naming
    name_raw_clean = _get_cfg_required(cfg, ["names", "raw_clean_desc"])
    name_csd = _get_cfg_required(cfg, ["names", "csd_desc"])
    name_ica = _get_cfg_required(cfg, ["names", "ica_desc"])
    _ = name_ica  # kept for compatibility with previous outputs
    name_evoked = _get_cfg_required(cfg, ["names", "evoked_desc"])
    name_state = _get_cfg_required(cfg, ["names", "state_desc"])
    name_sleep = _get_cfg_required(cfg, ["names", "sleep_desc"])
    
    # Epoch parameters
    state_overlap = float(_get_cfg_required(cfg, ["state_windows_overlap_s"]))
    tmin = float(_get_cfg_required(cfg, ["epochs_evoked", "tmin"]))
    tmax = float(_get_cfg_required(cfg, ["epochs_evoked", "tmax"]))
    baseline = tuple(_get_cfg_required(cfg, ["epochs_evoked", "baseline"]))
    pre_probe_s = float(
        _get_cfg_required(cfg, ["state_windows", "pre_probe_s"])
    )
    mini_epoch_s = float(
        _get_cfg_required(cfg, ["state_windows", "mini_epoch_s"])
    )
    sleep_pre_probe_s = float(
        _get_cfg_required(cfg, ["sleep_windows", "pre_probe_s"])
    )
    sleep_reject_by_annotation = bool(
        cfg.get("sleep_windows", {}).get("reject_by_annotation", True)
    )
    
    # Coarse epoch rejection threshold (applied before AutoReject)
    # Optional: if not specified, no coarse rejection is applied
    reject_threshold = cfg.get("epoch_reject_threshold", None)
    
    # ICA parameters
    ica_method = _get_cfg_required(cfg, ["ica", "method"])
    n_components = _get_cfg_required(cfg, ["ica", "n_components"])
    max_iter = _get_cfg_required(cfg, ["ica", "max_iter"])
    random_state = int(_get_cfg_required(cfg, ["ica", "random_state"]))
    
    # ICA component selection parameters (variance-weighted system)
    ica_selection = _get_cfg_required(cfg, ["ica_selection"])
    prob_min = float(ica_selection["iclabel_prob_min"])
    muscle_threshold = float(ica_selection["muscle_threshold"])
    variance_penalty_factor = float(ica_selection["variance_penalty_factor"])
    max_excluded_variance_ratio = float(
        ica_selection["max_excluded_variance_ratio"]
    )
    max_threshold_cap = float(ica_selection.get("max_threshold_cap", 0.99))
    pattern_bonus = float(ica_selection.get("pattern_bonus", 0.10))

    # =========================================================================
    # 3. DATA LOADING AND INITIAL PROCESSING
    # =========================================================================
    print("Loading raw data...")
    # Read raw from BIDS (annotations already harmonized by previous step)
    raw = bidsio.read_raw(subject=subject, task=task, preload=True)
    # Ensure EOG channels are correctly typed
    eog_candidates = {}
    for ch_name in ["VEOG", "HEOG"]:
        if ch_name in raw.ch_names:
            eog_candidates[ch_name] = "eog"
    if eog_candidates:
        raw.set_channel_types(eog_candidates)
    # Build events directly from existing annotations
    events, event_id = mne.events_from_annotations(raw)
    
    # DEBUG: Voltage after loading
    _compute_voltage_stats(raw, "1. RAW (after loading)")

    # =========================================================================
    # 4. NOTCH FILTERING AND PSD ANALYSIS
    # =========================================================================
    print("Applying notch filter...")
    # QA PSD pre-notch
    fig_pre = compute_psd_figure(raw)

    # Apply notch filter for line noise removal
    raw.load_data()
    raw.notch_filter(freqs=notch_freqs, method="fir", n_jobs=1)
    fig_post = compute_psd_figure(raw)
    
    # DEBUG: Voltage after notch
    _compute_voltage_stats(raw, "2. After NOTCH filter")
    
    # Add PSD comparison to report
    add_psd_figures(report, fig_pre, fig_post)

    # =========================================================================
    # 4a. OPTIONAL ASR (ARTIFACT SUBSPACE RECONSTRUCTION)
    # =========================================================================
    # Apply ASR if configured (before bad channel detection)
    raw_asr, asr_applied = apply_asr_if_configured(
        raw,
        asr_cfg,
        sfreq=float(raw.info["sfreq"]),
    )
    if asr_applied:
        # Add ASR PSD comparison to report
        fig_pre_asr = fig_post  # Post-notch is pre-ASR
        fig_post_asr = compute_psd_figure(raw_asr)
        report.add_figure(
            fig=fig_pre_asr,
            title="PSD Pre-ASR",
            caption="Power spectral density before ASR artifact removal",
            section="ASR",
        )
        report.add_figure(
            fig=fig_post_asr,
            title="PSD Post-ASR",
            caption="Power spectral density after ASR artifact removal",
            section="ASR",
        )
        raw = raw_asr  # Use ASR-cleaned data for downstream processing
        del raw_asr
        gc.collect()

    # =========================================================================
    # 5. BAD CHANNEL DETECTION
    # =========================================================================
    print("Detecting bad channels...")
    # Capture initial bad channels before additional detection
    prelim_bads = list(raw.info.get("bads", []))

    # Detect noisy channels using PyPREP RANSAC
    prep_bads = run_pyprep_noisychannels(
        raw,
        do_detrend=do_detrend,
        ransac=use_ransac,
        channel_wise=channel_wise,
        random_state=random_state,
        prefilter=prefilter_for_pyprep,
        pre_ica_opts=pre_ica_opts,
    )
    
    # Combine bad channels from different sources
    all_bads = (raw.info.get("bads") or []) + (prep_bads or [])
    raw.info["bads"] = sorted(set(all_bads))
    
    # DEBUG: Voltage after bad channel detection (before interpolation)
    _compute_voltage_stats(raw, f"3. After BAD detection ({len(raw.info['bads'])} bads marked)")

    # =========================================================================
    # 7. INTERPOLATION, MONTAGE, AND REFERENCE SETUP
    # =========================================================================
    print("Interpolating bad channels and setting up montage...")
    
    # Capture bad channels before interpolation for BIDS output
    bads_pre_interp = list(raw.info.get("bads", []))
    
    # 1) Interpolate bad channels first
    raw.interpolate_bads(reset_bads=True, mode="accurate")
    
    # DEBUG: Voltage after interpolation
    _compute_voltage_stats(raw, f"4. After INTERPOLATION ({len(bads_pre_interp)} channels)")

    # 2) Set montage using CACS-64_REF.bvef
    montage = mne.channels.read_custom_montage("Preprocessing_pipeline_new/CACS-64_REF.bvef")
    raw.set_montage(montage)
    
    # 3) Define EEG picks excluding reference and auxiliary channels
    ref_like = ["REF", "EXG1", "EXG2", "HEOG", "VEOG", "M1", "M2", "GND"]
    picks_eeg = mne.pick_types(raw.info, eeg=True, exclude=ref_like)
    
    # 4) Apply average reference ONLY to EEG channels
    if avg_ref_projector:
        raw.set_eeg_reference("average", projection=False)
    
    # DEBUG: Voltage after reference
    _compute_voltage_stats(raw, f"5. After REFERENCE (using {len(picks_eeg)} EEG channels)")

    # =========================================================================
    # 8. FILTERING (NO RESAMPLING)
    # =========================================================================
    print("Applying filters...")
    
    # Create two copies for different processing streams
    # ICA copy: dedicated for ICA fitting, can be freed early
    raw_for_ica = raw.copy().load_data()
    raw_for_ica.filter(
        l_freq=filters_ica.get("l_freq"),
        h_freq=filters_ica.get("h_freq"),
        phase="zero",
        n_jobs=1,
    )
    # NO RESAMPLING - removed sr_target resampling
    
    # DEBUG: Voltage of ICA copy after filtering
    _compute_voltage_stats(
        raw_for_ica, 
        f"6a. ICA copy after filter ({filters_ica.get('l_freq')}-{filters_ica.get('h_freq')} Hz)"
    )

    # Analysis copy: for final analysis and epoch extraction
    raw_analysis = raw
    raw_analysis.filter(
        l_freq=filters_analysis.get("l_freq"),
        h_freq=filters_analysis.get("h_freq"),
        phase="zero",
        n_jobs=1,
    )
    
    # DEBUG: Voltage of analysis copy after filtering
    _compute_voltage_stats(
        raw_analysis, 
        f"6b. Analysis copy after filter ({filters_analysis.get('l_freq')}-{filters_analysis.get('h_freq')} Hz)"
    )
    # =========================================================================
    # 8. ICA ARTIFACT REMOVAL
    # =========================================================================
    print("Fitting ICA...")
    # Fit ICA using deterministic method
    ica = fit_ica_deterministic(
        raw_for_ica,
        ica_method,
        n_components,
        max_iter,
        random_state,
        report,
    )
    
    # Automatically select components to exclude (variance-weighted)
    ica_exclude, combination_method, ica_selection_log = (
        auto_select_ica_components(
            raw_for_ica,
            ica,
            prob_min,
            muscle_threshold,
            variance_penalty_factor,
            max_excluded_variance_ratio,
            max_threshold_cap,
            pattern_bonus,
        )
    )
    ica.exclude = ica_exclude
    print(f"ICA combination method used: {combination_method}")
    
    # Log ICA selection details
    logger.log_detail("ica_selection_method", combination_method)
    logger.log_detail("ica_selection_details", ica_selection_log)

    # Apply ICA to analysis copy
    raw_clean = raw_analysis
    
    # DEBUG: Voltage BEFORE ICA application
    _compute_voltage_stats(raw_clean, "7a. BEFORE ICA.apply()")
    
    # DEBUG: Print ICA exclude list
    print(f"\n{'='*70}")
    print(f"ICA EXCLUSION CHECK:")
    print(f"  ica.exclude = {ica.exclude}")
    print(f"  Number of components to exclude: {len(ica.exclude) if ica.exclude else 0}")
    print(f"  Total ICA components: {ica.n_components_}")
    print(f"{'='*70}\n")
    
    if ica.exclude:
        print(f"Applying ICA to remove {len(ica.exclude)} components...")
        ica.apply(raw_clean)
        print("ICA application completed.")
        # DEBUG: Voltage AFTER ICA application
        _compute_voltage_stats(
            raw_clean, 
            f"7b. AFTER ICA.apply() (removed {len(ica.exclude)} components: {ica.exclude})"
        )
    else:
        print("  [WARNING] No ICA components excluded, skipping ICA.apply()")
        print("  This means no artifacts were detected or thresholds are too strict!")
    
    # Add ICA-related sections to report
    add_excluded_ics(report, ica, raw_clean)
    add_ica_selection_table(report, ica_selection_log)
    report.add_raw(raw=raw_clean, title="Raw clean", psd=True)


    # Memory cleanup: ICA copy no longer needed
    del raw_for_ica
    gc.collect()

    # =========================================================================
    # 8a. OPTIONAL CSD COMPUTATION (AFTER ICA - CLEAN DATA)
    # =========================================================================
    # Compute CSD after ICA cleaning to get clean spatial-transform data
    # Following MNE example: CSD on clean, artifact-removed data
    raw_csd = None
    if use_csd:
        print("Computing Current Source Density on cleaned data...")
        # CSD benefits from being applied to clean data (after ICA)
        # Bad channels are already cleared, no need to worry about them
        raw_csd = mne.preprocessing.compute_current_source_density(raw_clean.copy())
        print(f"  [INFO] CSD computed on ICA-cleaned data")
    
    # =========================================================================
    # 9. FILE SAVING
    # =========================================================================
    print("Saving processed data...")
    # Save ICA-cleaned data to derivatives
    bidsio.write_derivative_raw(
        raw_clean,
        derivatives_root,
        subject,
        task,
        desc=name_raw_clean,
        overwrite=overwrite,
        bad_channels_pre_interp=bads_pre_interp,
    )
    
    # Save CSD data (if computed)
    # Note: raw_csd is already computed on filtered+ICA-cleaned data
    # so it doesn't need additional filtering
    if raw_csd is not None:
        print("Saving CSD-transformed data...")
        # Save CSD to derivatives
        bidsio.write_derivative_raw(
            raw_csd,
            derivatives_root,
            subject,
            task,
            desc=name_csd,
            overwrite=overwrite,
        )
        print(f"  [INFO] CSD data saved with desc='{name_csd}'")
        
        # Memory cleanup: CSD data no longer needed after saving
        del raw_csd
        gc.collect()

    # =========================================================================
    # 10. EVOKED EPOCHS PROCESSING
    # =========================================================================
    print("Creating evoked epochs...")
    # Optionally restrict to go/nogo events based on config
    if bool(cfg.get("evoked_filter_go_nogo_only", True)):
        go_nogo_event_id = {
            k: v for k, v in event_id.items()
            if (isinstance(k, str) and ("go" in k or "nogo" in k))
        }
        event_id_use = go_nogo_event_id if go_nogo_event_id else event_id
    else:
        event_id_use = event_id
        
    # Create evoked epochs
    epochs = make_evoked_epochs(
        raw_clean,
        events,
        event_id_use,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject_by_annotation=True,
        reject=reject_threshold,
    )
    
    # Apply picks_eeg to epochs after creation
    # epochs.pick(picks_eeg)

    # =========================================================================
    # 11. AUTOREJECT ARTIFACT REJECTION
    # =========================================================================
    print("Running AutoReject...")
    # Apply AutoReject for automated artifact rejection
    ar_epochs, ar_reject_epochs_idx = run_autoreject(
        epochs,
        cv=int(_get_cfg_required(cfg, ["autoreject", "cv"])),
        random_state=int(
            _get_cfg_required(cfg, ["autoreject", "random_state"])
        ),
        logger=logger,
        log_prefix="evoked",
        n_jobs=(
            int(_get_cfg_required(cfg, ["autoreject", "n_jobs"]))
            if "n_jobs" in cfg.get("autoreject", {})
            else 1
        ),
        thresh_method=cfg.get("autoreject", {}).get("thresh_method"),
        n_interpolate=cfg.get("autoreject", {}).get("n_interpolate"),
        n_consensus=cfg.get("autoreject", {}).get("n_consensus"),
        # picks=picks_eeg,
    )
    
    # Debug AutoReject output
    print(
        f"[DEBUG] AR returned: ar_reject_epochs_idx = {ar_reject_epochs_idx}"
    )
    if ar_reject_epochs_idx is not None:
        print(f"[DEBUG] AR rejected {len(ar_reject_epochs_idx)} epochs")
    else:
        print("[DEBUG] AR returned None")
    
    # Add drop-log visualization before calling drop_bad
    try:
        add_drop_log(
            report,
            ar_epochs,
            title="Epochs drop log (including AutoReject)",
        )
    except Exception:
        pass
    
    # Apply picks_eeg to AutoReject results
    # ar_epochs.pick(picks_eeg)
    
    # Store epochs after AutoReject but before drop_bad for counting
    ar_epochs_before_drop = ar_epochs.copy()
    ar_epochs.drop_bad()
    n_before_ar, pre_len_after_ar, n_dropped_post_ar = (
        compute_epoch_counts_around_autoreject(
            epochs,
            ar_epochs_before_drop,
            ar_epochs,
        )
    )

    # =========================================================================
    # 12. EVOKED EPOCHS SAVING AND REPORTING
    # =========================================================================
    print("Processing evoked epochs for saving...")
    # Build ERP evoked responses for QA
    evk = []
    go_labels = [
        k for k in ar_epochs.event_id.keys()
        if isinstance(k, str) and k.lower().startswith("go")
    ]
    nogo_labels = [
        k for k in ar_epochs.event_id.keys()
        if isinstance(k, str) and k.lower().startswith("nogo")
    ]
    if go_labels:
        evk.append(ar_epochs[go_labels].average())
    if nogo_labels:
        evk.append(ar_epochs[nogo_labels].average())
    if evk:
        report.add_evokeds(
            evokeds=evk,
            titles=["Go (all)", "NoGo (all)"][: len(evk)],
        )

    # Save evoked epochs to derivatives
    bidsio.write_derivative_epochs(
        ar_epochs,
        derivatives_root,
        subject,
        task,
        desc=name_evoked,
    )
    
    # Add before/after epochs and comparison to report
    try:
        from steps_report import add_epochs_sections, add_evoked_comparison
        # Note: using ar_epochs for both before/after since original epochs
        # was cleaned up
        add_epochs_sections(report, ar_epochs, ar_epochs, label="Evoked")
        add_evoked_comparison(report, ar_epochs)
    except Exception:
        pass

    # =========================================================================
    # 13. PREPARE FOR QUALITY ASSESSMENT (BEFORE MEMORY CLEANUP)
    # =========================================================================
    print("Preparing epoch metrics for quality assessment...")
    # Store epoch metrics for later QA computation (after state epochs are
    # processed)
    epoch_metrics = compute_evoked_epoch_metrics(
        epochs=epochs,
        ar_epochs=ar_epochs,
        ar_reject_epochs_idx=ar_reject_epochs_idx,
        epochs_state=None,  # Will be updated later if state epochs exist
        state_rejected_by_ar=None,
        state_input_total=None,
        state_after_construct=None,
        state_dropped_post_ar=None
    )
    
    # Update with post-AutoReject counts
    epoch_metrics['n_dropped_post_ar'] = n_dropped_post_ar
    
    # Memory cleanup: original epochs no longer needed after metrics
    # computation
    del epochs
    gc.collect()

    # =========================================================================
    # 14. STATE EPOCHS PROCESSING
    # =========================================================================
    print("Creating state epochs...")
    # State windows pre-probe using existing THOUGHT_PROBE annotations
    raw_state = raw_clean.copy()
    
    # For state epochs, don't apply coarse rejection - let AutoReject handle it
    # State/resting data naturally has more variability than evoked responses
    epochs_state = make_state_preprobe_epochs(
        raw_state,
        pre_probe_s=pre_probe_s,
        mini_epoch_s=mini_epoch_s,
        overlap_s=state_overlap,
        reject=None,  # No coarse rejection for state epochs
    )
    
    # # Apply picks_eeg to state epochs after creation
    # if epochs_state is not None:
    #     epochs_state.pick(picks_eeg)
    
    # Memory cleanup: raw_state no longer needed
    del raw_state
    gc.collect()
    
    if epochs_state is not None:
        # Track state epoch counts
        try:
            state_input_total = len(getattr(epochs_state, "drop_log", []))
        except Exception:
            state_input_total = None
        try:
            state_after_construct = int(len(epochs_state.events))
        except Exception:
            state_after_construct = None
            
        # Apply AutoReject to state epochs
        ar_state_epochs, ar_reject_epochs_idx_state = run_autoreject(
            epochs_state,
            cv=int(_get_cfg_required(cfg, ["autoreject", "cv"])),
            random_state=int(
                _get_cfg_required(cfg, ["autoreject", "random_state"])
            ),
            logger=logger,
            log_prefix="state",
            n_jobs=(
                int(_get_cfg_required(cfg, ["autoreject", "n_jobs"]))
                if "n_jobs" in cfg.get("autoreject", {})
                else 1
            ),
            thresh_method=cfg.get("autoreject", {}).get("thresh_method"),
            n_interpolate=cfg.get("autoreject", {}).get("n_interpolate"),
            n_consensus=cfg.get("autoreject", {}).get("n_consensus"),
            # picks=picks_eeg,
        )
        
        # Debug AutoReject output for state epochs
        print(
            "[DEBUG] State AR returned: "
            f"ar_reject_epochs_idx_state = {ar_reject_epochs_idx_state}"
        )
        if ar_reject_epochs_idx_state is not None:
            print(
                f"[DEBUG] State AR rejected {len(ar_reject_epochs_idx_state)} "
                "epochs"
            )
        else:
            print("[DEBUG] State AR returned None")
        
        # Apply picks_eeg to state AutoReject results
        # ar_state_epochs.pick(picks_eeg)
        
        # Store epochs after AutoReject but before drop_bad for counting
        ar_state_epochs_before_drop = ar_state_epochs.copy()
        ar_state_epochs.drop_bad()
        state_before_ar, pre_len_state, n_dropped_post_ar_state = (
            compute_epoch_counts_around_autoreject(
                epochs_state,
                ar_state_epochs_before_drop,
                ar_state_epochs,
            )
        )
        n_after_state_drop = int(len(ar_state_epochs))
        # Calculate epochs rejected by AutoReject (before drop_bad)
        rejected_by_ar_state = max(0, state_before_ar - pre_len_state)
        
        # Update epoch metrics with state information
        epoch_metrics.update({
            'state_input_total': state_input_total,
            'state_after_construct': state_after_construct,
            'state_rejected_by_ar': rejected_by_ar_state,
            'state_dropped_post_ar': n_dropped_post_ar_state
        })
    # Mark presence of state epochs and store final kept count for
    # CSV/reporting
        epoch_metrics['has_state_epochs'] = True
        epoch_metrics['n_state_epochs'] = n_after_state_drop
        
        # Save state epochs to derivatives
        bidsio.write_derivative_epochs(
            ar_state_epochs,
            derivatives_root,
            subject,
            task,
            desc=name_state,
        )
        
        # Add before/after to report
        try:
            from steps_report import add_epochs_sections
            add_epochs_sections(
                report, ar_state_epochs, ar_state_epochs, label="State"
            )
        except Exception:
            pass
        
        # Memory cleanup
        gc.collect()

    # =========================================================================
    # 14b. SLEEP (LONG PRE-PROBE) EPOCHS PROCESSING
    # =========================================================================
    print("Creating sleep epochs...")
    raw_sleep = raw_clean.copy()

    # For sleep epochs, don't apply coarse rejection - let AutoReject handle it
    # Long resting-state windows need permissive thresholds
    epochs_sleep = make_sleep_preprobe_epochs(
        raw_sleep,
        pre_probe_s=sleep_pre_probe_s,
        reject=None,  # No coarse rejection for sleep epochs
        reject_by_annotation=sleep_reject_by_annotation,
    )

    # raw_sleep no longer needed after constructing epochs
    del raw_sleep
    gc.collect()

    if epochs_sleep is not None:
        # Track sleep epoch counts
        try:
            sleep_input_total = len(getattr(epochs_sleep, "drop_log", []))
        except Exception:
            sleep_input_total = None
        try:
            sleep_after_construct = int(len(epochs_sleep.events))
        except Exception:
            sleep_after_construct = None
        
        # For sleep epochs, skip AutoReject and rely only on the coarse
        # amplitude-based rejection defined at epoch construction
        print("Skipping AutoReject for sleep epochs; using fixed reject thresholds only.")
        sleep_epochs_clean = epochs_sleep.copy()
        
        # Update epoch metrics with sleep information
        epoch_metrics.update({
            'has_sleep_epochs': True,
            'sleep_input_total': sleep_input_total,
            'sleep_after_construct': sleep_after_construct,
            'n_sleep_epochs': int(len(sleep_epochs_clean)),
        })

        # Save sleep epochs to derivatives
        bidsio.write_derivative_epochs(
            sleep_epochs_clean,
            derivatives_root,
            subject,
            task,
            desc=name_sleep,
        )

        from steps_report import add_epochs_sections
        add_epochs_sections(
            report, sleep_epochs_clean, sleep_epochs_clean, label="Sleep"
        )

        # Memory cleanup
        gc.collect()

    # =========================================================================
    # 15. COMPLETE QUALITY ASSESSMENT
    # =========================================================================
    print("Running complete quality assessment...")
    # Run complete QA assessment with all epoch metrics (evoked + state)
    # Include QA threshold for per-epoch rejection flagging in CSV (optional)
    try:
        epoch_metrics['qa_thr_bad_epochs_ratio'] = float(
            _get_cfg_required(cfg, ["qa_thresholds", "max_bad_epochs_ratio"])
        )
    except Exception:
        pass
    qa_results = run_complete_qa_assessment(
        raw_before_ica=raw_analysis,  # Pre-ICA for comparison
        raw_after_ica=raw_clean,      # Post-ICA for comparison
        ica_exclude=ica_exclude,
        ica_n_components=ica.n_components_,
        bads_pre_interp=bads_pre_interp,
        epoch_metrics=epoch_metrics,  # Now includes evoked + state metrics
        derivatives_root=derivatives_root,
        subject=subject,
        task=task,
        cfg=cfg,
        logger=logger,
        prelim_bads=prelim_bads,
        prep_bads=prep_bads,
        ica_selection_log=ica_selection_log,  # Pass variance selection log
    )


    # =========================================================================
    # 16. FINAL REPORTING AND CLEANUP
    # =========================================================================
    print("Generating final report...")
    # Add final drop-log figure (after drop_bad)
    add_drop_log(report, ar_epochs, title="Final clean epochs drop log")

    # Add comprehensive QA metrics summary
    try:
        # Prepare evoked and state metrics for reports
        evoked_metrics, state_metrics = prepare_epoch_metrics_for_reports(
            epoch_metrics
        )
        
        # Prepare global metrics with detailed channel and ICA info
        eeg_picks = mne.pick_types(raw_clean.info, eeg=True)
        bad_channels_ratio = (
            len(bads_pre_interp) / len(eeg_picks) 
            if bads_pre_interp else 0.0
        )
        global_metrics = {
            'bad_channels_ratio': bad_channels_ratio,
            'n_bad_channels': len(bads_pre_interp) if bads_pre_interp else 0,
            'bad_channels': bads_pre_interp,
            'ica_excluded_ratio': qa_results['qa_metrics']['ica_excl_ratio'],
            'n_ica_excluded': len(ica_exclude) if ica_exclude else 0,
            'ica_excluded_indices': ica_exclude,
            'line_noise_improvement': qa_results['qa_metrics']['delta_50_db']
        }
        
        add_qa_metrics_summary(
            report=report,
            evoked_metrics=evoked_metrics,
            state_metrics=state_metrics,
            global_metrics=global_metrics
        )
    except Exception as e:
        print(f"Warning: Could not add QA metrics summary: {e}")

    # Save report
    report_path = os.path.join(
    reports_dir,
        f"sub-{subject}_task-{task}_desc-autoPreproc_eeg.html"
    )
    report.save(report_path, overwrite=True, open_browser=False)
    print(f"[OK] Saved report: {report_path}")

    # =========================================================================
    # 17. FINAL MEMORY CLEANUP
    # =========================================================================
    # Clean up remaining large objects
    try:
        del ar_epochs
    except Exception:
        pass
    try:
        del ar_state_epochs
    except Exception:
        pass
    try:
        del raw_csd
    except Exception:
        pass
    try:
        del ica
    except Exception:
        pass
    
    # Final garbage collection
    gc.collect()
    print(f"Finished processing subject {subject}, task {task}")


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================


def main():
    """
    Main function for the EEG preprocessing pipeline.
    
    Handles command-line arguments and orchestrates the preprocessing
    of single subject-task combinations or batch processing of all
    subjects and tasks specified in the configuration file.
    """
    import argparse

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description=(
            "Automatic EEG preprocessing pipeline (continuous -> epochs), "
            "BIDS in/out"
        )
    )
    parser.add_argument(
        "--config",
        default="./Preprocessing_pipeline_new/config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--subject",
        help="Process only this subject",
    )
    parser.add_argument(
        "--task",
        help="Process only this task",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Process single subject-task if specified
    if args.subject and args.task:
        process_subject_task(cfg, args.subject, args.task)
        return

    # Process all subjects and tasks from config
    print("Processing all subjects and tasks from configuration...")
    
    # Parse subjects from config
    subjects_cfg = cfg.get("subjects", [])
    if isinstance(subjects_cfg, str):
        s = subjects_cfg.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1]
        subjects = [
            t.strip() for t in s.replace(",", " ").split() if t.strip()
        ]
    else:
        subjects = [str(s) for s in subjects_cfg]

    # Get tasks from config
    tasks = cfg.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks provided in config under 'tasks'")

    # Process all subject-task combinations
    total_combinations = len(subjects) * len(tasks)
    current = 0
    
    for sub in subjects:
        for t in tasks:
            current += 1
            print(f"\n{'='*60}")
            print(f"Processing combination {current}/{total_combinations}: "
                  f"subject {sub}, task {t}")
            print(f"{'='*60}")
            try:
                process_subject_task(cfg, sub, str(t))
                print(f"✓ Successfully completed subject {sub}, task {t}")
            except Exception as e:
                print(f"✗ Error processing subject {sub}, task {t}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print("Preprocessing pipeline completed!")
    print(f"Processed {total_combinations} subject-task combinations")
    print(f"{'='*60}")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
