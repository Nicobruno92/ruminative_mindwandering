import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from helpers import (
        load_yaml_config,
        build_evoked_dir,
    )
except Exception:
    _this_dir = os.path.dirname(__file__)
    if _this_dir not in sys.path:
        sys.path.append(_this_dir)
    from helpers import (
        load_yaml_config,
        build_evoked_dir,
    )

# Try to import statsmodels for LMM
try:
    from statsmodels.formula.api import mixedlm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available. LMM analysis will be disabled.")

# Try to import Julia LMM bridge for advanced temporal analysis (Unfold.jl)
try:
    # Only import when needed to avoid unused import warnings
    HAS_JULIA_LMM = True
except ImportError:
    HAS_JULIA_LMM = False
    warnings.warn("Julia LMM bridge not available. Will use Python LMM only.")


def _find_probe_evokeds(features_root: str, subject: str, task: str) -> List[str]:
    """Find all probe evoked files for a subject-task in the new BIDS structure."""
    evoked_dir = build_evoked_dir(features_root, subject)
    if not os.path.isdir(evoked_dir):
        print(f"[DEBUG] No evoked directory found: {evoked_dir}")
        return []
    files = []
    for name in sorted(os.listdir(evoked_dir)):
        if (
            name.endswith("_ave.fif")
            and f"sub-{subject}_task-{task}_desc-probe-" in name
        ):
            files.append(os.path.join(evoked_dir, name))
    return files


def _split_by_label(paths: List[str]) -> Tuple[List[str], List[str]]:
    """Split evoked files by on-task/off-task labels."""
    on_paths, off_paths = [], []
    for p in paths:
        base = os.path.basename(p)
        if "_onTask_" in base or base.endswith("_onTask_ave.fif"):
            on_paths.append(p)
        elif "_offTask_" in base or base.endswith("_offTask_ave.fif"):
            off_paths.append(p)
    return on_paths, off_paths


def _roi_picks(info: mne.Info, roi_channels: List[str]) -> np.ndarray:
    """Get channel picks for a given ROI."""
    picks = [ch for ch in roi_channels if ch in info["ch_names"]]
    return mne.pick_channels(info["ch_names"], include=picks, exclude=[])


def _load_and_roi_average(path: str, roi_picks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Load evoked file and compute ROI average."""
    evk = mne.read_evokeds(path, condition=0, verbose=False)
    data = evk.data[roi_picks, :]
    roi_avg = data.mean(axis=0)
    
    # Convert to microvolts if data seems to be in volts (very small values)
    if np.abs(roi_avg).max() < 1e-3:
        roi_avg = roi_avg * 1e6
    
    return evk.times, roi_avg


def _extract_probe_info(path: str) -> Tuple[str, int, str]:
    """Extract subject, probe number, and condition from evoked file path."""
    basename = os.path.basename(path)
    # Expected format: sub-XX_task-YY_desc-probe-###_(onTask|offTask)_ave.fif
    parts = basename.replace(".fif", "").split("_")
    
    subject = None
    probe_num = None
    condition = None
    
    for part in parts:
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("desc-probe-"):
            probe_num = int(part.replace("desc-probe-", ""))
        elif part in ["onTask", "offTask", "unknown"]:
            condition = part
    
    return subject, probe_num, condition


def _detect_outlier_probes(probe_curves: List[Tuple[np.ndarray, np.ndarray]], 
                          z_threshold: float = 3.0) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """Remove outlier probes based on z-score of mean amplitude."""
    if len(probe_curves) < 3:  # Need at least 3 probes for meaningful outlier detection
        return probe_curves, []
    
    # Compute mean amplitude across time for each probe
    mean_amps = [np.mean(np.abs(v)) for _, v in probe_curves]
    z_scores = np.abs(stats.zscore(mean_amps))
    
    # Keep probes with z-score below threshold
    outlier_indices = np.where(z_scores > z_threshold)[0].tolist()
    clean_curves = [probe_curves[i] for i in range(len(probe_curves)) if i not in outlier_indices]
    
    return clean_curves, outlier_indices


def _compute_window_amplitude(
    times: np.ndarray,
    values: np.ndarray,
    window: Tuple[float, float],
    baseline: Optional[Tuple[float, float]] = None,
) -> float:
    """Compute mean amplitude in a time window."""
    # Get time indices for the window
    time_mask = (times >= window[0]) & (times <= window[1])
    if not np.any(time_mask):
        return np.nan
    
    # Extract data for time window
    window_data = values[time_mask]
    
    # Baseline correction if requested
    if baseline is not None:
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        if not np.any(baseline_mask):
            # Cannot find baseline window - this is critical for analysis
            print(f"[ERROR] Cannot find baseline window {baseline} in time range [{times.min():.3f}, {times.max():.3f}]")
            return np.nan
        
        baseline_data = values[baseline_mask]
        if len(baseline_data) == 0:
            print(f"[ERROR] Empty baseline data for window {baseline}")
            return np.nan
            
        baseline_mean = baseline_data.mean()
        if np.isnan(baseline_mean):
            print(f"[ERROR] Baseline mean is NaN for window {baseline}")
            return np.nan
            
        window_data = window_data - baseline_mean
    
    # Return mean amplitude across time
    return window_data.mean()


def collect_probe_data_for_lmm(cfg: Dict) -> pd.DataFrame:
    """
    Collect probe evoked data for LMM analysis.
    
    This function loads the probe evoked files created by make_probe_evokeds.py
    and extracts amplitude values for specified time windows and ROIs.
    """
    proj = cfg.get("project", {})
    features_root = proj.get("features_root")
    if not features_root:
        raise ValueError("project.features_root must be set in config")
    
    rois = cfg.get("erp_rois", {})
    lmm_cfg = cfg.get("lmm_analysis", {})
    time_windows = lmm_cfg.get("time_windows", {})
    baseline_window = lmm_cfg.get("baseline_window", [-0.1, 0.0])
    if not baseline_window:
        raise ValueError("baseline_window must be configured in lmm_analysis config. Cannot proceed without baseline.")
    baseline_window = tuple(baseline_window)
    
    # Validate baseline window
    if len(baseline_window) != 2:
        raise ValueError(f"baseline_window must have exactly 2 values [start, end], got {len(baseline_window)}")
    if baseline_window[0] >= baseline_window[1]:
        raise ValueError(f"baseline_window start ({baseline_window[0]}) must be < end ({baseline_window[1]})")
    
    # Get outlier detection parameters
    outlier_cfg = cfg.get("outlier_detection", {})
    z_threshold = float(outlier_cfg.get("epoch_z_threshold", 3.0))  # Use epoch threshold for probe-level outlier detection
    
    subjects_cfg = cfg.get("subjects", [])
    subjects = [str(s) for s in subjects_cfg]
    
    tasks_cfg = cfg.get("tasks", [])
    tasks = [str(t) for t in tasks_cfg]
    
    all_data = []

    print("Collecting probe evoked data for LMM analysis...")
    print(f"Looking for probe evoked files in: {features_root}")
    print(f"Subjects: {subjects[:3]}... (showing first 3)")
    print(f"Tasks: {tasks}")
    print(f"Outlier detection Z-threshold: {z_threshold}")
    
    for subject in subjects:
        subject_probes = {"onTask": 0, "offTask": 0}
        
        for task in tasks:
            # Find probe evoked files for this subject-task
            probe_paths = _find_probe_evokeds(features_root, subject, task)
            if not probe_paths:
                print(f"[WARN] No probe evoked files found for sub-{subject} task-{task}")
                continue
            
            print(f"Processing sub-{subject} task-{task}: {len(probe_paths)} probe files")
            
            # Split by condition
            on_paths, off_paths = _split_by_label(probe_paths)
            subject_probes["onTask"] += len(on_paths)
            subject_probes["offTask"] += len(off_paths)
            
            # Process each condition
            for condition, paths in [("onTask", on_paths), ("offTask", off_paths)]:
                if not paths:
                    continue

                # Load one file to get info for ROI picks
                if paths:
                    info_evoked = mne.read_evokeds(paths[0], condition=0, verbose=False).info
                
                # Process each ROI
                for roi_name, roi_channels in rois.items():
                    picks = _roi_picks(info_evoked, list(roi_channels))
                    if picks.size == 0:
                        continue

                    # Load all probes for this ROI-condition
                    probe_curves = []
                    probe_info = []
                    
                    for path in paths:
                        try:
                            times, roi_avg = _load_and_roi_average(path, picks)
                            probe_curves.append((times, roi_avg))
                            
                            # Extract probe information
                            _, probe_num, _ = _extract_probe_info(path)
                            probe_info.append(probe_num)
                        except Exception as e:
                            print(f"[WARN] Failed to load {path}: {e}")
                            continue
                    
                    if not probe_curves:
                        continue
            
                    # Apply outlier detection at probe level
                    clean_curves, outlier_indices = _detect_outlier_probes(probe_curves, z_threshold)
                    
                    if len(outlier_indices) > 0:
                        print(f"  Removed {len(outlier_indices)} outlier probes for {subject} {task} {roi_name} {condition}")
                    
                    # Process each clean probe
                    for i, (times, values) in enumerate(clean_curves):
                        # Get the original probe number (accounting for outlier removal)
                        original_idx = [j for j in range(len(probe_curves)) if j not in outlier_indices][i]
                        probe_num = probe_info[original_idx]
                        
                        # Process each time window
                        for window_name, window in time_windows.items():
                            # Compute amplitude for this window
                            amplitude = _compute_window_amplitude(times, values, window, baseline_window)
                            
                            if not np.isnan(amplitude):
                                # MANDATORY baseline computation - no escape allowed
                                if baseline_window is None:
                                    raise ValueError("Baseline window must be configured. baseline_window cannot be None.")
                                
                                baseline_value = _compute_window_amplitude(times, values, baseline_window, None)
                                if np.isnan(baseline_value):
                                    print(f"[WARN] Skipping probe {probe_num} due to invalid baseline computation")
                                    continue
                                
                                all_data.append({
                                        "subject": subject,
                                        "task": task,
                                        "probe": int(probe_num),
                                        "condition": condition,
                                        "roi": roi_name,
                                    "window": window_name,
                                        "amplitude": amplitude,
                                        "baseline": baseline_value,
                                    "window_start": window[0],
                                    "window_end": window[1],
                                    })

        print(f"Subject {subject}: {subject_probes['onTask']} onTask probes, {subject_probes['offTask']} offTask probes")
    
    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data collected - DataFrame is empty")
        return df
    
    print(f"Collected {len(df)} data points from {df['subject'].nunique()} subjects")
    return df


def run_lmm_analysis(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Run Linear Mixed Models for each ROI-window combination."""
    if not HAS_STATSMODELS:
        print("[ERROR] statsmodels not available. Cannot run LMM analysis.")
        return pd.DataFrame()
    
    alpha = float(cfg.get("lmm_analysis", {}).get("alpha_level", 0.05))
    
    # Get and validate formula
    formula = cfg.get("lmm_analysis", {}).get("formula", "amplitude ~ condition_code + baseline_centered")
    print(f"Using LMM formula: {formula}")
    
    results = []
    print("Running LMM analysis...")
    
    # Group by ROI and window
    for (roi, window), group_df in df.groupby(["roi", "window"]):
        print(f"  Analyzing {roi} - {window}")
        
        # Check if we have enough data
        n_subjects = group_df["subject"].nunique()
        if n_subjects < 3:
            print(f"    [SKIP] Insufficient subjects: {n_subjects}")
            continue
        
        # Check conditions
        conditions = group_df["condition"].unique()
        if len(conditions) < 2:
            print(f"    [SKIP] Insufficient conditions: {conditions}")
            continue
        
        try:
            # Prepare data for LMM
            lmm_df = group_df.copy()
            
            # MANDATORY baseline computation - no escape allowed
            if "baseline" not in lmm_df.columns:
                raise ValueError(f"Missing baseline data for {roi}-{window}. Baseline computation is mandatory.")
            
            # Check for missing baseline values
            missing_baseline = lmm_df["baseline"].isna().sum()
            if missing_baseline > 0:
                raise ValueError(f"Found {missing_baseline} missing baseline values for {roi}-{window}. All baseline values must be computed.")
            
            # Center baseline to improve interpretability and convergence
            lmm_df["baseline_centered"] = (
                lmm_df["baseline"] - lmm_df["baseline"].mean()
            )
            
            # Create dummy coding for condition (onTask = 0, offTask = 1)
            lmm_df["condition_code"] = (
                lmm_df["condition"] == "offTask"
            ).astype(int)
            
            # Run LMM with configurable formula and subject as random effect
            model = mixedlm(
                formula,
                lmm_df,
                groups=lmm_df["subject"],
            )
            result = model.fit(method=["lbfgs"])
            
            # Extract results for condition effect
            if "condition_code" in result.params:
                coef = result.params.get("condition_code", np.nan)
                se = result.bse.get("condition_code", np.nan)
                t_val = result.tvalues.get("condition_code", np.nan)
                p_val = result.pvalues.get("condition_code", np.nan)
                ci_lower, ci_upper = result.conf_int().loc["condition_code"]
            else:
                print(f"    [WARNING] 'condition_code' not found in formula results for {roi}-{window}")
                coef = se = t_val = p_val = ci_lower = ci_upper = np.nan

            # Baseline effect statistics
            base_coef = result.params.get("baseline_centered", np.nan)
            base_se = result.bse.get("baseline_centered", np.nan)
            base_t = result.tvalues.get("baseline_centered", np.nan)
            base_p = result.pvalues.get("baseline_centered", np.nan)
            if "baseline_centered" in result.params:
                base_ci_low, base_ci_up = result.conf_int().loc["baseline_centered"]
            else:
                base_ci_low, base_ci_up = (np.nan, np.nan)
            
            # Compute effect size (Cohen's d approximation)
            pooled_std = np.sqrt(
                (
                    lmm_df[lmm_df["condition"] == "onTask"]["amplitude"].var()
                    + lmm_df[lmm_df["condition"] == "offTask"]["amplitude"].var()
                )
                / 2
            )
            cohens_d = coef / pooled_std if pooled_std > 0 else np.nan
            
            # Compute group means for interpretation
            on_mean = lmm_df[lmm_df["condition"] == "onTask"]["amplitude"].mean()
            off_mean = lmm_df[lmm_df["condition"] == "offTask"]["amplitude"].mean()
            
            results.append({
                "roi": roi,
                "window": window,
                "window_start": group_df["window_start"].iloc[0],
                "window_end": group_df["window_end"].iloc[0],
                "formula": formula,
                "n_subjects": n_subjects,
                "n_observations": len(lmm_df),
                "on_task_mean": on_mean,
                "off_task_mean": off_mean,
                "difference": coef,  # off-task - on-task
                "std_error": se,
                "t_value": t_val,
                "p_value": p_val,
                "significant": p_val < alpha,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "cohens_d": cohens_d,
                "aic": result.aic,
                "bic": result.bic,
                # Baseline stats
                "baseline_coef": base_coef,
                "baseline_se": base_se,
                "baseline_t_value": base_t,
                "baseline_p_value": base_p,
                "baseline_ci_lower": base_ci_low,
                "baseline_ci_upper": base_ci_up,
            })
            
        except Exception as e:
            print(f"    [ERROR] LMM failed: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    print(f"Completed LMM analysis for {len(results_df)} ROI-window combinations")
    return results_df


def collect_temporal_probe_data(cfg: Dict, time_step: float, window_width: float) -> pd.DataFrame:
    """
    Collect probe data for temporal LMM analysis (sliding window approach).
    
    This creates data points for each time point using sliding windows
    to generate beta time-course plots.
    """
    proj = cfg.get("project", {})
    features_root = proj.get("features_root")
    if not features_root:
        raise ValueError("project.features_root must be set in config")
    
    rois = cfg.get("erp_rois", {})
    baseline_window = cfg.get("lmm_analysis", {}).get("baseline_window", [-0.1, 0.0])
    if not baseline_window:
        raise ValueError("baseline_window must be configured in lmm_analysis config. Cannot proceed without baseline.")
    baseline_window = tuple(baseline_window)
    
    # Validate baseline window
    if len(baseline_window) != 2:
        raise ValueError(f"baseline_window must have exactly 2 values [start, end], got {len(baseline_window)}")
    if baseline_window[0] >= baseline_window[1]:
        raise ValueError(f"baseline_window start ({baseline_window[0]}) must be < end ({baseline_window[1]})")
    
    # Get outlier detection parameters
    outlier_cfg = cfg.get("outlier_detection", {})
    z_threshold = float(outlier_cfg.get("epoch_z_threshold", 3.0))  # Use epoch threshold for probe-level outlier detection
    
    subjects_cfg = cfg.get("subjects", [])
    subjects = [str(s) for s in subjects_cfg]
    
    tasks_cfg = cfg.get("tasks", [])
    tasks = [str(t) for t in tasks_cfg]
    
    all_data = []

    print("Collecting temporal probe data for LMM analysis...")
    print(f"Time step: {time_step*1000:.0f}ms, Window width: {window_width*1000:.0f}ms")
    
    for subject in subjects:
        for task in tasks:
            # Find probe evoked files for this subject-task
            probe_paths = _find_probe_evokeds(features_root, subject, task)
            if not probe_paths:
                continue
            
            # Split by condition
            on_paths, off_paths = _split_by_label(probe_paths)
            
            # Process each condition
            for condition, paths in [("onTask", on_paths), ("offTask", off_paths)]:
                if not paths:
                    continue
                
                # Load one file to get info for ROI picks and time points
                if paths:
                    info_evoked = mne.read_evokeds(paths[0], condition=0, verbose=False)
                    times = info_evoked.times
                
                # Define time points for sliding windows
                time_points = np.arange(
                    times.min() + window_width/2,
                    times.max() - window_width/2,
                    time_step,
                )
                
                # Process each ROI
                for roi_name, roi_channels in rois.items():
                    picks = _roi_picks(info_evoked.info, list(roi_channels))
                    if picks.size == 0:
                        continue
                    
                    # Load all probes for this ROI-condition
                    probe_curves = []
                    probe_info = []
                    
                    for path in paths:
                        try:
                            times, roi_avg = _load_and_roi_average(path, picks)
                            probe_curves.append((times, roi_avg))
                            
                            # Extract probe information
                            _, probe_num, _ = _extract_probe_info(path)
                            probe_info.append(probe_num)
                        except Exception as e:
                            print(f"[WARN] Failed to load {path}: {e}")
                            continue
                    
                    if not probe_curves:
                        continue
                    
                    # Apply outlier detection at probe level
                    clean_curves, outlier_indices = _detect_outlier_probes(probe_curves, z_threshold)
                    
                    # Process each clean probe and time point
                    for i, (times, values) in enumerate(clean_curves):
                        # Get the original probe number (accounting for outlier removal)
                        original_idx = [j for j in range(len(probe_curves)) if j not in outlier_indices][i]
                        probe_num = probe_info[original_idx]
                        
                        # Process each time point
                        for time_point in time_points:
                            window_start = time_point - window_width/2
                            window_end = time_point + window_width/2
                            window = (window_start, window_end)
                            
                            # Compute amplitude for this window
                            amplitude = _compute_window_amplitude(times, values, window, baseline_window)
                            
                            if not np.isnan(amplitude):
                                # MANDATORY baseline computation - no escape allowed
                                if baseline_window is None:
                                    raise ValueError("Baseline window must be configured. baseline_window cannot be None.")
                                
                                baseline_value = _compute_window_amplitude(times, values, baseline_window, None)
                                if np.isnan(baseline_value):
                                    print(f"[WARN] Skipping probe {probe_num} due to invalid baseline computation")
                                    continue
                                
                                all_data.append({
                                    "subject": subject,
                                    "task": task,
                                    "probe": int(probe_num),
                                    "condition": condition,
                                    "roi": roi_name,
                                    "time_point": float(time_point),
                                    "amplitude": amplitude,
                                    "baseline": baseline_value,
                                })
    
    df = pd.DataFrame(all_data)
    if df.empty:
        print("No temporal data collected - DataFrame is empty")
        return df
    
    print(f"Collected {len(df)} temporal data points from {df['subject'].nunique()} subjects")
    return df


def run_temporal_lmm_analysis(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Run sliding window LMM analysis for beta time-course plots.
    
    This runs LMM at each time point to generate beta coefficient time-courses
    showing statistical effects over time (option B).
    """
    if not HAS_STATSMODELS:
        print("[ERROR] statsmodels not available. Cannot run temporal LMM analysis.")
        return pd.DataFrame()
    
    # Get temporal analysis parameters
    plotting_cfg = cfg.get("plotting", {})
    time_step = float(plotting_cfg.get("time_step", 0.01))
    window_width = float(plotting_cfg.get("window_width", 0.05))
    alpha = float(cfg.get("lmm_analysis", {}).get("alpha_level", 0.05))
    
    # Collect temporal data
    temporal_data = collect_temporal_probe_data(cfg, time_step, window_width)
    
    if temporal_data.empty:
        print("[ERROR] No temporal data collected for LMM analysis")
        return pd.DataFrame()
    
    # Get and validate formula
    formula = cfg.get("lmm_analysis", {}).get("formula", "amplitude ~ condition_code + baseline_centered")
    print(f"Using LMM formula for temporal analysis: {formula}")
    
    results = []
    print("Running temporal LMM analysis...")
    
    # Group by ROI and time point
    for (roi, time_point), group_df in temporal_data.groupby(["roi", "time_point"]):
        # Check if we have enough data
        n_subjects = group_df["subject"].nunique()
        if n_subjects < 3:
            continue
        
        # Check conditions
        conditions = group_df["condition"].unique()
        if len(conditions) < 2:
            continue
        
        try:
            # Prepare data for LMM
            lmm_df = group_df.copy()
            
            # MANDATORY baseline computation - no escape allowed
            if "baseline" not in lmm_df.columns:
                raise ValueError(f"Missing baseline data for {roi}-t{time_point:.3f}. Baseline computation is mandatory.")
            
            # Check for missing baseline values
            missing_baseline = lmm_df["baseline"].isna().sum()
            if missing_baseline > 0:
                raise ValueError(f"Found {missing_baseline} missing baseline values for {roi}-t{time_point:.3f}. All baseline values must be computed.")
            
            # Center baseline to improve interpretability and convergence
            lmm_df["baseline_centered"] = (
                lmm_df["baseline"] - lmm_df["baseline"].mean()
            )
            
            # Create dummy coding for condition (onTask = 0, offTask = 1)
            lmm_df["condition_code"] = (
                lmm_df["condition"] == "offTask"
            ).astype(int)

            # Run LMM with configurable formula and subject as random effect
            model = mixedlm(
                formula,
                lmm_df,
                groups=lmm_df["subject"],
            )
            result = model.fit(method=["lbfgs"])
            
            # Extract results for condition effect
            if "condition_code" in result.params:
                coef = result.params.get("condition_code", np.nan)
                se = result.bse.get("condition_code", np.nan)
                t_val = result.tvalues.get("condition_code", np.nan)
                p_val = result.pvalues.get("condition_code", np.nan)
                ci_lower, ci_upper = result.conf_int().loc["condition_code"]
            else:
                coef = se = t_val = p_val = ci_lower = ci_upper = np.nan
            
            results.append({
                "roi": roi,
                "time_point": time_point,
                "formula": formula,
                "n_subjects": n_subjects,
                "n_observations": len(lmm_df),
                "beta": coef,  # off-task - on-task effect
                "std_error": se,
                "t_value": t_val,
                "p_value": p_val,
                "significant": p_val < alpha,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })
            
        except Exception as e:
            # Silently continue for convergence issues in temporal analysis
            continue
    
    results_df = pd.DataFrame(results)
    print(f"Completed temporal LMM analysis for {len(results_df)} time points")
    return results_df


def save_lmm_results(
    results_df: pd.DataFrame, 
    data_df: pd.DataFrame, 
    cfg: Dict,
    temporal_results_df: Optional[pd.DataFrame] = None
) -> None:
    """Save LMM results and create visualization."""
    proj = cfg.get("project", {})
    results_root = proj.get("results_root", "results/ERPs_new")
    
    # Create LMM results directory
    lmm_dir = os.path.join(results_root, "lmm")
    os.makedirs(lmm_dir, exist_ok=True)
    
    # Save detailed results
    results_path = os.path.join(lmm_dir, "lmm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[OK] Saved LMM results: {results_path}")
    
    # Save raw data
    data_path = os.path.join(lmm_dir, "lmm_data.csv")
    data_df.to_csv(data_path, index=False)
    print(f"[OK] Saved LMM data: {data_path}")
    
    # Save temporal results if available
    if temporal_results_df is not None and not temporal_results_df.empty:
        temporal_path = os.path.join(lmm_dir, "lmm_temporal_results.csv")
        temporal_results_df.to_csv(temporal_path, index=False)
        print(f"[OK] Saved temporal LMM results: {temporal_path}")
        
        # Create temporal visualization
        _create_temporal_lmm_plots(temporal_results_df, lmm_dir, cfg)
    
    # Create visualization
    if not results_df.empty:
        _create_lmm_plots(results_df, lmm_dir)


def _create_lmm_plots(results_df: pd.DataFrame, output_dir: str) -> None:
    """Create visualization plots for LMM results."""
    
    # 1. Heatmap of p-values
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    pivot_df = results_df.pivot(
        index="roi", columns="window", values="p_value"
    )
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdYlBu_r",
                center=0.05, vmin=0, vmax=0.1,
                cbar_kws={"label": "p-value"})
    plt.title(
        "LMM p-values: On-task vs Off-task Comparison\\n(Red = significant)"
    )
    plt.xlabel("Time Window")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lmm_pvalues_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    
    # 2. Effect sizes heatmap
    plt.figure(figsize=(12, 8))
    
    pivot_effect = results_df.pivot(
        index="roi", columns="window", values="cohens_d"
    )
    sns.heatmap(pivot_effect, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                cbar_kws={"label": "Cohen's d"})
    plt.title(
        "Effect Sizes (Cohen's d): Off-task - On-task\\n"
        "(Red = off-task > on-task)"
    )
    plt.xlabel("Time Window")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lmm_effect_sizes_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    
    # 3. Significance summary
    plt.figure(figsize=(10, 6))
    
    sig_summary = (
        results_df.groupby(["roi", "window"])["significant"]
        .sum()
        .reset_index()
    )
    sig_pivot = sig_summary.pivot(
        index="roi", columns="window", values="significant"
    )
    
    sns.heatmap(sig_pivot, annot=True, fmt="d", cmap="Reds",
                cbar_kws={"label": "Significant"})
    plt.title("Significant Effects Summary")
    plt.xlabel("Time Window")
    plt.ylabel("ROI")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lmm_significance_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    
    print(f"[OK] Saved LMM plots in: {output_dir}")


def _create_temporal_lmm_plots(temporal_results_df: pd.DataFrame, output_dir: str, cfg: Dict) -> None:
    """Create visualization plots for temporal LMM results (beta time-courses)."""
    plotting_cfg = cfg.get("plotting", {})
    smooth_betas = plotting_cfg.get("smooth_betas", True)
    smoothing_window = plotting_cfg.get("smoothing_window", 0.02)
    alpha = float(cfg.get("lmm_analysis", {}).get("alpha_level", 0.05))
    
    # Get unique ROIs
    rois = temporal_results_df["roi"].unique()
    
    # Create subplot figure for all ROIs
    n_rois = len(rois)
    fig, axes = plt.subplots(n_rois, 1, figsize=(12, 4*n_rois), sharex=True)
    if n_rois == 1:
        axes = [axes]
    
    for i, roi in enumerate(rois):
        roi_data = temporal_results_df[temporal_results_df["roi"] == roi].copy()
        roi_data = roi_data.sort_values("time_point")
        
        times = roi_data["time_point"].values
        betas = roi_data["beta"].values
        ci_lower = roi_data["ci_lower"].values
        ci_upper = roi_data["ci_upper"].values
        p_values = roi_data["p_value"].values
        
        # Apply smoothing if requested
        if smooth_betas and len(times) > 1:
            # Convert smoothing window from seconds to samples
            time_step = np.mean(np.diff(times))
            sigma = smoothing_window / time_step
            betas = gaussian_filter1d(betas, sigma=sigma)
            ci_lower = gaussian_filter1d(ci_lower, sigma=sigma)
            ci_upper = gaussian_filter1d(ci_upper, sigma=sigma)
        
        ax = axes[i]
        
        # Plot beta time-course with confidence intervals
        ax.plot(times, betas, 'b-', linewidth=2, label='Off-task - On-task effect')
        ax.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='blue')
        
        # Mark significant time points
        sig_mask = p_values < alpha
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            sig_betas = betas[sig_mask]
            ax.scatter(sig_times, sig_betas, c='red', s=20, alpha=0.7, 
                      label=f'Significant (p < {alpha})', zorder=5)
        
        # Add reference lines
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_ylabel('Beta coefficient (µV)')
        ax.set_title(f'ROI: {roi} - Beta Time-course')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on confidence intervals
        y_min = np.min(ci_lower) * 1.1
        y_max = np.max(ci_upper) * 1.1
        ax.set_ylim(y_min, y_max)
    
    # Set x-axis label only for bottom subplot
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lmm_beta_timecourse.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    
    # Create individual plots for each ROI
    for roi in rois:
        roi_data = temporal_results_df[temporal_results_df["roi"] == roi].copy()
        roi_data = roi_data.sort_values("time_point")
        
        times = roi_data["time_point"].values
        betas = roi_data["beta"].values
        ci_lower = roi_data["ci_lower"].values
        ci_upper = roi_data["ci_upper"].values
        p_values = roi_data["p_value"].values
        
        # Apply smoothing if requested
        if smooth_betas and len(times) > 1:
            time_step = np.mean(np.diff(times))
            sigma = smoothing_window / time_step
            betas = gaussian_filter1d(betas, sigma=sigma)
            ci_lower = gaussian_filter1d(ci_lower, sigma=sigma)
            ci_upper = gaussian_filter1d(ci_upper, sigma=sigma)
        
        plt.figure(figsize=(10, 6))
        
        # Plot beta time-course with confidence intervals
        plt.plot(times, betas, 'b-', linewidth=3, label='Off-task - On-task effect')
        plt.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
        
        # Mark significant time points
        sig_mask = p_values < alpha
        if np.any(sig_mask):
            sig_times = times[sig_mask]
            sig_betas = betas[sig_mask]
            plt.scatter(sig_times, sig_betas, c='red', s=30, alpha=0.8, 
                       label=f'Significant (p < {alpha})', zorder=5)
            
            # Highlight significant clusters as background
            if len(sig_times) > 1:
                # Find continuous clusters
                time_diffs = np.diff(sig_times)
                cluster_breaks = np.where(time_diffs > 2 * np.mean(np.diff(times)))[0]
                
                if len(cluster_breaks) == 0:
                    # Single cluster
                    plt.axvspan(sig_times[0], sig_times[-1], alpha=0.1, color='red')
                else:
                    # Multiple clusters
                    start_idx = 0
                    for break_idx in cluster_breaks:
                        plt.axvspan(sig_times[start_idx], sig_times[break_idx], 
                                   alpha=0.1, color='red')
                        start_idx = break_idx + 1
                    # Last cluster
                    plt.axvspan(sig_times[start_idx], sig_times[-1], 
                               alpha=0.1, color='red')
        
        # Add reference lines
        plt.axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        plt.axvline(0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # Formatting
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Beta coefficient (µV)', fontsize=12)
        plt.title(f'ROI {roi}: Off-task vs On-task Beta Time-course', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        y_min = np.min(ci_lower) * 1.1
        y_max = np.max(ci_upper) * 1.1
        plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"lmm_beta_timecourse_roi-{roi}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    
    print(f"[OK] Saved temporal LMM plots in: {output_dir}")


def run_julia_unfold_analysis(cfg: Dict) -> pd.DataFrame:
    """
    Run Unfold.jl analysis with continuous temporal data.
    
    This function collects continuous temporal data from preprocessed epochs
    and runs the full Unfold.jl pipeline.
    """
    if not HAS_JULIA_LMM:
        print("[ERROR] Julia LMM bridge not available for Unfold analysis")
        return pd.DataFrame()
    
    # Import the bridge
    from unfold_bridge import JuliaLMMBridge
    
    # Initialize bridge
    bridge = JuliaLMMBridge()
    
    # Check Julia environment
    if not bridge.check_julia_environment():
        print("[ERROR] Julia environment not ready for Unfold analysis")
        return pd.DataFrame()
    
    print("🚀 Running Unfold.jl analysis with continuous temporal data...")
    
    try:
        # Collect continuous temporal data from preprocessed epochs
        print("📊 Collecting continuous temporal data...")
        continuous_data = bridge.collect_continuous_temporal_data(cfg)
        
        if continuous_data.empty:
            print("[ERROR] No continuous temporal data collected")
            return pd.DataFrame()
        
        # Prepare data for Julia
        print("🔧 Preparing data for Julia Unfold analysis...")
        data_path = bridge.prepare_julia_data(continuous_data, cfg)
        
        # Run Julia analysis
        print("🚀 Running Julia Unfold analysis...")
        results_df = bridge.run_julia_lmm(data_path, cfg)
        
        # Convert results format
        final_results = bridge.convert_julia_results(results_df)
        
        # Cleanup
        bridge.cleanup_temp_workspace()
        
        return final_results
        
    except Exception as e:
        print(f"[ERROR] Unfold.jl analysis failed: {e}")
        bridge.cleanup_temp_workspace()
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Linear Mixed Model analysis on probe evoked responses"
        )
    )
    parser.add_argument(
        "--config",
        default="./ERPs_new/config.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--use-unfold",
        action="store_true",
        help="Use Unfold.jl with continuous temporal data"
    )
    args = parser.parse_args()
    
    cfg = load_yaml_config(args.config)
    
    # Check if LMM analysis is enabled
    if not cfg.get("lmm_analysis", {}).get("enabled", False):
        print("[INFO] LMM analysis is disabled in config")
        return
    
    if not HAS_STATSMODELS:
        print("[ERROR] statsmodels package required for LMM analysis")
        print("Install with: pip install statsmodels")
        return
    
    # Check which plotting types are requested
    plotting_cfg = cfg.get("plotting", {})
    plot_type = plotting_cfg.get("plot_type", "classic_erp")
    
    # Check if Unfold should be used (either via flag or config)
    use_julia_config = cfg.get("lmm_analysis", {}).get("use_julia", False)
    
    if args.use_unfold or use_julia_config:
        print("🔬 Using Unfold.jl with continuous temporal data...")
        if use_julia_config and not args.use_unfold:
            print("  ℹ️  Triggered by config: lmm_analysis.use_julia = true")
        
        results_df = run_julia_unfold_analysis(cfg)
        
        if results_df.empty:
            print("[ERROR] Unfold.jl analysis failed")
            return
        
        # Save results (no windowed data for Unfold)
        data_df = pd.DataFrame()  # Empty dataframe for windowed data
        temporal_results_df = results_df  # Unfold results are temporal
        
        save_lmm_results(results_df, data_df, cfg, temporal_results_df)
        
        print("🎉 Unfold.jl analysis completed successfully!")
        return
    
    # For traditional analysis, use Python LMM
    print("🐍 Using Python for LMM analysis (windowed + temporal)...")
    
    # Collect probe evoked data for windowed analysis (always needed)
    data_df = collect_probe_data_for_lmm(cfg)
    
    if data_df.empty:
        print("[ERROR] No data collected for LMM analysis")
        return
    
    # Run windowed LMM analysis using Python
    print("Running windowed LMM analysis with Python...")
    results_df = run_lmm_analysis(data_df, cfg)
    
    if results_df.empty:
        print("[ERROR] No windowed LMM results generated")
        return
    
    # Run temporal LMM analysis if beta time-course plots are requested
    temporal_results_df = None
    if plot_type in ["beta_timecourse", "both"]:
        print("\\nRunning temporal LMM analysis with Python...")
        temporal_results_df = run_temporal_lmm_analysis(data_df, cfg)
        
        if temporal_results_df.empty:
            print("[WARNING] No temporal LMM results generated")
        else:
            print(f"Generated temporal results for {temporal_results_df['roi'].nunique()} ROIs")
    
    # Save results and create plots
    save_lmm_results(results_df, data_df, cfg, temporal_results_df)
    
    # Create combined ERP + Beta plots if temporal results are available
    if temporal_results_df is not None and not temporal_results_df.empty:
        plotting_cfg = cfg.get("plotting", {})
        create_combined_plots = plotting_cfg.get("create_combined_erp_beta", True)
        
        if create_combined_plots:
            print("\\n🎨 Creating combined ERP + Beta time-course plots...")
            try:
                from combined_erp_beta_plots import run_combined_erp_beta_analysis
                run_combined_erp_beta_analysis(cfg)
            except ImportError:
                print("⚠️  Combined plotting module not available")
            except Exception as e:
                print(f"⚠️  Combined plotting failed: {e}")
    
    # Print summary
    sig_results = results_df[results_df["significant"]]
    print("\\nSUMMARY:")
    print(f"Total tests: {len(results_df)}")
    print(f"Significant effects: {len(sig_results)}")
    
    if len(sig_results) > 0:
        print("\\nSignificant effects:")
        for _, row in sig_results.iterrows():
            print(
                f"  {row['roi']} - {row['window']}: "
                f"p={row['p_value']:.3f}, d={row['cohens_d']:.2f}"
            )


if __name__ == "__main__":
    main()