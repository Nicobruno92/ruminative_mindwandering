import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import stats
from scipy.ndimage import gaussian_filter1d

try:
    from helpers import load_yaml_config, build_evoked_dir
except Exception:
    _this_dir = os.path.dirname(__file__)
    if _this_dir not in sys.path:
        sys.path.append(_this_dir)
    from helpers import load_yaml_config, build_evoked_dir


def _load_lmm_results(cfg: Dict) -> Optional[pd.DataFrame]:
    """Load LMM results if available."""
    try:
        proj = cfg.get("project", {})
        results_root = proj.get("results_root", "results/ERPs_new")
        lmm_path = os.path.join(results_root, "lmm", "lmm_results.csv")
        
        if os.path.exists(lmm_path):
            return pd.read_csv(lmm_path)
        else:
            print(f"[INFO] LMM results not found at {lmm_path}")
            return None
    except Exception as e:
        print(f"[WARNING] Could not load LMM results: {e}")
        return None


def _load_temporal_lmm_results(cfg: Dict) -> Optional[pd.DataFrame]:
    """Load temporal LMM results if available."""
    try:
        proj = cfg.get("project", {})
        results_root = proj.get("results_root", "results/ERPs_new")
        temporal_lmm_path = os.path.join(results_root, "lmm", "lmm_temporal_results.csv")
        
        if os.path.exists(temporal_lmm_path):
            return pd.read_csv(temporal_lmm_path)
        else:
            print(f"[INFO] Temporal LMM results not found at {temporal_lmm_path}")
            return None
    except Exception as e:
        print(f"[WARNING] Could not load temporal LMM results: {e}")
        return None


def _get_significant_windows(lmm_results: pd.DataFrame, roi_name: str, alpha: float = 0.05) -> List[Tuple[float, float]]:
    """Extract significant time windows for a given ROI from LMM results."""
    if lmm_results is None or lmm_results.empty:
        return []
    
    # Filter for this ROI and significant results
    roi_results = lmm_results[
        (lmm_results["roi"] == roi_name) & 
        (lmm_results["p_value"] < alpha)
    ]
    
    # Extract time windows
    windows = []
    for _, row in roi_results.iterrows():
        windows.append((row["window_start"], row["window_end"]))
    
    return windows


def _find_probe_evokeds(features_root: str, subject: str, task: str) -> List[str]:
    evoked_dir = build_evoked_dir(features_root, subject)
    if not os.path.isdir(evoked_dir):
        return []
    files = []
    for name in sorted(os.listdir(evoked_dir)):
        if name.endswith("_ave.fif") and f"sub-{subject}_task-{task}_desc-probe-" in name:
            files.append(os.path.join(evoked_dir, name))
    return files


def _split_by_label(paths: List[str]) -> Tuple[List[str], List[str]]:
    on_paths, off_paths = [], []
    for p in paths:
        base = os.path.basename(p)
        if "_onTask_" in base or base.endswith("_onTask_ave.fif"):
            on_paths.append(p)
        elif "_offTask_" in base or base.endswith("_offTask_ave.fif"):
            off_paths.append(p)
    return on_paths, off_paths


def _roi_picks(info: mne.Info, roi_channels: List[str]) -> np.ndarray:
    picks = [ch for ch in roi_channels if ch in info["ch_names"]]
    return mne.pick_channels(info["ch_names"], include=picks, exclude=[])


def _load_and_roi_average(path: str, roi_picks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evk = mne.read_evokeds(path, condition=0, verbose=False)
    data = evk.data[roi_picks, :]
    roi_avg = data.mean(axis=0)
    
    # Debug: print data range to check scaling
    # print(f"Debug - {os.path.basename(path)}: data range [{data.min():.2e}, {data.max():.2e}], ROI avg range [{roi_avg.min():.2e}, {roi_avg.max():.2e}]")
    
    # Convert to microvolts if data seems to be in volts (very small values)
    if np.abs(roi_avg).max() < 1e-3:
        # print(f"  -> Converting to microvolts (multiplying by 1e6)")
        roi_avg = roi_avg * 1e6
    
    return evk.times, roi_avg


def _ensure_same_timebase(curves: List[Tuple[np.ndarray, np.ndarray]], rtol: float = 1e-6, atol: float = 1e-8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Ensure all curves share the same time vector by keeping only those
    matching the most common timebase. This avoids shape errors when stacking.

    Parameters
    ----------
    curves : list of (times, values)
        Input curves that may have slightly different time vectors.
    rtol, atol : float
        Tolerances for time vector comparison (np.allclose).

    Returns
    -------
    list of (times, values)
        Filtered curves sharing the dominant time vector. If multiple
        timebases exist, curves not matching the dominant one are dropped.
    """
    if not curves:
        return []
    # Build simple keys based on (length, start, end) to find dominant timebase
    keys: List[Tuple[int, float, float]] = []
    for t, _ in curves:
        if t is None or len(t) == 0:
            keys.append((0, 0.0, 0.0))
        else:
            keys.append((len(t), float(t[0]), float(t[-1])))
    # Choose the most frequent key
    from collections import Counter
    key_counts = Counter(keys)
    dominant_key, _ = key_counts.most_common(1)[0]
    # Reference time for the dominant key
    ref_time = None
    for (t, _), k in zip(curves, keys):
        if k == dominant_key:
            ref_time = t
            break
    if ref_time is None:
        return curves
    # Keep only curves matching reference time within tolerance
    filtered: List[Tuple[np.ndarray, np.ndarray]] = []
    for (t, v) in curves:
        if len(t) == len(ref_time) and np.allclose(t, ref_time, rtol=rtol, atol=atol):
            filtered.append((t, v))
    # If filtering removes everything (shouldn't), fall back to original
    return filtered if filtered else curves


def _plot_subject(on_curves: List[Tuple[np.ndarray, np.ndarray]],
                  off_curves: List[Tuple[np.ndarray, np.ndarray]],
                  title: str, out_path: str,
                  significant_windows: List[Tuple[float, float]] = None) -> None:
    # Ensure same time base within each condition before stacking
    on_curves = _ensure_same_timebase(on_curves)
    off_curves = _ensure_same_timebase(off_curves)
    plt.figure(figsize=(8, 4))
    # Each curve is (times, values)
    for t, v in on_curves:
        plt.plot(t, v, color="#2ca02c", alpha=0.4, linewidth=1.0)
    for t, v in off_curves:
        plt.plot(t, v, color="#d62728", alpha=0.4, linewidth=1.0)

    # Grand means per subject, if any
    if on_curves:
        t0, on_mean = _safe_vstack_mean(on_curves)
        if len(on_mean) > 0:
            plt.plot(t0, on_mean, color="#1f7a1f", linewidth=2.0,
                     label="on-task mean")
    if off_curves:
        t1, off_mean = _safe_vstack_mean(off_curves)
        if len(off_mean) > 0:
            plt.plot(t1, off_mean, color="#8b0000", linewidth=2.0,
                     label="off-task mean")

    # Add significant windows as gray rectangles
    if significant_windows:
        for i, (win_start, win_end) in enumerate(significant_windows):
            label = 'Significant window' if i == 0 else ""
            plt.axvspan(win_start, win_end, alpha=0.2, color='gray',
                        label=label)

    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_grand_average_plotly(subject_mean_on: List[Tuple[np.ndarray, np.ndarray]], 
                               subject_mean_off: List[Tuple[np.ndarray, np.ndarray]], 
                               title: str, out_path: str,
                               significant_windows: List[Tuple[float, float]] = None) -> None:
    """Create interactive plotly version of group ERP plot."""
    # Ensure same time base for means before stacking
    subject_mean_on = _ensure_same_timebase(subject_mean_on)
    subject_mean_off = _ensure_same_timebase(subject_mean_off)
    fig = go.Figure()
    
    # Add individual subject traces (transparent)
    if subject_mean_on:
        for i, (t, v) in enumerate(subject_mean_on):
            fig.add_trace(go.Scatter(
                x=t, y=v, mode='lines',
                line=dict(color='rgba(44, 160, 44, 0.15)', width=1),
                name=f'on-task subj {i+1}',
                showlegend=False,
                hovertemplate='Subject %{customdata}<br>Time: %{x:.3f}s<br>Amplitude: %{y:.2f}µV<extra></extra>',
                customdata=[f'on-task {i+1}'] * len(t)
            ))
    
    if subject_mean_off:
        for i, (t, v) in enumerate(subject_mean_off):
            fig.add_trace(go.Scatter(
                x=t, y=v, mode='lines',
                line=dict(color='rgba(214, 39, 40, 0.15)', width=1),
                name=f'off-task subj {i+1}',
                showlegend=False,
                hovertemplate='Subject %{customdata}<br>Time: %{x:.3f}s<br>Amplitude: %{y:.2f}µV<extra></extra>',
                customdata=[f'off-task {i+1}'] * len(t)
            ))
    
    # Compute and add grand averages with error bands
    if subject_mean_on:
        t0, aligned_on = _align_temporal_arrays(subject_mean_on)
        if aligned_on:
            on_data = np.array(aligned_on)
            on_mean = on_data.mean(axis=0)
            on_sem = on_data.std(axis=0) / np.sqrt(on_data.shape[0])
        else:
            t0, on_mean, on_sem = np.array([]), np.array([]), np.array([])
        
        # Add traces only if we have data
        if len(on_mean) > 0 and len(t0) > 0:
            # Error band
            fig.add_trace(go.Scatter(
                x=np.concatenate([t0, t0[::-1]]),
                y=np.concatenate([on_mean + on_sem, (on_mean - on_sem)[::-1]]),
                fill='toself',
                fillcolor='rgba(31, 122, 31, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='on-task SEM',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=t0, y=on_mean, mode='lines',
                line=dict(color='rgb(31, 122, 31)', width=3),
                name=f'on-task mean (n={len(aligned_on)})',
                hovertemplate='On-task mean<br>Time: %{x:.3f}s<br>Amplitude: %{y:.2f}µV<extra></extra>'
            ))
    
    if subject_mean_off:
        t1, aligned_off = _align_temporal_arrays(subject_mean_off)
        if aligned_off:
            off_data = np.array(aligned_off)
            off_mean = off_data.mean(axis=0)
            off_sem = off_data.std(axis=0) / np.sqrt(off_data.shape[0])
        else:
            t1, off_mean, off_sem = np.array([]), np.array([]), np.array([])
        
        # Add traces only if we have data
        if len(off_mean) > 0 and len(t1) > 0:
            # Error band
            fig.add_trace(go.Scatter(
                x=np.concatenate([t1, t1[::-1]]),
                y=np.concatenate([off_mean + off_sem, (off_mean - off_sem)[::-1]]),
                fill='toself',
                fillcolor='rgba(139, 0, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='off-task SEM',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Mean line
            fig.add_trace(go.Scatter(
                x=t1, y=off_mean, mode='lines',
                line=dict(color='rgb(139, 0, 0)', width=3),
                name=f'off-task mean (n={len(aligned_off)})',
                hovertemplate='Off-task mean<br>Time: %{x:.3f}s<br>Amplitude: %{y:.2f}µV<extra></extra>'
            ))
    
    # Add reference lines
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)
    
    # Add significant windows as gray rectangles
    if significant_windows:
        for i, (win_start, win_end) in enumerate(significant_windows):
            fig.add_vrect(
                x0=win_start, x1=win_end,
                fillcolor="gray", opacity=0.2,
                layer="below", line_width=0,
                annotation_text="Significant" if i == 0 else "",
                annotation_position="top left"
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        template="plotly_white",
        hovermode='x unified',
        width=900,
        height=500
    )
    
    # Save as HTML
    fig.write_html(out_path.replace('.png', '_interactive.html'))
    
    # Also save static matplotlib version
    _plot_grand_average_matplotlib(subject_mean_on, subject_mean_off, title, 
                                   out_path, significant_windows)


def _plot_grand_average_matplotlib(subject_mean_on: List[Tuple[np.ndarray, np.ndarray]], 
                                   subject_mean_off: List[Tuple[np.ndarray, np.ndarray]], 
                                   title: str, out_path: str,
                                   significant_windows: List[Tuple[float, float]] = None) -> None:
    """Create matplotlib version of group ERP plot."""
    # Ensure same time base for means before stacking
    subject_mean_on = _ensure_same_timebase(subject_mean_on)
    subject_mean_off = _ensure_same_timebase(subject_mean_off)
    plt.figure(figsize=(10, 6))
    
    # Plot individual subject ERPs as transparent lines
    if subject_mean_on:
        for t, v in subject_mean_on:
            plt.plot(t, v, color="#2ca02c", alpha=0.15, linewidth=0.8)
    if subject_mean_off:
        for t, v in subject_mean_off:
            plt.plot(t, v, color="#d62728", alpha=0.15, linewidth=0.8)
    
    # Compute and plot grand averages with error shadows
    if subject_mean_on:
        t0, aligned_on = _align_temporal_arrays(subject_mean_on)
        if aligned_on and len(t0) > 0:
            on_data = np.array(aligned_on)
            on_mean = on_data.mean(axis=0)
            on_sem = on_data.std(axis=0) / np.sqrt(on_data.shape[0])
            plt.plot(t0, on_mean, color="#1f7a1f", linewidth=3.0, label=f"on-task (n={len(aligned_on)})")
            plt.fill_between(t0, on_mean - on_sem, on_mean + on_sem, color="#1f7a1f", alpha=0.3)
    
    if subject_mean_off:
        t1, aligned_off = _align_temporal_arrays(subject_mean_off)
        if aligned_off and len(t1) > 0:
            off_data = np.array(aligned_off)
            off_mean = off_data.mean(axis=0)
            off_sem = off_data.std(axis=0) / np.sqrt(off_data.shape[0])
            plt.plot(t1, off_mean, color="#8b0000", linewidth=3.0, label=f"off-task (n={len(aligned_off)})")
            plt.fill_between(t1, off_mean - off_sem, off_mean + off_sem, color="#8b0000", alpha=0.3)

    # Add significant windows as gray rectangles
    if significant_windows:
        for i, (win_start, win_end) in enumerate(significant_windows):
            label = 'Significant window' if i == 0 else ""
            plt.axvspan(win_start, win_end, alpha=0.2, color='gray',
                        label=label)

    plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.axhline(0.0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude (µV)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _sequential_outlier_detection_epochs(epochs, reject_ptp=200e-6, reject_amp=150e-6, z_threshold=4.0):
    """
    Sequential outlier detection using multiple criteria (more conservative approach).
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object to clean
    reject_ptp : float
        Peak-to-peak rejection threshold in Volts
    reject_amp : float
        Amplitude rejection threshold in Volts  
    z_threshold : float
        Z-score threshold for variance-based outlier detection
        
    Returns
    -------
    clean_epochs : mne.Epochs
        Epochs with outliers removed
    n_removed_ptp : int
        Number of epochs removed by peak-to-peak criterion
    n_removed_amp : int
        Number of epochs removed by amplitude criterion
    n_removed_var : int
        Number of epochs removed by variance criterion
    """
    epochs_temp = epochs.copy()
    n_initial = len(epochs_temp)
    
    # Paso 1: Peak-to-peak (más agresivo)
    reject_ptp_dict = {'eeg': reject_ptp}
    epochs_temp.drop_bad(reject=reject_ptp_dict)
    n_after_ptp = len(epochs_temp)
    n_removed_ptp = n_initial - n_after_ptp
    
    # Paso 2: Amplitud
    reject_amp_dict = {'eeg': reject_amp}
    epochs_temp.drop_bad(reject=reject_amp_dict)
    n_after_amp = len(epochs_temp)
    n_removed_amp = n_after_ptp - n_after_amp
    
    # Paso 3: Z-score manual para casos extremos (variance-based)
    if len(epochs_temp) > 2:  # Need at least 3 epochs for z-score calculation
        data = epochs_temp.get_data()
        epoch_vars = np.var(data, axis=(1, 2))  # Variance across channels and time
        z_vars = np.abs(stats.zscore(epoch_vars))
        bad_variance = z_vars > z_threshold
        
        if np.any(bad_variance):
            epochs_final = epochs_temp.drop(bad_variance)
            n_removed_var = np.sum(bad_variance)
        else:
            epochs_final = epochs_temp
            n_removed_var = 0
    else:
        epochs_final = epochs_temp
        n_removed_var = 0
    
    return epochs_final, n_removed_ptp, n_removed_amp, n_removed_var


def _sequential_outlier_detection_curves(curves: List[Tuple[np.ndarray, np.ndarray]], 
                                       reject_ptp: float = 200e-6, reject_amp: float = 150e-6, 
                                       z_threshold: float = 4.0) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Sequential outlier detection for curve data (adapted from epochs-based method).
    
    Parameters
    ----------
    curves : list of tuples
        List of (times, amplitudes) for each probe epoch
    reject_ptp : float
        Peak-to-peak rejection threshold in Volts
    reject_amp : float
        Amplitude rejection threshold in Volts
    z_threshold : float
        Z-score threshold for variance-based outlier detection
        
    Returns
    -------
    clean_curves : list
        Curves with outliers removed
    outlier_indices : list
        Indices of removed outlier epochs
    """
    if len(curves) < 3:
        return curves, []

    # Initialize mask for keeping epochs
    keep_mask = np.ones(len(curves), dtype=bool)

    # Check if data is in microvolts (large values) vs volts (small values)
    # If values are in microvolts range, convert thresholds accordingly
    sample_values = curves[0][1] if curves else np.array([0])
    data_range = np.abs(sample_values).max() if len(sample_values) > 0 else 0
    
    if data_range > 1e-3:  # Data is likely in microvolts
        # Convert thresholds from volts to microvolts
        reject_ptp_uv = reject_ptp * 1e6
        reject_amp_uv = reject_amp * 1e6
        # Data is in microvolts, convert thresholds accordingly
    else:
        # Data is in volts, use thresholds as-is
        reject_ptp_uv = reject_ptp
        reject_amp_uv = reject_amp

    # Paso 1: Peak-to-peak (más agresivo) computed per-curve to handle variable lengths
    ptp_values = np.array([float(np.ptp(v)) for _, v in curves])
    ptp_outliers = ptp_values > float(reject_ptp_uv)
    n_ptp_outliers = np.sum(ptp_outliers)
    keep_mask[ptp_outliers] = False

    # Paso 2: Amplitud máxima
    max_amps = np.array([float(np.max(np.abs(v))) for _, v in curves])
    amp_outliers = max_amps > float(reject_amp_uv)
    n_amp_outliers = np.sum(amp_outliers)
    keep_mask[amp_outliers] = False
    
    # Debug info for troubleshooting (uncomment if needed)
    # if n_ptp_outliers > 0 or n_amp_outliers > 0:
    #     print(f"[DEBUG] PTP outliers: {n_ptp_outliers}/{len(curves)} (threshold={reject_ptp_uv:.1f}), "
    #           f"AMP outliers: {n_amp_outliers}/{len(curves)} (threshold={reject_amp_uv:.1f})")
    #     if n_ptp_outliers > 0:
    #         print(f"[DEBUG] PTP values: {ptp_values}")
    #     if n_amp_outliers > 0:
    #         print(f"[DEBUG] MAX AMP values: {max_amps}")

    # Paso 3: Z-score para casos extremos (variance-based)
    remaining_indices = np.where(keep_mask)[0]
    if remaining_indices.size > 2:  # Need at least 3 epochs for z-score calculation
        epoch_vars = np.array([float(np.var(curves[i][1])) for i in remaining_indices])
        if epoch_vars.size > 0 and np.nanstd(epoch_vars) > 0:
            z_vars = np.abs(stats.zscore(epoch_vars))
            bad_variance = z_vars > float(z_threshold)
            if np.any(bad_variance):
                var_outlier_indices = remaining_indices[bad_variance]
                keep_mask[var_outlier_indices] = False

    # Convert back to curve format
    outlier_indices = np.where(~keep_mask)[0].tolist()
    clean_curves = [curves[i] for i in range(len(curves)) if keep_mask[i]]

    return clean_curves, outlier_indices


def _detect_outlier_epochs(curves: List[Tuple[np.ndarray, np.ndarray]], z_threshold: float = 3.0) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    """Remove outlier epochs based on z-score of mean amplitude."""
    if len(curves) < 3:  # Need at least 3 epochs for meaningful outlier detection
        return curves, []
    
    # Compute mean amplitude across time for each epoch
    mean_amps = [np.mean(np.abs(v)) for _, v in curves]
    z_scores = np.abs(stats.zscore(mean_amps))
    
    # Keep epochs with z-score below threshold
    outlier_indices = np.where(z_scores > z_threshold)[0].tolist()
    clean_curves = [curves[i] for i in range(len(curves)) if i not in outlier_indices]
    
    return clean_curves, outlier_indices


def _aggregate_subject_condition(curves: List[Tuple[np.ndarray, np.ndarray]], subject: str, condition: str, 
                                z_threshold: float = 3.0, use_sequential: bool = False,
                                reject_ptp: float = 200e-6, reject_amp: float = 150e-6) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Aggregate epochs for one subject-condition, removing outliers."""
    if not curves:
        return None, None, {"n_epochs_total": 0, "n_epochs_clean": 0, "outlier_indices": []}
    
    # Remove outlier epochs
    if use_sequential and len(curves) >= 3:
        # Use sequential multi-criteria outlier detection
        clean_curves, outlier_indices = _sequential_outlier_detection_curves(
            curves, reject_ptp, reject_amp, z_threshold
        )
        outlier_method = "sequential"
    else:
        # Use simple z-score based detection
        clean_curves, outlier_indices = _detect_outlier_epochs(curves, z_threshold)
        outlier_method = "zscore"
    
    qa_info = {
        "subject": subject,
        "condition": condition,
        "n_epochs_total": len(curves),
        "n_epochs_clean": len(clean_curves),
        "outlier_indices": outlier_indices,
        "outlier_ratio": len(outlier_indices) / len(curves) if curves else 0.0,
        "outlier_method": outlier_method
    }
    
    if not clean_curves:
        print(f"[WARN] All epochs were outliers for {subject} {condition}")
        return None, None, qa_info
    
    # Ensure same time base before stacking
    filtered_curves = _ensure_same_timebase(clean_curves)
    if len(filtered_curves) != len(clean_curves):
        qa_info["n_dropped_timebase_mismatch"] = len(clean_curves) - len(filtered_curves)
    else:
        qa_info["n_dropped_timebase_mismatch"] = 0

    if not filtered_curves:
        print(f"[WARN] All clean epochs dropped due to timebase mismatch for {subject} {condition}")
        return None, None, qa_info

    # Compute subject-condition average
    times, values = _safe_vstack_mean(filtered_curves)
    
    return times, values, qa_info


def process_subject_task(cfg: Dict, subject: str, task: str,
                         lmm_results: Optional[pd.DataFrame] = None) -> Dict:
    """Process subject-task and return QA info for all ROIs."""
    proj = cfg.get("project", {})
    features_root = proj.get("features_root")
    results_root = proj.get("results_root", os.path.join(features_root, "figs"))
    if not features_root:
        raise ValueError("project.features_root must be set in config")

    rois = cfg.get("erp_rois", {})
    outlier_cfg = cfg.get("outlier_detection", {})
    z_threshold = float(outlier_cfg.get("epoch_z_threshold", 3.0))
    use_sequential = outlier_cfg.get("use_sequential_detection", False)
    reject_ptp = float(outlier_cfg.get("reject_ptp", 200e-6))
    reject_amp = float(outlier_cfg.get("reject_amp", 150e-6))
    
    print(f"[INFO] Outlier detection method: {'Sequential (multi-criteria)' if use_sequential else 'Z-score only'}")
    if use_sequential:
        print(f"[INFO] Sequential thresholds: PTP={reject_ptp*1e6:.1f}µV, AMP={reject_amp*1e6:.1f}µV, Z-score={z_threshold}")
    else:
        print(f"[INFO] Z-score threshold: {z_threshold}")
    
    # Save individual participant figures under participants/sub-XX/figures/ (BIDS-like)
    out_dir = os.path.join(results_root, "participants", f"sub-{subject}", "figures")
    os.makedirs(out_dir, exist_ok=True)

    paths = _find_probe_evokeds(features_root, subject, task)
    if not paths:
        print(f"[WARN] No per-probe Evokeds found for sub-{subject} task-{task}")
        return {}
    on_paths, off_paths = _split_by_label(paths)

    # Load one file to compute picks per ROI (assumes same montage across probes)
    info_probe = mne.read_evokeds(on_paths[0] if on_paths else off_paths[0], condition=0, verbose=False).info

    subject_qa = {}
    
    # For each ROI, create subject-level figure and compute clean subject averages
    for roi_name, roi_channels in rois.items():
        picks = _roi_picks(info_probe, list(roi_channels))
        if picks.size == 0:
            print(f"[WARN] No channels from ROI '{roi_name}' present in data for sub-{subject}")
            continue

        # Load all probe curves for this ROI
        on_curves, off_curves = [], []
        for p in on_paths:
            t, v = _load_and_roi_average(p, picks)
            on_curves.append((t, v))
        for p in off_paths:
            t, v = _load_and_roi_average(p, picks)
            off_curves.append((t, v))

        # Aggregate with outlier removal
        on_times, on_values, on_qa = _aggregate_subject_condition(
            on_curves, subject, "onTask", z_threshold, use_sequential, reject_ptp, reject_amp
        )
        off_times, off_values, off_qa = _aggregate_subject_condition(
            off_curves, subject, "offTask", z_threshold, use_sequential, reject_ptp, reject_amp
        )
        
        subject_qa[roi_name] = {
            "onTask": on_qa,
            "offTask": off_qa
        }

        # Get significant windows for this ROI
        significant_windows = _get_significant_windows(lmm_results, roi_name)

        # Subject-level plot (all probe curves + subject mean)
        subj_fig = os.path.join(out_dir, f"sub-{subject}_task-{task}_roi-{roi_name}_erps.png")
        _plot_subject(on_curves, off_curves, 
                     title=f"sub-{subject} {task} ROI {roi_name}", 
                     out_path=subj_fig,
                     significant_windows=significant_windows)

        # Store CLEAN subject averages for group analysis
        subj_means_dir = os.path.join(results_root, "participants", f"sub-{subject}", "roi_means", roi_name)
        os.makedirs(subj_means_dir, exist_ok=True)
        
        if on_times is not None and on_values is not None:
            np.savez(os.path.join(subj_means_dir, f"sub-{subject}_task-{task}_onTask_mean.npz"), 
                    times=on_times, values=on_values, **on_qa)
        if off_times is not None and off_values is not None:
            np.savez(os.path.join(subj_means_dir, f"sub-{subject}_task-{task}_offTask_mean.npz"), 
                    times=off_times, values=off_values, **off_qa)
    
    return subject_qa


def _align_temporal_arrays(subject_data: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Align temporal arrays to a common time grid for averaging.
    
    Parameters
    ----------
    subject_data : list of tuples
        List of (times, values) tuples from different subjects
        
    Returns
    -------
    common_times : np.ndarray
        Common time grid
    aligned_values : list of np.ndarray
        Values interpolated to common time grid
    """
    if not subject_data:
        return np.array([]), []
    
    # Find the time range that is common to all subjects
    min_time = max(times[0] for times, _ in subject_data)  # Latest start
    max_time = min(times[-1] for times, _ in subject_data)  # Earliest end
    
    # Use the finest resolution available
    dt_values = []
    for times, _ in subject_data:
        if len(times) > 1:
            dt_values.append(np.mean(np.diff(times)))
    
    if not dt_values:
        # Fallback for single timepoint
        return subject_data[0][0], [values for _, values in subject_data]
    
    dt = min(dt_values)
    
    # Create common time grid
    common_times = np.arange(min_time, max_time + dt/2, dt)
    
    # Interpolate all subjects to common grid
    aligned_values = []
    for times, values in subject_data:
        if len(times) == len(values):
            # Interpolate to common grid
            aligned = np.interp(common_times, times, values)
            aligned_values.append(aligned)
        else:
            print(f"[WARN] Skipping subject with mismatched times/values lengths: {len(times)} vs {len(values)}")
    
    return common_times, aligned_values


def _safe_vstack_mean(curves: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely compute mean of multiple time series with potentially different lengths.
    
    Parameters
    ----------
    curves : list of tuples
        List of (times, values) tuples
        
    Returns
    -------
    common_times : np.ndarray
        Common time grid
    mean_values : np.ndarray
        Mean values across all curves
    """
    if not curves:
        return np.array([]), np.array([])
    
    # Check if all arrays have the same length
    lengths = [len(v) for _, v in curves]
    if len(set(lengths)) == 1:
        # All same length, use fast path
        times = curves[0][0]
        values = np.vstack([v for _, v in curves]).mean(axis=0)
        return times, values
    else:
        # Different lengths, need alignment
        print(f"[INFO] Aligning {len(curves)} curves with different lengths: {set(lengths)}")
        common_times, aligned_values = _align_temporal_arrays(curves)
        if aligned_values:
            mean_values = np.mean(aligned_values, axis=0)
            return common_times, mean_values
        else:
            return np.array([]), np.array([])


def _detect_outlier_subjects(subject_data: List[Tuple[np.ndarray, np.ndarray, str]], z_threshold: float = 2.5) -> Tuple[List[Tuple[np.ndarray, np.ndarray, str]], List[str]]:
    """Remove outlier subjects based on z-score of mean amplitude."""
    if len(subject_data) < 3:
        return subject_data, []
    
    # Compute mean amplitude across time for each subject
    mean_amps = [np.mean(np.abs(v)) for _, v, _ in subject_data]
    z_scores = np.abs(stats.zscore(mean_amps))
    
    # Keep subjects with z-score below threshold
    outlier_indices = np.where(z_scores > z_threshold)[0]
    outlier_subjects = [subject_data[i][2] for i in outlier_indices]
    clean_data = [subject_data[i] for i in range(len(subject_data)) if i not in outlier_indices]
    
    return clean_data, outlier_subjects


def _plot_beta_timecourse_plotly(temporal_results: pd.DataFrame, roi_name: str, 
                                  title: str, out_path: str, cfg: Dict) -> None:
    """Create interactive plotly version of beta time-course plot."""
    plotting_cfg = cfg.get("plotting", {})
    smooth_betas = plotting_cfg.get("smooth_betas", True)
    smoothing_window = plotting_cfg.get("smoothing_window", 0.02)
    alpha = float(cfg.get("lmm_analysis", {}).get("alpha_level", 0.05))
    
    # Filter data for this ROI
    roi_data = temporal_results[temporal_results["roi"] == roi_name].copy()
    if roi_data.empty:
        print(f"[WARN] No temporal LMM data for ROI {roi_name}")
        return
    
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
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(31, 122, 31, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% CI',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add beta time-course
    fig.add_trace(go.Scatter(
        x=times, y=betas, mode='lines',
        line=dict(color='rgb(31, 122, 31)', width=3),
        name='Off-task - On-task Effect',
        hovertemplate='Time: %{x:.3f}s<br>Beta: %{y:.3f}µV<extra></extra>'
    ))
    
    # Mark significant time points
    sig_mask = p_values < alpha
    if np.any(sig_mask):
        sig_times = times[sig_mask]
        sig_betas = betas[sig_mask]
        fig.add_trace(go.Scatter(
            x=sig_times, y=sig_betas, mode='markers',
            marker=dict(color='red', size=6, opacity=0.8),
            name=f'Significant (p < {alpha})',
            hovertemplate='Time: %{x:.3f}s<br>Beta: %{y:.3f}µV<br>Significant<extra></extra>'
        ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.7)
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (s)",
        yaxis_title="Beta coefficient (µV)",
        template="plotly_white",
        hovermode='x unified',
        width=900,
        height=500
    )
    
    # Save as HTML
    fig.write_html(out_path.replace('.png', '_beta_timecourse_interactive.html'))
    
    # Also save static matplotlib version
    _plot_beta_timecourse_matplotlib(temporal_results, roi_name, title, out_path, cfg)


def _plot_beta_timecourse_matplotlib(temporal_results: pd.DataFrame, roi_name: str, 
                                     title: str, out_path: str, cfg: Dict) -> None:
    """Create matplotlib version of beta time-course plot."""
    plotting_cfg = cfg.get("plotting", {})
    smooth_betas = plotting_cfg.get("smooth_betas", True)
    smoothing_window = plotting_cfg.get("smoothing_window", 0.02)
    alpha = float(cfg.get("lmm_analysis", {}).get("alpha_level", 0.05))
    
    # Filter data for this ROI
    roi_data = temporal_results[temporal_results["roi"] == roi_name].copy()
    if roi_data.empty:
        print(f"[WARN] No temporal LMM data for ROI {roi_name}")
        return
    
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
    plt.plot(times, betas, 'b-', linewidth=3, label='Off-task - On-task Effect')
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
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def _write_qa_report(group_dir: str, roi_qa: Dict, roi_name: str) -> None:
    """Write detailed QA report for group analysis."""
    qa_rows = []
    
    for condition in ["onTask", "offTask"]:
        for subj_id, subj_data in roi_qa[condition].items():
            if subj_data:  # Only include subjects with data
                qa_rows.append({
                    "roi": roi_name,
                    "condition": condition,
                    "subject": subj_id,
                    "n_epochs_total": subj_data.get("n_epochs_total", 0),
                    "n_epochs_clean": subj_data.get("n_epochs_clean", 0),
                    "n_tasks": subj_data.get("n_tasks", 0),
                    "outlier_ratio": subj_data.get("outlier_ratio", 0.0),
                    "outlier_method": subj_data.get("outlier_method", "unknown"),
                    "outlier_indices": ",".join(map(str, subj_data.get("outlier_indices", []))),
                    "included_in_group": subj_id not in roi_qa.get("outlier_subjects", {}).get(condition, [])
                })
    
    if qa_rows:
        df = pd.DataFrame(qa_rows)
        qa_path = os.path.join(group_dir, f"qa_report_roi-{roi_name}.csv")
        df.to_csv(qa_path, index=False)
        print(f"[QA] Saved QA report: {qa_path}")


def make_group_figures(cfg: Dict, 
                      lmm_results: Optional[pd.DataFrame] = None,
                      temporal_lmm_results: Optional[pd.DataFrame] = None) -> None:
    proj = cfg.get("project", {})
    features_root = proj.get("features_root")
    results_root = proj.get("results_root", os.path.join(features_root, "figs"))
    rois = cfg.get("erp_rois", {})
    subject_z_threshold = float(cfg.get("outlier_detection", {}).get("subject_z_threshold", 2.5))
    
    # Get plotting configuration
    plotting_cfg = cfg.get("plotting", {})
    plot_type = plotting_cfg.get("plot_type", "classic_erp")
    
    group_dir = os.path.join(results_root, "group")
    participants_dir = os.path.join(results_root, "participants")
    os.makedirs(group_dir, exist_ok=True)
    
    print(f"[INFO] Generating plots of type: {plot_type}")
    
    for roi_name in rois.keys():
        print(f"Processing group analysis for ROI: {roi_name}")
        
        # Collect and AGGREGATE across tasks per subject
        subject_aggregated_on = {}  # subject_id -> list of task ERPs
        subject_aggregated_off = {}
        roi_qa = {"onTask": {}, "offTask": {}, "outlier_subjects": {"onTask": [], "offTask": []}}
        
        if os.path.isdir(participants_dir):
            for sub_name in sorted(os.listdir(participants_dir)):
                if not sub_name.startswith("sub-"):
                    continue
                subject_id = sub_name.replace("sub-", "")
                
                roi_dir = os.path.join(participants_dir, sub_name, "roi_means", roi_name)
                if not os.path.isdir(roi_dir):
                    continue
                
                # Collect all task ERPs for this subject
                subject_on_tasks = []
                subject_off_tasks = []
                
                # Load on-task data across all tasks
                on_files = [os.path.join(roi_dir, f) for f in os.listdir(roi_dir) if f.endswith("_onTask_mean.npz")]
                total_epochs_on = 0
                clean_epochs_on = 0
                all_outlier_indices_on = []
                
                for f in on_files:
                    d = np.load(f, allow_pickle=True)
                    if "values" in d and d["values"].size > 0:
                        subject_on_tasks.append((d["times"], d["values"]))
                        total_epochs_on += int(d.get("n_epochs_total", 0))
                        clean_epochs_on += int(d.get("n_epochs_clean", 0))
                        task_outliers = d.get("outlier_indices", [])
                        if hasattr(task_outliers, 'tolist'):
                            all_outlier_indices_on.extend(task_outliers.tolist())
                        else:
                            all_outlier_indices_on.extend(list(task_outliers))
                
                # Load off-task data across all tasks
                off_files = [os.path.join(roi_dir, f) for f in os.listdir(roi_dir) if f.endswith("_offTask_mean.npz")]
                total_epochs_off = 0
                clean_epochs_off = 0
                all_outlier_indices_off = []
                
                for f in off_files:
                    d = np.load(f, allow_pickle=True)
                    if "values" in d and d["values"].size > 0:
                        subject_off_tasks.append((d["times"], d["values"]))
                        total_epochs_off += int(d.get("n_epochs_total", 0))
                        clean_epochs_off += int(d.get("n_epochs_clean", 0))
                        task_outliers = d.get("outlier_indices", [])
                        if hasattr(task_outliers, 'tolist'):
                            all_outlier_indices_off.extend(task_outliers.tolist())
                        else:
                            all_outlier_indices_off.extend(list(task_outliers))
                
                # Average across tasks for this subject
                if subject_on_tasks:
                    common_times_on, aligned_values_on = _align_temporal_arrays(subject_on_tasks)
                    if aligned_values_on:
                        subject_on_mean = np.mean(aligned_values_on, axis=0)
                        subject_aggregated_on[subject_id] = (common_times_on, subject_on_mean)
                    roi_qa["onTask"][subject_id] = {
                        "n_epochs_total": total_epochs_on,
                        "n_epochs_clean": clean_epochs_on,
                        "outlier_ratio": len(all_outlier_indices_on) / total_epochs_on if total_epochs_on > 0 else 0.0,
                        "outlier_indices": all_outlier_indices_on,
                        "n_tasks": len(subject_on_tasks)
                    }
                
                if subject_off_tasks:
                    common_times_off, aligned_values_off = _align_temporal_arrays(subject_off_tasks)
                    if aligned_values_off:
                        subject_off_mean = np.mean(aligned_values_off, axis=0)
                        subject_aggregated_off[subject_id] = (common_times_off, subject_off_mean)
                    roi_qa["offTask"][subject_id] = {
                        "n_epochs_total": total_epochs_off,
                        "n_epochs_clean": clean_epochs_off,
                        "outlier_ratio": len(all_outlier_indices_off) / total_epochs_off if total_epochs_off > 0 else 0.0,
                        "outlier_indices": all_outlier_indices_off,
                        "n_tasks": len(subject_off_tasks)
                    }
        
        # Now we have ONE ERP per subject per condition (aggregated across tasks)
        subject_data_on = [(t, v, subj_id) for subj_id, (t, v) in subject_aggregated_on.items()]
        subject_data_off = [(t, v, subj_id) for subj_id, (t, v) in subject_aggregated_off.items()]
        
        # Remove outlier subjects
        clean_on, outlier_on = _detect_outlier_subjects(subject_data_on, subject_z_threshold)
        clean_off, outlier_off = _detect_outlier_subjects(subject_data_off, subject_z_threshold)
        
        roi_qa["outlier_subjects"]["onTask"] = outlier_on
        roi_qa["outlier_subjects"]["offTask"] = outlier_off
        
        print(f"  On-task: {len(clean_on)}/{len(subject_data_on)} subjects (removed {len(outlier_on)} outliers)")
        print(f"  Off-task: {len(clean_off)}/{len(subject_data_off)} subjects (removed {len(outlier_off)} outliers)")
        
        # Write QA report
        _write_qa_report(group_dir, roi_qa, roi_name)
        
        # Create group plots with clean data
        if clean_on or clean_off:
            subject_mean_on = [(t, v) for t, v, _ in clean_on]
            subject_mean_off = [(t, v) for t, v, _ in clean_off]
            
            # Get significant windows for this ROI
            significant_windows = _get_significant_windows(lmm_results, roi_name)
            
            # Generate plots based on configuration
            if plot_type in ["classic_erp", "both"]:
                # Option A: Classic ERP plots (traditional approach)
                print(f"  Generating classic ERP plots for ROI {roi_name}")
                fig_path = os.path.join(group_dir, f"group_roi-{roi_name}_erps_classic.png")
                _plot_grand_average_plotly(subject_mean_on, subject_mean_off, 
                                         title=f"Group ROI {roi_name} - Classic ERP", 
                                         out_path=fig_path,
                                         significant_windows=significant_windows)
            
            if plot_type in ["beta_timecourse", "both"]:
                # Option B: Beta time-course plots (statistical effects)
                if temporal_lmm_results is not None and not temporal_lmm_results.empty:
                    print(f"  Generating beta time-course plots for ROI {roi_name}")
                    beta_fig_path = os.path.join(group_dir, f"group_roi-{roi_name}_beta_timecourse.png")
                    _plot_beta_timecourse_plotly(temporal_lmm_results, roi_name,
                                               title=f"ROI {roi_name}: Off-task vs On-task Beta Time-course",
                                               out_path=beta_fig_path, cfg=cfg)
                else:
                    print(f"  [WARN] No temporal LMM results available for beta time-course plots")

            # Save numerical arrays for the group grand mean (for classic ERPs)
            if plot_type in ["classic_erp", "both"]:
                os.makedirs(os.path.join(group_dir, "roi_arrays"), exist_ok=True)
                if subject_mean_on:
                    # Align temporal arrays before averaging
                    common_times_on, aligned_values_on = _align_temporal_arrays(subject_mean_on)
                    if aligned_values_on:
                        on_mean = np.mean(aligned_values_on, axis=0)
                        np.savez(os.path.join(group_dir, "roi_arrays", f"group_roi-{roi_name}_onTask_mean.npz"), 
                                times=common_times_on, values=on_mean, n_subjects=len(aligned_values_on))
                
                if subject_mean_off:
                    # Align temporal arrays before averaging
                    common_times_off, aligned_values_off = _align_temporal_arrays(subject_mean_off)
                    if aligned_values_off:
                        off_mean = np.mean(aligned_values_off, axis=0)
                        np.savez(os.path.join(group_dir, "roi_arrays", f"group_roi-{roi_name}_offTask_mean.npz"), 
                                times=common_times_off, values=off_mean, n_subjects=len(aligned_values_off))


def main():
    parser = argparse.ArgumentParser(description="Make ERP figures comparing on-task vs off-task per ROI")
    parser.add_argument("--config", default="./ERPs_new/config.yaml", help="Path to YAML config")
    parser.add_argument("--subject", help="Process only this subject")
    parser.add_argument("--task", help="Process only this task")
    parser.add_argument("--group-only", action="store_true", help="Generate only group figures (skip individual)")
    parser.add_argument("--plot-type", choices=["classic_erp", "beta_timecourse", "both"], 
                       help="Type of plots to generate (overrides config)")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    
    # Override plot type from CLI if provided
    if args.plot_type:
        if "plotting" not in cfg:
            cfg["plotting"] = {}
        cfg["plotting"]["plot_type"] = args.plot_type
        print(f"[INFO] Plot type overridden to: {args.plot_type}")
    
    # Load LMM results if available
    lmm_results = _load_lmm_results(cfg)
    if lmm_results is not None:
        print(f"[INFO] Loaded LMM results with {len(lmm_results)} ROI-window combinations")
    else:
        print("[INFO] No LMM results found - plots will not show significant windows")
    
    # Load temporal LMM results if available
    temporal_lmm_results = _load_temporal_lmm_results(cfg)
    if temporal_lmm_results is not None:
        print(f"[INFO] Loaded temporal LMM results with {len(temporal_lmm_results)} time points")
    else:
        print("[INFO] No temporal LMM results found - beta time-course plots will not be available")

    subjects_cfg = cfg.get("subjects", [])
    if args.subject:
        subjects = [str(args.subject)]
    else:
        subjects = [str(s) for s in subjects_cfg]

    tasks_cfg = cfg.get("tasks", [])
    if args.task:
        tasks = [str(args.task)]
    else:
        tasks = [str(t) for t in tasks_cfg]

    # Always process individual subjects first (unless --group-only)
    if not args.group_only:
        print("Processing individual subjects...")
        all_qa = {}
        for sub in subjects:
            for t in tasks:
                qa_info = process_subject_task(cfg, sub, t, lmm_results)
                if qa_info:
                    all_qa[f"{sub}_{t}"] = qa_info

    # Always generate group figures at the end
    print("Generating group figures...")
    make_group_figures(cfg, lmm_results, temporal_lmm_results)


if __name__ == "__main__":
    main()


