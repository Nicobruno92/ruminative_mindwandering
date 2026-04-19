#!/usr/bin/env python3
"""
Plot ERP waveform for significant LMM cluster electrodes: On-task vs Off-task.

Reads preprocessed epochs directly from BIDS derivatives and computes a
group-level ERP by classifying probes into on-task / off-task categories
using one of three configurable methods.

Classification methods
----------------------
global_median
    Threshold = median of all probe onoff scores across all subjects and tasks.
within_subject_median
    Threshold = per-subject median (each subject is split around their own
    median onoff score).
fixed
    Threshold = configurable constant (default 50; equal-to-threshold →
    off-task, same convention as the rest of the ERPs_new pipeline).

Probe selection
---------------
By default (``use_junifer_probe_selection: true``) the script loads the probe
list and onoff scores directly from the junifer aggregated summary CSVs —
the same source used by the LMM clustering pipeline.  This ensures the ERP
reflects exactly the trials included in the statistical analysis.

When disabled, probe selection is re-derived from the events.tsv using the
same go/correct + distance-range criteria, which may differ slightly.

Configuration
-------------
All parameters are read from ``ERPs_new/config.yaml`` under the key
``cluster_erp``.  Cluster channels can be specified either as an explicit
list (``cluster_channels``) or auto-loaded from the LMM cluster-summary CSV
(``cluster_csv`` + ``cluster_name``).

Usage
-----
::

    python ERPs_new/plot_cluster_erp.py --config ERPs_new/config.yaml
    python ERPs_new/plot_cluster_erp.py --config ERPs_new/config.yaml \\
        --method within_subject_median
    python ERPs_new/plot_cluster_erp.py --config ERPs_new/config.yaml \\
        --method fixed --threshold 50
"""

# ===== Imports =====
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from helpers import (
    load_yaml_config,
    read_derivative_epochs,
    parse_events_tsv,
    enrich_events_with_parsed_fields,
    select_trials_for_probes,
    load_qa_summary,
    get_qa_exclusions,
    should_process_subject_task,
)


# ===== Cluster channel utilities =====

def load_cluster_channels(cfg: Dict) -> List[str]:
    """
    Resolve the list of cluster channels to include in the ERP plot.

    Priority order:

    1. ``cluster_erp.cluster_csv`` + ``cluster_erp.cluster_name`` — reads the
       channel list from the LMM cluster-summary CSV (the ``channels`` column
       of the row with the largest cluster statistic matching ``cluster_name``).
    2. ``cluster_erp.cluster_channels`` — explicit list in config.

    Parameters
    ----------
    cfg : Dict
        Full configuration dictionary.

    Returns
    -------
    List[str]
        Channel names belonging to the significant cluster.

    Raises
    ------
    ValueError
        When no cluster channels can be resolved.
    """
    cluster_cfg = cfg.get("cluster_erp", {})
    cluster_csv = cluster_cfg.get("cluster_csv") or ""
    cluster_name = cluster_cfg.get("cluster_name") or ""

    if cluster_csv and os.path.exists(cluster_csv):
        df = pd.read_csv(cluster_csv)
        if cluster_name:
            candidate = df[df["marker_name"] == cluster_name]
        else:
            candidate = df[df["significant"].astype(bool)]
        if candidate.empty:
            raise ValueError(
                f"No significant cluster found in {cluster_csv} "
                f"for cluster_name='{cluster_name}'"
            )
        row = candidate.sort_values("statistic", ascending=False).iloc[0]
        return [ch.strip() for ch in str(row["channels"]).split(",")]

    channels = cluster_cfg.get("cluster_channels") or []
    if not channels:
        raise ValueError(
            "No cluster channels defined in config. "
            "Set cluster_erp.cluster_channels or cluster_erp.cluster_csv."
        )
    return list(channels)


# ===== Junifer probe metadata =====

def load_junifer_probe_metadata(
    features_root: str,
    subject: str,
    tasks: List[str],
) -> pd.DataFrame:
    """
    Load valid probe list and onoff scores from junifer aggregated summaries.

    This mirrors the exact probe selection used by the LMM clustering pipeline:
    only probes present in the junifer ``summary_*.csv`` files with
    ``marker_type == "evoked"`` are included.  The onoff score is read from
    the first row of the per-probe evoked CSV (same value across all channels).

    Parameters
    ----------
    features_root : str
        Root features directory (``project.features_root`` in config).
    subject : str
        Zero-padded subject identifier, e.g. ``"04"``.
    tasks : List[str]
        Task names to search, e.g. ``["Sart1", "Sart2", "Sart3", "Sart4"]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``task``, ``probe_number`` (int), ``onoff_score`` (float),
        ``n_epochs`` (int), ``label_junifer`` (str).
        Empty DataFrame when no data is found.
    """
    rows: List[Dict] = []
    junifer_dir = os.path.join(
        features_root, f"sub-{subject}", "eeg", "junifer_aggregated"
    )

    for task in tasks:
        summary_path = os.path.join(
            junifer_dir, f"summary_sub-{subject}_task-{task}.csv"
        )
        if not os.path.exists(summary_path):
            continue

        summary_df = pd.read_csv(summary_path)
        evoked_rows = summary_df[summary_df["marker_type"] == "evoked"]

        for _, row in evoked_rows.iterrows():
            evoked_csv = str(row["output_path"])
            if not os.path.exists(evoked_csv):
                continue

            # onoff is constant across all channel rows — read first row only
            evoked_data = pd.read_csv(evoked_csv, nrows=1)
            onoff_score: float = (
                float(evoked_data["onoff"].iloc[0])
                if "onoff" in evoked_data.columns
                else np.nan
            )

            rows.append({
                "task": task,
                "probe_number": int(row["probe_number"]),
                "onoff_score": onoff_score,
                "n_epochs": int(row["n_epochs"]),
                "label_junifer": str(row["label"]),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["task", "probe_number", "onoff_score", "n_epochs", "label_junifer"]
    )


# ===== Per-probe ERP extraction =====

def _get_cluster_picks(info: mne.Info, cluster_channels: List[str]) -> np.ndarray:
    """
    Return channel indices for the cluster channels present in the recording.

    Parameters
    ----------
    info : mne.Info
        Info object from the loaded epochs.
    cluster_channels : List[str]
        Requested cluster channel names.

    Returns
    -------
    np.ndarray
        Integer indices into ``info["ch_names"]``.

    Raises
    ------
    RuntimeError
        When none of the requested channels are found.
    """
    present = [ch for ch in cluster_channels if ch in info["ch_names"]]
    missing = [ch for ch in cluster_channels if ch not in info["ch_names"]]
    if missing:
        print(f"[WARN] Cluster channels absent from recording: {missing}")
    if not present:
        raise RuntimeError(
            f"None of the requested cluster channels found. "
            f"Requested: {cluster_channels}"
        )
    return mne.pick_channels(info["ch_names"], include=present, exclude=[])


def extract_probe_erps(
    cfg: Dict,
    subject: str,
    task: str,
    cluster_channels: List[str],
    onoff_override: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Extract per-probe ERP waveforms averaged over cluster channels.

    Loads preprocessed epochs, selects go/correct trials within the
    pre-probe distance window, computes one evoked per probe, and averages
    over the cluster channels.

    Parameters
    ----------
    cfg : Dict
        Full configuration dictionary.
    subject : str
        Zero-padded subject identifier, e.g. ``"04"``.
    task : str
        Task name, e.g. ``"Sart1"``.
    cluster_channels : List[str]
        Channel names to average over for the cluster ROI.
    onoff_override : Dict[int, float] or None
        When provided, maps ``probe_number → onoff_score`` and restricts
        processing to those probes only.  Onoff scores come from the junifer
        pipeline rather than being re-extracted from events.  When ``None``,
        all probes passing trial-selection criteria are used and onoff is
        extracted from event descriptions.

    Returns
    -------
    pd.DataFrame
        One row per probe with columns:
        ``subject``, ``task``, ``probe_number``, ``onoff_score``,
        ``times`` (np.ndarray), ``erp`` (np.ndarray, µV, mean over cluster
        channels).  Empty DataFrame when no valid data is found.
    """
    project = cfg["project"]
    trial_sel = cfg.get("trial_selection", {})
    evk_cfg = cfg.get("evoked", {})

    derivatives_root = project["derivatives_root"]
    bids_root = project.get("bids_root")
    evoked_desc_in = project.get("input_evoked_desc", "evoked")

    # Trial selection strategy:
    #   Junifer mode (onoff_override provided): match junifer exactly →
    #     go trials only, correctness ignored (filter_go=True, filter_correct=False).
    #   Standalone mode: use only_go_correct from config (default True).
    distance_min = int(trial_sel.get("distance_min", -5))
    distance_max = int(trial_sel.get("distance_max", -1))
    min_req = int(trial_sel.get("min_required_distances", 3))
    if onoff_override is not None:
        only_go_correct = False  # will pre-filter go below
    else:
        only_go_correct = bool(trial_sel.get("only_go_correct", True))

    apply_baseline = bool(evk_cfg.get("apply_baseline", False))
    baseline = tuple(evk_cfg.get("baseline", (-0.1, 0.0)))

    epochs, events_tsv, _ = read_derivative_epochs(
        derivatives_root=derivatives_root,
        subject=subject,
        task=task,
        desc=evoked_desc_in,
        preload=True,
        bids_root=bids_root,
    )

    cluster_picks = _get_cluster_picks(epochs.info, cluster_channels)

    events_df = parse_events_tsv(events_tsv)
    events_df = enrich_events_with_parsed_fields(events_df)
    events_df = events_df.rename(columns={"row_index": "epoch_index"})

    # In junifer mode, pre-filter to go trials only (nogo excluded, but
    # incorrect go trials kept — same as junifer filter_go=True, filter_correct=False)
    events_for_selection = (
        events_df[events_df["trial_type"] == "go"].copy()
        if onoff_override is not None
        else events_df
    )

    selected = select_trials_for_probes(
        events_for_selection,
        only_go_correct=only_go_correct,
        distance_min=distance_min,
        distance_max=distance_max,
        min_required_distances=min_req,
    )

    if selected.empty:
        print(f"[WARN] No trials selected for sub-{subject} task-{task}")
        return pd.DataFrame()

    rows: List[Dict] = []
    for probe_num, grp in selected.groupby("probe_number", sort=False):
        probe_int = int(probe_num)

        # When using junifer probe selection: skip probes not in the override
        # dict and use its pre-computed onoff score instead of re-extracting.
        if onoff_override is not None:
            if probe_int not in onoff_override:
                continue
            onoff_score = onoff_override[probe_int]
        else:
            onoff_values = grp["onoff"].dropna()
            if onoff_values.empty:
                continue
            onoff_score = float(np.median(onoff_values))

        epoch_indices = grp["epoch_index"].astype(int).tolist()
        if not epoch_indices:
            continue

        ep = epochs.copy()[epoch_indices]
        if apply_baseline:
            ep.apply_baseline(baseline=baseline)

        evk = ep.average()
        # Average over cluster channels → single time series per probe
        cluster_mean = evk.data[cluster_picks, :].mean(axis=0)

        # Convert to µV when data is stored in volts
        if np.abs(cluster_mean).max() < 1e-3:
            cluster_mean = cluster_mean * 1e6

        rows.append({
            "subject": subject,
            "task": task,
            "probe_number": int(probe_num),
            "onoff_score": onoff_score,
            "times": evk.times.copy(),
            "erp": cluster_mean,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["subject", "task", "probe_number", "onoff_score", "times", "erp"]
    )


# ===== Classification =====

def compute_thresholds(
    probe_df: pd.DataFrame,
    method: str,
    fixed_threshold: float,
) -> pd.DataFrame:
    """
    Add a ``threshold`` column to the probe DataFrame.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Combined probe data with an ``onoff_score`` column.
    method : str
        One of ``"global_median"``, ``"within_subject_median"``, ``"fixed"``.
    fixed_threshold : float
        Threshold value when ``method="fixed"``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a ``threshold`` column added in-place.

    Raises
    ------
    ValueError
        When ``method`` is not one of the three recognised options, or when
        no valid onoff scores are available for ``global_median``.
    """
    df = probe_df.copy()

    if method == "global_median":
        valid = df["onoff_score"].dropna()
        if valid.empty:
            raise ValueError(
                "No valid onoff scores available for global_median computation."
            )
        thr = float(np.median(valid))
        df["threshold"] = thr
        print(f"[INFO] Global median threshold: {thr:.2f}  (n={len(valid)} probes)")

    elif method == "within_subject_median":
        sub_medians = (
            df.groupby("subject")["onoff_score"]
            .median()
            .rename("threshold")
        )
        df = df.join(sub_medians, on="subject")
        print("[INFO] Within-subject median thresholds:")
        print(sub_medians.to_string())

    elif method == "fixed":
        df["threshold"] = float(fixed_threshold)
        print(f"[INFO] Fixed threshold: {fixed_threshold}")

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: global_median, within_subject_median, fixed"
        )

    return df


def assign_labels(probe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign ``"onTask"`` / ``"offTask"`` / ``"unknown"`` to each probe.

    Convention: ``onoff_score > threshold`` → ``"onTask"``;
    ``onoff_score <= threshold`` → ``"offTask"``.
    Probes with a missing score or threshold become ``"unknown"``.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Output of :func:`compute_thresholds`.

    Returns
    -------
    pd.DataFrame
        Input with a ``label`` column added.
    """
    df = probe_df.copy()

    def _label(row: pd.Series) -> str:
        if pd.isna(row["onoff_score"]) or pd.isna(row["threshold"]):
            return "unknown"
        return "onTask" if row["onoff_score"] > row["threshold"] else "offTask"

    df["label"] = df.apply(_label, axis=1)
    return df


# ===== Subject-level averaging =====

def _align_and_mean(
    curves: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Average a list of (times, values) time series onto a common time grid.

    When all curves share the same time vector, no interpolation is performed.
    Otherwise all curves are linearly interpolated onto the finest common grid.

    Parameters
    ----------
    curves : List[Tuple[np.ndarray, np.ndarray]]
        Each tuple is ``(times, amplitude_µV)``.

    Returns
    -------
    times : np.ndarray or None
        Common time vector.
    mean : np.ndarray or None
        Mean amplitude across curves.
    """
    if not curves:
        return None, None

    lengths = {len(v) for _, v in curves}
    if len(lengths) == 1:
        t = curves[0][0]
        return t, np.vstack([v for _, v in curves]).mean(axis=0)

    min_t = max(t[0] for t, _ in curves)
    max_t = min(t[-1] for t, _ in curves)
    dt_vals = [np.mean(np.diff(t)) for t, _ in curves if len(t) > 1]
    dt = min(dt_vals) if dt_vals else 0.001
    common_t = np.arange(min_t, max_t + dt / 2, dt)
    aligned = [np.interp(common_t, t, v) for t, v in curves if len(t) == len(v)]
    if not aligned:
        return None, None
    return common_t, np.mean(aligned, axis=0)


def aggregate_to_subjects(
    probe_df: pd.DataFrame,
    min_probes_per_condition: int,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, str]],
    List[Tuple[np.ndarray, np.ndarray, str]],
]:
    """
    Average probe ERPs within each condition for every subject.

    Subjects with fewer than ``min_probes_per_condition`` probes in either
    condition are excluded with a warning.  Aggregation is across all tasks
    (Sart1–Sart4) within each subject.

    Parameters
    ----------
    probe_df : pd.DataFrame
        Output of :func:`assign_labels` with ``erp`` and ``times`` columns.
    min_probes_per_condition : int
        Minimum probes required in *each* condition to include a subject.

    Returns
    -------
    on_task : List[Tuple[times, erp, subject_id]]
    off_task : List[Tuple[times, erp, subject_id]]
    """
    on_task: List[Tuple[np.ndarray, np.ndarray, str]] = []
    off_task: List[Tuple[np.ndarray, np.ndarray, str]] = []

    labeled = probe_df[probe_df["label"].isin(["onTask", "offTask"])].copy()

    for subject, sub_df in labeled.groupby("subject"):
        on_curves = [
            (row["times"], row["erp"])
            for _, row in sub_df[sub_df["label"] == "onTask"].iterrows()
        ]
        off_curves = [
            (row["times"], row["erp"])
            for _, row in sub_df[sub_df["label"] == "offTask"].iterrows()
        ]

        n_on = len(on_curves)
        n_off = len(off_curves)

        if n_on < min_probes_per_condition:
            print(
                f"[SKIP] sub-{subject}: {n_on} on-task probes "
                f"< min={min_probes_per_condition}"
            )
            continue
        if n_off < min_probes_per_condition:
            print(
                f"[SKIP] sub-{subject}: {n_off} off-task probes "
                f"< min={min_probes_per_condition}"
            )
            continue

        on_t, on_v = _align_and_mean(on_curves)
        off_t, off_v = _align_and_mean(off_curves)

        if on_t is not None and off_t is not None:
            on_task.append((on_t, on_v, str(subject)))
            off_task.append((off_t, off_v, str(subject)))
            print(
                f"[OK] sub-{subject}: "
                f"on-task={n_on} probes, off-task={n_off} probes"
            )

    return on_task, off_task


# ===== Grand average utilities =====

def _grand_average_with_sem(
    subject_data: List[Tuple[np.ndarray, np.ndarray, str]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute grand average and SEM across subjects.

    Parameters
    ----------
    subject_data : List[Tuple[times, erp, subject_id]]

    Returns
    -------
    times : np.ndarray or None
    grand_mean : np.ndarray or None
    sem : np.ndarray or None
    """
    if not subject_data:
        return None, None, None

    curves = [(t, v) for t, v, _ in subject_data]
    common_t, _ = _align_and_mean(curves)
    if common_t is None:
        return None, None, None

    aligned = [np.interp(common_t, t, v) for t, v, _ in subject_data]
    arr = np.array(aligned)
    grand_mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
    return common_t, grand_mean, sem


# ===== Plotting =====

def plot_cluster_erp(
    on_task_data: List[Tuple[np.ndarray, np.ndarray, str]],
    off_task_data: List[Tuple[np.ndarray, np.ndarray, str]],
    cluster_channels: List[str],
    method: str,
    output_dir: str,
    cfg: Dict,
) -> None:
    """
    Produce and save a group ERP figure for the significant cluster channels.

    The figure shows:
    * Transparent thin lines for each subject's condition mean.
    * Grand average ± SEM per condition in bold.
    * Shaded P3b time window (from ``lmm_analysis.time_windows.P3b`` in config).

    Both a static PNG and accompanying NumPy arrays are saved under
    ``output_dir``.

    Parameters
    ----------
    on_task_data : List[Tuple[times, erp, subject_id]]
    off_task_data : List[Tuple[times, erp, subject_id]]
    cluster_channels : List[str]
        Names of the cluster channels (for the title).
    method : str
        Classification method label (used in title and filename).
    output_dir : str
        Directory where outputs are saved.
    cfg : Dict
        Full configuration dictionary (used to read P3b window).
    """
    os.makedirs(output_dir, exist_ok=True)

    on_t, on_mean, on_sem = _grand_average_with_sem(on_task_data)
    off_t, off_mean, off_sem = _grand_average_with_sem(off_task_data)

    n_on = len(on_task_data)
    n_off = len(off_task_data)

    ch_short = ", ".join(cluster_channels[:5])
    if len(cluster_channels) > 5:
        ch_short += f" … (+{len(cluster_channels) - 5})"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual subject traces (very transparent)
    for t, v, _ in on_task_data:
        ax.plot(t, v, color="#2ca02c", alpha=0.10, linewidth=0.7)
    for t, v, _ in off_task_data:
        ax.plot(t, v, color="#d62728", alpha=0.10, linewidth=0.7)

    # Grand averages with SEM shading
    if on_t is not None:
        ax.fill_between(
            on_t, on_mean - on_sem, on_mean + on_sem,
            color="#1f7a1f", alpha=0.25,
        )
        ax.plot(
            on_t, on_mean,
            color="#1f7a1f", linewidth=2.5,
            label=f"On-task (n={n_on})",
        )

    if off_t is not None:
        ax.fill_between(
            off_t, off_mean - off_sem, off_mean + off_sem,
            color="#8b0000", alpha=0.25,
        )
        ax.plot(
            off_t, off_mean,
            color="#8b0000", linewidth=2.5,
            label=f"Off-task (n={n_off})",
        )

    # P3b window highlight
    p3b_win = (
        cfg.get("lmm_analysis", {})
           .get("time_windows", {})
           .get("P3b")
    )
    if p3b_win:
        ax.axvspan(
            p3b_win[0], p3b_win[1],
            alpha=0.12, color="steelblue",
            label=f"P3b window ({p3b_win[0]}–{p3b_win[1]} s)",
        )

    ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0, color="k", linestyle="-",  linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(
        f"P3b Cluster ERP — On-task vs Off-task\n"
        f"Method: {method}  |  {len(cluster_channels)} channels: {ch_short}",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(output_dir, f"cluster_erp_{method}.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {out_png}")

    # Save numerical arrays for downstream use
    for label, t_arr, mean_arr, sem_arr, n_subs in [
        ("onTask",  on_t,  on_mean,  on_sem,  n_on),
        ("offTask", off_t, off_mean, off_sem, n_off),
    ]:
        if t_arr is not None:
            npz_path = os.path.join(
                output_dir, f"cluster_erp_{method}_{label}.npz"
            )
            np.savez(
                npz_path,
                times=t_arr, mean=mean_arr, sem=sem_arr,
                n_subjects=n_subs,
            )
            print(f"[OK] Saved arrays: {npz_path}")


# ===== Main =====

def main() -> None:
    """Entry point: parse arguments, run pipeline, save outputs."""
    parser = argparse.ArgumentParser(
        description="Plot group ERP for significant LMM cluster channels."
    )
    parser.add_argument(
        "--config",
        default="./ERPs_new/config.yaml",
        help="Path to ERPs_new config YAML (default: ./ERPs_new/config.yaml)",
    )
    parser.add_argument(
        "--method",
        choices=["global_median", "within_subject_median", "fixed"],
        help="Classification method (overrides cluster_erp.classification_method)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Fixed threshold value (only with --method fixed; overrides config)",
    )
    parser.add_argument(
        "--subject",
        help="Restrict processing to a single subject",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cluster_cfg = cfg.get("cluster_erp", {})
    project = cfg.get("project", {})

    # ---- Resolve runtime parameters ----
    method: str = args.method or cluster_cfg.get(
        "classification_method", "global_median"
    )
    fixed_threshold: float = (
        args.threshold
        if args.threshold is not None
        else float(cluster_cfg.get("fixed_threshold", 50.0))
    )
    min_probes: int = int(cluster_cfg.get("min_probes_per_condition", 3))
    output_dir: str = cluster_cfg.get("output_dir", "results/ERPs_new/cluster_erp")
    use_junifer: bool = bool(cluster_cfg.get("use_junifer_probe_selection", True))

    subjects: List[str] = (
        [str(args.subject)]
        if args.subject
        else [str(s) for s in cfg.get("subjects", [])]
    )
    tasks: List[str] = [str(t) for t in cfg.get("tasks", [])]
    features_root: str = project["features_root"]

    print(f"[INFO] Classification method    : {method}")
    print(f"[INFO] Junifer probe selection  : {use_junifer}")
    print(f"[INFO] Subjects                 : {len(subjects)}")
    print(f"[INFO] Tasks                    : {tasks}")
    print(f"[INFO] Output directory         : {output_dir}")

    # ---- Load cluster channels ----
    cluster_channels = load_cluster_channels(cfg)
    print(f"[INFO] Cluster channels ({len(cluster_channels)}): {cluster_channels}")

    # ---- Optional QA filtering ----
    qa_summary_path = project.get("qa_summary_path")
    exclude_failed_qa = bool(project.get("exclude_failed_qa", False))
    qa_exclusions = None
    if qa_summary_path and exclude_failed_qa:
        qa_df = load_qa_summary(qa_summary_path, verbose=False)
        qa_exclusions = get_qa_exclusions(qa_df, epoch_type="evoked")
        print(f"[INFO] QA: {len(qa_exclusions)} subject-task pairs excluded")

    # ---- Extract per-probe ERPs (single pass through all epochs) ----
    all_rows: List[pd.DataFrame] = []
    for sub in subjects:
        # Load junifer probe metadata once per subject (all tasks)
        junifer_meta: Optional[pd.DataFrame] = None
        if use_junifer:
            junifer_meta = load_junifer_probe_metadata(features_root, sub, tasks)
            if junifer_meta.empty:
                print(f"[WARN] sub-{sub}: no junifer summaries found, skipping")
                continue

        for task in tasks:
            if qa_exclusions is not None and not should_process_subject_task(
                sub, task, qa_exclusions
            ):
                print(f"[SKIP] sub-{sub} task-{task}: failed QA")
                continue

            # Build per-task onoff override from junifer metadata
            onoff_override: Optional[Dict[int, float]] = None
            if use_junifer and junifer_meta is not None:
                task_meta = junifer_meta[junifer_meta["task"] == task]
                if task_meta.empty:
                    print(f"[SKIP] sub-{sub} task-{task}: no junifer data")
                    continue
                onoff_override = dict(
                    zip(
                        task_meta["probe_number"].astype(int),
                        task_meta["onoff_score"].astype(float),
                    )
                )

            print(f"[INFO] Processing sub-{sub} task-{task} …")
            df = extract_probe_erps(
                cfg, sub, task, cluster_channels,
                onoff_override=onoff_override,
            )
            if not df.empty:
                all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            "No probe ERPs extracted. "
            "Verify that derivatives_root is set correctly in config and that "
            "preprocessed epochs (desc-evoked_epo.fif) exist."
        )

    probe_df = pd.concat(all_rows, ignore_index=True)
    n_probes_total = len(probe_df)
    n_subjects_found = probe_df["subject"].nunique()
    print(
        f"\n[INFO] Extracted {n_probes_total} probes "
        f"from {n_subjects_found} subjects"
    )

    # Drop probes lacking an onoff rating
    n_before = len(probe_df)
    probe_df = probe_df.dropna(subset=["onoff_score"])
    n_dropped = n_before - len(probe_df)
    if n_dropped > 0:
        print(f"[INFO] Dropped {n_dropped} probes with missing onoff scores")

    # ---- Classify probes ----
    probe_df = compute_thresholds(
        probe_df, method=method, fixed_threshold=fixed_threshold
    )
    probe_df = assign_labels(probe_df)

    counts = probe_df["label"].value_counts()
    print(f"[INFO] Label distribution: {counts.to_dict()}")

    # ---- Aggregate to subject level ----
    on_task, off_task = aggregate_to_subjects(
        probe_df, min_probes_per_condition=min_probes
    )

    if not on_task or not off_task:
        raise RuntimeError(
            f"Insufficient data after quality filtering: "
            f"{len(on_task)} on-task subjects, "
            f"{len(off_task)} off-task subjects."
        )

    print(
        f"\n[INFO] Final: {len(on_task)} on-task subjects, "
        f"{len(off_task)} off-task subjects"
    )

    # ---- Plot and save ----
    plot_cluster_erp(
        on_task, off_task, cluster_channels, method, output_dir, cfg
    )

    # ---- Save used config ----
    os.makedirs(output_dir, exist_ok=True)
    used_cfg = {
        "classification_method": method,
        "fixed_threshold": float(fixed_threshold),
        "cluster_channels": cluster_channels,
        "n_subjects_on_task": len(on_task),
        "n_subjects_off_task": len(off_task),
        "n_probes_total": int(n_probes_total),
        "subjects": subjects,
        "tasks": tasks,
    }
    config_out = os.path.join(output_dir, "used_config.yaml")
    with open(config_out, "w") as fh:
        yaml.dump(used_cfg, fh, default_flow_style=False)
    print(f"[OK] Saved used config: {config_out}")


if __name__ == "__main__":
    main()
