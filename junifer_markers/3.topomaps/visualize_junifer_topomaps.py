#!/usr/bin/env python3
"""
Visualize topographic maps of Junifer EEG markers by mental state.

For each marker extracted by the Junifer pipeline, generates publication-quality
topomap figures with the spatial scalp distribution grouped by mental state.

Layout (2 rows × N columns where N = len(mental_states))
--------------------------------------------------------
Row 0 — condition means:
    on-task | about-task | distracted | deliberate | spontaneous | blank

Row 1 — differences vs reference (on-task):
    on-task (ref) | diff | diff | diff | diff | diff

Per marker × epoch_type, one figure per task combination is produced
(combined + Sart1..Sart4 individually by default).

Pipeline
--------
1. Glob per-probe CSV files from features_root/sub-*/eeg/junifer_aggregated/
2. Two-level averaging: probes → subject mean per channel → group mean
3. Plot two-row topomap figures and save PNG + SVG

Usage
-----
::

    conda activate eeg
    python visualize_junifer_topomaps.py
    python visualize_junifer_topomaps.py --epoch-type sleep
    python visualize_junifer_topomaps.py --marker power_alpha
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

# Metadata columns present in every per-probe CSV (NOT markers).
# This project has no `session` column.
METADATA_COLS = {
    "subject", "task", "probe_number", "label", "n_epochs",
    "channel", "ontask_label", "content", "confidence_level", "depth_level",
}


# =============================================================================
# HELPERS
# =============================================================================

def get_project_root() -> Path:
    """Resolve project root via ``git rev-parse --show-toplevel``.

    Returns
    -------
    Path
        Repository root.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : Path

    Returns
    -------
    dict
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(project_root: Path, value: str) -> Path:
    """Return ``value`` as absolute path (anchored at project_root if relative).

    Parameters
    ----------
    project_root : Path
    value : str

    Returns
    -------
    Path
    """
    p = Path(value)
    return p if p.is_absolute() else project_root / p


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_probe_data(
    project_root: Path,
    cfg: Dict,
    epoch_type: str,
    tasks: Optional[List[str]],
) -> pd.DataFrame:
    """Load and concatenate all per-probe CSV files for one epoch type.

    Each CSV has one row per channel per probe. Metadata columns + marker
    columns (named ``{marker}_{agg_method}``) are preserved as-is.

    Parameters
    ----------
    project_root : Path
    cfg : Dict
    epoch_type : str
        One of ``"evoked"``, ``"state"``, ``"sleep"``.
    tasks : Optional[List[str]]
        Task filter (None = all tasks).

    Returns
    -------
    pd.DataFrame
        Concatenated probe data; empty DataFrame if no files match.
    """
    features_root = resolve_path(project_root, cfg["project"]["features_root"])
    # No `ses-*` segment in this project layout.
    pattern = f"sub-*/eeg/junifer_aggregated/*_{epoch_type}.csv"

    csv_files = sorted(features_root.glob(pattern))
    assert csv_files, (
        f"No CSV files found for epoch_type='{epoch_type}' under {features_root}. "
        f"Run the 2.aggregate_probes pipeline first."
    )

    # Drop *_agg.csv (these are ROI-aggregated outputs from step 2; we want
    # the per-channel files for topomap rendering).
    csv_files = [f for f in csv_files if not f.name.endswith(f"_agg.csv")
                 and not f.name.endswith(f"_agg_{epoch_type}.csv")]

    dfs: List[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f)
        if tasks is not None and "task" in df.columns:
            if df["task"].iloc[0] not in tasks:
                continue
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def detect_marker_columns(df: pd.DataFrame, agg_method: str) -> List[str]:
    """Detect marker columns by their aggregation-method suffix.

    Excludes known metadata columns; returns only columns ending with
    ``_{agg_method}`` (e.g. ``_mean``).

    Parameters
    ----------
    df : pd.DataFrame
    agg_method : str
        Suffix used by step 2 aggregation, e.g. ``"mean"``.

    Returns
    -------
    list of str
    """
    suffix = f"_{agg_method}"
    return [
        c for c in df.columns
        if c not in METADATA_COLS and c.endswith(suffix)
    ]


def strip_agg_suffix(col: str, agg_method: str) -> str:
    """Strip the aggregation-method suffix from a marker column name."""
    suffix = f"_{agg_method}"
    return col[: -len(suffix)] if col.endswith(suffix) else col


# =============================================================================
# CONDITION AVERAGING
# =============================================================================

def compute_group_means(
    df: pd.DataFrame,
    group_col: str,
    marker_col: str,
    conditions: List[str],
    channels: List[str],
    min_probes: int,
) -> Dict[str, np.ndarray]:
    """Compute group-level mean per channel via two-level averaging.

    Level 1: average across probes within each (subject, condition, channel).
    Level 2: average the subject-level means across subjects.

    Two-level averaging removes the confound of subjects with many probes
    dominating the group mean.

    Parameters
    ----------
    df : pd.DataFrame
        Per-probe long-format data.
    group_col : str
        Column to stratify by (here always ``"content"``).
    marker_col : str
        Marker column to average.
    conditions : list of str
        Condition labels to compute.
    channels : list of str
        Ordered list of channel names for the output arrays.
    min_probes : int
        Minimum probes per (subject, condition, channel) to include.

    Returns
    -------
    dict
        ``condition → 1-D array of length len(channels)``; NaN for channels
        without enough probes in any subject.
    """
    if marker_col not in df.columns:
        return {}

    result: Dict[str, np.ndarray] = {}

    for cond in conditions:
        subset = df[df[group_col] == cond][["subject", "channel", marker_col]]
        if subset.empty:
            continue

        # Level 1: per-subject mean across probes (per channel).
        subj_means = (
            subset.groupby(["subject", "channel"])[marker_col]
            .agg(n="count", mean="mean")
            .reset_index()
        )
        subj_means = subj_means[subj_means["n"] >= min_probes]
        if subj_means.empty:
            continue

        # Level 2: grand mean per channel.
        grand = subj_means.groupby("channel")["mean"].mean()
        result[cond] = grand.reindex(channels).values

    return result


# =============================================================================
# MNE SETUP
# =============================================================================

def create_mne_info(channels: List[str], montage_path: Path) -> mne.Info:
    """Create MNE Info with the project montage.

    Parameters
    ----------
    channels : list of str
        EEG channel names present in the data.
    montage_path : Path
        Path to the montage file (BVEF).

    Returns
    -------
    mne.Info
    """
    montage = mne.channels.read_custom_montage(str(montage_path), coord_frame="head")
    info = mne.create_info(ch_names=list(channels), sfreq=256.0, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore", verbose=False)
    return info


# =============================================================================
# TOPOMAP UTILITIES
# =============================================================================

def plot_single_topomap(
    ax: plt.Axes,
    values: Optional[np.ndarray],
    info: mne.Info,
    channels: List[str],
    title: str,
    vlim: Tuple[float, float],
    cmap: str,
    contours: int,
) -> Optional[plt.cm.ScalarMappable]:
    """Plot one topomap onto an existing axes.

    Parameters
    ----------
    ax : plt.Axes
    values : Optional[np.ndarray]
        Per-channel values aligned to ``channels``. ``None`` ⇒ blank panel.
    info : mne.Info
    channels : list of str
        Channel names corresponding to ``values``.
    title : str
    vlim : tuple of float
    cmap : str
    contours : int

    Returns
    -------
    Optional[ScalarMappable]
        Image returned by ``mne.viz.plot_topomap`` (for colorbar attachment),
        or ``None`` if the panel was skipped.
    """
    ax.set_title(title, fontsize=8.5, pad=3)

    if values is None or not np.any(np.isfinite(values)):
        ax.axis("off")
        return None

    valid_ch = [
        ch for ch, val in zip(channels, values)
        if ch in info.ch_names and np.isfinite(val)
    ]
    if len(valid_ch) < 4:
        ax.axis("off")
        return None

    picks = mne.pick_channels(info.ch_names, include=valid_ch)
    info_sub = mne.pick_info(info, picks, copy=True)

    ch_to_val = dict(zip(channels, values))
    plot_vals = np.array([ch_to_val[ch] for ch in info_sub.ch_names])

    im, _ = mne.viz.plot_topomap(
        plot_vals,
        info_sub,
        axes=ax,
        show=False,
        cmap=cmap,
        vlim=vlim,
        contours=contours,
        sensors=True,
        extrapolate="head",
        image_interp="cubic",
        res=128,
        sphere="auto",
        outlines="head",
        border="mean",
        ch_type="eeg",
    )
    return im


def compute_vlim(
    arrays: List[Optional[np.ndarray]],
    symmetric: bool = False,
    percentile: float = 5.0,
) -> Tuple[float, float]:
    """Compute robust color limits across multiple value arrays.

    Parameters
    ----------
    arrays : list of Optional[np.ndarray]
    symmetric : bool
        Force ``vmin = -vmax``.
    percentile : float
        Tail percentile to clip (e.g., ``5`` ⇒ 5th–95th percentile range).

    Returns
    -------
    tuple of float
    """
    finite = [
        a[np.isfinite(a)] for a in arrays
        if a is not None and np.any(np.isfinite(a))
    ]
    if not finite:
        return (-1.0, 1.0)
    all_vals = np.concatenate(finite)

    vmin = float(np.percentile(all_vals, percentile))
    vmax = float(np.percentile(all_vals, 100.0 - percentile))

    if symmetric:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    return vmin, vmax


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    im: plt.cm.ScalarMappable,
    label: str = "",
) -> None:
    """Append a thin colorbar to the right of ``ax``.

    Parameters
    ----------
    fig : plt.Figure
    ax : plt.Axes
    im : ScalarMappable
    label : str
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# =============================================================================
# FIGURE CONSTRUCTION
# =============================================================================

def create_topomap_figure(
    cond_means: Dict[str, np.ndarray],
    channels: List[str],
    info: mne.Info,
    marker_name: str,
    epoch_type: str,
    task_label: str,
    cfg: Dict,
) -> plt.Figure:
    """Build a single-row topomap figure for one marker × task combination.

    Layout
    ------
    One row × N+1 columns where N = number of conditions:
        ``[ref] [other1] [other2] ... [otherN-1] [non-ref − ref]``

    For the binary on-task vs off-task case used in this project, the row is
    ``[onTask] [offTask] [offTask − onTask]`` (3 panels).

    Parameters
    ----------
    cond_means : dict
        ``condition → per-channel array``.
    channels : list of str
    info : mne.Info
    marker_name : str
    epoch_type : str
    task_label : str
    cfg : Dict

    Returns
    -------
    plt.Figure
    """
    conditions = cfg["conditions"]
    ref_cond = cfg["reference_condition"]
    other_conds = [c for c in conditions if c != ref_cond]
    topomap_size = cfg["figure"]["topomap_size"]
    contours = cfg["figure"]["contours"]
    cmap_abs = cfg["figure"]["colormap"]
    cmap_diff = cfg["figure"]["colormap_diff"]

    ref_vals = cond_means.get(ref_cond)
    cond_diffs: Dict[str, Optional[np.ndarray]] = {
        c: (cond_means[c] - ref_vals) if (cond_means.get(c) is not None and ref_vals is not None) else None
        for c in other_conds
    }

    abs_vlim = compute_vlim([cond_means.get(c) for c in conditions])
    diff_vlim = compute_vlim(list(cond_diffs.values()), symmetric=True)

    n_cols = len(conditions) + len(other_conds)  # one diff panel per non-ref condition
    n_rows = 1

    fig_w = topomap_size * n_cols + 1.0
    fig_h = topomap_size * n_rows + 1.4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.4, "wspace": 0.10},
        squeeze=False,
    )
    axes = axes[0]  # collapse the row dimension for indexing

    fig.suptitle(
        f"{marker_name}   |   {epoch_type}   |   {task_label}",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )

    last_im_abs = None
    last_im_diff = None

    # Absolute panels: ref first, then other conditions.
    for col_i, cond in enumerate(conditions):
        ax = axes[col_i]
        vals = cond_means.get(cond)
        title = f"{cond}\n(ref)" if cond == ref_cond else cond
        im = plot_single_topomap(ax, vals, info, channels, title,
                                 abs_vlim, cmap_abs, contours)
        if im is not None:
            last_im_abs = im
        color = "#2ca02c" if cond == ref_cond else "black"
        ax.set_title(title, fontsize=8.5, pad=3, color=color)

    # Diff panels (always vs reference).
    base_diff_col = len(conditions)
    for offset, cond in enumerate(other_conds):
        ax = axes[base_diff_col + offset]
        diff_vals = cond_diffs.get(cond)
        title = f"{cond}\n− {ref_cond}"
        im = plot_single_topomap(ax, diff_vals, info, channels, title,
                                 diff_vlim, cmap_diff, contours)
        if im is not None:
            last_im_diff = im

    if last_im_abs is not None:
        add_colorbar(fig, axes[len(conditions) - 1], last_im_abs)
    if last_im_diff is not None:
        add_colorbar(fig, axes[-1], last_im_diff, label="Δ")

    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(cfg: Dict, project_root: Path, args: argparse.Namespace) -> None:
    """Main pipeline: load data → compute means → save figures.

    Parameters
    ----------
    cfg : Dict
    project_root : Path
    args : argparse.Namespace
        CLI arguments (may override epoch_type and marker filters).
    """
    results_root = resolve_path(project_root, cfg["project"]["results_root"])
    montage_path = resolve_path(project_root, cfg["project"]["montage_file"])
    agg_method = cfg["aggregation_method"]
    min_probes = cfg["min_probes_per_condition"]
    conditions = cfg["conditions"]
    condition_column = cfg["condition_column"]
    output_formats = cfg["figure"]["formats"]

    assert montage_path.exists(), f"Montage file not found: {montage_path}"

    epoch_types = [args.epoch_type] if args.epoch_type else cfg["epoch_types"]
    task_combinations: Dict[str, Optional[List[str]]] = cfg["task_combinations"]

    for epoch_type in epoch_types:
        print(f"\n{'='*70}")
        print(f"Epoch type: {epoch_type}")
        print(f"{'='*70}")

        all_data = load_all_probe_data(project_root, cfg, epoch_type, tasks=None)
        if all_data.empty:
            print(f"  [SKIP] No data found for epoch_type={epoch_type}")
            continue

        all_marker_cols = detect_marker_columns(all_data, agg_method)
        if not all_marker_cols:
            print(f"  [SKIP] No marker columns found (agg_method='{agg_method}')")
            continue

        if args.marker:
            all_marker_cols = [c for c in all_marker_cols if args.marker in c]
            assert all_marker_cols, (
                f"No marker columns matching --marker '{args.marker}'. "
                f"Available: {detect_marker_columns(all_data, agg_method)}"
            )

        channels = sorted(all_data["channel"].dropna().unique())
        info = create_mne_info(channels, montage_path)

        print(f"  Markers:   {len(all_marker_cols)}")
        print(f"  Channels:  {len(channels)}")
        print(f"  Subjects:  {all_data['subject'].nunique()}")
        print(f"  Probes:    {len(all_data) // len(channels)}")

        for task_label, task_filter in task_combinations.items():
            print(f"\n  Task combo: {task_label}")

            df = all_data if task_filter is None else \
                all_data[all_data["task"].isin(task_filter)]

            if df.empty:
                print(f"    [SKIP] No data after task filter")
                continue

            out_dir = results_root / task_label / epoch_type
            out_dir.mkdir(parents=True, exist_ok=True)

            for marker_col in all_marker_cols:
                marker_name = strip_agg_suffix(marker_col, agg_method)
                print(f"    Marker: {marker_name}", end="  ", flush=True)

                cond_means = compute_group_means(
                    df=df,
                    group_col=condition_column,
                    marker_col=marker_col,
                    conditions=conditions,
                    channels=channels,
                    min_probes=min_probes,
                )

                n_cond = sum(v is not None for v in cond_means.values())
                if n_cond == 0:
                    print("→ no data, skip")
                    continue

                fig = create_topomap_figure(
                    cond_means=cond_means,
                    channels=channels,
                    info=info,
                    marker_name=marker_name,
                    epoch_type=epoch_type,
                    task_label=task_label,
                    cfg=cfg,
                )

                for fmt in output_formats:
                    out_path = out_dir / f"{marker_name}.{fmt}"
                    fig.savefig(out_path, dpi=cfg["figure"]["dpi"],
                                bbox_inches="tight", facecolor="white")
                plt.close(fig)
                print(f"→ saved ({n_cond} conditions)")

    print(f"\nDone. Figures saved to: {results_root}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Visualize Junifer EEG marker topomaps by mental state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  All markers, all epoch types:
    python visualize_junifer_topomaps.py

  Only sleep epochs:
    python visualize_junifer_topomaps.py --epoch-type sleep

  Single marker (partial match):
    python visualize_junifer_topomaps.py --marker power_alpha
        """,
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epoch-type", dest="epoch_type", default=None,
        choices=["evoked", "state", "sleep"],
        help="Process only this epoch type",
    )
    parser.add_argument(
        "--marker", default=None,
        help="Process only markers whose name contains this substring",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = get_project_root()

    run_pipeline(cfg, project_root, args)


if __name__ == "__main__":
    main()
