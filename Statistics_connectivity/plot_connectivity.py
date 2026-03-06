"""
Visualization module for connectivity statistics pipeline.

Produces paper-style connectivity matrix heatmaps showing LMM results
with FDR-corrected significance masking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from itertools import combinations_with_replacement


# =============================================================================
# CONSTANTS
# =============================================================================

PAIR_SEPARATOR = "--"

# Greek band labels for display
BAND_LABELS = {
    "delta": "δ",
    "theta": "θ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
}

# Default ROI display order (anterior → posterior)
DEFAULT_ROI_ORDER = [
    "frontal_left",
    "frontal_right",
    "central_left",
    "midline",
    "central_right",
    "posterior_left",
    "posterior_right",
]

# Short ROI labels for axis ticks
ROI_SHORT_LABELS = {
    "frontal_left": "L Frontal",
    "frontal_right": "R Frontal",
    "central_left": "L Central",
    "central_right": "R Central",
    "posterior_left": "L Posterior",
    "posterior_right": "R Posterior",
    "midline": "Midline",
}


# =============================================================================
# SINGLE CONTRAST MATRIX
# =============================================================================

def plot_contrast_matrix(
    results_df: pd.DataFrame,
    roi_order: Optional[List[str]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    title: str = "Connectivity Contrast",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdBu_r",
    mask_nonsig: bool = True,
    figsize: Tuple[float, float] = (7, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot a single connectivity contrast matrix (ROI × ROI).

    Non-significant connections are masked (greyed out).

    Parameters
    ----------
    results_df : pd.DataFrame
        LMM results with connection_id, t_statistic, p_value_fdr, significant_fdr.
    roi_order : Optional[List[str]]
        Order of ROIs on axes. None = default.
    value_col : str
        Column to plot as color values.
    sig_col : str
        Column indicating significance (bool).
    title : str
        Plot title.
    vmin, vmax : Optional[float]
        Color scale limits. None = symmetric around 0.
    cmap : str
        Colormap name.
    mask_nonsig : bool
        If True, grey out non-significant connections.
    figsize : Tuple[float, float]
        Figure size.
    ax : Optional[plt.Axes]
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    if roi_order is None:
        roi_order = DEFAULT_ROI_ORDER

    n_rois = len(roi_order)

    # Build matrix
    value_matrix = np.full((n_rois, n_rois), np.nan)
    sig_matrix = np.zeros((n_rois, n_rois), dtype=bool)

    roi_to_idx = {r: i for i, r in enumerate(roi_order)}

    for _, row in results_df.iterrows():
        conn_id = row["connection_id"]
        parts = conn_id.split(PAIR_SEPARATOR)
        if len(parts) != 2:
            continue

        r1, r2 = parts[0].strip(), parts[1].strip()
        if r1 not in roi_to_idx or r2 not in roi_to_idx:
            continue

        i, j = roi_to_idx[r1], roi_to_idx[r2]
        val = row[value_col] if pd.notna(row[value_col]) else np.nan
        sig = bool(row[sig_col]) if pd.notna(row.get(sig_col, np.nan)) else False

        value_matrix[i, j] = val
        value_matrix[j, i] = val  # Symmetric
        sig_matrix[i, j] = sig
        sig_matrix[j, i] = sig

    # Set color limits
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(value_matrix))
        if np.isnan(abs_max) or abs_max == 0:
            abs_max = 1.0
        vmin = -abs_max
        vmax = abs_max

    # Create figure
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot heatmap
    display_matrix = value_matrix.copy()

    # Create masked version for non-significant
    if mask_nonsig:
        masked_matrix = np.ma.array(display_matrix, mask=~sig_matrix)
        bg_matrix = np.ma.array(display_matrix, mask=sig_matrix)

        # Plot background (non-significant) in grey
        ax.imshow(
            np.ones_like(display_matrix) * 0.5,
            cmap="Greys",
            vmin=0,
            vmax=1,
            alpha=0.15,
            aspect="equal",
        )

        # Plot significant values
        im = ax.imshow(
            masked_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
    else:
        im = ax.imshow(
            display_matrix,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

    # Add grid lines
    for i in range(n_rois + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
        ax.axvline(i - 0.5, color="white", linewidth=0.5)

    # Labels
    tick_labels = [ROI_SHORT_LABELS.get(r, r) for r in roi_order]
    ax.set_xticks(range(n_rois))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rois))
    ax.set_yticklabels(tick_labels, fontsize=9)

    ax.set_xlabel("ROI 1", fontsize=10)
    ax.set_ylabel("ROI 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Colorbar
    if created_fig:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(value_col.replace("_", " ").title(), fontsize=10)

    if created_fig:
        fig.tight_layout()

    return fig


# =============================================================================
# MULTI-BAND CONTRAST GRID
# =============================================================================

def plot_multi_band_grid(
    all_results: Dict[str, pd.DataFrame],
    roi_order: Optional[List[str]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    suptitle: str = "wSMI Connectivity: LMM Results",
    cmap: str = "RdBu_r",
    shared_scale: bool = True,
    figsize_per_panel: Tuple[float, float] = (5, 4.5),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a grid of contrast matrices, one per frequency band.

    Inspired by Figure 3 from the source-space connectivity paper.

    Parameters
    ----------
    all_results : Dict[str, pd.DataFrame]
        Mapping: {band_name: results_df}. Each results_df has columns
        connection_id, t_statistic, p_value_fdr, significant_fdr.
    roi_order : Optional[List[str]]
        ROI ordering for axes.
    value_col : str
        Column to plot.
    sig_col : str
        Column for significance masking.
    suptitle : str
        Figure super-title.
    cmap : str
        Colormap.
    shared_scale : bool
        If True, use same color limits across all bands.
    figsize_per_panel : Tuple[float, float]
        Size of each panel.
    save_path : Optional[str]
        Path to save figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    bands = list(all_results.keys())
    n_bands = len(bands)

    # Compute shared color limits
    vmin, vmax = None, None
    if shared_scale:
        all_vals = []
        for df in all_results.values():
            vals = df[value_col].dropna().values
            if len(vals) > 0:
                all_vals.extend(vals)
        if all_vals:
            abs_max = max(abs(v) for v in all_vals)
            vmin, vmax = -abs_max, abs_max

    # Create figure
    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(figsize_per_panel[0] * n_bands, figsize_per_panel[1]),
        squeeze=False,
    )

    for idx, band in enumerate(bands):
        ax = axes[0, idx]
        band_label = BAND_LABELS.get(band, band)

        plot_contrast_matrix(
            results_df=all_results[band],
            roi_order=roi_order,
            value_col=value_col,
            sig_col=sig_col,
            title=f"wSMI {band_label}",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# =============================================================================
# CHANNEL-LEVEL MATRIX (larger matrix)
# =============================================================================

def plot_channel_contrast_matrix(
    results_df: pd.DataFrame,
    channel_order: Optional[List[str]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    title: str = "Channel-level Connectivity",
    cmap: str = "RdBu_r",
    mask_nonsig: bool = True,
    figsize: Tuple[float, float] = (14, 12),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a channel × channel connectivity matrix.

    For channel-level analysis with many connections.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with connection_id (channel pairs like "Fp1--F3").
    channel_order : Optional[List[str]]
        Order of channels. None = alphabetical from data.
    value_col : str
        Column to plot.
    sig_col : str
        Significance column.
    title : str
        Title.
    cmap : str
        Colormap.
    mask_nonsig : bool
        Grey out non-significant.
    figsize : Tuple[float, float]
        Figure size.
    save_path : Optional[str]
        Save path.
    dpi : int
        DPI.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    # Extract unique channels
    all_channels = set()
    for conn_id in results_df["connection_id"]:
        parts = conn_id.split(PAIR_SEPARATOR)
        if len(parts) == 2:
            all_channels.add(parts[0].strip())
            all_channels.add(parts[1].strip())

    if channel_order is None:
        channel_order = sorted(all_channels)

    n_ch = len(channel_order)
    ch_to_idx = {ch: i for i, ch in enumerate(channel_order)}

    # Build matrices
    value_matrix = np.full((n_ch, n_ch), np.nan)
    sig_matrix = np.zeros((n_ch, n_ch), dtype=bool)

    for _, row in results_df.iterrows():
        parts = row["connection_id"].split(PAIR_SEPARATOR)
        if len(parts) != 2:
            continue
        ch1, ch2 = parts[0].strip(), parts[1].strip()
        if ch1 not in ch_to_idx or ch2 not in ch_to_idx:
            continue
        i, j = ch_to_idx[ch1], ch_to_idx[ch2]
        val = row[value_col] if pd.notna(row[value_col]) else np.nan
        sig = bool(row[sig_col]) if pd.notna(row.get(sig_col, np.nan)) else False
        value_matrix[i, j] = val
        value_matrix[j, i] = val
        sig_matrix[i, j] = sig
        sig_matrix[j, i] = sig

    # Color limits
    abs_max = np.nanmax(np.abs(value_matrix))
    if np.isnan(abs_max) or abs_max == 0:
        abs_max = 1.0

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    display = value_matrix.copy()

    if mask_nonsig:
        masked = np.ma.array(display, mask=~sig_matrix)
        ax.imshow(
            np.ones_like(display) * 0.5, cmap="Greys",
            vmin=0, vmax=1, alpha=0.15, aspect="equal",
        )
        im = ax.imshow(masked, cmap=cmap, vmin=-abs_max, vmax=abs_max, aspect="equal")
    else:
        im = ax.imshow(display, cmap=cmap, vmin=-abs_max, vmax=abs_max, aspect="equal")

    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channel_order, rotation=90, fontsize=6)
    ax.set_yticks(range(n_ch))
    ax.set_yticklabels(channel_order, fontsize=6)
    ax.set_title(title, fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label(value_col.replace("_", " ").title(), fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    return fig


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def create_summary_table(
    all_results: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create summary table across all bands.

    Parameters
    ----------
    all_results : Dict[str, pd.DataFrame]
        {band: results_df} mapping.
    save_path : Optional[str]
        Path to save CSV.

    Returns
    -------
    pd.DataFrame
        Combined summary table.
    """
    rows = []
    for band, df in all_results.items():
        for _, row in df.iterrows():
            rows.append({
                "band": band,
                "connection_id": row["connection_id"],
                "t_statistic": row.get("t_statistic", np.nan),
                "p_value": row.get("p_value", np.nan),
                "p_value_fdr": row.get("p_value_fdr", np.nan),
                "coefficient": row.get("coefficient", np.nan),
                "significant_fdr": row.get("significant_fdr", False),
                "n_observations": row.get("n_observations", 0),
                "n_subjects": row.get("n_subjects", 0),
                "converged": row.get("converged", False),
            })

    summary = pd.DataFrame(rows)

    if save_path:
        summary.to_csv(save_path, index=False)
        print(f"  Saved summary: {save_path}")

    return summary
