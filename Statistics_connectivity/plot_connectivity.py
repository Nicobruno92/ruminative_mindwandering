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

PAIR_SEPARATOR = "-"

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
        abs_vals = np.abs(value_matrix)
        if np.all(np.isnan(abs_vals)):
            abs_max = 1.0
        else:
            abs_max = float(np.nanmax(abs_vals))
            if abs_max == 0:
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
    abs_vals = np.abs(value_matrix)
    if np.all(np.isnan(abs_vals)):
        abs_max = 1.0
    else:
        abs_max = float(np.nanmax(abs_vals))
        if abs_max == 0:
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
# CIRCULAR CONNECTIVITY PLOTS
# =============================================================================

# Channel positions for topographic ordering (degrees, counterclockwise from top=90°)
# Layout: Fz at top (90°), left hemisphere 90°→270°, occipital at bottom (270°),
# right hemisphere 270°→90° (through 0°). All 64 BioSemi channels have unique angles,
# evenly spaced at 5.625° intervals to prevent node overlap in circle plots.
#
# Counterclockwise order starting from Fz (frontal midline at top):
# Fz → left-frontal → left-temporal → left-parietal → occipital-midline →
# right-parietal → right-temporal → right-frontal → Fp2
CHANNEL_ANGLES = {
    # Frontal midline at top
    "Fz":   90,
    # Left frontal
    "Fp1":  96, "AF7": 101, "AF3": 107, "F7": 113, "F5": 118,
    "F3":  124, "F1":  129, "AFz": 135,
    # Left fronto-central / temporal
    "FC1": 141, "FC3": 146, "FC5": 152, "FT7": 158, "FT9": 163,
    # Left central
    "T7":  169, "C5":  174, "C3":  180, "C1":  186,
    # Left centro-parietal
    "CP5": 191, "CP3": 197, "CP1": 203, "TP7": 208, "TP9": 214,
    # Left parieto-occipital
    "P7":  219, "P5":  225, "P3":  231, "P1":  236,
    "PO7": 242, "PO3": 248, "O1":  253,
    # Posterior midline (bottom)
    "CPz": 259, "Cz":  264, "Pz":  270, "POz": 276,
    "Oz":  281, "Iz":  287,
    # Right parieto-occipital
    "O2":  293, "PO4": 298, "PO8": 304,
    "P2":  309, "P4":  315, "P6":  321, "P8":  326,
    # Right centro-parietal
    "TP10": 332, "TP8": 338, "CP6": 343, "CP4": 349, "CP2": 354,
    # Right central
    "T8":    0, "C6":    6, "C4":   11, "C2":   17,
    # Right fronto-central / temporal
    "FT10":  23, "FT8":  28, "FC6":  34, "FC4":  39, "FC2":  45,
    # Right frontal
    "F2":   51, "F4":   56, "F6":   62, "F8":   68,
    "AF4":  73, "AF8":  79, "Fp2":  84,
    # FCz not in data but kept for compatibility (between Fp2 and Fz)
    "FCz":  87,
}

# ROI angular positions (center of each region, arranged on the circle)
ROI_ANGLES = {
    "frontal_left": 60,
    "frontal_right": 120,
    "central_left": 20,
    "central_right": 160,
    "midline": 270,
    "posterior_left": 310,
    "posterior_right": 230,
}

# ROI colors for circle plot
ROI_COLORS = {
    "frontal_left": "#E74C3C",
    "frontal_right": "#E67E22",
    "central_left": "#2ECC71",
    "central_right": "#1ABC9C",
    "posterior_left": "#3498DB",
    "posterior_right": "#9B59B6",
    "midline": "#F1C40F",
}


def plot_connectivity_circle(
    results_df: pd.DataFrame,
    node_names: Optional[List[str]] = None,
    node_angles: Optional[Dict[str, float]] = None,
    node_colors: Optional[Dict[str, str]] = None,
    node_group: Optional[Dict[str, str]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    title: str = "Connectivity Circle",
    linewidth_range: Tuple[float, float] = (0.5, 4.0),
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (10, 10),
    show_nonsig: bool = False,
    nonsig_alpha: float = 0.05,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a circular connectivity diagram (chord plot).

    Nodes are arranged on a circle, significant connections are drawn as
    curved lines between them, color-coded by t-statistic direction and
    width-coded by effect magnitude.

    Parameters
    ----------
    results_df : pd.DataFrame
        LMM results with connection_id, t_statistic, significance columns.
    node_names : Optional[List[str]]
        Node names in desired circular order. None = auto-detect from data.
    node_angles : Optional[Dict[str, float]]
        Angle (degrees) for each node. None = use defaults or evenly space.
    node_colors : Optional[Dict[str, str]]
        Color for each node. None = assign by ROI group or default palette.
    node_group : Optional[Dict[str, str]]
        Group label for each node (for color grouping).
    value_col : str
        Column for edge color values.
    sig_col : str
        Column indicating significance.
    title : str
        Plot title.
    linewidth_range : Tuple[float, float]
        Min and max linewidth for connections.
    cmap : str
        Colormap for connection values.
    figsize : Tuple[float, float]
        Figure size.
    show_nonsig : bool
        If True, draw non-significant connections as faint grey lines.
    nonsig_alpha : float
        Opacity for non-significant connections.
    save_path : Optional[str]
        Path to save figure.
    dpi : int
        Figure DPI.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import FancyArrowPatch, PathPatch
    import matplotlib.patches as mpatches

    # ── Extract unique nodes ─────────────────────────────────────────────
    all_nodes = set()
    for conn_id in results_df["connection_id"]:
        parts = conn_id.split(PAIR_SEPARATOR)
        if len(parts) == 2:
            all_nodes.add(parts[0].strip())
            all_nodes.add(parts[1].strip())

    if node_names is None:
        # Try topographic ordering
        if node_angles is None:
            node_angles = _get_default_angles(all_nodes)
        node_names = sorted(all_nodes, key=lambda n: node_angles.get(n, 0))

    n_nodes = len(node_names)

    # ── Compute node positions on the circle ─────────────────────────────
    if node_angles is None:
        # Evenly spaced
        angles_deg = np.linspace(90, 90 + 360, n_nodes, endpoint=False)
        node_angle_map = {name: ang for name, ang in zip(node_names, angles_deg)}
    else:
        node_angle_map = node_angles

    radius = 1.0
    node_positions = {}
    for name in node_names:
        ang = np.radians(node_angle_map.get(name, 0))
        node_positions[name] = (radius * np.cos(ang), radius * np.sin(ang))

    # ── Assign node colors ───────────────────────────────────────────────
    if node_colors is None and node_group is not None:
        group_names = sorted(set(node_group.values()))
        palette = list(ROI_COLORS.values())
        # Extend palette if needed
        while len(palette) < len(group_names):
            palette.extend(plt.cm.Set3(np.linspace(0, 1, 12)).tolist())
        group_color_map = {g: palette[i % len(palette)] for i, g in enumerate(group_names)}
        node_colors = {n: group_color_map[node_group[n]] for n in node_names if n in node_group}

    if node_colors is None:
        node_colors = {n: "#555555" for n in node_names}

    # ── Create figure ────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")

    # ── Draw connections ─────────────────────────────────────────────────
    sig_rows = results_df[results_df[sig_col] == True] if sig_col in results_df.columns else results_df
    all_values = results_df[value_col].dropna().values

    if len(all_values) > 0:
        abs_max = max(abs(v) for v in all_values)
        if abs_max == 0:
            abs_max = 1.0
    else:
        abs_max = 1.0

    colormap = plt.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

    # Draw non-significant first (background)
    if show_nonsig:
        nonsig_rows = results_df[results_df[sig_col] == False] if sig_col in results_df.columns else pd.DataFrame()
        for _, row in nonsig_rows.iterrows():
            parts = row["connection_id"].split(PAIR_SEPARATOR)
            if len(parts) != 2:
                continue
            n1, n2 = parts[0].strip(), parts[1].strip()
            if n1 not in node_positions or n2 not in node_positions:
                continue
            _draw_curved_edge(
                ax, node_positions[n1], node_positions[n2],
                color="#cccccc", linewidth=0.3, alpha=nonsig_alpha,
            )

    # Draw significant connections
    for _, row in sig_rows.iterrows():
        parts = row["connection_id"].split(PAIR_SEPARATOR)
        if len(parts) != 2:
            continue
        n1, n2 = parts[0].strip(), parts[1].strip()
        if n1 not in node_positions or n2 not in node_positions:
            continue

        val = row[value_col] if pd.notna(row[value_col]) else 0
        color = colormap(norm(val))
        # Scale linewidth by absolute value
        lw_min, lw_max = linewidth_range
        lw = lw_min + (lw_max - lw_min) * (abs(val) / abs_max)

        _draw_curved_edge(
            ax, node_positions[n1], node_positions[n2],
            color=color, linewidth=lw, alpha=0.8,
        )

    # ── Draw nodes ───────────────────────────────────────────────────────
    for name in node_names:
        x, y = node_positions[name]
        color = node_colors.get(name, "#555555")
        ax.plot(x, y, "o", color=color, markersize=8, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)

        # Label
        ang = node_angle_map.get(name, 0)
        label_radius = radius + 0.08
        lx = label_radius * np.cos(np.radians(ang))
        ly = label_radius * np.sin(np.radians(ang))
        ha = "left" if lx >= 0 else "right"
        rotation = ang % 360
        if 90 < rotation < 270:
            rotation += 180
        ax.text(
            lx, ly, name, fontsize=7, ha=ha, va="center",
            rotation=rotation % 360, rotation_mode="anchor",
        )

    # ── Draw outer ring arcs per group ───────────────────────────────────
    if node_group is not None:
        _draw_group_arcs(ax, node_names, node_angle_map, node_group, radius=1.12)

    # ── Colorbar ─────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(value_col.replace("_", " ").title(), fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold", y=1.05)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")

    return fig


def plot_roi_connectivity_circle(
    results_df: pd.DataFrame,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    title: str = "ROI Connectivity Circle",
    figsize: Tuple[float, float] = (10, 10),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a circular connectivity diagram for ROI-level results.

    Convenience wrapper around plot_connectivity_circle with ROI-specific
    defaults (colors, positions, labels).

    Parameters
    ----------
    results_df : pd.DataFrame
        ROI-level LMM results.
    value_col : str
        Column for edge color.
    sig_col : str
        Significance column.
    title : str
        Plot title.
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
    return plot_connectivity_circle(
        results_df=results_df,
        node_names=list(DEFAULT_ROI_ORDER),
        node_angles=ROI_ANGLES,
        node_colors=ROI_COLORS,
        value_col=value_col,
        sig_col=sig_col,
        title=title,
        linewidth_range=(1.5, 6.0),
        figsize=figsize,
        show_nonsig=True,
        nonsig_alpha=0.08,
        save_path=save_path,
        dpi=dpi,
    )


def plot_channel_connectivity_circle(
    results_df: pd.DataFrame,
    rois: Optional[Dict[str, List[str]]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    title: str = "Channel Connectivity Circle",
    figsize: Tuple[float, float] = (14, 14),
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a circular connectivity diagram for channel-level results.

    Channels are arranged topographically on the circle. If ROI definitions
    are provided, channels are color-coded by ROI membership and grouped
    with arcs on the outer ring.

    Parameters
    ----------
    results_df : pd.DataFrame
        Channel-level LMM results.
    rois : Optional[Dict[str, List[str]]]
        ROI definitions for grouping/coloring channels.
    value_col : str
        Column for edge color.
    sig_col : str
        Significance column.
    title : str
        Plot title.
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
    # Build channel-to-ROI mapping for coloring
    node_group = None
    node_colors = None
    if rois is not None:
        node_group = {}
        for roi_name, channels in rois.items():
            for ch in channels:
                node_group[ch] = roi_name
        # Build color map from ROI colors
        node_colors = {}
        for ch, roi in node_group.items():
            node_colors[ch] = ROI_COLORS.get(roi, "#555555")

    return plot_connectivity_circle(
        results_df=results_df,
        node_angles=CHANNEL_ANGLES,
        node_colors=node_colors,
        node_group=node_group,
        value_col=value_col,
        sig_col=sig_col,
        title=title,
        linewidth_range=(0.5, 3.0),
        figsize=figsize,
        show_nonsig=False,
        save_path=save_path,
        dpi=dpi,
    )


def plot_multi_band_circle_grid(
    all_results: Dict[str, pd.DataFrame],
    level: str = "roi",
    rois: Optional[Dict[str, List[str]]] = None,
    value_col: str = "t_statistic",
    sig_col: str = "significant_fdr",
    suptitle: str = "wSMI Connectivity Circles",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a row of circular connectivity diagrams, one per band.

    Parameters
    ----------
    all_results : Dict[str, pd.DataFrame]
        {band: results_df} mapping.
    level : str
        "roi" or "channel".
    rois : Optional[Dict[str, List[str]]]
        ROI definitions (used for channel-level coloring).
    value_col : str
        Column for color values.
    sig_col : str
        Significance column.
    suptitle : str
        Super title.
    save_path : Optional[str]
        Save path.
    dpi : int
        DPI.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    bands = list(all_results.keys())
    n_bands = len(bands)

    fig_width = 8 * n_bands
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))

    for idx, band in enumerate(bands):
        ax = fig.add_subplot(1, n_bands, idx + 1, aspect="equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

        band_label = BAND_LABELS.get(band, band)
        results = all_results[band]

        # Extract nodes and build positions
        all_nodes = set()
        for conn_id in results["connection_id"]:
            parts = conn_id.split(PAIR_SEPARATOR)
            if len(parts) == 2:
                all_nodes.update(p.strip() for p in parts)

        if level == "roi":
            node_names = [r for r in DEFAULT_ROI_ORDER if r in all_nodes]
            angle_map = ROI_ANGLES
            color_map = ROI_COLORS
            lw_range = (1.5, 6.0)
        else:
            angle_map = _get_default_angles(all_nodes)
            node_names = sorted(all_nodes, key=lambda n: angle_map.get(n, 0))
            color_map = {}
            if rois:
                for roi_name, channels in rois.items():
                    for ch in channels:
                        color_map[ch] = ROI_COLORS.get(roi_name, "#555555")
            lw_range = (0.5, 3.0)

        # Compute positions
        node_positions = {}
        for name in node_names:
            ang = np.radians(angle_map.get(name, 0))
            node_positions[name] = (np.cos(ang), np.sin(ang))

        # Get significant connections
        sig_mask = results[sig_col] == True if sig_col in results.columns else pd.Series([True] * len(results))
        sig_rows = results[sig_mask]
        all_values = results[value_col].dropna().values
        abs_max = max(abs(v) for v in all_values) if len(all_values) > 0 else 1.0
        if abs_max == 0:
            abs_max = 1.0

        colormap = plt.get_cmap("RdBu_r")
        norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

        # Draw significant edges
        for _, row in sig_rows.iterrows():
            parts = row["connection_id"].split(PAIR_SEPARATOR)
            if len(parts) != 2:
                continue
            n1, n2 = parts[0].strip(), parts[1].strip()
            if n1 not in node_positions or n2 not in node_positions:
                continue
            val = row[value_col] if pd.notna(row[value_col]) else 0
            color = colormap(norm(val))
            lw_min, lw_max = lw_range
            lw = lw_min + (lw_max - lw_min) * (abs(val) / abs_max)
            _draw_curved_edge(ax, node_positions[n1], node_positions[n2],
                              color=color, linewidth=lw, alpha=0.8)

        # Draw nodes
        for name in node_names:
            x, y = node_positions[name]
            c = color_map.get(name, "#555555")
            ax.plot(x, y, "o", color=c, markersize=6, markeredgecolor="white",
                    markeredgewidth=0.3, zorder=5)
            # Labels
            ang = angle_map.get(name, 0)
            lx = 1.08 * np.cos(np.radians(ang))
            ly = 1.08 * np.sin(np.radians(ang))
            ha = "left" if lx >= 0 else "right"
            fs = 8 if level == "roi" else 5
            label_text = ROI_SHORT_LABELS.get(name, name) if level == "roi" else name
            ax.text(lx, ly, label_text, fontsize=fs, ha=ha, va="center")

        n_sig = len(sig_rows)
        ax.set_title(f"wSMI {band_label}\n({n_sig} sig. edges)",
                     fontsize=12, fontweight="bold")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {save_path}")

    return fig


# =============================================================================
# CIRCLE PLOT HELPERS
# =============================================================================

def _draw_curved_edge(
    ax: plt.Axes,
    pos1: Tuple[float, float],
    pos2: Tuple[float, float],
    color: str = "grey",
    linewidth: float = 1.0,
    alpha: float = 0.5,
) -> None:
    """
    Draw a curved (Bezier) edge between two points, curving toward the center.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    pos1 : Tuple[float, float]
        Start position (x, y).
    pos2 : Tuple[float, float]
        End position (x, y).
    color : str
        Line color.
    linewidth : float
        Line width.
    alpha : float
        Opacity.
    """
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches

    x1, y1 = pos1
    x2, y2 = pos2

    # Control point: midpoint pulled toward center (origin)
    # The pull factor controls how curved the line is
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Distance between points determines curvature
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    pull = 0.3 + 0.3 * (dist / 2.0)  # More distant nodes → more curvature

    ctrl_x = mid_x * (1 - pull)
    ctrl_y = mid_y * (1 - pull)

    verts = [(x1, y1), (ctrl_x, ctrl_y), (x2, y2)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)

    patch = mpatches.PathPatch(
        path, facecolor="none", edgecolor=color,
        linewidth=linewidth, alpha=alpha, capstyle="round",
    )
    ax.add_patch(patch)


def _get_default_angles(nodes: set) -> Dict[str, float]:
    """
    Get default angular positions for nodes using CHANNEL_ANGLES or even spacing.

    Parameters
    ----------
    nodes : set
        Node names.

    Returns
    -------
    Dict[str, float]
        Angle (degrees) for each node.
    """
    angles = {}
    unknown = []
    for n in nodes:
        if n in CHANNEL_ANGLES:
            angles[n] = CHANNEL_ANGLES[n]
        elif n in ROI_ANGLES:
            angles[n] = ROI_ANGLES[n]
        else:
            unknown.append(n)

    # Evenly space unknown nodes in remaining gaps
    if unknown:
        used_angles = set(angles.values())
        for i, n in enumerate(sorted(unknown)):
            angle = (360 / (len(unknown) + 1)) * (i + 1)
            while angle in used_angles:
                angle += 1
            angles[n] = angle
            used_angles.add(angle)

    return angles


def _draw_group_arcs(
    ax: plt.Axes,
    node_names: List[str],
    node_angles: Dict[str, float],
    node_group: Dict[str, str],
    radius: float = 1.12,
) -> None:
    """
    Draw colored arcs on the outer ring to indicate ROI groupings.

    Parameters
    ----------
    ax : plt.Axes
        Axes to draw on.
    node_names : List[str]
        Ordered node names.
    node_angles : Dict[str, float]
        Angle for each node.
    node_group : Dict[str, str]
        Group label for each node.
    radius : float
        Radius of the outer arc.
    """
    from matplotlib.patches import Arc

    # Group nodes by their group label
    groups = {}
    for name in node_names:
        grp = node_group.get(name)
        if grp is None:
            continue
        if grp not in groups:
            groups[grp] = []
        groups[grp].append(node_angles.get(name, 0))

    for grp, grp_angles in groups.items():
        if len(grp_angles) < 1:
            continue
        ang_min = min(grp_angles) - 3
        ang_max = max(grp_angles) + 3
        color = ROI_COLORS.get(grp, "#555555")

        arc = Arc(
            (0, 0), 2 * radius, 2 * radius,
            angle=0, theta1=ang_min, theta2=ang_max,
            color=color, linewidth=4, alpha=0.7,
        )
        ax.add_patch(arc)


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
            entry = {
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
            }
            # NBS columns (channel-level)
            if "p_value_nbs" in row.index:
                entry["p_value_nbs"] = row.get("p_value_nbs", np.nan)
                entry["significant_nbs"] = row.get("significant_nbs", False)
                entry["nbs_component_id"] = row.get("nbs_component_id", -1)
            # Max-t permutation columns
            if "p_value_perm" in row.index:
                entry["p_value_perm"] = row.get("p_value_perm", np.nan)
                entry["significant_perm"] = row.get("significant_perm", False)
            rows.append(entry)

    summary = pd.DataFrame(rows)

    if save_path:
        summary.to_csv(save_path, index=False)
        print(f"  Saved summary: {save_path}")

    return summary
