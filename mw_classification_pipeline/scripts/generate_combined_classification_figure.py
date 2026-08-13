#!/usr/bin/env python
"""
Generate combined classification figures showing WS and LOSO results.

Outputs (in results/combined_figures/):
    group_ws.{png,svg}        — WS group-level AUC per dimension
    group_loso.{png,svg}      — LOSO group-level AUC per dimension
    individual_ws.{png,svg}   — WS per-subject AUC per dimension
    individual_loso.{png,svg} — LOSO per-subject AUC per dimension
    marker_direction_all_dimensions.{png,svg}
                              — WS vs LOSO SHAP direction per marker, all
                                dimensions. Consumes the tables written by
                                scripts/feature_consistency_analysis.py and is
                                skipped with a message if they are absent.
    spatial_comparison_ws_loso.{png,svg}
                              — WS (row 1) vs LOSO (row 2) spatial (per-electrode)
                                decoding topomaps, one column per dimension
                                (the 5 canonical dimensions). Reads per_channel_metrics.csv
                                from results/MW_Classification/SpatialDecoding/;
                                skipped with a message for any dimension missing
                                on either pipeline.
    group_comparison_spatial_combined.{png,svg}
                              — WS | LOSO "Global Decoding" (True/Residualized/
                                Shuffled AUC density) next to "Spatial Decoding"
                                (topomap) per dimension, one combined panel per
                                pipeline. Pure Matplotlib (see
                                plot_group_spatial_combined docstring for why).
Each dimension has a consistent color used across all four figures.
Full (true labels) = dimension color; Shuffled = grey.

Usage (from project root):
    /path/to/miniforge3/envs/plots/bin/python \
        mw_classification_pipeline/scripts/generate_combined_classification_figure.py

Must run in the `plots` env, not `ML`: the batched writer below uses the
`kaleido.Kaleido` context manager, which exists only in kaleido >= 1.0. The ML
env pins kaleido 0.2.1, whose API is the old `fig.write_image()`, so every
figure builds there and then the run dies at the final write step.
"""

# =============================================================================
# Imports
# =============================================================================

import asyncio
import os
import sys
import warnings
import pickle
import yaml
import kaleido as _kaleido
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# Shared project color palette (single source of truth). Resolved path-relative
# to this script so it works regardless of the current working directory.
PALETTE = yaml.safe_load(
    open(Path(__file__).resolve().parents[2] / "color_palette.yaml")
)
DIM_COLORS = PALETTE["dimensions"]

RESULTS_ROOT    = Path("mw_classification_pipeline/results/MW_Classification")
CONSISTENCY_DIR = Path("mw_classification_pipeline/results/feature_consistency")
OUTPUT_DIR      = Path("mw_classification_pipeline/results/combined_figures")
PROBE_DATA_PATH = Path("results/Behavior/probe_data/probe_level_aggregated_data.csv")

# Spatial (per-electrode searchlight) decoding results, one merged
# per_channel_metrics.csv per {pipeline}/{dimension}/{family}/{model}.
SPATIAL_RESULTS_ROOT = RESULTS_ROOT / "SpatialDecoding"
SPATIAL_FAMILY = "all"
SPATIAL_MODEL  = "rf"

# WS-vs-LOSO feature scatter figures: show the union of each pipeline's
# top-N features (by its own ranking metric).
SHAP_SCATTER_TOP_N = 10

# Per-dimension color, consistent across every figure
DIMENSIONS = [
    {
        "key":          "on_off",
        "ws_dir":       "on_vs_off_within_median",
        "loso_dir":     "ON_vs_OFF_within_median",
        "label":        "On/Off-Task",
        "color":        DIM_COLORS["onoff"],   # red
        "label_source": "onoff",
    },
    {
        "key":          "valence",
        "ws_dir":       "valence_within_median",
        "loso_dir":     "valence_within_median",
        "label":        "Valence",
        "color":        DIM_COLORS["valence"],   # blue
        "label_source": "valence",
    },
    {
        "key":          "selfother",
        "ws_dir":       "selfother_within_median",
        "loso_dir":     "selfother_within_median",
        "label":        "Self/Other",
        "color":        DIM_COLORS["selfother"],   # green
        "label_source": "selfother",
    },
    {
        "key":          "time",
        "ws_dir":       "time_within_median",
        "loso_dir":     "time_within_median",
        "label":        "Time",
        "color":        DIM_COLORS["time"],   # purple
        "label_source": "time",
    },
    {
        "key":          "confidence",
        "ws_dir":       "confidence_within_median",
        "loso_dir":     "confidence_within_median",
        "label":        "Confidence",
        "color":        DIM_COLORS["confidence"],   # orange
        "label_source": "confidence",
    },
]

PERM_COLOR   = PALETTE["neutral"]["permutation"]   # gray
CHANCE       = 0.5
AUC_RANGE    = (0.35, 1.0)

# Residualized-contrast result directories, one per canonical dimension (same
# folder name under WithinSubject/ and LOSO/ for every dimension — see
# docs/superpowers/specs/2026-07-30-cross-dimension-residualized-contrasts-design.md).
# Only the *_with_residualized figures read these; the canonical figures above
# never do.
RESIDUALIZED_DIR = {
    "on_off":     "onoff_within_median_res",
    "valence":    "valence_within_median_res",
    "time":       "time_within_median_res",
    "selfother":  "selfother_within_median_res",
    "confidence": "confidence_within_median_res",
}

# SQ_DIMENSIONS (valence_sq / time_sq) removed 2026-08-13 together with the
# quadratic construct itself — see Stats_andrillon/config_andrillon.yaml. The
# group-level row order is therefore just the canonical 5 dimensions; there are
# no longer quadratic rows interleaved after their linear parents.
GROUP_ROW_DIMENSIONS = list(DIMENSIONS)

RESIDUALIZED_ALPHA = 0.30  # well below True's 0.72, so it reads as a translucent layer
# Thin aggregate marks (inner donut ring, stacked bar segment) need more
# opacity than a full violin area to stay legible at small size — still
# visually distinct from the fully-opaque canonical marks they sit beside.
RESIDUALIZED_RING_ALPHA = 0.55
BW_GROUP     = 0.4   # Scott multiplier for group-level KDE
BW_INDIV     = 0.5   # Scott multiplier for individual KDE
MIN_BW_INDIV = 0.025 # minimum bandwidth in AUC units (prevents spike artefacts)
DPI          = 200

# =============================================================================
# Data loading
# =============================================================================


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def load_group_data(
    pipeline: str, dim_info: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (true_mean_aucs, perm_mean_aucs) arrays across all runs."""
    if pipeline == "ws":
        base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
    else:
        base = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"

    def _collect(subdir: str) -> np.ndarray:
        vals = []
        for csv_f in sorted(base.glob(f"{subdir}/run_*/*_summary.csv")):
            df = _safe_read_csv(csv_f)
            if df is not None and "mean_auc" in df.columns:
                vals.extend(df["mean_auc"].dropna().tolist())
        return np.array(vals)

    return _collect("true_runs"), _collect("permuted_runs")


def load_subject_data(
    pipeline: str, dim_info: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (true_df, perm_df) with columns [subject, auc].

    WS  : subject metric column is ``mean_auc``.
    LOSO: metric column is ``auc`` per fold — averaged per subject per run.
    """
    if pipeline == "ws":
        base       = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        true_glob  = "true_runs/run_*/*_ws_subject_metrics.csv"
        perm_glob  = "permuted_runs/run_*/*_ws_subject_metrics.csv"
        auc_col    = "mean_auc"
        agg_folds  = False
    else:
        base       = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
        true_glob  = "true_runs/run_*/*_loso_subject_metrics.csv"
        perm_glob  = "permuted_runs/run_*/*_loso_subject_metrics.csv"
        auc_col    = "auc"
        agg_folds  = True

    def _collect(glob_pat: str) -> pd.DataFrame:
        dfs = []
        for csv_f in sorted(base.glob(glob_pat)):
            df = _safe_read_csv(csv_f)
            if df is None or "subject" not in df.columns or auc_col not in df.columns:
                continue
            df["subject"] = df["subject"].astype(str)
            if agg_folds:
                df = df.groupby("subject")[auc_col].mean().reset_index()
            df = (
                df[["subject", auc_col]]
                .rename(columns={auc_col: "auc"})
                .dropna(subset=["auc"])
            )
            dfs.append(df)
        if not dfs:
            return pd.DataFrame(columns=["subject", "auc"])
        return pd.concat(dfs, ignore_index=True)

    return _collect(true_glob), _collect(perm_glob)


def _residualized_dim_info(dim_info: dict) -> dict:
    """
    Synthetic dim_info pointing at a dimension's residualized-contrast result
    directory, so load_group_data / load_subject_data can be reused unchanged.
    """
    res_dir = RESIDUALIZED_DIR[dim_info["key"]]
    return {**dim_info, "ws_dir": res_dir, "loso_dir": res_dir}


# =============================================================================
# Statistics
# =============================================================================


def empirical_pvalue(true_vals: np.ndarray, perm_vals: np.ndarray) -> float:
    if len(true_vals) == 0 or len(perm_vals) == 0:
        return np.nan
    return float(np.mean(perm_vals >= np.median(true_vals)))


def fdr_correct(pvals: list[float]) -> np.ndarray:
    arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(arr)
    adj = arr.copy()
    if valid.sum() > 0:
        _, adj[valid], _, _ = multipletests(arr[valid], method="fdr_bh")
    return adj


def format_pval(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


def sig_stars(p: float) -> str:
    if np.isnan(p) or p >= 0.05:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _fmt_q(q: float) -> str:
    """Compact FDR-corrected q-value, e.g. '<.001' or '0.028'."""
    if np.isnan(q):
        return "n/a"
    return "<.001" if q < 0.001 else f"{q:.3f}"


def _true_res_annotation_text(
    t: np.ndarray, p_adj: float, t_res: np.ndarray, res_p_adj: float,
) -> str:
    """
    Compact 2-line "Full / Residualized" label — mean AUC, significance
    stars, FDR-corrected p (p_FDR, matching the project-wide convention in
    CLAUDE.md rather than the q-value name from genomics/Storey) — meant to
    sit INSIDE a violin panel rather than a separate margin column, so it
    stays short by design ("Full"/"Res." rather than the full words). Shared
    by plot_group_level_with_residualized and
    plot_group_comparison_with_residualized so the two figures read
    identically.
    """
    mean_t     = float(np.mean(t))     if len(t)     else float("nan")
    mean_t_res = float(np.mean(t_res)) if len(t_res) else float("nan")
    line1 = f"<b>Full {mean_t:.2f}{sig_stars(p_adj)}</b> p_FDR={_fmt_q(p_adj)}"
    line2 = f"<i>Res. {mean_t_res:.2f}{sig_stars(res_p_adj)}</i> p_FDR={_fmt_q(res_p_adj)}"
    return f"{line1}<br>{line2}"


def _true_res_annotation_lines(
    t: np.ndarray, p_adj: float, t_res: np.ndarray, res_p_adj: float,
) -> tuple[str, str]:
    """
    Plain-text (no HTML) two-line "Full / Residualized" label — same content
    and wording as :func:`_true_res_annotation_text`, for Matplotlib
    ``ax.text`` rather than a Plotly annotation (used by
    :func:`plot_group_spatial_combined`, the one Matplotlib figure among the
    otherwise-Plotly group-level annotations).
    """
    mean_t     = float(np.mean(t))     if len(t)     else float("nan")
    mean_t_res = float(np.mean(t_res)) if len(t_res) else float("nan")
    line1 = f"Full {mean_t:.2f}{sig_stars(p_adj)}  p_FDR={_fmt_q(p_adj)}"
    line2 = f"Res. {mean_t_res:.2f}{sig_stars(res_p_adj)}  p_FDR={_fmt_q(res_p_adj)}"
    return line1, line2


# =============================================================================
# KDE helper
# =============================================================================


def _kde_fill(
    ax: plt.Axes,
    values: np.ndarray,
    color: str,
    alpha: float,
    bw: float = BW_GROUP,
    min_bw: float = 0.0,
) -> None:
    """Plot a unit-height filled KDE on *ax*.

    Parameters
    ----------
    min_bw : float
        Minimum bandwidth in AUC data units.  Prevents spike artefacts when
        a subject's AUC values have very low variance across runs.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return
    x = np.linspace(AUC_RANGE[0], AUC_RANGE[1], 400)

    # Compute Scott's bandwidth estimate (scipy's baseline before bw multiplier)
    std = float(np.std(v))
    n   = len(v)
    if std > 0 and min_bw > 0:
        scott_h   = std * (n ** -0.2)        # Scott's rule: σ × n^(-1/5)
        bw_needed = min_bw / scott_h          # factor to reach min_bw in data units
        bw_actual = max(bw, bw_needed)
    else:
        bw_actual = bw

    try:
        kde = gaussian_kde(v, bw_method=bw_actual)
    except np.linalg.LinAlgError:
        return
    y = kde(x)
    peak = y.max()
    if peak > 0:
        y /= peak
    ax.fill_between(x, y, alpha=alpha, color=color, linewidth=0)
    ax.plot(x, y, color=color, alpha=min(1.0, alpha + 0.25), linewidth=1.5)


def _clean_kde_ax(
    ax: plt.Axes,
    show_xticks: bool = False,
    show_xlabel: bool = False,
    fixed_ylim: bool = False,
) -> None:
    ax.set_xlim(AUC_RANGE)
    if fixed_ylim:
        # Fixed scale for individual cells: all peaks are unit-normalised,
        # so 1.25 gives head-room for star annotations.
        ax.set_ylim(0, 1.25)
    else:
        ax.set_ylim(bottom=0)
    # Thin baseline visible in every cell (mirrors reference style)
    ax.axhline(0, color="#BBBBBB", linewidth=0.6, zorder=0)
    ax.axvline(CHANCE, color="#333333", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    if show_xticks:
        ax.set_xticks([0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.5", ".75", "1.0"], fontsize=7)
        if show_xlabel:
            ax.set_xlabel("AUC", fontsize=9)
    else:
        ax.tick_params(bottom=False, labelbottom=False)


# =============================================================================
# Figure A — Group-level (one per pipeline)
# =============================================================================


def plot_group_level(pipeline: str, title_prefix: str) -> go.Figure:
    """
    Plotly horizontal half-violin figure (5 rows, one per dimension).
    True (dim color) overlaid on permuted (grey).  Mirrors reference style.
    """
    n_dims = len(DIMENSIONS)

    raw_pvals, dim_data = [], []
    for dim_info in DIMENSIONS:
        t, p = load_group_data(pipeline, dim_info)
        raw_pvals.append(empirical_pvalue(t, p))
        dim_data.append((t, p))
        print(
            f"  [{pipeline.upper()}] {dim_info['label']}: "
            f"n_true={len(t)}, n_perm={len(p)}, med_true={np.median(t):.3f}"
        )

    adj_pvals = fdr_correct(raw_pvals)

    fig = make_subplots(
        rows=n_dims, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    for i, (dim_info, (t, p), p_raw, p_adj) in enumerate(
        zip(DIMENSIONS, dim_data, raw_pvals, adj_pvals)
    ):
        row = i + 1
        color = dim_info["color"]

        # Permuted
        fig.add_trace(go.Violin(
            x=p, y=[0] * len(p),
            orientation="h", side="positive",
            line_color=PERM_COLOR, fillcolor=PERM_COLOR,
            opacity=0.55, points=False, width=1.0, bandwidth=0.015,
            name="Shuffled", legendgroup="Shuffled",
            showlegend=(i == 0),
        ), row=row, col=1)

        # True
        fig.add_trace(go.Violin(
            x=t, y=[0] * len(t),
            orientation="h", side="positive",
            line_color=color, fillcolor=color,
            opacity=0.72, points=False, width=1.0, bandwidth=0.015,
            name="Full", legendgroup="Full",
            showlegend=(i == 0),
        ), row=row, col=1)

        # Chance line
        x_ref = "x" if row == 1 else f"x{row}"
        y_ref = "y" if row == 1 else f"y{row}"
        fig.add_shape(
            type="line", x0=0.5, x1=0.5, y0=0, y1=1,
            xref=x_ref, yref=f"{y_ref} domain",
            line=dict(color="black", dash="dash", width=1.5),
        )

        # Dimension label (left)
        fig.add_annotation(
            text=f"<b>{dim_info['label']}</b>",
            xref="paper", yref=f"{y_ref} domain",
            x=-0.01, y=0.5,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=12, family="Times New Roman"),
        )

        # P-value (right)
        stars = sig_stars(p_adj)
        line1 = f"{stars} {format_pval(p_raw)}".strip() if stars else format_pval(p_raw)
        line2 = f"(FDR: {format_pval(p_adj)})"
        fig.add_annotation(
            text=f"<b>{line1}</b><br>{line2}",
            xref="paper", yref=f"{y_ref} domain",
            x=1.01, y=0.5,
            xanchor="left", yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=10, family="Times New Roman"),
        )

        # Hide y-tick labels per row
        y_ax = "yaxis" if row == 1 else f"yaxis{row}"
        fig.layout[y_ax].showticklabels = False
        fig.layout[y_ax].showgrid = False
        fig.layout[y_ax].zeroline = False

    # x-axis: show ticks only on last row
    for row in range(1, n_dims + 1):
        x_ax = "xaxis" if row == 1 else f"xaxis{row}"
        fig.layout[x_ax].range = [0.35, 1.0]
        fig.layout[x_ax].showticklabels = (row == n_dims)
        if row == n_dims:
            fig.layout[x_ax].tickmode = "array"
            fig.layout[x_ax].tickvals = [0.5, 0.75, 1.0]
            fig.layout[x_ax].ticktext = ["0.5", ".75", "1.0"]
            fig.layout[x_ax].title = dict(text="AUC", font=dict(size=12))

    fig.update_layout(
        title=dict(
            text=f"{title_prefix} — Group-level AUC per Dimension",
            font=dict(size=14, family="Times New Roman"),
        ),
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=max(400, n_dims * 90 + 100),
        width=720,
        margin=dict(l=130, r=230, t=80, b=60),
        font=dict(family="Times New Roman", size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
    )
    return fig


# =============================================================================
# Figure A2 — Group-level + residualized overlay (one per pipeline)
# =============================================================================


def plot_group_level_with_residualized(pipeline: str, title_prefix: str) -> go.Figure:
    """
    Same layout as :func:`plot_group_level`, with a third half-violin per row:
    the True-AUC distribution from that dimension's residualized-contrast
    re-run (RESIDUALIZED_DIR). Drawn in the dimension color at
    RESIDUALIZED_ALPHA — well below True's opacity — so it reads as a
    translucent layer rather than a competing solid distribution.

    Rows are GROUP_ROW_DIMENSIONS (the 5 canonical dimensions). No figure title, and the
    True/Residualized mean-AUC + significance readout (_true_res_annotation_text)
    sits INSIDE each row's own panel (top-right corner) instead of a separate
    margin column — compact by design, not just cropped.
    """
    row_dims = GROUP_ROW_DIMENSIONS
    n_dims   = len(row_dims)

    raw_pvals, dim_data = [], []
    res_raw_pvals, res_dim_data = [], []
    for dim_info in row_dims:
        t, p = load_group_data(pipeline, dim_info)
        raw_pvals.append(empirical_pvalue(t, p))
        dim_data.append((t, p))

        t_res, p_res = load_group_data(pipeline, _residualized_dim_info(dim_info))
        res_raw_pvals.append(empirical_pvalue(t_res, p_res))
        res_dim_data.append((t_res, p_res))
        print(
            f"  [{pipeline.upper()}] {dim_info['label']} (residualized): "
            f"n_true={len(t_res)}, n_perm={len(p_res)}, "
            f"med_true={np.median(t_res) if len(t_res) else np.nan:.3f}"
        )

    adj_pvals     = fdr_correct(raw_pvals)
    res_adj_pvals = fdr_correct(res_raw_pvals)

    fig = make_subplots(
        rows=n_dims, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
    )

    for i, dim_info in enumerate(row_dims):
        row   = i + 1
        color = dim_info["color"]
        t, p         = dim_data[i]
        t_res, p_res = res_dim_data[i]
        p_adj     = adj_pvals[i]
        res_p_adj = res_adj_pvals[i]

        # Permuted
        fig.add_trace(go.Violin(
            x=p, y=[0] * len(p),
            orientation="h", side="positive",
            line_color=PERM_COLOR, fillcolor=PERM_COLOR,
            opacity=0.55, points=False, width=1.0, bandwidth=0.015,
            name="Shuffled", legendgroup="Shuffled",
            showlegend=(i == 0),
        ), row=row, col=1)

        # True
        fig.add_trace(go.Violin(
            x=t, y=[0] * len(t),
            orientation="h", side="positive",
            line_color=color, fillcolor=color,
            opacity=0.72, points=False, width=1.0, bandwidth=0.015,
            name="Full", legendgroup="Full",
            showlegend=(i == 0),
        ), row=row, col=1)

        # Residualized (True) — same color, low alpha, drawn last (on top)
        fig.add_trace(go.Violin(
            x=t_res, y=[0] * len(t_res),
            orientation="h", side="positive",
            line_color=color, fillcolor=color,
            opacity=RESIDUALIZED_ALPHA, points=False, width=1.0, bandwidth=0.015,
            name="Residualized", legendgroup="Residualized",
            showlegend=(i == 0),
        ), row=row, col=1)

        # Chance line
        x_ref = "x" if row == 1 else f"x{row}"
        y_ref = "y" if row == 1 else f"y{row}"
        fig.add_shape(
            type="line", x0=0.5, x1=0.5, y0=0, y1=1,
            xref=x_ref, yref=f"{y_ref} domain",
            line=dict(color="black", dash="dash", width=1.5),
        )

        # Dimension label (left)
        fig.add_annotation(
            text=f"<b>{dim_info['label']}</b>",
            xref="paper", yref=f"{y_ref} domain",
            x=-0.01, y=0.5,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=11, family="Times New Roman"),
        )

        # True/Residualized mean AUC + significance — inside the panel
        # (top-right corner), not a separate margin column.
        fig.add_annotation(
            text=_true_res_annotation_text(t, p_adj, t_res, res_p_adj),
            xref=f"{x_ref} domain", yref=f"{y_ref} domain",
            x=0.97, y=0.80,
            xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(color=color, size=8, family="Times New Roman"),
        )

        # Hide y-tick labels per row
        y_ax = "yaxis" if row == 1 else f"yaxis{row}"
        fig.layout[y_ax].showticklabels = False
        fig.layout[y_ax].showgrid = False
        fig.layout[y_ax].zeroline = False

    # x-axis: show ticks only on last row
    for row in range(1, n_dims + 1):
        x_ax = "xaxis" if row == 1 else f"xaxis{row}"
        fig.layout[x_ax].range = [0.35, 1.0]
        fig.layout[x_ax].showticklabels = (row == n_dims)
        if row == n_dims:
            fig.layout[x_ax].tickmode = "array"
            fig.layout[x_ax].tickvals = [0.5, 0.75, 1.0]
            fig.layout[x_ax].ticktext = ["0.5", ".75", "1.0"]
            fig.layout[x_ax].title = dict(text="AUC", font=dict(size=12))

    fig.update_layout(
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=max(430, n_dims * 85 + 55),
        width=620,
        margin=dict(l=140, r=20, t=55, b=50),
        font=dict(family="Times New Roman", size=12),
        # yanchor="bottom" at y=1.0 (not "top"): grows the legend up into
        # margin.t, not down into row 1 -- see plot_group_comparison_with_residualized.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0,
            xanchor="left", x=0, font=dict(size=10),
        ),
    )
    return fig


# =============================================================================
# Figure B — Individual-level (one per pipeline)
# =============================================================================


def _compute_subject_pvals(
    sorted_subjects: list[str], all_true: dict, all_perm: dict
) -> tuple[dict, dict]:
    """Return (raw_pvals, adj_pvals) both as {dim_key: {subject: float}}."""
    raw_pvals: dict = {}
    adj_pvals: dict = {}
    for dim_info in DIMENSIONS:
        dk = dim_info["key"]
        t_df = all_true.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        p_df = all_perm.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        raw_list, subj_list = [], []
        for s in sorted_subjects:
            tv = t_df.loc[t_df["subject"] == s, "auc"].values
            pv = p_df.loc[p_df["subject"] == s, "auc"].values
            raw_list.append(empirical_pvalue(tv, pv))
            subj_list.append(s)
        adj = fdr_correct(raw_list)
        raw_pvals[dk] = dict(zip(subj_list, raw_list))
        adj_pvals[dk] = dict(zip(subj_list, adj.tolist()))
    return raw_pvals, adj_pvals


def plot_individual(pipeline: str, title_prefix: str, stem: str) -> bool:
    """
    Per-subject × dimension half-violin figure using Plotly (mirrors reference style).

    Uses go.Violin(side='positive', orientation='h') for horizontal half-violins.
    Saves HTML + PNG + SVG to OUTPUT_DIR/{stem}.{ext}.
    Returns True if saved, False if no data.
    """
    print(f"\nLoading {pipeline.upper()} individual data…")
    all_true, all_perm = {}, {}
    for dim_info in DIMENSIONS:
        t_df, p_df = load_subject_data(pipeline, dim_info)
        all_true[dim_info["key"]] = t_df
        all_perm[dim_info["key"]] = p_df
        print(f"  {dim_info['label']}: n_true={len(t_df)}, n_perm={len(p_df)}")

    # --- Subject union: keep all subjects present in ANY dim with >= 5 runs ---
    subjs_per_dim: dict[str, set] = {}
    for dim_info in DIMENSIONS:
        dk = dim_info["key"]
        t_df = all_true.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        if t_df.empty:
            subjs_per_dim[dk] = set()
        else:
            counts = t_df.groupby("subject").size()
            subjs_per_dim[dk] = set(counts[counts >= 5].index.astype(str))

    all_subj_sets = [s for s in subjs_per_dim.values() if s]
    if not all_subj_sets:
        print(f"  Insufficient data for {pipeline}, skipping.")
        return False

    union_subjects = set.union(*all_subj_sets)
    sorted_subjects = sorted(union_subjects, key=lambda x: int(x) if x.isdigit() else x)
    n_subjs = len(sorted_subjects)

    if n_subjs == 0:
        print(f"  No subjects found for {pipeline}, skipping.")
        return False

    print(f"  Subjects (union across dims): {n_subjs} → {sorted_subjects}")

    # n_dims run per subject (for denominator in sig-bar)
    n_dims_per_subject = {
        s: sum(1 for dk in subjs_per_dim if s in subjs_per_dim[dk])
        for s in sorted_subjects
    }

    # y-positions: i + 0.2 (matches reference tick alignment)
    subj_y_map = {s: i + 0.2 for i, s in enumerate(sorted_subjects)}
    subj_idx_map = {s: i for i, s in enumerate(sorted_subjects)}  # integer index for bar
    n_dims       = len(DIMENSIONS)
    raw_pvals, adj_pvals = _compute_subject_pvals(sorted_subjects, all_true, all_perm)

    tick_labels = [str(int(s) if s.isdigit() else s) for s in sorted_subjects]

    # --- Build Plotly figure (row 1 = violins, row 2 = donuts) ---
    specs = [
        [{"type": "xy"}]     * (n_dims + 1),
        [{"type": "domain"}] * n_dims + [None],
    ]
    subplot_titles = [d["label"] for d in DIMENSIONS] + ["Total<br>Sig."]

    fig = make_subplots(
        rows=2, cols=n_dims + 1,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        row_heights=[0.85, 0.15],
        specs=specs,
    )

    # Color subplot title text per dimension; leave last col title blank (avoid legend collision)
    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color = dim_info["color"]
        fig.layout.annotations[i].font.size  = 14
    # Clear the auto-generated "Total Sig." annotation — we'll add it as a custom annotation
    if len(fig.layout.annotations) > n_dims:
        fig.layout.annotations[n_dims].text = ""

    sig_count_per_subject = {s: 0 for s in sorted_subjects}

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col   = col_idx + 1
        dk    = dim_info["key"]
        color = dim_info["color"]

        t_df = all_true.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        p_df = all_perm.get(dk, pd.DataFrame(columns=["subject", "auc"]))

        t_df = t_df[t_df["subject"].isin(sorted_subjects)].copy()
        p_df = p_df[p_df["subject"].isin(sorted_subjects)].copy()
        t_df["subj_y"] = t_df["subject"].map(subj_y_map)
        p_df["subj_y"] = p_df["subject"].map(subj_y_map)

        # Permuted violin (background, grey)
        fig.add_trace(go.Violin(
            x=p_df["auc"].values,
            y=p_df["subj_y"].values,
            legendgroup="Permuted",
            scalegroup=f"Permuted_{dk}",
            name="Permuted",
            line_color=PERM_COLOR,
            fillcolor=PERM_COLOR,
            opacity=0.4,
            showlegend=(col_idx == 0),
            points=False,
            width=0.8,
            bandwidth=0.04,
            side="positive",
            orientation="h",
        ), row=1, col=col)

        # True violin (foreground, dimension color)
        fig.add_trace(go.Violin(
            x=t_df["auc"].values,
            y=t_df["subj_y"].values,
            legendgroup="Full",
            scalegroup=f"True_{dk}",
            name="Full",
            line_color=color,
            fillcolor=color,
            opacity=0.7,
            showlegend=(col_idx == 0),
            points=False,
            width=0.8,
            bandwidth=0.04,
            meanline_visible=True,
            side="positive",
            orientation="h",
        ), row=1, col=col)

        # Dashed chance line spanning all subjects
        x_axis_name = "x" if col == 1 else f"x{col}"
        fig.add_shape(
            type="line", x0=0.5, x1=0.5, y0=0, y1=1,
            xref=x_axis_name, yref="y domain",
            line=dict(color="black", dash="dash"),
        )

        # Significance stars
        for subj in sorted_subjects:
            p_adj_val = adj_pvals.get(dk, {}).get(subj, np.nan)
            stars = sig_stars(p_adj_val)
            if stars:
                sig_count_per_subject[subj] += 1
                subj_y = subj_y_map[subj]
                fig.add_annotation(
                    x=0.99, y=subj_y,
                    text=f"<b>{stars}</b>",
                    showarrow=False,
                    xanchor="right", yanchor="middle",
                    row=1, col=col,
                    font=dict(color="black", size=14),
                )

        # Donut chart — denominator = subjects with data in THIS dim
        n_with_data = len([s for s in sorted_subjects if s in subjs_per_dim.get(dk, set())])
        n_sig_d = sum(
            1 for s in sorted_subjects
            if s in subjs_per_dim.get(dk, set())
            and not np.isnan(adj_pvals.get(dk, {}).get(s, np.nan))
            and adj_pvals.get(dk, {}).get(s, 1.0) < 0.05
        )
        pct_d = n_sig_d / n_with_data * 100 if n_with_data > 0 else 0
        fig.add_trace(go.Pie(
            labels=["Signif.", "Not"],
            values=[max(pct_d, 0.001), max(100 - pct_d, 0.001)],
            marker=dict(colors=[color, PERM_COLOR]),
            hole=0.4,
            textinfo="text",
            text=[f"{pct_d:.0f}%", ""],
            textfont=dict(size=16, color="white"),
            textposition="inside",
            showlegend=False,
            sort=False,
        ), row=2, col=col)

    # "Total Sig." column header (custom annotation — avoids collision with legend)
    fig.add_annotation(
        text="<b>Total<br>Sig.</b>",
        xref=f"x{n_dims + 1} domain", yref="paper",
        x=0.5, y=1.01,
        showarrow=False,
        font=dict(size=13, color="#333333"),
        xanchor="center", yanchor="bottom",
    )

    # Total significance bar — denominator = dims actually run per subject
    sig_pcts = [
        sig_count_per_subject[s] / n_dims_per_subject[s] * 100
        if n_dims_per_subject[s] > 0 else 0
        for s in sorted_subjects
    ]
    fig.add_trace(go.Bar(
        x=sig_pcts,
        y=[i + 0.2 for i in range(n_subjs)],
        orientation="h",
        marker_color=PALETTE["neutral"]["accent"],
        showlegend=False,
        text=[f"{v:.0f}%" if v > 0 else "" for v in sig_pcts],
        textposition="outside",
        textfont=dict(size=12, color="black"),
    ), row=1, col=n_dims + 1)

    # --- Layout ---
    total_height = max(500, n_subjs * 25 + 200)
    total_width  = 1600

    fig.update_layout(
        title_text=f"{title_prefix} — Individual-level AUC per Participant",
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=total_height,
        width=total_width,
        margin=dict(l=80, r=40, t=140, b=50),
        font=dict(family="Times New Roman", size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="Subject ID", font=dict(size=18)),
            automargin=True,
            tickmode="array",
            tickvals=[i + 0.2 for i in range(n_subjs)],
            ticktext=tick_labels,
            tickfont=dict(size=16),
            range=[-0.5, n_subjs],
        ),
    )

    # Shared y-axis ticks (all violin subplots share y)
    fig.update_yaxes(
        tickmode="array",
        tickvals=[i + 0.2 for i in range(n_subjs)],
        ticktext=tick_labels,
        range=[-0.5, n_subjs],
        row=1,
    )

    # x-axis range for violin columns — use direct layout assignment (matches reference)
    for j in range(n_dims):
        ax = "xaxis" if j == 0 else f"xaxis{j + 1}"
        fig.layout[ax].range = [0.25, 1.0]
        fig.layout[ax].tickmode = "array"
        fig.layout[ax].tickvals = [0.5, 0.75, 1.0]
        fig.layout[ax].ticktext = ["0.5", ".75", "1.0"]
        fig.layout[ax].showticklabels = True
        fig.layout[ax].tickfont = dict(size=12)

    # x-axis for sig-bar column — direct layout assignment
    bar_ax = f"xaxis{n_dims + 1}"
    fig.layout[bar_ax].title = dict(text="% Signif.", font=dict(size=16))
    fig.layout[bar_ax].tickfont = dict(size=12)
    fig.layout[bar_ax].showgrid = True
    fig.layout[bar_ax].tickmode = "linear"
    fig.layout[bar_ax].dtick = 25
    fig.layout[bar_ax].range = [0, 115]

    return fig, total_width, total_height


# =============================================================================
# Figure B2 — Individual-level + residualized overlay (one per pipeline)
# =============================================================================


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#RRGGBB' hex color to an 'rgba(r,g,b,a)' string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _shrink_domain(domain: dict[str, list[float]], factor: float) -> dict[str, list[float]]:
    """
    Shrink a Plotly domain dict toward its own center by `factor` (0-1).

    Used to nest a second pie trace inside a subplot cell already occupied by
    a first one, producing a concentric double-donut instead of two
    identically-sized pies stacked flat on top of each other.
    """
    def _shrink_axis(lo: float, hi: float) -> list[float]:
        center = (lo + hi) / 2
        half   = (hi - lo) / 2 * factor
        return [center - half, center + half]

    return {"x": _shrink_axis(*domain["x"]), "y": _shrink_axis(*domain["y"])}


def plot_individual_with_residualized(
    pipeline: str, title_prefix: str, stem: str
) -> tuple[go.Figure, int, int] | bool:
    """
    Same layout as :func:`plot_individual`, with one extra half-violin per
    subject/dimension cell: that subject's True-AUC distribution from the
    dimension's residualized-contrast re-run (RESIDUALIZED_DIR), drawn in the
    dimension color at RESIDUALIZED_ALPHA so it sits as a translucent layer
    behind the canonical True violin rather than a fourth distinct series.

    Per-cell canonical significance stars are unchanged from plot_individual
    (bold black, right edge of each panel). Residualized significance gets
    its own per-cell marker too — same tiered stars, pale dimension color, a
    few AUC-units to the left — so the aggregate marks below can actually be
    counted against something on the plot, not just read as a number with no
    visible per-subject referent. Both aggregates use their own denominator
    (residualized data covers a different, usually larger, subject pool than
    canonical — see subjs_per_dim_res / n_dims_per_subject_res below), and
    both print the literal number, not just an implied shape:
      - Donut: an inner, thinner, lower-alpha ring nested inside the
        canonical outer ring, always drawn as a full ring (colored + gray)
        so 0% significant reads as empty rather than vanishing, plus the
        literal % printed in the ring's empty center.
      - "Total Sig." bar: unchanged bar (length = canonical True rate), with
        the residualized rate appended as "(res Y%)" text rather than a
        second overlapping bar — two bars sharing a row are ambiguous to
        read when one is longer, shorter, or zero while the other isn't.
    """
    print(f"\nLoading {pipeline.upper()} individual data (+ residualized)…")
    all_true, all_perm, all_res, all_res_perm = {}, {}, {}, {}
    for dim_info in DIMENSIONS:
        t_df, p_df = load_subject_data(pipeline, dim_info)
        all_true[dim_info["key"]] = t_df
        all_perm[dim_info["key"]] = p_df
        r_df, r_perm_df = load_subject_data(pipeline, _residualized_dim_info(dim_info))
        all_res[dim_info["key"]] = r_df
        all_res_perm[dim_info["key"]] = r_perm_df
        print(f"  {dim_info['label']}: n_true={len(t_df)}, n_perm={len(p_df)}, n_res={len(r_df)}")

    # --- Subject union: keep all subjects present in ANY dim with >= 5 runs ---
    subjs_per_dim: dict[str, set] = {}
    for dim_info in DIMENSIONS:
        dk = dim_info["key"]
        t_df = all_true.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        if t_df.empty:
            subjs_per_dim[dk] = set()
        else:
            counts = t_df.groupby("subject").size()
            subjs_per_dim[dk] = set(counts[counts >= 5].index.astype(str))

    all_subj_sets = [s for s in subjs_per_dim.values() if s]
    if not all_subj_sets:
        print(f"  Insufficient data for {pipeline}, skipping.")
        return False

    union_subjects = set.union(*all_subj_sets)
    sorted_subjects = sorted(union_subjects, key=lambda x: int(x) if x.isdigit() else x)
    n_subjs = len(sorted_subjects)

    if n_subjs == 0:
        print(f"  No subjects found for {pipeline}, skipping.")
        return False

    print(f"  Subjects (union across dims): {n_subjs} → {sorted_subjects}")

    n_dims_per_subject = {
        s: sum(1 for dk in subjs_per_dim if s in subjs_per_dim[dk])
        for s in sorted_subjects
    }

    # Residualized data covers a different (usually larger) subject pool than
    # the canonical runs — tracked separately so the residualized % below is
    # always a literal count over subjects actually tested residualized, not
    # diluted or inflated by the canonical denominator.
    subjs_per_dim_res: dict[str, set] = {}
    for dim_info in DIMENSIONS:
        dk = dim_info["key"]
        r_df = all_res.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        if r_df.empty:
            subjs_per_dim_res[dk] = set()
        else:
            counts = r_df.groupby("subject").size()
            subjs_per_dim_res[dk] = set(counts[counts >= 5].index.astype(str)) & set(sorted_subjects)

    n_dims_per_subject_res = {
        s: sum(1 for dk in subjs_per_dim_res if s in subjs_per_dim_res[dk])
        for s in sorted_subjects
    }

    subj_y_map = {s: i + 0.2 for i, s in enumerate(sorted_subjects)}
    n_dims       = len(DIMENSIONS)
    raw_pvals, adj_pvals         = _compute_subject_pvals(sorted_subjects, all_true, all_perm)
    raw_pvals_res, adj_pvals_res = _compute_subject_pvals(sorted_subjects, all_res, all_res_perm)

    tick_labels = [str(int(s) if s.isdigit() else s) for s in sorted_subjects]

    specs = [
        [{"type": "xy"}]     * (n_dims + 1),
        [{"type": "domain"}] * n_dims + [None],
    ]
    subplot_titles = [d["label"] for d in DIMENSIONS] + ["Total<br>Sig."]

    fig = make_subplots(
        rows=2, cols=n_dims + 1,
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.015,
        vertical_spacing=0.05,
        row_heights=[0.85, 0.15],
        specs=specs,
    )

    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color = dim_info["color"]
        fig.layout.annotations[i].font.size  = 14
    if len(fig.layout.annotations) > n_dims:
        fig.layout.annotations[n_dims].text = ""

    sig_count_per_subject     = {s: 0 for s in sorted_subjects}
    sig_count_per_subject_res = {s: 0 for s in sorted_subjects}

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col   = col_idx + 1
        dk    = dim_info["key"]
        color = dim_info["color"]

        t_df = all_true.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        p_df = all_perm.get(dk, pd.DataFrame(columns=["subject", "auc"]))
        r_df = all_res.get(dk, pd.DataFrame(columns=["subject", "auc"]))

        t_df = t_df[t_df["subject"].isin(sorted_subjects)].copy()
        p_df = p_df[p_df["subject"].isin(sorted_subjects)].copy()
        r_df = r_df[r_df["subject"].isin(sorted_subjects)].copy()
        t_df["subj_y"] = t_df["subject"].map(subj_y_map)
        p_df["subj_y"] = p_df["subject"].map(subj_y_map)
        r_df["subj_y"] = r_df["subject"].map(subj_y_map)

        # Permuted violin (background, grey)
        fig.add_trace(go.Violin(
            x=p_df["auc"].values,
            y=p_df["subj_y"].values,
            legendgroup="Permuted",
            scalegroup=f"Permuted_{dk}",
            name="Permuted",
            line_color=PERM_COLOR,
            fillcolor=PERM_COLOR,
            opacity=0.4,
            showlegend=(col_idx == 0),
            points=False,
            width=0.8,
            bandwidth=0.04,
            side="positive",
            orientation="h",
        ), row=1, col=col)

        # True violin (foreground, dimension color)
        fig.add_trace(go.Violin(
            x=t_df["auc"].values,
            y=t_df["subj_y"].values,
            legendgroup="Full",
            scalegroup=f"True_{dk}",
            name="Full",
            line_color=color,
            fillcolor=color,
            opacity=0.7,
            showlegend=(col_idx == 0),
            points=False,
            width=0.8,
            bandwidth=0.04,
            meanline_visible=True,
            side="positive",
            orientation="h",
        ), row=1, col=col)

        # Residualized violin (translucent overlay, same color, drawn last)
        fig.add_trace(go.Violin(
            x=r_df["auc"].values,
            y=r_df["subj_y"].values,
            legendgroup="Residualized",
            scalegroup=f"Residualized_{dk}",
            name="Residualized",
            line_color=color,
            fillcolor=color,
            opacity=RESIDUALIZED_ALPHA,
            showlegend=(col_idx == 0),
            points=False,
            width=0.8,
            bandwidth=0.04,
            side="positive",
            orientation="h",
        ), row=1, col=col)

        # Dashed chance line spanning all subjects
        x_axis_name = "x" if col == 1 else f"x{col}"
        fig.add_shape(
            type="line", x0=0.5, x1=0.5, y0=0, y1=1,
            xref=x_axis_name, yref="y domain",
            line=dict(color="black", dash="dash"),
        )

        # Significance stars (canonical True vs Permuted only). Column x-range
        # is widened to [0.25, 1.15] (below) specifically to give both star
        # columns a blank margin past the real AUC<=1 data, so neither one
        # sits on top of a violin — x=1.10 here, residualized at x=0.98.
        for subj in sorted_subjects:
            p_adj_val = adj_pvals.get(dk, {}).get(subj, np.nan)
            stars = sig_stars(p_adj_val)
            if stars:
                sig_count_per_subject[subj] += 1
                subj_y = subj_y_map[subj]
                fig.add_annotation(
                    x=1.10, y=subj_y,
                    text=f"<b>{stars}</b>",
                    showarrow=False,
                    xanchor="right", yanchor="middle",
                    row=1, col=col,
                    font=dict(color="black", size=14),
                )

        # Residualized significance — drawn as its own marker (pale color,
        # its own margin column left of the canonical stars) so the
        # aggregate % in the bar/donut can actually be counted by eye
        # against something on the plot, instead of being a number with no
        # visible per-subject referent. Kept out of the data area (unlike an
        # earlier version at x=0.90, which landed on top of violins peaking
        # near that AUC and was unreadable there).
        for subj in sorted_subjects:
            res_stars = sig_stars(adj_pvals_res.get(dk, {}).get(subj, np.nan))
            if res_stars:
                sig_count_per_subject_res[subj] += 1
                subj_y = subj_y_map[subj]
                fig.add_annotation(
                    x=0.98, y=subj_y,
                    text=f"<b>{res_stars}</b>",
                    showarrow=False,
                    xanchor="right", yanchor="middle",
                    row=1, col=col,
                    font=dict(color=_hex_to_rgba(color, RESIDUALIZED_RING_ALPHA), size=14),
                )

        # Donut — outer ring: canonical True-vs-Permuted, denominator =
        # subjects with data in THIS dim. Inner ring (nested, thinner, lower
        # alpha): residualized significance rate over subjects who actually
        # have residualized data for THIS dim (a different, usually larger,
        # pool than the canonical denominator — using it here keeps the %
        # a literal count, not diluted/inflated by the canonical n). Always
        # drawn as a full ring (colored + gray), same as the outer one, so
        # "0% significant" reads as an empty gray ring rather than vanishing.
        n_with_data = len([s for s in sorted_subjects if s in subjs_per_dim.get(dk, set())])
        n_sig_d = sum(
            1 for s in sorted_subjects
            if s in subjs_per_dim.get(dk, set())
            and not np.isnan(adj_pvals.get(dk, {}).get(s, np.nan))
            and adj_pvals.get(dk, {}).get(s, 1.0) < 0.05
        )
        n_with_res_data = len([s for s in sorted_subjects if s in subjs_per_dim_res.get(dk, set())])
        n_sig_res_d = sum(
            1 for s in sorted_subjects
            if s in subjs_per_dim_res.get(dk, set())
            and not np.isnan(adj_pvals_res.get(dk, {}).get(s, np.nan))
            and adj_pvals_res.get(dk, {}).get(s, 1.0) < 0.05
        )
        pct_d     = n_sig_d / n_with_data * 100 if n_with_data > 0 else 0
        pct_res_d = n_sig_res_d / n_with_res_data * 100 if n_with_res_data > 0 else 0

        fig.add_trace(go.Pie(
            labels=["Signif.", "Not"],
            values=[max(pct_d, 0.001), max(100 - pct_d, 0.001)],
            marker=dict(colors=[color, PERM_COLOR]),
            hole=0.55,
            textinfo="text",
            text=[f"{pct_d:.0f}%", ""],
            textfont=dict(size=14, color="white"),
            textposition="inside",
            showlegend=False,
            sort=False,
        ), row=2, col=col)

        outer_domain = fig.data[-1].domain
        inner_domain = _shrink_domain(
            {"x": list(outer_domain.x), "y": list(outer_domain.y)}, factor=0.55
        )
        fig.add_trace(go.Pie(
            labels=["Res. Signif.", "Not"],
            values=[max(pct_res_d, 0.001), max(100 - pct_res_d, 0.001)],
            marker=dict(colors=[_hex_to_rgba(color, RESIDUALIZED_RING_ALPHA), "whitesmoke"]),
            hole=0.35,
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
            sort=False,
            domain=inner_domain,
        ))

        # The ring alone only gives magnitude at a glance; print the literal
        # % in the donut's empty center so it can be read and checked, not
        # just eyeballed off arc thickness.
        fig.add_annotation(
            text=f"{pct_res_d:.0f}%",
            xref="paper", yref="paper",
            x=sum(inner_domain["x"]) / 2, y=sum(inner_domain["y"]) / 2,
            showarrow=False,
            font=dict(size=9, color=color),
            xanchor="center", yanchor="middle",
        )

    # "Total Sig." column header (custom annotation — avoids collision with legend)
    fig.add_annotation(
        text="<b>Total<br>Sig.</b>",
        xref=f"x{n_dims + 1} domain", yref="paper",
        x=0.5, y=1.01,
        showarrow=False,
        font=dict(size=13, color="#333333"),
        xanchor="center", yanchor="bottom",
    )

    # Total significance bar — a single bar (bar length = canonical True sig.
    # rate, unchanged from plot_individual), with the residualized rate
    # appended as literal text in parentheses rather than drawn as a second
    # overlapping bar: two bars sharing one row are ambiguous to read when
    # one is longer than the other (or when True is 0% and only Residualized
    # has a value) — a number is not. Each rate keeps its own denominator
    # (dims actually run per subject, canonical vs. residualized).
    sig_pcts = [
        sig_count_per_subject[s] / n_dims_per_subject[s] * 100
        if n_dims_per_subject[s] > 0 else 0
        for s in sorted_subjects
    ]
    sig_pcts_res = [
        sig_count_per_subject_res[s] / n_dims_per_subject_res[s] * 100
        if n_dims_per_subject_res[s] > 0 else 0
        for s in sorted_subjects
    ]
    bar_text = [
        (f"{v:.0f}%" if v > 0 else "") + (f" (res {r:.0f}%)" if r > 0 else "")
        for v, r in zip(sig_pcts, sig_pcts_res)
    ]
    fig.add_trace(go.Bar(
        x=sig_pcts,
        y=[i + 0.2 for i in range(n_subjs)],
        orientation="h",
        marker_color=PALETTE["neutral"]["accent"],
        showlegend=False,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=11, color="black"),
    ), row=1, col=n_dims + 1)

    # --- Layout ---
    total_height = max(500, n_subjs * 25 + 200)
    total_width  = 1600

    fig.update_layout(
        title_text=f"{title_prefix} — Individual-level AUC per Participant (+ Residualized)",
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=total_height,
        width=total_width,
        margin=dict(l=80, r=40, t=140, b=50),
        font=dict(family="Times New Roman", size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="Subject ID", font=dict(size=18)),
            automargin=True,
            tickmode="array",
            tickvals=[i + 0.2 for i in range(n_subjs)],
            ticktext=tick_labels,
            tickfont=dict(size=16),
            range=[-0.5, n_subjs],
        ),
    )

    # Shared y-axis ticks (all violin subplots share y)
    fig.update_yaxes(
        tickmode="array",
        tickvals=[i + 0.2 for i in range(n_subjs)],
        ticktext=tick_labels,
        range=[-0.5, n_subjs],
        row=1,
    )

    # x-axis range for violin columns — widened past [0.25, 1.0] (real AUC
    # ceiling) to leave a blank margin for the two star columns (canonical at
    # x=1.10, residualized at x=0.98) so neither sits on top of a violin.
    # Ticks stay at the real 0.5/.75/1.0 scale — the extra room is margin only.
    for j in range(n_dims):
        ax = "xaxis" if j == 0 else f"xaxis{j + 1}"
        fig.layout[ax].range = [0.25, 1.15]
        fig.layout[ax].tickmode = "array"
        fig.layout[ax].tickvals = [0.5, 0.75, 1.0]
        fig.layout[ax].ticktext = ["0.5", ".75", "1.0"]
        fig.layout[ax].showticklabels = True
        fig.layout[ax].tickfont = dict(size=12)

    # x-axis for sig-bar column — direct layout assignment
    bar_ax = f"xaxis{n_dims + 1}"
    fig.layout[bar_ax].title = dict(text="% Signif.", font=dict(size=16))
    fig.layout[bar_ax].tickfont = dict(size=12)
    fig.layout[bar_ax].showgrid = True
    fig.layout[bar_ax].tickmode = "linear"
    fig.layout[bar_ax].dtick = 25
    fig.layout[bar_ax].range = [0, 145]

    return fig, total_width, total_height


# =============================================================================
# Figure C — Combined group-level WS vs LOSO side by side
# =============================================================================


def plot_group_comparison() -> go.Figure:
    """
    Plotly half-violin figure with 2 columns (WS | LOSO), n_dims rows.
    Mirrors reference style (go.Violin side='positive', orientation='h').
    """
    n_dims = len(DIMENSIONS)

    ws_data, loso_data = [], []
    ws_pvals_raw, loso_pvals_raw = [], []
    for dim_info in DIMENSIONS:
        t_ws, p_ws = load_group_data("ws",   dim_info)
        t_lo, p_lo = load_group_data("loso", dim_info)
        ws_data.append((t_ws, p_ws))
        loso_data.append((t_lo, p_lo))
        ws_pvals_raw.append(empirical_pvalue(t_ws, p_ws))
        loso_pvals_raw.append(empirical_pvalue(t_lo, p_lo))

    ws_pvals_adj   = fdr_correct(ws_pvals_raw)
    loso_pvals_adj = fdr_correct(loso_pvals_raw)

    fig = make_subplots(
        rows=n_dims, cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.02,
        horizontal_spacing=0.12,
        column_titles=["Within-Subject", "LOSO"],
    )

    # Axis numbering in make_subplots(rows=n, cols=2):
    # (r, 1) → axis index (r-1)*2 + 1; (r, 2) → (r-1)*2 + 2
    def _ax_idx(row: int, col: int) -> str:
        idx = (row - 1) * 2 + col
        return "" if idx == 1 else str(idx)

    for i, dim_info in enumerate(DIMENSIONS):
        row   = i + 1
        color = dim_info["color"]

        for col, (data_list, p_raws, p_adjs) in enumerate(
            [
                (ws_data,   ws_pvals_raw,   ws_pvals_adj),
                (loso_data, loso_pvals_raw, loso_pvals_adj),
            ],
            start=1,
        ):
            t, p    = data_list[i]
            p_raw   = p_raws[i]
            p_adj   = p_adjs[i]
            sfx     = _ax_idx(row, col)
            x_ref   = f"x{sfx}"
            y_ref   = f"y{sfx}"

            # Permuted
            fig.add_trace(go.Violin(
                x=p, y=[0] * len(p),
                orientation="h", side="positive",
                line_color=PERM_COLOR, fillcolor=PERM_COLOR,
                opacity=0.55, points=False, width=1.0, bandwidth=0.015,
                name="Shuffled", legendgroup="Shuffled",
                showlegend=(i == 0 and col == 1),
            ), row=row, col=col)

            # True
            fig.add_trace(go.Violin(
                x=t, y=[0] * len(t),
                orientation="h", side="positive",
                line_color=color, fillcolor=color,
                opacity=0.72, points=False, width=1.0, bandwidth=0.015,
                name="Full", legendgroup="Full",
                showlegend=(i == 0 and col == 1),
            ), row=row, col=col)

            # Chance line
            fig.add_shape(
                type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                xref=x_ref, yref=f"{y_ref} domain",
                line=dict(color="black", dash="dash", width=1.5),
            )

            # P-value annotation (right of each panel)
            stars   = sig_stars(p_adj)
            line1   = f"{stars} {format_pval(p_raw)}".strip() if stars else format_pval(p_raw)
            line2   = f"(FDR: {format_pval(p_adj)})"
            fig.add_annotation(
                text=f"<b>{line1}</b><br>{line2}",
                xref=f"{x_ref} domain", yref=f"{y_ref} domain",
                x=1.03, y=0.5,
                xanchor="left", yanchor="middle",
                showarrow=False,
                font=dict(color=color, size=9, family="Times New Roman"),
            )

            # Hide y-axis labels
            y_ax = f"yaxis{sfx}"
            fig.layout[y_ax].showticklabels = False
            fig.layout[y_ax].showgrid = False
            fig.layout[y_ax].zeroline = False

            # x-axis ticks (only last row)
            x_ax = f"xaxis{sfx}"
            fig.layout[x_ax].range = [0.35, 1.0]
            fig.layout[x_ax].showticklabels = (row == n_dims)
            if row == n_dims:
                fig.layout[x_ax].tickmode = "array"
                fig.layout[x_ax].tickvals = [0.5, 0.75, 1.0]
                fig.layout[x_ax].ticktext = ["0.5", ".75", "1.0"]
                fig.layout[x_ax].title = dict(text="AUC", font=dict(size=12))

        # Dimension label (left of WS panel)
        sfx_left = _ax_idx(row, 1)
        y_ref_left = f"y{sfx_left}"
        fig.add_annotation(
            text=f"<b>{dim_info['label']}</b>",
            xref="paper", yref=f"{y_ref_left} domain",
            x=-0.01, y=0.5,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=12, family="Times New Roman"),
        )

    fig.update_layout(
        title=dict(
            text="Group-level AUC: Within-Subject vs LOSO",
            font=dict(size=14, family="Times New Roman"),
        ),
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=max(450, n_dims * 90 + 120),
        width=1000,
        margin=dict(l=130, r=60, t=100, b=60),
        font=dict(family="Times New Roman", size=12),
        # Centered (not left-anchored): this figure has WS and LOSO side by
        # side as two columns, so the legend belongs over the midpoint between
        # them, not over the WS column alone.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
    )
    return fig


# =============================================================================
# Figure C2 — Combined group-level WS vs LOSO side by side + residualized
# =============================================================================


def plot_group_comparison_with_residualized() -> go.Figure:
    """
    Same layout as :func:`plot_group_comparison`, with a third half-violin per
    panel: that dimension's True-AUC distribution from its residualized-
    contrast re-run (RESIDUALIZED_DIR), drawn in the dimension color at
    RESIDUALIZED_ALPHA so it reads as a translucent layer rather than a
    competing solid distribution — same convention as
    plot_group_level_with_residualized.

    Rows are GROUP_ROW_DIMENSIONS (the 5 canonical dimensions). No figure title, no
    column_titles ("Within-Subject"/"LOSO" collided with the legend) —
    pipeline identity is carried by each column's bottom x-axis title
    ("WS AUC" / "LOSO AUC") instead. The True/Residualized mean-AUC +
    significance readout (_true_res_annotation_text, shared with
    plot_group_level_with_residualized so both figures read identically)
    sits INSIDE each panel's own top-right corner rather than a separate
    margin column.
    """
    row_dims = GROUP_ROW_DIMENSIONS
    n_dims   = len(row_dims)

    ws_data, loso_data = [], []
    ws_pvals_raw, loso_pvals_raw = [], []
    ws_res_data, loso_res_data = [], []
    ws_res_pvals_raw, loso_res_pvals_raw = [], []
    for dim_info in row_dims:
        t_ws, p_ws = load_group_data("ws",   dim_info)
        t_lo, p_lo = load_group_data("loso", dim_info)
        ws_data.append((t_ws, p_ws))
        loso_data.append((t_lo, p_lo))
        ws_pvals_raw.append(empirical_pvalue(t_ws, p_ws))
        loso_pvals_raw.append(empirical_pvalue(t_lo, p_lo))

        res_dim_info = _residualized_dim_info(dim_info)
        t_ws_res, p_ws_res = load_group_data("ws",   res_dim_info)
        t_lo_res, p_lo_res = load_group_data("loso", res_dim_info)
        ws_res_data.append((t_ws_res, p_ws_res))
        loso_res_data.append((t_lo_res, p_lo_res))
        ws_res_pvals_raw.append(empirical_pvalue(t_ws_res, p_ws_res))
        loso_res_pvals_raw.append(empirical_pvalue(t_lo_res, p_lo_res))

    ws_pvals_adj       = fdr_correct(ws_pvals_raw)
    loso_pvals_adj     = fdr_correct(loso_pvals_raw)
    ws_res_pvals_adj   = fdr_correct(ws_res_pvals_raw)
    loso_res_pvals_adj = fdr_correct(loso_res_pvals_raw)

    fig = make_subplots(
        rows=n_dims, cols=2,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.02,
        horizontal_spacing=0.05,
    )

    # Axis numbering in make_subplots(rows=n, cols=2):
    # (r, 1) → axis index (r-1)*2 + 1; (r, 2) → (r-1)*2 + 2
    def _ax_idx(row: int, col: int) -> str:
        idx = (row - 1) * 2 + col
        return "" if idx == 1 else str(idx)

    for i, dim_info in enumerate(row_dims):
        row   = i + 1
        color = dim_info["color"]

        for col, (pipeline_label, data_list, p_adjs, res_data_list, res_p_adjs) in enumerate(
            [
                ("WS",   ws_data,   ws_pvals_adj,   ws_res_data,   ws_res_pvals_adj),
                ("LOSO", loso_data, loso_pvals_adj, loso_res_data, loso_res_pvals_adj),
            ],
            start=1,
        ):
            t, p     = data_list[i]
            p_adj    = p_adjs[i]
            t_res, p_res = res_data_list[i]
            res_p_adj    = res_p_adjs[i]
            sfx     = _ax_idx(row, col)
            x_ref   = f"x{sfx}"
            y_ref   = f"y{sfx}"

            # Permuted
            fig.add_trace(go.Violin(
                x=p, y=[0] * len(p),
                orientation="h", side="positive",
                line_color=PERM_COLOR, fillcolor=PERM_COLOR,
                opacity=0.55, points=False, width=1.0, bandwidth=0.015,
                name="Shuffled", legendgroup="Shuffled",
                showlegend=(i == 0 and col == 1),
            ), row=row, col=col)

            # True
            fig.add_trace(go.Violin(
                x=t, y=[0] * len(t),
                orientation="h", side="positive",
                line_color=color, fillcolor=color,
                opacity=0.72, points=False, width=1.0, bandwidth=0.015,
                name="Full", legendgroup="Full",
                showlegend=(i == 0 and col == 1),
            ), row=row, col=col)

            # Residualized (True) — same color, low alpha, drawn last (on top)
            fig.add_trace(go.Violin(
                x=t_res, y=[0] * len(t_res),
                orientation="h", side="positive",
                line_color=color, fillcolor=color,
                opacity=RESIDUALIZED_ALPHA, points=False, width=1.0, bandwidth=0.015,
                name="Residualized", legendgroup="Residualized",
                showlegend=(i == 0 and col == 1),
            ), row=row, col=col)

            # Chance line
            fig.add_shape(
                type="line", x0=0.5, x1=0.5, y0=0, y1=1,
                xref=x_ref, yref=f"{y_ref} domain",
                line=dict(color="black", dash="dash", width=1.5),
            )

            # True/Residualized mean AUC + significance — inside the panel
            # (top-right corner), not a separate margin column.
            fig.add_annotation(
                text=_true_res_annotation_text(t, p_adj, t_res, res_p_adj),
                xref=f"{x_ref} domain", yref=f"{y_ref} domain",
                x=0.97, y=0.68,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(color=color, size=7.5, family="Times New Roman"),
            )

            # Hide y-axis labels
            y_ax = f"yaxis{sfx}"
            fig.layout[y_ax].showticklabels = False
            fig.layout[y_ax].showgrid = False
            fig.layout[y_ax].zeroline = False

            # x-axis ticks (only last row). Title carries pipeline identity
            # ("WS AUC" / "LOSO AUC") since there's no column_titles anymore.
            x_ax = f"xaxis{sfx}"
            # WS AUCs run higher than LOSO's (On/Off-Task's Full violin peaks
            # at .72, close enough to the annotation at domain x=0.97 that its
            # right tail touched the text). Extending WS's range past 1.0
            # (ticks stay at .5/.75/1.0 either way) pushes that same domain
            # fraction further right in data terms, opening a gap without
            # moving the annotation itself. LOSO's violins run low enough
            # not to need it.
            fig.layout[x_ax].range = [0.35, 1.12] if col == 1 else [0.35, 1.0]
            fig.layout[x_ax].showticklabels = (row == n_dims)
            if row == n_dims:
                fig.layout[x_ax].tickmode = "array"
                fig.layout[x_ax].tickvals = [0.5, 0.75, 1.0]
                fig.layout[x_ax].ticktext = ["0.5", ".75", "1.0"]
                fig.layout[x_ax].title = dict(
                    text=f"{pipeline_label} AUC", font=dict(size=12)
                )

        # Dimension label (left of WS panel)
        sfx_left = _ax_idx(row, 1)
        y_ref_left = f"y{sfx_left}"
        fig.add_annotation(
            text=f"<b>{dim_info['label']}</b>",
            xref="paper", yref=f"{y_ref_left} domain",
            x=-0.01, y=0.5,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=12, family="Times New Roman"),
        )

    fig.update_layout(
        violinmode="overlay",
        violingap=0,
        template="plotly_white",
        height=max(400, n_dims * 68 + 55),
        width=760,
        margin=dict(l=150, r=20, t=55, b=50),
        font=dict(family="Times New Roman", size=12),
        # yanchor="top" at y=1.0 puts the legend's top edge AT the plot area's
        # own top edge, growing downward INTO row 1 rather than up into the
        # margin reserved for it -- that's what was cutting into the first
        # panel. yanchor="bottom" grows it upward into margin.t instead.
        # Centered (not left-anchored) so it sits over the midpoint between
        # the WS and LOSO columns, not just over WS.
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
    )
    return fig


# =============================================================================
# Figure C3 — Spatial decoding comparison: WS vs LOSO topomaps
# =============================================================================


def load_spatial_metrics(
    pipeline: str, dim_info: dict,
    family: str = SPATIAL_FAMILY, model: str = SPATIAL_MODEL,
) -> pd.DataFrame | None:
    """
    Load one dimension/pipeline's merged spatial ``per_channel_metrics.csv``.

    Returns None (not an error) if the file doesn't exist yet (see
    ``plot_spatial_comparison_panel``).
    """
    sub     = "WithinSubject" if pipeline == "ws" else "LOSO"
    dir_key = "ws_dir" if pipeline == "ws" else "loso_dir"
    metrics_csv = (
        SPATIAL_RESULTS_ROOT / sub / dim_info[dir_key] / family / model
        / "per_channel_metrics.csv"
    )
    if not metrics_csv.exists():
        return None
    return _safe_read_csv(metrics_csv)


def plot_spatial_comparison_panel() -> Path | None:
    """
    Build the WS-vs-LOSO spatial (per-electrode) decoding comparison panel:
    2 rows (Within-Subject, LOSO) x one column per dimension, shared AUC
    colour scale, FWER-significant electrodes marked (see
    ``utils.spatial_decoding_utils.plot_pipeline_comparison_topomap_panel``).

    Columns follow GROUP_ROW_DIMENSIONS (the 5 canonical dimensions — same
    order as the "+ residualized" group-level figures), so a reader moving
    between figures does not have to re-locate a dimension.

    A Matplotlib figure, not a Plotly one, so it's saved directly (PNG + SVG)
    rather than queued through the kaleido batch writer used by every other
    figure in this script. Any dimension missing a merged
    ``per_channel_metrics.csv`` on either pipeline is skipped with a message.

    Returns the PNG path, or None if fewer than one dimension has data on
    both pipelines.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.spatial_decoding_utils import plot_pipeline_comparison_topomap_panel

    ws_metrics, loso_metrics = {}, {}
    for dim_info in GROUP_ROW_DIMENSIONS:
        ws_df   = load_spatial_metrics("ws", dim_info)
        loso_df = load_spatial_metrics("loso", dim_info)
        if ws_df is None or loso_df is None:
            missing = "WS" if ws_df is None else "LOSO"
            print(f"  ! {dim_info['label']}: missing spatial per_channel_metrics.csv "
                  f"for {missing}; skipping from comparison panel.")
            continue
        ws_metrics[dim_info["label"]]   = ws_df
        loso_metrics[dim_info["label"]] = loso_df

    if not ws_metrics:
        print("  No dimension has spatial data on both pipelines; skipping spatial comparison panel.")
        return None

    dim_colors = {d["label"]: d["color"] for d in GROUP_ROW_DIMENSIONS}
    out_png = OUTPUT_DIR / "spatial_comparison_ws_loso.png"
    out_svg = OUTPUT_DIR / "spatial_comparison_ws_loso.svg"
    for out_path in (out_png, out_svg):
        plot_pipeline_comparison_topomap_panel(
            ws_metrics, loso_metrics, value_col="mean_auc", out_path=str(out_path),
            mask_col="sig", dim_colors=dim_colors,
        )
    return out_png


# =============================================================================
# Figure C4 — Global Decoding + Spatial Decoding, combined (WS | LOSO)
# =============================================================================


def plot_group_spatial_combined() -> Path | None:
    """
    Composite "Global Decoding + Spatial Decoding" figure: for each pipeline
    (Within-Subject, LOSO/Between-Subject), a column of per-dimension AUC
    density curves (Shuffled / Full / Residualized — same three layers as
    :func:`plot_group_level_with_residualized`) sits next to a column of
    that dimension's spatial-decoding topomap, GROUP_ROW_DIMENSIONS rows.

    Pure Matplotlib, not Plotly: MNE can only draw a topomap onto a
    Matplotlib Axes, and CLAUDE.md's "Figure Assembly" rule is to compose
    natively-rendered halves rather than fake one plotting library inside
    the other — so instead of rasterizing a topomap into a Plotly figure,
    the *entire* figure (distributions included) is built in Matplotlib.
    This reuses the ``_kde_fill``/``_clean_kde_ax`` helpers (unused until
    now — they render a vertical filled density curve, the Matplotlib
    equivalent of the Plotly half-violins used everywhere else in this
    script) alongside ``utils.spatial_decoding_utils._draw_cbpt_topomap``
    for the spatial half, so both halves can share one Figure/GridSpec.

    Saved directly as PNG + SVG (bypasses the kaleido batch queue — no
    Plotly figure is involved). Returns the PNG path, or None if neither
    pipeline has spatial data for any dimension.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.spatial_decoding_utils import build_info_from_channels, _draw_cbpt_topomap

    row_dims  = GROUP_ROW_DIMENSIONS
    n_dims    = len(row_dims)
    pipelines = [("ws", "Within-Subject Classification"), ("loso", "Between-Subject Classification")]

    # ---- Load distribution (True/Perm/Residualized) + spatial data ----
    dist_rows, spatial_by_pipeline = {}, {}
    for pkey, _ in pipelines:
        raw_pvals, res_raw_pvals, rows = [], [], []
        spatial = {}
        for dim_info in row_dims:
            t, p         = load_group_data(pkey, dim_info)
            t_res, p_res = load_group_data(pkey, _residualized_dim_info(dim_info))
            raw_pvals.append(empirical_pvalue(t, p))
            res_raw_pvals.append(empirical_pvalue(t_res, p_res))
            rows.append((t, p, t_res, p_res))
            spatial[dim_info["label"]] = load_spatial_metrics(pkey, dim_info)
        adj     = fdr_correct(raw_pvals)
        res_adj = fdr_correct(res_raw_pvals)
        dist_rows[pkey] = [(*row, a, ra) for row, a, ra in zip(rows, adj, res_adj)]
        spatial_by_pipeline[pkey] = spatial

    all_frames = [df for sp in spatial_by_pipeline.values() for df in sp.values() if df is not None]
    if not all_frames:
        print("  No spatial data on either pipeline; skipping distribution+spatial combined panel.")
        return None
    vmin, vmax = 0.5, max(float(df["mean_auc"].max()) for df in all_frames)

    # ---- Figure / GridSpec ----
    # Columns: row-label | WS-dist | WS-topo | WS-cbar | spacer | LOSO-dist | LOSO-topo | LOSO-cbar
    #
    # width_ratios/height_ratios are set in literal inches (figsize is their
    # sum, not an independent value scaled up from it) so the topo column's
    # width exactly equals the data-row height, i.e. every topomap axes is
    # square. mne.viz.plot_topomap enforces equal aspect internally, and the
    # circular clip path in _draw_cbpt_topomap is drawn in ax.transAxes
    # (0-1 assuming a square axes) — on a non-square axes the two disagree
    # and the head comes out visibly stretched/oval. wspace/hspace stay
    # small and identical in both directions so the shrink they introduce
    # doesn't reintroduce a width/height mismatch.
    ROW_H  = 1.65   # data-row height == topo column width (keeps topomaps square)
    TITLE_FRAC = 0.20   # fraction of ROW_H reserved above the dist axes for its stats text
    label_w, dist_w, topo_w, cbar_w, spacer_w = 0.42, 2.5, ROW_H, 0.13, 0.22
    col_w = [label_w, dist_w, topo_w, cbar_w, spacer_w, dist_w, topo_w, cbar_w]
    block_cols = {"ws": dict(dist=1, topo=2, cbar=3), "loso": dict(dist=5, topo=6, cbar=7)}
    header_h = 1.5

    fig = plt.figure(figsize=(sum(col_w), header_h + ROW_H * n_dims))
    gs = fig.add_gridspec(
        nrows=n_dims + 1, ncols=len(col_w),
        width_ratios=col_w, height_ratios=[header_h] + [ROW_H] * n_dims,
        wspace=0.06, hspace=0.06,
    )

    def _row_y(r: int) -> tuple[float, float]:
        p = gs[r, 1].get_position(fig)
        return p.y0, p.y1

    def _col_x(c0: int, c1: int) -> tuple[float, float]:
        p0 = gs[0, c0].get_position(fig)
        p1 = gs[0, c1].get_position(fig)
        return p0.x0, p1.x1

    # ---- Header row: block title banner + "Global/Spatial Decoding" sub-headers ----
    # Offsets are a fraction of the header row's OWN height (not a fixed
    # figure-fraction) so banner/sub-header spacing stays correct regardless
    # of how header_h is tuned — a fixed-fraction offset sized for one
    # header_h silently collapses the two into each other at a smaller one.
    header_y0, header_y1 = _row_y(0)
    header_span = header_y1 - header_y0
    for pkey, block_title in pipelines:
        cols = block_cols[pkey]
        bx0, bx1 = _col_x(cols["dist"], cols["cbar"])
        fig.text((bx0 + bx1) / 2, header_y1 - 0.10 * header_span, block_title,
                 ha="center", va="top", fontsize=13, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#333333", edgecolor="none"))

        dx0, dx1 = _col_x(cols["dist"], cols["dist"])
        fig.text((dx0 + dx1) / 2, header_y0 + 0.10 * header_span, "Global Decoding",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

        tx0, tx1 = _col_x(cols["topo"], cols["cbar"])
        fig.text((tx0 + tx1) / 2, header_y0 + 0.10 * header_span, "Spatial Decoding",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ---- Per-dimension rows ----
    last_im_by_pipeline: dict[str, object] = {}
    for r, dim_info in enumerate(row_dims):
        row   = r + 1
        color = dim_info["color"]

        # Rotated text has no auto-fit: a fixed size overflows into
        # neighbouring rows for the longer compound labels ("Neutral/
        # Emotional", "Present/NotPresent" — up to 19 characters vs. 4-11 for
        # the rest), so size down once a label crosses that length.
        label_fontsize = 11 if len(dim_info["label"]) <= 12 else 8
        lp = gs[row, 0].get_position(fig)
        fig.text((lp.x0 + lp.x1) / 2, (lp.y0 + lp.y1) / 2, dim_info["label"],
                 ha="center", va="center", rotation=90, fontsize=label_fontsize,
                 fontweight="bold", color=color)

        for pkey, _ in pipelines:
            cols = block_cols[pkey]
            t, p, t_res, p_res, p_adj, res_p_adj = dist_rows[pkey][r]

            # --- Global Decoding: vertical filled density curves ---
            # The stats text used to sit INSIDE this axes (top-right corner)
            # and visually collided with the curves it was labelling. It now
            # lives in a dedicated strip above the axes instead: the dist
            # axes is built manually at TITLE_FRAC-shrunk height (from the
            # top only, so the bottom — where the last row's x-axis/"AUC"
            # label live — doesn't move), and the text is placed in the
            # freed strip via fig.text. This costs the dist column some
            # height, not the topo column, so it doesn't shrink the topomaps.
            cell = gs[row, cols["dist"]].get_position(fig)
            title_h = TITLE_FRAC * (cell.y1 - cell.y0)
            ax_d = fig.add_axes([cell.x0, cell.y0, cell.x1 - cell.x0,
                                 (cell.y1 - cell.y0) - title_h])
            _kde_fill(ax_d, p,     PERM_COLOR, 0.55, bw=BW_GROUP)
            _kde_fill(ax_d, t,     color,      0.72, bw=BW_GROUP)
            _kde_fill(ax_d, t_res, color,      RESIDUALIZED_ALPHA, bw=BW_GROUP)
            _clean_kde_ax(ax_d, show_xticks=(row == n_dims), show_xlabel=(row == n_dims),
                         fixed_ylim=False)
            line1, line2 = _true_res_annotation_lines(t, p_adj, t_res, res_p_adj)
            xc = (cell.x0 + cell.x1) / 2
            fig.text(xc, cell.y1 - 0.10 * title_h, line1, ha="center", va="top",
                      fontsize=7.5, fontweight="bold", color=color)
            fig.text(xc, cell.y1 - 0.55 * title_h, line2, ha="center", va="top",
                      fontsize=7.5, fontstyle="italic", color=color)

            # --- Spatial Decoding: topomap ---
            ax_s = fig.add_subplot(gs[row, cols["topo"]])
            df = spatial_by_pipeline[pkey].get(dim_info["label"])
            if df is not None:
                info = build_info_from_channels(df["channel"].tolist())
                mask = df["sig"].to_numpy(dtype=bool) if "sig" in df.columns else None
                last_im_by_pipeline[pkey] = _draw_cbpt_topomap(
                    ax_s, df["mean_auc"].to_numpy(dtype=float), info, mask,
                    vmin, vmax, "viridis", markersize=4,
                )
            else:
                ax_s.axis("off")
                ax_s.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="#999999")

    # ---- One shared colorbar per pipeline block, spanning all data rows ----
    for pkey, _ in pipelines:
        if pkey not in last_im_by_pipeline:
            continue
        cax = fig.add_subplot(gs[1:, block_cols[pkey]["cbar"]])
        cbar = fig.colorbar(last_im_by_pipeline[pkey], cax=cax)
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
        cbar.set_label("mean_auc", fontsize=9)

    out_png = OUTPUT_DIR / "group_comparison_spatial_combined.png"
    out_svg = OUTPUT_DIR / "group_comparison_spatial_combined.svg"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


# =============================================================================
# Figure D — LOSO vs WS scatter (one panel per dimension)
# =============================================================================


def plot_loso_vs_ws_scatter() -> go.Figure:
    """
    One panel per dimension: scatter of per-subject mean AUC (LOSO vs WS).
    Only subjects with data in both pipelines for a given dimension are shown.
    Diagonal (y=x) and chance lines (0.5) drawn for reference.
    """
    n_dims = len(DIMENSIONS)

    fig = make_subplots(
        rows=1, cols=n_dims,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        subplot_titles=[d["label"] for d in DIMENSIONS],
    )

    # Color subplot titles
    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color = dim_info["color"]
        fig.layout.annotations[i].font.size  = 14
        fig.layout.annotations[i].font.family = "Times New Roman"

    auc_min, auc_max = 0.35, 1.0

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col   = col_idx + 1
        color = dim_info["color"]

        # Load mean AUC per subject for both pipelines
        t_ws, _   = load_subject_data("ws",   dim_info)
        t_lo, _   = load_subject_data("loso", dim_info)

        ws_means   = t_ws.groupby("subject")["auc"].mean().rename("ws")
        loso_means = t_lo.groupby("subject")["auc"].mean().rename("loso")

        merged = pd.concat([ws_means, loso_means], axis=1).dropna()
        if merged.empty:
            continue

        merged.index = merged.index.astype(str)
        subj_labels  = [str(int(s)) if s.isdigit() else s for s in merged.index]

        # Scatter points
        fig.add_trace(go.Scatter(
            x=merged["ws"].values,
            y=merged["loso"].values,
            mode="markers+text",
            text=subj_labels,
            textposition="top center",
            textfont=dict(size=9, family="Times New Roman"),
            marker=dict(color=color, size=7, opacity=0.85,
                        line=dict(color="white", width=0.5)),
            showlegend=False,
            name=dim_info["label"],
        ), row=1, col=col)

        # Diagonal (y = x)
        fig.add_shape(
            type="line", x0=auc_min, x1=auc_max, y0=auc_min, y1=auc_max,
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col}" if col > 1 else "y",
            line=dict(color="#888888", dash="dot", width=1.2),
        )

        # Chance lines
        for axis_pair in [("x", "y"), ("y", "x")]:
            fig.add_shape(
                type="line",
                x0=0.5 if axis_pair[0] == "x" else auc_min,
                x1=0.5 if axis_pair[0] == "x" else auc_max,
                y0=0.5 if axis_pair[1] == "y" else auc_min,
                y1=0.5 if axis_pair[1] == "y" else auc_max,
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                line=dict(color="#BBBBBB", dash="dash", width=1.0),
            )

        # Regression line + annotation
        if len(merged) >= 3:
            m, b = np.polyfit(merged["ws"].values, merged["loso"].values, 1)
            x_reg = np.array([auc_min, auc_max])
            y_reg = m * x_reg + b
            fig.add_trace(go.Scatter(
                x=x_reg, y=y_reg,
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                showlegend=False,
            ), row=1, col=col)

            r = float(np.corrcoef(merged["ws"].values, merged["loso"].values)[0, 1])
            n = len(merged)
            fig.add_annotation(
                text=f"r = {r:.2f}, n = {n}",
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.05, y=0.96,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=10, color=color, family="Times New Roman"),
            )

        # Axis formatting
        x_ax = f"xaxis{col}" if col > 1 else "xaxis"
        y_ax = f"yaxis{col}" if col > 1 else "yaxis"
        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range = [auc_min, auc_max]
            fig.layout[ax_name].tickmode = "array"
            fig.layout[ax_name].tickvals = [0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext = ["0.5", ".75", "1.0"]
            fig.layout[ax_name].tickfont = dict(size=10)
            fig.layout[ax_name].showgrid = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline = False

        fig.layout[x_ax].title = dict(text="WS AUC", font=dict(size=12))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO AUC", font=dict(size=12))

    fig.update_layout(
        title=dict(
            text="LOSO vs Within-Subject AUC per Participant",
            font=dict(size=14, family="Times New Roman"),
        ),
        template="plotly_white",
        height=380,
        width=320 * n_dims,
        margin=dict(l=60, r=30, t=100, b=60),
        font=dict(family="Times New Roman", size=12),
    )
    return fig


# =============================================================================
# Figure E — Regression lines overlay (all dimensions on one panel)
# =============================================================================


def plot_regression_overlay() -> go.Figure:
    """
    Single panel: regression lines LOSO~WS for each dimension overlaid.
    Points omitted; only regression lines + CI bands shown per dimension.
    """
    auc_min, auc_max = 0.35, 1.0
    x_reg = np.linspace(auc_min, auc_max, 200)

    fig = go.Figure()

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[auc_min, auc_max], y=[auc_min, auc_max],
        mode="lines",
        line=dict(color="#AAAAAA", dash="dot", width=1.5),
        showlegend=True, name="y = x",
    ))

    # Chance lines
    for v in [0.5]:
        fig.add_vline(x=v, line_dash="dash", line_color="#CCCCCC", line_width=1.0)
        fig.add_hline(y=v, line_dash="dash", line_color="#CCCCCC", line_width=1.0)

    for dim_info in DIMENSIONS:
        color = dim_info["color"]

        t_ws, _ = load_subject_data("ws",   dim_info)
        t_lo, _ = load_subject_data("loso", dim_info)

        ws_means   = t_ws.groupby("subject")["auc"].mean().rename("ws")
        loso_means = t_lo.groupby("subject")["auc"].mean().rename("loso")
        merged = pd.concat([ws_means, loso_means], axis=1).dropna()

        if len(merged) < 3:
            continue

        x = merged["ws"].values
        y = merged["loso"].values
        n = len(merged)
        m, b = np.polyfit(x, y, 1)
        r    = float(np.corrcoef(x, y)[0, 1])

        # Bootstrap 95% CI band on the regression line
        rng = np.random.default_rng(42)
        y_boots = np.zeros((500, len(x_reg)))
        for k in range(500):
            idx = rng.integers(0, n, size=n)
            mk, bk = np.polyfit(x[idx], y[idx], 1)
            y_boots[k] = mk * x_reg + bk
        ci_lo = np.percentile(y_boots, 2.5, axis=0)
        ci_hi = np.percentile(y_boots, 97.5, axis=0)

        # CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_reg, x_reg[::-1]]),
            y=np.concatenate([ci_hi, ci_lo[::-1]]),
            fill="toself",
            fillcolor=color,
            opacity=0.15,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Regression line
        fig.add_trace(go.Scatter(
            x=x_reg, y=m * x_reg + b,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{dim_info['label']} (r={r:.2f}, n={n})",
        ))

    fig.update_layout(
        title=dict(
            text="LOSO vs WS AUC — Regression Lines by Dimension",
            font=dict(size=14, family="Times New Roman"),
        ),
        template="plotly_white",
        height=480, width=520,
        margin=dict(l=70, r=30, t=80, b=70),
        font=dict(family="Times New Roman", size=12),
        xaxis=dict(
            title=dict(text="WS AUC", font=dict(size=13)),
            range=[auc_min, auc_max],
            tickmode="array", tickvals=[0.5, 0.75, 1.0],
            ticktext=["0.5", ".75", "1.0"],
            showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="LOSO AUC", font=dict(size=13)),
            range=[auc_min, auc_max],
            tickmode="array", tickvals=[0.5, 0.75, 1.0],
            ticktext=["0.5", ".75", "1.0"],
            showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        ),
        legend=dict(
            font=dict(size=11), bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#DDDDDD", borderwidth=1,
        ),
    )
    return fig


# =============================================================================
# WS-vs-LOSO feature scatter helpers (shared by Gini / SHAP figures)
# =============================================================================


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """
    Scale an array to [0, 1] via min-max normalisation.

    A constant array (max == min) maps to all zeros rather than dividing by
    zero. Used so WS and LOSO importance/SHAP values — which live on very
    different absolute scales because of differing sample sizes and
    aggregation depth — can be plotted on a shared 0-1 axis.

    NOTE: Only appropriate for non-negative magnitudes (Gini, |SHAP|). For
    signed values use _symmetric_scale instead.
    """
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def _symmetric_scale(values: np.ndarray) -> np.ndarray:
    """
    Scale signed values to [-1, 1] by dividing by max(|values|).

    Preserves 0 at 0, so a feature with zero net SHAP contribution maps to 0
    on both axes — unlike min-max, which would map it to an arbitrary
    non-zero position depending on the distribution asymmetry.
    """
    v_max = np.abs(values).max()
    if v_max == 0:
        return np.zeros_like(values)
    return values / v_max


def _select_top_union(rank_x: np.ndarray, rank_y: np.ndarray, top_n: int) -> list[int]:
    """
    Indices of the union of the top-N entries of ``rank_x`` and of ``rank_y``.

    Ensures a feature important in *either* pipeline is shown, instead of only
    those that rank highly in a single one (e.g. WS).
    """
    top_x = set(np.argsort(rank_x)[-top_n:])
    top_y = set(np.argsort(rank_y)[-top_n:])
    return sorted(top_x | top_y)


def _common_feature_indices(
    x_feats: list[str], y_feats: list[str]
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Return (x_index_map, y_index_map, common_features) for two feature-name lists."""
    x_idx = {f: i for i, f in enumerate(x_feats)}
    y_idx = {f: i for i, f in enumerate(y_feats)}
    common = [f for f in x_feats if f in y_idx]
    return x_idx, y_idx, common


def _new_dimension_grid_fig() -> go.Figure:
    """Create a 1xN subplot grid (one panel per dimension) with colored titles."""
    n_dims = len(DIMENSIONS)
    fig = make_subplots(
        rows=1, cols=n_dims,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.06,
        subplot_titles=[d["label"] for d in DIMENSIONS],
    )
    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color  = dim_info["color"]
        fig.layout.annotations[i].font.size   = 14
        fig.layout.annotations[i].font.family = "Times New Roman"
    return fig


def _add_scatter_panel(
    fig: go.Figure, col: int, dim_info: dict,
    x_common: np.ndarray, y_common: np.ndarray, common_feats: list[str],
    highlight_idx: list[int],
    x_label: str, y_label: str,
    axis_range: list[float] | None = None,
) -> None:
    """
    Add a WS-vs-LOSO scatter panel to ``fig`` at column ``col``.

    Every common feature is drawn, so the cloud the reader sees is the same
    population the reported statistics describe.  ``highlight_idx`` entries are
    drawn opaque and labelled; the rest form a faint background.

    The correlation and the regression line are computed over **all** common
    features, never over ``highlight_idx`` alone.  Restricting them to a
    top-N-of-x ∪ top-N-of-y subset is a selection-on-extremes artifact: when the
    true relationship is near zero, every "top-x only" point sits at high-x/low-y
    and every "top-y only" point at low-x/high-y, manufacturing a steep negative
    slope out of noise.  Measured here on ``onoff``: r = 0.04 over all 177 common
    features vs r = -0.86 over the 20-point union, the latter sitting at the 25th
    percentile of a permutation null that breaks the WS↔LOSO pairing entirely.

    Parameters
    ----------
    fig : go.Figure
        Figure to draw into.
    col : int
        1-based subplot column.
    dim_info : dict
        Dimension metadata; ``color`` and ``label`` are used.
    x_common, y_common : np.ndarray
        Importance values for every common feature, index-aligned to
        ``common_feats``.
    common_feats : list[str]
        Feature names for the common set.
    highlight_idx : list[int]
        Indices into ``common_feats`` to draw opaque and label.
    x_label, y_label : str
        Axis titles.
    axis_range : list[float], optional
        [lo, hi] for both axes.  Defaults to [-0.05, 1.05] (min-max figures).
        Pass [-1.05, 1.05] for signed/directional figures.
    """
    if axis_range is None:
        axis_range = [-0.05, 1.05]
    color = dim_info["color"]

    # A constant axis makes the correlation and the regression undefined.
    # np.polyfit answers with an opaque "SVD did not converge" several frames
    # down, so the condition is named here instead: a panel drawn from a
    # degenerate input would otherwise look like a legitimate flat relationship.
    for values, side in ((x_common, "WS"), (y_common, "LOSO")):
        if np.ptp(values) == 0:
            raise ValueError(
                f"{dim_info['label']}: every {side} value is identical "
                f"({values[0]:g}), so no correlation is defined over the "
                f"{len(values)} common features. Fix the upstream export rather "
                f"than plotting this."
            )

    highlight = np.zeros(len(common_feats), dtype=bool)
    highlight[highlight_idx] = True

    # Background: every common feature not singled out for labelling.
    fig.add_trace(go.Scatter(
        x=x_common[~highlight], y=y_common[~highlight],
        mode="markers",
        marker=dict(color=color, size=4, opacity=0.18,
                    line=dict(color="white", width=0.3)),
        customdata=[f for f, h in zip(common_feats, highlight) if not h],
        hovertemplate="%{customdata}<br>WS: %{x:.3f}<br>LOSO: %{y:.3f}<extra></extra>",
        showlegend=False,
        name=f"{dim_info['label']} (other)",
    ), row=1, col=col)

    highlight_feats = [f for f, h in zip(common_feats, highlight) if h]
    fig.add_trace(go.Scatter(
        x=x_common[highlight], y=y_common[highlight],
        mode="markers+text",
        marker=dict(color=color, size=7, opacity=0.9,
                    line=dict(color="white", width=0.5)),
        text=[f.split("_")[0] for f in highlight_feats],
        textposition="top center",
        textfont=dict(size=7, family="Times New Roman"),
        customdata=highlight_feats,
        hovertemplate="%{customdata}<br>WS: %{x:.3f}<br>LOSO: %{y:.3f}<extra></extra>",
        showlegend=False,
        name=dim_info["label"],
    ), row=1, col=col)

    x_ref = f"x{col}" if col > 1 else "x"
    y_ref = f"y{col}" if col > 1 else "y"
    diag_lo, diag_hi = axis_range[0], axis_range[1]

    # Diagonal y=x
    fig.add_shape(
        type="line", x0=diag_lo, x1=diag_hi, y0=diag_lo, y1=diag_hi,
        xref=x_ref, yref=y_ref,
        line=dict(color="#AAAAAA", dash="dot", width=1.2),
    )

    # Regression line — fitted on the full common set.
    m, b = np.polyfit(x_common, y_common, 1)
    x_reg = np.array([diag_lo, diag_hi])
    fig.add_trace(go.Scatter(
        x=x_reg, y=m * x_reg + b,
        mode="lines",
        line=dict(color=color, width=2.0),
        showlegend=False,
    ), row=1, col=col)

    # Correlation — over the full common set.
    r = float(np.corrcoef(x_common, y_common)[0, 1])
    fig.add_annotation(
        text=f"r = {r:.2f}, n={len(x_common)}",
        xref=f"{x_ref} domain" if col > 1 else "x domain",
        yref=f"{y_ref} domain" if col > 1 else "y domain",
        x=0.05, y=0.97,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=10, color=color, family="Times New Roman"),
    )

    x_ax = f"xaxis{col}" if col > 1 else "xaxis"
    y_ax = f"yaxis{col}" if col > 1 else "yaxis"
    fig.layout[x_ax].range = axis_range
    fig.layout[y_ax].range = axis_range
    show_zeroline = diag_lo < 0  # signed [-1,1] range → show zero reference
    for ax_name in (x_ax, y_ax):
        fig.layout[ax_name].showgrid   = True
        fig.layout[ax_name].gridcolor  = "#EEEEEE"
        fig.layout[ax_name].zeroline   = show_zeroline
        fig.layout[ax_name].zerolinecolor = "#AAAAAA"
        fig.layout[ax_name].zerolinewidth = 1.0
        fig.layout[ax_name].tickfont   = dict(size=9)
    fig.layout[x_ax].title = dict(text=x_label, font=dict(size=11))
    if col == 1:
        fig.layout[y_ax].title = dict(text=y_label, font=dict(size=11))


# =============================================================================
# Feature-importance / SHAP loading helpers
# =============================================================================


def _load_mean_gini_importance(
    pipeline: str, dim_info: dict
) -> tuple[np.ndarray, list[str]]:
    """
    Return (gini_importance, feature_names) averaged across all true runs.

    WS:   mean over true_runs/run_N/rf_ws_feature_importances.csv
    LOSO: rf_loso_100runs_feature_importances_data.csv (already aggregated)
    """
    if pipeline == "ws":
        base     = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        run_dfs  = []
        for fi_csv in sorted(base.glob("true_runs/run_*/rf_ws_feature_importances.csv")):
            df = pd.read_csv(fi_csv).rename(columns={"feature": "feature_name",
                                                      "importance": "mean_importance"})
            run_dfs.append(df.set_index("feature_name")["mean_importance"])
        if not run_dfs:
            print(f"  ! No rf_ws_feature_importances.csv found under {base}/true_runs/ "
                  f"— panel will be empty, not just sparse. Check the glob pattern still "
                  f"matches the on-disk filename convention.")
            return np.array([]), []
        combined = pd.concat(run_dfs, axis=1).fillna(0.0)
        mean_fi  = combined.mean(axis=1)
        feature_names = list(mean_fi.index)
        return mean_fi.values, feature_names

    else:
        base   = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
        fi_csv = base / "rf_loso_100runs_feature_importances_data.csv"
        if not fi_csv.exists():
            return np.array([]), []
        df = pd.read_csv(fi_csv)
        feature_names = list(df["feature_name"])
        return df["mean_importance"].values, feature_names


def _load_shap_summary(
    pipeline: str, dim_info: dict
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return (mean_abs_shap, mean_signed_shap, feature_names) averaged over all
    true runs.

    Per run, SHAP values (n_samples x n_features) are reduced to per-feature
    mean(|SHAP|) (overall contribution magnitude) and mean(SHAP) (signed: net
    push toward the positive class). These per-run vectors are then averaged
    across the 100 true runs. Unselected features have SHAP == 0 for every
    sample (see compute_shap_values_for_pipeline), so they contribute 0 to both
    reductions without special-casing.

    WS:   true_runs/run_N/rf_ws_shap_values_stacked.pkl
    LOSO: true_runs/run_N/rf_loso_100runs_shap_values.pkl
    """
    if pipeline == "ws":
        base     = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        pkl_glob = "true_runs/run_*/rf_ws_shap_values_stacked.pkl"
    else:
        base     = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
        pkl_glob = "true_runs/run_*/rf_loso_100runs_shap_values.pkl"

    abs_per_run:    list[np.ndarray] = []
    signed_per_run: list[np.ndarray] = []
    feature_names: list[str] = []
    for pkl_path in sorted(base.glob(pkl_glob)):
        with open(pkl_path, "rb") as f:
            run_data = pickle.load(f)
        shap_vals = np.asarray(run_data["shap_values"])
        y_true    = np.asarray(run_data["y_true"])
        if not feature_names:
            feature_names = list(run_data["feature_names"])
        abs_per_run.append(np.abs(shap_vals).mean(axis=0))
        # Signed direction: contrast between on-task (y=1) and off-task (y=0)
        # trials. mean(shap over all samples) cancels to ~0 for balanced classes
        # by construction — per-class contrast gives meaningful direction.
        shap_cls1 = shap_vals[y_true == 1].mean(axis=0)
        shap_cls0 = shap_vals[y_true == 0].mean(axis=0)
        signed_per_run.append(shap_cls1 - shap_cls0)

    if not abs_per_run:
        print(f"  ! No SHAP pkl matched '{pkl_glob}' under {base} "
              f"— panel will be empty, not just sparse. Check the glob pattern still "
              f"matches the on-disk filename convention.")
        return np.array([]), np.array([]), []

    mean_abs_shap    = np.mean(abs_per_run, axis=0)
    mean_signed_shap = np.mean(signed_per_run, axis=0)
    return mean_abs_shap, mean_signed_shap, feature_names


# =============================================================================
# Figure F — WS-vs-LOSO feature scatter: Gini importance, |SHAP|, signed SHAP
# =============================================================================


def _build_gini_scatter_fig() -> go.Figure | None:
    """
    One panel per dimension: WS vs LOSO RF Gini feature importance.

    Every common feature is drawn; the union of each pipeline's top-N
    (``SHAP_SCATTER_TOP_N``) is labelled. Both axes are min-max scaled to [0, 1]
    over each pipeline's own feature space so the two — whose raw importances
    live on very different scales — are visually comparable.

    Returns
    -------
    go.Figure or None
        ``None`` when the importances on disk are degenerate, with the reason
        printed. Both pipelines run with ``feature_selection.method: "none"``,
        and until the 2026-07-30 fix to ``_extract_fold_importances`` the
        exporter only populated importances inside its ``feature_selection``
        branch — so every ``*_feature_importances.csv`` currently on disk is
        uniformly zero. That is a stale-data condition a re-run clears, not a
        code path worth drawing, and returning ``None`` lets the SHAP figures
        (whose data is intact) still be produced.
    """
    fig = _new_dimension_grid_fig()

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col = col_idx + 1
        print(f"  Loading Gini FI [{dim_info['label']}]…", flush=True)

        ws_fi, ws_feats = _load_mean_gini_importance("ws",   dim_info)
        lo_fi, lo_feats = _load_mean_gini_importance("loso", dim_info)
        if len(ws_fi) == 0 or len(lo_fi) == 0:
            continue

        for values, side in ((ws_fi, "WS"), (lo_fi, "LOSO")):
            if np.ptp(values) == 0:
                print(
                    f"  ! {dim_info['label']}: all {len(values)} {side} Gini "
                    f"importances equal {values[0]:g}. The RF importances were "
                    f"exported as zeros while feature selection was disabled "
                    f"(fixed in utils/ml_utils.py:_extract_fold_importances); "
                    f"re-run the pipeline to populate them. Skipping this figure."
                )
                return None

        x_idx, y_idx, common = _common_feature_indices(ws_feats, lo_feats)
        if not common:
            continue

        # Scale over each pipeline's full feature space so that 1.0 = top within
        # that pipeline (not just within the common intersection).
        ws_fi_sc = _minmax_scale(ws_fi)
        lo_fi_sc = _minmax_scale(lo_fi)
        x_common = np.array([ws_fi_sc[x_idx[f]] for f in common])
        y_common = np.array([lo_fi_sc[y_idx[f]] for f in common])

        top_idx = _select_top_union(x_common, y_common, SHAP_SCATTER_TOP_N)

        _add_scatter_panel(
            fig, col, dim_info, x_common, y_common, common, top_idx,
            x_label="WS Gini importance (min-max)",
            y_label="LOSO Gini importance (min-max)",
        )

    fig.update_layout(
        title=dict(
            text=(f"LOSO vs WS — RF Feature Importance "
                  f"(Gini, min-max scaled; r over all common features, "
                  f"top-{SHAP_SCATTER_TOP_N} of each pipeline labelled)"),
            font=dict(size=14, family="Times New Roman"),
        ),
        template="plotly_white",
        height=380,
        width=320 * len(DIMENSIONS),
        margin=dict(l=60, r=30, t=100, b=60),
        font=dict(family="Times New Roman", size=12),
    )
    return fig


def _build_shap_scatter_figs() -> tuple[go.Figure, go.Figure]:
    """
    Build the WS-vs-LOSO mean |SHAP| and mean signed-SHAP scatter figures.

    Both figures share a single data-loading pass (loading the 100 per-run SHAP
    pkls per pipeline/dimension is the dominant cost) and the same top-N-union
    feature selection by mean |SHAP|, so the absolute and directional panels
    show identical feature sets — one for contribution magnitude, one for
    direction (positive mean SHAP = pushes toward the positive class).

    Returns
    -------
    tuple[go.Figure, go.Figure]
        (fig_absolute, fig_directional)
    """
    fig_abs = _new_dimension_grid_fig()
    fig_dir = _new_dimension_grid_fig()

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col = col_idx + 1
        print(f"  Loading SHAP [{dim_info['label']}]…", flush=True)

        ws_abs, ws_signed, ws_feats = _load_shap_summary("ws",   dim_info)
        lo_abs, lo_signed, lo_feats = _load_shap_summary("loso", dim_info)
        if len(ws_abs) == 0 or len(lo_abs) == 0:
            continue

        x_idx, y_idx, common = _common_feature_indices(ws_feats, lo_feats)
        if not common:
            continue

        # Scale over each pipeline's full feature space (not just the common
        # intersection) so that 1.0 means "top feature in that pipeline's own
        # feature space" — making the two axes truly comparable.
        ws_abs_sc     = _minmax_scale(ws_abs)
        lo_abs_sc     = _minmax_scale(lo_abs)
        ws_signed_sc  = _symmetric_scale(ws_signed)
        lo_signed_sc  = _symmetric_scale(lo_signed)

        ws_abs_c    = np.array([ws_abs_sc[x_idx[f]]    for f in common])
        lo_abs_c    = np.array([lo_abs_sc[y_idx[f]]    for f in common])
        ws_signed_c = np.array([ws_signed_sc[x_idx[f]] for f in common])
        lo_signed_c = np.array([lo_signed_sc[y_idx[f]] for f in common])

        # Feature selection (and pairing between the two figures) is driven by
        # |SHAP|; the directional figure re-uses the same features.
        top_idx = _select_top_union(ws_abs_c, lo_abs_c, SHAP_SCATTER_TOP_N)

        _add_scatter_panel(
            fig_abs, col, dim_info, ws_abs_c, lo_abs_c, common, top_idx,
            x_label="WS mean |SHAP| (min-max)",
            y_label="LOSO mean |SHAP| (min-max)",
        )
        _add_scatter_panel(
            fig_dir, col, dim_info, ws_signed_c, lo_signed_c, common, top_idx,
            x_label="WS mean SHAP, signed (symmetric)",
            y_label="LOSO mean SHAP, signed (symmetric)",
            axis_range=[-1.05, 1.05],
        )

    for fig, title in (
        (fig_abs, (f"LOSO vs WS — Mean |SHAP| "
                   f"(min-max scaled; r over all common features, "
                   f"top-{SHAP_SCATTER_TOP_N} of each pipeline labelled)")),
        (fig_dir, (f"LOSO vs WS — Mean SHAP, signed "
                   f"(same features as |SHAP|, symmetric scaled ÷ max|SHAP|)")),
    ):
        fig.update_layout(
            title=dict(text=title, font=dict(size=14, family="Times New Roman")),
            template="plotly_white",
            height=380,
            width=320 * len(DIMENSIONS),
            margin=dict(l=60, r=30, t=100, b=60),
            font=dict(family="Times New Roman", size=12),
        )

    return fig_abs, fig_dir


def _pretty_marker_name(name: str) -> str:
    """Shorten a raw marker name for an axis label, keeping its family tag."""
    for old, new in (("psd_relative_", "PSD "), ("kolmogorov_complexity", "Kolmogorov"),
                     ("slowwaves_", "SW "), ("wsmi_", "wSMI "), ("PE_", "PE ")):
        if name.startswith(old) or name == old.rstrip("_"):
            return name.replace(old, new)
    return name


# =============================================================================
# Figure — plain vs residualized: does residualizing change which markers matter?
# =============================================================================

_RESIDUAL_PROFILES = CONSISTENCY_DIR / "residual_marker_profiles.csv"
_RESIDUAL_TOP_N = 5
_RESIDUAL_WIDTH_MM, _RESIDUAL_HEIGHT_MM = 180.0, 250.0
_RESIDUAL_SINGLE_MM = (150.0, 78.0)
# Labels name the top markers of each side; the union is at most twice this and
# collapses to fewer when the two sides agree, which is itself informative.
_RESIDUAL_LABEL_N = 5


def _residual_panel(
    fig: go.Figure, row: int, col: int, block: pd.DataFrame, color: str
) -> None:
    """
    Draw one plain-vs-residualized scatter and return its two summary statistics.

    Parameters
    ----------
    block : pd.DataFrame
        The 23 marker rows for one (dimension, pipeline) combination.

    Notes
    -----
    Axes are the absolute mean(|SHAP|), not a share of the dimension's total. A
    share is relative — a marker's share falls whenever the others rise — so a
    share plot cannot separate "this marker lost attribution" from "everything
    else gained". On absolute axes the identity line is meaningful: a cloud
    sitting below it means residualizing cost the model attribution outright.

    Labels name the union of each side's top ``_RESIDUAL_LABEL_N`` markers — the
    leaders of the full model and the leaders of the residualized one. Taking the
    union rather than one side's ranking is what makes a promotion visible: a
    marker that only matters after residualizing has no reason to appear in the
    full model's top five, and labelling by the full model alone would hide
    exactly the change the figure exists to show. Where the two lists agree the
    union shrinks below ten, which is itself a reading of the panel.

    Labelling the markers furthest from the diagonal was tried first and reads
    badly: the biggest movers are often small markers whose displacement is
    large only relative to their own size, leaving the points that carry the
    model anonymous.

    This is a labelling choice only. Both statistics cover all 23 markers, so no
    selected subset feeds a number.
    """
    from scipy.stats import spearmanr

    full = block["abs_full"].to_numpy()
    residual = block["abs_residual"].to_numpy()
    names = [_pretty_marker_name(m) for m in block["marker"]]

    top_full = block.nlargest(_RESIDUAL_LABEL_N, "abs_full")["marker"]
    top_residual = block.nlargest(_RESIDUAL_LABEL_N, "abs_residual")["marker"]
    rho, _ = spearmanr(full, residual)

    # Axes span the data, not [0, max]. Anchoring at zero pushed the cloud into
    # a corner occupying 41-70% of the panel, where any positive-valued scatter
    # reads as a tight diagonal band no matter what its correlation is — which is
    # why the annotated rho looked wrong against the picture. The identity line
    # still carries "did attribution drop", and now the visible spread matches
    # the number.
    low = float(min(full.min(), residual.min()))
    high = float(max(full.max(), residual.max()))
    pad = 0.10 * (high - low)
    axis_lo, axis_hi = low - pad, high + pad
    fig.add_shape(type="line", x0=axis_lo, y0=axis_lo, x1=axis_hi, y1=axis_hi,
                  line=dict(color=PALETTE["neutral"]["permutation"],
                            width=0.9, dash="dot"),
                  layer="below", row=row, col=col)

    named = set(top_full) | set(top_residual)
    mask = block["marker"].isin(named).to_numpy()

    fig.add_trace(go.Scatter(
        x=full[~mask], y=residual[~mask], mode="markers",
        marker=dict(color=color, size=5, opacity=0.55,
                    line=dict(color="white", width=0.4)),
        customdata=[n for n, m in zip(names, mask) if not m],
        hovertemplate="%{customdata}<br>full %{x:.4f}<br>residualized %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    # Labels are pushed away from the cloud's centre — points on the left of the
    # panel get their name on the left, points on the right get it on the right —
    # so a label never lies over the points it is not naming. Ties in the same
    # half alternate vertically as a second separation.
    # Three vertical offsets per side, not two: the labelled markers are the
    # largest ones and in the within-subject panels they cluster in the same
    # corner, so a two-way alternation still stacked names on top of each other.
    x_mid = float(np.median(full))
    order = np.argsort(np.argsort(residual[mask]))
    left_cycle = np.array(["middle left", "top left", "bottom left"])
    right_cycle = np.array(["middle right", "top right", "bottom right"])
    positions = np.where(
        full[mask] < x_mid, left_cycle[order % 3], right_cycle[order % 3]
    )
    fig.add_trace(go.Scatter(
        x=full[mask], y=residual[mask], mode="markers+text",
        marker=dict(color=color, size=7, line=dict(color="white", width=0.6)),
        text=[n for n, m in zip(names, mask) if m],
        textposition=positions, textfont=dict(size=6.5, color="#333333"),
        cliponaxis=False,
        hovertemplate="%{text}<br>full %{x:.4f}<br>residualized %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.add_annotation(
        text=f"Spearman ρ = {rho:.2f}",
        xref="x domain", yref="y domain", x=0.03, y=0.97,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=6.5, color=color), align="left", row=row, col=col,
    )
    fig.update_xaxes(range=[axis_lo, axis_hi], nticks=4, tickfont=dict(size=6),
                     row=row, col=col)
    fig.update_yaxes(range=[axis_lo, axis_hi], nticks=4, tickfont=dict(size=6),
                     row=row, col=col)


def _residual_color(contrast_key: str) -> str:
    """Palette colour for a dimension, covering the quadratic keys too."""
    return DIM_COLORS.get(
        {"on_off": "onoff"}.get(contrast_key, contrast_key),
        PALETTE["quadratic"].get(contrast_key, PALETTE["neutral"]["accent"]),
    )


def _residual_dimension_order(profiles: pd.DataFrame) -> list[str]:
    """
    Dimension order for the residual figures: the one this script already uses.

    ``GROUP_ROW_DIMENSIONS`` drives the other group-level figures here
    (the "+ residualized" group-level and spatial-comparison panels), so the
    residual panels follow it too rather than the order they happen to appear
    in ``config_feature_consistency.yaml`` — a reader moving between figures
    should not have to re-locate a dimension. Since the quadratic terms were
    removed (2026-08-13) it is simply the canonical 5 (they aren't in that
    5-entry list), which is exactly the "appended, not interleaved" ordering
    CLAUDE.md's "Dimension Order (Figures)" rule forbids — this function must
    use the 7-entry list, not the 5-entry one.

    ``color_palette.yaml``, ``config_feature_consistency.yaml`` and
    ``GROUP_ROW_DIMENSIONS`` here all agree on dimension order (see CLAUDE.md
    "Dimension Order (Figures)").
    """
    present = list(dict.fromkeys(profiles["contrast"]))
    canonical = [entry["key"] for entry in GROUP_ROW_DIMENSIONS]
    ordered = [key for key in canonical if key in present]
    return ordered + [key for key in present if key not in ordered]


def _load_residual_profiles() -> pd.DataFrame | None:
    """Read the precomputed profiles, or explain why the figures are skipped."""
    if not _RESIDUAL_PROFILES.exists():
        print(f"  ! {_RESIDUAL_PROFILES} not found. "
              f"Run scripts/compute_residual_profiles.py first; skipping.")
        return None
    profiles = pd.read_csv(_RESIDUAL_PROFILES)
    if "abs_full" not in profiles.columns:
        print(f"  ! {_RESIDUAL_PROFILES} predates the absolute-scale columns "
              f"(abs_full / abs_residual). Re-run scripts/compute_residual_profiles.py; "
              f"skipping rather than falling back to shares, which are relative and "
              f"answer a different question.")
        return None
    return profiles


def _build_residual_comparison_figure() -> go.Figure | None:
    """
    Marker attribution before vs after residualizing, all dimensions.

    One row per dimension, two columns facing each other: within-subject on the
    left, LOSO on the right. Facing columns rather than a flat grid so the two
    pipelines for a dimension sit side by side and the question "does
    residualizing hit both the same way?" is answered by a glance along a row.

    Each dimension is paired with the contrast that residualizes it against the
    other probe dimensions (within-subject OLS): if the same markers carry the
    signal at the same strength before and after, it is specific to that
    dimension; if the cloud reorders or drops toward the axis, it was shared.

    Returns
    -------
    go.Figure or None
        ``None`` when the precomputed profiles are absent or predate the
        absolute-scale columns, with the reason printed.
    """
    profiles = _load_residual_profiles()
    if profiles is None:
        return None

    dimensions = _residual_dimension_order(profiles)
    pipelines = [("ws", "within-subject"), ("loso", "LOSO")]

    fig = make_subplots(
        rows=len(dimensions), cols=2,
        horizontal_spacing=0.10, vertical_spacing=0.045,
        subplot_titles=[
            f"{profiles.loc[profiles['contrast'] == key, 'label'].iloc[0]} · {name}"
            for key in dimensions for _, name in pipelines
        ],
    )
    for index, key in enumerate(dimensions):
        for offset in (0, 1):
            fig.layout.annotations[index * 2 + offset].font.size = 7.5
            fig.layout.annotations[index * 2 + offset].font.color = _residual_color(key)

    for row, key in enumerate(dimensions, start=1):
        for col, (pipeline, _) in enumerate(pipelines, start=1):
            block = profiles[(profiles["contrast"] == key)
                             & (profiles["pipeline"] == pipeline)]
            _residual_panel(fig, row, col, block, _residual_color(key))
            if row == len(dimensions):
                fig.update_xaxes(title_text="full model — mean |SHAP|",
                                 title_font=dict(size=7), row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="residualized model",
                                 title_font=dict(size=7), row=row, col=col)

    fig.update_layout(
        template="plotly_white",
        width=int(_RESIDUAL_WIDTH_MM * _MM_TO_PX),
        height=int(_RESIDUAL_HEIGHT_MM * _MM_TO_PX),
        margin=dict(l=52, r=18, t=28, b=40),
        font=dict(size=8),
    )
    return fig


def _build_residual_single_figures() -> list[tuple[go.Figure, str]]:
    """
    One standalone figure per dimension, its two pipelines facing each other.

    The combined grid is 14 panels on a page and each is small; these are the
    same panels at a size where the marker labels are comfortable to read.

    Returns
    -------
    list[tuple[go.Figure, str]]
        ``(figure, output stem)`` pairs, empty when the profiles are missing.
    """
    profiles = _load_residual_profiles()
    if profiles is None:
        return []

    pipelines = [("ws", "within-subject"), ("loso", "LOSO")]
    figures: list[tuple[go.Figure, str]] = []

    for key in _residual_dimension_order(profiles):
        label = profiles.loc[profiles["contrast"] == key, "label"].iloc[0]
        color = _residual_color(key)
        fig = make_subplots(
            rows=1, cols=2, horizontal_spacing=0.12,
            subplot_titles=[f"{label} · {name}" for _, name in pipelines],
        )
        for annotation in fig.layout.annotations:
            annotation.font.size = 9
            annotation.font.color = color

        for col, (pipeline, _) in enumerate(pipelines, start=1):
            block = profiles[(profiles["contrast"] == key)
                             & (profiles["pipeline"] == pipeline)]
            _residual_panel(fig, 1, col, block, color)
            fig.update_xaxes(title_text="full model — mean |SHAP|",
                             title_font=dict(size=8), tickfont=dict(size=7),
                             row=1, col=col)
            fig.update_yaxes(tickfont=dict(size=7), row=1, col=col)
        fig.update_yaxes(title_text="residualized model", title_font=dict(size=8),
                         row=1, col=1)

        fig.update_layout(
            template="plotly_white",
            width=int(_RESIDUAL_SINGLE_MM[0] * _MM_TO_PX),
            height=int(_RESIDUAL_SINGLE_MM[1] * _MM_TO_PX),
            margin=dict(l=54, r=18, t=30, b=44),
            font=dict(size=9),
        )
        figures.append((fig, f"residual_vs_plain_{key}"))
    return figures


# =============================================================================
# Figure — WS-vs-LOSO SHAP direction per marker, every dimension in one panel
# =============================================================================


# Millimetres to pixels at the 96 dpi plotly assumes, so the figure keeps the
# physical size it was designed at when this script queues it by pixel count.
_MM_TO_PX = 96.0 / 25.4
_DIRECTION_WIDTH_MM, _DIRECTION_HEIGHT_MM = 180.0, 185.0


def _build_marker_direction_figure() -> go.Figure | None:
    """
    Build the combined directional forest: do the two pipelines use each marker
    the same way, for every dimension at once.

    Returns
    -------
    go.Figure or None
        ``None`` when the feature-consistency tables have not been generated
        yet, with the reason printed. Those tables come from a separate,
        expensive pass over the per-run SHAP pickles
        (``scripts/feature_consistency_analysis.py``), so this figure is a
        consumer of that analysis rather than something this script can compute
        on its own — and a missing input is a "run the analysis first"
        condition, not a reason to fail the whole figure batch.

    Notes
    -----
    The builder lives in ``make_fig_marker_direction.py`` and is imported rather
    than duplicated, so the standalone per-dimension figures and this one can
    never drift apart. That module registers sciplot's ``sci`` template as the
    plotly default; the default is restored afterwards so the other figures in
    this script keep the ``plotly_white`` styling they were designed against.
    """
    required = ["group_summary.csv", "marker_level_consistency.csv",
                "marker_direction_per_subject.csv"]
    missing = [name for name in required if not (CONSISTENCY_DIR / name).exists()]
    if missing:
        print(f"  ! {CONSISTENCY_DIR} is missing {', '.join(missing)}. "
              f"Run scripts/feature_consistency_analysis.py first; skipping this figure.")
        return None

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_fig_marker_direction import (  # noqa: E402
        build_combined_figure,
        init_palette,
        subject_classification_significance,
    )

    summary = pd.read_csv(CONSISTENCY_DIR / "group_summary.csv")
    per_marker = pd.read_csv(CONSISTENCY_DIR / "marker_level_consistency.csv")
    per_subject_direction = pd.read_csv(CONSISTENCY_DIR / "marker_direction_per_subject.csv")

    config = yaml.safe_load(
        (Path(__file__).resolve().parent / "config_feature_consistency.yaml").read_text()
    )
    contrasts = {entry["key"]: entry for entry in config["contrasts"]}

    # Canonical dimension order (CLAUDE.md "Dimension Order (Figures)"), taken
    # from config_feature_consistency.yaml's own contrasts[] — one of that
    # rule's documented "wired into" locations, and already listed on/off,
    # valence, selfother, time, confidence. Panel position
    # here is literally row-major over this row order (build_combined_figure
    # fills a 2-column grid by summary index), so sorting by legacy_group_rho
    # instead (as make_fig_marker_direction.py's own standalone load_tables()
    # does, for its ranked-by-consistency figure) scrambles the grid relative
    # to every other dimension-ordered figure in this script.
    canonical_order = {entry["key"]: i for i, entry in enumerate(config["contrasts"])}
    summary = (
        summary.assign(_order=summary["contrast"].map(canonical_order))
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    summary["label"] = summary["contrast"].map(
        {key: entry["label"] for key, entry in contrasts.items()}
    )

    significance_by_contrast = {
        key: subject_classification_significance(contrasts[key])
        for key in summary["contrast"]
    }

    previous_default = pio.templates.default
    pal = init_palette()
    fig = build_combined_figure(summary, per_marker, per_subject_direction,
                                significance_by_contrast, pal)
    pio.templates.default = previous_default

    fig.update_layout(width=int(_DIRECTION_WIDTH_MM * _MM_TO_PX),
                      height=int(_DIRECTION_HEIGHT_MM * _MM_TO_PX))
    return fig


# =============================================================================
# Figure G — Dimension-level median summary (SD + mean vs LOSO AUC)
# =============================================================================


def plot_dimension_median_summary() -> go.Figure:
    """
    Two-panel scatter figure at the dimension level (5 points each).

    Panel 1 — SD: x = SD across subjects of per-subject within-subject medians,
               y = dimension's mean LOSO AUC.
    Panel 2 — Mean: x = mean across subjects of per-subject within-subject medians,
               y = dimension's mean LOSO AUC.

    Tests whether a dimension's group-level decodability is related to (a) how
    consistent subjects' scale-position is, and (b) how far the average subject
    sits from the midpoint.  Both panels share the same y-axis range.

    Data sources:
    - Per-subject medians: PROBE_DATA_PATH, filtered to each dimension's subjects_final
    - Mean LOSO AUC: np.mean of true_mean_aucs from load_group_data("loso", dim_info)
    - subjects_final: loaded from used_config.yaml in the LOSO results dir
    """
    probe_df = pd.read_csv(PROBE_DATA_PATH)
    probe_df["subject"] = probe_df["subject"].astype(str).str.zfill(2)

    from scipy.stats import pearsonr, linregress

    rows = []
    for dim_info in DIMENSIONS:
        loso_cfg_path = (
            RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf" / "used_config.yaml"
        )
        if not loso_cfg_path.exists():
            print(f"  Warning: missing used_config.yaml for {dim_info['label']} — skipping")
            continue
        with open(loso_cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        subjects_final = [str(s).zfill(2) for s in cfg["_data_provenance"]["subjects_final"]]

        label_col = dim_info["label_source"]
        if label_col not in probe_df.columns:
            print(f"  Warning: column '{label_col}' not in probe data — skipping {dim_info['label']}")
            continue

        sub_medians = (
            probe_df[probe_df["subject"].isin(subjects_final)]
            .groupby("subject")[label_col]
            .median()
        )
        if sub_medians.empty:
            continue

        true_aucs, _ = load_group_data("loso", dim_info)
        if len(true_aucs) == 0:
            continue

        rows.append({
            "label":       dim_info["label"],
            "color":       dim_info["color"],
            "median_sd":   float(sub_medians.std(ddof=1)),
            "median_mean": float(sub_medians.mean()),
            "mean_auc":    float(np.mean(true_aucs)),
            "n_subjects":  len(sub_medians),
        })

    if not rows:
        print("  No data available for dimension median summary — skipping.")
        return go.Figure()

    dim_df = pd.DataFrame(rows)

    def _panel_traces(
        fig: go.Figure,
        x_col: str,
        col: int,
        show_yaxis_title: bool,
    ) -> None:
        x_vals = dim_df[x_col].values.astype(float)
        y_vals = dim_df["mean_auc"].values.astype(float)

        for _, row in dim_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row[x_col]],
                y=[row["mean_auc"]],
                mode="markers+text",
                marker=dict(color=row["color"], size=14,
                            line=dict(color="white", width=1.5)),
                text=[f"<b>{row['label']}</b>"],
                textposition="top center",
                textfont=dict(size=11, color=row["color"]),
                showlegend=False,
                hovertemplate=(
                    f"{row['label']}<br>"
                    f"{x_col}: %{{x:.2f}}<br>"
                    f"mean AUC: %{{y:.3f}}<extra></extra>"
                ),
            ), row=1, col=col)

        if len(x_vals) >= 3:
            r_val, p_val = pearsonr(x_vals, y_vals)
            slope, intercept, *_ = linregress(x_vals, y_vals)
            x_line = np.array([x_vals.min(), x_vals.max()])
            fig.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                showlegend=False,
            ), row=1, col=col)

            x_ref = "x" if col == 1 else f"x{col}"
            y_ref = "y" if col == 1 else f"y{col}"
            fig.add_annotation(
                text=f"r = {r_val:.2f}, p = {p_val:.3f}",
                xref=f"{x_ref} domain", yref=f"{y_ref} domain",
                x=0.05, y=0.97, xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=12, color="black"),
            )

        # chance line
        fig.add_hline(
            y=0.5, line_dash="dash", line_color=PERM_COLOR, opacity=0.5, row=1, col=col,
        )

    fig = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.12,
        subplot_titles=[
            "SD of per-subject medians vs LOSO AUC",
            "Mean of per-subject medians vs LOSO AUC",
        ],
    )

    _panel_traces(fig, "median_sd",   col=1, show_yaxis_title=True)
    _panel_traces(fig, "median_mean", col=2, show_yaxis_title=False)

    y_all = dim_df["mean_auc"].values
    y_lo  = max(0.45, float(y_all.min()) - 0.05)
    y_hi  = min(1.0,  float(y_all.max()) + 0.07)

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>LOSO Decodability vs Dimension-level Median Position</b>",
            font=dict(size=15, family="Times New Roman"),
        ),
        height=480,
        width=900,
        margin=dict(l=70, r=40, t=100, b=70),
        font=dict(family="Times New Roman", size=13),
    )

    fig.update_yaxes(
        title_text="Mean LOSO AUC",
        range=[y_lo, y_hi],
        showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        row=1, col=1,
    )
    fig.update_yaxes(
        range=[y_lo, y_hi],
        showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        showticklabels=False,
        row=1, col=2,
    )
    fig.update_xaxes(
        title_text="SD of per-subject medians",
        showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        row=1, col=1,
    )
    fig.update_xaxes(
        title_text="Mean of per-subject medians (0–100 scale)",
        showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        row=1, col=2,
    )

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.family = "Times New Roman"

    return fig


# =============================================================================
# Probe-level probability scatter: WS proba vs LOSO proba
# =============================================================================


def _normalise_subject(values: pd.Series) -> pd.Series:
    """
    Coerce a subject column to the project's zero-padded string form ("03").

    The two sides of the probe-level merge read different files: the
    within-subject consolidated CSV stores subjects as strings, while the raw
    LOSO per-run CSVs store them as integers. Merging those raises a dtype
    error in pandas — which is the good case. Had one side been "3" and the
    other "03", both object dtype, the merge would have succeeded and returned
    zero rows, i.e. an empty panel with no error at all.
    """
    return values.astype(str).str.strip().str.zfill(2)


def _load_probe_predictions_ws(dim_info: dict) -> pd.DataFrame:
    """
    Load WS consolidated sample predictions for a dimension.

    Returns DataFrame with columns [subject, task, probe_number, ws_proba, y_true].
    """
    base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
    path = base / "rf_ws_consolidated_sample_predictions.csv"
    if not path.exists():
        print(f"  ! {path} not found — panel will be empty, not just sparse. "
              f"Check the on-disk filename convention still matches.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["subject"] = _normalise_subject(df["subject"])
    return df[["subject", "task", "probe_number", "proba_mean", "y_true_first"]].rename(
        columns={"proba_mean": "ws_proba", "y_true_first": "y_true"}
    )


def _load_probe_predictions_loso(dim_info: dict) -> pd.DataFrame:
    """
    Load and aggregate LOSO sample predictions across all runs for a dimension.

    Returns DataFrame with columns [subject, task, probe_number, loso_proba, y_true].
    """
    base = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
    run_csvs = sorted(base.glob("true_runs/run_*/*_sample_predictions.csv"))
    if not run_csvs:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in run_csvs]
    all_runs = pd.concat(dfs, ignore_index=True)
    all_runs["subject"] = _normalise_subject(all_runs["subject"])
    agg = (
        all_runs.groupby(["subject", "task", "probe_number"])
        .agg(loso_proba=("y_proba", "mean"), y_true=("y_true", "first"))
        .reset_index()
    )
    return agg


def plot_proba_ws_vs_loso_scatter() -> go.Figure:
    """
    One panel per dimension: scatter of per-probe predicted probability (WS vs LOSO).

    Each point is one probe. Fill encodes true label per project convention:
      - filled marker  → y_true = 1 (high end of dimension)
      - hollow marker  → y_true = 0 (low end of dimension)
    Diagonal y=x and chance lines at 0.5 drawn for reference.
    """
    n_dims = len(DIMENSIONS)

    fig = make_subplots(
        rows=1, cols=n_dims,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.05,
        subplot_titles=[d["label"] for d in DIMENSIONS],
    )

    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color  = dim_info["color"]
        fig.layout.annotations[i].font.size   = 14
        fig.layout.annotations[i].font.family = "Times New Roman"

    proba_min, proba_max = 0.0, 1.0

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col   = col_idx + 1
        color = dim_info["color"]

        ws_df   = _load_probe_predictions_ws(dim_info)
        loso_df = _load_probe_predictions_loso(dim_info)

        if ws_df.empty or loso_df.empty:
            continue

        merged = ws_df.merge(loso_df, on=["subject", "task", "probe_number"], suffixes=("_ws", "_loso"))
        merged = merged.dropna(subset=["ws_proba", "loso_proba"])
        if merged.empty:
            print(f"  ! {dim_info['label']}: the WS ({len(ws_df)} rows) and LOSO "
                  f"({len(loso_df)} rows) probe tables share no "
                  f"(subject, task, probe_number) key — the panel will be empty, "
                  f"not merely sparse. Check the key dtypes and zero-padding.")
            continue

        y_true = merged["y_true_ws"] if "y_true_ws" in merged.columns else merged["y_true"]

        for label_val, symbol, marker_label in [
            (1, "circle",      "High (y=1)"),
            (0, "circle-open", "Low (y=0)"),
        ]:
            mask = y_true == label_val
            show_leg = (col_idx == 0)
            fig.add_trace(go.Scatter(
                x=merged.loc[mask, "ws_proba"].values,
                y=merged.loc[mask, "loso_proba"].values,
                mode="markers",
                name=marker_label,
                marker=dict(
                    symbol=symbol,
                    color=color,
                    size=5,
                    opacity=0.65,
                    line=dict(color=color, width=1.2),
                ),
                showlegend=show_leg,
                legendgroup=marker_label,
            ), row=1, col=col)

        # Diagonal y = x
        fig.add_shape(
            type="line",
            x0=proba_min, x1=proba_max,
            y0=proba_min, y1=proba_max,
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col}" if col > 1 else "y",
            line=dict(color="#888888", dash="dot", width=1.2),
        )

        # Chance lines at 0.5
        for x0, x1, y0, y1 in [
            (0.5, 0.5, proba_min, proba_max),
            (proba_min, proba_max, 0.5, 0.5),
        ]:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=y0, y1=y1,
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                line=dict(color="#BBBBBB", dash="dash", width=1.0),
            )

        # Pearson r + n annotation
        if len(merged) >= 3:
            r = float(np.corrcoef(merged["ws_proba"].values, merged["loso_proba"].values)[0, 1])
            n = len(merged)
            fig.add_annotation(
                text=f"r = {r:.2f}<br>n = {n}",
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.05, y=0.97,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=10, color=color, family="Times New Roman"),
            )

        x_ax = f"xaxis{col}" if col > 1 else "xaxis"
        y_ax = f"yaxis{col}" if col > 1 else "yaxis"
        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range     = [proba_min, proba_max]
            fig.layout[ax_name].tickmode  = "array"
            fig.layout[ax_name].tickvals  = [0.0, 0.25, 0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext  = ["0", ".25", ".5", ".75", "1"]
            fig.layout[ax_name].tickfont  = dict(size=10)
            fig.layout[ax_name].showgrid  = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline  = False

        fig.layout[x_ax].title = dict(text="WS predicted probability", font=dict(size=11))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO predicted probability", font=dict(size=11))

    fig.update_layout(
        title=dict(
            text="Per-probe predicted probability: Within-Subject vs LOSO",
            font=dict(size=16, family="Times New Roman"),
            x=0.5,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman"),
        width=350 * n_dims,
        height=400,
        legend=dict(
            orientation="v",
            x=1.01,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(size=11, family="Times New Roman"),
            title=dict(text="True label", font=dict(size=11)),
        ),
        margin=dict(t=80, b=60, l=70, r=130),
    )

    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate (fig, path, opts) — written in one kaleido/Chrome session at the end
    # so Chrome launches only once instead of once per write_image() call.
    pending: list[tuple[go.Figure, Path, dict]] = []

    def _queue(fig: go.Figure, stem: str) -> None:
        w = int(fig.layout.width  or 800)
        h = int(fig.layout.height or 600)
        for fmt in ("png", "svg"):
            pending.append((fig, OUTPUT_DIR / f"{stem}.{fmt}",
                            {"format": fmt, "width": w, "height": h, "scale": 2}))

    tasks = [
        ("ws",   "Within-Subject", "group_ws",   "individual_ws"),
        ("loso", "LOSO",           "group_loso",  "individual_loso"),
    ]

    for pipeline, title_prefix, group_stem, indiv_stem in tasks:
        print("=" * 60)
        print(f"Group-level: {title_prefix}")
        print("=" * 60)
        fig_g = plot_group_level(pipeline, title_prefix)
        _queue(fig_g, group_stem)

        print("=" * 60)
        print(f"Individual-level: {title_prefix}")
        print("=" * 60)
        result = plot_individual(pipeline, title_prefix, indiv_stem)
        if result is not None:
            fig_i, w_i, h_i = result
            for fmt in ("png", "svg"):
                pending.append((fig_i, OUTPUT_DIR / f"{indiv_stem}.{fmt}",
                                {"format": fmt, "width": int(w_i), "height": int(h_i), "scale": 2}))

        print("=" * 60)
        print(f"Group-level (+ residualized): {title_prefix}")
        print("=" * 60)
        fig_g_res = plot_group_level_with_residualized(pipeline, title_prefix)
        _queue(fig_g_res, f"{group_stem}_residualized")

        print("=" * 60)
        print(f"Individual-level (+ residualized): {title_prefix}")
        print("=" * 60)
        result_res = plot_individual_with_residualized(
            pipeline, title_prefix, f"{indiv_stem}_residualized"
        )
        if isinstance(result_res, tuple):
            fig_i_res, w_i_res, h_i_res = result_res
            for fmt in ("png", "svg"):
                pending.append((fig_i_res, OUTPUT_DIR / f"{indiv_stem}_residualized.{fmt}",
                                {"format": fmt, "width": int(w_i_res), "height": int(h_i_res), "scale": 2}))

    print("=" * 60)
    print("Group-level comparison: WS vs LOSO")
    print("=" * 60)
    _queue(plot_group_comparison(), "group_comparison")

    print("=" * 60)
    print("Group-level comparison (+ residualized): WS vs LOSO")
    print("=" * 60)
    _queue(plot_group_comparison_with_residualized(), "group_comparison_residualized")

    print("=" * 60)
    print("Spatial decoding comparison: WS vs LOSO topomaps")
    print("=" * 60)
    spatial_path = plot_spatial_comparison_panel()
    if spatial_path is not None:
        print(f"  Saved: {spatial_path}")
        print(f"  Saved: {spatial_path.with_suffix('.svg')}")

    print("=" * 60)
    print("Global Decoding + Spatial Decoding combined: WS | LOSO")
    print("=" * 60)
    combined_path = plot_group_spatial_combined()
    if combined_path is not None:
        print(f"  Saved: {combined_path}")
        print(f"  Saved: {combined_path.with_suffix('.svg')}")

    print("=" * 60)
    print("LOSO vs WS scatter")
    print("=" * 60)
    _queue(plot_loso_vs_ws_scatter(), "scatter_loso_vs_ws")

    print("=" * 60)
    print("Regression overlay")
    print("=" * 60)
    _queue(plot_regression_overlay(), "scatter_regression_overlay")

    print("=" * 60)
    print("Feature importance scatter — Gini (WS vs LOSO)")
    print("=" * 60)
    fig_gini = _build_gini_scatter_fig()
    if fig_gini is not None:
        _queue(fig_gini, "scatter_feature_importance")

    print("=" * 60)
    print("SHAP scatter — absolute & directional (WS vs LOSO)")
    print("=" * 60)
    fig_shap_abs, fig_shap_dir = _build_shap_scatter_figs()
    _queue(fig_shap_abs, "scatter_shap_absolute")
    _queue(fig_shap_dir, "scatter_shap_directional")

    print("=" * 60)
    print("Probe-level probability scatter: WS vs LOSO")
    print("=" * 60)
    _queue(plot_proba_ws_vs_loso_scatter(), "scatter_proba_ws_vs_loso")

    print("=" * 60)
    print("SHAP direction per marker — WS vs LOSO, all dimensions")
    print("=" * 60)
    fig_direction = _build_marker_direction_figure()
    if fig_direction is not None:
        _queue(fig_direction, "marker_direction_all_dimensions")

    print("=" * 60)
    print("Plain vs residualized — top markers per dimension and pipeline")
    print("=" * 60)
    fig_residual = _build_residual_comparison_figure()
    if fig_residual is not None:
        _queue(fig_residual, "residual_vs_plain_markers")
    for fig_single, stem in _build_residual_single_figures():
        _queue(fig_single, stem)

    print("=" * 60)
    print("Dimension median summary (SD + mean vs LOSO AUC)")
    print("=" * 60)
    fig_ms = plot_dimension_median_summary()
    if fig_ms.data:
        _queue(fig_ms, "dimension_median_summary")

    # Write all images through ONE kaleido/Chrome instance (one cold-start total)
    print("=" * 60)
    print(f"Writing {len(pending)} images (1 kaleido instance)…")
    print("=" * 60)

    async def _write_all() -> None:
        async with _kaleido.Kaleido() as k:
            for fig, path, opts in pending:
                await k.write_fig(fig, path=path, opts=opts)
                print(f"  Saved: {path}", flush=True)

    asyncio.run(_write_all())
    print("\nDone. Figures in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
