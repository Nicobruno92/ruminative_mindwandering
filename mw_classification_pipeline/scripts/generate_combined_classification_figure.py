#!/usr/bin/env python
"""
Generate combined classification figures showing WS and LOSO results.

Outputs (in results/MW_Classification/combined_figures/):
    group_ws.{png,svg}        — WS group-level AUC per dimension
    group_loso.{png,svg}      — LOSO group-level AUC per dimension
    individual_ws.{png,svg}   — WS per-subject AUC per dimension
    individual_loso.{png,svg} — LOSO per-subject AUC per dimension

Each dimension has a consistent color used across all four figures.
True = dimension color; Shuffled = grey.

Usage (from project root):
    /path/to/miniforge3/envs/ML/bin/python \
        mw_classification_pipeline/scripts/generate_combined_classification_figure.py
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
OUTPUT_DIR      = Path("mw_classification_pipeline/results/combined_figures")
PROBE_DATA_PATH = Path("results/Behavior/probe_data/probe_level_aggregated_data.csv")

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
        "key":          "time",
        "ws_dir":       "time_within_median",
        "loso_dir":     "time_within_median",
        "label":        "Time",
        "color":        DIM_COLORS["time"],   # purple
        "label_source": "time",
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
            name="True", legendgroup="True",
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
            line_color="lightgray",
            fillcolor="lightgray",
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
            legendgroup="True",
            scalegroup=f"True_{dk}",
            name="True",
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
            marker=dict(colors=[color, "lightgray"]),
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
                name="True", legendgroup="True",
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
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
    )
    return fig


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
    x_vals: np.ndarray, y_vals: np.ndarray, feat_names: list[str],
    x_label: str, y_label: str,
    axis_range: list[float] | None = None,
) -> None:
    """
    Add a WS-vs-LOSO scatter panel to ``fig`` at column ``col``.

    Draws the marker+text scatter, a y=x diagonal, a linear regression line,
    and a correlation annotation, then styles both axes to ``axis_range``.

    Parameters
    ----------
    axis_range : list[float], optional
        [lo, hi] for both axes.  Defaults to [-0.05, 1.05] (min-max figures).
        Pass [-1.05, 1.05] for signed/directional figures.
    """
    if axis_range is None:
        axis_range = [-0.05, 1.05]
    color = dim_info["color"]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        marker=dict(color=color, size=7, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        text=[f.split("_")[0] for f in feat_names],
        textposition="top center",
        textfont=dict(size=7, family="Times New Roman"),
        customdata=feat_names,
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

    # Regression line
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_reg = np.array([diag_lo, diag_hi])
    fig.add_trace(go.Scatter(
        x=x_reg, y=m * x_reg + b,
        mode="lines",
        line=dict(color=color, width=2.0),
        showlegend=False,
    ), row=1, col=col)

    # Correlation
    r = float(np.corrcoef(x_vals, y_vals)[0, 1])
    fig.add_annotation(
        text=f"r = {r:.2f}, n={len(x_vals)}",
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

    WS:   mean over true_runs/run_N/rf_loso_feature_importances.csv
    LOSO: rf_loso_100runs_feature_importances_data.csv (already aggregated)
    """
    if pipeline == "ws":
        base     = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        run_dfs  = []
        for fi_csv in sorted(base.glob("true_runs/run_*/rf_loso_feature_importances.csv")):
            df = pd.read_csv(fi_csv).rename(columns={"feature": "feature_name",
                                                      "importance": "mean_importance"})
            run_dfs.append(df.set_index("feature_name")["mean_importance"])
        if not run_dfs:
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

    WS:   true_runs/run_N/rf_loso_shap_values_stacked.pkl
    LOSO: true_runs/run_N/rf_loso_100runs_shap_values.pkl
    """
    if pipeline == "ws":
        base     = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
        pkl_glob = "true_runs/run_*/rf_loso_shap_values_stacked.pkl"
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
        return np.array([]), np.array([]), []

    mean_abs_shap    = np.mean(abs_per_run, axis=0)
    mean_signed_shap = np.mean(signed_per_run, axis=0)
    return mean_abs_shap, mean_signed_shap, feature_names


# =============================================================================
# Figure F — WS-vs-LOSO feature scatter: Gini importance, |SHAP|, signed SHAP
# =============================================================================


def _build_gini_scatter_fig() -> go.Figure:
    """
    One panel per dimension: WS vs LOSO RF Gini feature importance.

    Features shown are the union of each pipeline's top-N (``SHAP_SCATTER_TOP_N``)
    by Gini importance; both axes are min-max scaled to [0, 1] over the common
    feature set so the two pipelines — whose raw importances live on very
    different scales — are visually comparable.
    """
    fig = _new_dimension_grid_fig()

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col = col_idx + 1
        print(f"  Loading Gini FI [{dim_info['label']}]…", flush=True)

        ws_fi, ws_feats = _load_mean_gini_importance("ws",   dim_info)
        lo_fi, lo_feats = _load_mean_gini_importance("loso", dim_info)
        if len(ws_fi) == 0 or len(lo_fi) == 0:
            continue

        x_idx, y_idx, common = _common_feature_indices(ws_feats, lo_feats)
        if not common:
            continue

        # Scale over each pipeline's full feature space so that 1.0 = top within
        # that pipeline (not just within the common intersection).
        ws_fi_sc = _minmax_scale(ws_fi)
        lo_fi_sc = _minmax_scale(lo_fi)
        x_common = np.array([ws_fi_sc[x_idx[f]] for f in common])
        y_common = np.array([lo_fi_sc[y_idx[f]] for f in common])

        top_idx    = _select_top_union(x_common, y_common, SHAP_SCATTER_TOP_N)
        feat_names = [common[i] for i in top_idx]
        x_vals     = x_common[top_idx]
        y_vals     = y_common[top_idx]

        _add_scatter_panel(
            fig, col, dim_info, x_vals, y_vals, feat_names,
            x_label="WS Gini importance (min-max)",
            y_label="LOSO Gini importance (min-max)",
        )

    fig.update_layout(
        title=dict(
            text=(f"LOSO vs WS — RF Feature Importance "
                  f"(Gini, top-{SHAP_SCATTER_TOP_N} ∪ top-{SHAP_SCATTER_TOP_N}, min-max scaled)"),
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
        top_idx    = _select_top_union(ws_abs_c, lo_abs_c, SHAP_SCATTER_TOP_N)
        feat_names = [common[i] for i in top_idx]

        ws_abs_scaled    = ws_abs_c[top_idx]
        lo_abs_scaled    = lo_abs_c[top_idx]
        ws_signed_scaled = ws_signed_c[top_idx]
        lo_signed_scaled = lo_signed_c[top_idx]

        _add_scatter_panel(
            fig_abs, col, dim_info, ws_abs_scaled, lo_abs_scaled, feat_names,
            x_label="WS mean |SHAP| (min-max)",
            y_label="LOSO mean |SHAP| (min-max)",
        )
        _add_scatter_panel(
            fig_dir, col, dim_info, ws_signed_scaled, lo_signed_scaled, feat_names,
            x_label="WS mean SHAP, signed (symmetric)",
            y_label="LOSO mean SHAP, signed (symmetric)",
            axis_range=[-1.05, 1.05],
        )

    for fig, title in (
        (fig_abs, (f"LOSO vs WS — Mean |SHAP| "
                   f"(top-{SHAP_SCATTER_TOP_N} ∪ top-{SHAP_SCATTER_TOP_N}, min-max scaled)")),
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
            y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=col,
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


def _load_probe_predictions_ws(dim_info: dict) -> pd.DataFrame:
    """
    Load WS consolidated sample predictions for a dimension.

    Returns DataFrame with columns [subject, task, probe_number, ws_proba, y_true].
    """
    base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
    path = base / "rf_loso_consolidated_sample_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
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
    print("Group-level comparison: WS vs LOSO")
    print("=" * 60)
    _queue(plot_group_comparison(), "group_comparison")

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
    _queue(_build_gini_scatter_fig(), "scatter_feature_importance")

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
