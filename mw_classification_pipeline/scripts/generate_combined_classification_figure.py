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

import os
import sys
import warnings
import pickle
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

RESULTS_ROOT = Path("mw_classification_pipeline/results/MW_Classification")
OUTPUT_DIR   = Path("mw_classification_pipeline/results/combined_figures")

# Per-dimension color, consistent across every figure
DIMENSIONS = [
    {
        "key":      "on_off",
        "ws_dir":   "on_vs_off_within_median",
        "loso_dir": "ON_vs_OFF_within_median",
        "label":    "On/Off-Task",
        "color":    "#DE237B",   # pink
    },
    {
        "key":      "valence",
        "ws_dir":   "valence_within_median",
        "loso_dir": "valence_within_median",
        "label":    "Valence",
        "color":    "#7B4FBA",   # purple
    },
    {
        "key":      "selfother",
        "ws_dir":   "selfother_within_median",
        "loso_dir": "selfother_within_median",
        "label":    "Self/Other",
        "color":    "#E67E22",   # orange
    },
    {
        "key":      "confidence",
        "ws_dir":   "confidence_within_median",
        "loso_dir": "confidence_within_median",
        "label":    "Confidence",
        "color":    "#27AE60",   # green
    },
    {
        "key":      "time",
        "ws_dir":   "time_within_median",
        "loso_dir": "time_within_median",
        "label":    "Time",
        "color":    "#2980B9",   # blue
    },
]

PERM_COLOR   = "#C8C8C8"
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
        marker_color="#DE237B",
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

    # --- Save ---
    stem_path = OUTPUT_DIR / stem
    fig.write_html(str(stem_path.with_suffix(".html")))
    fig.write_image(str(stem_path.with_suffix(".png")), width=total_width, height=total_height, scale=2)
    fig.write_image(str(stem_path.with_suffix(".svg")), width=total_width, height=total_height, scale=2)
    return True


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
# SHAP loading helpers
# =============================================================================


def _load_mean_shap(
    pipeline: str, dim_info: dict
) -> tuple[np.ndarray, list[str]]:
    """
    Return (gini_importance, feature_names) averaged across all runs.

    Uses RF Gini feature importance (not SHAP) because LOSO SHAP values are
    not reliably saved by the current pipeline. Gini is available for both
    WS (per-run CSVs) and LOSO (aggregate CSV) and is directly comparable.

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


# =============================================================================
# Figure F — SHAP scatter: LOSO vs WS (absolute and directional)
# =============================================================================


def _build_shap_scatter_fig() -> go.Figure:
    """
    One panel per dimension: scatter of per-feature RF Gini importance (LOSO vs WS).

    Uses Gini importance (not SHAP) because LOSO SHAP values are not reliably
    saved by the current pipeline. Top-20 features selected by WS importance.
    """
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

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col   = col_idx + 1
        color = dim_info["color"]
        print(f"  Loading FI [{dim_info['label']}]…", flush=True)

        ws_fi, ws_feats = _load_mean_shap("ws",   dim_info)
        lo_fi, lo_feats = _load_mean_shap("loso", dim_info)

        if len(ws_fi) == 0 or len(lo_fi) == 0:
            continue

        # Align on common features
        ws_feat_idx = {f: i for i, f in enumerate(ws_feats)}
        lo_feat_idx = {f: i for i, f in enumerate(lo_feats)}
        common = [f for f in ws_feats if f in lo_feat_idx]

        if not common:
            continue

        x_common = np.array([ws_fi[ws_feat_idx[f]] for f in common])
        y_common = np.array([lo_fi[lo_feat_idx[f]] for f in common])

        # Keep top-20 by WS Gini importance
        top_k   = 20
        top_idx = np.argsort(x_common)[-top_k:]
        x_vals  = x_common[top_idx]
        y_vals  = y_common[top_idx]
        top_feats = [common[i] for i in top_idx]

        # Scatter
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers+text",
            marker=dict(color=color, size=7, opacity=0.75,
                        line=dict(color="white", width=0.5)),
            text=[f.split("_")[0] for f in top_feats],
            textposition="top center",
            textfont=dict(size=7, family="Times New Roman"),
            customdata=top_feats,
            hovertemplate="%{customdata}<br>WS: %{x:.4f}<br>LOSO: %{y:.4f}<extra></extra>",
            showlegend=False,
            name=dim_info["label"],
        ), row=1, col=col)

        # Axis refs
        x_ref = f"x{col}" if col > 1 else "x"
        y_ref = f"y{col}" if col > 1 else "y"

        # Axis range: floor at 0, 5% padding above max
        pad = 0.05
        xr = [0, x_vals.max() * (1 + pad)]
        yr = [0, max(y_vals.max(), x_vals.max()) * (1 + pad)]
        v_range = [0, max(xr[1], yr[1])]

        # Diagonal y=x
        fig.add_shape(
            type="line", x0=0, x1=v_range[1], y0=0, y1=v_range[1],
            xref=x_ref, yref=y_ref,
            line=dict(color="#AAAAAA", dash="dot", width=1.2),
        )

        # Regression line
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_reg = np.array([0, xr[1]])
        fig.add_trace(go.Scatter(
            x=x_reg, y=m * x_reg + b,
            mode="lines",
            line=dict(color=color, width=2.0),
            showlegend=False,
        ), row=1, col=col)

        # Correlation
        r = float(np.corrcoef(x_vals, y_vals)[0, 1])
        fig.add_annotation(
            text=f"r = {r:.2f}, top {top_k}",
            xref=f"{x_ref} domain" if col > 1 else "x domain",
            yref=f"{y_ref} domain" if col > 1 else "y domain",
            x=0.05, y=0.97,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=10, color=color, family="Times New Roman"),
        )

        # Axes
        x_ax = f"xaxis{col}" if col > 1 else "xaxis"
        y_ax = f"yaxis{col}" if col > 1 else "yaxis"
        fig.layout[x_ax].range     = xr
        fig.layout[y_ax].range     = yr
        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].showgrid  = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline  = False
            fig.layout[ax_name].tickfont  = dict(size=9)
        fig.layout[x_ax].title = dict(text="WS Gini importance", font=dict(size=11))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO Gini importance", font=dict(size=11))

    fig.update_layout(
        title=dict(
            text="LOSO vs WS — RF Feature Importance (top 20 by WS, Gini)",
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
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("ws",   "Within-Subject", "group_ws",   "individual_ws"),
        ("loso", "LOSO",           "group_loso",  "individual_loso"),
    ]

    def _save_plotly(fig: go.Figure, stem: str) -> None:
        total_width  = fig.layout.width  or 800
        total_height = fig.layout.height or 600
        for fmt in ("html", "png", "svg"):
            p = OUTPUT_DIR / f"{stem}.{fmt}"
            if fmt == "html":
                fig.write_html(str(p))
            else:
                fig.write_image(str(p), width=total_width, height=total_height, scale=2)
            print(f"  Saved: {p}")

    for pipeline, title_prefix, group_stem, indiv_stem in tasks:
        # Group-level (now Plotly)
        print("=" * 60)
        print(f"Group-level: {title_prefix}")
        print("=" * 60)
        fig_g = plot_group_level(pipeline, title_prefix)
        _save_plotly(fig_g, group_stem)

        # Individual-level (Plotly — saves internally)
        print("=" * 60)
        print(f"Individual-level: {title_prefix}")
        print("=" * 60)
        saved = plot_individual(pipeline, title_prefix, indiv_stem)
        if saved:
            for fmt in ("png", "svg", "html"):
                print(f"  Saved: {OUTPUT_DIR / f'{indiv_stem}.{fmt}'}")

    # Combined group-level comparison (now Plotly)
    print("=" * 60)
    print("Group-level comparison: WS vs LOSO")
    print("=" * 60)
    fig_c = plot_group_comparison()
    _save_plotly(fig_c, "group_comparison")

    # LOSO vs WS scatter
    print("=" * 60)
    print("LOSO vs WS scatter")
    print("=" * 60)
    fig_s = plot_loso_vs_ws_scatter()
    _save_plotly(fig_s, "scatter_loso_vs_ws")

    # Regression overlay
    print("=" * 60)
    print("Regression overlay")
    print("=" * 60)
    fig_r = plot_regression_overlay()
    _save_plotly(fig_r, "scatter_regression_overlay")

    # Feature importance scatter: WS vs LOSO (Gini)
    print("=" * 60)
    print("Feature importance scatter — Gini (WS vs LOSO)")
    print("=" * 60)
    fig_fi = _build_shap_scatter_fig()
    _save_plotly(fig_fi, "scatter_feature_importance")

    print("\nDone. Figures in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
