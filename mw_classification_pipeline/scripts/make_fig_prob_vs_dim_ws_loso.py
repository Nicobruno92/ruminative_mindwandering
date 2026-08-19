#!/usr/bin/env python
"""
Combined WS vs LOSO scatter figures: predicted probability vs raw probe score.

Extends the WS-only ``prob_vs_{dim}_{general,faceted}.png`` pair produced by
``utils/plotting_utils.py::plot_probability_vs_raw`` (WithinSubject only, no
LOSO equivalent exists) into a Within-Subject | LOSO side-by-side comparison,
for both the probed dimension's own raw score and the confidence rating given
alongside every probe. Sits next to ``scatter_proba_ws_vs_loso.*`` in
``results/combined_figures/`` — same probe-level data, same DIMENSIONS/PALETTE,
same WS/LOSO loading pattern from ``generate_combined_classification_figure.py``
(reused here, not duplicated), so the two figures read as one family.

Outputs (in results/combined_figures/prob_vs_dim/):
    {dim_key}_prob_vs_raw_ws_loso_{general,faceted}.{png,svg}
        Per dimension: predicted probability vs that dimension's own raw score
        (0-100), WS in the left panel/column, LOSO in the right. Skipped for
        the confidence dimension (a raw-score-vs-confidence panel already
        exists per dimension below; plotting confidence against itself would
        be redundant).
    {dim_key}_prob_vs_confidence_ws_loso_{general,faceted}.{png,svg}
        Per dimension: predicted probability vs confidence rating (0-100),
        same WS | LOSO layout. Skipped for the confidence dimension itself
        (identical to the raw-score panel above).
    all_dims_prob_vs_raw_ws_loso_general.{png,svg}
        All 5 dimensions in one figure: row 1 = WS, row 2 = LOSO, one column
        per dimension. Pooled across subjects (general style only — a
        per-subject facet grid crossed with 5 dimensions and 2 pipelines
        would not fit on a page).
    all_dims_prob_vs_confidence_ws_loso_general.{png,svg}
        Same layout, x-axis = confidence rating, over the 4 non-confidence
        dimensions (confidence has no separate "vs confidence" panel).
    {dim_key}_prob_vs_confidence_mirrored_ws_loso_{general,faceted}.{png,svg}
    all_dims_prob_vs_confidence_mirrored_ws_loso_general.{png,svg}
        Same as the "vs confidence" figures above, but the y-axis is the
        probability mirrored to the true class (see _mirrored_proba) instead
        of raw class-1-anchored proba. Confidence is orthogonal to the
        target's own label, so a real, symmetric confidence effect (proba
        pulled toward 1 on y=1 trials, toward 0 on y=0 trials) has opposite
        sign on the raw axis and cancels out in one pooled regression;
        mirroring puts both on the same axis instead.
    {dim_key}_prob_vs_confidence_raw_vs_mirrored.{png,svg}
        Per dimension, both of the above stacked as one figure: panel A (top
        row) = raw proba vs confidence, panel B (bottom row) = mirrored proba
        vs confidence, each row itself WS | LOSO. Lets the two be read
        together instead of flipping between two separate files.

"general" pools all probes/subjects into one scatter with an OLS line and
beta/r/p/n annotated. "faceted" gives one row per subject (WS | LOSO columns),
sharing axes so panels are directly comparable.

Must run in the `plots` env (kaleido >= 1.0 batched writer, see
generate_combined_classification_figure.py's module docstring for why).

Usage (from project root):
    /path/to/miniforge3/envs/plots/bin/python \
        mw_classification_pipeline/scripts/make_fig_prob_vs_dim_ws_loso.py
"""

# =============================================================================
# Imports
# =============================================================================

import asyncio
import sys
import warnings
from pathlib import Path

import kaleido as _kaleido
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from generate_combined_classification_figure import (  # noqa: E402
    AUC_RANGE,
    CHANCE,
    DIMENSIONS,
    PROBE_DATA_PATH,
    RESULTS_ROOT,
    _export_fig_csv,
    _normalise_subject,
    load_subject_data,
)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path("mw_classification_pipeline/results/combined_figures/prob_vs_dim")

SCORE_RANGE = (-5, 105)     # raw score / confidence, both on the 0-100 MDES scale
PROBA_RANGE = (-0.05, 1.05)
MARKER_SIZE_GENERAL = 5
MARKER_SIZE_FACETED = 4
MARKER_OPACITY = 0.5
REGRESSION_LINE_COLOR = "black"

# The confidence rating shares one pole pair (low/high) with every dimension's
# "vs confidence" x-axis, regardless of which dimension is being probed —
# looked up once here rather than threading dim_info for the probed dimension
# through code paths that plot against confidence, not the probed dimension.
_CONFIDENCE_DIM = next(d for d in DIMENSIONS if d["label_source"] == "confidence")

# Font sizes (pt). Kept in step with the matching r/p annotations in
# generate_combined_classification_figure.py's scatter_loso_vs_ws /
# scatter_proba_ws_vs_loso / scatter_regression_overlay (also 18pt) so the
# whole "family" of correlation scatter figures reads as one style.
BASE_FONT_SIZE = 14           # figure-wide default (axis titles, tick labels)
PANEL_TITLE_FONT_SIZE = 18    # "Within-Subject"/"LOSO" column headers, per-dim titles
ANNOTATION_FONT_SIZE = 18     # r/p readout, general (pooled) panels
ANNOTATION_FONT_SIZE_SMALL = 14   # r/p readout, faceted (one row per subject) panels
FACETED_TICK_FONT_SIZE = 10
SUBJECT_LABEL_FONT_SIZE = 10  # per-row "sub-XX" label in faceted figures
PANEL_LETTER_FONT_SIZE = 22   # bold A/B panel letters

# =============================================================================
# Data loading
# =============================================================================


def _load_ws_probe_data(dim_info: dict) -> pd.DataFrame:
    """
    WS probe-level table for one dimension.

    Returns columns [subject, task, probe_number, proba, y_true, raw_score,
    confidence] — raw_score/confidence omitted if their source column is
    absent from this dimension's consolidated CSV.
    """
    base = RESULTS_ROOT / "WithinSubject" / dim_info["ws_dir"] / "all" / "rf"
    path = base / "rf_ws_consolidated_sample_predictions.csv"
    if not path.exists():
        print(f"  ! {path} not found — WS side will be empty for {dim_info['label']}.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    valid_subject = df["subject"].astype(str).str.strip().str.match(r"^\d+$")
    if (~valid_subject).any():
        bad = df.loc[~valid_subject, "subject"].unique().tolist()
        print(f"  ! {path.name}: dropping {(~valid_subject).sum()} row(s) with "
              f"non-numeric subject id {bad} (corrupt on-disk data, not a real subject).")
        df = df.loc[valid_subject].copy()
    df["subject"] = _normalise_subject(df["subject"])

    out = df[["subject", "task", "probe_number"]].copy()
    out["proba"] = df["proba_mean"]
    out["y_true"] = df["y_true_first"]

    # Built as independent column assignments, not a source-keyed rename dict:
    # for the confidence dimension, raw_col and "confidence_first" are the
    # SAME source column, and a rename dict keyed by source name can only
    # hold one destination per source — the second assignment would silently
    # clobber the first and leave raw_score missing.
    raw_col = f"{dim_info['label_source']}_first"
    if raw_col in df.columns:
        out["raw_score"] = df[raw_col]
    if "confidence_first" in df.columns:
        out["confidence"] = df["confidence_first"]

    return out


def _load_loso_probe_data(dim_info: dict) -> pd.DataFrame:
    """
    LOSO probe-level table for one dimension, aggregated across true_runs.

    Returns the same columns as :func:`_load_ws_probe_data`. Raw score and
    confidence are constant per probe across runs, so they are aggregated
    with "first" rather than "mean" (only y_proba varies run to run).
    """
    base = RESULTS_ROOT / "LOSO" / dim_info["loso_dir"] / "all" / "rf"
    run_csvs = sorted(base.glob("true_runs/run_*/*_sample_predictions.csv"))
    if not run_csvs:
        print(f"  ! No LOSO true_runs sample_predictions found for {dim_info['label']}.")
        return pd.DataFrame()

    all_runs = pd.concat([pd.read_csv(p) for p in run_csvs], ignore_index=True)
    all_runs["subject"] = _normalise_subject(all_runs["subject"])

    agg_kwargs = {"proba": ("y_proba", "mean"), "y_true": ("y_true", "first")}
    raw_col = dim_info["label_source"]
    if raw_col in all_runs.columns:
        agg_kwargs["raw_score"] = (raw_col, "first")
    if "confidence" in all_runs.columns:
        agg_kwargs["confidence"] = ("confidence", "first")

    return (
        all_runs.groupby(["subject", "task", "probe_number"])
        .agg(**agg_kwargs)
        .reset_index()
    )


# =============================================================================
# Shared scatter-panel drawing
# =============================================================================


def _format_p(p_value: float) -> str:
    """
    APA-style p-value: decimal (never scientific notation), no leading zero.

    Below the 3-decimal floor (.001), the exact value is noise to a reader —
    report the conventional "<.001" instead of e.g. "2.4e-30".
    """
    if p_value < 0.001:
        return "p<.001"
    return f"p={p_value:.3f}".replace("=0.", "=.")


def _add_scatter_panel(
    fig: go.Figure,
    row: int,
    col: int,
    x: pd.Series,
    y: pd.Series,
    color: str,
    marker_size: int,
    annotate: bool,
) -> None:
    """Scatter + OLS line + beta/r/p/n annotation, drawn into one subplot cell."""
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    if len(x_arr) < 2:
        return

    fig.add_trace(
        go.Scatter(
            x=x_arr, y=y_arr, mode="markers",
            marker=dict(color=color, size=marker_size, opacity=MARKER_OPACITY, line=dict(width=0)),
            showlegend=False,
        ),
        row=row, col=col,
    )

    if len(x_arr) < 3 or np.std(x_arr) == 0:
        return

    slope, intercept, r_value, p_value, _ = stats.linregress(x_arr, y_arr)
    x_line = np.array([x_arr.min(), x_arr.max()])
    fig.add_trace(
        go.Scatter(
            x=x_line, y=slope * x_line + intercept, mode="lines",
            line=dict(color=REGRESSION_LINE_COLOR, width=1.5),
            showlegend=False,
        ),
        row=row, col=col,
    )

    if annotate:
        fig.add_annotation(
            text=f"r={r_value:.2f}<br>{_format_p(p_value)}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=ANNOTATION_FONT_SIZE, family="Times New Roman", color=color),
            row=row, col=col,
        )
    else:
        fig.add_annotation(
            text=f"r={r_value:.2f}, {_format_p(p_value)}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=ANNOTATION_FONT_SIZE_SMALL, family="Times New Roman", color=color),
            row=row, col=col,
        )


def _add_subject_scatter_panel(
    fig: go.Figure, row: int, col: int, x: pd.Series, y: pd.Series, color: str,
) -> None:
    """
    Subject-level scatter + OLS line + r/p annotation, one point per subject,
    each labeled with its subject number.

    Deliberately not :func:`_add_scatter_panel`: at ~20-30 points per panel a
    colored regression line and a "sub-N" label per point are legible and
    useful (subject-level outliers are worth being able to name at a glance);
    at the probe-level panels' point-cloud density (~1000+ points) both would
    be unreadable, which is why those keep the plain black line and no labels.
    """
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    labels = [str(int(s)) if str(s).isdigit() else str(s) for s in x.index[mask]]
    if len(x_arr) < 2:
        return

    fig.add_trace(
        go.Scatter(
            x=x_arr, y=y_arr, mode="markers+text",
            text=labels, textposition="top center",
            textfont=dict(size=10, family="Times New Roman"),
            marker=dict(color=color, size=8, opacity=0.85, line=dict(color="white", width=0.5)),
            showlegend=False,
        ),
        row=row, col=col,
    )

    if len(x_arr) < 3 or np.std(x_arr) == 0:
        return

    slope, intercept, r_value, p_value, _ = stats.linregress(x_arr, y_arr)
    x_line = np.array([x_arr.min(), x_arr.max()])
    fig.add_trace(
        go.Scatter(
            x=x_line, y=slope * x_line + intercept, mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.add_annotation(
        text=f"r={r_value:.2f}<br>{_format_p(p_value)}",
        xref="x domain", yref="y domain",
        x=0.05, y=0.95, xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=ANNOTATION_FONT_SIZE, family="Times New Roman", color=color),
        row=row, col=col,
    )


def _mirrored_proba(df: pd.DataFrame) -> pd.Series:
    """
    Predicted probability flipped around 0.5 on y_true==0 trials — i.e. the
    probability the model assigned to the TRUE class, regardless of whether
    that class was coded 1 or 0.

    Confidence is a probe dimension orthogonal to the target dimension's own
    label (a high-confidence trial can equally be y=1 or y=0). Raw proba is
    anchored to class 1, so if confidence genuinely relates to how decisively
    the model classifies a trial, it pushes proba toward 1 on y=1 trials and
    toward 0 on y=0 trials — two effects with opposite sign on the raw proba
    axis that cancel out in one pooled regression against confidence. Mirroring
    puts both on the same axis: higher mirrored proba always means "more
    confident and correct", so a real symmetric effect shows up as a single
    non-zero slope instead of washing out to ~0.
    """
    return df["proba"].where(df["y_true"] == 1, 1 - df["proba"])


# =============================================================================
# Per-dimension figures — general (pooled, WS | LOSO)
# =============================================================================


def build_general_fig(dim_info: dict, value_type: str, mirror: bool = False) -> go.Figure | None:
    """
    One dimension, one panel per pipeline: predicted probability vs raw_score
    (value_type="raw") or vs confidence (value_type="confidence"), WS left /
    LOSO right, sharing axes. mirror=True plots :func:`_mirrored_proba`
    instead of raw proba on the y-axis (see its docstring) — only meaningful
    for value_type="confidence", since the dimension's own raw_score is
    already monotonically tied to y_true by construction (median split).
    """
    label = dim_info["label"]
    color = dim_info["color"]
    y_col = "raw_score" if value_type == "raw" else "confidence"
    # Pole wording (CLAUDE.md "Dimension Labels & Pole Wording"), not a
    # generic "Raw score (0-100)" / "Confidence rating (0-100)" — the axis
    # endpoints ARE off-task/on-task, negative/positive, etc., and the
    # generic label doesn't say what "0" and "100" mean.
    pole_dim = dim_info if value_type == "raw" else _CONFIDENCE_DIM
    x_title = f"{pole_dim['pole_low']} → {pole_dim['pole_high']}"

    ws_df = _load_ws_probe_data(dim_info)
    loso_df = _load_loso_probe_data(dim_info)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Within-Subject", "LOSO"],
        horizontal_spacing=0.05,
    )
    any_data = False
    for col, df in [(1, ws_df), (2, loso_df)]:
        if df.empty or y_col not in df.columns:
            continue
        any_data = True
        y_vals = _mirrored_proba(df) if mirror else df["proba"]
        _add_scatter_panel(fig, 1, col, df[y_col], y_vals, color, MARKER_SIZE_GENERAL, annotate=True)

    if not any_data:
        print(f"  Skipping {label} / {value_type} (general): no data on either side.")
        return None

    y_title = "Predicted probability of true class (mirrored)" if mirror else "Predicted probability"
    for col in (1, 2):
        fig.update_xaxes(title_text=x_title, range=list(SCORE_RANGE), row=1, col=col)
        fig.update_yaxes(range=list(PROBA_RANGE), row=1, col=col)
    fig.update_yaxes(title_text=y_title, row=1, col=1)

    for ann in fig.layout.annotations[:2]:
        ann.font.family = "Times New Roman"
        ann.font.size = PANEL_TITLE_FONT_SIZE

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=BASE_FONT_SIZE),
        width=700, height=360,
        margin=dict(t=25, b=45, l=60, r=15),
    )
    return fig


# =============================================================================
# Per-dimension figures — faceted (one row per subject, WS | LOSO)
# =============================================================================


def build_faceted_fig(
    dim_info: dict, value_type: str, mirror: bool = False,
) -> tuple[go.Figure, int, int] | None:
    """
    One dimension, one row per subject: predicted probability vs raw_score
    or confidence, WS in column 1 / LOSO in column 2, sharing axes across
    every panel so subjects are directly comparable. mirror=True plots
    :func:`_mirrored_proba` instead of raw proba — see build_general_fig.
    """
    label = dim_info["label"]
    color = dim_info["color"]
    y_col = "raw_score" if value_type == "raw" else "confidence"
    # Pole wording, same rule as build_general_fig — see its comment.
    pole_dim = dim_info if value_type == "raw" else _CONFIDENCE_DIM
    x_title = f"{pole_dim['pole_low']} → {pole_dim['pole_high']}"

    ws_df = _load_ws_probe_data(dim_info)
    loso_df = _load_loso_probe_data(dim_info)
    if (ws_df.empty or y_col not in ws_df.columns) and (loso_df.empty or y_col not in loso_df.columns):
        print(f"  Skipping {label} / {value_type} (faceted): no data on either side.")
        return None

    ws_subjects = set(ws_df["subject"]) if not ws_df.empty else set()
    loso_subjects = set(loso_df["subject"]) if not loso_df.empty else set()
    subjects = sorted(ws_subjects | loso_subjects, key=lambda s: int(s) if s.isdigit() else s)
    n = len(subjects)
    if n == 0:
        return None

    subplot_titles = ["Within-Subject", "LOSO"] + [""] * (2 * (n - 1))
    fig = make_subplots(
        rows=n, cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=min(0.3 / n, 0.015),
    )

    for i, subj in enumerate(subjects):
        row = i + 1
        for col, df in [(1, ws_df), (2, loso_df)]:
            if df.empty or y_col not in df.columns:
                continue
            sub_df = df[df["subject"] == subj]
            if sub_df.empty:
                continue
            y_vals = _mirrored_proba(sub_df) if mirror else sub_df["proba"]
            _add_scatter_panel(
                fig, row, col, sub_df[y_col], y_vals, color,
                MARKER_SIZE_FACETED, annotate=False,
            )
            fig.update_xaxes(range=list(SCORE_RANGE), row=row, col=col,
                              showticklabels=(row == n), tickfont=dict(size=FACETED_TICK_FONT_SIZE))
            fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=col,
                              showticklabels=(col == 1), tickfont=dict(size=FACETED_TICK_FONT_SIZE))
        fig.add_annotation(
            text=f"<b>sub-{subj}</b>",
            xref="paper", yref="y domain", row=row, col=1,
            x=-0.02, y=0.5, xanchor="right", yanchor="middle",
            showarrow=False, font=dict(size=SUBJECT_LABEL_FONT_SIZE, family="Times New Roman"),
        )

    y_title = "Predicted probability of true class (mirrored)" if mirror else "Predicted probability"
    fig.update_xaxes(title_text=x_title, row=n, col=1)
    fig.update_xaxes(title_text=x_title, row=n, col=2)
    fig.update_yaxes(title_text=y_title, row=(n + 1) // 2, col=1)

    for ann in fig.layout.annotations[:2]:
        ann.font.family = "Times New Roman"
        ann.font.size = PANEL_TITLE_FONT_SIZE

    width, height = 580, max(240, n * 115 + 45)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=BASE_FONT_SIZE),
        width=width, height=height,
        margin=dict(t=25, b=40, l=60, r=15),
    )
    return fig, width, height


# =============================================================================
# Per-dimension figures — raw vs mirrored comparison (panel A / panel B)
# =============================================================================


def build_mirror_comparison_fig(dim_info: dict) -> go.Figure | None:
    """
    One dimension, "vs confidence" only: panel A (top row) is the raw
    class-1-anchored probability, panel B (bottom row) is the same data with
    proba mirrored to the true class (see :func:`_mirrored_proba`). Each
    panel is itself WS | LOSO side by side, i.e. this stacks
    build_general_fig(..., "confidence", mirror=False) over
    build_general_fig(..., "confidence", mirror=True) as one figure so the
    two are read together rather than as two separate files.

    Not meaningful for value_type="raw" — the dimension's own raw score is
    already monotonically tied to y_true by construction, so there is no
    unmirrored/mirrored pair to compare there.
    """
    label = dim_info["label"]
    color = dim_info["color"]

    ws_df = _load_ws_probe_data(dim_info)
    loso_df = _load_loso_probe_data(dim_info)
    if (ws_df.empty or "confidence" not in ws_df.columns) and \
       (loso_df.empty or "confidence" not in loso_df.columns):
        print(f"  Skipping {label} mirror comparison: no data on either side.")
        return None

    subplot_titles = ["Within-Subject", "LOSO", "", ""]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.13,
    )

    any_data = False
    for row, mirror in [(1, False), (2, True)]:
        for col, df in [(1, ws_df), (2, loso_df)]:
            if df.empty or "confidence" not in df.columns:
                continue
            any_data = True
            y_vals = _mirrored_proba(df) if mirror else df["proba"]
            _add_scatter_panel(fig, row, col, df["confidence"], y_vals, color, MARKER_SIZE_GENERAL, annotate=True)
        conf_x_title = f"{_CONFIDENCE_DIM['pole_low']} → {_CONFIDENCE_DIM['pole_high']}"
        fig.update_xaxes(title_text=conf_x_title, range=list(SCORE_RANGE), row=row, col=1)
        fig.update_xaxes(title_text=conf_x_title, range=list(SCORE_RANGE), row=row, col=2)
        fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=1)
        fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=2)

    if not any_data:
        print(f"  Skipping {label} mirror comparison: no data.")
        return None

    fig.update_yaxes(title_text="Predicted probability", row=1, col=1)
    fig.update_yaxes(title_text="P(true class), mirrored", row=2, col=1)

    for ann in fig.layout.annotations[:2]:
        ann.font.family = "Times New Roman"
        ann.font.size = PANEL_TITLE_FONT_SIZE

    # Panel letters — bold, top-left corner of each row's first (WS) panel,
    # clearly outside the plot area so they read as structure rather than a
    # prefix on the "Within-Subject"/"LOSO" column headers above row 1.
    for row, letter in [(1, "A"), (2, "B")]:
        fig.add_annotation(
            text=f"<b>{letter}</b>",
            xref="x domain", yref="y domain", row=row, col=1,
            x=-0.14, y=1.20, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=PANEL_LETTER_FONT_SIZE, family="Times New Roman", color="#333333"),
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=BASE_FONT_SIZE),
        width=680, height=640,
        margin=dict(t=50, b=45, l=60, r=15),
    )
    return fig


# =============================================================================
# All-dimensions combined figure — general only (2 rows x N cols)
# =============================================================================


def build_all_dims_combined_fig(value_type: str, mirror: bool = False) -> go.Figure | None:
    """
    All dimensions pooled into one figure: row 1 = WS, row 2 = LOSO, one
    column per dimension. General/pooled style only. mirror=True plots
    :func:`_mirrored_proba` instead of raw proba — see build_general_fig.
    """
    dims = DIMENSIONS if value_type == "raw" else [
        d for d in DIMENSIONS if d["label_source"] != "confidence"
    ]
    y_col = "raw_score" if value_type == "raw" else "confidence"
    n = len(dims)

    subplot_titles = [d["label"] for d in dims] + [""] * n
    fig = make_subplots(
        rows=2, cols=n,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.025,
    )

    any_data = False
    for i, dim_info in enumerate(dims):
        col = i + 1
        color = dim_info["color"]
        ws_df = _load_ws_probe_data(dim_info)
        loso_df = _load_loso_probe_data(dim_info)
        for row, df in [(1, ws_df), (2, loso_df)]:
            if df.empty or y_col not in df.columns:
                continue
            any_data = True
            y_vals = _mirrored_proba(df) if mirror else df["proba"]
            _add_scatter_panel(fig, row, col, df[y_col], y_vals, color, MARKER_SIZE_GENERAL, annotate=False)
            fig.update_xaxes(range=list(SCORE_RANGE), row=row, col=col,
                              showticklabels=(row == 2))
            fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=col,
                              showticklabels=(col == 1))
        fig.layout.annotations[i].font.color = color
        fig.layout.annotations[i].font.size = PANEL_TITLE_FONT_SIZE
        fig.layout.annotations[i].font.family = "Times New Roman"

    if not any_data:
        print(f"  Skipping all-dims combined ({value_type}): no data.")
        return None

    # Short form here (vs. the full "...of true class (mirrored)" used in the
    # single-dimension figures): with two rows this close together, the long
    # title's rotated text is taller than each row's cell and the row-1/row-2
    # labels visually overlap at the seam.
    y_title = "P(true class)" if mirror else "Predicted probability"
    # Pole wording (CLAUDE.md "Dimension Labels & Pole Wording"), not a
    # generic "Raw score (0-100)". For value_type="raw" every column is a
    # DIFFERENT dimension with its own pole pair, so the title is set per
    # column; for "confidence" every column shares the same x-axis (the
    # confidence rating), so one shared title is correct.
    if value_type == "raw":
        for i, dim_info in enumerate(dims):
            fig.update_xaxes(
                title_text=f"{dim_info['pole_low']} → {dim_info['pole_high']}",
                row=2, col=i + 1,
            )
    else:
        fig.update_xaxes(
            title_text=f"{_CONFIDENCE_DIM['pole_low']} → {_CONFIDENCE_DIM['pole_high']}",
            row=2,
        )

    # The row-group label ("Within-Subject" / "LOSO") must read as the
    # OUTERMOST element (label, then axis title, then plot) — but a
    # paper-positioned annotation can't be trusted to land there: Plotly's
    # add_annotation(row=, col=) silently discards an explicit xref="paper"
    # and substitutes the subplot's own x DATA axis whenever yref contains
    # "domain" (see plotly basedatatypes.py _add_annotation_like), and even
    # without row/col, a rotated "paper"-xref annotation's rendered position
    # did not match its nominal paper fraction under kaleido here — gluing
    # the label to the data or landing deep inside the plot either way.
    # Folding the row label into the y-axis title text itself sidesteps the
    # coordinate system entirely: Plotly's own automargin places axis titles
    # correctly by construction, so this is the only placement that is
    # actually reliable.
    for row, pipeline_label in [(1, "Within-Subject"), (2, "LOSO")]:
        fig.update_yaxes(
            title_text=f"<b>{pipeline_label}</b><br>{y_title}",
            title_font=dict(size=PANEL_TITLE_FONT_SIZE),
            row=row, col=1,
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=BASE_FONT_SIZE),
        width=290 * n, height=490,
        margin=dict(t=30, b=58, l=150, r=15),
    )
    return fig


# =============================================================================
# Performance + probabilities by confidence — combined (4 rows x N cols)
# =============================================================================


def build_performance_and_prob_by_confidence_combined() -> go.Figure:
    """
    Union of two "does self-reported confidence track decodability?" families
    into one figure, built as a single ``make_subplots`` grid rather than two
    independently-styled figures pasted together — same font, same p-value
    format, same pole wording, same tick convention throughout, so the two
    halves read as one figure, not two.

    Panel A (rows 1-2, "Performance by Confidence") — one point per subject:
    x = that subject's mean 'confidence' rating across every probe, y = mean
    AUC for that dimension/pipeline. Same data as
    ``generate_combined_classification_figure.py::plot_auc_vs_confidence_scatter``.
    All 5 dimensions.

    Panel B (rows 3-4, "Probabilities by Confidence") — one point per probe:
    x = confidence rating on that probe, y = predicted probability mirrored
    to the true class (see :func:`_mirrored_proba`). Same data as
    ``build_all_dims_combined_fig("confidence", mirror=True)``. 4 dimensions
    — confidence has no "vs itself" panel, so that cell is left blank and
    marked rather than silently empty.

    EXPLORATORY, same as both source figures: no correction across the 18
    populated panels here.
    """
    n = len(DIMENSIONS)
    dims_b = [d for d in DIMENSIONS if d["label_source"] != "confidence"]

    subplot_titles = [d["label"] for d in DIMENSIONS] + [""] * (3 * n)
    fig = make_subplots(
        rows=4, cols=n,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].text = (
            f"{dim_info['label']}<br><sub>{dim_info['pole_low']} ↔ {dim_info['pole_high']}</sub>"
        )
        fig.layout.annotations[i].font.color = dim_info["color"]
        fig.layout.annotations[i].font.size = PANEL_TITLE_FONT_SIZE
        fig.layout.annotations[i].font.family = "Times New Roman"

    # Reserve the row2/row3 gap (between the "A" and "B" groups) up front by
    # giving it more of the vertical budget than the row1/row2 and row3/row4
    # gaps (within-group) get — the panel-B letter + "Probabilities by
    # Confidence" title sit there, on top of row 2's own x-axis title/ticks
    # immediately above it. Growing this via margin or a bigger header
    # y-offset afterward doesn't work (margin pads the outer canvas, it
    # doesn't add space between elements placed inside the plot area) — see
    # scientific-plots skill's paper-figures.md. Total gap budget matches the
    # uniform-grid default (2*0.05 + 0.05 = 0.15) so overall figure height is
    # unchanged; only its distribution across the 3 gaps changes.
    gap_within, gap_between = 0.03, 0.09
    row_h = (1.0 - 2 * gap_within - gap_between) / 4
    row_tops = [1.0]
    for gap in (gap_within, gap_between, gap_within):
        row_tops.append(row_tops[-1] - row_h - gap)
    row_domains = {
        r: (max(0.0, min(1.0, row_tops[r - 1] - row_h)), max(0.0, min(1.0, row_tops[r - 1])))
        for r in (1, 2, 3, 4)
    }
    for row in (1, 2, 3, 4):
        for col in range(1, n + 1):
            fig.update_yaxes(domain=list(row_domains[row]), row=row, col=col)

    conf_x_title = f"{_CONFIDENCE_DIM['pole_low']} → {_CONFIDENCE_DIM['pole_high']}"

    # ---- Rows 1-2 (Panel A): subject-level AUC vs mean confidence ----
    probe_df = pd.read_csv(PROBE_DATA_PATH)
    probe_df["subject"] = probe_df["subject"].astype(str).str.zfill(2)
    subj_confidence = probe_df.groupby("subject")["confidence"].mean().rename("confidence")
    x_min_a = float(subj_confidence.min()) - 3
    x_max_a = float(subj_confidence.max()) + 3
    auc_lo, auc_hi = AUC_RANGE
    y_bottom_a, y_top_a = auc_lo, 1.16  # headroom above 1.0 for the r/p annotation

    for row, pipeline, pipeline_label in [(1, "ws", "WS"), (2, "loso", "LOSO")]:
        for col_idx, dim_info in enumerate(DIMENSIONS):
            col, color = col_idx + 1, dim_info["color"]
            idx = (row - 1) * n + col
            xref = "x" if idx == 1 else f"x{idx}"
            yref = "y" if idx == 1 else f"y{idx}"

            t_df, _ = load_subject_data(pipeline, dim_info)
            if t_df.empty:
                continue
            auc_means = t_df.groupby("subject")["auc"].mean().rename("auc")
            auc_means.index = auc_means.index.astype(str)
            merged = pd.concat([subj_confidence, auc_means], axis=1).dropna()
            if merged.empty:
                continue

            fig.add_shape(
                type="line", x0=x_min_a, x1=x_max_a, y0=CHANCE, y1=CHANCE,
                xref=xref, yref=yref,
                line=dict(color="#BBBBBB", dash="dash", width=1.0),
            )
            _add_subject_scatter_panel(fig, row, col, merged["confidence"], merged["auc"], color)

            fig.update_xaxes(range=[x_min_a, x_max_a], row=row, col=col,
                              showticklabels=(row == 2), tickfont=dict(size=12))
            fig.update_yaxes(range=[y_bottom_a, y_top_a], row=row, col=col,
                              tickmode="array", tickvals=[0.5, 0.75, 1.0],
                              ticktext=["0.5", ".75", "1.0"], tickfont=dict(size=12),
                              showticklabels=(col == 1))
        fig.update_yaxes(
            title_text=f"<b>{pipeline_label}</b><br>AUC",
            title_font=dict(size=PANEL_TITLE_FONT_SIZE), row=row, col=1,
        )
    fig.update_xaxes(title_text=conf_x_title, row=2)

    # ---- Rows 3-4 (Panel B): probe-level mirrored proba vs confidence ----
    for row, pipeline_label, loader in [
        (3, "WS", _load_ws_probe_data), (4, "LOSO", _load_loso_probe_data),
    ]:
        for col_idx, dim_info in enumerate(dims_b):
            col, color = col_idx + 1, dim_info["color"]
            df = loader(dim_info)
            if df.empty or "confidence" not in df.columns:
                continue
            _add_scatter_panel(
                fig, row, col, df["confidence"], _mirrored_proba(df), color,
                MARKER_SIZE_GENERAL, annotate=True,
            )
            fig.update_xaxes(range=list(SCORE_RANGE), row=row, col=col,
                              showticklabels=(row == 4))
            fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=col,
                              showticklabels=(col == 1))
        fig.update_yaxes(
            title_text=f"<b>{pipeline_label}</b><br>P(true class)",
            title_font=dict(size=PANEL_TITLE_FONT_SIZE), row=row, col=1,
        )
    for col in range(1, len(dims_b) + 1):
        fig.update_xaxes(title_text=conf_x_title, row=4, col=col)

    # Confidence has no "vs itself" panel in Panel B (see docstring) — mark
    # the cell instead of leaving an unexplained blank box.
    for row in (3, 4):
        fig.update_xaxes(visible=False, row=row, col=n)
        fig.update_yaxes(visible=False, row=row, col=n)
    # Positioned in "paper" coordinates from column 5's own (populated, known-
    # good) x-domain and row 3's y-domain, rather than "x15/y15 domain" — an
    # axis with visible=False and no trace on it does not reliably anchor an
    # annotation's domain-relative position.
    col_n_x0, col_n_x1 = fig.layout.xaxis5.domain
    row3_y0, row3_y1 = row_domains[3]
    fig.add_annotation(
        text="<i>not shown<br>(redundant with<br>confidence's own axis)</i>",
        xref="paper", yref="paper",
        x=(col_n_x0 + col_n_x1) / 2, y=(row3_y0 + row3_y1) / 2,
        xanchor="center", yanchor="middle", showarrow=False,
        font=dict(size=11, family="Times New Roman", color="#999999"),
    )

    # Panel letters + per-panel title — top-left corner of each group's first
    # row/column, per project convention (CLAUDE.md "Figure Assembly"): one
    # letter per distinct *kind* of panel, not one per column, since every
    # column within a row-pair is the same kind of panel repeated. Mirrors
    # generate_combined_classification_figure.py::plot_cv_stability_combined's
    # letter styling exactly, so this reads as the same figure family.
    #
    # Positioned with a small PIXEL yshift off a known-good anchor, not a
    # hand-picked fraction added to a "y" value — "paper" fractions span the
    # *whole* canvas including margins while axis "domain" fractions span
    # only the plot area, so adding a domain-scale offset (e.g. 0.05) to a
    # paper-scale y silently mismatches units. That was the bug: the row-1
    # header ended up shifted almost the entire top margin, reading as
    # "floating far above its panel" instead of sitting snug against it.
    FIG_WIDTH, FIG_HEIGHT = 300 * n, 1420
    MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 110, 20, 100, 60
    x1_left = fig.layout.xaxis.domain[0]
    x_right = fig.layout[f"xaxis{n}"].domain[1]
    x_center = (x1_left + x_right) / 2

    # Row 1's column headers (label + pole, two lines at PANEL_TITLE_FONT_SIZE
    # and smaller) already sit just above the row-1 panels at their own
    # (correct) "paper" y. The group title only needs to clear that block —
    # about 2 lines tall — plus a small gap.
    row1_anchor_y = fig.layout.annotations[0].y
    row1_yshift = 1.9 * PANEL_TITLE_FONT_SIZE + 14

    # Row 3 has no column headers repeated above it — just convert its own
    # domain top to the equivalent "paper" y (paper spans the full canvas,
    # domain only the plot area inside the margins) and clear it by a small
    # fixed gap.
    plot_area_frac = (FIG_HEIGHT - MARGIN_T - MARGIN_B) / FIG_HEIGHT
    row3_domain_top = row_domains[3][1]
    row3_anchor_y = MARGIN_B / FIG_HEIGHT + row3_domain_top * plot_area_frac
    row3_yshift = 10

    for letter, panel_title, anchor_y, yshift in [
        ("A", "Performance by Confidence", row1_anchor_y, row1_yshift),
        ("B", "Probabilities by Confidence", row3_anchor_y, row3_yshift),
    ]:
        # Structural letter — top-left corner, per project convention
        # (CLAUDE.md "Figure Assembly": letters read as structure, never
        # centered, even when a panel already carries its own title).
        fig.add_annotation(
            text=f"<b>{letter}</b>",
            xref="paper", yref="paper",
            x=x1_left - 0.06, y=anchor_y, yshift=yshift,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=30, family="Times New Roman", color="black"),
        )
        # Descriptive title — bold, large, centered over the group's full
        # column span, sitting snug just above its own panel/headers rather
        # than floating in the margin.
        fig.add_annotation(
            text=f"<b>{panel_title}</b>",
            xref="paper", yref="paper",
            x=x_center, y=anchor_y, yshift=yshift,
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=24, family="Times New Roman", color="black"),
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=BASE_FONT_SIZE),
        width=FIG_WIDTH, height=FIG_HEIGHT,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # Not wiped before regenerating (see generate_combined_classification_figure.py's
    # main() for why a blanket rmtree here is unsafe): each stem overwrites its
    # own file in place.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[go.Figure, Path, dict]] = []

    def _queue(fig: go.Figure | None, stem: str, width: int | None = None, height: int | None = None,
               out_dir: Path = OUTPUT_DIR) -> None:
        if fig is None:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        _export_fig_csv(fig, out_dir / f"{stem}.csv")
        w = int(width or fig.layout.width or 800)
        h = int(height or fig.layout.height or 600)
        for fmt in ("png", "svg"):
            pending.append((fig, out_dir / f"{stem}.{fmt}",
                             {"format": fmt, "width": w, "height": h, "scale": 2}))

    for dim_info in DIMENSIONS:
        dim_key = dim_info["key"]
        value_types = ["raw"] if dim_info["label_source"] == "confidence" else ["raw", "confidence"]
        for value_type in value_types:
            print("=" * 60)
            print(f"{dim_info['label']} — prob vs {value_type} (WS vs LOSO)")
            print("=" * 60)
            _queue(build_general_fig(dim_info, value_type), f"{dim_key}_prob_vs_{value_type}_ws_loso_general")

            result = build_faceted_fig(dim_info, value_type)
            if result is not None:
                fig, w, h = result
                _queue(fig, f"{dim_key}_prob_vs_{value_type}_ws_loso_faceted", width=w, height=h)

            # Mirrored variant only makes sense for value_type="confidence":
            # confidence is orthogonal to the target's own label, so raw
            # class-1-anchored proba can hide a real symmetric effect (see
            # _mirrored_proba docstring). The dimension's own raw_score is
            # already monotonically tied to y_true by construction, so
            # mirroring it would add nothing.
            if value_type == "confidence":
                print(f"  ({dim_info['label']} — mirrored proba variant)")
                mirrored_stem = f"{dim_key}_prob_vs_{value_type}_mirrored_ws_loso"
                _queue(build_general_fig(dim_info, value_type, mirror=True), f"{mirrored_stem}_general")
                result_m = build_faceted_fig(dim_info, value_type, mirror=True)
                if result_m is not None:
                    fig_m, w_m, h_m = result_m
                    _queue(fig_m, f"{mirrored_stem}_faceted", width=w_m, height=h_m)

                _queue(build_mirror_comparison_fig(dim_info), f"{dim_key}_prob_vs_confidence_raw_vs_mirrored")

    print("=" * 60)
    print("All dimensions combined — prob vs raw score")
    print("=" * 60)
    _queue(build_all_dims_combined_fig("raw"), "all_dims_prob_vs_raw_ws_loso_general")

    print("=" * 60)
    print("All dimensions combined — prob vs confidence")
    print("=" * 60)
    _queue(build_all_dims_combined_fig("confidence"), "all_dims_prob_vs_confidence_ws_loso_general")

    print("=" * 60)
    print("All dimensions combined — prob vs confidence (mirrored to true class)")
    print("=" * 60)
    _queue(
        build_all_dims_combined_fig("confidence", mirror=True),
        "all_dims_prob_vs_confidence_mirrored_ws_loso_general",
    )

    print("=" * 60)
    print("Performance + probabilities by confidence, combined")
    print("=" * 60)
    _queue(
        build_performance_and_prob_by_confidence_combined(),
        "performance_and_probabilities_by_confidence",
        out_dir=OUTPUT_DIR.parent,
    )

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
