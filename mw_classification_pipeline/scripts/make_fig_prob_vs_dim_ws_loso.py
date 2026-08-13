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
    DIMENSIONS,
    RESULTS_ROOT,
    _normalise_subject,
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
            text=f"β={slope:.3f}<br>r={r_value:.2f}<br>p={p_value:.1e}<br>n={len(x_arr)}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=9, family="Times New Roman", color=color),
            row=row, col=col,
        )
    else:
        fig.add_annotation(
            text=f"r={r_value:.2f}, n={len(x_arr)}",
            xref="x domain", yref="y domain",
            x=0.05, y=0.95, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=7, family="Times New Roman", color=color),
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
    x_title = f"{label} raw score (0–100)" if value_type == "raw" else "Confidence rating (0–100)"

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
        ann.font.size = 13

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=11),
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
    x_title = f"{label} raw score" if value_type == "raw" else "Confidence rating"

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
                              showticklabels=(row == n), tickfont=dict(size=8))
            fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=col,
                              showticklabels=(col == 1), tickfont=dict(size=8))
        fig.add_annotation(
            text=f"<b>sub-{subj}</b>",
            xref="paper", yref="y domain", row=row, col=1,
            x=-0.02, y=0.5, xanchor="right", yanchor="middle",
            showarrow=False, font=dict(size=8, family="Times New Roman"),
        )

    y_title = "Predicted probability of true class (mirrored)" if mirror else "Predicted probability"
    fig.update_xaxes(title_text=x_title, row=n, col=1)
    fig.update_xaxes(title_text=x_title, row=n, col=2)
    fig.update_yaxes(title_text=y_title, row=(n + 1) // 2, col=1)

    for ann in fig.layout.annotations[:2]:
        ann.font.family = "Times New Roman"
        ann.font.size = 13

    width, height = 580, max(240, n * 115 + 45)
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=11),
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
        fig.update_xaxes(title_text="Confidence rating (0–100)", range=list(SCORE_RANGE), row=row, col=1)
        fig.update_xaxes(title_text="Confidence rating (0–100)", range=list(SCORE_RANGE), row=row, col=2)
        fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=1)
        fig.update_yaxes(range=list(PROBA_RANGE), row=row, col=2)

    if not any_data:
        print(f"  Skipping {label} mirror comparison: no data.")
        return None

    fig.update_yaxes(title_text="Predicted probability", row=1, col=1)
    fig.update_yaxes(title_text="P(true class), mirrored", row=2, col=1)

    for ann in fig.layout.annotations[:2]:
        ann.font.family = "Times New Roman"
        ann.font.size = 13

    # Panel letters — bold, top-left corner of each row's first (WS) panel,
    # clearly outside the plot area so they read as structure rather than a
    # prefix on the "Within-Subject"/"LOSO" column headers above row 1.
    for row, letter in [(1, "A"), (2, "B")]:
        fig.add_annotation(
            text=f"<b>{letter}</b>",
            xref="x domain", yref="y domain", row=row, col=1,
            x=-0.14, y=1.20, xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=20, family="Times New Roman", color="#333333"),
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=11),
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
        horizontal_spacing=0.03,
        vertical_spacing=0.10,
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
        fig.layout.annotations[i].font.size = 13
        fig.layout.annotations[i].font.family = "Times New Roman"

    if not any_data:
        print(f"  Skipping all-dims combined ({value_type}): no data.")
        return None

    x_title = "Raw score (0–100)" if value_type == "raw" else "Confidence rating (0–100)"
    # Short form here (vs. the full "...of true class (mirrored)" used in the
    # single-dimension figures): with two rows this close together, the long
    # title's rotated text is taller than each row's cell and the row-1/row-2
    # labels visually overlap at the seam.
    y_title = "P(true class)" if mirror else "Predicted probability"
    fig.update_xaxes(title_text=x_title, row=2)
    fig.update_yaxes(title_text=y_title, col=1)

    for row, pipeline_label in [(1, "Within-Subject"), (2, "LOSO")]:
        fig.add_annotation(
            text=f"<b>{pipeline_label}</b>",
            xref="paper", yref="y domain", row=row, col=1,
            x=-0.075, y=0.5, xanchor="right", yanchor="middle",
            textangle=-90, showarrow=False,
            font=dict(size=12, family="Times New Roman"),
        )

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Times New Roman", size=11),
        width=260 * n, height=480,
        margin=dict(t=30, b=50, l=95, r=15),
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[go.Figure, Path, dict]] = []

    def _queue(fig: go.Figure | None, stem: str, width: int | None = None, height: int | None = None) -> None:
        if fig is None:
            return
        w = int(width or fig.layout.width or 800)
        h = int(height or fig.layout.height or 600)
        for fmt in ("png", "svg"):
            pending.append((fig, OUTPUT_DIR / f"{stem}.{fmt}",
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
