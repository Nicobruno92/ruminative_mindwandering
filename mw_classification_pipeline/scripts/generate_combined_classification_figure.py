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
    scatter_shap_marker.{png,svg}
                              — WS vs LOSO mean |SHAP|, one point per CBPT
                                marker (mean over ROI), one panel per
                                dimension. Built from the same true_runs SHAP
                                pkls as scatter_shap_absolute — unlike
                                marker_direction_all_dimensions above, it does
                                not depend on feature_consistency_analysis.py.
    scatter_shap_marker_{key}.{png,svg}
                              — the same marker-level comparison, one
                                standalone file per dimension (key = on_off,
                                valence, selfother, time, confidence) at a size
                                comfortable to read every label, mirroring
                                residual_vs_plain_{key} below.
    spatial_comparison_ws_loso.{png,svg}
                              — WS (row 1) vs LOSO (row 2) spatial (per-electrode)
                                decoding topomaps, one column per dimension
                                (the 5 canonical dimensions). Reads per_channel_metrics.csv
                                from results/MW_Classification/SpatialDecoding/;
                                skipped with a message for any dimension missing
                                on either pipeline.
    spatial_comparison_ws_loso_residualized.{png,svg}
                              — Same 5 dimension columns as
                                spatial_comparison_ws_loso, but 4 rows instead
                                of 2: WS-Full, WS-Residualized, LOSO-Full,
                                LOSO-Residualized (RESIDUALIZED_DIR). Does not
                                modify spatial_comparison_ws_loso.{png,svg}.
    group_comparison_spatial_combined.{png,svg}
                              — WS | LOSO "Global Decoding" (True/Residualized/
                                Shuffled AUC density) next to "Spatial Decoding"
                                (topomap) per dimension, one combined panel per
                                pipeline. Pure Matplotlib (see
                                plot_group_spatial_combined docstring for why).
    cv_stability_combined.{png,svg}
                              — Two-panel union of scatter_loso_vs_ws.{png,svg}
                                (Panel A, per-subject mean AUC) and
                                scatter_proba_ws_vs_loso.{png,svg} (Panel B,
                                per-probe predicted probability), stacked one
                                row per panel, one column per dimension. Reads
                                the same source data as those two standalone
                                figures and does not replace them (kept as
                                separate outputs per project convention).
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
import re
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
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, pearsonr
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

# Per-dimension color, consistent across every figure. pole_low/pole_high are
# the canonical SHORT_POLES wording (CLAUDE.md "Dimension Labels & Pole
# Wording" — same pairs as Stats_andrillon/plot_paper_figures.py's
# SHORT_POLES and Behavior/Probe_analysis/probe_dimension_cloud_plot.py's
# DIMENSIONS): raw-score axis labels must name the actual poles, never a
# generic "Raw score (0-100)".
DIMENSIONS = [
    {
        "key":          "on_off",
        "ws_dir":       "on_vs_off_within_median",
        "loso_dir":     "ON_vs_OFF_within_median",
        "label":        "On/Off-Task",
        "color":        DIM_COLORS["onoff"],   # red
        "label_source": "onoff",
        "pole_low":     "off-task",
        "pole_high":    "on-task",
    },
    {
        "key":          "valence",
        "ws_dir":       "valence_within_median",
        "loso_dir":     "valence_within_median",
        "label":        "Valence",
        "color":        DIM_COLORS["valence"],   # blue
        "label_source": "valence",
        "pole_low":     "negative",
        "pole_high":    "positive",
    },
    {
        "key":          "selfother",
        "ws_dir":       "selfother_within_median",
        "loso_dir":     "selfother_within_median",
        "label":        "Self/Other",
        "color":        DIM_COLORS["selfother"],   # green
        "label_source": "selfother",
        "pole_low":     "self-focused",
        "pole_high":    "other-focused",
    },
    {
        "key":          "time",
        "ws_dir":       "time_within_median",
        "loso_dir":     "time_within_median",
        "label":        "Time",
        "color":        DIM_COLORS["time"],   # purple
        "label_source": "time",
        "pole_low":     "past",
        "pole_high":    "future",
    },
    {
        "key":          "confidence",
        "ws_dir":       "confidence_within_median",
        "loso_dir":     "confidence_within_median",
        "label":        "Confidence",
        "color":        DIM_COLORS["confidence"],   # orange
        "label_source": "confidence",
        "pole_low":     "low",
        "pole_high":    "high",
    },
]

PERM_COLOR   = PALETTE["neutral"]["permutation"]   # gray
CHANCE       = 0.5
AUC_RANGE    = (0.35, 1.0)

# Same lookup as make_fig_prob_vs_dim_ws_loso.py's _CONFIDENCE_DIM — the two
# scripts are documented as one figure family (see that module's docstring)
# and must share pole wording for any confidence axis, not just a color.
_CONFIDENCE_DIM = next(d for d in DIMENSIONS if d["label_source"] == "confidence")

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


def _format_p_compact(p: float) -> str:
    """
    APA-style p-value: decimal, no leading zero, no space around "=" — the
    same convention as make_fig_prob_vs_dim_ws_loso.py's _format_p. Kept as
    a separate helper rather than changing format_pval itself, which two
    dozen other figures in this file rely on for its "p = 0.032" spacing;
    used only by the handful of scatter figures meant to read as one family
    with prob_vs_dim/ (scatter_loso_vs_ws, scatter_proba_ws_vs_loso,
    scatter_regression_overlay).
    """
    if np.isnan(p):
        return "p=n/a"
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}".replace("=0.", "=.")


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
# CSV export helpers
# =============================================================================

# Text an axis-anchored annotation must match (after stripping HTML tags) to
# be trusted as a row/column label rather than a p-value readout or other
# incidental annotation — the canonical dimension labels, the two pipeline
# column titles, and per-subject "sub-NN" row labels (plot_individual /
# plot_individual_with_residualized).
_KNOWN_AXIS_LABELS = {d["label"] for d in DIMENSIONS} | {"Within-Subject", "LOSO"}
_SUBJECT_LABEL_RE = re.compile(r"^sub-\d+$")


def _infer_axis_labels(fig: go.Figure) -> dict[str, str]:
    """
    Best-effort ``{axis_id: label}`` map built from a figure's OWN row/column
    annotations (e.g. the ``"<b>On/Off-Task</b>"`` text every multi-panel
    figure in this script already places at each row's y-domain, or the
    ``"Within-Subject"``/``"LOSO"`` column titles) — reused by
    :func:`_export_fig_csv` to label exported rows without editing every
    individual ``fig.add_trace`` call site.
    """
    axis_labels: dict[str, str] = {}
    for ann in fig.layout.annotations:
        text = re.sub(r"<[^>]+>", "", ann.text or "").strip()
        if text not in _KNOWN_AXIS_LABELS and not _SUBJECT_LABEL_RE.match(text):
            continue
        for ref in (ann.xref, ann.yref):
            if ref and ref.endswith(" domain"):
                axis_labels.setdefault(ref.split()[0], text)

    # make_subplots(subplot_titles=...) / column_titles=... annotations carry
    # no axis id at all (xref=yref="paper") — match by x-position against
    # each xaxis's own domain instead. Y is intentionally NOT matched this
    # way: column titles at a shared top y would collide across columns.
    col_anns = [
        (re.sub(r"<[^>]+>", "", a.text or "").strip(), a.x)
        for a in fig.layout.annotations
        if a.xref == "paper" and a.yref == "paper"
    ]
    col_anns = [(t, x) for t, x in col_anns if t in _KNOWN_AXIS_LABELS]
    if col_anns:
        for axis_name in fig.layout:
            axis_name = str(axis_name)
            if not axis_name.startswith("xaxis"):
                continue
            domain = getattr(fig.layout[axis_name], "domain", None)
            if not domain:
                continue
            axis_id = "x" + axis_name[5:]
            for text, ax in col_anns:
                if domain[0] - 1e-6 <= ax <= domain[1] + 1e-6:
                    axis_labels.setdefault(axis_id, text)
    return axis_labels


#: An annotation is treated as a plotted *statistic* (r/p, z/p, AUC, β...)
#: rather than a structural label (dimension name, pole wording, subject id,
#: panel letter) when it is anchored relative to a specific panel's axis
#: (at least one of xref/yref ends in " domain" — a bare "x"/"y"/"paper" ref
#: is a data-coordinate arrow target or a whole-figure title/letter, not a
#: per-panel readout) *and* its text contains a digit — every stats
#: annotation in this file has one (an estimate, a p-value, a percentage);
#: every purely-verbal label (a pole pair, a category name) does not.
_STATS_ANNOTATION_DIGIT_RE = re.compile(r"\d")


def _export_annotation_stats(fig: go.Figure, axis_labels: dict[str, str]) -> pd.DataFrame:
    """
    Every per-panel statistic *annotation* (r/p, z/p, AUC, β, LRT...) as one
    tidy row per annotation — the numbers actually printed on the figure,
    which :func:`_export_fig_csv`'s trace dump never captures because a
    statistic lives in ``fig.add_annotation(text=...)``, not in any trace's
    x/y. Without this, a reviewer reading the points CSV has to recompute
    r/p themselves to check what the figure claims, instead of reading it
    off a table — see the ``scientific-plots`` skill's
    ``references/learnings/scatter-correlation.md`` for why both tables are
    required, not just the raw one.
    """
    known_text = _KNOWN_AXIS_LABELS
    rows: list[dict] = []
    for ann in fig.layout.annotations:
        raw_text = ann.text or ""
        stripped = re.sub(r"<[^>]+>", " ", raw_text).strip()
        if not stripped or stripped in known_text or _SUBJECT_LABEL_RE.match(stripped):
            continue
        if not _STATS_ANNOTATION_DIGIT_RE.search(stripped):
            continue
        refs_anchored = any(
            isinstance(r, str) and r.endswith(" domain") for r in (ann.xref, ann.yref)
        )
        if not refs_anchored:
            continue
        xaxis = ann.xref.split()[0] if isinstance(ann.xref, str) else ann.xref
        yaxis = ann.yref.split()[0] if isinstance(ann.yref, str) else ann.yref
        rows.append(dict(
            row_label=axis_labels.get(yaxis, ""), col_label=axis_labels.get(xaxis, ""),
            xref=ann.xref, yref=ann.yref, x=ann.x, y=ann.y,
            text=re.sub(r"\s*<br\s*/?>\s*", " | ", raw_text),
        ))
    return pd.DataFrame(rows)


def _export_fig_csv(fig: go.Figure, csv_path: Path) -> None:
    """
    Dump every trace's plotted values — the exact numbers rendered, not a
    KDE/derived summary — to one tidy long-format CSV next to the figure's
    PNG/SVG. Generic over every trace type used in this project: Violin,
    Scatter, and Bar via x/y; Pie via labels/values. Per-point ``subject``
    is included where the trace carries it via ``customdata`` (only
    plot_individual / plot_individual_with_residualized set this — every
    other figure's data is already fully identified by x/y alone).

    Also writes a companion ``{stem}_stats.csv`` — see
    :func:`_export_annotation_stats` — whenever the figure carries at least
    one per-panel statistic annotation, so the numbers backing the plot
    (r/p, z/p, AUC...) are on disk next to the raw points that produced
    them, not only visible as text baked into the PNG/SVG.
    """
    axis_labels = _infer_axis_labels(fig)
    stats_df = _export_annotation_stats(fig, axis_labels)
    if not stats_df.empty:
        stats_df.to_csv(csv_path.with_name(f"{csv_path.stem}_stats.csv"), index=False)
    rows: list[dict] = []
    for i, trace in enumerate(fig.data):
        # Pie traces are domain-type (no xaxis/yaxis attribute at all).
        xaxis = getattr(trace, "xaxis", None) or "x"
        yaxis = getattr(trace, "yaxis", None) or "y"
        common = dict(
            trace_index=i, trace_type=trace.type, trace_name=trace.name or "",
            row_label=axis_labels.get(yaxis, ""), col_label=axis_labels.get(xaxis, ""),
        )
        if trace.type == "pie":
            for lab, val in zip(trace.labels or [], trace.values or []):
                rows.append({**common, "label": lab, "value": val})
            continue
        xs = list(trace.x) if trace.x is not None else []
        ys = list(trace.y) if trace.y is not None else []
        subj = list(trace.customdata) if trace.customdata is not None else []
        for j in range(max(len(xs), len(ys))):
            row = {**common,
                   "x": xs[j] if j < len(xs) else None,
                   "y": ys[j] if j < len(ys) else None}
            if j < len(subj):
                s = subj[j]
                row["subject"] = s[0] if isinstance(s, (list, tuple, np.ndarray)) else s
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)


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
            customdata=p_df["subject"].values,
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
            customdata=t_df["subject"].values,
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
        customdata=sorted_subjects,
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
            customdata=p_df["subject"].values,
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
            customdata=t_df["subject"].values,
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
            customdata=r_df["subject"].values,
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
        customdata=sorted_subjects,
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

    labeled = []
    for pipeline_label, metrics in [("WS", ws_metrics), ("LOSO", loso_metrics)]:
        for dim_label, df in metrics.items():
            d = df.copy()
            d["dimension"] = dim_label
            d["pipeline"] = pipeline_label
            labeled.append(d)
    pd.concat(labeled, ignore_index=True).to_csv(
        OUTPUT_DIR / "spatial_comparison_ws_loso.csv", index=False
    )
    return out_png


def plot_spatial_comparison_panel_with_residualized() -> Path | None:
    """
    Variant of :func:`plot_spatial_comparison_panel` with 4 rows instead of 2:
    WS-Full, WS-Residualized, LOSO-Full, LOSO-Residualized — keeping the SAME
    5 dimension columns as the canonical panel (not doubled), so a reader
    scans down each column to see how residualizing that one dimension
    changes its own pipeline's topomap, with WS and LOSO each getting their
    own full/residualized row pair.

    Reuses ``plot_pipeline_comparison_topomap_panel_multirow`` (the N-row
    generalization of the 2-row function the canonical panel uses), with the
    residualized-contrast data (RESIDUALIZED_DIR) swapped in for rows 2 and 4.

    Does not modify or overwrite spatial_comparison_ws_loso.{png,svg} — saves
    to its own spatial_comparison_ws_loso_residualized.{png,svg}.

    Returns the PNG path, or None if fewer than one dimension has data in
    all four (WS-full, WS-res, LOSO-full, LOSO-res) row sets.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.spatial_decoding_utils import plot_pipeline_comparison_topomap_panel_multirow

    ws_full, ws_res, loso_full, loso_res, dim_colors = {}, {}, {}, {}, {}
    for dim_info in GROUP_ROW_DIMENSIONS:
        label = dim_info["label"]
        res_dim_info = _residualized_dim_info(dim_info)

        ws_df       = load_spatial_metrics("ws", dim_info)
        loso_df     = load_spatial_metrics("loso", dim_info)
        ws_res_df   = load_spatial_metrics("ws", res_dim_info)
        loso_res_df = load_spatial_metrics("loso", res_dim_info)
        missing = [name for name, df in [
            ("WS full", ws_df), ("LOSO full", loso_df),
            ("WS res", ws_res_df), ("LOSO res", loso_res_df),
        ] if df is None]
        if missing:
            print(f"  ! {label}: missing spatial per_channel_metrics.csv for "
                  f"{', '.join(missing)}; skipping from residualized comparison panel.")
            continue

        ws_full[label]   = ws_df
        loso_full[label] = loso_df
        ws_res[label]    = ws_res_df
        loso_res[label]  = loso_res_df
        dim_colors[label] = dim_info["color"]

    if not ws_full:
        print("  No dimension has both full and residualized spatial data on both "
              "pipelines; skipping residualized spatial comparison panel.")
        return None

    # Inner row labels are just "Full"/"Residualized" — the outer group label
    # (Within-Subject/LOSO, matching the canonical panel's row naming) is
    # supplied separately via row_groups so it's drawn once per pipeline,
    # spanning both its rows, instead of repeated on every row.
    row_specs = [
        ("Full",           ws_full),
        ("Residualized",   ws_res),
        ("Full",           loso_full),
        ("Residualized",   loso_res),
    ]
    row_groups = [("Within-Subject", 2), ("LOSO", 2)]
    out_png = OUTPUT_DIR / "spatial_comparison_ws_loso_residualized.png"
    out_svg = OUTPUT_DIR / "spatial_comparison_ws_loso_residualized.svg"
    for out_path in (out_png, out_svg):
        plot_pipeline_comparison_topomap_panel_multirow(
            row_specs, value_col="mean_auc", out_path=str(out_path),
            mask_col="sig", dim_colors=dim_colors, row_groups=row_groups,
        )

    labeled = []
    for pipeline_label, contrast_label, metrics in [
        ("WS", "Full", ws_full), ("WS", "Residualized", ws_res),
        ("LOSO", "Full", loso_full), ("LOSO", "Residualized", loso_res),
    ]:
        for dim_label, df in metrics.items():
            d = df.copy()
            d["dimension"] = dim_label
            d["pipeline"] = pipeline_label
            d["contrast"] = contrast_label
            labeled.append(d)
    pd.concat(labeled, ignore_index=True).to_csv(
        OUTPUT_DIR / "spatial_comparison_ws_loso_residualized.csv", index=False
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
    # Columns: row-label | WS-panel | WS-cbar | spacer | LOSO-panel | LOSO-cbar
    #
    # One axes per (row, pipeline) now, not two: the topomap is a small inset
    # INSIDE the "Global Decoding" axes (top-right corner) rather than a
    # separate column next to it. width_ratios/height_ratios are literal
    # inches (figsize is their exact sum) so the inset's inches-per-fraction
    # can be computed directly and forced square — same reasoning as before
    # about mne.viz.plot_topomap's equal-aspect + the transAxes-based clip
    # circle in _draw_cbpt_topomap disagreeing on a non-square axes.
    # ROW_H/PANEL_W set the actual on-screen size of the curve panel —
    # INSET_IN stays fixed (small, per earlier feedback) so growing these
    # makes the inset proportionally SMALLER relative to its panel, i.e.
    # more "aligned"/tucked-in rather than dominating it.
    ROW_H   = 2.0
    PANEL_W = 4.6
    INSET_IN = 0.85   # inset side length in inches — keeps the topo genuinely small
    label_w, cbar_w, spacer_w = 0.42, 0.13, 0.25
    col_w = [label_w, PANEL_W, cbar_w, spacer_w, PANEL_W, cbar_w]
    block_cols = {"ws": dict(panel=1, cbar=2), "loso": dict(panel=4, cbar=5)}
    header_h = 1.3

    # KDE ylim: curves peak at data-y=1.0 (fixed by _kde_fill's normalisation).
    # Y_TOP is picked so that fraction 1.0/Y_TOP (the peak line) sits safely
    # below INSET_Y0 (below) — both the stats text and the inset live in the
    # headroom band above the peak, separated from each other horizontally
    # (text left-of-centre, inset at the right edge) rather than stacked
    # vertically, so neither has to fight the other for space. Lower than
    # before (2.3 vs 3.0) so the curves themselves fill more of the panel —
    # a bigger ROW_H shrinks the inset's fraction enough to still clear the
    # peak with margin at this Y_TOP (checked below via INSET_Y0).
    Y_TOP = 2.3
    INSET_W_FRAC = INSET_IN / PANEL_W
    INSET_H_FRAC = INSET_IN / ROW_H
    INSET_X0 = 0.98 - INSET_W_FRAC
    # Bottom-right, not top-right: sits at the same height as the curves
    # (the curve zone runs from y-fraction 0 to ~peak_fraction=1/Y_TOP,
    # about 0.435 here) rather than up in the text's headroom band. A small
    # margin (0.03) off the baseline keeps it off the AUC axis line.
    INSET_Y0 = 0.03

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

    # ---- Header row: block title banner + "Global Decoding" sub-header ----
    # Offsets are a fraction of the header row's OWN height (not a fixed
    # figure-fraction) so banner/sub-header spacing stays correct regardless
    # of how header_h is tuned — a fixed-fraction offset sized for one
    # header_h silently collapses the two into each other at a smaller one.
    header_y0, header_y1 = _row_y(0)
    header_span = header_y1 - header_y0
    for pkey, block_title in pipelines:
        cols = block_cols[pkey]
        bx0, bx1 = _col_x(cols["panel"], cols["cbar"])
        fig.text((bx0 + bx1) / 2, header_y1 - 0.10 * header_span, block_title,
                 ha="center", va="top", fontsize=13, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#333333", edgecolor="none"))

        fig.text((bx0 + bx1) / 2, header_y0 + 0.10 * header_span,
                 "Global Decoding",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ---- Per-dimension rows ----
    last_im_by_pipeline: dict[str, object] = {}
    for r, dim_info in enumerate(row_dims):
        row   = r + 1
        color = dim_info["color"]

        label_fontsize = 11 if len(dim_info["label"]) <= 12 else 8
        lp = gs[row, 0].get_position(fig)
        fig.text((lp.x0 + lp.x1) / 2, (lp.y0 + lp.y1) / 2, dim_info["label"],
                 ha="center", va="center", rotation=90, fontsize=label_fontsize,
                 fontweight="bold", color=color)

        for pkey, _ in pipelines:
            cols = block_cols[pkey]
            t, p, t_res, p_res, p_adj, res_p_adj = dist_rows[pkey][r]

            # --- Global Decoding: vertical filled density curves ---
            ax_d = fig.add_subplot(gs[row, cols["panel"]])
            _kde_fill(ax_d, p,     PERM_COLOR, 0.55, bw=BW_GROUP)
            _kde_fill(ax_d, t,     color,      0.72, bw=BW_GROUP)
            _kde_fill(ax_d, t_res, color,      RESIDUALIZED_ALPHA, bw=BW_GROUP)
            _clean_kde_ax(ax_d, show_xticks=(row == n_dims), show_xlabel=(row == n_dims),
                         fixed_ylim=True)
            ax_d.set_ylim(0, Y_TOP)
            # x needs axes-fraction (0.32 == left-of-centre, clear of the
            # inset's footprint at the right) while y needs data coordinates
            # (anchored to the curves' fixed peak at 1.0) — a blended
            # transform mixes the two per-axis; a bare x with no transform
            # defaults to DATA coordinates on BOTH axes, which previously
            # put the text near AUC=x on the curve axis instead of at a
            # fixed fraction of the panel.
            blend = transforms.blended_transform_factory(ax_d.transAxes, ax_d.transData)
            line1, line2 = _true_res_annotation_lines(t, p_adj, t_res, res_p_adj)
            ax_d.text(0.32, 1.30, line2, transform=blend, ha="center", va="bottom",
                      fontsize=7, fontstyle="italic", color=color)
            ax_d.text(0.32, 1.60, line1, transform=blend, ha="center", va="bottom",
                      fontsize=7, fontweight="bold", color=color)

            # --- Spatial Decoding: small topomap inset, top-right corner ---
            # inset_axes bounds are axes-fraction by default (no transform
            # kwarg needed), unlike the text above.
            ax_s = ax_d.inset_axes([INSET_X0, INSET_Y0, INSET_W_FRAC, INSET_H_FRAC])
            df = spatial_by_pipeline[pkey].get(dim_info["label"])
            if df is not None:
                info = build_info_from_channels(df["channel"].tolist())
                mask = df["sig"].to_numpy(dtype=bool) if "sig" in df.columns else None
                last_im_by_pipeline[pkey] = _draw_cbpt_topomap(
                    ax_s, df["mean_auc"].to_numpy(dtype=float), info, mask,
                    vmin, vmax, "viridis", markersize=2.5,
                )
            else:
                ax_s.axis("off")
                ax_s.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, color="#999999")

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

    # Global-decoding half: the True/Shuffled/Residualized AUC arrays behind
    # each panel's density curves (same arrays fed to _kde_fill above).
    global_rows = []
    for pkey, _ in pipelines:
        for r, dim_info in enumerate(row_dims):
            t, p, t_res, p_res, _, _ = dist_rows[pkey][r]
            for series_name, vals in [
                ("Full_True", t), ("Full_Shuffled", p),
                ("Residualized_True", t_res), ("Residualized_Shuffled", p_res),
            ]:
                for v in vals:
                    global_rows.append(dict(
                        pipeline=pkey, dimension=dim_info["label"],
                        series=series_name, auc=v,
                    ))
    pd.DataFrame(global_rows).to_csv(
        OUTPUT_DIR / "group_comparison_spatial_combined_global.csv", index=False
    )

    # Spatial-decoding half: the per_channel_metrics.csv rows behind each
    # dimension's topomap inset.
    spatial_rows = []
    for pkey, sp in spatial_by_pipeline.items():
        for dim_label, df in sp.items():
            if df is None:
                continue
            d = df.copy()
            d["pipeline"] = pkey
            d["dimension"] = dim_label
            spatial_rows.append(d)
    if spatial_rows:
        pd.concat(spatial_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "group_comparison_spatial_combined_spatial.csv", index=False
        )

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
        fig.layout.annotations[i].font.size  = 18
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
            textfont=dict(size=11, family="Times New Roman"),
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

            r, p = pearsonr(merged["ws"].values, merged["loso"].values)
            fig.add_annotation(
                text=f"r={r:.2f}<br>{_format_p_compact(p)}",
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.05, y=0.96,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=18, color=color, family="Times New Roman"),
            )

        # Axis formatting
        x_ax = f"xaxis{col}" if col > 1 else "xaxis"
        y_ax = f"yaxis{col}" if col > 1 else "yaxis"
        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range = [auc_min, auc_max]
            fig.layout[ax_name].tickmode = "array"
            fig.layout[ax_name].tickvals = [0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext = ["0.5", ".75", "1.0"]
            fig.layout[ax_name].tickfont = dict(size=13)
            fig.layout[ax_name].showgrid = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline = False

        fig.layout[x_ax].title = dict(text="WS AUC", font=dict(size=15))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO AUC", font=dict(size=15))

    fig.update_layout(
        title=dict(
            text="LOSO vs Within-Subject AUC per Participant",
            font=dict(size=18, family="Times New Roman"),
        ),
        template="plotly_white",
        height=380,
        width=320 * n_dims,
        margin=dict(l=60, r=30, t=100, b=60),
        font=dict(family="Times New Roman", size=15),
    )
    return fig


# =============================================================================
# Figure D2 — AUC vs self-reported confidence (exploratory)
# =============================================================================


def plot_auc_vs_confidence_scatter() -> go.Figure:
    """
    Two rows (WS, LOSO) x one column per dimension: per-subject AUC against
    that subject's mean self-reported confidence.

    x = subject's mean 'confidence' rating across every probe of every task
    (PROBE_DATA_PATH), independent of which dimension is being decoded.
    y = subject's mean AUC for that dimension/pipeline (load_subject_data).

    Tests whether subjects who report higher confidence on average are also
    better individually decoded — for any of the five dimensions, not only
    the 'confidence' dimension itself. EXPLORATORY: not part of the
    confirmatory Section 2 (CBPT) / Section 3 (classification) analyses: no
    correction is applied across the 10 panels here.
    """
    from scipy.stats import pearsonr

    probe_df = pd.read_csv(PROBE_DATA_PATH)
    probe_df["subject"] = probe_df["subject"].astype(str).str.zfill(2)
    subj_confidence = probe_df.groupby("subject")["confidence"].mean().rename("confidence")

    n_dims = len(DIMENSIONS)
    pipelines = [("ws", "WS AUC"), ("loso", "LOSO AUC")]

    subplot_titles = [d["label"] for d in DIMENSIONS] + [""] * n_dims
    fig = make_subplots(
        rows=2, cols=n_dims,
        # 'columns' / True share one axis object per column/row so only the
        # bottom row keeps x tick labels and only the left column keeps y
        # tick labels — lets the panels sit flush with near-zero spacing
        # instead of each repeating its own tick numbers.
        shared_xaxes="columns",
        shared_yaxes=True,
        horizontal_spacing=0.008,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
    )

    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color = dim_info["color"]
        fig.layout.annotations[i].font.size = 20
        fig.layout.annotations[i].font.family = "Times New Roman"

    x_min = float(subj_confidence.min()) - 3
    x_max = float(subj_confidence.max()) + 3

    # AUC data never exceeds 1.0, so the band above it (up to Y_TOP) is
    # reserved for the r/n annotation — keeps it clear of subject points and
    # their labels regardless of where a given panel's cloud happens to sit.
    y_bottom, y_top = AUC_RANGE[0], 1.16
    annot_y = 1.09

    for row_idx, (pipeline, y_title) in enumerate(pipelines):
        row = row_idx + 1
        for col_idx, dim_info in enumerate(DIMENSIONS):
            col   = col_idx + 1
            color = dim_info["color"]
            idx   = (row - 1) * n_dims + col
            x_ref = "x" if idx == 1 else f"x{idx}"
            y_ref = "y" if idx == 1 else f"y{idx}"
            x_ax  = "xaxis" if idx == 1 else f"xaxis{idx}"
            y_ax  = "yaxis" if idx == 1 else f"yaxis{idx}"

            t_df, _ = load_subject_data(pipeline, dim_info)
            if t_df.empty:
                continue
            auc_means = t_df.groupby("subject")["auc"].mean().rename("auc")
            auc_means.index = auc_means.index.astype(str)

            merged = pd.concat([subj_confidence, auc_means], axis=1).dropna()
            if merged.empty:
                continue

            subj_labels = [str(int(s)) if s.isdigit() else s for s in merged.index]

            # Chance line
            fig.add_shape(
                type="line", x0=x_min, x1=x_max, y0=CHANCE, y1=CHANCE,
                xref=x_ref, yref=y_ref,
                line=dict(color=PERM_COLOR, dash="dash", width=1.0),
            )

            fig.add_trace(go.Scatter(
                x=merged["confidence"].values,
                y=merged["auc"].values,
                mode="markers+text",
                text=subj_labels,
                textposition="top center",
                textfont=dict(size=11, family="Times New Roman"),
                marker=dict(color=color, size=8, opacity=0.85,
                            line=dict(color="white", width=0.5)),
                showlegend=False,
                name=f"{dim_info['label']} ({y_title})",
            ), row=row, col=col)

            if len(merged) >= 3:
                m, b = np.polyfit(merged["confidence"].values, merged["auc"].values, 1)
                x_reg = np.array([x_min, x_max])
                y_reg = m * x_reg + b
                fig.add_trace(go.Scatter(
                    x=x_reg, y=y_reg,
                    mode="lines",
                    line=dict(color=color, width=2, dash="solid"),
                    showlegend=False,
                ), row=row, col=col)

                r, p = pearsonr(merged["confidence"].values, merged["auc"].values)
                fig.add_annotation(
                    text=f"r = {r:.2f}, {format_pval(p)}",
                    xref=f"{x_ref} domain", yref=y_ref,
                    x=0.05, y=annot_y,
                    xanchor="left", yanchor="middle",
                    showarrow=False,
                    font=dict(size=14, color=color, family="Times New Roman"),
                )

            fig.layout[y_ax].range = [y_bottom, y_top]
            fig.layout[y_ax].tickmode = "array"
            fig.layout[y_ax].tickvals = [0.5, 0.75, 1.0]
            fig.layout[y_ax].ticktext = ["0.5", ".75", "1.0"]
            fig.layout[y_ax].tickfont = dict(size=15)
            fig.layout[y_ax].showgrid = True
            fig.layout[y_ax].gridcolor = "#EEEEEE"
            fig.layout[y_ax].zeroline = False

            fig.layout[x_ax].range = [x_min, x_max]
            fig.layout[x_ax].tickfont = dict(size=15)
            fig.layout[x_ax].showgrid = True
            fig.layout[x_ax].gridcolor = "#EEEEEE"
            fig.layout[x_ax].zeroline = False

            if col == 1:
                fig.layout[y_ax].title = dict(text=y_title, font=dict(size=17))
            if row == 2:
                x_title = f"{_CONFIDENCE_DIM['pole_low']} → {_CONFIDENCE_DIM['pole_high']}"
                fig.layout[x_ax].title = dict(text=x_title, font=dict(size=15))

    fig.update_layout(
        template="plotly_white",
        height=780,
        width=340 * n_dims,
        margin=dict(l=70, r=20, t=50, b=70),
        font=dict(family="Times New Roman", size=15),
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
        r, p = pearsonr(x, y)

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
            name=f"{dim_info['label']} (r={r:.2f}, {_format_p_compact(p)})",
        ))

    fig.update_layout(
        title=dict(
            text="LOSO vs WS AUC — Regression Lines by Dimension",
            font=dict(size=16, family="Times New Roman"),
        ),
        template="plotly_white",
        height=480, width=620,
        margin=dict(l=70, r=30, t=80, b=70),
        font=dict(family="Times New Roman", size=13),
        xaxis=dict(
            title=dict(text="WS AUC", font=dict(size=15)),
            range=[auc_min, auc_max],
            tickmode="array", tickvals=[0.5, 0.75, 1.0],
            ticktext=["0.5", ".75", "1.0"],
            tickfont=dict(size=13),
            showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="LOSO AUC", font=dict(size=15)),
            range=[auc_min, auc_max],
            tickmode="array", tickvals=[0.5, 0.75, 1.0],
            ticktext=["0.5", ".75", "1.0"],
            tickfont=dict(size=13),
            showgrid=True, gridcolor="#EEEEEE", zeroline=False,
        ),
        legend=dict(
            font=dict(size=14, family="Times New Roman"),
            bgcolor="rgba(255,255,255,0.8)",
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
        horizontal_spacing=0.07,
        subplot_titles=[d["label"] for d in DIMENSIONS],
    )
    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].font.color  = dim_info["color"]
        fig.layout.annotations[i].font.size   = 16
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

    # Spread labels around their point instead of a fixed "top center": with
    # up to 2*SHAP_SCATTER_TOP_N labelled points in one panel, a shared fixed
    # position stacks nearby labels directly on top of each other (see
    # _residual_panel above, which the same fix was needed for). Points left
    # of the labelled subset's own median go left, the rest go right; within
    # each side a 3-way top/middle/bottom cycle (ordered by y) keeps close
    # neighbours from landing on the same spot.
    x_hi, y_hi = x_common[highlight], y_common[highlight]
    x_mid = float(np.median(x_hi))
    order = np.argsort(np.argsort(y_hi))
    left_cycle = np.array(["middle left", "top left", "bottom left"])
    right_cycle = np.array(["middle right", "top right", "bottom right"])
    goes_left = x_hi < x_mid
    # A point within the outer 8% of the axis range would have its "outward"
    # label push past the panel/figure boundary and get clipped — anchor
    # those toward the inside of the axis instead, overriding the median split.
    edge_margin = 0.08 * (axis_range[1] - axis_range[0])
    goes_left = np.where(x_hi > axis_range[1] - edge_margin, True, goes_left)
    goes_left = np.where(x_hi < axis_range[0] + edge_margin, False, goes_left)
    label_positions = np.where(goes_left, left_cycle[order % 3], right_cycle[order % 3])

    fig.add_trace(go.Scatter(
        x=x_hi, y=y_hi,
        mode="markers+text",
        marker=dict(color=color, size=7, opacity=0.9,
                    line=dict(color="white", width=0.5)),
        text=[_pretty_feature_label(f) for f in highlight_feats],
        textposition=label_positions,
        textfont=dict(size=9, family="Times New Roman"),
        cliponaxis=False,
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
        font=dict(size=12, color=color, family="Times New Roman"),
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
        fig.layout[ax_name].tickfont   = dict(size=11)
    fig.layout[x_ax].title = dict(text=x_label, font=dict(size=13))
    if col == 1:
        fig.layout[y_ax].title = dict(text=y_label, font=dict(size=13))


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
            font=dict(size=16, family="Times New Roman"),
        ),
        template="plotly_white",
        height=460,
        width=420 * len(DIMENSIONS),
        margin=dict(l=70, r=40, t=110, b=70),
        font=dict(family="Times New Roman", size=13),
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
            title=dict(text=title, font=dict(size=16, family="Times New Roman")),
            template="plotly_white",
            height=460,
            width=420 * len(DIMENSIONS),
            margin=dict(l=70, r=40, t=110, b=70),
            font=dict(family="Times New Roman", size=13),
        )

    return fig_abs, fig_dir


def _pretty_marker_name(name: str) -> str:
    """Shorten a raw marker name for an axis label, keeping its family tag."""
    for old, new in (("psd_relative_", "PSD "), ("kolmogorov_complexity", "Kolmogorov"),
                     ("slowwaves_", "SW "), ("wsmi_", "wSMI "), ("PE_", "PE ")):
        if name.startswith(old) or name == old.rstrip("_"):
            return name.replace(old, new)
    return name


# Compact marker×ROI labels for scatter-panel point labels, where up to
# SHAP_SCATTER_TOP_N*2 labels share one panel and every character counts —
# same compact tier (Greek band symbols) as
# make_fig_ws_loso_sign_forest.py's MARKER_LABELS, per CLAUDE.md "Marker
# Naming": both files must use the same stem word per family, so this dict
# only differs from that one by trimming entries not present in the 23-marker
# CBPT feature space (psd_bands_*, per_channel_*, spindles_*).
_FEATURE_LABEL_MARKERS: dict[str, str] = {
    "kolmogorov_complexity": "KoC",
    "psd_relative_alpha": "PSD α",
    "psd_relative_beta": "PSD β",
    "psd_relative_gamma": "PSD γ",
    "psd_relative_delta": "PSD δ",
    "psd_relative_theta": "PSD θ",
    "slowwaves_Density": "SW density",
    "slowwaves_Duration": "SW duration",
    "slowwaves_Frequency": "SW frequency",
    "slowwaves_PTP": "SW PTP",
    "slowwaves_Slope": "SW slope",
    "wsmi_alpha": "wSMI α",
    "wsmi_beta": "wSMI β",
    "wsmi_gamma": "wSMI γ",
    "wsmi_theta": "wSMI θ",
}
_FEATURE_LABEL_GREEK = {"theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ"}
_FEATURE_LABEL_REGION = {"frontal": "Front", "central": "Cent", "posterior": "Post"}
_FEATURE_LABEL_SIDE = {"left": "L", "right": "R", "mid": "M"}


def _pretty_feature_label(feature: str) -> str:
    """
    Full marker×ROI label for a point in the WS-vs-LOSO scatter panels.

    ``f.split("_")[0]`` (the previous implementation) kept only the family
    stem — e.g. every PSD band and every slow-wave statistic at every ROI all
    rendered as the bare word "psd" / "slowwaves", indistinguishable from one
    another once several such points cluster in one panel. This instead
    reproduces the full ``marker · ROI`` label (e.g. "PSD γ · Front R"),
    matching make_fig_ws_loso_sign_forest.py's pretty_feature_label so the two
    figures name the same feature the same way.

    'psd_relative_gamma_mean_frontal_right_trimmean' -> 'PSD γ · Front R'
    """
    marker, roi = feature.split("_mean_", 1)
    if marker in _FEATURE_LABEL_MARKERS:
        marker_label = _FEATURE_LABEL_MARKERS[marker]
    else:
        marker_label = marker
        for band, symbol in _FEATURE_LABEL_GREEK.items():
            if marker.endswith(f"_{band}"):
                marker_label = f"{marker[: -(len(band) + 1)]} {symbol}"
                break
    roi_parts = roi.replace("_trimmean", "").split("_")
    region = _FEATURE_LABEL_REGION.get(roi_parts[0], roi_parts[0].title())
    side = _FEATURE_LABEL_SIDE.get(roi_parts[1], roi_parts[1].title()) if len(roi_parts) > 1 else ""
    roi_label = f"{region} {side}".strip()
    return f"{marker_label} · {roi_label}"


# =============================================================================
# Figure — WS-vs-LOSO SHAP scatter, aggregated to the 23 CBPT markers
# =============================================================================
# Built from the same _load_shap_summary data (the true_runs SHAP pkls) as
# _build_shap_scatter_figs above, not from feature_consistency_analysis.py's
# separate per-subject pipeline — this keeps the marker-level view inside the
# same combined-figure generator instead of depending on another script's
# output tables. It answers a coarser-grained version of the same question:
# _build_shap_scatter_figs shows whether individual marker×ROI columns agree
# between pipelines (175 points), this collapses each marker's ROI columns to
# one point (mean, never sum, per CLAUDE.md "Marker Naming" — the 4 evoked
# markers own one ROI column and the 19 sleep markers own up to nine, so
# summing would hand sleep markers ~9x the mass before any data is seen).
_MARKER_SCATTER_LABEL_N = 8


def _build_shap_marker_scatter_fig() -> go.Figure:
    """
    One panel per dimension: WS vs LOSO mean |SHAP|, one point per CBPT marker.

    Styled like _residual_panel (dashed diagonal spanning the data with a 10%
    pad, Spearman rho over all markers, top-N-union labelled) with the full-vs-
    residualized axes swapped for WS-vs-LOSO — the same comparison as
    _build_shap_scatter_figs, one resolution coarser.
    """
    fig = _new_dimension_grid_fig()
    # Column titles enlarged to match _residual_panel's sibling figure
    # (residual_vs_plain_markers.png) — this and that one are read side by
    # side, so the same font scale keeps them equally legible.
    for i in range(len(DIMENSIONS)):
        fig.layout.annotations[i].font.size = 20

    for col_idx, dim_info in enumerate(DIMENSIONS):
        col = col_idx + 1
        print(f"  Loading marker-level SHAP [{dim_info['label']}]…", flush=True)

        ws_abs, _, ws_feats = _load_shap_summary("ws", dim_info)
        lo_abs, _, lo_feats = _load_shap_summary("loso", dim_info)
        if len(ws_abs) == 0 or len(lo_abs) == 0:
            continue

        x_idx, y_idx, common = _common_feature_indices(ws_feats, lo_feats)
        if not common:
            continue

        marker_of = {f: f.split("_mean_", 1)[0] for f in common}
        markers = sorted(set(marker_of.values()))
        ws_marker = np.array([
            np.mean([ws_abs[x_idx[f]] for f in common if marker_of[f] == marker])
            for marker in markers
        ])
        lo_marker = np.array([
            np.mean([lo_abs[y_idx[f]] for f in common if marker_of[f] == marker])
            for marker in markers
        ])

        _add_marker_scatter_panel(fig, col, dim_info, ws_marker, lo_marker, markers)

    # No figure-level title — the column titles plus each panel's own axis
    # labels and rho annotation already carry the comparison, matching
    # residual_vs_plain_markers.png's untitled layout.
    fig.update_layout(
        template="plotly_white",
        height=500,
        width=420 * len(DIMENSIONS),
        margin=dict(l=90, r=40, t=55, b=85),
        font=dict(family="Times New Roman", size=13),
    )
    return fig


def _add_marker_scatter_panel(
    fig: go.Figure, col: int, dim_info: dict,
    ws_marker: np.ndarray, lo_marker: np.ndarray, markers: list[str],
) -> None:
    """Add one WS-vs-LOSO marker-level scatter panel to ``fig`` at column ``col``."""
    from scipy.stats import spearmanr

    color = dim_info["color"]
    names = [_pretty_marker_name(m) for m in markers]
    rho, pval = spearmanr(ws_marker, lo_marker)

    # Axes span the data, not [0, max] — anchoring at zero pushes a positive-
    # valued cloud into one corner where any spread reads as a tight diagonal
    # band regardless of the actual correlation (see _residual_panel above,
    # which needed the same fix for the same reason).
    low = float(min(ws_marker.min(), lo_marker.min()))
    high = float(max(ws_marker.max(), lo_marker.max()))
    pad = 0.10 * (high - low)
    axis_lo, axis_hi = low - pad, high + pad

    x_ref = f"x{col}" if col > 1 else "x"
    y_ref = f"y{col}" if col > 1 else "y"
    fig.add_shape(
        type="line", x0=axis_lo, y0=axis_lo, x1=axis_hi, y1=axis_hi,
        xref=x_ref, yref=y_ref,
        line=dict(color=PERM_COLOR, width=0.9, dash="dot"), layer="below",
    )

    # Ranked by the mean of the two pipelines, not the max — a marker one
    # pipeline leans on and the other ignores would otherwise outrank one both
    # use steadily (same rationale as make_fig_feature_consistency.py's
    # build_marker_scatter_figure, which this panel mirrors at the group level).
    rank = (ws_marker + lo_marker) / 2.0
    top_idx = set(np.argsort(rank)[-_MARKER_SCATTER_LABEL_N:])
    mask = np.array([i in top_idx for i in range(len(markers))])

    fig.add_trace(go.Scatter(
        x=ws_marker[~mask], y=lo_marker[~mask], mode="markers",
        marker=dict(color=color, size=6, opacity=0.45, line=dict(color="white", width=0.4)),
        customdata=[n for n, m in zip(names, mask) if not m],
        hovertemplate="%{customdata}<br>WS %{x:.4f}<br>LOSO %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=col)

    # Spread labels left/right of their point (3-way top/middle/bottom cycle),
    # anchored inward near the axis edges so a label never pushes past the
    # panel boundary — same technique as _add_scatter_panel above.
    x_hi, y_hi = ws_marker[mask], lo_marker[mask]
    x_mid = float(np.median(x_hi))
    order = np.argsort(np.argsort(y_hi))
    left_cycle = np.array(["middle left", "top left", "bottom left"])
    right_cycle = np.array(["middle right", "top right", "bottom right"])
    goes_left = x_hi < x_mid
    edge_margin = 0.08 * (axis_hi - axis_lo)
    goes_left = np.where(x_hi > axis_hi - edge_margin, True, goes_left)
    goes_left = np.where(x_hi < axis_lo + edge_margin, False, goes_left)
    label_positions = np.where(goes_left, left_cycle[order % 3], right_cycle[order % 3])

    fig.add_trace(go.Scatter(
        x=x_hi, y=y_hi, mode="markers+text",
        marker=dict(color=color, size=8, opacity=0.9, line=dict(color="white", width=0.5)),
        text=[n for n, m in zip(names, mask) if m],
        textposition=label_positions, textfont=dict(size=11, family="Times New Roman"),
        cliponaxis=False,
        hovertemplate="%{text}<br>WS %{x:.4f}<br>LOSO %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=col)

    # Bottom-right rather than top-left: the labelled set skews toward high
    # WS-and-LOSO markers (ranked by their mean), so their text routinely
    # lands in the top-left corner and collided with the annotation there.
    fig.add_annotation(
        text=f"ρ = {rho:.2f}<br>{_format_p_compact(pval)}",
        xref=f"{x_ref} domain", yref=f"{y_ref} domain",
        x=0.95, y=0.05, xanchor="right", yanchor="bottom", showarrow=False,
        font=dict(size=17, color=color, family="Times New Roman"),
    )

    x_ax = f"xaxis{col}" if col > 1 else "xaxis"
    y_ax = f"yaxis{col}" if col > 1 else "yaxis"
    fig.layout[x_ax].range = [axis_lo, axis_hi]
    fig.layout[y_ax].range = [axis_lo, axis_hi]
    for ax_name in (x_ax, y_ax):
        fig.layout[ax_name].showgrid  = True
        fig.layout[ax_name].gridcolor = "#EEEEEE"
        fig.layout[ax_name].tickfont  = dict(size=16)
    fig.layout[x_ax].title = dict(text="WS mean |SHAP| (marker)", font=dict(size=15))
    if col == 1:
        fig.layout[y_ax].title = dict(text="LOSO mean |SHAP| (marker)", font=dict(size=15))


_MARKER_SCATTER_SINGLE_MM = (95.0, 90.0)


def _build_shap_marker_single_figures() -> list[tuple[go.Figure, str]]:
    """
    One standalone WS-vs-LOSO marker-SHAP figure per dimension.

    _build_shap_marker_scatter_fig above puts all five dimensions in one row
    at 420px per panel; fine as a five-dimension survey, tight for reading
    every marker label. These are the same panels at a size comfortable to
    read, one file per dimension — same naming convention as
    _build_residual_single_figures's "residual_vs_plain_{key}"
    ("scatter_shap_marker_{key}").

    Returns
    -------
    list[tuple[go.Figure, str]]
        ``(figure, output stem)`` pairs, skipping any dimension whose SHAP
        pkls are absent (matching _build_shap_marker_scatter_fig's per-column
        skip behaviour).
    """
    figures: list[tuple[go.Figure, str]] = []

    for dim_info in DIMENSIONS:
        print(f"  Loading marker-level SHAP (standalone) [{dim_info['label']}]…", flush=True)

        ws_abs, _, ws_feats = _load_shap_summary("ws", dim_info)
        lo_abs, _, lo_feats = _load_shap_summary("loso", dim_info)
        if len(ws_abs) == 0 or len(lo_abs) == 0:
            continue

        x_idx, y_idx, common = _common_feature_indices(ws_feats, lo_feats)
        if not common:
            continue

        marker_of = {f: f.split("_mean_", 1)[0] for f in common}
        markers = sorted(set(marker_of.values()))
        ws_marker = np.array([
            np.mean([ws_abs[x_idx[f]] for f in common if marker_of[f] == marker])
            for marker in markers
        ])
        lo_marker = np.array([
            np.mean([lo_abs[y_idx[f]] for f in common if marker_of[f] == marker])
            for marker in markers
        ])

        fig = make_subplots(rows=1, cols=1)
        _add_marker_scatter_panel(fig, 1, dim_info, ws_marker, lo_marker, markers)
        # _add_marker_scatter_panel sizes its axis text for the five-panel
        # combined figure (420 px/panel); rescaled down here for this
        # standalone panel's smaller 95 mm canvas, same as
        # _build_residual_single_figures does after calling _residual_panel.
        fig.update_xaxes(tickfont=dict(size=10), title_font=dict(size=11), row=1, col=1)
        fig.update_yaxes(tickfont=dict(size=10), title_font=dict(size=11), row=1, col=1)
        # The rho annotation's y=0.05 (domain fraction) put it comfortably
        # above the x-axis on the five-panel combined figure's taller canvas;
        # on this panel's shorter one it sat right down against the tick
        # labels. Raised rather than shrunk, to keep it as legible as the
        # combined figure's.
        fig.layout.annotations[-1].y = 0.13
        fig.update_layout(
            title=dict(
                # Two lines, not one long string — a single line at a legible
                # size ran past the panel's 95 mm width and got clipped by the
                # canvas edge (the dimension<br><sub>...</sub> pattern already
                # used for the column titles elsewhere in this file).
                text=f"{dim_info['label']}<br><sub>WS vs LOSO mean |SHAP| (marker)</sub>",
                font=dict(size=15, family="Times New Roman", color=dim_info["color"]),
            ),
            template="plotly_white",
            width=int(_MARKER_SCATTER_SINGLE_MM[0] * _MM_TO_PX),
            height=int(_MARKER_SCATTER_SINGLE_MM[1] * _MM_TO_PX),
            margin=dict(l=70, r=30, t=70, b=55),
            font=dict(family="Times New Roman", size=12),
        )
        figures.append((fig, f"scatter_shap_marker_{dim_info['key']}"))

    return figures


# =============================================================================
# Figure — plain vs residualized: does residualizing change which markers matter?
# =============================================================================

_RESIDUAL_PROFILES = CONSISTENCY_DIR / "residual_marker_profiles.csv"
_RESIDUAL_TOP_N = 5
_RESIDUAL_WIDTH_MM, _RESIDUAL_HEIGHT_MM = 380.0, 190.0
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
    rho, pval = spearmanr(full, residual)

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
        textposition=positions, textfont=dict(size=11, color="#333333"),
        cliponaxis=False,
        hovertemplate="%{text}<br>full %{x:.4f}<br>residualized %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    fig.add_annotation(
        text=f"Spearman ρ = {rho:.2f}<br>{_format_p_compact(pval)}",
        xref="x domain", yref="y domain", x=0.03, y=0.97,
        xanchor="left", yanchor="top", showarrow=False,
        font=dict(size=17, color=color), align="left", row=row, col=col,
    )
    fig.update_xaxes(range=[axis_lo, axis_hi], nticks=4, tickfont=dict(size=16),
                     row=row, col=col)
    fig.update_yaxes(range=[axis_lo, axis_hi], nticks=4, tickfont=dict(size=16),
                     row=row, col=col)


def _residual_color(contrast_key: str) -> str:
    """Palette colour for a dimension, falling back to the neutral accent."""
    return DIM_COLORS.get(
        {"on_off": "onoff"}.get(contrast_key, contrast_key),
        PALETTE["neutral"]["accent"],
    )


def _residual_dimension_order(profiles: pd.DataFrame) -> list[str]:
    """
    Dimension order for the residual figures: the one this script already uses.

    ``GROUP_ROW_DIMENSIONS`` drives the other group-level figures here
    (the "+ residualized" group-level and spatial-comparison panels), so the
    residual panels follow it too rather than the order they happen to appear
    in ``config_feature_consistency.yaml`` — a reader moving between figures
    should not have to re-locate a dimension. Since the quadratic terms were
    removed (2026-08-13) that list is simply the canonical 5.

    Any contrast present in ``profiles`` but absent from
    ``GROUP_ROW_DIMENSIONS`` is appended at the end rather than dropped, so a
    stale profiles table is visible in the figure instead of silently
    half-filtered. Re-run ``scripts/compute_residual_profiles.py`` if
    valence_sq / time_sq ever reappear there.

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

    Two rows, one per pipeline (within-subject on top, LOSO below), one column
    per dimension. Dimension name is a column header shown once (top row only);
    the pipeline name is a row-spanning supra-title shown once per row instead
    of being repeated inside every panel's own title — a reader lines a column
    up against the header once, not ten times.

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
    n_cols = len(dimensions)
    pipelines = [("ws", "Within-Subject"), ("loso", "LOSO")]

    subplot_titles = [
        profiles.loc[profiles["contrast"] == key, "label"].iloc[0] if row_idx == 0 else ""
        for row_idx in range(2) for key in dimensions
    ]
    fig = make_subplots(
        rows=2, cols=n_cols,
        horizontal_spacing=0.045, vertical_spacing=0.16,
        subplot_titles=subplot_titles,
    )
    for col_idx, key in enumerate(dimensions):
        fig.layout.annotations[col_idx].font.size = 20
        fig.layout.annotations[col_idx].font.color = _residual_color(key)

    for row_idx, (pipeline_key, _) in enumerate(pipelines):
        row = row_idx + 1
        for col_idx, key in enumerate(dimensions):
            col = col_idx + 1
            block = profiles[(profiles["contrast"] == key)
                             & (profiles["pipeline"] == pipeline_key)]
            _residual_panel(fig, row, col, block, _residual_color(key))
            if row == 2:
                fig.update_xaxes(title_text="full model — mean |SHAP|",
                                 title_font=dict(size=15), row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text="residualized model",
                                 title_font=dict(size=15), row=row, col=col)

    # Row-spanning supra-titles: "Within-Subject" above row 1, "LOSO" centred
    # in the gap between the two rows — the pipeline name shown once per row
    # rather than baked into every panel title (see docstring).
    row1_domain = fig.layout["yaxis"].domain
    row2_axis = "yaxis" if n_cols == 1 else f"yaxis{n_cols + 1}"
    row2_domain = fig.layout[row2_axis].domain
    supra_font = dict(size=21, color="#333333")
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=1.05,
                        xanchor="center", yanchor="bottom", showarrow=False,
                        text=f"<b>{pipelines[0][1]}</b>", font=supra_font)
    fig.add_annotation(xref="paper", yref="paper", x=0.5,
                        y=(row1_domain[0] + row2_domain[1]) / 2,
                        xanchor="center", yanchor="middle", showarrow=False,
                        text=f"<b>{pipelines[1][1]}</b>", font=supra_font)

    fig.update_layout(
        template="plotly_white",
        width=int(_RESIDUAL_WIDTH_MM * _MM_TO_PX),
        height=int(_RESIDUAL_HEIGHT_MM * _MM_TO_PX),
        margin=dict(l=110, r=95, t=68, b=75),
        font=dict(size=15),
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


#: Physical canvas for plot_proba_ws_vs_loso_scatter — a double-column-width
#: page figure, sized in mm like every other figure a paper actually uses
#: (see scientific-plots skill, references/learnings/multipanel-layout.md).
#: The previous version sized this canvas in raw pixels
#: (`350 * n_dims` x `430`), which at the 96 dpi Plotly/Kaleido assume is
#: ~463 mm wide — 2.5x the printable page — so every font constant on it was
#: chosen as if for a screen-sized canvas and read as tiny once the PNG was
#: actually placed at 180 mm. Margins below are hand-set (not automargin,
#: matching this file's other mm-sized figures, e.g. _build_marker_direction_figure)
#: because five equal-width panels sharing one y-axis need a predictable,
#: not auto-negotiated, left/right split.
_PROBA_SCATTER_WIDTH_MM  = 180.0
_PROBA_SCATTER_HEIGHT_MM = 68.0
_PROBA_SCATTER_MARGIN_MM = dict(l=15.0, r=27.0, t=29.0, b=14.0)
_PROBA_SCATTER_GUTTER_MM = 3.0

#: Type sizes in points at the figure's final print size (converted to the
#: px Plotly expects via _MM_TO_PX-equivalent pt->px, see below) — the house
#: floor from the scientific-plots skill (10 pt axis titles / 9 pt ticks),
#: not the smaller sizes this file's other panels sometimes use.
_PROBA_SCATTER_PT = dict(
    fig_title=13.0, panel_title=12.0, panel_pole=9.5,
    stat_annotation=10.5, axis_title=10.0, tick=9.0, legend=9.0,
)
_PT_TO_PX = 96.0 / 72.0  # points at final size -> the px Plotly/Kaleido lay out in


def plot_proba_ws_vs_loso_scatter() -> tuple[go.Figure, int, int]:
    """
    One panel per dimension: scatter of per-probe predicted probability (WS vs LOSO).

    Each point is one probe. Fill encodes true label per project convention:
      - filled marker  → y_true = 1 (high end of the dimension's 0-100 scale)
      - hollow marker  → y_true = 0 (low end of the dimension's 0-100 scale)
    Diagonal y=x and chance lines at 0.5 drawn for reference. The legend
    ("High end (y=1)" / "Low end (y=0)") is deliberately generic — it is one
    legend shared across all 5 panels, and each dimension's actual pole word
    (on-task, positive, ...) differs, so the legend can't literally name the
    pole and stays true to just the fill encoding; the per-panel subtitle
    gives the dimension-specific pole wording instead (CLAUDE.md "Dimension
    Labels & Pole Wording").

    Only the leftmost panel carries the shared 0-1 probability y-axis (title
    + tick labels); the other four keep the gridlines only, so a value still
    lines up visually across panels without five redundant copies of
    "0, .25, .5, .75, 1" eating into the plot area (see
    references/learnings/multipanel-layout.md in the scientific-plots skill
    — a repeated shared axis is the other common source of a facet row
    reading as cramped even though nothing is actually missing).

    Returns
    -------
    (figure, width_px, height_px)
        Matches the ``plot_individual``-style return so the caller can queue
        it with its own physical size and a 600 dpi export scale instead of
        the shared ``_queue()`` helper's screen-preview scale.
    """
    n_dims = len(DIMENSIONS)
    pt = _PROBA_SCATTER_PT

    width_mm, height_mm = _PROBA_SCATTER_WIDTH_MM, _PROBA_SCATTER_HEIGHT_MM
    margin_mm = _PROBA_SCATTER_MARGIN_MM
    plot_width_mm = width_mm - margin_mm["l"] - margin_mm["r"]
    gutters_mm = _PROBA_SCATTER_GUTTER_MM * (n_dims - 1)
    h_spacing = _PROBA_SCATTER_GUTTER_MM / plot_width_mm

    fig = make_subplots(
        rows=1, cols=n_dims,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=h_spacing,
        subplot_titles=None,  # built by hand below: two lines, two sizes, no HTML sub-scaling guess
    )

    for i, dim_info in enumerate(DIMENSIONS):
        col = i + 1
        xa = fig.layout[f"xaxis{col}" if col > 1 else "xaxis"]
        x_center = sum(xa.domain) / 2
        fig.add_annotation(
            text=f"<b>{dim_info['label']}</b>", x=x_center, y=1.0,
            xref="x domain" if col == 1 else f"x{col} domain", yref="paper",
            xanchor="center", yanchor="bottom", yshift=38,
            showarrow=False,
            font=dict(size=pt["panel_title"] * _PT_TO_PX, color=dim_info["color"],
                      family="Times New Roman"),
        )
        # Two lines, not one: the widest pole pair ("self-focused" /
        # "other-focused") does not fit one column's ~33 mm width on one
        # line at a legible size — wrapping (not abbreviating; CLAUDE.md
        # "Dimension Labels & Pole Wording" allows line-wrap but not a
        # shorter synonym) keeps every pole at full canonical wording.
        fig.add_annotation(
            text=f"{dim_info['pole_low']}<br>↔ {dim_info['pole_high']}",
            x=x_center, y=1.0,
            xref="x domain" if col == 1 else f"x{col} domain", yref="paper",
            xanchor="center", yanchor="top", yshift=32,
            showarrow=False,
            font=dict(size=pt["panel_pole"] * _PT_TO_PX, color=dim_info["color"],
                      family="Times New Roman"),
        )

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
            (1, "circle",      "High end (y=1)"),
            (0, "circle-open", "Low end (y=0)"),
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

        # Regression line + Pearson r / p-value annotation
        if len(merged) >= 3:
            m, b = np.polyfit(merged["ws_proba"].values, merged["loso_proba"].values, 1)
            x_reg = np.array([proba_min, proba_max])
            y_reg = m * x_reg + b
            fig.add_trace(go.Scatter(
                x=x_reg, y=y_reg,
                mode="lines",
                line=dict(color=color, width=2, dash="solid"),
                showlegend=False,
            ), row=1, col=col)

            r, p = pearsonr(merged["ws_proba"].values, merged["loso_proba"].values)
            fig.add_annotation(
                text=f"r={r:.2f}<br>{_format_p_compact(p)}",
                xref=f"x{col} domain" if col > 1 else "x domain",
                yref=f"y{col} domain" if col > 1 else "y domain",
                x=0.05, y=0.97,
                xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=pt["stat_annotation"] * _PT_TO_PX, color=color,
                          family="Times New Roman"),
            )

        x_ax = f"xaxis{col}" if col > 1 else "xaxis"
        y_ax = f"yaxis{col}" if col > 1 else "yaxis"
        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range     = [proba_min, proba_max]
            fig.layout[ax_name].tickmode  = "array"
            fig.layout[ax_name].tickvals  = [0.0, 0.25, 0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext  = ["0", ".25", ".5", ".75", "1"]
            fig.layout[ax_name].tickfont  = dict(size=pt["tick"] * _PT_TO_PX)
            fig.layout[ax_name].showgrid  = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline  = False

        # Shared 0-1 scale across all 5 panels: only the leftmost panel keeps
        # the y tick labels + title (gridlines stay on every panel so a value
        # still lines up visually across the row) — see
        # references/learnings/multipanel-layout.md in the scientific-plots
        # skill. The x title is dropped per-panel too, in favour of one
        # centered annotation below the whole row (added after this loop),
        # since all 5 columns share the identical "WS predicted probability"
        # text.
        if col == 1:
            fig.layout[y_ax].title = dict(
                text="LOSO predicted probability",
                font=dict(size=pt["axis_title"] * _PT_TO_PX),
            )
        else:
            fig.layout[y_ax].showticklabels = False

    x_domain_lo = fig.layout.xaxis.domain[0]
    x_domain_hi = fig.layout[f"xaxis{n_dims}"].domain[1]
    fig.add_annotation(
        text="WS predicted probability",
        x=(x_domain_lo + x_domain_hi) / 2, y=0,
        xref="paper", yref="paper",
        xanchor="center", yanchor="top", yshift=-32,
        showarrow=False,
        font=dict(size=pt["axis_title"] * _PT_TO_PX, family="Times New Roman"),
    )

    width_px  = round(width_mm * _MM_TO_PX)
    height_px = round(height_mm * _MM_TO_PX)
    fig.update_layout(
        title=dict(
            text="Per-probe predicted probability: Within-Subject vs LOSO",
            font=dict(size=pt["fig_title"] * _PT_TO_PX, family="Times New Roman"),
            x=0.5, y=0.99, yanchor="top",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Times New Roman", size=pt["tick"] * _PT_TO_PX),
        width=width_px,
        height=height_px,
        legend=dict(
            orientation="v",
            x=1.0, y=0.5,
            xanchor="left", yanchor="middle",
            font=dict(size=pt["legend"] * _PT_TO_PX, family="Times New Roman"),
            title=dict(text="True label", font=dict(size=pt["legend"] * _PT_TO_PX)),
        ),
        margin=dict(
            l=round(margin_mm["l"] * _MM_TO_PX), r=round(margin_mm["r"] * _MM_TO_PX),
            t=round(margin_mm["t"] * _MM_TO_PX), b=round(margin_mm["b"] * _MM_TO_PX),
        ),
    )

    return fig, width_px, height_px


# =============================================================================
# Figure D3 — CV-scheme stability: performance (A) + probability agreement (B)
# =============================================================================


def _subplot_axis_ids(row: int, col: int, n_cols: int) -> tuple[str, str, str, str]:
    """
    ``(xaxis_key, yaxis_key, xref, yref)`` for a ``make_subplots`` cell.

    ``plot_loso_vs_ws_scatter``/``plot_proba_ws_vs_loso_scatter`` each only
    ever address row 1, so their inline ``f"xaxis{col}"`` is correct as
    written; this combined figure stacks two rows, so the same cell needs a
    row-major index (``(row-1)*n_cols+col``) instead of just ``col``.
    """
    idx = (row - 1) * n_cols + col
    suffix = "" if idx == 1 else str(idx)
    return f"xaxis{suffix}", f"yaxis{suffix}", f"x{suffix}", f"y{suffix}"


def _add_diag_and_chance_lines(fig: go.Figure, xref: str, yref: str, lo: float, hi: float) -> None:
    """Diagonal (y=x) + chance crosshair (0.5) shared by both rows of the CV-stability figure."""
    fig.add_shape(
        type="line", x0=lo, x1=hi, y0=lo, y1=hi,
        xref=xref, yref=yref,
        line=dict(color="#888888", dash="dot", width=1.2),
    )
    for x0, x1, y0, y1 in [(0.5, 0.5, lo, hi), (lo, hi, 0.5, 0.5)]:
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y0, y1=y1,
            xref=xref, yref=yref,
            line=dict(color="#BBBBBB", dash="dash", width=1.0),
        )


def _add_regression_and_annotation(
    fig: go.Figure, row: int, col: int, xref: str, yref: str,
    x: np.ndarray, y: np.ndarray, lo: float, hi: float, color: str,
) -> None:
    """Fit line + Pearson r/p annotation, shared by both rows of the CV-stability figure."""
    m, b = np.polyfit(x, y, 1)
    x_reg = np.array([lo, hi])
    y_reg = m * x_reg + b
    fig.add_trace(go.Scatter(
        x=x_reg, y=y_reg, mode="lines",
        line=dict(color=color, width=2, dash="solid"),
        showlegend=False,
    ), row=row, col=col)

    r, p = pearsonr(x, y)
    fig.add_annotation(
        text=f"r={r:.2f}<br>{_format_p_compact(p)}",
        xref=f"{xref} domain", yref=f"{yref} domain",
        x=0.05, y=0.96, xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=16, color=color, family="Times New Roman"),
    )


def plot_cv_stability_combined() -> go.Figure:
    """
    Two-panel figure: does WS agree with LOSO, at both the performance level
    and the prediction level?

    Panel A (row 1) — per-subject mean AUC, WS vs LOSO (same data as
    ``plot_loso_vs_ws_scatter``): does a dimension's decoding *performance*
    replicate across CV schemes?
    Panel B (row 2) — per-probe predicted probability, WS vs LOSO (same data
    as ``plot_proba_ws_vs_loso_scatter``): do the two schemes agree on
    individual predictions, not just on the summary AUC? A dimension can
    tie on A while disagreeing probe-by-probe on B, or vice versa, so the two
    rows are deliberately kept as one figure rather than two standalone ones.

    One column per canonical dimension, shared across both rows. Letters
    follow the project convention (CLAUDE.md "Figure Assembly"): one per
    distinct *kind* of panel, at that row's first column, top-left corner —
    not one per column, since all five columns within a row are the same
    kind of panel repeated.
    """
    n_dims = len(DIMENSIONS)

    fig = make_subplots(
        rows=2, cols=n_dims,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.045,
        vertical_spacing=0.30,
        subplot_titles=[d["label"] for d in DIMENSIONS] + [""] * n_dims,
    )

    for i, dim_info in enumerate(DIMENSIONS):
        fig.layout.annotations[i].text = (
            f"<b>{dim_info['label']}</b><br>"
            f"<sub>{dim_info['pole_low']} ↔ {dim_info['pole_high']}</sub>"
        )
        fig.layout.annotations[i].font.color  = dim_info["color"]
        fig.layout.annotations[i].font.size   = 22
        fig.layout.annotations[i].font.family = "Times New Roman"

    auc_min, auc_max = AUC_RANGE
    proba_min, proba_max = 0.0, 1.0

    # ---- Row 1 (Panel A): per-subject mean AUC ----
    for col_idx, dim_info in enumerate(DIMENSIONS):
        col, color = col_idx + 1, dim_info["color"]
        x_ax, y_ax, xref, yref = _subplot_axis_ids(1, col, n_dims)

        t_ws, _   = load_subject_data("ws",   dim_info)
        t_lo, _   = load_subject_data("loso", dim_info)
        ws_means   = t_ws.groupby("subject")["auc"].mean().rename("ws")
        loso_means = t_lo.groupby("subject")["auc"].mean().rename("loso")
        merged = pd.concat([ws_means, loso_means], axis=1).dropna()
        if merged.empty:
            continue

        fig.add_trace(go.Scatter(
            x=merged["ws"].values, y=merged["loso"].values,
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.85,
                        line=dict(color="white", width=0.5)),
            showlegend=False, name=dim_info["label"],
        ), row=1, col=col)

        _add_diag_and_chance_lines(fig, xref, yref, auc_min, auc_max)
        if len(merged) >= 3:
            _add_regression_and_annotation(
                fig, 1, col, xref, yref,
                merged["ws"].values, merged["loso"].values, auc_min, auc_max, color,
            )

        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range     = [auc_min, auc_max]
            fig.layout[ax_name].tickmode  = "array"
            fig.layout[ax_name].tickvals  = [0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext  = ["0.5", ".75", "1.0"]
            fig.layout[ax_name].tickfont  = dict(size=12)
            fig.layout[ax_name].showgrid  = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline  = False
        fig.layout[x_ax].title = dict(text="WS AUC", font=dict(size=13))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO AUC", font=dict(size=13))

    # ---- Row 2 (Panel B): per-probe predicted probability ----
    for col_idx, dim_info in enumerate(DIMENSIONS):
        col, color = col_idx + 1, dim_info["color"]
        x_ax, y_ax, xref, yref = _subplot_axis_ids(2, col, n_dims)

        ws_df   = _load_probe_predictions_ws(dim_info)
        loso_df = _load_probe_predictions_loso(dim_info)
        if ws_df.empty or loso_df.empty:
            continue

        merged = ws_df.merge(loso_df, on=["subject", "task", "probe_number"], suffixes=("_ws", "_loso"))
        merged = merged.dropna(subset=["ws_proba", "loso_proba"])
        if merged.empty:
            print(f"  ! {dim_info['label']}: WS/LOSO probe tables share no "
                  f"(subject, task, probe_number) key — panel B will be empty for this column.")
            continue

        y_true = merged["y_true_ws"] if "y_true_ws" in merged.columns else merged["y_true"]

        for label_val, symbol, marker_label in [
            (1, "circle",      "High end (y=1)"),
            (0, "circle-open", "Low end (y=0)"),
        ]:
            mask = y_true == label_val
            fig.add_trace(go.Scatter(
                x=merged.loc[mask, "ws_proba"].values,
                y=merged.loc[mask, "loso_proba"].values,
                mode="markers",
                name=marker_label,
                marker=dict(symbol=symbol, color=color, size=4.5, opacity=0.6,
                            line=dict(color=color, width=1.1)),
                showlegend=(col_idx == 0),
                legendgroup=marker_label,
            ), row=2, col=col)

        _add_diag_and_chance_lines(fig, xref, yref, proba_min, proba_max)
        if len(merged) >= 3:
            _add_regression_and_annotation(
                fig, 2, col, xref, yref,
                merged["ws_proba"].values, merged["loso_proba"].values, proba_min, proba_max, color,
            )

        for ax_name in (x_ax, y_ax):
            fig.layout[ax_name].range     = [proba_min, proba_max]
            fig.layout[ax_name].tickmode  = "array"
            fig.layout[ax_name].tickvals  = [0.0, 0.25, 0.5, 0.75, 1.0]
            fig.layout[ax_name].ticktext  = ["0", ".25", ".5", ".75", "1"]
            fig.layout[ax_name].tickfont  = dict(size=12)
            fig.layout[ax_name].showgrid  = True
            fig.layout[ax_name].gridcolor = "#EEEEEE"
            fig.layout[ax_name].zeroline  = False
        fig.layout[x_ax].title = dict(text="WS probability", font=dict(size=13))
        if col == 1:
            fig.layout[y_ax].title = dict(text="LOSO probability", font=dict(size=13))

    # Panel letters (top-left corner, per project convention — one per row
    # since all 5 columns are the same kind of panel repeated) + a centered
    # per-panel title for each row. No figure-wide title.
    #
    # Positioning uses `yshift` (fixed pixels) rather than a fractional `y`
    # offset: the column-title annotations above row 1 (dimension name + pole
    # subtitle, 2 lines) have a pixel-sized vertical extent that a fractional
    # guess kept colliding with. A pixel offset is measured from each row's
    # own domain edge and stays correct regardless of that text's height.
    y1_top = fig.layout.yaxis.domain[1]
    _, y2_ax, _, _ = _subplot_axis_ids(2, 1, n_dims)
    y2_top = fig.layout[y2_ax].domain[1]
    x1_left = fig.layout.xaxis.domain[0]

    header_specs = [
        ("A", "Performance",   y1_top, 95),  # clears the 2-line column titles above it
        ("B", "Probabilities", y2_top, 95),  # clears the legend sitting below it (yshift=25)
    ]
    for letter, panel_title, y_base, yshift in header_specs:
        fig.add_annotation(
            text=f"<b>{letter}</b>",
            xref="paper", yref="paper",
            x=x1_left - 0.03, y=y_base, yshift=yshift,
            xanchor="right", yanchor="bottom",
            showarrow=False,
            font=dict(size=30, family="Times New Roman", color="black"),
        )
        fig.add_annotation(
            text=f"<b>{panel_title}</b>",
            xref="paper", yref="paper",
            x=0.5, y=y_base, yshift=yshift,
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(size=18, family="Times New Roman", color="#333333"),
        )

    # Legend: horizontal, centered, inside the figure in the row1/row2 gap —
    # not off to the right (which only pads the canvas). Sits on its own line
    # below panel B's "B  Probabilities" header, directly above panel B's
    # plots. `layout.legend` has no `yshift` (annotations do), so the same
    # ~25px offset used for the header is pre-converted to a plot-area
    # fraction here (25 / (height - margin.t - margin.b) = 25/630).
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=320 * n_dims,
        height=840,
        margin=dict(l=130, r=30, t=150, b=60),
        font=dict(family="Times New Roman", size=15),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=y2_top + 25 / 630, yanchor="bottom",
            font=dict(size=13, family="Times New Roman"),
            title=dict(text="True label:  ", font=dict(size=13), side="left"),
            bgcolor="rgba(255,255,255,0)",
        ),
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # NOT wiped before regenerating: OUTPUT_DIR is also the parent of
    # prob_vs_dim/, owned by make_fig_prob_vs_dim_ws_loso.py — an
    # rmtree here silently deletes that script's output too. Each stem this
    # script writes overwrites its own file in place; a stem removed
    # upstream can leave an orphan behind, but that is the safer failure mode.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Accumulate (fig, path, opts) — written in one kaleido/Chrome session at the end
    # so Chrome launches only once instead of once per write_image() call.
    pending: list[tuple[go.Figure, Path, dict]] = []

    def _queue(fig: go.Figure, stem: str) -> None:
        _export_fig_csv(fig, OUTPUT_DIR / f"{stem}.csv")
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
            _export_fig_csv(fig_i, OUTPUT_DIR / f"{indiv_stem}.csv")
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
            _export_fig_csv(fig_i_res, OUTPUT_DIR / f"{indiv_stem}_residualized.csv")
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
    print("Spatial decoding comparison (+ residualized): WS vs LOSO topomaps")
    print("=" * 60)
    spatial_res_path = plot_spatial_comparison_panel_with_residualized()
    if spatial_res_path is not None:
        print(f"  Saved: {spatial_res_path}")
        print(f"  Saved: {spatial_res_path.with_suffix('.svg')}")

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
    print("AUC vs self-reported confidence (exploratory)")
    print("=" * 60)
    _queue(plot_auc_vs_confidence_scatter(), "scatter_auc_vs_confidence")

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
    print("SHAP scatter — marker-aggregated (WS vs LOSO)")
    print("=" * 60)
    _queue(_build_shap_marker_scatter_fig(), "scatter_shap_marker")
    for fig_marker_single, stem in _build_shap_marker_single_figures():
        _queue(fig_marker_single, stem)

    print("=" * 60)
    print("Probe-level probability scatter: WS vs LOSO")
    print("=" * 60)
    # Bypasses _queue()'s screen-preview scale=2 (192 dpi): this figure is
    # sized to its real 180x60 mm print footprint (see
    # plot_proba_ws_vs_loso_scatter), so it needs its own export scale to
    # hit the project's 600 dpi minimum for a paper PNG — scale is relative
    # to the 96 dpi Plotly/Kaleido assume, so 600/96.
    fig_proba, w_proba, h_proba = plot_proba_ws_vs_loso_scatter()
    _export_fig_csv(fig_proba, OUTPUT_DIR / "scatter_proba_ws_vs_loso.csv")
    for fmt in ("png", "svg"):
        pending.append((fig_proba, OUTPUT_DIR / f"scatter_proba_ws_vs_loso.{fmt}",
                        {"format": fmt, "width": w_proba, "height": h_proba,
                         "scale": 600.0 / 96.0 if fmt == "png" else 1}))

    print("=" * 60)
    print("CV-scheme stability combined: performance (A) + probability (B)")
    print("=" * 60)
    _queue(plot_cv_stability_combined(), "cv_stability_combined")

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
