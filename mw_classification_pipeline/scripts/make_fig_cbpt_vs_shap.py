#!/usr/bin/env python
"""
CBPT effect size vs classification SHAP importance, per marker, per dimension.

Reads the Andrillon CBPT per-marker cluster statistics (Paper 2 Section 2) and
the WS/LOSO SHAP marker profiles already computed by
``feature_consistency_analysis.py`` (Paper 2 Section 3), and asks whether a
marker's group-level neural effect predicts how much a classifier leans on it
for individual-level decoding.

CBPT never persists a beta/regression coefficient — ``Statistics/lmm_model.py``
and ``lmm_r_backend.py`` compute ``Estimate`` internally and discard it, saving
only t-statistics, p-values and the cluster statistic (a summed t-mass). The
x-axis here is therefore the **peak cluster-mass statistic** per (dimension,
marker) — ``max(|statistic|)`` across that marker's clusters, itself a *sum*
of per-electrode t-values within a spatially contiguous cluster (not a single
electrode's t, which is why it reaches +-100-150 rather than +-2..6) — signed,
the same "rank by |cluster_stat|" convention CLAUDE.md already uses to compare
marker effects. It is an effect-size proxy, not a beta.

Two SHAP-side quantities, from two different upstream tables:

Full
    ``marker_level_consistency.csv``'s ``mean_share_{ws,loso}`` (each marker's
    share of a subject's total mean(|SHAP|), averaged over subjects) signed by
    ``mean_direction_{ws,loso}`` (the SHAP-value/feature-value correlation).
Residualized
    ``residual_marker_profiles.csv``'s ``share_residual`` for the
    cross-dimension-residualized contrasts. This table only has magnitude
    (``abs_residual``/``share_residual``) — no per-marker direction was
    computed for the residualized contrasts upstream — so this axis is
    unsigned, unlike Full.

Four figures — every (pipeline, panel-kind) combination kept in its own file:
``cbpt_vs_shap_{ws,loso}_{full,residualized}``. Each is one row of five panels,
one per probe dimension (canonical order), sized to the actual printed page
(``WIDTH_MM`` = double-column, per CLAUDE.md "Figure Assembly (Paper 2)") —
not to a screen-sized canvas, which is what made every earlier pass at this
figure's type read "too small" once placed in the paper: the canvas was built
several times wider than a printed page and every font constant was chosen to
look right on that oversized canvas (see the scientific-plots skill's
`references/learnings/scatter-correlation.md`, 2026-08-18 entry, for the same
failure mode on a sibling figure). Uses ``sciplot``'s pure sizing helpers
(``mm2px``/``pt2px``/``facet_subplots``) without ``sp.init()``, per that same
entry, so this figure keeps the Times New Roman house style
``generate_combined_classification_figure.py``'s ~20 sibling figures share
rather than switching to the skill's own template for one figure out of a
matched set.

Panel styling: a plain coloured dimension name as the panel title, a fitted
trend line, and the Spearman rho/p-value as in-panel text — sitting in a
headroom band reserved above the highest data point so it never collides with
a point or its label, rather than merged into the title (matching this repo's
``plot_auc_vs_confidence_scatter``). Markers are named with
``_pretty_marker_name`` (the union of each axis's top-N-by-|value| markers;
the rest sit as a faint background), and fill is solid when the marker
survived within-family BH correction in CBPT, hollow otherwise — the one
per-marker significance test this figure has; SHAP importance has no
analogous per-marker null test in this pipeline.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_cbpt_vs_shap.py
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from scipy.stats import spearmanr

_SKILL_SCRIPTS = Path("/network/iss/home/nicolas.bruno/.claude/skills/scientific-plots/scripts")
sys.path.insert(0, str(_SKILL_SCRIPTS))
import sciplot as sp  # noqa: E402  (pure helpers only — no sp.init(), see module docstring)

# =============================================================================
# Configuration
# =============================================================================
_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parents[1]

# Shared project color palette (single source of truth), resolved the same way
# generate_combined_classification_figure.py does — this repo's actual
# color_palette.yaml predates sciplot's own schema, so it is read directly
# rather than through sciplot's Palette class (see RepoPalette precedent in
# generate_combined_classification_figure.py).
PALETTE = yaml.safe_load(open(REPO_ROOT / "color_palette.yaml"))
DIM_COLORS = PALETTE["dimensions"]
PERM_COLOR = PALETTE["neutral"]["permutation"]

ANDRILLON_FORMULA = "onoff_valence_selfother_time_time_on_task_confidence"
ANDRILLON_ROOT = REPO_ROOT / "results" / "andrillon_cluster"
FEATURE_CONSISTENCY_DIR = _SCRIPTS_DIR.parent / "results" / "feature_consistency"
OUT_DIR = _SCRIPTS_DIR.parent / "results" / "figures" / "cbpt_vs_shap"

# Canonical dimension order (CLAUDE.md "Dimension Order (Figures)"), and the
# rename from CBPT's target-directory spelling to the SHAP tables' contrast
# spelling (only "onoff" vs "on_off" differs). Pole wording is CLAUDE.md's
# SHORT_POLES table.
DIMENSIONS = [
    {"key": "onoff", "contrast": "on_off", "label": "On/Off-Task",
     "pole_low": "off-task", "pole_high": "on-task"},
    {"key": "valence", "contrast": "valence", "label": "Valence",
     "pole_low": "negative", "pole_high": "positive"},
    {"key": "selfother", "contrast": "selfother", "label": "Self/Other",
     "pole_low": "self-focused", "pole_high": "other-focused"},
    {"key": "time", "contrast": "time", "label": "Time",
     "pole_low": "past", "pole_high": "future"},
    {"key": "confidence", "contrast": "confidence", "label": "Confidence",
     "pole_low": "low", "pole_high": "high"},
]
CONTRAST_KEY = {d["key"]: d["contrast"] for d in DIMENSIONS}

# CBPT sign convention (POLE_LABELS in Stats_andrillon/plot_paper_figures.py):
# negative = low pole, positive = high pole. The SHAP tables follow the same
# sense: target=1 (the classifier's positive class) is always the raw
# above-median / high-pole value (mw_classification_pipeline/utils/data_utils.py
# positive_above=True, verified by tracing the label-encoding code), so both
# axes read "negative = low pole, positive = high pole" with no per-axis
# exception.
PANEL_KINDS = {
    "full":         dict(y_col="shap_full",     y_signed=True,  y_title_key="y_title_full"),
    "residualized": dict(y_col="shap_residual", y_signed=False, y_title_key="y_title_residual"),
}

# Not a single t-value — the cluster-mass statistic (sum of per-electrode
# t-values within a spatially contiguous cluster; CLAUDE.md e.g. "PE-beta
# cluster stat -75.6 over 22 electrodes"), which is why it reaches +-100-150
# rather than +-2..6.  The display string is in LAYOUT["shared_x_label"].

# =============================================================================
# Layout & appearance — every tunable in one place.
# Units: mm for physical lengths, pt for font sizes, Plotly px for markers.
# =============================================================================
LAYOUT = dict(

    # ── Figure canvas (mm) ────────────────────────────────────────────────
    # Total printed dimensions including all margins.
    width_mm  = 590.0,
    height_mm = 215.0,

    # ── Outer margins (mm) ────────────────────────────────────────────────
    # Space between the canvas edge and the axes.
    # margin_b must fit: tick labels + x-axis pole title + shared x-label +
    #   legend.  margin_l must fit the y-axis quantity title.
    margin_l = 44.0,
    margin_r = 24.0,
    margin_t = 18.0,   # top margin (holds panel title overhang)
    margin_b = 72.0,   # bottom margin (tick labels + x-titles + legend row)

    # ── Panel gap ────────────────────────────────────────────────────────
    # Fraction of total axis-area width reserved as gutter between subplot
    # columns.  Increase to give rotated y-axis pole labels more room.
    panel_gap = 0.075,

    # ── Bottom annotation stack (mm below the axis baseline) ─────────────
    # Tick labels + x-axis pole title reach this far below the axis.
    # The shared x-label and legend are placed just below it.
    # Must be < margin_b, otherwise labels overflow into the plot area.
    bottom_stack_mm = 46.0,

    # ── X-axis data padding ───────────────────────────────────────────────
    # Fraction of the data x-range added as padding on each side of the axis.
    # Wider = more room for extreme-value labels; narrower = more plot density.
    x_pad_frac = 0.60,

    # ── Y-axis headroom bands (fraction of data y-range) ─────────────────
    # label_band_frac : space above the highest point for "top"-anchored
    #   marker name labels.  Increase if top labels collide with the rho/p text.
    # annot_band_frac : space above label_band for the rho/p annotation.
    # y_bottom_frac   : fraction added below the lowest point.
    label_band_frac = 0.38,
    annot_band_frac = 0.28,
    y_bottom_frac   = 0.18,

    # ── Number of labelled markers per panel ─────────────────────────────
    # Union of top-N by |cluster_stat| and top-N by |SHAP importance|.
    # Fewer labels → less overlap; more labels → more context.
    n_labels = 4,

    # ── Scatter marker sizes (Plotly px, ~screen px at 96 dpi) ───────────
    marker_size_bg    = 10.0,   # unlabelled (background) markers
    marker_size_label = 12.0,  # labelled (foreground) markers
    marker_edge_bg    = 0.8,   # edge line width — background markers
    marker_edge_label = 1.1,   # edge line width — labelled markers
    legend_marker_size = 10.0, # markers in the fill-encoding legend row

    # ── Line widths (Plotly px) ───────────────────────────────────────────
    trendline_width = 1.6,   # fitted OLS trend line
    refline_width   = 0.9,   # zero-crossing dashed/dotted guide lines

    # ── Font sizes (typographic points, converted to px via sp.pt2px) ────
    # pt_panel_title   : bold dimension name above each subplot panel
    # pt_axis_title    : x-axis pole direction ("off-task ← → on-task") and
    #                    y-axis quantity + pole label (column 1 only)
    # pt_pole_annot    : rotated y-axis pole text for interior columns (2–5)
    # pt_tick          : axis tick-number labels
    # pt_marker_label  : text next to each labelled scatter point
    # pt_annotation    : in-panel Spearman ρ / p-value text
    # pt_shared_x_label: shared x-axis label centred below all panels
    # pt_legend        : fill-encoding legend at the very bottom
    pt_panel_title    = 26.0,
    pt_axis_title     = 20.0,
    pt_pole_annot     = 20.0,
    pt_tick           = 13.0,
    pt_marker_label   = 16.0,
    pt_annotation     = 18.0,
    pt_shared_x_label = 20.0,
    pt_legend         = 15.0,

    # ── Text labels that appear in the figure ────────────────────────────
    # shared_x_label   : label centred below all 5 panels (the x-axis quantity)
    # y_title_full     : y-axis quantity name for the signed / full panel kind
    # y_title_residual : y-axis quantity name for the residualized panel kind
    # legend_sig       : legend entry for CBPT BH-significant markers (filled)
    # legend_nonsig    : legend entry for non-significant markers (hollow)
    shared_x_label   = "CBPT cluster-mass T-value (signed)",
    y_title_full     = "Relative SHAP importance (signed)",
    y_title_residual = "Relative SHAP importance<br>(residualized, unsigned)",
    legend_sig       = "CBPT BH-significant",
    legend_nonsig    = "not significant",
)


def format_pval(p: float) -> str:
    """APA-style p-value, matching generate_combined_classification_figure.py."""
    if np.isnan(p):
        return "p = n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


# =============================================================================
# Marker naming (CLAUDE.md "Marker Naming (Figures)")
# =============================================================================
def _pretty_marker_name(name: str) -> str:
    """Shorten a raw marker name for display, keeping its family tag."""
    for old, new in (("psd_relative_", "PSD "), ("kolmogorov_complexity", "Kolmogorov"),
                     ("slowwaves_", "SW "), ("wsmi_", "wSMI "), ("PE_", "PE ")):
        if name.startswith(old) or name == old.rstrip("_"):
            return name.replace(old, new)
    return name


# =============================================================================
# Data
# =============================================================================
def load_cbpt_effects() -> pd.DataFrame:
    """
    Peak cluster statistic per (dimension, marker), signed, with its
    within-family BH-significance flag.

    Read from each marker's ``results.pkl`` (``cluster_stats``,
    ``marker_rejected``) rather than the derived ``cluster_summary_corrected
    .csv`` — 18 of 115 dimension x marker combinations found **zero** clusters
    at all (no electrode crossed the cluster-forming threshold in either
    direction), and the postprocessing step only writes the CSV when there is
    at least one cluster to summarize. That is a legitimate null result, not
    missing data: ``cluster_stats`` is an empty array and ``marker_rejected``
    is ``False``, so the peak statistic is correctly 0.0.

    Returns
    -------
    pd.DataFrame
        Columns ``dimension, marker, family, cluster_stat, cbpt_significant``.
        115 rows: 5 dimensions x 23 markers (4 evoked + 19 sleep).
    """
    rows = []
    for dim in DIMENSIONS:
        dim_dir = ANDRILLON_ROOT / f"{ANDRILLON_FORMULA}__target_{dim['key']}"
        for marker_dir in sorted(p for p in dim_dir.iterdir() if p.is_dir()):
            family, marker = marker_dir.name.split("_", 1)
            with open(marker_dir / "results.pkl", "rb") as fh:
                result = pickle.load(fh)
            cluster_stats = result["cluster_stats"]
            peak_stat = float(cluster_stats[np.argmax(np.abs(cluster_stats))]) if len(cluster_stats) else 0.0
            rows.append(dict(
                dimension=dim["key"], marker=marker, family=family,
                cluster_stat=peak_stat,
                cbpt_significant=bool(result["marker_rejected"]),
            ))
    return pd.DataFrame(rows)


def load_shap_full() -> pd.DataFrame:
    """
    Signed SHAP effect per (dimension, marker, pipeline) for the raw contrasts.

    ``mean_share`` (relative mean(|SHAP|)) signed by the sign of
    ``mean_direction`` (the SHAP-value/feature-value correlation).
    """
    mlc = pd.read_csv(FEATURE_CONSISTENCY_DIR / "marker_level_consistency.csv")
    rename = {v: k for k, v in CONTRAST_KEY.items()}
    mlc["dimension"] = mlc["contrast"].map(rename)
    frames = []
    for pipeline in ("ws", "loso"):
        frame = mlc[["dimension", "marker"]].copy()
        frame["pipeline"] = pipeline
        frame["shap_full"] = (
            mlc[f"mean_share_{pipeline}"] * np.sign(mlc[f"mean_direction_{pipeline}"])
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_shap_residual() -> pd.DataFrame:
    """
    Unsigned SHAP effect per (dimension, marker, pipeline) for the
    cross-dimension-residualized contrasts (``share_residual``).

    No direction/sign was computed upstream for these contrasts, so this stays
    a magnitude, unlike :func:`load_shap_full`.
    """
    residual = pd.read_csv(FEATURE_CONSISTENCY_DIR / "residual_marker_profiles.csv")
    rename = {v: k for k, v in CONTRAST_KEY.items()}
    residual["dimension"] = residual["contrast"].map(rename)
    return residual.rename(columns={"share_residual": "shap_residual"})[
        ["dimension", "marker", "pipeline", "shap_residual"]
    ]


def load_pipeline_table(pipeline: str) -> pd.DataFrame:
    """CBPT effects joined to both SHAP tables for one classification pipeline."""
    cbpt = load_cbpt_effects()
    full = load_shap_full()
    residual = load_shap_residual()
    full = full[full["pipeline"] == pipeline].drop(columns="pipeline")
    residual = residual[residual["pipeline"] == pipeline].drop(columns="pipeline")
    table = cbpt.merge(full, on=["dimension", "marker"], how="inner")
    table = table.merge(residual, on=["dimension", "marker"], how="inner")
    assert len(table) == len(cbpt) == 115, (
        f"expected 115 dimension x marker rows for pipeline={pipeline!r}, got "
        f"cbpt={len(cbpt)} joined={len(table)}"
    )
    return table


# =============================================================================
# Panel
# =============================================================================
def _add_panel(fig: go.Figure, col: int, block: pd.DataFrame, color: str,
                *, y_col: str, y_signed: bool) -> float:
    """
    Draw one CBPT-vs-SHAP scatter panel (23 markers of one dimension) into
    column ``col`` of a 1-row figure, and return its Spearman rho.

    Every marker is drawn — the reported rho always covers all 23 — with the
    union of each axis's top ``LABEL_N`` markers by |value| named
    (``_pretty_marker_name``) and the rest as a faint background. Labels are
    pushed left/right of the labelled subset's own x-median and cycle through
    three vertical offsets, so they separate even when several labelled
    markers share similar values.

    The y-axis reserves a headroom band above the highest data point — wide
    enough for both a "top"-positioned label and the rho/p annotation sitting
    above that — so the in-panel statistic text never collides with a point
    or its label regardless of where the labelled extremes happen to fall.

    Fill adds the one significance test this figure has: solid marker = the
    CBPT peak cluster survived within-family BH correction; hollow (white
    face, coloured edge — ``sciplot.hollow``) = it did not. SHAP has no
    analogous per-marker null test, so significance is never encoded on the
    y-axis.
    """
    x = block["cluster_stat"].to_numpy()
    y = block[y_col].to_numpy()
    names = [_pretty_marker_name(m) for m in block["marker"]]
    sig = block["cbpt_significant"].to_numpy()

    n_lab = LAYOUT["n_labels"]
    top_x = block.iloc[block["cluster_stat"].abs().to_numpy().argsort()[::-1][:n_lab]]["marker"]
    top_y = block.iloc[block[y_col].abs().to_numpy().argsort()[::-1][:n_lab]]["marker"]
    named = set(top_x) | set(top_y)
    mask = block["marker"].isin(named).to_numpy()

    x_lo, x_hi = float(x.min()), float(x.max())
    x_pad = LAYOUT["x_pad_frac"] * (x_hi - x_lo)
    x_range = [x_lo - x_pad, x_hi + x_pad]

    y_lo, y_hi = float(y.min()), float(y.max())
    y_span = y_hi - y_lo
    label_band = LAYOUT["label_band_frac"] * y_span
    annot_band = LAYOUT["annot_band_frac"] * y_span
    y_bottom = y_lo - LAYOUT["y_bottom_frac"] * y_span
    y_top = y_hi + label_band + annot_band
    annot_y = y_hi + label_band + annot_band * 0.5

    x_ref = "x" if col == 1 else f"x{col}"
    y_ref = "y" if col == 1 else f"y{col}"

    if y_signed:
        fig.add_shape(type="line", x0=x_range[0], x1=x_range[1], y0=0, y1=0,
                      xref=x_ref, yref=y_ref,
                      line=dict(color=PERM_COLOR, dash="dash", width=LAYOUT["refline_width"]))
    fig.add_shape(type="line", x0=0, x1=0, y0=y_bottom, y1=y_top,
                  xref=x_ref, yref=y_ref,
                  line=dict(color=PERM_COLOR, dash="dot", width=LAYOUT["refline_width"]))

    face = np.where(sig, color, "white")

    # Background: markers not singled out for labelling.
    fig.add_trace(go.Scatter(
        x=x[~mask], y=y[~mask], mode="markers",
        marker=dict(color=list(face[~mask]), size=LAYOUT["marker_size_bg"], opacity=0.6,
                    line=dict(color=color, width=LAYOUT["marker_edge_bg"])),
        customdata=[n for n, m in zip(names, mask) if not m],
        hovertemplate="%{customdata}<br>CBPT %{x:.1f}<br>SHAP %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=col)

    # Spread labels by y-rank within each side so they occupy distinct vertical
    # slots rather than colliding when several points share similar y-values.
    x_mid = float(np.median(x[mask]))
    x_m = x[mask]
    y_m = y[mask]
    n_lab = int(mask.sum())
    positions = np.empty(n_lab, dtype=object)
    left_opts  = ["bottom left",  "middle left",  "top left"]
    right_opts = ["bottom right", "middle right", "top right"]
    for side_mask, opts in ((x_m < x_mid, left_opts), (x_m >= x_mid, right_opts)):
        idx = np.where(side_mask)[0]
        if len(idx) == 0:
            continue
        y_ranks = np.argsort(np.argsort(y_m[idx]))   # 0 = lowest y
        for k, i in enumerate(idx):
            positions[i] = opts[y_ranks[k] % len(opts)]

    fig.add_trace(go.Scatter(
        x=x[mask], y=y[mask], mode="markers+text",
        marker=dict(color=list(face[mask]), size=LAYOUT["marker_size_label"],
                    line=dict(color=color, width=LAYOUT["marker_edge_label"])),
        text=[n for n, m in zip(names, mask) if m],
        textposition=positions,
        textfont=dict(size=sp.pt2px(LAYOUT["pt_marker_label"]), color="#222222"),
        # Unclipped: at this panel width (~30 mm), the padding needed to keep
        # every left-side label from touching the axis edge would eat too
        # much of the already-scarce plot width. The gutter (where the
        # rotated pole annotation sits, x=-0.16 domain) has enough spare room
        # that a few extra px of text spilling past the axis doesn't reach it.
        cliponaxis=False,
        customdata=[n for n, m in zip(names, mask) if m],
        hovertemplate="%{customdata}<br>CBPT %{x:.1f}<br>SHAP %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=col)

    # Trend line, fitted on all 23 markers and drawn across the full visible
    # x-range (not just the data extent).
    m, b = np.polyfit(x, y, 1)
    x_reg = np.array(x_range)
    fig.add_trace(go.Scatter(
        x=x_reg, y=m * x_reg + b, mode="lines",
        line=dict(color=color, width=LAYOUT["trendline_width"], dash="solid"), showlegend=False,
    ), row=1, col=col)

    rho, p = spearmanr(x, y)
    fig.add_annotation(
        text=f"ρ = {rho:.2f}, {format_pval(p)}",
        xref=f"{x_ref} domain", yref=y_ref,
        x=0.05, y=annot_y, xanchor="left", yanchor="middle", showarrow=False,
        font=dict(size=sp.pt2px(LAYOUT["pt_annotation"]), color=color),
    )

    fig.update_xaxes(range=x_range, tickfont=dict(size=sp.pt2px(LAYOUT["pt_tick"])), nticks=4,
                     showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=1, col=col)
    fig.update_yaxes(range=[y_bottom, y_top], tickfont=dict(size=sp.pt2px(LAYOUT["pt_tick"])),
                     showgrid=True, gridcolor="#EEEEEE", zeroline=False, row=1, col=col)
    return float(rho)


# =============================================================================
# Figure
# =============================================================================
def build_dimension_row_figure(table: pd.DataFrame, panel_kind: str) -> tuple[go.Figure, pd.DataFrame]:
    """
    One row of 5 panels (one per dimension) for one (pipeline, panel_kind).

    Returns
    -------
    (go.Figure, pd.DataFrame)
        The figure, and the per-panel Spearman rho/p table the panel
        annotations show (the in-panel statistic, kept separate from the raw
        per-marker points table per the scientific-plots skill's rule that a
        plotted statistic needs its own row in the backing data).
    """
    kind = PANEL_KINDS[panel_kind]
    y_title = LAYOUT[kind["y_title_key"]]
    n_dims = len(DIMENSIONS)

    # Plain make_subplots, not sciplot's facet_subplots: that helper's tight
    # physical gutter is built for the "one shared axis title, interior
    # panels cleared" case, and reserves no automargin room for a title on
    # interior columns — exactly what this figure needs, since every column's
    # y-axis carries its own (different) pole-pair title. A wider fractional
    # horizontal_spacing gives plotly's own per-axis automargin room to place
    # each one without it landing on top of the data.
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=n_dims, horizontal_spacing=LAYOUT["panel_gap"],
        subplot_titles=[d["label"] for d in DIMENSIONS],
    )
    for i, dim in enumerate(DIMENSIONS):
        fig.layout.annotations[i].text = f"<b>{dim['label']}</b>"
        fig.layout.annotations[i].font.color = DIM_COLORS[dim["key"]]
        fig.layout.annotations[i].font.size = sp.pt2px(LAYOUT["pt_panel_title"])

    stats_rows = []
    for col, dim in enumerate(DIMENSIONS, start=1):
        color = DIM_COLORS[dim["key"]]
        block = table[table["dimension"] == dim["key"]]
        rho = _add_panel(fig, col, block, color, y_col=kind["y_col"], y_signed=kind["y_signed"])
        _, p = spearmanr(block["cluster_stat"], block[kind["y_col"]])
        stats_rows.append(dict(dimension=dim["key"], label=dim["label"], rho=rho, p=float(p)))

        pole_pair = f'{dim["pole_low"]} ←  → {dim["pole_high"]}'
        # Two lines for the horizontal x-axis title only: at ~30 mm per
        # column, one line of "self-focused <- -> other-focused" (or, at the
        # larger type sizes this figure moved to, any of the pole pairs) is
        # wider than the column and bleeds into its neighbours. The rotated
        # y-axis version (below) doesn't need this — height, not width, is
        # its scarce dimension, and there is room to spare there.
        # At ~100 mm per panel there is enough horizontal room for a single
        # line; the <br> that was here was a workaround for the old ~30 mm
        # columns and now just breaks the arrows into two separate lines.
        pole_pair_x = f'{dim["pole_low"]} ← → {dim["pole_high"]}'
        fig.update_xaxes(
            title=dict(text=pole_pair_x,
                       font=dict(size=sp.pt2px(LAYOUT["pt_axis_title"]), color=color)),
            row=1, col=col)

        if col == 1:
            # Colour mixing: the quantity name is not dimension-specific (black);
            # the pole pair is dimension-coloured.  A plotly axis title only takes
            # one font.color, so each half gets its own <span>.
            axis_pt = sp.pt2px(LAYOUT["pt_axis_title"])
            if kind["y_signed"]:
                y_axis_text = (f'<span style="color:#000000;font-size:{axis_pt:.1f}px">{y_title}</span>'
                               f'<br><span style="color:{color};font-size:{axis_pt:.1f}px">{pole_pair}</span>')
            else:
                y_axis_text = f'<span style="color:#000000;font-size:{axis_pt:.1f}px">{y_title}</span>'
            fig.update_yaxes(title=dict(text=y_axis_text, font=dict(size=axis_pt)), row=1, col=col)
        elif kind["y_signed"]:
            # Interior y-axis pole labels sit in the panel gutter as manual
            # annotations (automargin cannot widen an interior gutter).
            fig.add_annotation(
                text=pole_pair, xref=f"x{col} domain", yref=f"y{col} domain",
                x=-0.24, y=0.5, xanchor="center", yanchor="middle", textangle=-90,
                showarrow=False, font=dict(size=sp.pt2px(LAYOUT["pt_pole_annot"]), color=color),
            )

    # One shared x-axis label (every panel's x is the same quantity) and the
    # significance legend, stacked below the per-panel pole titles — placed
    # as small *positive* fractions of the known MARGIN_MM["b"] band (the
    # explicit bottom margin set below), not large negative offsets guessed
    # against the figure's total height. Those negative offsets were tuned
    # for one specific HEIGHT_MM and silently stopped meaning the same
    # physical gap the moment that constant changed elsewhere — a huge dead
    # band one time, a clipped edge another. A fraction of a margin whose
    # size is fixed by hand is the same physical gap regardless of what else
    # in the figure changes.
    # Below BOTTOM_STACK_MM (where the automatic tick labels + pole title
    # end), the remaining MARGIN_MM["b"] - BOTTOM_STACK_MM of margin is clear
    # — split it 2:1 between the shared label (closer to the pole titles) and
    # the legend (nearer the true bottom edge).
    # Plot-area height in mm (the denominator for paper-coord fractions below).
    # paper y=0 is the axis bottom; one unit = one plot_h. Dividing by HEIGHT_MM
    # here was the bug that left 20+ mm of dead space below the legend.
    _plot_h = LAYOUT["height_mm"] - LAYOUT["margin_t"] - LAYOUT["margin_b"]
    shared_label_y = -(LAYOUT["bottom_stack_mm"] + 7.0)  / _plot_h
    legend_y       = -(LAYOUT["bottom_stack_mm"] + 17.0) / _plot_h
    fig.add_annotation(
        text=f'<span style="color:#000000">{LAYOUT["shared_x_label"]}</span>',
        xref="paper", yref="paper", x=0.5, y=shared_label_y,
        xanchor="center", yanchor="middle", showarrow=False,
        font=dict(size=sp.pt2px(LAYOUT["pt_shared_x_label"])),
    )

    for face, label in (("filled", LAYOUT["legend_sig"]), ("hollow", LAYOUT["legend_nonsig"])):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=LAYOUT["legend_marker_size"],
                        color=PERM_COLOR if face == "filled" else "white",
                        line=dict(color=PERM_COLOR, width=1.5)),
            name=label, showlegend=True, hoverinfo="skip",
        ))

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=sp.pt2px(LAYOUT["pt_tick"])),
        margin=dict(l=sp.mm2px(LAYOUT["margin_l"]), r=sp.mm2px(LAYOUT["margin_r"]),
                    t=sp.mm2px(LAYOUT["margin_t"]), b=sp.mm2px(LAYOUT["margin_b"])),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=legend_y, yanchor="middle",
                    font=dict(size=sp.pt2px(LAYOUT["pt_legend"]))),
    )
    return fig, pd.DataFrame(stats_rows)


# =============================================================================
# Save helper (bypasses sp.save's 300 mm guard for this supplementary figure)
# =============================================================================
def _save(fig: go.Figure, outdir: str | Path, name: str, *,
          width_mm: float, height_mm: float, dpi: float,
          data: dict) -> None:
    """Write SVG + PNG + CSVs; same logic as sp.save without the width cap."""
    w_px = round(sp.mm2px(width_mm))
    h_px = round(sp.mm2px(height_mm))
    fig.update_layout(width=w_px, height=h_px)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    scale = dpi / sp.CSS_DPI
    fig.write_image(out / f"{name}.svg", format="svg", width=w_px, height=h_px, scale=1)
    fig.write_image(out / f"{name}.png", format="png", width=w_px, height=h_px, scale=scale)
    sp.save_data(out, name, data)
    print(f"[save] {name}: {width_mm:.0f}x{height_mm:.0f} mm ({w_px}x{h_px} px), PNG at {dpi:.0f} dpi")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None  # e.g. "ws_full" for a cheap single-figure iteration
    dpi = 150 if only else sp.DEFAULT_DPI

    for pipeline, pipeline_label in (("ws", "Within-Subject"), ("loso", "LOSO")):
        table = load_pipeline_table(pipeline)
        print(f"\n{pipeline_label}")

        for panel_kind in ("full", "residualized"):
            out_stem = f"cbpt_vs_shap_{pipeline}_{panel_kind}"
            if only and f"{pipeline}_{panel_kind}" != only:
                continue
            kind = PANEL_KINDS[panel_kind]
            fig, stats = build_dimension_row_figure(table, panel_kind)

            print(f"  {panel_kind}:")
            for row in stats.itertuples():
                print(f"    {row.label:<14} rho={row.rho:+.2f} p={row.p:.3f}")

            points = table[["dimension", "marker", "family", "cluster_stat", "cbpt_significant",
                            kind["y_col"]]].rename(columns={kind["y_col"]: "y_value"})
            _save(fig, OUT_DIR, out_stem,
                  width_mm=LAYOUT["width_mm"], height_mm=LAYOUT["height_mm"], dpi=dpi,
                  data={"points": points, "stats": stats})


if __name__ == "__main__":
    main()
