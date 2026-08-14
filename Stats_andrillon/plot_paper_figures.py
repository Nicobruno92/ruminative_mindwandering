"""
Paper-ready figures for the Andrillon CBPT analysis (Paper 2, Section 2).

WHAT THIS SCRIPT IS, AND IS NOT
-------------------------------
It is *packaging only*. Every topographic map is drawn with the exact same
``mne.viz.plot_topomap`` call used by the production pipeline
(``Statistics/plot_results.py::plot_cluster_topomap``, lines 256-275): same
colormap, same interpolation, same head outline, same significance mask style.
Those keyword arguments are reproduced verbatim in ``TOPOMAP_KWARGS`` below so a
reader can diff them against the original. Nothing about how a topography is
computed or rendered is re-invented here.

What this script adds is composition: which panels go on which figure, a colour
scale shared across the panels of a figure so they are visually comparable, and
a colour bar whose poles are labelled with what they mean for the probe
dimension being plotted.

WHY IT READS results.pkl AND NOT multiple_comparisons_summary.csv
-----------------------------------------------------------------
The evoked-family fits were regenerated on 2026-07-28/29 for every target, but
the marker-wise correction step was not re-run afterwards, so the on-disk
``multiple_comparisons_summary.csv`` files are stale for that family (issue
SCI-001 in reports/report_pipeline_review_2026-07-29.md). Reading the raw
per-marker ``results.pkl`` and re-deriving the Benjamini-Hochberg correction
here keeps the figure internally consistent with the fits it is drawing. The
procedure applied is the one the pipeline documents: one representative
p-value per marker (its max-statistic cluster), corrected within each
declared family, never across families.

This is a stopgap for figure generation, not a replacement for re-running
``Statistics/apply_mcc_postprocessing.py``.

SIGN CONVENTION — READ BEFORE INTERPRETING ANY COLOUR
-----------------------------------------------------
Probe dimensions are recorded on a 0-100 slider where 100 is the second pole
listed in the project README (100 = on-task, positive, future-oriented,
other-focused, high-confidence). A positive t-statistic therefore means the EEG
marker *increases* toward that pole.

Two independent physiological anchors in the observed data confirm this
orientation rather than assuming it:
  - ``evoked/P3b`` for onoff has a positive cluster statistic, i.e. larger P3b
    toward high onoff. P3 amplitude is well established to be reduced during
    mind-wandering, so high onoff = on-task.
  - ``sleep/slowwaves_Density`` for onoff has a negative cluster statistic, i.e.
    more slow waves toward low onoff. Andrillon et al. (2021) report exactly
    this: sleep-like slow waves increase during attentional lapses.
Both anchors agree, so low onoff = off-task (mind-wandering).

The orthogonalised quadratic predictors ``valence_sq``/``time_sq`` were removed
from the analysis on 2026-08-13, so every column of every figure here is now a
plain linear dimension and the sign convention above applies uniformly. See
``Stats_andrillon/config_andrillon.yaml`` for why they were retired.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Statistics"))

from Statistics.plot_results import (  # noqa: E402
    overlay_uncorrected_only_channels,
    save_figure_multiformat,
    validate_montage_and_channels,
)

# Chrome (separator lines, secondary labels) uses the project's neutral grey
# rather than an ad-hoc hex, per color_palette.yaml's own rule: key into this
# file, never hardcode. Loaded once at import time since this colour is used
# by several helpers that don't otherwise need the full palette dict.
with open(REPO_ROOT / "color_palette.yaml") as _palette_handle:
    NEUTRAL_GRAY = yaml.safe_load(_palette_handle)["neutral"]["covariate"]

# =============================================================================
# CONFIGURATION
# =============================================================================

# Reproduced verbatim from Statistics/plot_results.py::plot_cluster_topomap
# (lines 256-275). Do not tune these here — if the pipeline's topography
# rendering changes, change it there and mirror it back.
TOPOMAP_KWARGS = dict(
    cmap="RdBu_r",
    sensors=False,
    contours=6,
    ch_type="eeg",
    sphere="auto",
    outlines="head",
    extrapolate="head",
    image_interp="cubic",
    border="mean",
    res=128,
)

# Significance mask style, adapted from plot_cluster_topomap: same marker
# shape and colour. Originally sized smaller than the pipeline's
# single-topomap figures because these panels used to be ~1/4 their area;
# since the 2026-08-13 pass makes every topomap fill its column (see
# solve_square_topomap_row_height_in), that area gap mostly closed, and the
# old 4.2pt dot read as barely-visible ("los puntos... quedaron muy chicos")
# against a head circle several times its old size. Scaled up to match.
MASK_PARAMS = dict(
    marker="o",
    markerfacecolor="k",
    markeredgecolor="k",
    linewidth=0,
    markersize=10.5,
)

# Style for channels that are part of a raw-significant cluster (p < alpha)
# whose MARKER does not survive the family-level BH correction. Drawn as a
# hollow ring (Statistics/plot_results.py::overlay_uncorrected_only_channels,
# reused verbatim) rather than omitted — the marker still carries genuine
# nominal signal, it just doesn't clear the multiple-comparisons bar on its
# own (see the "distributed but real" tier in CLAUDE.md).
HOLLOW_MARKERSIZE = 11.5
HOLLOW_MARKEREDGEWIDTH = 2.6

# Panels per family (onoff figure) / per dimension (other-dimensions figure).
# Kept small and equal-sized on purpose: a main figure shows the strongest
# effects, not every effect, and a fixed count per group is what makes every
# group (including a null one) equally visible instead of some groups crowding
# others out of a pooled top-N.
N_TOP_MARKERS_PER_FAMILY = 3
N_TOP_MARKERS_PER_DIMENSION = 3

# =============================================================================
# LAYOUT — chrome budgeted in inches, not guessed as fractions
# =============================================================================
# A fraction of panel height (e.g. "top=0.85") reserves a DIFFERENT absolute
# amount of space depending on how tall the panel is — the same fraction that
# looks right in a tall standalone figure reserves too little (or too much) in
# a short subfigure of a combined layout. Chrome (titles, headers, colour bars)
# is text: it needs a fixed amount of space in INCHES regardless of the panel's
# height. These constants are that fixed budget; ``fractions_from_budget``
# converts them to the top/bottom fractions GridSpec actually wants, given
# whatever panel height a specific caller is using. This is what lets the same
# drawing function produce a tightly-packed panel whether it is saved on its
# own or embedded at a different size inside the combined figure.
#
# Budgets are DERIVED from the font sizes below via ``text_block_height_in``,
# not hand-picked independently of them — every previous round of "make the
# text bigger" desynced a font size from a budget that was tuned for the old,
# smaller size, which is exactly what caused new collisions each time. Bump a
# *_FONTSIZE constant and its budget grows with it automatically.
SUPTITLE_FONTSIZE = 40
# Bare panel letter (A/B/C) drawn in the combined figure instead of a
# descriptive suptitle — see the panel_letter parameter on draw_heatmap /
# draw_onoff_panel / draw_other_dimensions_panel. Still clearly larger than
# HEADER_FONTSIZE/PANEL_TITLE_FONTSIZE per the project's panel-letter
# convention. Raised twice on 2026-08-13 per explicit user direction
# ("agranda muuucho las letras de A B C", then "agrandalas bastante mas AUN
# bastante" after the first 30->58 pass still wasn't enough) — every other
# bump in this pass was a moderate "make it readable" step; this one was
# called out twice on its own as needing to be much larger than that. Shares
# its row with the significance legend rather than getting a reserved row of
# its own — a lone character doesn't carry the same visual weight as the
# sentence it replaced, so giving it an equally large dedicated row just left
# that row looking empty.
PANEL_LETTER_FONTSIZE = 92
HEADER_FONTSIZE = 36
PANEL_TITLE_FONTSIZE = 32
# Gap, in points, between the marker-name title (e.g. "P3b") and its axes'
# top edge (``ax.set_title(..., pad=...)`` in draw_onoff_panel /
# draw_other_dimensions_panel). MNE's head outline draws almost to the very
# top of the axes bounding box, so at the small pad (3pt) this used before
# the topomaps were enlarged to fill their columns, the bigger title text now
# reads as touching the head outline below it.
MARKER_TITLE_PAD_PT = 14
# Bumped to 28 in an earlier round to make legends easier to read at a
# glance, but that was tuned back when the legend still shared its row with
# a full descriptive suptitle; once the suptitle shrank to a bare letter the
# legend read as oversized next to it. 24 keeps it close to HEADER_FONTSIZE
# (still far from a "secondary/footnote" size) without dominating the row.
# Raised again to 30 in the 2026-08-13 "almost all text is too small" pass —
# HEADER_FONTSIZE itself grew a lot in that pass, so a legend still at 24
# would have re-created the exact "legend reads as oversized/undersized next
# to its neighbour" problem this comment originally describes, just inverted.
LEGEND_FONTSIZE = 30
# The onoff directional bar (add_directional_colorbar) spans the middle
# columns of a 5-column panel and its pole text overflows OUTWARD into the
# (otherwise empty, at that row) outer columns — there is nothing there for it
# to collide with, so this one tolerates a much bigger font than the
# per-dimension bars below.
COLORBAR_LABEL_FONTSIZE = 32
# The per-dimension column colour bars (add_column_colorbar) are only ~1/6 of
# the figure width, unlike the single wide onoff bar that COLORBAR_LABEL_FONTSIZE
# is sized for, and neighbouring columns each have their own label to collide
# with — measured with matplotlib's own text extents: the longest pole pair
# ("self-focused" / "other-focused") needs >=3.74in of column width at 17pt,
# which is why figure_other_dimensions/figure_combined widen those panels
# below to fit it (Statistics/plot_results.py has no equivalent — this bar is
# unique to this script). Was 21pt — at the combined figure's true print size
# this read as essentially blank ("el texto del cbar no se ve nada"), so this
# is the single largest proportional jump in the 2026-08-13 pass; the column
# width safety margin below (COLUMN_COLORBAR_WIDTH_SAFETY_IN) grew to match.
COLUMN_COLORBAR_LABEL_FONTSIZE = 34
COLORBAR_TICK_FONTSIZE = 30
# Tuned against measured text extents (not guessed): at HEATMAP_TICK_FONTSIZE
# pt, the widest column header ("quadratic") and the left-margin row/group
# labels ("wSMI gamma" / "Complexity /") need the panel widths and
# left_margin fractions set below in figure_heatmap/figure_combined — bump
# this without also widening those panels and the headers start touching
# again (that is exactly the bug this round fixed; see git history).
HEATMAP_TICK_FONTSIZE = 34
HEATMAP_TITLE_FONTSIZE = 40
# The in-cell significance glyphs (dot/asterisks) are isolated inside a solid
# colour patch with no neighbouring text to collide with — unlike every other
# label in this figure, there is no width budget to solve for here, so these
# were the one place still worth a large, un-tuned bump on their own.
HEATMAP_GLYPH_FONTSIZE = 44
HEATMAP_DOT_GLYPH_FONTSIZE = 32
# Family-group row label ("Evoked (ERP)", "Complexity /\ninformation"), drawn
# left of the family-boundary line in draw_heatmap. Used to be a fraction of
# HEATMAP_TICK_FONTSIZE (0.95x, i.e. deliberately *smaller* than the column
# headers) — at that ratio it was the least legible text on the whole figure
# ("ni el nombre de las familias [se ve]"). A family name labels 2-5 marker
# rows at once, more chart furniture than a per-cell tick, so it gets its own
# constant, sized larger than the ticks rather than as a fraction of them.
HEATMAP_FAMILY_LABEL_FONTSIZE = 40
# Marker-name row label ("P3b", "wSMI gamma"). Was tick_fontsize*1.15; bumped
# to 1.3x here for the same "ylab not visible" complaint — these are short
# single/double-word strings with slack in the left margin, unlike the
# 2-line column headers left_margin is actually sized against (see the
# draw_heatmap docstring).
HEATMAP_YLABEL_FONTSIZE_RATIO = 1.3


def text_block_height_in(fontsize_pt: float, n_lines: int = 1, linespacing: float = 1.25) -> float:
    """
    Approximate the vertical space a text block needs, in inches, with margin.

    Font metrics (ascender/descender/internal leading) mean a line of text
    occupies noticeably more vertical space than its point size alone would
    suggest; the 1.15 factor is a safety margin against that, not a precise
    typographic constant. Used to derive every layout budget below from the
    font size that will actually be rendered, instead of the two drifting
    apart the next time a font size changes. Lowered from 1.4 to 1.15 (still a
    real margin, just not a padded one) because every chrome block in this
    figure (suptitle, legend row, column headers, per-panel title) draws from
    this same function, so the old factor's slack compounded into a visibly
    "dead" strip above every panel — shrinking it here reclaims that space
    everywhere at once without touching any font size.

    Parameters
    ----------
    fontsize_pt : float
        Font size in points.
    n_lines : int
        Number of stacked lines in the block.
    linespacing : float
        Line spacing multiplier (matplotlib's ``linespacing``).

    Returns
    -------
    float
        Height budget in inches.
    """
    return fontsize_pt * n_lines * linespacing / 72.0 * 1.15


SUPTITLE_BUDGET_IN = text_block_height_in(SUPTITLE_FONTSIZE)
# The panel-letter row shares a single line with the significance legend
# (letter left, legend right) rather than getting a row of its own the way
# the standalone suptitle does — so its budget only needs to cover the
# taller of the two, not both stacked.
PANEL_LETTER_ROW_BUDGET_IN = max(
    text_block_height_in(PANEL_LETTER_FONTSIZE), text_block_height_in(LEGEND_FONTSIZE)
)
# Sized for a two-line header (e.g. "Complexity /\ninformation") — the onoff
# panel's family names need two lines; giving the (shorter, single-line)
# other-dimensions header the same budget just leaves it a little unused
# space, which is a cheaper mistake than a collision.
HEADER_BUDGET_IN = text_block_height_in(HEADER_FONTSIZE, n_lines=2)
# The per-panel ax.set_title() (marker name) renders ABOVE the axes' own
# bounding box, using its own font height + pad — space GridSpec's "top"
# fraction does not know about and will not reserve on its own. Skipping this
# budget line was what made the column header text land right on top of the
# marker-name titles: both were being placed only a hair above the same
# tops[0] edge.
PANEL_TITLE_BUDGET_IN = text_block_height_in(PANEL_TITLE_FONTSIZE)
GAP_BUDGET_IN = 0.06
# The visible colour gradient is a deliberately thin strip (COLORBAR_BAR_IN),
# not the whole reserved row — a colour bar reads as data, and making it as
# tall as the text next to it makes the panel look bottom-heavy. Most of the
# row's height budget goes to the (now much larger) pole-label text under it.
COLORBAR_BAR_IN = 0.16
COLORBAR_ROW_BUDGET_IN = (
    COLORBAR_BAR_IN + GAP_BUDGET_IN + text_block_height_in(COLORBAR_LABEL_FONTSIZE)
)
# Stacked, bottom to top: 2-line xtick labels, glyph-key legend, main title.
HEATMAP_XTICK_BUDGET_IN = text_block_height_in(HEATMAP_TICK_FONTSIZE, n_lines=2)
HEATMAP_LEGEND_BUDGET_IN = text_block_height_in(LEGEND_FONTSIZE)
# Same line height, reused by draw_onoff_panel/draw_other_dimensions_panel to
# give the significance legend its own row under the suptitle instead of
# sharing one row with it (see add_significance_legend).
LEGEND_ROW_BUDGET_IN = HEATMAP_LEGEND_BUDGET_IN
HEATMAP_TITLE_BUDGET_IN = text_block_height_in(HEATMAP_TITLE_FONTSIZE)
# Panel-letter mode (combined figure): letter and glyph-legend share one row
# instead of each getting a stacked row of their own — same reasoning as
# PANEL_LETTER_ROW_BUDGET_IN above.
HEATMAP_TOP_BUDGET_LETTER_IN = (
    HEATMAP_XTICK_BUDGET_IN + PANEL_LETTER_ROW_BUDGET_IN + 2 * GAP_BUDGET_IN
)
HEATMAP_TOP_BUDGET_IN = (
    HEATMAP_XTICK_BUDGET_IN
    + HEATMAP_LEGEND_BUDGET_IN
    + HEATMAP_TITLE_BUDGET_IN
    + 3 * GAP_BUDGET_IN
)
HEATMAP_BOTTOM_BUDGET_IN = 0.04

# GridSpec margins shared by draw_onoff_panel/draw_other_dimensions_panel and
# the row-height solver below — the solver has to reproduce the exact same
# left/right/wspace GridSpec uses, or its "what column width will GridSpec
# actually give each axes" answer would drift from reality.
TOPOMAP_GRID_LEFT = 0.005
TOPOMAP_GRID_RIGHT = 0.995
ONOFF_GRID_WSPACE = 0.004
# Wider than ONOFF_GRID_WSPACE: pole labels sit flush against their own
# column's edge (see draw_other_dimensions_panel's GridSpec call for why),
# so this wspace IS the visible gap between neighbouring pole labels.
OTHER_DIMS_GRID_WSPACE = 0.03
TOPOMAP_PANEL_BOTTOM_BUDGET_IN = 0.06
# Shared by draw_onoff_panel/draw_other_dimensions_panel's panel_letter
# branch — both compute this identically, so it is a single constant rather
# than two copies that could silently drift apart (as they briefly did
# during the 2026-08-13 font-size pass).
TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN = (
    PANEL_LETTER_ROW_BUDGET_IN + HEADER_BUDGET_IN + PANEL_TITLE_BUDGET_IN + 3 * GAP_BUDGET_IN
)
# Same idea for the "suptitle" branch (standalone figure_onoff/
# figure_other_dimensions exports) — was duplicated inline in both
# draw_onoff_panel and draw_other_dimensions_panel; a single constant here
# is also what lets figure_onoff/figure_other_dimensions solve their own
# panel_height_in bottom-up (see solve_topomap_panel_height_in) without
# reproducing the formula a third time.
TOPOMAP_SUPTITLE_TOP_BUDGET_IN = (
    SUPTITLE_BUDGET_IN + LEGEND_ROW_BUDGET_IN + HEADER_BUDGET_IN + PANEL_TITLE_BUDGET_IN
    + 4 * GAP_BUDGET_IN
)
# Column width these topomap panels are tuned against, in inches — shared by
# the combined figure's panel B/C (as bc_width_in) and by the standalone
# figure_onoff/figure_other_dimensions exports, so both places produce
# exactly the same column width and the same "does this header/pole-label
# text fit" answer instead of two independently-guessed numbers drifting
# apart (which is what caused the standalone exports' headers to start
# colliding in the 2026-08-13 font pass while the combined figure, already
# re-derived, stayed fine).
TOPOMAP_PANEL_WIDTH_IN = 28.0
# Applied to the solved row height only (never to the column width itself,
# which must stay exactly what GridSpec will produce): without a hair of
# margin, floating-point rounding in GridSpec's own width solve can leave a
# fraction of a point of blank column edge, re-creating the exact gap this
# solver exists to remove. Also covers the row-height GridSpec loses to
# ROW_HSPACE between data rows (solve_topomap_panel_height_in's n_rows*height
# term does not itself subtract that hspace loss). 1.06 was the smallest
# value that stayed gap-free at ROW_HSPACE=0.08 when checked against the
# rendered PNG.
SQUARE_TOPOMAP_ROW_SAFETY = 1.06
# GridSpec vertical spacing between topomap data rows, shared by
# draw_onoff_panel/draw_other_dimensions_panel. Raised from 0.06 to make room
# for MARKER_TITLE_PAD_PT — that pad sits in exactly this inter-row gap for
# every row below the first (the first row's title clears the header instead,
# via PANEL_TITLE_BUDGET_IN).
ROW_HSPACE = 0.08


def fractions_from_budget(
    panel_height_in: float, top_budget_in: float, bottom_budget_in: float
) -> Tuple[float, float]:
    """
    Convert absolute-inch chrome budgets to the top/bottom fractions GridSpec needs.

    Parameters
    ----------
    panel_height_in : float
        Height of the figure or subfigure the grid will be drawn into.
    top_budget_in : float
        Space to reserve above the data rows, in inches.
    bottom_budget_in : float
        Space to reserve below the data rows, in inches.

    Returns
    -------
    tuple of float
        ``(top, bottom)`` fractions for ``GridSpec``.
    """
    top = 1.0 - top_budget_in / panel_height_in
    bottom = bottom_budget_in / panel_height_in
    return top, bottom


def solve_colorbar_row_ratio(n_data_rows: int, data_span_in: float, colorbar_budget_in: float) -> float:
    """
    Solve for the GridSpec height ratio that gives a colour-bar row a fixed inch height.

    ``GridSpec`` only takes relative weights, so getting an absolute row height
    (in inches) out of it means solving for the weight that produces it, given
    how much vertical space (``data_span_in``) is actually available and how
    many equal-weight data rows (weight 1.0 each) share it.

    Parameters
    ----------
    n_data_rows : int
        Number of data rows, each with height ratio 1.0.
    data_span_in : float
        Total height, in inches, available to the whole grid (data rows plus
        the colour-bar row).
    colorbar_budget_in : float
        Desired absolute height of the colour-bar row, in inches.

    Returns
    -------
    float
        Height ratio to assign the colour-bar row.
    """
    fraction = colorbar_budget_in / data_span_in
    return fraction * n_data_rows / (1.0 - fraction)


def solve_square_topomap_row_height_in(
    panel_width_in: float, n_cols: int, wspace: float
) -> float:
    """
    Solve the data-row height, in inches, that makes each topomap column square.

    ``mne.viz.plot_topomap`` always draws a circular head, so it is bound by
    the SMALLER of an axes' width and height and centres itself in the other
    — leaving blank margin on whichever side is bigger. A GridSpec column is
    wider than it is tall whenever the row height is left to "whatever space
    is left over" (the previous behaviour here), which is exactly what left
    visible blank strips beside every topomap in the combined figure. Solving
    for the row height that reproduces GridSpec's own column width — rather
    than picking a panel height first and letting columns fall out wider or
    narrower than that — is what makes the column square and the topomap
    fill it, without touching the column width (panel width / n_cols) at all.

    Parameters
    ----------
    panel_width_in : float
        Width, in inches, of the figure/subfigure the topomap GridSpec is
        drawn into. This is the dimension the caller must NOT enlarge — the
        function only ever solves the complementary row height.
    n_cols : int
        Number of topomap columns (families or dimensions).
    wspace : float
        GridSpec ``wspace`` used for this panel's grid (``ONOFF_GRID_WSPACE``
        or ``OTHER_DIMS_GRID_WSPACE``) — must match the value the caller
        passes to the actual ``GridSpec`` call, or the solved height will not
        match the column width GridSpec actually produces.

    Returns
    -------
    float
        Row height, in inches, that makes one data row as tall as one
        column is wide (times ``SQUARE_TOPOMAP_ROW_SAFETY`` for GridSpec
        rounding slack).
    """
    usable_width_in = panel_width_in * (TOPOMAP_GRID_RIGHT - TOPOMAP_GRID_LEFT)
    col_width_in = usable_width_in / (n_cols + (n_cols - 1) * wspace)
    return col_width_in * SQUARE_TOPOMAP_ROW_SAFETY


def solve_topomap_panel_height_in(
    top_budget_in: float, n_rows: int, row_height_in: float, colorbar_budget_in: float
) -> float:
    """
    Derive the subfigure height, in inches, that fits square topomap rows.

    Inverts the usual flow in this module (panel height picked first, chrome
    and data rows fitted into whatever is left): here the row height is fixed
    by ``solve_square_topomap_row_height_in`` first, and the panel height is
    whatever it takes to fit ``n_rows`` of that height plus chrome — so the
    combined figure's overall height is a consequence of "topomaps must be
    square", not a number chosen independently of it.

    Parameters
    ----------
    top_budget_in : float
        Chrome reserved above the data rows (``TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN``).
    n_rows : int
        Number of topomap rows (``n_per_family`` / ``n_per_dimension``).
    row_height_in : float
        Per-row height from ``solve_square_topomap_row_height_in``.
    colorbar_budget_in : float
        Height of the colour-bar row (``COLORBAR_ROW_BUDGET_IN``).

    Returns
    -------
    float
        Required subfigure height, in inches.
    """
    return (
        top_budget_in
        + n_rows * row_height_in
        + colorbar_budget_in
        + TOPOMAP_PANEL_BOTTOM_BUDGET_IN
    )

# Display grouping. This is for READING ONLY and is deliberately finer than the
# correction family: multiplicity is corrected over evoked (m=4) and sleep
# (m=19), never over these five groups.
DISPLAY_GROUPS: Dict[str, List[str]] = {
    "Evoked (ERP)": ["evoked/P1", "evoked/N1", "evoked/P3a", "evoked/P3b"],
    "Spectral (relative power)": [
        "sleep/psd_relative_delta",
        "sleep/psd_relative_theta",
        "sleep/psd_relative_alpha",
        "sleep/psd_relative_beta",
        "sleep/psd_relative_gamma",
    ],
    "Complexity / information": [
        "sleep/PE_theta",
        "sleep/PE_alpha",
        "sleep/PE_beta",
        "sleep/PE_gamma",
        "sleep/kolmogorov_complexity",
    ],
    "Connectivity (wSMI)": [
        "sleep/wsmi_theta",
        "sleep/wsmi_alpha",
        "sleep/wsmi_beta",
        "sleep/wsmi_gamma",
    ],
    "Slow waves": [
        "sleep/slowwaves_Density",
        "sleep/slowwaves_Duration",
        "sleep/slowwaves_Frequency",
        "sleep/slowwaves_PTP",
        "sleep/slowwaves_Slope",
    ],
}

MARKER_DISPLAY_NAMES: Dict[str, str] = {
    "evoked/P1": "P1",
    "evoked/N1": "N1",
    "evoked/P3a": "P3a",
    "evoked/P3b": "P3b",
    "sleep/psd_relative_delta": "PSD delta",
    "sleep/psd_relative_theta": "PSD theta",
    "sleep/psd_relative_alpha": "PSD alpha",
    "sleep/psd_relative_beta": "PSD beta",
    "sleep/psd_relative_gamma": "PSD gamma",
    "sleep/PE_theta": "PE theta",
    "sleep/PE_alpha": "PE alpha",
    "sleep/PE_beta": "PE beta",
    "sleep/PE_gamma": "PE gamma",
    "sleep/kolmogorov_complexity": "Kolmogorov",
    "sleep/wsmi_theta": "wSMI theta",
    "sleep/wsmi_alpha": "wSMI alpha",
    "sleep/wsmi_beta": "wSMI beta",
    "sleep/wsmi_gamma": "wSMI gamma",
    "sleep/slowwaves_Density": "SW density",
    "sleep/slowwaves_Duration": "SW duration",
    "sleep/slowwaves_Frequency": "SW frequency",
    "sleep/slowwaves_PTP": "SW amplitude",
    "sleep/slowwaves_Slope": "SW slope",
}

# Pole meaning per dimension, from the MDES scale convention in the project
# README. (low_pole, high_pole) == (negative t, positive t).
POLE_LABELS: Dict[str, Tuple[str, str]] = {
    "onoff": ("higher when OFF-task\n(mind-wandering)", "higher when ON-task"),
    "valence": ("higher for\nnegative affect", "higher for\npositive affect"),
    "selfother": ("higher when\nself-focused", "higher when\nother-focused"),
    "time": ("higher for\npast-oriented", "higher for\nfuture-oriented"),
    "confidence": ("higher when\nlow confidence", "higher when\nhigh confidence"),
}

# Compact pole names, for annotations where the full sentence in POLE_LABELS
# would not fit beside a neighbouring panel.
#
# confidence's poles are "low"/"high", not "unconfident"/"confident" — fixed
# 2026-08-13. The SART Task Design section of CLAUDE.md defines Q3's own
# poles literally as "(low ↔ high)"; a 2026-08-12 pass had flagged "low"/
# "high" as an inconsistent shortening and "corrected" it to "unconfident"/
# "confident", which was backwards — "unconfident" carries connotations
# (anxiety, insecurity) the slider never measured, it only asked how
# confident the self-assessment was. "low"/"high" reads correctly here
# because the column header directly above always says "Confidence".
SHORT_POLES: Dict[str, Tuple[str, str]] = {
    "onoff": ("off-task", "on-task"),
    "valence": ("negative", "positive"),
    "selfother": ("self-focused", "other-focused"),
    "time": ("past", "future"),
    "confidence": ("low", "high"),
}

# Two-line break points for SHORT_POLES entries too wide to render as one
# line at COLUMN_COLORBAR_LABEL_FONTSIZE within a fixed-width panel-C column
# (see add_column_colorbar) — a display-only line break, not a wording
# change, so SHORT_POLES itself (the canonical short pole text per CLAUDE.md)
# stays untouched. Entries absent here (e.g. "past"/"future", "low"/"high")
# are short enough to render on one line at any size this figure uses. A
# dict, not a generic char-count heuristic, because the break point has to
# land on a real syllable/hyphen boundary, which isn't derivable from the
# string alone.
COLUMN_COLORBAR_POLE_LINE_BREAKS: Dict[str, str] = {
    "self-focused": "self-\nfocused",
    "other-focused": "other-\nfocused",
}

# Dimensions shown on the secondary figure, in CANONICAL order (CLAUDE.md
# "Dimension Order") minus onoff, which has its own figure.
#
# This list used to be ordered by CBPT tier — concentrated, then diffuse, then
# null — and CLAUDE.md documented that as the one licensed deviation from
# canonical order, on the grounds that the ordering *was* the finding. It was
# reverted to canonical order on 2026-08-13 because those tiers were read off
# the quadratic-specification results, which have been retired: two of the six
# columns (valence_sq, time_sq) no longer exist, and the linear valence/time
# estimates are being recomputed without the quadratic covariates that were
# distorting them. Baking a superseded ranking into panel position would prime
# the reader toward a conclusion the current data has not re-established.
# Do not restore a tier ordering until the linear-only re-run is in and the
# tiers have actually been recomputed.
SECONDARY_DIMENSIONS = [
    "valence",
    "selfother",
    "time",
    "confidence",
]

# Column headers for the heatmap and other-dimensions figure: short enough to
# fit side by side at five columns. "Valence"/"Time" are single-line on
# purpose — per CLAUDE.md's "Dimension Labels & Pole Wording" table their
# canonical label is the bare word, with no "linear" qualifier. That qualifier
# was only ever needed to disambiguate from a quadratic column; with the
# quadratic terms gone (2026-08-13) there is nothing left to disambiguate
# against, so the bare word is now both canonical and unambiguous. xtick
# labels on a top-anchored axis grow upward from a shared bottom edge, so a
# shorter single-line label still bottom-aligns with its two-line neighbours —
# no visual inconsistency from the line-count mismatch.
HEATMAP_COLUMN_LABELS: Dict[str, str] = {
    "onoff": "On/Off-\nTask",
    "valence": "Valence",
    "selfother": "Self/\nOther",
    "time": "Time",
    "confidence": "Confi-\ndence",
}

# Family-group row labels for the heatmap, pre-wrapped to two short lines.
# DISPLAY_GROUPS keys are used as-is elsewhere (e.g. as figure_onoff column
# headers, where the panel is wide enough for one line); a generic
# "wrap at ' ('" heuristic used to be applied here instead, but it left
# "Complexity / information" — the one group name with no "(" — as a single
# 25-character line, wide enough to run into the family-boundary line in the
# narrower, embedded copy of this heatmap (the combined figure's Panel A).
HEATMAP_GROUP_LABELS: Dict[str, str] = {
    "Evoked (ERP)": "Evoked\n(ERP)",
    "Spectral (relative power)": "Spectral\n(rel. power)",
    "Complexity / information": "Complexity /\ninformation",
    "Connectivity (wSMI)": "Connectivity\n(wSMI)",
    "Slow waves": "Slow\nwaves",
}

# Column headers for the other-dimensions figure: full single-line names, since
# that figure has fewer, wider columns than the heatmap.
SECONDARY_COLUMN_LABELS: Dict[str, str] = {
    "valence": "Valence",
    "time": "Time",
    "selfother": "Self/Other",
    "confidence": "Confidence",
}

HEATMAP_DIMENSION_ORDER = [
    "onoff",
    "valence",
    "selfother",
    "time",
    "confidence",
]

def resolve_dimension_color(palette: dict, dimension: str) -> str:
    """
    Resolve a probe dimension to its project-wide colour.

    Quadratic predictors have their own lighter tint of the parent hue in
    ``color_palette.yaml``, so they read as the same MDES question in a
    different functional form rather than as an unrelated series.

    Parameters
    ----------
    palette : dict
        Parsed ``color_palette.yaml``.
    dimension : str
        Probe dimension name.

    Returns
    -------
    str
        Hex colour.
    """
    if dimension in palette["quadratic"]:
        return palette["quadratic"][dimension]
    return palette["dimensions"][dimension]


# =============================================================================
# HELPER FUNCTIONS — DATA
# =============================================================================


def load_config(config_path: Path) -> dict:
    """
    Load the Andrillon pipeline configuration.

    Parameters
    ----------
    config_path : Path
        Path to ``config_andrillon.yaml``.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_path) as handle:
        return yaml.safe_load(handle)


def load_color_palette(palette_path: Path) -> dict:
    """
    Load the project-wide colour palette.

    Parameters
    ----------
    palette_path : Path
        Path to ``color_palette.yaml`` at the repository root.

    Returns
    -------
    dict
        Parsed palette.
    """
    with open(palette_path) as handle:
        return yaml.safe_load(handle)


from helpers import get_model_folder_name  # noqa: E402


def find_model_dir(output_root: Path, target: str, formula: Optional[str] = None) -> Path:
    """
    Locate the results directory for one predictor-of-interest.

    When ``formula`` is given, the directory name is *derived* from it with the
    same helper the pipeline uses to create it, so exactly one candidate is
    named up front. Without it, this falls back to globbing on the
    ``__target_<name>`` suffix, which is ambiguous whenever results from two
    formulas coexist under ``output_root`` — the normal state of affairs, since
    changing the formula writes to a new directory and leaves the old one in
    place. That ambiguity is a hard error rather than a silent pick: after the
    quadratic terms were dropped (2026-08-13) the glob matched both
    ``onoff_valence_valence_sq_..._target_onoff`` and
    ``onoff_valence_..._target_onoff``, and choosing either implicitly would
    have mixed specifications in one figure.

    Parameters
    ----------
    output_root : Path
        Root of the cluster results tree.
    target : str
        Predictor of interest, e.g. ``"onoff"``.
    formula : str, optional
        LMM formula whose fixed effects name the directory. Strongly preferred;
        pass ``config["lmm"]["formula"]``.

    Returns
    -------
    Path
        The single matching model directory.
    """
    if formula is not None:
        expected = output_root / get_model_folder_name(formula, target)
        if not expected.is_dir():
            raise RuntimeError(
                f"No results directory for target '{target}' under the configured "
                f"formula. Expected: {expected.name}"
            )
        return expected

    matches = sorted(output_root.glob(f"*__target_{target}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one results directory for target '{target}', "
            f"found {len(matches)}: {[m.name for m in matches]}. Pass `formula` "
            f"to disambiguate — results from more than one specification are "
            f"present under {output_root}."
        )
    return matches[0]


def load_marker_result(model_dir: Path, marker_name: str) -> Optional[dict]:
    """
    Load one marker's cluster-permutation result.

    Parameters
    ----------
    model_dir : Path
        Directory for one predictor of interest.
    marker_name : str
        Marker in ``"<epoch_type>/<marker>"`` form.

    Returns
    -------
    dict or None
        Result payload, or None when the marker was not run for this target.
    """
    marker_dir = model_dir / marker_name.replace("/", "_")
    pkl_path = marker_dir / "results.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)


def representative_p_value(result: dict) -> float:
    """
    Reduce a marker to the single p-value that represents it in its family.

    Only the maximum-statistic cluster is calibrated against the max-statistic
    permutation null; sub-maximal clusters carry conservative p-values. A marker
    with no candidate cluster contributed no evidence and is represented by 1.0,
    but stays in the family because it was tested.

    Parameters
    ----------
    result : dict
        Payload from ``load_marker_result``.

    Returns
    -------
    float
        Representative (uncorrected) p-value for the marker.
    """
    cluster_p_values = np.asarray(result["cluster_p_values"], dtype=float)
    if cluster_p_values.size == 0:
        return 1.0
    return float(np.min(cluster_p_values))


def collect_target_results(
    model_dir: Path, marker_names: Sequence[str], alpha: float
) -> pd.DataFrame:
    """
    Assemble every marker's result for one target and correct within family.

    Benjamini-Hochberg (primary) and Benjamini-Yekutieli (dependence-robust
    sensitivity) are both applied, separately within each declared family.

    Parameters
    ----------
    model_dir : Path
        Directory for one predictor of interest.
    marker_names : Sequence[str]
        Declared family membership, in ``"<epoch_type>/<marker>"`` form.
    alpha : float
        Family-wise alpha level.

    Returns
    -------
    pd.DataFrame
        One row per marker with raw and corrected p-values.
    """
    records = []
    for marker_name in marker_names:
        result = load_marker_result(model_dir, marker_name)
        if result is None:
            raise RuntimeError(
                f"Declared marker '{marker_name}' has no results.pkl under "
                f"{model_dir.name}. Refusing to shrink the correction family."
            )
        cluster_stats = np.asarray(result["cluster_stats"], dtype=float)
        cluster_p_values = np.asarray(result["cluster_p_values"], dtype=float)
        if cluster_p_values.size == 0:
            best_stat = np.nan
            best_n_elec = 0
        else:
            best_idx = int(np.argmin(cluster_p_values))
            best_stat = float(cluster_stats[best_idx])
            best_n_elec = int(len(result["clusters"][best_idx]))
        records.append(
            {
                "marker_name": marker_name,
                "marker_type": result["marker_type"],
                "p_raw": representative_p_value(result),
                "cluster_stat": best_stat,
                "n_electrodes": best_n_elec,
                "n_clusters": int(result["n_clusters"]),
                "n_subjects": int(result["n_subjects"]),
            }
        )

    df = pd.DataFrame(records)
    df["p_bh"] = np.nan
    df["p_by"] = np.nan
    for family, family_df in df.groupby("marker_type"):
        _, q_bh, _, _ = multipletests(
            family_df["p_raw"].to_numpy(), alpha=alpha, method="fdr_bh"
        )
        _, q_by, _, _ = multipletests(
            family_df["p_raw"].to_numpy(), alpha=alpha, method="fdr_by"
        )
        df.loc[family_df.index, "p_bh"] = q_bh
        df.loc[family_df.index, "p_by"] = q_by
    df["sig_bh"] = df["p_bh"] < alpha
    df["sig_by"] = df["p_by"] < alpha
    return df


def significant_channel_masks(
    result: dict, alpha: float, sig_bh: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a marker's raw-significant channels into "survives BH" vs. "raw only".

    Every channel returned here belongs to a cluster with its own permutation
    p-value below alpha — that part of the gate never changes. What changes is
    which of the two returned masks those channels land in: if the marker's
    representative p-value survives the family-level BH correction, they are
    the marker's headline result (drawn filled); if it does not, they are
    still real, nominally-significant channels that a marker-wise multiple-
    comparisons correction wipes out on its own (drawn hollow) — the same
    "distributed but real" signal the family-level omnibus test exists to
    catch (see CLAUDE.md). A marker with no candidate cluster below alpha at
    all returns two empty masks; callers use that to leave the panel blank.

    Parameters
    ----------
    result : dict
        Payload from ``load_marker_result``.
    alpha : float
        Significance threshold.
    sig_bh : bool
        Whether this marker survived the family-level BH correction.

    Returns
    -------
    tuple of np.ndarray
        ``(mask_survives_bh, mask_raw_only)`` boolean masks over channels.
    """
    n_channels = len(result["t_stats"])
    mask_raw = np.zeros(n_channels, dtype=bool)
    for cluster, p_value in zip(result["clusters"], result["cluster_p_values"]):
        if p_value < alpha:
            mask_raw[cluster] = True
    if sig_bh:
        return mask_raw, np.zeros(n_channels, dtype=bool)
    return np.zeros(n_channels, dtype=bool), mask_raw


# =============================================================================
# HELPER FUNCTIONS — DRAWING
# =============================================================================


def draw_topomap_panel(
    ax: plt.Axes,
    result: dict,
    mask_survives_bh: np.ndarray,
    mask_raw_only: np.ndarray,
    vlim: Tuple[float, float],
) -> matplotlib.image.AxesImage:
    """
    Render one topography using the pipeline's exact rendering parameters.

    Significance is drawn, never stated in text: a filled dot marks a channel
    whose marker survives the family-level BH correction; a hollow ring marks
    a channel that is part of a raw-significant cluster (p < alpha) whose
    marker does not. Nothing is drawn for channels in neither mask.

    Parameters
    ----------
    ax : plt.Axes
        Axis to draw into.
    result : dict
        Payload from ``load_marker_result``.
    mask_survives_bh : np.ndarray
        Boolean channel mask, drawn filled.
    mask_raw_only : np.ndarray
        Boolean channel mask, drawn hollow.
    vlim : tuple of float
        Shared (vmin, vmax) for the figure this panel belongs to.

    Returns
    -------
    matplotlib.image.AxesImage
        The rendered image, for colour-bar construction.
    """
    info, t_stats = validate_montage_and_channels(result["info"], result["t_stats"])

    # Same NaN handling as plot_cluster_topomap: MNE cannot interpolate NaN.
    t_stats_plot = np.copy(np.asarray(t_stats, dtype=float))
    t_stats_plot[np.isnan(t_stats_plot)] = 0.0

    image, _ = mne.viz.plot_topomap(
        t_stats_plot,
        info,
        axes=ax,
        show=False,
        vlim=vlim,
        mask=mask_survives_bh,
        mask_params=MASK_PARAMS,
        **TOPOMAP_KWARGS,
    )
    overlay_uncorrected_only_channels(
        ax=ax,
        info=info,
        n_channels=len(t_stats),
        mask_uncorrected=mask_raw_only,
        mask_corrected=np.zeros(len(t_stats), dtype=bool),
        markersize=HOLLOW_MARKERSIZE,
        markeredgewidth=HOLLOW_MARKEREDGEWIDTH,
    )
    return image


def annotate_column_groups(
    fig: plt.Figure,
    grid: GridSpec,
    col_labels: Sequence[str],
    n_data_rows: int,
    panel_height_in: float,
    sub_labels: Optional[Sequence[str]] = None,
    draw_separators: bool = True,
) -> None:
    """
    Label each column of a grid once and draw separators between columns.

    A group name stated once above its column is chart furniture, like an
    axis label — it replaces repeating the same caption under every panel in
    that column.

    Parameters
    ----------
    fig : plt.Figure
        Figure the grid belongs to.
    grid : GridSpec
        Grid laying out the topomap panels (data rows only; a further row for
        the colour bar is allowed to exist below them).
    col_labels : Sequence[str]
        One label per column, left to right.
    n_data_rows : int
        Number of topomap rows in the grid (excludes any colour-bar row).
    panel_height_in : float
        Height in inches of the figure/subfigure ``grid`` was built for — used
        to convert the fixed ``GAP_BUDGET_IN`` into a fraction of this panel.
    sub_labels : Sequence[str], optional
        One compact secondary label per column (e.g. pole hints), drawn
        smaller and directly below the main label.
    draw_separators : bool
        Whether to draw vertical separator lines between columns.
    """
    bottoms, tops, lefts, rights = grid.get_grid_positions(fig)
    gap_fraction = GAP_BUDGET_IN / panel_height_in
    # The panel's own per-axes title (marker name) floats above tops[0] using
    # its own budget (PANEL_TITLE_BUDGET_IN) — the header has to clear THAT
    # space, not just the bare axes edge, or the two collide.
    header_y = tops[0] + (PANEL_TITLE_BUDGET_IN + GAP_BUDGET_IN) / panel_height_in
    sep_top = tops[0] + gap_fraction * 0.2
    sep_bottom = bottoms[n_data_rows - 1]

    for col, label in enumerate(col_labels):
        x_center = (lefts[col] + rights[col]) / 2
        fig.text(
            x_center,
            header_y,
            label,
            ha="center",
            va="bottom",
            fontsize=HEADER_FONTSIZE,
            fontweight="bold",
            linespacing=1.15,
        )
        if sub_labels is not None:
            fig.text(
                x_center,
                header_y - gap_fraction,
                sub_labels[col],
                ha="center",
                va="top",
                fontsize=LEGEND_FONTSIZE * 0.75,
                color=NEUTRAL_GRAY,
            )
        if draw_separators and col > 0:
            x_sep = (rights[col - 1] + lefts[col]) / 2
            # NOTE: must use transSubfigure, not transFigure. For a SubFigure,
            # .transFigure resolves relative to the OUTER (root) figure, not
            # the subfigure's own box, so a line placed with fractions meant
            # to be local to this panel would instead be stretched across
            # whatever else shares the parent figure (verified empirically —
            # this is what caused separator lines to cut through unrelated
            # panels in the combined figure). transSubfigure is local in both
            # cases (it is the same object as transFigure for a plain,
            # non-sub Figure), so this is always correct.
            fig.add_artist(
                plt.Line2D(
                    [x_sep, x_sep],
                    [sep_bottom, sep_top],
                    transform=fig.transSubfigure,
                    color=NEUTRAL_GRAY,
                    linewidth=1.0,
                )
            )


def compute_shared_vlim(results: Sequence[dict]) -> Tuple[float, float]:
    """
    Compute one symmetric colour range shared by every panel of a figure.

    A shared range is what makes panels comparable by eye; a per-panel range
    would make a weak effect look as strong as a strong one.

    Parameters
    ----------
    results : Sequence[dict]
        Payloads for every panel on the figure.

    Returns
    -------
    tuple of float
        Symmetric (vmin, vmax).
    """
    abs_max = 0.0
    for result in results:
        marker_max = np.nanmax(np.abs(np.asarray(result["t_stats"], dtype=float)))
        if not np.isnan(marker_max):
            abs_max = max(abs_max, float(marker_max))
    if abs_max == 0.0:
        abs_max = 1.0
    return -abs_max, abs_max


def add_directional_colorbar(
    fig: plt.Figure,
    image: matplotlib.image.AxesImage,
    cax: plt.Axes,
    dimension: str,
    label_poles: bool = True,
) -> None:
    """
    Draw a horizontal colour bar whose ends state what each sign means.

    A bare "t-statistic" colour bar forces the reader to remember the coding of
    the probe scale. Naming both poles removes that step, which is the single
    most common source of sign confusion when reading these maps.

    Parameters
    ----------
    fig : plt.Figure
        Figure being drawn.
    image : matplotlib.image.AxesImage
        Any rendered panel sharing the figure's colour scale.
    cax : plt.Axes
        Axis reserved for the colour bar.
    dimension : str
        Probe dimension, used to look up the pole labels.
    label_poles : bool
        When False, only the neutral t-statistic label is drawn.
    """
    # The visible gradient is a thin strip pinned to the top of the reserved
    # row (COLORBAR_BAR_IN of COLORBAR_ROW_BUDGET_IN); a colour bar as tall as
    # its own labels reads as a second data panel instead of a legend.
    bar_fraction = COLORBAR_BAR_IN / COLORBAR_ROW_BUDGET_IN
    cax.axis("off")
    bar_ax = cax.inset_axes([0, 1 - bar_fraction, 1, bar_fraction])
    colorbar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
    colorbar.outline.set_linewidth(0.6)
    bar_ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE, length=2, width=0.6)

    if not label_poles:
        return

    low_label, high_label = POLE_LABELS[dimension]
    bar_ax.text(
        -0.03,
        0.5,
        f"← {low_label}",
        transform=bar_ax.transAxes,
        ha="right",
        va="center",
        fontsize=COLORBAR_LABEL_FONTSIZE,
        linespacing=1.2,
    )
    bar_ax.text(
        1.03,
        0.5,
        f"{high_label} →",
        transform=bar_ax.transAxes,
        ha="left",
        va="center",
        fontsize=COLORBAR_LABEL_FONTSIZE,
        linespacing=1.2,
    )


def add_column_colorbar(
    fig: plt.Figure,
    image: matplotlib.image.AxesImage,
    cax: plt.Axes,
    dimension: str,
) -> None:
    """
    Draw a narrow, per-dimension colour bar with its poles named beneath it.

    Every dimension gets its own labelled bar — the same device used for the
    single onoff figure, repeated per column — instead of one shared, unlabelled
    bar plus a caption of arrows disconnected from any actual colour. The
    colour *scale* (vmin/vmax) is still shared across all columns of the
    figure this belongs to; only the bar and its labels are drawn per column,
    so panels stay comparable while each one is still self-explanatory.

    Parameters
    ----------
    fig : plt.Figure
        Figure being drawn.
    image : matplotlib.image.AxesImage
        Any rendered panel sharing the figure's (shared) colour scale.
    cax : plt.Axes
        Axis reserved for this column's colour bar.
    dimension : str
        Probe dimension, used to look up the pole labels.
    """
    # Thin gradient strip at the top of the reserved row, poles named below it
    # (see add_directional_colorbar for why the bar is deliberately thin).
    bar_fraction = COLORBAR_BAR_IN / COLORBAR_ROW_BUDGET_IN
    cax.axis("off")
    bar_ax = cax.inset_axes([0, 1 - bar_fraction, 1, bar_fraction])
    colorbar = fig.colorbar(image, cax=bar_ax, orientation="horizontal")
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_ticks([])

    text_y = 1 - bar_fraction - (GAP_BUDGET_IN / COLORBAR_ROW_BUDGET_IN) * 0.5
    low_label, high_label = SHORT_POLES[dimension]
    low_label = COLUMN_COLORBAR_POLE_LINE_BREAKS.get(low_label, low_label)
    high_label = COLUMN_COLORBAR_POLE_LINE_BREAKS.get(high_label, high_label)
    cax.text(
        0.0,
        text_y,
        low_label,
        transform=cax.transAxes,
        ha="left",
        va="top",
        fontsize=COLUMN_COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        linespacing=1.1,
    )
    cax.text(
        1.0,
        text_y,
        high_label,
        transform=cax.transAxes,
        ha="right",
        va="top",
        fontsize=COLUMN_COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
        color="black",
        linespacing=1.1,
    )


def add_significance_legend(fig: plt.Figure, y: float) -> None:
    """
    State the filled/hollow significance dots as text, on its own row.

    A graphical proxy-marker legend anchored to a figure corner kept landing
    on top of the colour bar's pole labels, which also live in a figure
    corner — every corner of this layout is already claimed by something.
    The right edge stays open, but sharing the suptitle's own row with it
    only works while both strings are short: at the current (much larger)
    SUPTITLE_FONTSIZE/LEGEND_FONTSIZE, "B · On/off-task: effect per family"
    and "survives correction / p < .05, uncorrected" are each wide enough
    that a shared row makes them collide. Callers instead give this its own
    row below the suptitle, sized by LEGEND_ROW_BUDGET_IN.

    Parameters
    ----------
    fig : plt.Figure
        Figure to attach the legend to.
    y : float
        Figure-fraction y-coordinate, top-aligned.
    """
    fig.text(
        0.995,
        y,
        "●  survives correction     ○  p < .05, uncorrected",
        ha="right",
        va="top",
        fontsize=LEGEND_FONTSIZE,
        fontweight="bold",
        color="black",
    )


# =============================================================================
# FIGURES
# =============================================================================


def draw_onoff_panel(
    fig: plt.Figure,
    panel_height_in: float,
    model_dir: Path,
    summary: pd.DataFrame,
    alpha: float,
    n_per_family: int = N_TOP_MARKERS_PER_FAMILY,
    suptitle: Optional[str] = "On/off-task: effect per family",
    panel_letter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Draw the strongest on/off-task effect in each marker family into ``fig``.

    Selection rule, applied before looking at any topography: within each of
    the five display families, rank markers by absolute cluster statistic and
    take the top ``n_per_family``. Ranking within family — rather than pooling
    all 23 markers and taking an overall top-N — is what keeps every family
    represented: a pooled ranking would let the largest family (sleep, 19
    markers) crowd out the smallest (evoked, 4), which says nothing about
    whether that family carries a real effect.

    Significance is drawn, not stated (``significant_channel_masks``): filled
    dots mark a channel whose marker survives the family-wise BH correction;
    hollow dots mark one with a raw-significant cluster (p < alpha) that does
    not survive on its own. A panel is left blank only when nothing clears
    even the uncorrected bar.

    Chrome (suptitle, column headers, colour bar) is budgeted in inches
    (module-level ``*_BUDGET_IN`` constants) and converted to the fractions
    ``GridSpec`` needs via ``panel_height_in`` — so topomaps get everything
    left over, and get it back proportionally if ``panel_height_in`` shrinks.

    Parameters
    ----------
    fig : plt.Figure
        Figure (or subfigure) to draw into.
    panel_height_in : float
        Actual height of ``fig`` in inches.
    model_dir : Path
        Results directory for the onoff target.
    summary : pd.DataFrame
        Output of ``collect_target_results`` for onoff.
    alpha : float
        Significance threshold.
    n_per_family : int
        Number of topographies per family.
    suptitle : str, optional
        Descriptive title drawn centred above the panel. Used only for the
        standalone export, which has no sibling panels to distinguish itself
        from; ignored when ``panel_letter`` is given.
    panel_letter : str, optional
        Bare panel letter (e.g. ``"B"``) drawn bold at the panel's top-left
        corner per the project's multi-panel-figure convention — used instead
        of ``suptitle`` when this panel sits inside a combined figure, where a
        full descriptive sentence per panel is redundant with the column
        headers already naming what's shown.

    Returns
    -------
    pd.DataFrame
        The markers actually plotted, with their family and within-family rank.
    """
    group_order = list(DISPLAY_GROUPS.keys())

    rows = []
    for family_label, members in DISPLAY_GROUPS.items():
        family_df = summary[summary["marker_name"].isin(members)].copy()
        family_df["abs_stat"] = family_df["cluster_stat"].abs()
        top = family_df.sort_values("abs_stat", ascending=False).head(n_per_family)
        for rank, (_, row) in enumerate(top.iterrows()):
            record = row.to_dict()
            record["display_group"] = family_label
            record["rank"] = rank
            rows.append(record)
    selected = pd.DataFrame(rows)

    results = {
        name: load_marker_result(model_dir, name) for name in selected["marker_name"]
    }
    vlim = compute_shared_vlim(results.values())

    n_cols = len(group_order)
    n_rows = n_per_family

    if panel_letter:
        top_budget = TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN
    elif suptitle:
        top_budget = TOPOMAP_SUPTITLE_TOP_BUDGET_IN
    else:
        top_budget = HEADER_BUDGET_IN + PANEL_TITLE_BUDGET_IN + 2 * GAP_BUDGET_IN
    bottom_budget = TOPOMAP_PANEL_BOTTOM_BUDGET_IN
    top, bottom = fractions_from_budget(panel_height_in, top_budget, bottom_budget)
    data_span_in = panel_height_in - top_budget - bottom_budget
    colorbar_ratio = solve_colorbar_row_ratio(n_rows, data_span_in, COLORBAR_ROW_BUDGET_IN)

    grid = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1.0] * n_rows + [colorbar_ratio],
        hspace=ROW_HSPACE,
        wspace=ONOFF_GRID_WSPACE,
        left=TOPOMAP_GRID_LEFT,
        right=TOPOMAP_GRID_RIGHT,
        top=top,
        bottom=bottom,
    )

    image = None
    for _, row in selected.iterrows():
        col = group_order.index(row["display_group"])
        ax = fig.add_subplot(grid[int(row["rank"]), col])
        ax.set_title(MARKER_DISPLAY_NAMES[row["marker_name"]], fontsize=PANEL_TITLE_FONTSIZE, pad=MARKER_TITLE_PAD_PT)
        if row["p_raw"] >= alpha:
            # No cluster below alpha even before correction: nothing to show.
            ax.axis("off")
            continue
        result = results[row["marker_name"]]
        mask_bh, mask_raw_only = significant_channel_masks(
            result, alpha, sig_bh=bool(row["sig_bh"])
        )
        image = draw_topomap_panel(ax, result, mask_bh, mask_raw_only, vlim)

    col_labels = [HEATMAP_GROUP_LABELS[g] for g in group_order]
    annotate_column_groups(fig, grid, col_labels, n_data_rows=n_rows, panel_height_in=panel_height_in)

    cax = fig.add_subplot(grid[n_rows, 1: n_cols - 1])
    add_directional_colorbar(fig, image, cax, "onoff")

    if panel_letter is not None:
        fig.text(0.008, 1.0, panel_letter, ha="left", va="top", fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold")
        add_significance_legend(fig, y=1.0)
    elif suptitle is not None:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE, fontweight="bold", y=1.0, va="top")
        legend_y = 1.0 - (SUPTITLE_BUDGET_IN + GAP_BUDGET_IN) / panel_height_in
        add_significance_legend(fig, y=legend_y)

    return selected


def figure_onoff(
    model_dir: Path,
    summary: pd.DataFrame,
    alpha: float,
    output_path: Path,
    n_per_family: int = N_TOP_MARKERS_PER_FAMILY,
) -> pd.DataFrame:
    """
    Standalone export of the onoff panel. See ``draw_onoff_panel``.

    Parameters
    ----------
    model_dir : Path
        Results directory for the onoff target.
    summary : pd.DataFrame
        Output of ``collect_target_results`` for onoff.
    alpha : float
        Significance threshold.
    output_path : Path
        Base path (extension replaced per format).
    n_per_family : int
        Number of topographies per family.

    Returns
    -------
    pd.DataFrame
        The markers actually plotted, with their family and within-family rank.
    """
    # Same TOPOMAP_PANEL_WIDTH_IN/row-height solve the combined figure's
    # panel B uses (see figure_combined) — panel_height_in is a consequence
    # of "topomap rows must be as tall as their column is wide", not a
    # separately-guessed formula that can drift out of sync with it.
    row_height_in = solve_square_topomap_row_height_in(
        TOPOMAP_PANEL_WIDTH_IN, len(DISPLAY_GROUPS), ONOFF_GRID_WSPACE
    )
    panel_height_in = solve_topomap_panel_height_in(
        TOPOMAP_SUPTITLE_TOP_BUDGET_IN, n_per_family, row_height_in, COLORBAR_ROW_BUDGET_IN
    )
    fig = plt.figure(figsize=(TOPOMAP_PANEL_WIDTH_IN, panel_height_in))
    selected = draw_onoff_panel(fig, panel_height_in, model_dir, summary, alpha, n_per_family)
    save_figure_multiformat(fig, output_path)
    plt.close(fig)
    return selected


def draw_other_dimensions_panel(
    fig: plt.Figure,
    panel_height_in: float,
    output_root: Path,
    summaries: Dict[str, pd.DataFrame],
    alpha: float,
    omnibus: Optional[pd.DataFrame],
    formula: str,
    n_per_dimension: int = N_TOP_MARKERS_PER_DIMENSION,
    suptitle: Optional[str] = "Other dimensions: strongest markers",
    panel_letter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Draw the strongest available markers for every other dimension into ``fig``.

    Selection rule: within each dimension, rank all 23 markers by absolute
    cluster statistic and take the top ``n_per_dimension`` — the same rule
    used for onoff, applied per dimension instead of per family. A dimension
    with no signal even before correction shows three blank panels; one whose
    signal is real but distributed (no single marker survives the family-wise
    correction) shows hollow-dot topographies instead of being represented by
    a single cherry-picked "best" marker whose significance would have to be
    asserted in a caption. Significance is drawn (``significant_channel_masks``:
    filled = survives BH, hollow = raw p < alpha only), never asserted in text.

    Each column gets its own colour bar (``add_column_colorbar``): the colour
    *scale* is still shared across all six dimensions (so panels stay
    comparable), but every column states its own poles under an actual
    gradient, the same way the onoff figure does with its single bar.

    Parameters
    ----------
    fig : plt.Figure
        Figure (or subfigure) to draw into.
    panel_height_in : float
        Actual height of ``fig`` in inches — see ``draw_onoff_panel`` for why
        chrome is budgeted from this rather than a fixed fraction.
    output_root : Path
        Root of the cluster results tree.
    summaries : Dict[str, pd.DataFrame]
        Per-target outputs of ``collect_target_results``.
    alpha : float
        Significance threshold.
    omnibus : pd.DataFrame or None
        Family-level omnibus results, when available. Folded into the
        returned table (for the supplementary CSV) but not drawn on the
        figure.
    n_per_dimension : int
        Number of topographies per dimension.
    suptitle : str, optional
        Descriptive title drawn centred above the panel. Used only for the
        standalone export; ignored when ``panel_letter`` is given.
    panel_letter : str, optional
        Bare panel letter drawn bold at the panel's top-left corner — see
        ``draw_onoff_panel`` for the rationale.

    Returns
    -------
    pd.DataFrame
        The markers actually plotted, with dimension and within-dimension rank.
    """
    rows = []
    results: Dict[Tuple[str, str], dict] = {}
    for dimension in SECONDARY_DIMENSIONS:
        summary = summaries[dimension].copy()
        summary["abs_stat"] = summary["cluster_stat"].abs()
        top = summary.sort_values("abs_stat", ascending=False).head(n_per_dimension)
        model_dir = find_model_dir(output_root, dimension, formula)
        for rank, (_, row) in enumerate(top.iterrows()):
            record = row.to_dict()
            record["dimension"] = dimension
            record["rank"] = rank
            rows.append(record)
            results[(dimension, row["marker_name"])] = load_marker_result(
                model_dir, row["marker_name"]
            )

    selected = pd.DataFrame(rows)

    if omnibus is not None:
        omnibus_sleep = omnibus[omnibus["family"] == "sleep"].set_index("target")
        selected["omnibus_count_p_bh"] = selected["dimension"].map(
            omnibus_sleep["count_p_bh"]
        )

    n_cols = len(SECONDARY_DIMENSIONS)
    n_rows = n_per_dimension
    vlim = compute_shared_vlim(results.values())

    if panel_letter:
        top_budget = TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN
    elif suptitle:
        top_budget = TOPOMAP_SUPTITLE_TOP_BUDGET_IN
    else:
        top_budget = HEADER_BUDGET_IN + PANEL_TITLE_BUDGET_IN + 2 * GAP_BUDGET_IN
    bottom_budget = TOPOMAP_PANEL_BOTTOM_BUDGET_IN
    top, bottom = fractions_from_budget(panel_height_in, top_budget, bottom_budget)
    data_span_in = panel_height_in - top_budget - bottom_budget
    colorbar_ratio = solve_colorbar_row_ratio(n_rows, data_span_in, COLORBAR_ROW_BUDGET_IN)

    grid = GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1.0] * n_rows + [colorbar_ratio],
        hspace=ROW_HSPACE,
        # Pole labels are anchored flush against their own column's edge
        # (ha="left" at x=0, ha="right" at x=1) rather than inset, so the
        # visible gap between one column's high-pole label and the next
        # column's low-pole label IS wspace's physical gap, not a function
        # of column width — wspace=0 makes adjacent labels touch exactly
        # (verified: "extremenegative" with zero pixels between). 0.03 is
        # the smallest tested value that keeps a real gap at
        # COLUMN_COLORBAR_LABEL_FONTSIZE=34 and the current panel width.
        wspace=OTHER_DIMS_GRID_WSPACE,
        left=TOPOMAP_GRID_LEFT,
        right=TOPOMAP_GRID_RIGHT,
        top=top,
        bottom=bottom,
    )

    # Keyed per dimension when that dimension has at least one marker with a
    # candidate cluster below alpha (drawn, whether or not it survives BH); a
    # dimension where nothing clears even the uncorrected bar never gets an
    # entry and falls back to reference_image below — the colour SCALE is
    # shared across every panel regardless, so any drawn image is a valid
    # mappable for any column's colour bar.
    images: Dict[str, matplotlib.image.AxesImage] = {}
    reference_image: Optional[matplotlib.image.AxesImage] = None
    for _, row in selected.iterrows():
        col = SECONDARY_DIMENSIONS.index(row["dimension"])
        ax = fig.add_subplot(grid[int(row["rank"]), col])
        ax.set_title(MARKER_DISPLAY_NAMES[row["marker_name"]], fontsize=PANEL_TITLE_FONTSIZE, pad=MARKER_TITLE_PAD_PT)
        if row["p_raw"] >= alpha:
            # No cluster below alpha even before correction: nothing to show.
            ax.axis("off")
            continue
        result = results[(row["dimension"], row["marker_name"])]
        mask_bh, mask_raw_only = significant_channel_masks(
            result, alpha, sig_bh=bool(row["sig_bh"])
        )
        image = draw_topomap_panel(ax, result, mask_bh, mask_raw_only, vlim)
        images[row["dimension"]] = image
        reference_image = image

    col_labels = [SECONDARY_COLUMN_LABELS[d] for d in SECONDARY_DIMENSIONS]
    annotate_column_groups(fig, grid, col_labels, n_data_rows=n_rows, panel_height_in=panel_height_in)

    for col, dimension in enumerate(SECONDARY_DIMENSIONS):
        cax = fig.add_subplot(grid[n_rows, col])
        add_column_colorbar(fig, images.get(dimension, reference_image), cax, dimension)

    if panel_letter is not None:
        fig.text(0.008, 1.0, panel_letter, ha="left", va="top", fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold")
        add_significance_legend(fig, y=1.0)
    elif suptitle is not None:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE, fontweight="bold", y=1.0, va="top")
        legend_y = 1.0 - (SUPTITLE_BUDGET_IN + GAP_BUDGET_IN) / panel_height_in
        add_significance_legend(fig, y=legend_y)

    return selected


def figure_other_dimensions(
    output_root: Path,
    summaries: Dict[str, pd.DataFrame],
    alpha: float,
    omnibus: Optional[pd.DataFrame],
    output_path: Path,
    formula: str,
    n_per_dimension: int = N_TOP_MARKERS_PER_DIMENSION,
) -> pd.DataFrame:
    """
    Standalone export of the other-dimensions panel. See ``draw_other_dimensions_panel``.

    Parameters
    ----------
    output_root : Path
        Root of the cluster results tree.
    summaries : Dict[str, pd.DataFrame]
        Per-target outputs of ``collect_target_results``.
    alpha : float
        Significance threshold.
    omnibus : pd.DataFrame or None
        Family-level omnibus results, when available.
    output_path : Path
        Base path (extension replaced per format).
    n_per_dimension : int
        Number of topographies per dimension.

    Returns
    -------
    pd.DataFrame
        The markers actually plotted, with dimension and within-dimension rank.
    """
    # Same TOPOMAP_PANEL_WIDTH_IN/row-height solve the combined figure's
    # panel C uses (see figure_combined) — this is also what gives each
    # column's per-dimension colour bar room for COLUMN_COLORBAR_LABEL_FONTSIZE's
    # pole-label pairs (the widest being "self-focused"/"other-focused",
    # wrapped to two lines via COLUMN_COLORBAR_POLE_LINE_BREAKS).
    row_height_in = solve_square_topomap_row_height_in(
        TOPOMAP_PANEL_WIDTH_IN, len(SECONDARY_DIMENSIONS), OTHER_DIMS_GRID_WSPACE
    )
    panel_height_in = solve_topomap_panel_height_in(
        TOPOMAP_SUPTITLE_TOP_BUDGET_IN, n_per_dimension, row_height_in, COLORBAR_ROW_BUDGET_IN
    )
    fig = plt.figure(figsize=(TOPOMAP_PANEL_WIDTH_IN, panel_height_in))
    selected = draw_other_dimensions_panel(
        fig, panel_height_in, output_root, summaries, alpha, omnibus, formula,
        n_per_dimension,
    )
    save_figure_multiformat(fig, output_path)
    plt.close(fig)
    return selected


def draw_heatmap(
    fig: plt.Figure,
    panel_height_in: float,
    summaries: Dict[str, pd.DataFrame],
    palette: dict,
    alpha: float,
    title: Optional[str] = "Marker × dimension evidence map",
    left_margin: float = 0.34,
    tick_fontsize: float = HEATMAP_TICK_FONTSIZE,
    family_fontsize: float = HEATMAP_FAMILY_LABEL_FONTSIZE,
    panel_letter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Draw the marker-by-dimension evidence heatmap into ``fig``.

    Hue encodes the dimension (fixed project-wide assignment), never
    significance. Evidence strength is the colour's opacity; significance is a
    separate glyph (``*``/``**``, drawn in-cell), so a reader can read "which
    dimension" and "is it significant" independently.

    Parameters
    ----------
    fig : plt.Figure
        Figure (or subfigure) to draw into.
    panel_height_in : float
        Actual height of ``fig`` in inches — used to convert the fixed
        ``HEATMAP_TOP_BUDGET_IN``/``HEATMAP_BOTTOM_BUDGET_IN`` chrome budgets
        into the fractions ``subplots_adjust`` needs (see ``draw_onoff_panel``
        for why this is budgeted in inches rather than a hardcoded fraction).
    summaries : Dict[str, pd.DataFrame]
        Per-target outputs of ``collect_target_results``.
    palette : dict
        Parsed ``color_palette.yaml``.
    alpha : float
        Significance threshold.
    title : str, optional
        Descriptive title drawn centred above the panel. Used only for the
        standalone export; ignored when ``panel_letter`` is given.
    left_margin : float
        Fraction of the figure width reserved for the y-axis marker labels and
        family-group labels. A narrower panel (e.g. embedded in a combined
        figure) needs a larger fraction for the same label text to still clear
        the family-boundary line.
    tick_fontsize : float
        Font size for the row/column tick labels. A narrower embedded panel
        needs this turned down a step — column labels like "quadratic" wrap
        onto neighbouring columns at the standalone figure's font size once
        the panel is narrow enough.
    family_fontsize : float
        Font size for the family-group row label ("Evoked (ERP)"). Kept
        independent of ``tick_fontsize`` (rather than a fraction of it) so a
        panel that must stay a fixed width (the combined figure's panel A —
        see figure_combined) can turn this down a step without also shrinking
        the column tick labels it isn't competing with for space.
    panel_letter : str, optional
        Bare panel letter drawn bold at the panel's top-left corner — see
        ``draw_onoff_panel`` for the rationale.

    Returns
    -------
    pd.DataFrame
        Tidy table backing the heatmap.
    """
    ordered_markers: List[str] = []
    group_boundaries: List[Tuple[str, int, int]] = []
    for group, members in DISPLAY_GROUPS.items():
        start = len(ordered_markers)
        ordered_markers.extend(members)
        group_boundaries.append((group, start, len(ordered_markers)))

    records = []
    for dimension in HEATMAP_DIMENSION_ORDER:
        summary = summaries[dimension].set_index("marker_name")
        for marker_name in ordered_markers:
            row = summary.loc[marker_name]
            records.append(
                {
                    "dimension": dimension,
                    "marker_name": marker_name,
                    "p_raw": float(row["p_raw"]),
                    "p_bh": float(row["p_bh"]),
                    "sig_bh": bool(row["sig_bh"]),
                    "sig_by": bool(row["sig_by"]),
                    "cluster_stat": float(row["cluster_stat"])
                    if not np.isnan(row["cluster_stat"])
                    else np.nan,
                }
            )
    tidy = pd.DataFrame(records)

    ax = fig.add_subplot(111)

    for col_idx, dimension in enumerate(HEATMAP_DIMENSION_ORDER):
        rgb = matplotlib.colors.to_rgb(resolve_dimension_color(palette, dimension))
        for row_idx, marker_name in enumerate(ordered_markers):
            record = tidy[
                (tidy["dimension"] == dimension) & (tidy["marker_name"] == marker_name)
            ].iloc[0]

            # Opacity ramps with evidence, capped at the permutation floor.
            evidence = -np.log10(max(record["p_raw"], 1e-4))
            opacity = float(np.clip(evidence / 3.7, 0.0, 1.0))

            ax.add_patch(
                plt.Rectangle(
                    (col_idx, row_idx),
                    1,
                    1,
                    facecolor=rgb + (opacity,),
                    edgecolor="white",
                    linewidth=1.4,
                )
            )
            if record["sig_bh"]:
                # Asterisk count encodes how significant the surviving
                # cluster is (standard p-value tiers on the BH-corrected
                # p-value), not a second, separate correction method — BY was
                # dropped from this figure per user direction: it read as an
                # unexplained second hurdle rather than "how strong is this
                # result", which is what a reader actually wants from the
                # glyph count once a marker has already cleared BH.
                if record["p_bh"] < 0.001:
                    glyph = "***"
                elif record["p_bh"] < 0.01:
                    glyph = "**"
                else:
                    glyph = "*"
                glyph_fontsize = HEATMAP_GLYPH_FONTSIZE
            elif record["p_raw"] < alpha:
                # Nominally significant (p < alpha) but does not survive the
                # within-family BH correction on its own — the "distributed
                # but real" signal the family-level omnibus test is built to
                # catch (see CLAUDE.md). A single dot, not an asterisk: this
                # is evidence at a different, weaker standard, not a smaller
                # version of the same claim.
                glyph, glyph_fontsize = "•", HEATMAP_DOT_GLYPH_FONTSIZE
            else:
                glyph, glyph_fontsize = "", 0
            if glyph:
                ax.text(
                    col_idx + 0.5,
                    row_idx + 0.52,
                    glyph,
                    ha="center",
                    va="center",
                    fontsize=glyph_fontsize,
                    fontweight="bold",
                    color="black",
                )

    ax.set_xlim(0, len(HEATMAP_DIMENSION_ORDER))
    ax.set_ylim(0, len(ordered_markers))
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(HEATMAP_DIMENSION_ORDER)) + 0.5)
    ax.set_xticklabels(
        [HEATMAP_COLUMN_LABELS[d] for d in HEATMAP_DIMENSION_ORDER],
        fontsize=tick_fontsize,
        linespacing=1.25,
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(ordered_markers)) + 0.5)
    # Marker-name row labels are short single words/short phrases (unlike the
    # column headers, which wrap to 2 lines and are what left_margin is
    # actually sized against) — they have slack to run bigger than
    # tick_fontsize without threatening that width budget.
    ax.set_yticklabels(
        [MARKER_DISPLAY_NAMES[m] for m in ordered_markers],
        fontsize=tick_fontsize * HEATMAP_YLABEL_FONTSIZE_RATIO,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for group, start, end in group_boundaries:
        ax.plot(
            [-0.34, -0.34],
            [start + 0.08, end - 0.08],
            transform=ax.get_yaxis_transform(),
            clip_on=False,
            color=NEUTRAL_GRAY,
            linewidth=2.2,
        )
        ax.text(
            -0.38,
            (start + end) / 2,
            HEATMAP_GROUP_LABELS[group],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=family_fontsize,
            fontweight="bold",
            color="black",
            linespacing=1.25,
        )

    top_budget_in = HEATMAP_TOP_BUDGET_LETTER_IN if panel_letter is not None else HEATMAP_TOP_BUDGET_IN
    top, bottom = fractions_from_budget(panel_height_in, top_budget_in, HEATMAP_BOTTOM_BUDGET_IN)
    gap_fraction = GAP_BUDGET_IN / panel_height_in

    if panel_letter is not None:
        # Anchored at the literal figure top (y=1.0, va="top"), exactly like
        # draw_onoff_panel/draw_other_dimensions_panel's panel_letter branch
        # (add_significance_legend) — this subfigure and the onoff subfigure
        # both span the full canvas height, so y=1.0 in either one IS the
        # same physical point, which is what makes "A" and "B" land aligned.
        # The previous version anchored "A" to the row va="bottom" instead
        # (a hair below 1.0), which is what the misalignment traced back to.
        fig.text(
            0.015, 1.0, panel_letter, ha="left", va="top",
            fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold",
        )
        fig.text(
            0.97, 1.0,
            "•  p < .05, uncorrected     *  p_FDR < .05     **  p_FDR < .01     ***  p_FDR < .001",
            ha="right", va="top", fontsize=LEGEND_FONTSIZE, fontweight="bold",
            color="black",
        )
    else:
        # Stacked bottom-to-top from the axis top edge: xtick labels (already
        # rendered by MNE-style set_xticklabels above the axis), then the
        # glyph legend, then the title — each with its own gap, all computed
        # from the same panel_height_in so nothing collides regardless of
        # standalone vs. embedded size (see draw_onoff_panel for why
        # fractions are derived this way instead of hardcoded).
        legend_y = top + HEATMAP_XTICK_BUDGET_IN / panel_height_in + gap_fraction
        title_y = legend_y + HEATMAP_LEGEND_BUDGET_IN / panel_height_in + gap_fraction
        if title is not None:
            fig.text(
                0.5, title_y, title, ha="center", va="bottom",
                fontsize=HEATMAP_TITLE_FONTSIZE, fontweight="bold",
            )
        fig.text(
            0.97,
            legend_y,
            "•  p < .05, uncorrected     *  p_FDR < .05     **  p_FDR < .01     ***  p_FDR < .001",
            ha="right",
            va="bottom",
            fontsize=LEGEND_FONTSIZE,
            fontweight="bold",
            color="black",
        )
    fig.subplots_adjust(left=left_margin, right=0.97, top=top, bottom=bottom)
    return tidy


def figure_heatmap(
    summaries: Dict[str, pd.DataFrame],
    palette: dict,
    alpha: float,
    output_path: Path,
) -> pd.DataFrame:
    """
    Standalone export of the evidence heatmap. See ``draw_heatmap``.

    Parameters
    ----------
    summaries : Dict[str, pd.DataFrame]
        Per-target outputs of ``collect_target_results``.
    palette : dict
        Parsed ``color_palette.yaml``.
    alpha : float
        Significance threshold.
    output_path : Path
        Base path (extension replaced per format).

    Returns
    -------
    pd.DataFrame
        Tidy table backing the heatmap.
    """
    panel_height_in = 14.8
    # Width and left_margin solved from measured text extents at
    # HEATMAP_TICK_FONTSIZE (this export uses the full, un-narrowed default,
    # unlike the combined figure's embedded copy) so the column headers
    # ("Valence" / "Neutral/Emotional") and the row/group labels each get
    # exactly the room they need — the previous fixed 20.0in/0.28 pairing was
    # sized for a smaller font and started colliding once HEATMAP_TICK_FONTSIZE
    # and HEATMAP_FAMILY_LABEL_FONTSIZE grew in the 2026-08-13 pass (see the
    # near-identical solve in figure_combined for panel A).
    heatmap_reserved_label_width_in = 10.0
    heatmap_col_width_in = 3.3
    heatmap_width_in = (
        heatmap_reserved_label_width_in + heatmap_col_width_in * len(HEATMAP_DIMENSION_ORDER)
    )
    heatmap_left_margin = heatmap_reserved_label_width_in / heatmap_width_in
    fig = plt.figure(figsize=(heatmap_width_in, panel_height_in))
    tidy = draw_heatmap(fig, panel_height_in, summaries, palette, alpha, left_margin=heatmap_left_margin)
    save_figure_multiformat(fig, output_path)
    plt.close(fig)
    return tidy


# =============================================================================
# COMBINED FIGURE
# =============================================================================


def figure_combined(
    output_root: Path,
    summaries: Dict[str, pd.DataFrame],
    palette: dict,
    alpha: float,
    omnibus: Optional[pd.DataFrame],
    output_path: Path,
    formula: str,
    n_per_family: int = N_TOP_MARKERS_PER_FAMILY,
    n_per_dimension: int = N_TOP_MARKERS_PER_DIMENSION,
) -> None:
    """
    Single paper-ready figure combining all three panels.

    Layout: the evidence heatmap (Panel A) runs the full height of the figure
    on the left, tall and vertical by construction (23 marker rows). To its
    right, stacked top to bottom: the onoff topographies (Panel B) and the
    other-dimensions topographies (Panel C) — together spanning the same
    height as Panel A, so the composite reads as one figure rather than three
    independently-sized ones pasted together.

    Parameters
    ----------
    output_root : Path
        Root of the cluster results tree.
    summaries : Dict[str, pd.DataFrame]
        Per-target outputs of ``collect_target_results``.
    palette : dict
        Parsed ``color_palette.yaml``.
    alpha : float
        Significance threshold.
    omnibus : pd.DataFrame or None
        Family-level omnibus results, when available.
    output_path : Path
        Base path (extension replaced per format).
    n_per_family : int
        Number of onoff topographies per marker family.
    n_per_dimension : int
        Number of topographies per non-onoff dimension.
    """
    # Both panel widths are fixed on purpose and must NOT grow, per explicit
    # user direction (2026-08-13): the combined figure's overall footprint —
    # panel A's width included — stays at its original size; only the
    # figure's HEIGHT is left free to grow (to let panels B/C's topomaps fill
    # their columns, see below). An earlier pass widened panel A instead, to
    # give its 7 column headers more room at a bigger font; the user rejected
    # that trade — text size in panel A now has to fit the original width
    # rather than the other way around, so heatmap_left_margin/
    # heatmap_tick_fontsize/heatmap_family_fontsize below are tuned down from
    # the standalone figure_heatmap's own (unconstrained) sizes to fit here.
    heatmap_width_in = 22.0
    bc_width_in = TOPOMAP_PANEL_WIDTH_IN
    fig_width_in = heatmap_width_in + bc_width_in
    width_ratios = [heatmap_width_in / fig_width_in, bc_width_in / fig_width_in]
    heatmap_left_margin = 0.38
    heatmap_tick_fontsize = 28.0
    heatmap_family_fontsize = 31.0

    # fig_height_in is DERIVED, not chosen: solve_square_topomap_row_height_in
    # asks "how tall must one topomap row be to equal the column width
    # GridSpec will actually give it" for panels B and C separately (they
    # have different column counts), then solve_topomap_panel_height_in
    # turns that row height plus each panel's chrome budget into a required
    # subfigure height. Their sum becomes the figure height. This is what
    # makes every topomap fill its column with no blank margin, without
    # ever touching bc_width_in above — see the module-level docstrings on
    # both solver functions for the full reasoning.
    onoff_row_height_in = solve_square_topomap_row_height_in(
        bc_width_in, len(DISPLAY_GROUPS), ONOFF_GRID_WSPACE
    )
    onoff_height_in = solve_topomap_panel_height_in(
        TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN, n_per_family, onoff_row_height_in, COLORBAR_ROW_BUDGET_IN
    )
    other_row_height_in = solve_square_topomap_row_height_in(
        bc_width_in, len(SECONDARY_DIMENSIONS), OTHER_DIMS_GRID_WSPACE
    )
    other_height_in = solve_topomap_panel_height_in(
        TOPOMAP_PANEL_LETTER_TOP_BUDGET_IN, n_per_dimension, other_row_height_in, COLORBAR_ROW_BUDGET_IN
    )
    height_ratios = [onoff_height_in, other_height_in]
    # subfigures()'s own hspace (below) eats a sliver of vertical space that
    # is not part of either target height — pad the total so both
    # subfigures still receive at least their solved height after that gap
    # is taken out, rather than coming up a hair short and reopening a thin
    # blank strip beside the topomaps this whole derivation exists to close.
    subfigure_vspace_padding = 1.02
    fig_height_in = (onoff_height_in + other_height_in) * subfigure_vspace_padding

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))
    heatmap_fig, right_fig = fig.subfigures(1, 2, width_ratios=width_ratios, wspace=0.003)

    heatmap_height_in = fig_height_in
    draw_heatmap(
        heatmap_fig,
        heatmap_height_in,
        summaries,
        palette,
        alpha,
        # Bare letter (not a descriptive sentence) per the multi-panel-figure
        # convention: the column headers and heatmap axis already say what
        # each part shows, so "Every marker × every dimension" repeated that.
        panel_letter="A",
        # heatmap_tick_fontsize/heatmap_family_fontsize/heatmap_left_margin
        # are all set above, below the standalone figure_heatmap's own
        # (unconstrained) sizes — this copy has to fit heatmap_width_in=22in
        # exactly, not the other way around (see the comment above this
        # function's width block).
        left_margin=heatmap_left_margin,
        tick_fontsize=heatmap_tick_fontsize,
        family_fontsize=heatmap_family_fontsize,
    )

    # onoff_height_in/other_height_in were already solved above (in inches);
    # height_ratios carries those same absolute values through as relative
    # GridSpec weights (matplotlib normalises them, they need not sum to 1).
    onoff_fig, other_fig = right_fig.subfigures(2, 1, height_ratios=height_ratios, hspace=0.012)

    draw_onoff_panel(
        onoff_fig,
        onoff_height_in,
        find_model_dir(output_root, "onoff", formula),
        summaries["onoff"],
        alpha,
        n_per_family,
        panel_letter="B",
    )
    draw_other_dimensions_panel(
        other_fig,
        other_height_in,
        output_root,
        summaries,
        alpha,
        omnibus,
        formula,
        n_per_dimension,
        panel_letter="C",
    )

    save_figure_multiformat(fig, output_path)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """
    Build every paper figure from the current on-disk cluster results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_andrillon.yaml",
        help="Path to config_andrillon.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write figures. Defaults to <output_path>/paper_figures.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    palette = load_color_palette(REPO_ROOT / "color_palette.yaml")
    alpha = config["multiple_comparisons"]["alpha"]
    output_root = Path(config["project"]["output_path"])
    # Names the model directories, so figures can never silently pick up results
    # from a retired specification left on disk (see find_model_dir).
    formula = config["lmm"]["formula"]
    output_dir = args.output_dir or (output_root / "paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_names = [m for members in DISPLAY_GROUPS.values() for m in members]

    summaries: Dict[str, pd.DataFrame] = {}
    for dimension in HEATMAP_DIMENSION_ORDER:
        model_dir = find_model_dir(output_root, dimension, formula)
        summaries[dimension] = collect_target_results(model_dir, marker_names, alpha)
        print(
            f"  {dimension:12s} BH-significant {int(summaries[dimension]['sig_bh'].sum()):2d}"
            f"/{len(marker_names)}   BY-significant {int(summaries[dimension]['sig_by'].sum()):2d}"
        )

    omnibus_path = output_root / "omnibus_test.csv"
    omnibus = pd.read_csv(omnibus_path) if omnibus_path.exists() else None

    print("\nFigure 1 — onoff topographies")
    selected_onoff = figure_onoff(
        find_model_dir(output_root, "onoff", formula),
        summaries["onoff"],
        alpha,
        output_dir / "figure1_onoff_topographies",
    )
    print(selected_onoff[["marker_name", "n_electrodes", "cluster_stat", "p_bh"]].to_string(index=False))

    print("\nFigure 2 — other dimensions")
    selected_other = figure_other_dimensions(
        output_root, summaries, alpha, omnibus,
        output_dir / "figure2_other_dimensions", formula,
    )
    print(selected_other[["dimension", "marker_name", "p_raw", "p_bh", "sig_bh"]].to_string(index=False))

    print("\nFigure 3 — marker x dimension heatmap")
    tidy = figure_heatmap(summaries, palette, alpha, output_dir / "figure3_evidence_heatmap")

    print("\nCombined figure — A (heatmap) + B (onoff) + C (other dimensions)")
    figure_combined(
        output_root,
        summaries,
        palette,
        alpha,
        omnibus,
        output_dir / "figure_combined",
        formula,
    )

    # Provenance: the figures are only as current as the fits they were built
    # from, and those fits are not all in sync (see module docstring).
    tidy.to_csv(output_dir / "figure_source_data.csv", index=False)
    manifest = {
        "alpha": alpha,
        "correction_families": {"evoked": 4, "sleep": 19},
        "corrections_recomputed_from": "per-marker results.pkl (not multiple_comparisons_summary.csv)",
        "figure1_markers": selected_onoff.groupby("display_group")["marker_name"]
        .apply(list)
        .to_dict(),
        "figure2_markers": selected_other.groupby("dimension")["marker_name"]
        .apply(list)
        .to_dict(),
        "marker_fit_timestamps": {
            dimension: {
                marker: load_marker_result(find_model_dir(output_root, dimension, formula), marker)[
                    "analysis_timestamp"
                ]
                for marker in marker_names
            }
            for dimension in HEATMAP_DIMENSION_ORDER
        },
    }
    with open(output_dir / "figure_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nFigures and provenance written to {output_dir}")


if __name__ == "__main__":
    main()
