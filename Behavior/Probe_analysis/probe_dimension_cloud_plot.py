#!/usr/bin/env python
"""
Probe-dimension distributions: group density + per-subject heterogeneity
+ inter-dimension correlation.

Panels 1-5, one per MDES probe dimension (onoff, valence, selfother, time,
confidence), each showing two densities on the same 0-100 probe scale:

- a small filled half-violin (KDE pooled over all trials/subjects), colored
  by the project's per-dimension palette, above a zero baseline;
- one thin gray KDE line per subject, computed from that subject's own trials
  and mirrored below the baseline.

The pooled density is the group-level shape; the per-subject lines are not a
group summary -- they exist to show that individual subjects do not sample
the 0-100 scale the same way (different peaks, different spread, some
bimodal), which the pooled density alone hides.

Panel 6 is a Spearman correlation matrix across the five raw dimensions. It
used to also carry the two curvature terms `valence_sq`/`time_sq`, on the
grounds that they were used in the paper's LMMs; those terms were removed from
every analysis on 2026-08-13 (see Stats_andrillon/config_andrillon.yaml), so
they no longer have a referent here and were dropped from the matrix.
Correlations pool all trials across subjects (matches the probe-probe method
in Behavior/Probe_analysis/Correlations/final_correlation_analysis.py); BH-FDR
is applied once across all 10 unique pairs. Non-significant pairs are left
uncolored (white) -- color/fill still means significance everywhere else in
the project's figures, so it means the same thing here.

Data: results/Behavior/probe_data/probe_level_aggregated_data.csv, one row
per probe with the five raw 0-100 dimension scores and a subject column.

USAGE
-----
    conda activate plots
    python Behavior/Probe_analysis/probe_dimension_cloud_plot.py
"""

# =============================================================================
# Imports
# =============================================================================
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, spearmanr
from statsmodels.stats.multitest import multipletests

_SKILL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "scientific_plots"
sys.path.insert(0, str(_SKILL_DIR))
import sciplot as sp  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_DATA_PATH = PROJECT_ROOT / "results" / "Behavior" / "probe_data" / "probe_level_aggregated_data.csv"
PALETTE_PATH = PROJECT_ROOT / "color_palette.yaml"

OUT_DIR = PROJECT_ROOT / "results" / "Behavior" / "probe_data" / "probe_dimension_distributions"
FIG_NAME = "probe_dimension_cloud"

WIDTH_MM = 220.0
HEIGHT_MM = 58.0

# Dimensions in display order, with their 0/100 poles (scale convention:
# 0 = off-task/negative/past/self-focused/low confidence; see CLAUDE.md).
DIMENSIONS: list[dict] = [
    {"key": "onoff", "label": "On/Off-Task", "pole_low": "off-task", "pole_high": "on-task"},
    {"key": "valence", "label": "Valence", "pole_low": "negative", "pole_high": "positive"},
    {"key": "selfother", "label": "Self/Other", "pole_low": "self-focused", "pole_high": "other-focused"},
    {"key": "time", "label": "Time", "pole_low": "past", "pole_high": "future"},
    {"key": "confidence", "label": "Confidence", "pole_low": "low", "pole_high": "high"},
]

SCALE_MIN, SCALE_MAX = 0.0, 100.0
GRID_N = 256  # KDE evaluation grid resolution

# Panel 6: correlation matrix across the five raw probe dimensions.
CORR_VARS: list[dict] = [
    {"key": "onoff", "label": "On/Off-Task"},
    {"key": "valence", "label": "Valence"},
    {"key": "selfother", "label": "Self/Other"},
    {"key": "time", "label": "Time"},
    {"key": "confidence", "label": "Confidence"},
]
CORR_ALPHA = 0.05
CORR_COLUMN_WEIGHT = 1.0  # same width as each density panel
CORR_CELL_FONT_PT = 5.5  # smaller than the template's 7pt tick size; 5x5 grid
                         # at 1-panel width has no room for a bigger label
# A blank spacer column between the last density panel and the matrix: the
# matrix's row labels sit outside its own plot area (standard for a heatmap),
# and with the panels directly adjacent that text renders on top of the
# "Confidence" panel next door. A real gap column reserves room for it instead
# of relying on axis automargin, which only pads the outer figure edge, not
# the seam between two inner subplots.
SPACER_COLUMN_WEIGHT = 0.5

# Relative height of the pooled (group) half-violin vs. the per-subject
# lines: the group density is deliberately drawn smaller so the panel reads
# as "many individual shapes below a small group summary", not the reverse.
GROUP_HEIGHT = 0.5
SUBJECT_HEIGHT = 0.65

# Per-subject curves are scaled by one shared reference peak so their relative
# heights stay informative (sharper subject = taller line). Using the 85th
# percentile of subject peaks rather than the max keeps the whole band compact
# -- the one or two most concentrated subjects run past the axis edge and get
# cropped, instead of every other subject being squashed flat near zero.
SUBJECT_SCALE_PERCENTILE = 85.0
SUBJECT_LINE_WIDTH = 0.7
SUBJECT_OPACITY = 0.45
GROUP_FILL_OPACITY = 0.6

RANDOM_SEED = 0  # unused (no resampling), kept for consistency with other figures


# =============================================================================
# Palette (adapter over color_palette.yaml; see make_fig_section3_decoding.py)
# =============================================================================
class RepoPalette:
    """Adapter exposing color_palette.yaml through the interface sciplot expects."""

    def __init__(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text())
        self.order: list[str] = data["order"]
        self.neutral: dict[str, str] = data.get("neutral", {})
        self.continuous: dict[str, str] = data.get("continuous", {})
        self._sources = (
            data.get("dimensions", {}),
            data.get("quadratic", {}),
            data.get("palette", {}),
            self.neutral,
        )

    def get(self, key: str) -> str:
        for source in self._sources:
            if key in source:
                return source[key]
        raise KeyError(f"'{key}' not found in {PALETTE_PATH}")


def init_palette() -> RepoPalette:
    pal = RepoPalette(PALETTE_PATH)
    pio.templates["sci"] = sp.make_template(pal)
    pio.templates.default = "sci"
    return pal


# =============================================================================
# Data
# =============================================================================
def load_probe_data() -> pd.DataFrame:
    """Load one row per probe with subject id and the five dimension scores."""
    cols = ["subject"] + [d["key"] for d in DIMENSIONS]
    df = pd.read_csv(PROBE_DATA_PATH, usecols=cols)
    df = df.dropna(subset=cols)
    return df


def compute_correlation_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Spearman correlations across ``CORR_VARS``, BH-FDR corrected.

    Returns
    -------
    corr : (n, n) array
        Spearman r for every pair (diagonal = 1).
    significant : (n, n) bool array
        True where the BH-FDR-corrected p-value (single correction across all
        10 unique off-diagonal pairs) is below ``CORR_ALPHA``. Diagonal is
        always True.
    """
    keys = [v["key"] for v in CORR_VARS]
    n = len(keys)
    corr = np.eye(n)
    significant = np.eye(n, dtype=bool)

    pair_idx: list[tuple[int, int]] = []
    pair_p: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            r, p = spearmanr(df[keys[i]], df[keys[j]])
            corr[i, j] = corr[j, i] = r
            pair_idx.append((i, j))
            pair_p.append(p)

    _, p_fdr, _, _ = multipletests(pair_p, method="fdr_bh")
    for (i, j), p_adj in zip(pair_idx, p_fdr):
        significant[i, j] = significant[j, i] = p_adj < CORR_ALPHA

    return corr, significant


def kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Raw Gaussian KDE of ``values`` evaluated on ``grid`` (unnormalized)."""
    return gaussian_kde(values)(grid)


TOP_STRIP_FRACTION = 0.86  # y-domains compressed into the bottom 86%, freeing a
                           # fixed top strip for the panel-group letters (A/B)
                           # above the per-panel dimension titles.


def reserve_top_strip(fig: go.Figure, top_fraction: float) -> None:
    """Compress every subplot's y-domain into ``[0, top_fraction]``.

    Needed because two header lines (group letter, then dimension title) sit
    above the plot area: growing ``margin.t`` after the fact does not create
    space for them since domains are fractions of the plot area itself, not
    of the full canvas -- see the identical fix in make_fig_ws_loso_sign_forest.py.
    """
    for key in fig.layout:
        if re.fullmatch(r"yaxis\d*", key):
            d0, d1 = fig.layout[key].domain
            fig.layout[key].domain = (d0 * top_fraction, d1 * top_fraction)


def centered_panel_titles(fig: go.Figure, labels: Sequence[str]) -> go.Figure:
    """Bold dimension name centered above each subplot (not left-anchored).

    ``sciplot.panel_labels`` anchors text at the domain's left edge, which
    reads as centered only for short single-letter labels; a longer word
    (e.g. "confidence") then grows past the panel and looks off-center. This
    centers the annotation on the subplot's own x-domain instead.
    """
    axes = sorted(
        (k for k in fig.layout if re.fullmatch(r"xaxis\d*", k)),
        key=lambda k: int(k[5:] or 1),
    )
    for name, label in zip(axes, labels):
        if not label:
            continue
        xa = fig.layout[name]
        ya = fig.layout["yaxis" + name[5:]]
        fig.add_annotation(
            x=(xa.domain[0] + xa.domain[1]) / 2, y=ya.domain[1],
            xref="paper", yref="paper",
            text=f"<b>{label}</b>", font=dict(size=sp.pt2px(sp.TYPE_PT["panel_label"])),
            xanchor="center", yanchor="bottom", showarrow=False,
        )
    return fig


GROUP_LETTER_SIZE_PT = 13.0    # a clear step above the 9pt dimension-name titles
GROUP_LETTER_GAP_MM = 7.0      # physical gap between the letter and the panel's
                                # own content, so it reads as a separate mark
                                # rather than a prefix stuck onto the title


def add_group_letters(fig: go.Figure, groups: Sequence[tuple[str, int]]) -> None:
    """Bold "A"/"B" letters at the top-left corner of a group's first panel.

    Per house rule (scientific-plots SKILL.md): a panel letter always sits at
    the top-left corner of a panel -- even when, as here, one letter stands
    for a whole row of panels -- never centered above the group. Corner
    anchoring is what ``sp.panel_labels`` does for a plain one-letter-per-panel
    grid; this is the same rule applied to a letter that spans several panels.

    For panel 1 (whose domain starts at x=0, the figure's own left edge) the
    letter has nowhere to sit "to the left of the panel" without living in the
    left margin -- so the margin is sized for it explicitly (``GROUP_LETTER_GAP_MM``)
    rather than clamping the letter back onto the panel and losing the
    separation from the panel's own title.

    Parameters
    ----------
    groups : sequence of (letter, first_col)
        E.g. ``("A", 1)`` anchors at panel 1's corner (labelling it and the
        four panels after it); ``("B", 7)`` anchors at panel 7's own corner.
    """
    ink = "#2B2B2B"
    dx = -GROUP_LETTER_GAP_MM / WIDTH_MM
    for letter, first_col in groups:
        xa = fig.layout[f"xaxis{'' if first_col == 1 else first_col}"]
        ya = fig.layout[f"yaxis{'' if first_col == 1 else first_col}"]
        fig.add_annotation(
            x=xa.domain[0] + dx, y=ya.domain[1],
            xref="paper", yref="paper",
            text=f"<b>{letter}</b>", font=dict(size=sp.pt2px(GROUP_LETTER_SIZE_PT), color=ink),
            xanchor="left", yanchor="bottom", showarrow=False,
        )


def format_r(value: float) -> str:
    """Compact r label: no leading zero, diagonal shown as a bare "1"."""
    if abs(value - 1.0) < 1e-9:
        return "1"
    text = f"{value:.2f}"
    return "-" + text[2:] if text.startswith("-0") else text[1:]


def add_correlation_panel(fig: go.Figure, df: pd.DataFrame, pal: RepoPalette, col: int) -> None:
    """Draw the lower-triangular Spearman correlation matrix in subplot ``col``.

    Significance is carried by fill (project convention): flat neutral gray for
    a tested-but-non-significant pair, the diverging r-color for a significant
    one, blank for the (redundant) upper triangle. r-values are printed inside
    every lower-triangle cell in a compact form (``format_r``, e.g. ".42",
    "-.03", "1") since a full "0.42" does not fit at this panel's width; the
    2-decimal values are also written in full to ``<name>_correlations.csv``.
    """
    corr, significant = compute_correlation_matrix(df)
    n = len(CORR_VARS)
    labels = [v["label"] for v in CORR_VARS]
    grey = pal.get("covariate")
    ink = pal.neutral.get("ink", "#2B2B2B")

    lower = np.tril(np.ones((n, n), dtype=bool))  # includes diagonal
    z_r = np.where(lower & significant, corr, np.nan)
    z_ns = np.where(lower & ~significant, 1.0, np.nan)

    # Flat gray base layer for tested/non-significant cells; the diverging,
    # significance-gated layer is drawn on top so its NaNs let this show through.
    fig.add_trace(
        go.Heatmap(
            z=z_ns, x=labels, y=labels, zmin=0, zmax=1,
            colorscale=[[0, grey], [1, grey]], showscale=False,
            xgap=1, ygap=1, hoverinfo="skip",
        ),
        row=1, col=col,
    )
    fig.add_trace(
        go.Heatmap(
            z=z_r, x=labels, y=labels,
            zmid=0, zmin=-1, zmax=1,
            colorscale=pal.continuous.get("diverging", "RdBu"),
            xgap=1, ygap=1,
            colorbar=dict(title=dict(text="r"), thickness=6, len=0.9, outlinewidth=0, x=1.0),
            hovertemplate="%{y} vs %{x}: r=%{z:.2f}<extra></extra>",
        ),
        row=1, col=col,
    )
    # A saturated fill (|r| large) needs white text to stay legible against it.
    for i in range(n):
        for j in range(i + 1):
            is_dark = significant[i, j] and abs(corr[i, j]) >= 0.6
            fig.add_annotation(
                x=labels[j], y=labels[i], text=format_r(corr[i, j]),
                font=dict(size=sp.pt2px(CORR_CELL_FONT_PT), color="white" if is_dark else ink),
                showarrow=False, row=1, col=col,
            )
    fig.update_xaxes(tickangle=-40, showgrid=False, row=1, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=col)
    add_correlation_legend(fig, pal, col=col, n=n)


def add_correlation_legend(fig: go.Figure, pal: RepoPalette, col: int, n: int) -> None:
    """Gray/blue swatch legend in the matrix's own blank upper-right triangle.

    For a lower-triangular n x n matrix, the top 2 rows are blank from their
    3rd column onward regardless of n (row 0 only has col 0 filled, row 1 only
    cols 0-1) -- so that block is always free real estate; no extra figure
    space is spent on the legend.
    """
    grey = pal.get("covariate")
    ink = pal.neutral.get("ink", "#2B2B2B")
    sig_color = pc.sample_colorscale(pal.continuous.get("diverging", "RdBu"), [0.8])[0]

    xa = fig.layout[f"xaxis{col}"]
    ya = fig.layout[f"yaxis{col}"]
    x_span = xa.domain[1] - xa.domain[0]
    y_span = ya.domain[1] - ya.domain[0]
    zone_x0 = xa.domain[0] + (2 / n) * x_span
    zone_y1 = ya.domain[1]
    zone_y0 = zone_y1 - (2 / n) * y_span

    # A swatch square defined as one fraction of x_span would render as a thin
    # bar: paper x/y fractions map to very different absolute sizes once
    # WIDTH_MM >> HEIGHT_MM. Sizing it in mm on each axis independently keeps
    # it visually square regardless of the panel's aspect ratio.
    swatch_mm = 2.0
    sw_x = swatch_mm / WIDTH_MM
    sw_y = swatch_mm / HEIGHT_MM
    x0 = zone_x0 + 0.08 * x_span
    for row, (color, label) in enumerate([(sig_color, "significant"), (grey, "not significant")]):
        y_center = zone_y1 - (row + 0.5) * (zone_y1 - zone_y0) / 2
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=x0, x1=x0 + sw_x, y0=y_center - sw_y / 2, y1=y_center + sw_y / 2,
            fillcolor=color, line_width=0,
        )
        fig.add_annotation(
            x=x0 + sw_x + 0.03 * x_span, y=y_center, xref="paper", yref="paper",
            text=label, font=dict(size=sp.pt2px(CORR_CELL_FONT_PT), color=ink),
            xanchor="left", yanchor="middle", showarrow=False,
        )


# =============================================================================
# Figure
# =============================================================================
def build_figure(df: pd.DataFrame, pal: RepoPalette) -> go.Figure:
    """Density panels (one per dimension) + a correlation-matrix panel."""
    grid = np.linspace(SCALE_MIN, SCALE_MAX, GRID_N)
    grey = pal.get("permutation")

    n_density = len(DIMENSIONS)
    spacer_col = n_density + 1
    corr_col = n_density + 2
    column_widths = [1.0] * n_density + [SPACER_COLUMN_WEIGHT, CORR_COLUMN_WEIGHT]
    fig = make_subplots(rows=1, cols=corr_col, column_widths=column_widths, horizontal_spacing=0.025)
    reserve_top_strip(fig, TOP_STRIP_FRACTION)
    fig.update_xaxes(visible=False, row=1, col=spacer_col)
    fig.update_yaxes(visible=False, row=1, col=spacer_col)

    for col, dim in enumerate(DIMENSIONS, start=1):
        key = dim["key"]
        color = pal.get(key)

        # Raw (unnormalized) per-subject densities: subjects with a sharper,
        # more concentrated distribution get a taller line, subjects with a
        # flatter one a shorter line -- that contrast is itself the signal,
        # so curves are scaled by one shared factor rather than each stretched
        # to the same peak.
        subject_values = [g[key].to_numpy() for _, g in df.groupby("subject")]
        subject_densities = [kde_density(values, grid) for values in subject_values]
        scale_ref = np.percentile([d.max() for d in subject_densities], SUBJECT_SCALE_PERCENTILE)
        for density in subject_densities:
            line = -SUBJECT_HEIGHT * density / scale_ref
            fig.add_trace(
                go.Scatter(
                    x=grid, y=line, mode="lines",
                    line=dict(color=grey, width=SUBJECT_LINE_WIDTH),
                    opacity=SUBJECT_OPACITY, hoverinfo="skip", showlegend=False,
                ),
                row=1, col=col,
            )

        pooled_density = kde_density(df[key].to_numpy(), grid)
        pooled = GROUP_HEIGHT * pooled_density / pooled_density.max()
        fig.add_trace(
            go.Scatter(
                x=grid, y=pooled, mode="lines", fill="tozeroy",
                line=dict(color=color, width=1.2),
                fillcolor=color, opacity=GROUP_FILL_OPACITY,
                hovertemplate=f"{dim['label']}: %{{x:.0f}}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col,
        )

        fig.add_hline(y=0, line=dict(color=pal.neutral.get("ink", "#2B2B2B"), width=0.6),
                       row=1, col=col, layer="below")

        # The dimension name is already the bold panel title above; the axis
        # title only needs to carry the pole direction, not repeat the name.
        fig.update_xaxes(
            title_text=f"{dim['pole_low']} → {dim['pole_high']}",
            range=[SCALE_MIN, SCALE_MAX], tickvals=[0, 50, 100],
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[-SUBJECT_HEIGHT * 1.08, GROUP_HEIGHT * 1.15],
            showticklabels=False, ticks="", showline=True,
            row=1, col=col,
        )

    sp.grid(fig, "x")
    add_correlation_panel(fig, df, pal, col=corr_col)
    centered_panel_titles(fig, [d["label"] for d in DIMENSIONS] + ["", "Correlation"])
    add_group_letters(fig, [("A", 1), ("B", corr_col)])
    # Left margin sized for the "A" letter sitting GROUP_LETTER_GAP_MM to the
    # left of panel 1's own edge (that panel starts at x=0, so the letter has
    # nowhere else to go); right margin for the correlation colorbar.
    fig.update_layout(margin=dict(t=8, l=sp.mm2px(GROUP_LETTER_GAP_MM + 2), r=32))
    return fig


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    pal = init_palette()
    df = load_probe_data()

    n_subjects = df["subject"].nunique()
    print(f"Probe data: {len(df)} trials from {n_subjects} subjects")
    for dim in DIMENSIONS:
        print(f"  {dim['key']:<12} mean={df[dim['key']].mean():6.1f}  sd={df[dim['key']].std():5.1f}")

    corr, significant = compute_correlation_matrix(df)
    corr_labels = [v["label"] for v in CORR_VARS]
    n_pairs = len(CORR_VARS) * (len(CORR_VARS) - 1) // 2
    print(f"\nSpearman correlations (BH-FDR across {n_pairs} pairs, alpha="
          f"{CORR_ALPHA}); * = significant:")
    for i, row_label in enumerate(corr_labels):
        cells = "  ".join(
            f"{corr[i, j]:+.2f}{'*' if significant[i, j] else ' '}" for j in range(len(corr_labels))
        )
        print(f"  {row_label:<12} {cells}")

    fig = build_figure(df, pal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr_df = pd.DataFrame(corr, index=corr_labels, columns=corr_labels)
    corr_df.to_csv(OUT_DIR / f"{FIG_NAME}_correlations.csv")
    sp.save(fig, str(OUT_DIR), FIG_NAME, width_mm=WIDTH_MM, height_mm=HEIGHT_MM, data=corr_df)
    print(f"\nWrote {OUT_DIR / FIG_NAME}.svg / .png")
    print(f"Wrote {OUT_DIR / FIG_NAME}_correlations.csv")


if __name__ == "__main__":
    main()
