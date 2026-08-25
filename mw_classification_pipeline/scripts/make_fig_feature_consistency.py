#!/usr/bin/env python
"""
Figures for the WS <-> LOSO feature-consistency analysis.

Reads the tables written by ``feature_consistency_analysis.py`` and renders:

``fig_feature_consistency``
    A — per-subject Spearman correlation between a subject's own model's
        mean(|SHAP|) profile and that of the group model trained without them,
        with the group-level correlation overlaid.
    B — the same group-level correlation at 175-feature and at 23-marker
        resolution, showing how much of the disagreement is arbitrary choice
        between collinear ROI columns.

``fig_marker_profiles``
    One panel per dimension: each of the 23 CBPT markers' share of total
    |SHAP|, within-subject against LOSO, as a scatter against the diagonal.
    Shows the alignment between the two pipelines directly.

``fig_marker_profiles_dots``
    The same numbers as a connected dot plot with every marker labelled. Better
    when the question is "where does marker X sit?" rather than "do the two
    agree?". Both views are kept; neither replaces the other.

A marker's ROI columns are combined by **mean**, not sum (``marker_aggregation``
in the analysis config). The four evoked markers own one column each and the
nineteen sleep markers own nine, so summing would bury an evoked marker such as
P3b behind an artefact of the feature-space layout.

The figure this replaces annotated its correlation over a top-10-of-WS union
top-10-of-LOSO subsample, which manufactures a steep negative slope out of
noise. Nothing here computes a statistic on a top-N subset.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_feature_consistency.py
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots

_SKILL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "scientific_plots"
sys.path.insert(0, str(_SKILL_DIR))
import sciplot as sp  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parents[1]
RESULTS_DIR = _SCRIPTS_DIR.parent / "results" / "feature_consistency"
OUT_DIR = _SCRIPTS_DIR.parent / "results" / "figures" / "feature_consistency"
PALETTE_PATH = REPO_ROOT / "color_palette.yaml"

MAIN_WIDTH_MM, MAIN_HEIGHT_MM = 180.0, 78.0
# Taller than the panels strictly need: the shared axis range is set by the
# single largest share across all dimensions, so most panels' points sit in the
# lower third and five labels each need vertical room not to collide.
SCATTER_WIDTH_MM, SCATTER_HEIGHT_MM = 180.0, 96.0
DOTPLOT_WIDTH_MM, DOTPLOT_HEIGHT_MM = 180.0, 105.0

# Maps a contrast key onto its name in color_palette.yaml.
PALETTE_KEY = {
    "on_off": "onoff", "valence": "valence", "confidence": "confidence",
    "selfother": "selfother", "time": "time",
}

JITTER_WIDTH = 0.18
ALPHA = 0.05
RANDOM_SEED = 0  # jitter only; no inference depends on it

# Points labelled per scatter panel: the top 5 by whichever pipeline uses them
# most. Labels name the cloud so the reader can see which markers dominate
# without a lookup table; no statistic uses them, and the annotated rho always
# covers all 23 markers.
N_SCATTER_LABELS = 5

# =============================================================================
# Palette
# =============================================================================


class RepoPalette:
    """
    Adapter exposing the project's ``color_palette.yaml`` through the small
    interface ``sciplot.make_template`` expects (``order`` and ``neutral``).

    The project file is the single source of truth named in CLAUDE.md and is
    read by several other scripts, so it is not rewritten into the plotting
    helper's own schema; the helper is adapted to it instead.
    """

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
        """Resolve a dimension, quadratic-term, swatch or neutral name to a hex."""
        for source in self._sources:
            if key in source:
                return source[key]
        raise KeyError(f"'{key}' not found in {PALETTE_PATH}")


def init_palette() -> RepoPalette:
    """Load the project palette and register sciplot's ``sci`` template with it."""
    pal = RepoPalette(PALETTE_PATH)
    pio.templates["sci"] = sp.make_template(pal)
    pio.templates.default = "sci"
    return pal


# =============================================================================
# Data
# =============================================================================


def marker_display_order(per_marker: pd.DataFrame) -> list[str]:
    """
    The one ordering rule every marker figure in this analysis uses.

    Markers are sorted by their mean share of |SHAP| averaged over both
    pipelines and all seven dimensions, ascending (so the most-used marker sits
    at the top of a reversed y axis).

    Returns
    -------
    list[str]
        Marker names, least- to most-used.

    Notes
    -----
    A single global order rather than a per-dimension one, deliberately: the
    dot plot puts seven dimensions on a shared y axis, so a per-dimension order
    is not even expressible there, and having each figure sort by its own
    criterion makes a row mean something different in each — which is exactly
    what made the ordering unreadable before. With one rule, a row is the same
    marker in every figure and the three can be read side by side.

    Shares are per-ROI means (see ``marker_aggregation`` in the analysis
    config), so single-ROI evoked markers compete on equal terms.
    """
    return list(
        per_marker.groupby("marker")[["mean_share_ws", "mean_share_loso"]]
        .mean().mean(axis=1).sort_values().index
    )


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read the three analysis tables, ordered by group-level agreement.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(summary, per_subject, per_marker)``. The contrast order established
        on ``summary`` is applied to the other two so every panel of every
        figure lists the dimensions the same way.
    """
    summary = pd.read_csv(RESULTS_DIR / "group_summary.csv")
    per_subject = pd.read_csv(RESULTS_DIR / "per_subject_consistency.csv")
    per_marker = pd.read_csv(RESULTS_DIR / "marker_level_consistency.csv")

    summary = summary.sort_values("legacy_group_rho", ascending=False).reset_index(drop=True)
    order = {key: i for i, key in enumerate(summary["contrast"])}
    for frame in (per_subject, per_marker):
        frame["_order"] = frame["contrast"].map(order)
        frame.sort_values("_order", inplace=True)
    return summary, per_subject, per_marker


# =============================================================================
# Figure 1 — consistency
# =============================================================================


def build_consistency_figure(
    summary: pd.DataFrame, per_subject: pd.DataFrame, pal: RepoPalette
) -> go.Figure:
    """
    Per-subject versus group-level agreement, and feature versus marker
    resolution.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    grey = pal.get("permutation")
    x_positions = np.arange(len(summary), dtype=float)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.10
    )

    # ---- Panel A ----------------------------------------------------------
    fig.add_hline(y=0.0, line=dict(color=grey, width=0.8, dash="dot"),
                  layer="below", row=1, col=1)

    for x, row in zip(x_positions, summary.itertuples()):
        color = pal.get(PALETTE_KEY[row.contrast])
        subject_rhos = per_subject.loc[
            per_subject["contrast"] == row.contrast, "rho_magnitude"
        ].to_numpy()
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=subject_rhos.size)
        fig.add_trace(go.Scatter(
            x=x + jitter, y=subject_rhos,
            mode="markers",
            marker=dict(color=color, size=3.5, opacity=0.35,
                        line=dict(width=0.3, color="white")),
            hovertemplate=f"{row.label}<br>subject rho: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        # Mean of the per-subject correlations, and the single group-level one.
        # Both are Spearman on the same scale, so the vertical gap between them
        # is directly readable — and it is the whole point of the panel.
        fig.add_trace(go.Scatter(
            x=[x - 0.16, x + 0.16], y=[row.mean_rho_magnitude, row.legacy_group_rho],
            mode="lines", line=dict(color=color, width=1.4),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    for column, symbol, size, offset in (
        ("mean_rho_magnitude", "circle", 10, -0.16),
        ("legacy_group_rho", "diamond", 9, 0.16),
    ):
        significant = (
            summary["n_sig_magnitude"] > 0 if column == "mean_rho_magnitude"
            else summary["legacy_group_p"] < ALPHA
        )
        colors = [pal.get(PALETTE_KEY[k]) for k in summary["contrast"]]
        fig.add_trace(go.Scatter(
            x=x_positions + offset, y=summary[column],
            mode="markers",
            marker=dict(symbol=symbol, size=size,
                        color=[c if s else "white" for c, s in zip(colors, significant)],
                        line=dict(color=colors, width=1.7)),
            hovertemplate="%{y:.3f}<extra></extra>", showlegend=False,
        ), row=1, col=1)

    # ---- Panel B ----------------------------------------------------------
    fig.add_hline(y=0.0, line=dict(color=grey, width=0.8, dash="dot"),
                  layer="below", row=1, col=2)
    for x, row in zip(x_positions, summary.itertuples()):
        color = pal.get(PALETTE_KEY[row.contrast])
        fig.add_trace(go.Scatter(
            x=[x, x], y=[row.legacy_group_rho, row.marker_group_rho],
            mode="lines", line=dict(color=color, width=1.4),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=2)

    for column, symbol, size, p_column in (
        ("legacy_group_rho", "diamond", 9, "legacy_group_p"),
        ("marker_group_rho", "square", 8, "marker_group_p"),
    ):
        colors = [pal.get(PALETTE_KEY[k]) for k in summary["contrast"]]
        significant = summary[p_column] < ALPHA
        fig.add_trace(go.Scatter(
            x=x_positions, y=summary[column],
            mode="markers",
            marker=dict(symbol=symbol, size=size,
                        color=[c if s else "white" for c, s in zip(colors, significant)],
                        line=dict(color=colors, width=1.7)),
            hovertemplate="%{y:.3f}<extra></extra>", showlegend=False,
        ), row=1, col=2)

    # ---- Legend -----------------------------------------------------------
    # Neutral ink: hue encodes the dimension everywhere else, so a coloured key
    # would imply it encodes the series. Solid, not hollow — hollow is already
    # spoken for as "not significant".
    for symbol, size, name in (
        ("circle", 8, "per subject (mean)"),
        ("diamond", 7.5, "group, 175 features"),
        ("square", 7, "group, 23 markers"),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey,
                        line=dict(color=grey, width=1.5)),
            name=name, showlegend=True, hoverinfo="skip",
        ))

    fig.update_xaxes(
        tickmode="array", tickvals=x_positions,
        ticktext=[f"{r.label}<br>n={r.n_subjects}" for r in summary.itertuples()],
        range=[-0.6, len(summary) - 0.4], row=1, col=1,
    )
    # Panel B repeats panel A's dimensions in the same order and the same
    # colours, so re-labelling them costs a band of rotated, colliding text to
    # restate what the reader already has one panel to the left.
    # ticks="" as well as the labels: left on, plotly places them at its own
    # automatic positions, which do not line up with the seven columns and read
    # as though they marked something.
    fig.update_xaxes(showticklabels=False, ticks="",
                     range=[-0.6, len(summary) - 0.4], row=1, col=2)

    all_subject_rhos = per_subject["rho_magnitude"].to_numpy()
    y_lo = min(all_subject_rhos.min(), 0.0)
    y_hi = max(all_subject_rhos.max(), summary["legacy_group_rho"].max(),
               summary["marker_group_rho"].max())
    pad = 0.06 * (y_hi - y_lo)
    fig.update_yaxes(title_text="Spearman ρ (WS vs LOSO)",
                     range=[y_lo - pad, y_hi + pad], row=1, col=1)
    fig.update_yaxes(range=[y_lo - pad, y_hi + pad], row=1, col=2)

    sp.grid(fig, "y")
    sp.legend_above(fig)
    sp.panel_labels(fig)
    return fig


# =============================================================================
# Figure 2 — marker profiles
# =============================================================================


def _pretty_marker(name: str) -> str:
    """
    Render a raw marker name for display without making two markers collide.

    The raw names carry their family as a prefix (``psd_relative_alpha``,
    ``PE_alpha``, ``wsmi_alpha``). Stripping any one prefix — as an earlier
    version did to ``psd_relative_`` — leaves a bare ``alpha`` sitting next to
    ``PE_alpha`` in the same axis, so the reader cannot tell which band belongs
    to which family. Every family therefore keeps a visible tag.
    """
    replacements = {
        "psd_relative_": "PSD ",
        "kolmogorov_complexity": "Kolmogorov",
        "slowwaves_": "SW ",
        "wsmi_": "wSMI ",
        "PE_": "PE ",
    }
    for old, new in replacements.items():
        if name.startswith(old) or name == old.rstrip("_"):
            return name.replace(old, new)
    return name


def build_marker_scatter_figure(
    summary: pd.DataFrame, per_marker: pd.DataFrame, pal: RepoPalette
) -> go.Figure:
    """
    Each marker's share of total |SHAP|, within-subject against LOSO, as a
    scatter — one panel per dimension, 23 points each.

    The correlation annotated on each panel covers all 23 markers. The two
    labelled points per panel are the largest contributors and are named for
    orientation only; no statistic uses them.

    Kept alongside the dot-plot view rather than replaced by it: the scatter
    shows the diagonal alignment between the two pipelines directly, which is
    the claim, while the dot plot is the better reference when the question is
    "where does marker X sit?".
    """
    contrasts = list(summary["contrast"])
    fig = make_subplots(
        rows=1, cols=len(contrasts), horizontal_spacing=0.022,
        subplot_titles=[str(lbl) for lbl in summary["label"]],
    )
    for i, key in enumerate(contrasts):
        fig.layout.annotations[i].font.color = pal.get(PALETTE_KEY[key])
        fig.layout.annotations[i].font.size = 8

    grey = pal.get("permutation")
    shares = per_marker[["mean_share_ws", "mean_share_loso"]].to_numpy()
    # Headroom above the highest point so its label is not clipped.
    hi = float(shares.max()) * 1.28

    for col, key in enumerate(contrasts, start=1):
        color = pal.get(PALETTE_KEY[key])
        block = per_marker[per_marker["contrast"] == key]
        x = block["mean_share_ws"].to_numpy()
        y = block["mean_share_loso"].to_numpy()

        fig.add_shape(type="line", x0=0, y0=0, x1=hi, y1=hi,
                      line=dict(color=grey, width=0.8, dash="dot"),
                      row=1, col=col, layer="below")

        # Ranked by the mean of the two pipelines, not the maximum. Taking the
        # max labels a marker that one pipeline leans on and the other ignores,
        # ahead of one both use steadily — a selection-on-extremes criterion, the
        # exact failure this whole analysis exists to undo. It also contradicted
        # the row ordering, which already averages the two: on/off-task P3b ranks
        # 7th by max and 5th by mean, so it was the one marker the direction
        # figure singled out and this one refused to name.
        rank = (x + y) / 2.0
        labelled = set(np.argsort(rank)[-N_SCATTER_LABELS:])
        names = [_pretty_marker(n) for n in block["marker"]]
        mask = np.array([i in labelled for i in range(len(x))])

        fig.add_trace(go.Scatter(
            x=x[~mask], y=y[~mask], mode="markers",
            marker=dict(color=color, size=4, opacity=0.45,
                        line=dict(color="white", width=0.3)),
            customdata=[n for n, m in zip(names, mask) if not m],
            hovertemplate="%{customdata}<br>WS %{x:.3f} / LOSO %{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col)
        # Label positions alternate above/below as the labelled points ascend.
        # The five most-used markers of a dimension sit within a few thousandths
        # of each other, so a single position stacks all five labels into one
        # illegible block — P3b was in on/off-task's top five and simply could
        # not be seen. Alternating doubles the vertical room without moving any
        # point.
        labelled_y = y[mask]
        vertical_rank = np.argsort(np.argsort(labelled_y))
        positions = np.where(vertical_rank % 2 == 0, "top center", "bottom center")

        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask], mode="markers+text",
            marker=dict(color=color, size=5.5,
                        line=dict(color="white", width=0.4)),
            text=[n for n, m in zip(names, mask) if m],
            textposition=positions, textfont=dict(size=5),
            hovertemplate="%{text}<br>WS %{x:.3f} / LOSO %{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col)

        rho = float(summary.loc[summary["contrast"] == key, "marker_group_rho"].iloc[0])
        fig.add_annotation(
            text=f"ρ = {rho:.2f}", xref="x domain", yref="y domain",
            x=0.04, y=0.98, xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=7, color=color), row=1, col=col,
        )

        fig.update_xaxes(range=[0, hi], dtick=0.05, row=1, col=col)
        fig.update_yaxes(range=[0, hi], dtick=0.05, showticklabels=(col == 1),
                         row=1, col=col)

    fig.update_xaxes(title_text="within-subject share of |SHAP|", row=1, col=4)
    fig.update_yaxes(title_text="LOSO share", row=1, col=1)
    sp.grid(fig, "y")
    # The template's tight top margin clips the subplot titles, which plotly
    # positions just above the axes.
    fig.update_layout(margin=dict(t=18))
    return fig


def build_marker_dotplot_figure(
    summary: pd.DataFrame, per_marker: pd.DataFrame, pal: RepoPalette
) -> go.Figure:
    """
    Each marker's share of total |SHAP|, within-subject against LOSO.

    A connected dot plot, one column per dimension, markers on the shared y
    axis. Agreement reads as a short connector, so the comparison the figure
    exists to make is a length rather than a position in a cloud.

    This replaces a 7-panel scatter of the same numbers. At 180 mm each panel
    was ~24 mm wide, too small to label 23 points, so only two per panel were
    named and the remaining 21 were anonymous — unable to answer "which marker
    is that?", the question the figure is for. Here every marker is labelled
    once, on the left, and the ordering is shared across panels so a row means
    the same thing throughout.
    """
    contrasts = list(summary["contrast"])
    grey = pal.get("permutation")

    # One ordering for every panel, by overall prominence: the reader learns the
    # row order once instead of re-reading axis labels seven times.
    markers = marker_display_order(per_marker)
    y_position = {m: i for i, m in enumerate(markers)}

    fig = make_subplots(
        rows=1, cols=len(contrasts), shared_yaxes=True, horizontal_spacing=0.012,
        subplot_titles=[str(lbl) for lbl in summary["label"]],
    )
    for i, key in enumerate(contrasts):
        fig.layout.annotations[i].font.color = pal.get(PALETTE_KEY[key])
        fig.layout.annotations[i].font.size = 7.5

    hi = float(per_marker[["mean_share_ws", "mean_share_loso"]].to_numpy().max()) * 1.10

    for col, key in enumerate(contrasts, start=1):
        color = pal.get(PALETTE_KEY[key])
        block = per_marker[per_marker["contrast"] == key].set_index("marker")
        ws = block.loc[markers, "mean_share_ws"].to_numpy()
        lo = block.loc[markers, "mean_share_loso"].to_numpy()
        ys = np.arange(len(markers), dtype=float)

        # Connectors first, as one trace with None separators, so they sit under
        # the points without adding 23 legend-less traces per panel.
        xs_seg, ys_seg = [], []
        for y, a, b in zip(ys, ws, lo):
            xs_seg += [a, b, None]
            ys_seg += [y, y, None]
        fig.add_trace(go.Scatter(
            x=xs_seg, y=ys_seg, mode="lines",
            line=dict(color=color, width=1.0), opacity=0.55,
            hoverinfo="skip", showlegend=False,
        ), row=1, col=col)

        for values, symbol, size, label in ((ws, "circle", 4.5, "within-subject"),
                                            (lo, "diamond", 4.5, "LOSO")):
            fig.add_trace(go.Scatter(
                x=values, y=ys, mode="markers",
                marker=dict(symbol=symbol, size=size, color=color,
                            line=dict(color="white", width=0.4)),
                name=label, legendgroup=label, showlegend=(col == 1),
                hovertemplate="%{y}<br>" + label + ": %{x:.3f}<extra></extra>",
            ), row=1, col=col)

        rho = float(summary.loc[summary["contrast"] == key, "marker_group_rho"].iloc[0])
        fig.add_annotation(
            text=f"ρ = {rho:.2f}", xref="x domain", yref="y domain",
            x=0.97, y=0.02, xanchor="right", yanchor="bottom", showarrow=False,
            font=dict(size=6.5, color=color), row=1, col=col,
        )

        fig.update_xaxes(range=[0, hi], dtick=0.05, tickfont=dict(size=6),
                         row=1, col=col)

    fig.update_yaxes(
        tickmode="array", tickvals=np.arange(len(markers)),
        ticktext=[_pretty_marker(m) for m in markers],
        tickfont=dict(size=6.5), range=[-0.8, len(markers) - 0.2],
        row=1, col=1,
    )
    fig.update_xaxes(title_text="share of total |SHAP|", row=1, col=4)
    sp.grid(fig, "x")
    # Legend above rather than inside: with seven narrow panels there is no free
    # corner, and the marker rows run the full height of every one.
    sp.legend_above(fig)
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Render both figures and report the numbers they encode."""
    pal = init_palette()
    summary, per_subject, per_marker = load_tables()

    print(f"{'dimension':<14}{'n':>4}{'mean rho':>10}{'sig':>7}"
          f"{'group(175)':>12}{'group(23)':>11}{'ceiling':>9}")
    for row in summary.itertuples():
        print(f"{row.label:<14}{row.n_subjects:>4}{row.mean_rho_magnitude:>10.3f}"
              f"{row.n_sig_magnitude:>4}/{row.n_subjects:<3}"
              f"{row.legacy_group_rho:>11.3f}{row.marker_group_rho:>11.3f}"
              f"{row.mean_noise_ceiling:>9.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp.save(build_consistency_figure(summary, per_subject, pal), str(OUT_DIR),
            "fig_feature_consistency", width_mm=MAIN_WIDTH_MM, height_mm=MAIN_HEIGHT_MM)
    sp.save(build_marker_scatter_figure(summary, per_marker, pal), str(OUT_DIR),
            "fig_marker_profiles",
            width_mm=SCATTER_WIDTH_MM, height_mm=SCATTER_HEIGHT_MM)
    sp.save(build_marker_dotplot_figure(summary, per_marker, pal), str(OUT_DIR),
            "fig_marker_profiles_dots",
            width_mm=DOTPLOT_WIDTH_MM, height_mm=DOTPLOT_HEIGHT_MM)
    print(f"\nWrote 3 figures (svg + png) to {OUT_DIR}")


if __name__ == "__main__":
    main()
