#!/usr/bin/env python
"""
Directional forest: do the two pipelines use each marker the same *way*?

The companion share figure asks whether the within-subject and LOSO models lean
on the same markers, and finds that they largely do (Spearman 0.43-0.87 over the
23 markers). Agreeing on which markers matter is not the same as agreeing on
what they mean: a marker can rank at the top of both pipelines while individual
subjects disagree on whether high values push toward or away from the positive
class. This figure asks the directional question.

Direction is the Pearson correlation between a feature's SHAP value and its own
value across a subject's trials — positive meaning higher values of the feature
push the model toward the positive class. A marker's ROI columns are combined as
a mean weighted by mean(|SHAP|), so an ROI the model barely uses cannot outvote
the ones carrying it.

Encoding
--------
Per subject, a circle at their own model's direction; a diamond at the mean
direction of the models trained without them. Colour is the dimension; fill
encodes significance, per the project-wide "colour is identity, fill is
significance" rule — here the significance of that marker's sign concordance
after FDR across the 23 markers, not of any single point.

The bars on the right are the statistic itself: the percentage of subjects whose
own direction has the same sign as the group's view of them, against chance at
50%. A marker at 50% is one where the group-level direction is a net average
over subjects who individually disagree.

Read the bars jointly with panel A. A marker whose group direction sits near
r = 0 has a sign decided by rounding, so its concordance says nothing about
agreement; the two track each other closely (Spearman rho = 0.78 over the 161
marker x dimension cells, printed by main()).

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_marker_direction.py
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recompute_headline_numbers import (  # noqa: E402
    empirical_pvalue,
    fdr_correct,
    load_subject_data,
)


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

WIDTH_MM, HEIGHT_MM = 180.0, 115.0

PALETTE_KEY = {
    "on_off": "onoff", "valence": "valence", "confidence": "confidence",
    "selfother": "selfother", "time": "time",
}

ALPHA = 0.05

# What the two classes are, per dimension: (class 0, class 1). Every contrast in
# the pipeline config uses `positive_above: true` with a within-subject median
# split, so class 1 is always the HIGH raw score. Verified directly against the
# data rather than taken from the config: on/off-task means 74.6 vs 34.5 and
# confidence 86.2 vs 52.7 for y=1 vs y=0.
#
# The poles follow the CLAUDE.md "Pole / extreme wording" short form (SHORT_POLES
# there) — bare pole word, no dimension name appended, since the panel title
# already says which dimension this is: 0 = off-task / negative / past /
# self-focused / low. NOTE that this contradicts `loso_pipeline/config.yaml` for
# self/other, which names the positive class "self" while the positive class is
# the high score, i.e. *other*-focused. The config's two class-name strings
# appear to be swapped; they are cosmetic there (nothing computes from them) but
# they would mislabel any figure that trusted them. Flagged, not silently
# resolved.
CLASS_POLES = {
    "on_off": ("off-task", "on-task"),
    "valence": ("negative", "positive"),
    "confidence": ("low", "high"),
    "selfother": ("self-focused", "other-focused"),
    "time": ("past", "future"),
}

# Dimensions whose plotted direction is negated *for display only* — empty by
# design (kept as machinery, not deleted, in case a future dimension needs it).
#
# self/other used to carry a -1.0 entry here. r is the correlation between a
# feature's SHAP value and its own value, so a positive r means high feature
# values push the model toward class 1 = the HIGH raw score (verified against
# the data: on/off 74.6 vs 34.5 for y=1 vs y=0), and for self/other the high
# raw score is *other*-focused (CLAUDE.md scale convention: 0 = self-focused,
# 100 = other-focused). That alone already lands the natural, un-negated axis
# on the canonical CLAUDE.md pole order (low = self-focused, high =
# other-focused) with no flip needed. The removed entry had negated it anyway,
# reasoning from `loso_pipeline/config.yaml`'s positive-class name ("self")
# rather than from CLAUDE.md's own SHORT_POLES table — but that config string
# is the cosmetic, independently-flagged bug noted above (nothing computes
# from it), not a second source of truth to reconcile with. Confirmed
# 2026-08-13 that nothing downstream depends on the removed flip:
# `marker_level_consistency.csv`, `mean_direction_*` and the sign-concordance
# statistic never read `DISPLAY_FLIP` — it only ever changed what this one
# figure draws.
DISPLAY_FLIP: dict[str, float] = {}


def display_sign(contrast_key: str) -> float:
    """Return -1.0 for dimensions drawn with a negated direction, else +1.0."""
    return DISPLAY_FLIP.get(contrast_key, 1.0)


def axis_title(contrast_key: str, verbose: bool = True) -> str:
    """
    X-axis title naming the two classes the SHAP direction points between.

    A bare "SHAP vs feature value (r)" leaves the reader to remember which class
    is 1, which is the whole meaning of the sign.
    """
    low, high = CLASS_POLES[contrast_key]
    middle = "SHAP vs feature value (r)" if verbose else "SHAP r"
    return f"\u2190 {low}    {middle}    {high} \u2192"

# Rows kept, by LOSO |SHAP|. Ten leaves 29 subject points per row legible;
# all 23 was what made this read as confetti.
N_MARKERS = 10
# Rows per panel in the combined figure — fewer, because seven panels share the
# page and each carries its own labels.
N_MARKERS_COMBINED = 5
# Kept well inside +/-0.5 so a point never strays into a neighbouring row's
# band, which is what makes the rows separable at all.
JITTER_WIDTH = 0.16
DIAMOND_PX = 11.0
# Bar thickness as a fraction of the row pitch. Thin: the bars are a reference
# scale beside the forest, not the subject of the figure.
BAR_WIDTH = 0.55
RANDOM_SEED = 0  # jitter only; no inference depends on it

# =============================================================================
# Palette
# =============================================================================


class RepoPalette:
    """Adapter exposing color_palette.yaml through sciplot's expected interface."""

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


def pretty_marker(name: str) -> str:
    """
    Render a raw marker name for display without making two markers collide.

    Every family keeps a visible tag: stripping ``psd_relative_`` alone would
    leave a bare ``alpha`` sitting in the same axis as ``PE_alpha``, with no way
    to tell which band belongs to which family.
    """
    for old, new in (("psd_relative_", "PSD "), ("kolmogorov_complexity", "Kolmogorov"),
                     ("slowwaves_", "SW "), ("wsmi_", "wSMI "), ("PE_", "PE ")):
        if name.startswith(old) or name == old.rstrip("_"):
            return name.replace(old, new)
    return name


# =============================================================================
# Data
# =============================================================================


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the marker-level tables written by ``feature_consistency_analysis.py``."""
    summary = pd.read_csv(RESULTS_DIR / "group_summary.csv")
    per_marker = pd.read_csv(RESULTS_DIR / "marker_level_consistency.csv")
    per_subject_direction = pd.read_csv(RESULTS_DIR / "marker_direction_per_subject.csv")
    summary = summary.sort_values("legacy_group_rho", ascending=False).reset_index(drop=True)
    return summary, per_marker, per_subject_direction




def subject_classification_significance(contrast: dict) -> dict[str, bool]:
    """
    Which subjects' own within-subject classification beat their permutation null.

    Parameters
    ----------
    contrast : dict
        A ``contrasts`` entry from the analysis config; ``ws_dir`` and
        ``loso_dir`` are used to locate the raw per-run metric files.

    Returns
    -------
    dict[str, bool]
        Zero-padded subject id -> significant after Benjamini-Hochberg across
        the subjects of this dimension.

    Notes
    -----
    This is the same per-subject test the Section 3 decoding figure uses, read
    from the raw per-run CSVs rather than any convenience summary. It lets the
    forest separate two very different subjects: one whose model decodes their
    mind-wandering and lands on a direction, and one whose model is at chance,
    for whom a "direction" is a number computed from noise.
    """
    true_df, perm_df = load_subject_data("ws", contrast)
    subjects = sorted(set(true_df["subject"]))
    p_values = [
        empirical_pvalue(
            true_df.loc[true_df["subject"] == s, "auc"].to_numpy(),
            perm_df.loc[perm_df["subject"] == s, "auc"].to_numpy(),
        )
        for s in subjects
    ]
    adjusted = fdr_correct(p_values)
    return {
        str(s).zfill(2): bool(p < ALPHA) if np.isfinite(p) else False
        for s, p in zip(subjects, adjusted)
    }


# =============================================================================
# Figure
# =============================================================================


def build_figure(
    contrast_key: str,
    label: str,
    per_marker: pd.DataFrame,
    per_subject_direction: pd.DataFrame,
    subject_significant: dict[str, bool],
    pal: RepoPalette,
) -> go.Figure:
    """
    Assemble the two-panel directional figure for one dimension.

    Rows are markers, ordered by the LOSO model's share of |SHAP|. The forest
    gives each marker's direction per subject against the group's; the bars give
    how often the two agree in sign. Diamond area still carries the share, so
    the ordering criterion remains readable without a panel of its own.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    grey = pal.get("permutation")
    color = pal.get(PALETTE_KEY[contrast_key])

    # Ordered by LOSO share: how much the group model — the one the diamonds
    # report and the one the concordance is measured against — leans on each
    # marker. Ranking by an average of both pipelines would sort the rows partly
    # by a quantity neither the diamonds nor the bars show.
    block = (
        per_marker[per_marker["contrast"] == contrast_key]
        .assign(_share=lambda d: d["mean_share_loso"])
        .nlargest(N_MARKERS, "_share")
        .sort_values("_share")           # ascending: y=0 is the bottom row
        .reset_index(drop=True)
    )
    markers = list(block["marker"])
    y_positions = np.arange(len(markers), dtype=float)

    fig = make_subplots(
        # The concordance bars sit hard against the forest so they read as an
        # extension of each row rather than a second chart the eye has to
        # re-associate with the labels.
        rows=1, cols=2, column_widths=[0.80, 0.20],
        shared_yaxes=True, horizontal_spacing=0.004,
        subplot_titles=(f"{label} — direction per subject",
                        "subjects agreeing"),
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 7.5
    fig.layout.annotations[0].font.color = color

    # Alternating row bands across all three panels. With 29 subjects on each of
    # 23 rows the strips bleed into one another and a point cannot be assigned to
    # a marker by eye — the single thing this figure has to make possible. Bands
    # do that without moving any data, and carry the eye from a row's label
    # through its forest strip to its bars.
    for index in range(0, len(markers), 2):
        for panel in (1, 2):
            fig.add_shape(
                type="rect", xref="x domain", yref="y",
                x0=0, x1=1, y0=index - 0.5, y1=index + 0.5,
                fillcolor=grey, opacity=0.10, line_width=0, layer="below",
                row=1, col=panel,
            )

    # ---- Panel A: per-subject directions ----------------------------------
    fig.add_vline(x=0, line=dict(color=grey, width=0.8, dash="dot"),
                  layer="below", row=1, col=1)

    directions = per_subject_direction[per_subject_direction["contrast"] == contrast_key]
    for y, marker in zip(y_positions, markers):
        rows = directions[directions["marker"] == marker]
        values = rows["direction_ws"].to_numpy()
        subjects = rows["subject"].astype(str).str.zfill(2).to_numpy()
        usable = np.isfinite(values)
        values, subjects = values[usable], subjects[usable]
        if values.size == 0:
            continue
        # Fill encodes whether that subject's own classification beat its null,
        # per the project-wide "colour is identity, fill is significance" rule.
        # A hollow point is a direction estimated from a model that never
        # decoded anything for that subject.
        solid = np.array([subject_significant.get(s, False) for s in subjects])
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=values.size)
        fig.add_trace(go.Scatter(
            x=values * display_sign(contrast_key), y=y + jitter, mode="markers",
            marker=dict(
                symbol="circle", size=4,
                color=np.where(solid, color, "rgba(0,0,0,0)"),
                line=dict(color=color, width=0.7),
                opacity=0.55,
            ),
            customdata=subjects,
            hovertemplate=(f"{pretty_marker(marker)}<br>sub %{{customdata}}"
                           "<br>r=%{x:.2f}<extra></extra>"),
            showlegend=False,
        ), row=1, col=1)

    # One size for every diamond. Scaling area by share made the small ones hard
    # to place on the axis, and the row ordering already carries the share.
    shares = block["_share"].to_numpy()
    fig.add_trace(go.Scatter(
        x=block["mean_direction_loso"] * display_sign(contrast_key),
        y=y_positions, mode="markers",
        marker=dict(symbol="diamond", size=DIAMOND_PX, color=color,
                    line=dict(color="white", width=0.6)),
        customdata=shares,
        hovertemplate="LOSO mean r=%{x:.2f}<br>share %{customdata:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ---- Panel B: concordance as a percentage -----------------------------
    percentages = block["sign_concordance"] * 100.0
    fig.add_trace(go.Bar(
        x=percentages, y=y_positions, orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        width=BAR_WIDTH,
        text=[f"{v:.0f}" for v in percentages],
        textposition="outside", textfont=dict(size=5),
        cliponaxis=False,
        hovertemplate="%{x:.0f}% of subjects<extra></extra>", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=50, line=dict(color=grey, width=1.0, dash="dash"),
                  layer="above", row=1, col=2)

    # ---- Legend proxies in neutral ink ------------------------------------
    for symbol, size, name in (
        ("circle", 6, "subject (solid = own classification p<.05)"),
        ("diamond", 8, "LOSO group (all subjects, trained without each)"),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey,
                        line=dict(color=grey, width=1.2)),
            name=name, showlegend=True, hoverinfo="skip",
        ), row=1, col=1)

    # ---- Axes --------------------------------------------------------------
    fig.update_yaxes(
        tickmode="array", tickvals=y_positions,
        ticktext=[pretty_marker(m) for m in markers],
        tickfont=dict(size=6.5), range=[-0.8, len(markers) - 0.2], row=1, col=1,
    )
    fig.update_xaxes(title_text=axis_title(contrast_key), range=[-1.05, 1.05],
                     dtick=0.5, tickfont=dict(size=6.5), row=1, col=1)
    fig.update_xaxes(title_text="% agreeing", range=[0, 132],
                     dtick=50, tickfont=dict(size=6), row=1, col=2)

    sp.grid(fig, "x")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.10,
                    xanchor="center", x=0.5, font=dict(size=6.5)),
        # Top margin as well as bottom: the template's tight default clips the
        # three subplot titles, which plotly positions just above the axes.
        margin=dict(b=40, t=20),
    )
    return fig


# =============================================================================
# Combined figure — every dimension at once
# =============================================================================


def build_combined_figure(
    summary: pd.DataFrame,
    per_marker: pd.DataFrame,
    per_subject_direction: pd.DataFrame,
    significance_by_contrast: dict[str, dict[str, bool]],
    pal: RepoPalette,
) -> go.Figure:
    """
    All dimensions in one figure, each panel showing its own top markers.

    Every panel lists the ``N_MARKERS_COMBINED`` markers with the largest LOSO
    share of |SHAP| **for that dimension**, so the rows differ between panels
    and each carries its own labels.

    A shared row set was tried and is wrong here: averaging the share over
    dimensions to pick one universal list fills each panel with markers that
    dimension may barely use, and buries the ones it actually relies on. The
    question the figure answers is "what does this dimension use, and in which
    direction" — that is a per-dimension question, so the rows have to be too.
    Cross-panel comparison of a given marker is what the single-dimension
    figures and `marker_level_consistency.csv` are for.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    grey = pal.get("permutation")
    contrasts = list(summary["contrast"])
    labels = dict(zip(summary["contrast"], summary["label"]))

    n_cols = 2
    n_rows = int(np.ceil(len(contrasts) / n_cols))
    fig = make_subplots(
        rows=n_rows, cols=n_cols, shared_yaxes=False, shared_xaxes=False,
        horizontal_spacing=0.13, vertical_spacing=0.095,
        subplot_titles=[str(labels[key]) for key in contrasts],
    )
    for index, key in enumerate(contrasts):
        fig.layout.annotations[index].font.size = 8
        fig.layout.annotations[index].font.color = pal.get(PALETTE_KEY[key])

    for index, contrast_key in enumerate(contrasts):
        row, col = index // n_cols + 1, index % n_cols + 1
        color = pal.get(PALETTE_KEY[contrast_key])
        block = (
            per_marker[per_marker["contrast"] == contrast_key]
            .nlargest(N_MARKERS_COMBINED, "mean_share_loso")
            .sort_values("mean_share_loso")       # ascending: y=0 is the bottom
            .reset_index(drop=True)
        )
        markers = list(block["marker"])
        y_positions = np.arange(len(markers), dtype=float)
        significant = significance_by_contrast[contrast_key]

        for band in range(0, len(markers), 2):
            fig.add_shape(
                type="rect", xref="x domain", yref="y",
                x0=0, x1=1, y0=band - 0.5, y1=band + 0.5,
                fillcolor=grey, opacity=0.10, line_width=0, layer="below",
                row=row, col=col,
            )
        fig.add_vline(x=0, line=dict(color=grey, width=0.7, dash="dot"),
                      layer="below", row=row, col=col)

        directions = per_subject_direction[
            per_subject_direction["contrast"] == contrast_key
        ]
        for y, marker in zip(y_positions, markers):
            rows_ = directions[directions["marker"] == marker]
            values = rows_["direction_ws"].to_numpy()
            subjects = rows_["subject"].astype(str).str.zfill(2).to_numpy()
            usable = np.isfinite(values)
            values, subjects = values[usable], subjects[usable]
            if values.size == 0:
                continue
            solid = np.array([significant.get(s, False) for s in subjects])
            fig.add_trace(go.Scatter(
                x=values * display_sign(contrast_key),
                y=y + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=values.size),
                mode="markers",
                marker=dict(symbol="circle", size=3.5,
                            color=np.where(solid, color, "rgba(0,0,0,0)"),
                            line=dict(color=color, width=0.55), opacity=0.6),
                customdata=subjects,
                hovertemplate="sub %{customdata}<br>r=%{x:.2f}<extra></extra>",
                showlegend=False,
            ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=block["mean_direction_loso"] * display_sign(contrast_key),
            y=y_positions, mode="markers",
            marker=dict(symbol="diamond", size=DIAMOND_PX * 0.8, color=color,
                        line=dict(color="white", width=0.5)),
            hovertemplate="LOSO r=%{x:.2f}<extra></extra>", showlegend=False,
        ), row=row, col=col)

        for y, value in zip(y_positions, block["sign_concordance"]):
            fig.add_annotation(
                text=f"{value:.0%}", x=1.30, y=y, xref="x", yref="y",
                xanchor="right", yanchor="middle", showarrow=False,
                font=dict(size=5.5, color=grey), row=row, col=col,
            )

        # Every panel gets its own pole labels: each is a different dimension,
        # so a single shared axis title at the bottom would name the wrong
        # classes for six of the seven.
        fig.update_xaxes(range=[-1.05, 1.34], dtick=0.5, tickfont=dict(size=6),
                         title_text=axis_title(contrast_key, verbose=False),
                         title_font=dict(size=6), title_standoff=3,
                         row=row, col=col)
        fig.update_yaxes(
            tickmode="array", tickvals=y_positions,
            ticktext=[pretty_marker(m) for m in markers],
            tickfont=dict(size=6.5), range=[-0.6, len(markers) - 0.4],
            row=row, col=col,
        )

    for symbol, size, name in (
        ("circle", 5, "subject (solid = own classification p<.05)"),
        ("diamond", 7, "LOSO group  \u00b7  % = subjects agreeing in sign"),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey,
                        line=dict(color=grey, width=1.0)),
            name=name, showlegend=True, hoverinfo="skip",
        ), row=1, col=1)

    sp.grid(fig, "x")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.045,
                    xanchor="center", x=0.5, font=dict(size=6.5)),
        margin=dict(b=30, t=18),
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Render the directional figure for every dimension."""
    pal = init_palette()
    summary, per_marker, per_subject_direction = load_tables()
    config = yaml.safe_load((_SCRIPTS_DIR / "config_feature_consistency.yaml").read_text())
    contrasts = {c["key"]: c for c in config["contrasts"]}
    # Labels come from the config, not from group_summary.csv: renaming a
    # dimension there should not require an 18-minute re-analysis to show up in
    # a figure. The CSV keeps whatever label was current when it was written.
    summary["label"] = summary["contrast"].map(
        {key: value["label"] for key, value in contrasts.items()}
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    significance_by_contrast: dict[str, dict[str, bool]] = {}
    print(f"{'dimension':<14}{'mean concordance':>18}{'>chance (FDR)':>16}{'subj sig':>10}")
    for row in summary.itertuples():
        block = per_marker[per_marker["contrast"] == row.contrast]
        above = block[(block["p_sign_concordance_fdr"] < ALPHA)
                      & (block["sign_concordance"] > 0.5)]
        subject_significant = subject_classification_significance(contrasts[row.contrast])
        n_sig = sum(subject_significant.values())
        print(f"{row.label:<14}{block['sign_concordance'].mean():>18.3f}"
              f"{len(above):>9}/{len(block):<6}{n_sig:>5}/{len(subject_significant):<4}")

        fig = build_figure(row.contrast, str(row.label), per_marker,
                           per_subject_direction, subject_significant, pal)
        sp.save(fig, str(OUT_DIR), f"fig_marker_direction_{row.contrast}",
                width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
        significance_by_contrast[row.contrast] = subject_significant

    # Concordance is only interpretable where a direction exists at all: a marker
    # whose group direction sits at r = 0 has a sign decided by rounding, so its
    # concordance says nothing about agreement. Printed because it is the
    # qualifier the figures need to be read correctly.
    sp.save(build_combined_figure(summary, per_marker, per_subject_direction,
                                  significance_by_contrast, pal),
            str(OUT_DIR), "fig_marker_direction_all_dimensions",
            width_mm=180.0, height_mm=185.0)

    strength = per_marker["mean_direction_loso"].abs()
    rho, p_value = spearmanr(strength, per_marker["sign_concordance"])
    print(f"\n|group direction| vs concordance: rho = {rho:.3f}, p = {p_value:.2g}, "
          f"n = {len(per_marker)} marker x dimension cells")
    bins = pd.cut(strength, [0, 0.1, 0.3, 0.5, 1.0],
                  labels=["<0.1", "0.1-0.3", "0.3-0.5", ">0.5"])
    print(per_marker.groupby(bins, observed=True)["sign_concordance"]
          .agg(["count", "mean"]).round(3).to_string())
    print(f"\nWrote {len(summary)} figures (svg + png) to {OUT_DIR}")


if __name__ == "__main__":
    main()
