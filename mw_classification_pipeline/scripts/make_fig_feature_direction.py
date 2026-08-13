#!/usr/bin/env python
"""
Per-ROI directional forest — the top features the LOSO model relies on.

Companion to ``make_fig_marker_direction.py``, not a replacement. That figure
asks the directional question of the 23 CBPT markers; this one asks it of the
individual ROI columns, which is where a *signed* quantity is actually well
defined.

Why both exist
--------------
|SHAP| is positive and additive, so averaging a marker's nine ROI columns is
sound and the marker-level share is solid. Directions are signed and cancel.
Measured on on/off-task, the sign coherence among a marker's ROIs is 0.73 within
subject and 0.77 for LOSO — nearly equal, so the ROI disagreement is a real
property of the data rather than within-subject noise. "The direction of PSD
alpha" is therefore not a well-defined quantity when its nine ROIs point
different ways, and averaging them to ~0 is honest but uninformative. At ROI
resolution nothing cancels.

The selection effect, declared
------------------------------
Rows are the ``N_FEATURES`` columns with the largest mean |SHAP| in the LOSO
model. This is a selection, and it inflates |r|: on on/off-task the fraction of
subject-level directions with |r| > 0.5 is 44% among the top 8 features versus
35% over all 175. It is *not* circular — the columns are chosen by |SHAP| and
read out on the sign of a correlation, which are different quantities — and the
same columns are used for both pipelines. But the numbers here are not
comparable to numbers computed over all features, and should not be quoted as if
they were.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_feature_direction.py
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fig_marker_direction import (  # noqa: E402
    RepoPalette,
    init_palette,
    subject_classification_significance,
)

_SKILL_DIR = Path("/network/iss/home/nicolas.bruno/.claude/skills/scientific-plots")
sys.path.insert(0, str(_SKILL_DIR / "scripts"))
import sciplot as sp  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
_SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPTS_DIR.parent / "results" / "feature_consistency"
OUT_DIR = _SCRIPTS_DIR.parent / "results" / "figures" / "feature_consistency"

# Ten rows keeps 29 subject points per row legible. The marker-level figure's 23
# rows x 29 points is what made that one read as confetti.
N_FEATURES = 10

WIDTH_MM, HEIGHT_MM = 180.0, 78.0

PALETTE_KEY = {
    "on_off": "onoff", "valence": "valence", "confidence": "confidence",
    "selfother": "selfother", "time": "time",
    "valence_sq": "valence_sq", "time_sq": "time_sq",
}

ALPHA = 0.05
JITTER_WIDTH = 0.17
MIN_DIAMOND_PX, MAX_DIAMOND_PX = 6.0, 14.0
BAR_WIDTH = 0.55
RANDOM_SEED = 0  # jitter only; no inference depends on it

# =============================================================================
# Labels
# =============================================================================

_MARKER_SHORT = {
    "psd_relative_": "PSD ", "kolmogorov_complexity": "Kolmogorov",
    "slowwaves_": "SW ", "wsmi_": "wSMI ", "PE_": "PE ",
}
_ROI_SHORT = {
    "frontal": "front", "central": "cent", "posterior": "post",
    "occipitoparietal": "occ-par", "frontocentral": "front-cent",
    "centroparietal": "cent-par",
}
_SIDE_SHORT = {"left": "L", "right": "R", "mid": "M",
               "midline": "mid", "lateral": "lat"}


def pretty_feature(name: str) -> str:
    """
    Turn ``PE_gamma_mean_frontal_right_trimmean`` into ``PE gamma · front-R``.

    The raw names are 35-45 characters and do not fit a 180 mm figure's label
    gutter without eating the plot area. Family prefixes are kept (``PE``,
    ``PSD``, ``wSMI``) because dropping one would leave a bare ``alpha`` next to
    ``PE alpha`` with no way to tell them apart.
    """
    match = re.match(r"^(.*)_mean_([a-z]+)_([a-z]+)_trimmean$", name)
    if match is None:
        raise ValueError(
            f"Feature '{name}' does not match the expected "
            f"<marker>_mean_<roi>_<side>_trimmean grammar; the naming "
            f"convention changed and this label map needs updating."
        )
    marker, roi, side = match.groups()
    for old, new in _MARKER_SHORT.items():
        if marker.startswith(old) or marker == old.rstrip("_"):
            marker = marker.replace(old, new)
            break
    return f"{marker} · {_ROI_SHORT.get(roi, roi)}-{_SIDE_SHORT.get(side, side)}"


# =============================================================================
# Figure
# =============================================================================


def build_figure(
    contrast_key: str,
    label: str,
    per_feature: pd.DataFrame,
    per_subject_feature: pd.DataFrame,
    subject_significant: dict[str, bool],
    pal: RepoPalette,
) -> go.Figure:
    """Assemble the per-ROI directional forest for one dimension."""
    rng = np.random.default_rng(RANDOM_SEED)
    grey = pal.get("permutation")
    color = pal.get(PALETTE_KEY[contrast_key])

    block = (
        per_feature[per_feature["contrast"] == contrast_key]
        .nlargest(N_FEATURES, "mean_share_loso")
        .sort_values("mean_share_loso")
        .reset_index(drop=True)
    )
    features = list(block["feature"])
    y_positions = np.arange(len(features), dtype=float)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.80, 0.20],
        shared_yaxes=True, horizontal_spacing=0.004,
        subplot_titles=(f"{label} — direction per subject", "subjects agreeing"),
    )
    for annotation in fig.layout.annotations:
        annotation.font.size = 7.5
    fig.layout.annotations[0].font.color = color

    # Alternating row bands carry the eye from a label through its strip to its
    # bar; without them the strips bleed together and a point cannot be assigned
    # to a row, which is the one thing the figure has to make possible.
    for index in range(0, len(features), 2):
        for panel in (1, 2):
            fig.add_shape(
                type="rect", xref="x domain", yref="y",
                x0=0, x1=1, y0=index - 0.5, y1=index + 0.5,
                fillcolor=grey, opacity=0.10, line_width=0, layer="below",
                row=1, col=panel,
            )

    fig.add_vline(x=0, line=dict(color=grey, width=0.8, dash="dot"),
                  layer="below", row=1, col=1)

    directions = per_subject_feature[per_subject_feature["contrast"] == contrast_key]
    for y, feature in zip(y_positions, features):
        rows = directions[directions["feature"] == feature]
        values = rows["direction_ws"].to_numpy()
        subjects = rows["subject"].astype(str).str.zfill(2).to_numpy()
        usable = np.isfinite(values)
        values, subjects = values[usable], subjects[usable]
        if values.size == 0:
            continue
        # Fill encodes whether that subject's own classification beat its null:
        # a hollow point is a direction read off a model that never decoded
        # anything for that person.
        solid = np.array([subject_significant.get(s, False) for s in subjects])
        fig.add_trace(go.Scatter(
            x=values, y=y + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=values.size),
            mode="markers",
            marker=dict(symbol="circle", size=5,
                        color=np.where(solid, color, "rgba(0,0,0,0)"),
                        line=dict(color=color, width=0.8), opacity=0.65),
            customdata=subjects,
            hovertemplate="sub %{customdata}<br>r=%{x:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

    shares = block["mean_share_loso"].to_numpy()
    sizes = MIN_DIAMOND_PX + (MAX_DIAMOND_PX - MIN_DIAMOND_PX) * (
        (shares - shares.min()) / np.ptp(shares)
    )
    fig.add_trace(go.Scatter(
        x=block["mean_direction_loso"], y=y_positions, mode="markers",
        marker=dict(symbol="diamond", size=sizes, color=color,
                    line=dict(color="white", width=0.6)),
        customdata=shares,
        hovertemplate="LOSO r=%{x:.2f}<br>share %{customdata:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    percentages = block["sign_concordance"] * 100.0
    fig.add_trace(go.Bar(
        x=percentages, y=y_positions, orientation="h",
        marker=dict(color=color, line=dict(width=0)), width=BAR_WIDTH,
        text=[f"{v:.0f}" for v in percentages],
        textposition="outside", textfont=dict(size=5.5), cliponaxis=False,
        hovertemplate="%{x:.0f}% of subjects<extra></extra>", showlegend=False,
    ), row=1, col=2)
    fig.add_vline(x=50, line=dict(color=grey, width=1.0, dash="dash"),
                  layer="above", row=1, col=2)

    for symbol, size, name in (
        ("circle", 6, "subject (solid = own classification p<.05)"),
        ("diamond", 8, "LOSO group (area = share of |SHAP|)"),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey,
                        line=dict(color=grey, width=1.2)),
            name=name, showlegend=True, hoverinfo="skip",
        ), row=1, col=1)

    fig.update_yaxes(
        tickmode="array", tickvals=y_positions,
        ticktext=[pretty_feature(f) for f in features],
        tickfont=dict(size=7), range=[-0.7, len(features) - 0.3], row=1, col=1,
    )
    fig.update_xaxes(title_text="SHAP vs feature value (r)", range=[-1.05, 1.05],
                     dtick=0.5, tickfont=dict(size=7), row=1, col=1)
    fig.update_xaxes(title_text="% agreeing", range=[0, 132], dtick=50,
                     tickfont=dict(size=6.5), row=1, col=2)

    sp.grid(fig, "x")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.16,
                    xanchor="center", x=0.5, font=dict(size=6.5)),
        margin=dict(b=44, t=20),
    )
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Render the per-ROI directional forest for every dimension."""
    pal = init_palette()
    per_feature = pd.read_csv(RESULTS_DIR / "feature_level_direction.csv")
    per_subject_feature = pd.read_csv(RESULTS_DIR / "feature_direction_per_subject.csv")
    config = yaml.safe_load(
        (_SCRIPTS_DIR / "config_feature_consistency.yaml").read_text()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{'dimension':<14}{'top feature (LOSO |SHAP|)':<34}"
          f"{'r':>7}{'agree':>8}")
    for contrast in config["contrasts"]:
        block = per_feature[per_feature["contrast"] == contrast["key"]]
        if block.empty:
            continue
        top = block.nlargest(1, "mean_share_loso").iloc[0]
        print(f"{contrast['label']:<14}{pretty_feature(top['feature']):<34}"
              f"{top['mean_direction_loso']:>7.2f}{top['sign_concordance']:>8.0%}")

        fig = build_figure(
            contrast["key"], str(contrast["label"]), per_feature,
            per_subject_feature,
            subject_classification_significance(contrast), pal,
        )
        sp.save(fig, str(OUT_DIR), f"fig_feature_direction_{contrast['key']}",
                width_mm=WIDTH_MM, height_mm=HEIGHT_MM)

    print(f"\nWrote {len(config['contrasts'])} figures (svg + png) to {OUT_DIR}")


if __name__ == "__main__":
    main()
