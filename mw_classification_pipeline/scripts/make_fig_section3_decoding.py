#!/usr/bin/env python
"""
Paper 2, Section 3 main figure — decodability of MW dimensions.

One panel. Per probe dimension: the within-subject per-subject AUCs as a jittered
strip, the within-subject group mean, the LOSO group mean, and the permutation
null as a grey band. Dimensions are ordered by within-subject AUC.

Encoding notes
--------------
A dot plot rather than a bar chart: AUC is bounded below by 0.5, so bars drawn
from zero (as bars must be) spend 70% of their length on an uninformative range
and compress the 0.50-0.75 span where the whole result lives. With points, the
vertical distance between the within-subject marker and the LOSO marker reads
directly as the generalisation gap, which is the comparison the figure exists to
make.

Colour encodes the dimension (from ``color_palette.yaml``); significance is
encoded by fill — solid = significant after FDR, hollow = not — per the
project-wide convention.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_section3_decoding.py
"""

# =============================================================================
# Imports
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml

_SKILL_DIR = Path("/network/iss/home/nicolas.bruno/.claude/skills/scientific-plots")
sys.path.insert(0, str(_SKILL_DIR / "scripts"))
import sciplot as sp  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
from recompute_headline_numbers import (  # noqa: E402
    DIMENSIONS,
    empirical_pvalue,
    fdr_correct,
    load_group_data,
    load_subject_data,
)

# =============================================================================
# Configuration
# =============================================================================
OUT_DIR = _SCRIPTS_DIR.parent / "results" / "figures" / "section3_decoding"
FIG_NAME = "fig_section3_decoding"

WIDTH_MM = 120.0
HEIGHT_MM = 78.0

# Maps the recompute script's dimension keys onto the palette's dimension names.
PALETTE_KEY = {
    "on_off": "onoff",
    "valence": "valence",
    "time": "time",
    "selfother": "selfother",
    "confidence": "confidence",
}

JITTER_WIDTH = 0.20
NULL_BAND_HALF_WIDTH = 0.30
RANDOM_SEED = 0  # jitter only; no inference depends on it

PALETTE_PATH = Path(__file__).resolve().parents[2] / "color_palette.yaml"

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
        # Named continuous scales; the project file declares none, so the
        # helper's own defaults apply.
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
# Data assembly
# =============================================================================


def collect_results() -> pd.DataFrame:
    """
    Pull every quantity the figure needs straight from the raw per-run CSVs.

    Returns
    -------
    pd.DataFrame
        One row per dimension with within-subject and LOSO group AUCs, their
        FDR-corrected p-values, the permutation-null percentiles, and the list
        of per-subject within-subject AUCs.
    """
    rows: list[dict] = []
    ws_pvals: list[float] = []
    loso_pvals: list[float] = []

    for dim in DIMENSIONS:
        ws_true, ws_perm = load_group_data("ws", dim)
        loso_true, loso_perm = load_group_data("loso", dim)

        ws_subj, _ = load_subject_data("ws", dim)
        subj_auc = ws_subj.groupby("subject")["auc"].mean()

        ws_pvals.append(empirical_pvalue(ws_true, ws_perm))
        loso_pvals.append(empirical_pvalue(loso_true, loso_perm))

        rows.append({
            "key": dim["key"],
            "label": dim["label"],
            "ws_auc": float(np.mean(ws_true)),
            "loso_auc": float(np.mean(loso_true)),
            "null_lo": float(np.percentile(ws_perm, 2.5)),
            "null_hi": float(np.percentile(ws_perm, 97.5)),
            "subject_aucs": subj_auc.to_numpy(),
            "n_subjects": int(subj_auc.size),
        })

    df = pd.DataFrame(rows)
    df["ws_p_fdr"] = fdr_correct(ws_pvals)
    df["loso_p_fdr"] = fdr_correct(loso_pvals)
    return df.sort_values("ws_auc", ascending=False).reset_index(drop=True)


# =============================================================================
# Figure
# =============================================================================


def build_figure(df: pd.DataFrame, pal: "RepoPalette") -> go.Figure:
    """Assemble the single-panel decoding figure."""
    rng = np.random.default_rng(RANDOM_SEED)
    fig = go.Figure()

    grey = pal.get("permutation")
    x_positions = np.arange(len(df), dtype=float)

    # Permutation null (2.5-97.5%) as a grey band behind everything: the
    # significance threshold, readable without consulting the caption.
    for x, row in zip(x_positions, df.itertuples()):
        fig.add_shape(
            type="rect",
            x0=x - NULL_BAND_HALF_WIDTH, x1=x + NULL_BAND_HALF_WIDTH,
            y0=row.null_lo, y1=row.null_hi,
            fillcolor=grey, opacity=0.22, line_width=0, layer="below",
        )

    fig.add_hline(y=0.5, line=dict(color=grey, width=0.8, dash="dot"), layer="below")

    # Per-subject within-subject AUCs.
    for x, row in zip(x_positions, df.itertuples()):
        color = pal.get(PALETTE_KEY[row.key])
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=row.subject_aucs.size)
        fig.add_trace(go.Scatter(
            x=x + jitter, y=row.subject_aucs,
            mode="markers",
            marker=dict(color=color, size=3.5, opacity=0.32,
                        line=dict(width=0.3, color="white")),
            hovertemplate=f"{row.label}<br>subject AUC: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        ))

    # Connector between the two group means. Keeping every subject point (some
    # subjects land below 0.25) forces a tall y-range that compresses the
    # 0.50-0.75 band where the group effects live, so the generalisation gap is
    # drawn explicitly as a segment: its length stays readable regardless of how
    # much the axis has to stretch for the outliers.
    for x, row in zip(x_positions, df.itertuples()):
        fig.add_trace(go.Scatter(
            x=[x - 0.13, x + 0.13], y=[row.ws_auc, row.loso_auc],
            mode="lines",
            line=dict(color=pal.get(PALETTE_KEY[row.key]), width=1.6),
            hoverinfo="skip", showlegend=False,
        ))

    # Group means. Fill carries significance, colour carries dimension.
    for pipeline, symbol, size, offset in (("ws", "circle", 11, -0.13),
                                           ("loso", "diamond", 10, 0.13)):
        xs, ys, marker_colors, line_colors = [], [], [], []
        for x, row in zip(x_positions, df.itertuples()):
            color = pal.get(PALETTE_KEY[row.key])
            auc = row.ws_auc if pipeline == "ws" else row.loso_auc
            sig = (row.ws_p_fdr if pipeline == "ws" else row.loso_p_fdr) < 0.05
            xs.append(x + offset)
            ys.append(auc)
            marker_colors.append(color if sig else "white")
            line_colors.append(color)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(symbol=symbol, size=size, color=marker_colors,
                        line=dict(color=line_colors, width=1.8)),
            hovertemplate="%{y:.3f}<extra></extra>",
            showlegend=False,
        ))

    # Legend proxies in neutral ink. The real markers are coloured per dimension,
    # so letting them into the legend would imply hue encodes the pipeline; here
    # hue encodes the dimension and the marker shape encodes the pipeline.
    # Solid grey, not hollow: hollow is already spoken for as "not significant",
    # so a hollow legend key would read as though LOSO were non-significant by
    # definition. Grey is the house neutral and reads as context, not result.
    for symbol, size, name in (("circle", 9, "within-subject"), ("diamond", 8, "LOSO")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey,
                        line=dict(color=grey, width=1.6)),
            name=name, showlegend=True, hoverinfo="skip",
        ))

    # Data-driven limits: a fixed range silently clipped the highest and lowest
    # per-subject AUCs, which is not a defensible thing for a figure to do.
    all_subject_aucs = np.concatenate(df["subject_aucs"].to_list())
    y_lo = min(all_subject_aucs.min(), df["null_lo"].min())
    y_hi = all_subject_aucs.max()
    pad = 0.04 * (y_hi - y_lo)

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=[f"{r.label}<br>n={r.n_subjects}" for r in df.itertuples()],
        range=[-0.55, len(df) - 0.45],
    )
    fig.update_yaxes(title_text="AUC", range=[y_lo - pad, y_hi + pad], dtick=0.1)

    sp.grid(fig, "y")
    sp.legend_above(fig)
    fig.update_layout(margin=dict(l=6))
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    pal = init_palette()
    df = collect_results()

    print(f"{'dimension':<14}{'WS AUC':>9}{'p_FDR':>9}{'LOSO AUC':>10}{'p_FDR':>9}{'gap':>8}{'n':>5}")
    for row in df.itertuples():
        print(f"{row.label:<14}{row.ws_auc:>9.3f}{row.ws_p_fdr:>9.4f}"
              f"{row.loso_auc:>10.3f}{row.loso_p_fdr:>9.4f}"
              f"{row.ws_auc - row.loso_auc:>8.3f}{row.n_subjects:>5d}")

    fig = build_figure(df, pal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp.save(fig, str(OUT_DIR), FIG_NAME, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    print(f"\nWrote {OUT_DIR / FIG_NAME}.svg / .png")


if __name__ == "__main__":
    main()
