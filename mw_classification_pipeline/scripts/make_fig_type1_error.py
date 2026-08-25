#!/usr/bin/env python
"""
Type I error calibration, combined across dimensions.

Each dimension's ``type1_error/run_type1_error_*.py`` reruns the full
classification pipeline (true + permutation passes) on synthetic null
features 200 times, keeping the real labels. Under the null, the pipeline's
own empirical p-value should be U[0,1], so the fraction of the 200 runs
called "significant" at a nominal alpha should equal that alpha. This figure
puts every completed dimension's calibration on one page instead of the
separate per-dimension plotly HTMLs each run already writes under
``.../rf/plots/``.

Only the LOSO pipeline has all five canonical dimensions run, in both their
plain and residualized-contrast forms (10 completed simulations). The
within-subject pipeline has only on/off-task done so far (the others were
never submitted) — it is drawn as a third, differently-shaped series in the
on/off panel only, not as a sixth panel, so its absence elsewhere is not
mistaken for zero rather than "not run".

Quadratic terms (``valence_sq``/``time_sq``) were removed from every project
analysis 2026-08-13 (CLAUDE.md "Quadratic Terms: Removed") — their Type1Error
directories are stale leftovers and are not read here.

Encoding
--------
Color is the dimension (project palette). Marker shape is which pipeline/
contrast: circle = LOSO plain, diamond = LOSO residualized, triangle-up =
within-subject plain. Fill follows the project-wide "color is identity, fill
is significance" rule, reused here for the question this figure actually
asks: solid = the empirical FPR's Wilson CI excludes the nominal alpha (this
point is significantly miscalibrated); hollow = the CI contains alpha
(consistent with correct calibration). The dashed gray diagonal is not a
model fit, it is the definition of "calibrated" (FPR = alpha).

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_type1_error.py
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

_SKILL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "scientific_plots"
sys.path.insert(0, str(_SKILL_DIR))
import sciplot as sp  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
_SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS_DIR.parents[1]
RESULTS_ROOT = _SCRIPTS_DIR.parent / "results" / "MW_Classification" / "Type1Error"
OUT_DIR = _SCRIPTS_DIR.parent / "results" / "combined_figures"
PALETTE_PATH = REPO_ROOT / "color_palette.yaml"

WIDTH_MM, HEIGHT_MM = 180.0, 78.0
# The project's headline decoding metric everywhere else (AUC), but the two
# type1_error runners name its summary row differently: LOSO's aggregator
# reports four metrics as "mean_{metric}" (it averages over n_runs true
# passes per simulation); the within-subject aggregator reports one metric
# per simulation and calls it "auc" with no "mean_" prefix. Confirmed against
# both on-disk summary CSVs 2026-08-18 — not a typo to "fix" in one of them.
PRIMARY_METRIC = {"loso_plain": "mean_auc", "loso_res": "mean_auc", "ws_plain": "auc"}

# Canonical dimension order (CLAUDE.md "Dimension Order (Figures)"). Directory
# names come straight from the on-disk Type1Error tree (verified via `find`);
# the on/off plain dir keeps its historical different casing from every other
# plain dir, and from its own residualized dir — same inconsistency already
# flagged in scripts/config_feature_consistency.yaml.
DIMENSIONS = [
    dict(key="onoff", label="On/Off-Task", palette_key="onoff",
         loso_plain="ON_vs_OFF_within_median", loso_res="onoff_within_median_res",
         ws_plain="on_vs_off_within_median"),
    dict(key="valence", label="Valence", palette_key="valence",
         loso_plain="valence_within_median", loso_res="valence_within_median_res",
         ws_plain=None),
    dict(key="selfother", label="Self/Other", palette_key="selfother",
         loso_plain="selfother_within_median", loso_res="selfother_within_median_res",
         ws_plain=None),
    dict(key="time", label="Time", palette_key="time",
         loso_plain="time_within_median", loso_res="time_within_median_res",
         ws_plain=None),
    dict(key="confidence", label="Confidence", palette_key="confidence",
         loso_plain="confidence_within_median", loso_res="confidence_within_median_res",
         ws_plain=None),
]

# x-offsets so plain / residualized / within-subject points at the same
# nominal alpha don't sit on top of each other or their error bars.
X_JITTER = {"loso_plain": -0.003, "loso_res": 0.0, "ws_plain": 0.003}
MARKER_SYMBOL = {"loso_plain": "circle", "loso_res": "diamond", "ws_plain": "triangle-up"}
SERIES_LABEL = {"loso_plain": "LOSO", "loso_res": "LOSO, residualized",
                "ws_plain": "within-subject"}
SERIES_DASH = {"loso_plain": "solid", "loso_res": "dash", "ws_plain": "dot"}


# =============================================================================
# Palette (repo schema differs from sciplot's default — see make_fig_marker_direction.py)
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


# =============================================================================
# Data
# =============================================================================
def load_summary(dim_key: str, series: str, subdir: str | None) -> pd.DataFrame | None:
    """Read ``type1_error_summary.csv`` for one dimension x series, if it exists."""
    if subdir is None:
        return None
    pipeline = "LOSO" if series.startswith("loso") else "WithinSubject"
    path = RESULTS_ROOT / pipeline / subdir / "all" / "rf" / "type1_error_summary.csv"
    if not path.exists():
        print(f"  [skip] {dim_key}/{series}: {path} not found")
        return None
    df = pd.read_csv(path)
    df = df[df["metric"] == PRIMARY_METRIC[series]].copy()
    if df.empty:
        raise RuntimeError(
            f"{path} has no rows with metric == {PRIMARY_METRIC[series]!r}; "
            f"metrics present: {sorted(pd.read_csv(path)['metric'].unique())}"
        )
    df["dimension"] = dim_key
    df["series"] = series
    return df


def load_all() -> pd.DataFrame:
    """Collect every completed dimension x series into one long table."""
    frames = []
    for dim in DIMENSIONS:
        for series, key in (("loso_plain", "loso_plain"), ("loso_res", "loso_res"),
                             ("ws_plain", "ws_plain")):
            frame = load_summary(dim["key"], series, dim[key])
            if frame is not None:
                frames.append(frame)
    if not frames:
        raise RuntimeError(f"No type1_error_summary.csv found under {RESULTS_ROOT}")
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Figure
# =============================================================================
def build_figure(data: pd.DataFrame, pal: RepoPalette) -> go.Figure:
    """One panel per dimension, FPR vs nominal alpha for every completed series."""
    grey = pal.get("permutation")
    keys = [d["key"] for d in DIMENSIONS]
    labels = {d["key"]: d["label"] for d in DIMENSIONS}

    alpha_max = data["alpha"].max()
    fpr_max = max(data["ci_upper"].max(), alpha_max) * 1.15

    fig = sp.facet_subplots(
        rows=1, cols=len(keys),
        subplot_titles=[labels[k] for k in keys],
        shared_yaxes=True,
        yaxis_title="empirical FPR",
        xaxis_title="nominal α",
        width_mm=WIDTH_MM, height_mm=HEIGHT_MM,
    )
    for index, key in enumerate(keys):
        fig.layout.annotations[index].font.color = pal.get(
            next(d["palette_key"] for d in DIMENSIONS if d["key"] == key)
        )

    for col, dim in enumerate(DIMENSIONS, start=1):
        color = pal.get(dim["palette_key"])
        block = data[data["dimension"] == dim["key"]]

        # Calibration reference: FPR = alpha, not a fitted line.
        fig.add_shape(
            type="line", x0=0, y0=0, x1=alpha_max * 1.1, y1=alpha_max * 1.1,
            line=dict(color=grey, width=1.0, dash="dash"),
            layer="below", row=1, col=col,
        )

        for series in ("loso_plain", "loso_res", "ws_plain"):
            rows = block[block["series"] == series].sort_values("alpha")
            if rows.empty:
                continue
            miscalibrated = (rows["alpha"] < rows["ci_lower"]) | (rows["alpha"] > rows["ci_upper"])
            fill_colors = [color if m else "white" for m in miscalibrated]
            x = rows["alpha"] + X_JITTER[series]
            fig.add_trace(go.Scatter(
                x=x, y=rows["fpr"], mode="lines+markers",
                line=dict(color=color, width=1.0, dash=SERIES_DASH[series]),
                marker=dict(
                    symbol=MARKER_SYMBOL[series], size=6.5,
                    color=fill_colors,
                    line=dict(color=color, width=1.1),
                ),
                customdata=rows[["n_simulations", "n_significant"]].to_numpy(),
                hovertemplate=(f"{SERIES_LABEL[series]}<br>α=%{{x:.2f}}"
                               "<br>FPR=%{y:.3f} (%{customdata[1]:.0f}/%{customdata[0]:.0f})"
                               "<extra></extra>"),
                error_y=dict(
                    type="data", symmetric=False,
                    array=rows["ci_upper"] - rows["fpr"],
                    arrayminus=rows["fpr"] - rows["ci_lower"],
                    color=color, thickness=0.9, width=2,
                ),
                showlegend=False,
            ), row=1, col=col)

        fig.update_xaxes(range=[0, alpha_max * 1.25],
                         tickvals=sorted(data["alpha"].unique()), row=1, col=col)

    fig.update_yaxes(range=[0, fpr_max], col=1)

    # ---- Legend proxies (neutral ink; shape/fill carry the meaning, not color) ---
    for series in ("loso_plain", "loso_res", "ws_plain"):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=MARKER_SYMBOL[series], size=8, color=grey,
                        line=dict(color=grey, width=1.2)),
            name=SERIES_LABEL[series], showlegend=True, hoverinfo="skip",
        ), row=1, col=1)
    for solid, name in ((True, "miscalibrated (CI excludes α)"),
                        (False, "calibrated (CI contains α)")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=8,
                        color=grey if solid else "white",
                        line=dict(color=grey, width=1.2)),
            name=name, showlegend=True, hoverinfo="skip",
        ), row=1, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5, font=dict(size=7.5)),
        margin=dict(b=52, t=20),
    )
    return fig


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Render the combined Type I error calibration figure."""
    pal = init_palette()
    print(f"Reading Type1Error summaries from {RESULTS_ROOT}")
    data = load_all()

    print(f"\n{'dimension':<12}{'series':<20}{'alpha':>7}{'fpr':>8}{'ci':>16}{'calibrated?':>13}")
    for row in data.sort_values(["dimension", "series", "alpha"]).itertuples():
        calibrated = row.ci_lower <= row.alpha <= row.ci_upper
        print(f"{row.dimension:<12}{row.series:<20}{row.alpha:>7.2f}{row.fpr:>8.3f}"
              f"  [{row.ci_lower:.3f},{row.ci_upper:.3f}]{str(calibrated):>13}")

    covered = sorted(data["dimension"].unique())
    missing_ws = [d["label"] for d in DIMENSIONS if d["ws_plain"] is None]
    print(f"\nDimensions with LOSO Type1Error data: {covered}")
    print(f"Within-subject Type1Error not yet run for: {missing_ws} "
          "(only on/off-task submitted so far)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure(data, pal)
    sp.save(fig, str(OUT_DIR), "type1_error_calibration",
            width_mm=WIDTH_MM, height_mm=HEIGHT_MM, data=data)
    print(f"\nWrote type1_error_calibration.{{svg,png,csv}} to {OUT_DIR}")


if __name__ == "__main__":
    main()
