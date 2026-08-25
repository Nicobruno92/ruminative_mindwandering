#!/usr/bin/env python
"""
WS-vs-LOSO SHAP-direction forest plot — per-subject sign heterogeneity.

A 2x3 grid: one small forest panel per probe dimension (top row: onoff,
valence, confidence; bottom row: selfother, time) plus, in the sixth slot, the
Spearman correlation matrix across the probe dimensions themselves (the same
panel built in Behavior/Probe_analysis/probe_dimension_cloud_plot.py, adapted
here for an arbitrary grid cell instead of a fixed single row).

Each dimension panel shows, for its top-5 LOSO features (by mean|SHAP|), the
LOSO group-level correlation between SHAP and feature value (a diamond)
against every within-subject model's own correlation for the same feature (a
jittered strip of circles). The question this figure answers: does the
group-level direction reflect a shared relationship, or is it a net average
over subjects who individually disagree on sign?

Encoding notes
--------------
Per-subject rows are computed directly from each pipeline's saved SHAP pkls
(nonzero rows only — a feature a subject's fold did not mRMR-select is a
structural zero, not a measured null effect, per
mw_classification_pipeline/utils/ml_utils.py:compute_shap_values_for_pipeline).

Colour is the dimension's own colour (from color_palette.yaml), carried by
both the markers and the feature's y-tick label; marker SHAPE encodes the
pipeline (circle = within-subject, diamond = LOSO), matching the convention in
make_fig_section3_decoding.py. Fill encodes significance of each individual
Pearson correlation (n varies per subject, so significance is not a fixed
function of |r|): solid = p<.05, hollow = not, per the project-wide "colour is
identity, fill is significance" rule. The correlation-matrix panel uses the
same convention with its own significance test (Spearman + BH-FDR, see
compute_correlation_matrix): flat gray = tested/non-significant, diverging
r-color = significant.

A first version stacked all five dimensions in one tall panel; splitting each
into its own small-multiple cell removes the need for the inter-dimension
spacing hack that version required, and lets every panel's own axis carry its
own feature labels without them fighting for space against a shared axis.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_ws_loso_sign_forest.py
"""

# =============================================================================
# Imports
# =============================================================================
import glob
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

_SKILL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "scientific_plots"
sys.path.insert(0, str(_SKILL_DIR))
import sciplot as sp  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PIPELINE_ROOT = _SCRIPTS_DIR.parent
_REPO_ROOT = _PIPELINE_ROOT.parent
sys.path.insert(0, str(_PIPELINE_ROOT))
from scripts.generate_pipeline_plots import load_all_results_from_model_dir  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
# Order matches the decodability hierarchy documented in CLAUDE.md (Section 3):
# onoff > valence > confidence > selfother > time. Grid position is (row, col)
# in a 2x3 layout; the 6th cell (2, 3) is the correlation-matrix panel.
DIMENSIONS: list[dict] = [
    {"key": "onoff", "label": "On/off", "loso_dir": "ON_vs_OFF_within_median", "ws_dir": "on_vs_off_within_median", "pos": (1, 1)},
    {"key": "valence", "label": "Valence", "loso_dir": "valence_within_median", "ws_dir": "valence_within_median", "pos": (1, 2)},
    {"key": "confidence", "label": "Confidence", "loso_dir": "confidence_within_median", "ws_dir": "confidence_within_median", "pos": (1, 3)},
    {"key": "selfother", "label": "Self/other", "loso_dir": "selfother_within_median", "ws_dir": "selfother_within_median", "pos": (2, 1)},
    {"key": "time", "label": "Time", "loso_dir": "time_within_median", "ws_dir": "time_within_median", "pos": (2, 2)},
]
CORR_POS = (2, 3)
GRID_ROWS, GRID_COLS = 2, 3

OUT_DIR = _PIPELINE_ROOT / "results/figures/ws_loso_sign_forest"
FIG_NAME = "fig_ws_loso_sign_forest"

N_TOP_FEATURES = 5            # top features per dimension, by mean|SHAP| in LOSO
MIN_NONZERO_ROWS = 6          # below this a per-subject r is not worth plotting
ALPHA = 0.05
JITTER_WIDTH = 0.17            # < ROW_STEP/2 so a row's own jitter cannot reach its neighbor
RANDOM_SEED = 0                 # jitter only; no inference depends on it

ROW_STEP = 1.0                  # vertical spacing between a panel's own feature rows
ROW_BAND_COLOR = "#EDEDED"      # alternating row background (neutral, not palette ink)

WIDTH_MM = 220.0
HEIGHT_MM = 165.0

PALETTE_PATH = _REPO_ROOT / "color_palette.yaml"
PROBE_DATA_PATH = _REPO_ROOT / "results/Behavior/probe_data/probe_level_aggregated_data.csv"

# Panel 6: Spearman correlation matrix across the five raw probe dimensions.
# The valence_sq/time_sq curvature terms this used to include were dropped
# project-wide (see CLAUDE.md "Quadratic Terms: Removed", 2026-08-13) --
# they no longer exist as analysis targets, so they don't belong in a
# descriptive matrix either. Labels match CLAUDE.md's canonical display
# labels for the five dimensions.
CORR_VARS: list[dict] = [
    {"key": "onoff", "label": "On/Off-Task"},
    {"key": "valence", "label": "Valence"},
    {"key": "selfother", "label": "Self/Other"},
    {"key": "time", "label": "Time"},
    {"key": "confidence", "label": "Confidence"},
]
CORR_ALPHA = 0.05
CORR_CELL_FONT_PT = 6.5

# =============================================================================
# Publication-style feature labels
# =============================================================================
# MARKER_LABELS originated as a verbatim copy from the now-orphaned
# Stats_andrillon/plot_cbpt_summary_figure.py (superseded by
# Stats_andrillon/plot_paper_figures.py, which is the CBPT topomap/heatmap
# generator actually wired into the pipeline — see CLAUDE.md "Paper
# Structure" Section 2). This dict is this script's own compact-label
# exception now (many marker x ROI y-tick labels need every character),
# documented in CLAUDE.md "Dimension Labels & Pole Wording (Figures)"-style:
# same underlying term as plot_paper_figures.py's MARKER_DISPLAY_NAMES (e.g.
# "PSD"), just abbreviated further (Greek symbol instead of spelled-out band).
# PE_* and evoked components (N1/P1/P3a/P3b) are not in this dict either —
# the project convention is to leave those as-is, they are already
# short/conventional.
MARKER_LABELS: dict[str, str] = {
    'kolmogorov_complexity': 'KoC',
    'per_channel_msf_psdsummary': 'MSF',
    'per_channel_sef90_psdsummary': 'SEF90',
    'per_channel_sef95_psdsummary': 'SEF95',
    'psd_bands_alpha': 'α power',
    'psd_bands_beta': 'β power',
    'psd_bands_gamma': 'γ power',
    'psd_bands_delta': 'δ power',
    'psd_bands_theta': 'θ power',
    'psd_bands_jota': 'ι power',
    'psd_bands_alpha_theta_ratio': 'α/θ ratio',
    'psd_bands_theta_beta_ratio': 'θ/β ratio',
    'psd_relative_alpha': 'PSD α',
    'psd_relative_beta': 'PSD β',
    'psd_relative_gamma': 'PSD γ',
    'psd_relative_delta': 'PSD δ',
    'psd_relative_theta': 'PSD θ',
    'psd_relative_jota': 'PSD ι',
    'psd_relative_alpha_theta_ratio': 'PSD α/θ ratio',
    'psd_relative_theta_beta_ratio': 'PSD θ/β ratio',
    'slowwaves_Density': 'SW density',
    'slowwaves_Duration': 'SW duration',
    'slowwaves_Frequency': 'SW frequency',
    'slowwaves_PTP': 'SW PTP',
    'slowwaves_Slope': 'SW slope',
    'spindles_Amplitude': 'spindle amp.',
    'spindles_Density': 'spindle density',
    'wsmi_alpha': 'wSMI α',
    'wsmi_beta': 'wSMI β',
    'wsmi_gamma': 'wSMI γ',
    'wsmi_theta': 'wSMI θ',
}
_GREEK = {"theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ"}
_REGION_LABELS = {"frontal": "Front", "central": "Cent", "posterior": "Post"}
_SIDE_LABELS = {"left": "L", "right": "R", "mid": "M"}


def _marker_label(marker: str) -> str:
    """Publication abbreviation for a marker name (falls back to the raw name)."""
    if marker in MARKER_LABELS:
        return MARKER_LABELS[marker]
    for band, symbol in _GREEK.items():
        if marker.endswith(f"_{band}"):
            return f"{marker[: -(len(band) + 1)]} {symbol}"
    return marker  # PE_* without a band suffix, evoked components (N1, P3b, ...)


def _roi_label(roi_part: str) -> str:
    """Compact region/side label, e.g. 'frontal_right_trimmean' -> 'Front R'."""
    parts = roi_part.replace("_trimmean", "").split("_")
    region = _REGION_LABELS.get(parts[0], parts[0].title())
    side = _SIDE_LABELS.get(parts[1], parts[1].title()) if len(parts) > 1 else ""
    return f"{region} {side}".strip()


def pretty_feature_label(feature: str) -> str:
    """'psd_relative_gamma_mean_frontal_right_trimmean' -> 'rel. γ · Front R'."""
    marker, roi = feature.split("_mean_", 1)
    return f"{_marker_label(marker)} · {_roi_label(roi)}"


# =============================================================================
# Palette (same adapter as make_fig_section3_decoding.py)
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
# Data assembly — SHAP sign heterogeneity (per dimension panel)
# =============================================================================


def collect_forest_data(loso_dir: Path, ws_dir: Path, n_top: int) -> dict:
    """
    Build (feature -> {loso_r, ws_points}) for a dimension's top LOSO features.

    Parameters
    ----------
    loso_dir : Path
        ``results/MW_Classification/LOSO/<dimension>/all/rf``.
    ws_dir : Path
        ``results/MW_Classification/WithinSubject/<dimension>/all/rf/true_runs/run_0``.
    n_top : int
        Number of top LOSO features (by mean|SHAP|) to keep.

    Returns
    -------
    dict
        ``{"features": [...], "loso": {feat: r}, "ws": {feat: [(r, p, n), ...]}}``,
        features ordered by descending |LOSO r|.
    """
    _, loso_shap_runs, loso_x_runs, loso_fn = load_all_results_from_model_dir(
        str(loso_dir), from_perms=False
    )
    sv, xt = loso_shap_runs[0], loso_x_runs[0]
    top_idx = np.argsort(-np.abs(sv).mean(axis=0))[:n_top]

    loso_r = {}
    for i in top_idx:
        feat = loso_fn[i]
        nz = sv[:, i] != 0
        loso_r[feat] = float(np.corrcoef(sv[nz, i], xt[nz, i])[0, 1])

    features = sorted(loso_r, key=lambda f: -abs(loso_r[f]))

    ws_pkls = sorted(glob.glob(str(ws_dir / "rf_ws_*_shap_values.pkl")))
    ws_points = {feat: [] for feat in features}
    for p in ws_pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        fn = d["feature_names"]
        for feat in features:
            if feat not in fn:
                continue
            idx = fn.index(feat)
            sv_s, xt_s = d["shap_values"][:, idx], d["x_test"][:, idx]
            nz = sv_s != 0
            if nz.sum() < MIN_NONZERO_ROWS:
                continue
            r, pval = pearsonr(sv_s[nz], xt_s[nz])
            ws_points[feat].append((float(r), float(pval), int(nz.sum())))

    return {"features": features, "loso": loso_r, "ws": ws_points}


def collect_all_dimensions() -> list[dict]:
    """Run :func:`collect_forest_data` for every entry in ``DIMENSIONS``."""
    blocks = []
    for dim in DIMENSIONS:
        loso_dir = _PIPELINE_ROOT / "results/MW_Classification/LOSO" / dim["loso_dir"] / "all/rf"
        ws_dir = (
            _PIPELINE_ROOT / "results/MW_Classification/WithinSubject" / dim["ws_dir"]
            / "all/rf/true_runs/run_0"
        )
        data = collect_forest_data(loso_dir, ws_dir, N_TOP_FEATURES)
        blocks.append({**dim, "data": data})
    return blocks


# =============================================================================
# Data assembly — probe-dimension correlation matrix (panel 6)
# =============================================================================


def load_probe_data() -> pd.DataFrame:
    """Load probe-level scores for the five canonical dimensions.

    The valence_sq / time_sq curvature columns this used to derive here were
    dropped with the quadratic construct (CLAUDE.md "Quadratic Terms:
    Removed", 2026-08-13); ``CORR_VARS`` no longer references them.
    """
    cols = ["subject", "onoff", "valence", "selfother", "time", "confidence"]
    df = pd.read_csv(PROBE_DATA_PATH, usecols=cols)
    return df.dropna(subset=cols)


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


def format_r(value: float) -> str:
    """Compact r label: no leading zero, diagonal shown as a bare "1"."""
    if abs(value - 1.0) < 1e-9:
        return "1"
    text = f"{value:.2f}"
    return "-" + text[2:] if text.startswith("-0") else text[1:]


def _axis_suffix(row: int, col: int) -> str:
    """Linear plotly axis suffix for a (row, col) cell in the GRID_ROWS x GRID_COLS grid."""
    n = (row - 1) * GRID_COLS + col
    return "" if n == 1 else str(n)


def add_correlation_panel(fig: go.Figure, df: pd.DataFrame, pal: RepoPalette, row: int, col: int) -> None:
    """Draw the lower-triangular Spearman correlation matrix in cell ``(row, col)``.

    Significance is carried by fill (project convention): flat neutral gray for
    a tested-but-non-significant pair, the diverging r-color for a significant
    one, blank for the (redundant) upper triangle. r-values are printed inside
    every lower-triangle cell in a compact form (``format_r``, e.g. ".42",
    "-.03", "1").
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
        row=row, col=col,
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
        row=row, col=col,
    )
    for i in range(n):
        for j in range(i + 1):
            is_dark = significant[i, j] and abs(corr[i, j]) >= 0.6
            fig.add_annotation(
                x=labels[j], y=labels[i], text=format_r(corr[i, j]),
                font=dict(size=sp.pt2px(CORR_CELL_FONT_PT), color="white" if is_dark else ink),
                showarrow=False, row=row, col=col,
            )
    fig.update_xaxes(tickangle=-40, showgrid=False, row=row, col=col)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=row, col=col)
    add_correlation_legend(fig, pal, row=row, col=col, n=n)


def add_correlation_legend(fig: go.Figure, pal: RepoPalette, row: int, col: int, n: int) -> None:
    """Gray/blue swatch legend in the matrix's own blank upper-right triangle.

    For a lower-triangular n x n matrix, the top 2 rows are blank from their
    3rd column onward regardless of n -- so that block is always free real
    estate; no extra figure space is spent on the legend.
    """
    grey = pal.get("covariate")
    ink = pal.neutral.get("ink", "#2B2B2B")
    sig_color = pc.sample_colorscale(pal.continuous.get("diverging", "RdBu"), [0.8])[0]

    suffix = _axis_suffix(row, col)
    xa = fig.layout[f"xaxis{suffix}"]
    ya = fig.layout[f"yaxis{suffix}"]
    x_span = xa.domain[1] - xa.domain[0]
    y_span = ya.domain[1] - ya.domain[0]
    zone_x0 = xa.domain[0] + (2 / n) * x_span
    zone_y1 = ya.domain[1]
    zone_y0 = zone_y1 - (2 / n) * y_span

    # A swatch square defined as one fraction of x_span would render as a thin
    # bar: paper x/y fractions map to very different absolute sizes once
    # WIDTH_MM != HEIGHT_MM. Sizing it in mm on each axis independently keeps
    # it visually square regardless of the panel's aspect ratio.
    swatch_mm = 2.2
    sw_x = swatch_mm / WIDTH_MM
    sw_y = swatch_mm / HEIGHT_MM
    x0 = zone_x0 + 0.08 * x_span
    for i, (color, label) in enumerate([(sig_color, "significant"), (grey, "not significant")]):
        y_center = zone_y1 - (i + 0.5) * (zone_y1 - zone_y0) / 2
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


def add_forest_panel(fig: go.Figure, block: dict, pal: RepoPalette, row: int, col: int, rng: np.random.Generator) -> None:
    """Draw one dimension's top-5 SHAP-sign forest in grid cell ``(row, col)``."""
    color = pal.get(block["key"])
    grey = pal.get("gray")
    features = block["data"]["features"]
    y_positions = np.arange(len(features), dtype=float)

    fig.add_vline(x=0, line=dict(color=grey, width=0.8, dash="dot"), layer="below", row=row, col=col)

    # Alternating row bands (zebra striping), confined to this panel's own
    # x-domain (not the whole figure, since there are five other panels).
    for i, y in enumerate(y_positions):
        if i % 2 == 1:
            fig.add_shape(
                type="rect", xref="x domain", x0=0, x1=1,
                yref="y", y0=y - ROW_STEP / 2, y1=y + ROW_STEP / 2,
                fillcolor=ROW_BAND_COLOR, opacity=0.6, line_width=0, layer="below",
                row=row, col=col,
            )

    for y, feat in zip(y_positions, features):
        points = block["data"]["ws"][feat]
        if not points:
            continue
        rs = np.array([pt[0] for pt in points])
        sig = np.array([pt[1] < ALPHA for pt in points])
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=rs.size)
        fig.add_trace(go.Scatter(
            x=rs, y=y + jitter,
            mode="markers",
            marker=dict(
                symbol="circle", size=5,
                color=[color if s else "white" for s in sig],
                opacity=0.55,
                line=dict(color=color, width=0.8),
            ),
            hovertemplate=f"{block['label']} / {feat}<br>r=%{{x:.3f}}<extra></extra>",
            showlegend=False,
        ), row=row, col=col)

    fig.add_trace(go.Scatter(
        x=[block["data"]["loso"][f] for f in features], y=y_positions,
        mode="markers",
        marker=dict(symbol="diamond", size=10, color=color, line=dict(color=color, width=1.4)),
        hovertemplate="LOSO r=%{x:.3f}<extra></extra>",
        showlegend=False,
    ), row=row, col=col)

    ticktext = [f"<span style='color:{color}'>{pretty_feature_label(f)}</span>" for f in features]
    fig.update_yaxes(
        tickmode="array", tickvals=y_positions, ticktext=ticktext,
        range=[-0.6, len(features) - 1 + 0.6], autorange="reversed",
        row=row, col=col,
    )
    fig.update_xaxes(
        title_text="Pearson r", range=[-1.05, 1.05], dtick=0.5,
        row=row, col=col,
    )


GRID_TOP_FRACTION = 0.88  # rescale every panel's y-domain into the bottom 88%,
                          # reserving a fixed top strip for titles + legend


def _reserve_top_strip(fig: go.Figure, top_fraction: float) -> None:
    """Compress every subplot's y-domain into ``[0, top_fraction]``.

    Passing ``subplot_titles`` to ``make_subplots`` only reserves a thin sliver
    right above row 1 -- not enough room for a legend as well, and increasing
    ``margin.t`` afterward does not help: domains are fractions of the *plot
    area*, so growing the margin shrinks that area and the title/legend gap
    shrinks right along with it. Explicitly rescaling the y-domains is the only
    way to carve out real, fixed space at the top before anything is placed in
    it.
    """
    for key in fig.layout:
        if re.fullmatch(r"yaxis\d*", key):
            d0, d1 = fig.layout[key].domain
            fig.layout[key].domain = (d0 * top_fraction, d1 * top_fraction)


def centered_panel_titles(fig: go.Figure, labels: list[str], ink: str) -> None:
    """Bold label centered above each of the 6 grid cells, in row-major order."""
    for i, label in enumerate(labels):
        if not label:
            continue
        suffix = "" if i == 0 else str(i + 1)
        xa = fig.layout[f"xaxis{suffix}"]
        ya = fig.layout[f"yaxis{suffix}"]
        fig.add_annotation(
            x=(xa.domain[0] + xa.domain[1]) / 2, y=ya.domain[1],
            xref="paper", yref="paper",
            text=f"<b>{label}</b>", font=dict(size=sp.pt2px(sp.TYPE_PT["panel_label"]), color=ink),
            xanchor="center", yanchor="bottom", showarrow=False,
        )


def build_figure(blocks: list[dict], probe_df: pd.DataFrame, pal: RepoPalette) -> go.Figure:
    """Assemble the 2x3 grid: 5 dimension forest panels + 1 correlation matrix."""
    rng = np.random.default_rng(RANDOM_SEED)
    grey = pal.get("gray")
    ink = pal.neutral.get("ink", "#2B2B2B")

    fig = make_subplots(rows=GRID_ROWS, cols=GRID_COLS, horizontal_spacing=0.09, vertical_spacing=0.20)
    _reserve_top_strip(fig, GRID_TOP_FRACTION)

    for block in blocks:
        row, col = block["pos"]
        add_forest_panel(fig, block, pal, row=row, col=col, rng=rng)

    add_correlation_panel(fig, probe_df, pal, row=CORR_POS[0], col=CORR_POS[1])

    # Legend proxies in neutral ink (colour is already spoken for by dimension,
    # carried by the marker fill and the y-tick label). Added once, at figure
    # level, since the legend is shared across all panels.
    for symbol, size, name in (("circle", 7, "within-subject (per subject)"),
                                ("diamond", 9, "LOSO (group)")):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=grey, line=dict(color=grey, width=1.6)),
            name=name, showlegend=True, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=7, color=grey, line=dict(color=grey, width=1.6)),
        name=f"p<{ALPHA} (solid) / not (hollow)", showlegend=True, hoverinfo="skip",
    ))

    sp.grid(fig, "x")
    sp.legend_above(fig)
    labels = [""] * (GRID_ROWS * GRID_COLS)
    for block in blocks:
        row, col = block["pos"]
        labels[(row - 1) * GRID_COLS + (col - 1)] = block["label"]
    labels[(CORR_POS[0] - 1) * GRID_COLS + (CORR_POS[1] - 1)] = "Correlation"
    centered_panel_titles(fig, labels, ink)
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    pal = init_palette()
    blocks = collect_all_dimensions()
    probe_df = load_probe_data()

    print(f"{'dimension':<12}{'feature':<50}{'LOSO r':>9}{'n_ws_subj':>11}{'frac_sign_flip':>16}")
    for block in blocks:
        data = block["data"]
        for feat in data["features"]:
            rs = np.array([pt[0] for pt in data["ws"][feat]])
            flip_frac = np.mean(np.sign(rs) != np.sign(data["loso"][feat])) if rs.size else float("nan")
            print(f"{block['label']:<12}{feat:<50}{data['loso'][feat]:>9.3f}{rs.size:>11d}{flip_frac:>16.2f}")

    fig = build_figure(blocks, probe_df, pal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp.save(fig, str(OUT_DIR), FIG_NAME, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    print(f"\nWrote {OUT_DIR / FIG_NAME}.svg / .png")


if __name__ == "__main__":
    main()
