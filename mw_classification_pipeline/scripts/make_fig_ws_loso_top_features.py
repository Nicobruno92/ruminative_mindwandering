#!/usr/bin/env python
"""
WS-vs-LOSO top-feature comparison — two panels, `onoff` only.

LOSO is one shared model, so its "top features" are unambiguous:
mean(|SHAP|) pooled over the 1216 trials (panel A). WS has no equivalent
single ranking — each of the 29 subjects independently fits its own model and
selects its own feature subset, so pooling/averaging across subjects (as
panel A's method would) washes out exactly the idiosyncratic effects this
analysis exists to show (see fig_ws_loso_sign_forest.py). Panel B instead
ranks by *how many subjects independently consider a feature one of their
own personal top-K* (K=10, by that subject's own mean(|SHAP|) over their own
non-zero rows) — a vote count across per-subject rankings, immune to
cross-subject scale/coverage differences.

Encoding notes
--------------
Two panels, one bar chart each: panel A's x-axis is mean(|SHAP|) magnitude
(LOSO group model); panel B's is a subject count out of 29 (WS vote count).
Different units by design — this is not one shared ranking split into two
views, it is two different questions ("how important, pooled" vs "how many
individuals agree"), which is the entire point of showing them side by side:
only one feature (PE_alpha_mean_central_right_trimmean) appears in both
top-10s.

Colour is the `onoff` dimension colour throughout (single-dimension figure).
No error bars: panel A is a point estimate from one run (run_0, deterministic
given the fixed seed sequence), panel B is a count, not a distribution.

Forking-path note: panel B's ranking is sensitive to K (the personal
top-K length). Spot-checked at K=5/10/15/20 — the top-5 shift partially but
not completely across that range; there is no single "correct" K. K=10 was
chosen and is not re-tuned to make the figure look a particular way.

USAGE
-----
    conda activate plots
    python mw_classification_pipeline/scripts/make_fig_ws_loso_top_features.py
"""

# =============================================================================
# Imports
# =============================================================================
import glob
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from plotly.subplots import make_subplots

_SKILL_DIR = Path(__file__).resolve().parents[2] / "vendor" / "scientific_plots"
sys.path.insert(0, str(_SKILL_DIR))
import sciplot as sp  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PIPELINE_ROOT = _SCRIPTS_DIR.parent
sys.path.insert(0, str(_PIPELINE_ROOT))
from scripts.generate_pipeline_plots import load_all_results_from_model_dir  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================
DIMENSION_KEY = "onoff"
LOSO_RESULTS_DIR = _PIPELINE_ROOT / "results/MW_Classification/LOSO/ON_vs_OFF_within_median/all/rf"
WS_RUN0_DIR = (
    _PIPELINE_ROOT
    / "results/MW_Classification/WithinSubject/on_vs_off_within_median/all/rf/true_runs/run_0"
)
OUT_DIR = _PIPELINE_ROOT / "results/figures/ws_loso_top_features"
FIG_NAME = "fig_ws_loso_top_features"

N_TOP = 10
PERSONAL_TOP_K = 10   # each WS subject's own personal top-K length (see forking-path note)

WIDTH_MM = 140.0
HEIGHT_MM = 170.0

PALETTE_PATH = _PIPELINE_ROOT.parent / "color_palette.yaml"

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
# Data assembly
# =============================================================================


def collect_loso_top() -> tuple[list[str], list[float]]:
    """LOSO top-N features by mean(|SHAP|) pooled over run_0's 1216 trials."""
    _, shap_runs, _, fn = load_all_results_from_model_dir(str(LOSO_RESULTS_DIR), from_perms=False)
    sv = shap_runs[0]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)[:N_TOP]
    return [fn[i] for i in order], [float(mean_abs[i]) for i in order]


def collect_ws_top() -> tuple[list[str], list[int], int]:
    """
    WS top-N features by vote count across subjects' own personal top-K.

    Each subject's personal ranking uses only their own non-zero rows (a
    feature their fold's mRMR did not select is a structural zero, not a
    measured null effect) — see compute_shap_values_for_pipeline in
    utils/ml_utils.py.
    """
    pkls = sorted(glob.glob(str(WS_RUN0_DIR / "rf_ws_*_shap_values.pkl")))
    vote_counter: Counter = Counter()
    n_subjects = 0
    for p in pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        fn = np.array(d["feature_names"])
        sv = d["shap_values"]
        nz_cols = np.where((sv != 0).any(axis=0))[0]
        per_feat_mean_abs = np.array([np.abs(sv[sv[:, c] != 0, c]).mean() for c in nz_cols])
        order = np.argsort(-per_feat_mean_abs)[:PERSONAL_TOP_K]
        vote_counter.update(fn[nz_cols][order])
        n_subjects += 1

    top = vote_counter.most_common(N_TOP)
    return [f for f, _ in top], [c for _, c in top], n_subjects


# =============================================================================
# Figure
# =============================================================================


def build_figure(
    loso_features: list[str], loso_vals: list[float],
    ws_features: list[str], ws_counts: list[int], n_ws_subjects: int,
    pal: "RepoPalette",
) -> go.Figure:
    """Assemble the two-panel top-feature comparison."""
    color = pal.get(DIMENSION_KEY)

    # Stacked vertically, not side by side: both panels' feature-name labels
    # are long (~45 chars) and a two-column layout put panel B's automargin'd
    # labels on top of panel A's bars regardless of horizontal_spacing — the
    # label reserves space beyond what a column split can give it. Full width
    # per panel removes the competition entirely.
    # No subplot_titles: the "what is panel A vs B" description belongs in the
    # caption (house style — no prose in the plot); sp.panel_labels adds the
    # bare A/B letters instead.
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14)

    # Panel A — LOSO, ranked descending top-to-bottom (reverse for horizontal bars).
    fig.add_trace(go.Bar(
        x=loso_vals[::-1], y=loso_features[::-1], orientation="h",
        marker_color=color, showlegend=False,
    ), row=1, col=1)
    fig.update_xaxes(title_text="mean(|SHAP|)", row=1, col=1)

    # Panel B — WS, vote count.
    fig.add_trace(go.Bar(
        x=ws_counts[::-1], y=ws_features[::-1], orientation="h",
        marker_color=color, showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text=f"# subjects (of {n_ws_subjects}) in own top-10", row=2, col=1, dtick=1)

    sp.grid(fig, "x")
    sp.panel_labels(fig)
    for r in (1, 2):
        fig.update_yaxes(automargin=True, row=r, col=1)
    return fig


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    pal = init_palette()
    loso_features, loso_vals = collect_loso_top()
    ws_features, ws_counts, n_ws_subjects = collect_ws_top()

    print(f"LOSO top-{N_TOP} (pooled mean|SHAP|):")
    for f, v in zip(loso_features, loso_vals):
        print(f"  {f:<50}{v:.5f}")

    print(f"\nWS top-{N_TOP} (votes in personal top-{PERSONAL_TOP_K}, n={n_ws_subjects} subjects):")
    for f, c in zip(ws_features, ws_counts):
        print(f"  {f:<50}{c}/{n_ws_subjects}")

    shared = set(loso_features) & set(ws_features)
    print(f"\nShared between both top-{N_TOP} lists: {sorted(shared) or 'none'}")

    fig = build_figure(loso_features, loso_vals, ws_features, ws_counts, n_ws_subjects, pal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp.save(fig, str(OUT_DIR), FIG_NAME, width_mm=WIDTH_MM, height_mm=HEIGHT_MM)
    print(f"\nWrote {OUT_DIR / FIG_NAME}.svg / .png")


if __name__ == "__main__":
    main()
