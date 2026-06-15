"""
LMM Analysis: Probe Dimensions → Objective Behavioral Markers

TWO ANALYSES:

1. ADDITIVE MODEL (one per marker):
    marker ~ onoff + valence + selfother + time + confidence + time_on_task
             + (1|subject)
   Tests the independent contribution of each probe dimension.

2. MODERATION ANALYSIS (one per marker × moderator):
    marker ~ onoff * moderator + (1|subject)
   For each non-onoff probe dimension, tests whether it moderates the effect
   of onoff on the objective marker. The interaction term onoff:moderator
   is the key test. FDR-BH corrected across all marker × moderator tests.
   Visualised as an interaction plot: onoff effect at Low vs High moderator
   (split at within-subject median).

Note: This analysis intentionally does not discriminate between Controls and
Risk of Depression (RoD) groups. For the purposes of this pipeline, all subjects
are treated as a single group in the models and visualizations.

Preprocessing mirrors objective_markers_analysis.py:
  - Merge objective markers with probe-level metadata (subject/task/probe)
  - Derive time_on_task = probe_number + 15 * (sart_number - 1)
  - Within-subject z-scoring of objective markers (APPLY_WITHIN_SUBJECT_Z)

Outputs (saved under OUTPUT_BASE_DIR/<dataset>/):
  present/<marker>_lmm_results.csv       — additive model results + FDR
  present/<marker>_combined_forest.png    — additive + moderation forest plot
  present/<marker>_scatter_grid.{html,png,pdf} — scatter grid per predictor
  present/<marker>_combined_panel.{html,png,pdf} — scatter grid + forest, one figure
  present/moderation_summary.csv          — interaction term stats, FDR-corrected
  present/moderation_forest_<marker>.png  — forest of interaction coefficients
  present/interaction_<marker>_<mod>.png  — interaction plot per significant pair
  present_vs_time_sq_comparison.csv       — cross-model comparison (dataset root)

SQUARE-EFFECTS SENSITIVITY MODELS (time_sq, valence_sq):
  `time` and `valence` are the two probe dimensions hypothesised to have
  quadratic (curvature) relationships with the markers. Each gets its own
  additive model with a centred-squared term added/swapped in (see
  compute_time_squared / compute_valence_squared), with the same full plot
  suite as present/ — scatter grid, additive-effects forest, onoff-moderation
  forest, and combined panel — saved under
  OUTPUT_BASE_DIR/<dataset>/sensitivity_square_effects/{time_sq,valence_sq}/.

  - time_sq: `present = 50 - |time-50|` and `time_sq = (time-50)^2/50` are
    both functions of |time-50| and become severely collinear once `time` is
    in the model, so `time_sq` SWAPS for `present` in a separate model
    (SENSITIVITY_PREDICTORS_TIME_SQ / FORMULA_TEMPLATE_TIME_SQ), with its
    moderators mirroring the same swap (SENSITIVITY_MODERATORS_TIME_SQ).
  - valence_sq: `valence_sq = (valence-50)^2/50` has no `present`-like
    collinear sibling, so it is ADDED to the primary predictor/moderator sets
    (SENSITIVITY_PREDICTORS_VALENCE_SQ / FORMULA_TEMPLATE_VALENCE_SQ /
    SENSITIVITY_MODERATORS_VALENCE_SQ).

  PREDICTOR_SETS ties each named predictor set (predictors, formula, output
  subdir, moderators) together; `plots_only(predictor_set_names=...)` /
  `--predictor-set` lets any set's plots be regenerated independently.

Author: Nicolas Bruno
"""

import os
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input data paths (same sources as objective_markers_analysis.py)
OBJECTIVE_MARKERS_PATHS: dict[str, str] = {
    "comunes": "results/Behavior/objective_markers/objective_markers_per_probe.csv",
    "n10": "results/Behavior/objective_markers/objective_markers_per_probe_n10.csv",
}
PROBE_DATA_PATH = (
    "results/Behavior/probe_data/probe_level_aggregated_data.csv"
)

# Output directory
OUTPUT_BASE_DIR = Path(
    "results/Behavior/objective_markers/lmm_probe_dimensions"
)

# Objective markers to model (dependent variables)
# total_errors = omission_rate + commission_rate (computed during data loading)
OBJECTIVE_MARKERS: list[str] = [
    "omission_rate",
    "commission_rate",
    "total_errors",
    "rtcv",
]

MARKER_LABELS: dict[str, str] = {
    "omission_rate": "Omission Rate",
    "commission_rate": "Commission Rate",
    "total_errors": "Total Errors",
    "rtcv": "RTCV (RT Variability)",
}

# Probe-level predictors (fixed effects, continuous, 0-100 scale)
PREDICTORS: list[str] = ["onoff", "valence", "selfother", "time", "present", "confidence", "time_on_task"]

PREDICTOR_LABELS: dict[str, str] = {
    "onoff": "On/Off Task",
    "valence": "Valence",
    "selfother": "Self/Other",
    "time": "Time (probe dim.)",
    "present": "Present",
    "confidence": "Confidence",
    "time_on_task": "Time on Task",
    "time_sq": "Time² (curvature)",        # only used by the time_sq sensitivity model below
    "valence_sq": "Valence² (curvature)",  # only used by the valence_sq sensitivity model below
}

# LMM formula template — marker name is substituted at runtime
FORMULA_TEMPLATE = (
    "{marker} ~ onoff + valence + selfother + time + present + confidence"
    " + time_on_task + (1|subject)"
)

# --- Square-effects sensitivity models: time_sq and valence_sq --------------
# `present = 50 - |time-50|` and `time_sq = (time-50)^2/50` (see
# compute_present_dimension / compute_time_squared) are both functions of
# |time-50| and become severely collinear (r=-0.97, VIF~16) once `time` is in
# the model, masking each other's effects on omission_rate. Rather than
# combine them into one model, fit this as a SEPARATE model that swaps
# `present` for `time_sq`, so the present/not-present framing of the primary
# model can be compared against a standard quadratic-in-time parametrization.
SENSITIVITY_PREDICTORS_TIME_SQ: list[str] = [
    "onoff", "valence", "selfother", "time", "time_sq", "confidence", "time_on_task",
]
FORMULA_TEMPLATE_TIME_SQ = (
    "{marker} ~ onoff + valence + selfother + time + time_sq + confidence"
    " + time_on_task + (1|subject)"
)
# Moderators for the time_sq model: same present -> time_sq swap as the
# predictors above, mirroring MODERATORS below.
SENSITIVITY_MODERATORS_TIME_SQ: list[str] = [
    "valence", "selfother", "time", "time_sq", "confidence", "time_on_task",
]

# Potential moderators of the onoff effect (all probe dimensions except onoff)
# Each will be tested in: marker ~ onoff * moderator + (1|subject)
MODERATORS: list[str] = ["valence", "selfother", "time", "present", "confidence", "time_on_task"]

# `valence_sq = (valence-50)^2/50` (see compute_valence_squared) tests whether
# `valence` has a curvature (U-shaped) relationship with the markers, in
# addition to its linear effect — the other probe dimension (besides `time`)
# hypothesised to have quadratic effects. Unlike time_sq, valence has no
# `present`-like collinear sibling, so valence_sq is ADDED to the primary
# predictor/moderator sets rather than swapped in.
SENSITIVITY_PREDICTORS_VALENCE_SQ: list[str] = PREDICTORS + ["valence_sq"]
FORMULA_TEMPLATE_VALENCE_SQ = (
    "{marker} ~ onoff + valence + selfother + time + present + confidence"
    " + time_on_task + valence_sq + (1|subject)"
)
SENSITIVITY_MODERATORS_VALENCE_SQ: list[str] = MODERATORS + ["valence_sq"]

# Optimisation settings (mirrors Statistics/config.yaml)
LMM_METHOD: str = "powell"  # gradient-free, robust
LMM_MAXITER: int = 500
RANDOM_STATE: int = 42

# Multiple-comparisons correction method
# For the additive model: applied within each model across predictors
# For the moderation model: applied across all marker × moderator interaction tests
MCC_METHOD: str = "fdr_bh"  # Benjamini-Hochberg FDR
MCC_ALPHA: float = 0.05

# Apply within-subject z-scoring to objective markers before modelling.
# Mirrors the APPLY_WITHIN_SUBJECT_Z flag in objective_markers_analysis.py.
APPLY_WITHIN_SUBJECT_Z: bool = False

# Figure settings
FIG_DPI: int = 300

# Shared project color palette (single source of truth). Resolved path-relative
# to this script so it works regardless of the current working directory.
PALETTE = yaml.safe_load(
    open(Path(__file__).resolve().parents[2] / "color_palette.yaml")
)

PALETTE_SIGNIFICANT = "#E74C3C"   # red — significant after FDR
PALETTE_NOT_SIGNIFICANT = PALETTE["neutral"]["covariate"]  # grey — not significant
# Interaction plot colours: Low/High moderator
PALETTE_MOD_LOW = "#2E86AB"   # blue — low moderator
PALETTE_MOD_HIGH = "#F24236"  # red — high moderator

# Per-dimension colours — sourced from the shared palette so probe dimensions
# are colour-coded consistently across the behavioural and EEG/classification
# figures of the paper. `time_on_task` is a covariate, not a probe dimension
# → neutral grey.
PREDICTOR_COLORS: dict[str, str] = {
    "onoff":        PALETTE["dimensions"]["onoff"],       # red    — On/Off-Task
    "valence":      PALETTE["dimensions"]["valence"],     # blue   — Valence
    "selfother":    PALETTE["dimensions"]["selfother"],   # green  — Self/Other
    "confidence":   PALETTE["dimensions"]["confidence"],  # orange — Confidence
    "time":         PALETTE["dimensions"]["time"],        # purple — Time (probe dim.)
    "present":      PALETTE["dimensions"]["present"],     # brown  — Present
    "time_sq":      PALETTE["dimensions"]["time"],        # purple — Time² (same dimension as `time`)
    "valence_sq":   PALETTE["dimensions"]["valence"],     # blue   — Valence² (same dimension as `valence`)
    "time_on_task": PALETTE["neutral"]["covariate"],      # grey   — covariate
}
DEFAULT_PREDICTOR_COLOR: str = PALETTE["neutral"]["covariate"]

# --- Predictor sets ----------------------------------------------------------
# Named, swappable predictor sets for the full plot suite (scatter grid +
# additive forest + onoff-moderation forest + combined panel). "present" is
# the primary model used in the paper; "time_sq" and "valence_sq" are the
# square-effects sensitivity models (see SENSITIVITY_PREDICTORS_TIME_SQ /
# SENSITIVITY_PREDICTORS_VALENCE_SQ above) — the two probe dimensions
# hypothesised to have quadratic relationships with the markers. All three are
# fit in run_pipeline_for_dataset and can be independently regenerated via
# `plots_only(predictor_set_names=...)` / `--predictor-set` on the CLI. Each
# set gets its own output subdirectory: <dataset>/present/,
# <dataset>/sensitivity_square_effects/time_sq/ and
# <dataset>/sensitivity_square_effects/valence_sq/. `moderators` is the list
# passed to `run_moderation_analysis` for that set's onoff-moderation forest
# panel.
PREDICTOR_SETS: dict[str, dict] = {
    "present": {
        "predictors": PREDICTORS,
        "formula_template": FORMULA_TEMPLATE,
        "output_subdir": "present",
        "moderators": MODERATORS,
        "with_moderation": True,
    },
    "time_sq": {
        "predictors": SENSITIVITY_PREDICTORS_TIME_SQ,
        "formula_template": FORMULA_TEMPLATE_TIME_SQ,
        "output_subdir": "sensitivity_square_effects/time_sq",
        "moderators": SENSITIVITY_MODERATORS_TIME_SQ,
        "with_moderation": True,
    },
    "valence_sq": {
        "predictors": SENSITIVITY_PREDICTORS_VALENCE_SQ,
        "formula_template": FORMULA_TEMPLATE_VALENCE_SQ,
        "output_subdir": "sensitivity_square_effects/valence_sq",
        "moderators": SENSITIVITY_MODERATORS_VALENCE_SQ,
        "with_moderation": True,
    },
}

# Binned within-subject scatter settings.
# Markers are z-scored within subject (APPLY_WITHIN_SUBJECT_Z), so a plain
# per-subject mean collapses the Y axis to ~0. Instead we bin the predictor
# into fixed-width bins *within each subject*, take the per-subject mean marker
# per bin, then plot the mean ± SE across subjects per bin. Fixed-width bins
# (not quantiles) keep the points evenly spaced even when the predictor is
# bimodal (e.g. onoff). This preserves the within-subject relationship that the
# mixed model estimates.
SCATTER_BIN_WIDTH: float = 10.0

# =============================================================================
# HELPERS
# =============================================================================


def normalize_task_label(raw_task: str) -> str:
    """Standardise task label to 'Sart1'–'Sart4' format.

    Parameters
    ----------
    raw_task : str
        Raw task string from CSV.

    Returns
    -------
    str
        Normalised task label.
    """
    t = str(raw_task).strip().lower().replace(" ", "").replace("-", "")
    if "sart" in t:
        for digit in ("1", "2", "3", "4"):
            if digit in t:
                return f"Sart{digit}"
    return raw_task


def add_time_on_task(df: pd.DataFrame) -> pd.DataFrame:
    """Add `time_on_task` column (global probe index across all SARTs).

    Mirrors the computation in objective_markers_analysis.py:
        time_on_task = probe_number + 15 * (sart_number - 1)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a 'task' column ('Sart1'–'Sart4') and
        'probe_number'.

    Returns
    -------
    pd.DataFrame
        Input dataframe with `sart_number` and `time_on_task` added in-place.
    """
    df["sart_number"] = df["task"].str.extract(r"(\d+)").astype(int)
    df["time_on_task"] = df["probe_number"] + 15 * (df["sart_number"] - 1)
    return df


def compute_present_dimension(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the `present` probe dimension from `time`.

    The time dimension runs 0 (past) → 50 (present) → 100 (future), so the
    distance from the present |time - 50| ranges 0-50. `present = 50 - |time
    - 50|` gives 50 when time=50 (fully present-focused) and 0 when time=0 or
    time=100 (fully past/future-focused).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a `time` column.

    Returns
    -------
    pd.DataFrame
        Input dataframe with `present` column added (0-50 scale).
    """
    df["present"] = 50 - (df["time"] - 50).abs()
    return df


def compute_time_squared(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a quadratic `time_sq` term from `time`.

    `present = 50 - |time - 50|` folds past and future together, which
    averages out an asymmetric U-shape (e.g. errors are higher for deep-past
    than deep-future probes). Adding `time_sq = (time - 50)^2 / 50` alongside
    the linear `time` term lets the additive model fit a parabola whose
    vertex can fall anywhere in [0, 100] (vertex at `time = 50 -
    beta_time / (2 * beta_time_sq)`), capturing that asymmetry while `present`
    retains its present/not-present interpretation. Centring on 50 before
    squaring reduces collinearity with the linear `time` term. The `/ 50`
    scaling puts `time_sq` on the same 0-50 range as `present`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a `time` column.

    Returns
    -------
    pd.DataFrame
        Input dataframe with `time_sq` column added (0-50 scale).
    """
    df["time_sq"] = (df["time"] - 50) ** 2 / 50
    return df


def compute_valence_squared(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a quadratic `valence_sq` term from `valence`.

    `valence_sq = (valence - 50)^2 / 50` tests whether `valence` has a
    curvature (U-shaped) relationship with the markers, in addition to its
    linear effect — `valence` is the other probe dimension (besides `time`)
    hypothesised to have quadratic effects. Centring on 50 before squaring
    follows the same "neutral midpoint" convention as `time_sq`
    (compute_time_squared), even though `valence` has no `present`-like
    collinear sibling to motivate it on collinearity grounds alone — kept for
    consistency/interpretability across the two square-effects models. The
    `/ 50` scaling puts `valence_sq` on the same 0-50 range as `time_sq`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing a `valence` column.

    Returns
    -------
    pd.DataFrame
        Input dataframe with `valence_sq` column added (0-50 scale).
    """
    df["valence_sq"] = (df["valence"] - 50) ** 2 / 50
    return df


def apply_within_subject_z_scoring(
    df: pd.DataFrame, markers: list[str]
) -> pd.DataFrame:
    """Z-score each objective marker within subject.

    Mirrors `apply_within_subject_z_scoring` in objective_markers_analysis.py:
    each subject's marker values are centred (mean=0, std=1) across all their
    available probes.

    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with objective markers and subject column.
    markers : list[str]
        Marker column names to z-score.

    Returns
    -------
    pd.DataFrame
        Copy of df with z-scored marker columns.
    """
    df_z = df.copy()
    for marker in markers:
        if marker not in df_z.columns:
            continue
        df_z[marker] = df_z.groupby("subject")[marker].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
            if x.std(ddof=0) not in (0, np.nan)
            else x
        )
    return df_z


def parse_lmm_formula(formula_template: str, marker: str) -> str:
    """Substitute marker name and remove random-effects for statsmodels.

    statsmodels `mixedlm` takes the fixed-effects formula separately from the
    grouping variable, so this strips the `(1|subject)` part.

    Parameters
    ----------
    formula_template : str
        Template with `{marker}` placeholder and `(1|subject)` term.
    marker : str
        Dependent variable name.

    Returns
    -------
    str
        Fixed-effects-only formula suitable for `smf.mixedlm`.
    """
    full = formula_template.format(marker=marker)
    # Remove random-effects term (1|subject)
    import re
    fixed = re.sub(r"\s*\+?\s*\([^)]*\|[^)]*\)", "", full).strip()
    return fixed


# =============================================================================
# LMM FITTING
# =============================================================================


def fit_lmm(
    data: pd.DataFrame,
    marker: str,
    formula_template: str,
    method: str,
    maxiter: int,
    predictors: list[str] | None = None,
) -> pd.DataFrame:
    """Fit a single LMM and return a tidy predictor-level results dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        Merged probe-level dataframe with marker and predictor columns.
    marker : str
        Name of the dependent variable.
    formula_template : str
        Formula template with `{marker}` placeholder.
    method : str
        Optimisation method passed to `model.fit()`.
    maxiter : int
        Maximum number of iterations.
    predictors : list[str] | None
        Predictor names used to select model columns and FDR-correct across.
        Defaults to the module-level `PREDICTORS` (primary model). Pass
        `SENSITIVITY_PREDICTORS_TIME_SQ` / `SENSITIVITY_PREDICTORS_VALENCE_SQ`
        for the square-effects sensitivity models.

    Returns
    -------
    pd.DataFrame
        One row per fixed effect with columns:
        predictor, estimate, std_error, t_value, p_value, conf_lower,
        conf_upper, p_fdr, significant_fdr.
    """
    if predictors is None:
        predictors = PREDICTORS

    fixed_formula = parse_lmm_formula(formula_template, marker)

    # Columns needed for this model
    needed_cols = ["subject", marker] + predictors
    present_cols = [c for c in needed_cols if c in data.columns]
    model_data = (
        data[present_cols]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )
    model_data["subject"] = model_data["subject"].astype(str)

    print(f"\n{'='*60}")
    print(f"MODEL: {marker}")
    print(f"Formula: {fixed_formula} + (1|subject)")
    print(f"N observations: {len(model_data)}  |  N subjects: {model_data['subject'].nunique()}")
    print("="*60)

    model = smf.mixedlm(fixed_formula, model_data, groups="subject")
    result = model.fit(method=method.upper(), maxiter=maxiter, reml=True, disp=False)
    print(result.summary())

    # Build tidy results table
    conf_int = result.conf_int()
    results_df = pd.DataFrame({
        "predictor": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "t_value": result.tvalues.values,
        "p_value": result.pvalues.values,
        "conf_lower": conf_int.iloc[:, 0].values,
        "conf_upper": conf_int.iloc[:, 1].values,
    })

    # Remove intercept and random effect rows for FDR (keep only predictors)
    predictor_mask = results_df["predictor"].isin(predictors)
    predictor_rows = results_df[predictor_mask].copy()

    # FDR-BH correction across predictors within this model
    _, p_fdr, _, _ = multipletests(predictor_rows["p_value"].values, method=MCC_METHOD)
    predictor_rows["p_fdr"] = p_fdr
    predictor_rows["significant_fdr"] = predictor_rows["p_fdr"] < MCC_ALPHA

    # Merge FDR back into full results_df
    results_df = results_df.merge(
        predictor_rows[["predictor", "p_fdr", "significant_fdr"]],
        on="predictor",
        how="left",
    )

    return results_df


# =============================================================================
# PLOTTING
# =============================================================================


def plot_combined_forest(
    additive_df: pd.DataFrame,
    moderation_df: pd.DataFrame | None,
    marker: str,
    output_dir: Path,
    figsize_override: tuple[float, float] | None = None,
    out_suffix: str = "",
    predictors: list[str] | None = None,
) -> Path:
    """Forest plot of additive effects, optionally paired with moderation terms.

    Left panel shows fixed-effect coefficients from the additive model given
    by ``predictors``. If ``moderation_df`` is provided, a right panel shows
    the `onoff:moderator` interaction coefficient for each moderator in that
    predictor set (see `PREDICTOR_SETS[<set>]["moderators"]` — every set,
    including the square-effects sensitivity models, now has a moderation
    panel). Marker **colour encodes the probe dimension** (shared with the MW
    classification figure, see ``PREDICTOR_COLORS``); statistical
    significance after FDR is encoded by marker fill and CI line style:
    filled marker + solid line = significant, hollow marker + dashed line =
    not significant.

    Parameters
    ----------
    additive_df : pd.DataFrame
        Output of `fit_lmm` for this marker (additive model results).
    moderation_df : pd.DataFrame | None
        Output of `run_moderation_analysis`, full table (all markers), with
        `p_fdr` and `significant_fdr` columns. Pass `None` to draw an
        additive-effects-only (single-panel) forest.
    marker : str
        Marker name (used in figure title and filename).
    output_dir : Path
        Output directory.
    predictors : list[str] | None
        Predictor set shown in the additive panel. Defaults to the
        module-level `PREDICTORS`. Pass `SENSITIVITY_PREDICTORS_TIME_SQ` /
        `SENSITIVITY_PREDICTORS_VALENCE_SQ` for the square-effects sensitivity
        models.
    """
    from matplotlib.lines import Line2D

    if predictors is None:
        predictors = PREDICTORS

    # --- Prepare additive panel data ---
    add_plot = additive_df[additive_df["predictor"].isin(predictors)].copy()
    add_plot = add_plot.sort_values("estimate", ascending=True).reset_index(drop=True)
    add_labels = [PREDICTOR_LABELS.get(p, p) for p in add_plot["predictor"]]
    add_colors = [PREDICTOR_COLORS.get(p, DEFAULT_PREDICTOR_COLOR)
                  for p in add_plot["predictor"]]
    add_sig = [bool(s) for s in add_plot["significant_fdr"].tolist()]

    # --- Prepare moderation panel data (optional) ---
    if moderation_df is not None:
        mod_plot = moderation_df[
            moderation_df["marker"] == marker
        ].copy().sort_values("estimate", ascending=True).reset_index(drop=True)
        mod_plot["ci_lower"] = mod_plot["estimate"] - 1.96 * mod_plot["std_error"]
        mod_plot["ci_upper"] = mod_plot["estimate"] + 1.96 * mod_plot["std_error"]
        mod_labels = [PREDICTOR_LABELS.get(m, m) for m in mod_plot["moderator"]]
        mod_colors = [PREDICTOR_COLORS.get(m, DEFAULT_PREDICTOR_COLOR)
                      for m in mod_plot["moderator"]]
        mod_sig = [bool(s) for s in mod_plot["significant_fdr"].tolist()]
    else:
        mod_plot = None

    # Dashed style used for non-significant CI lines
    dash_style = (0, (4, 2))

    # --- Build figure (one or two panels, shared colour legend) ---
    n_labels_for_size = [len(add_labels)]
    if mod_plot is not None:
        n_labels_for_size.append(len(mod_labels))
    default_figsize = (
        16 if mod_plot is not None else 8,
        max(5, max(n_labels_for_size) * 0.65 + 2.5),
    )
    figsize = figsize_override if figsize_override is not None else default_figsize
    if mod_plot is not None:
        fig, (ax_add, ax_mod) = plt.subplots(
            1, 2, figsize=figsize,
            constrained_layout=True,
        )
    else:
        fig, ax_add = plt.subplots(
            1, 1, figsize=figsize,
            constrained_layout=True,
        )

    # Helper to draw one panel
    def _draw_panel(ax, plot_df, labels, colors, sig_flags, ci_lower_col,
                    ci_upper_col, xlabel, title):
        for i, (_, row) in enumerate(plot_df.iterrows()):
            c = colors[i]
            sig = sig_flags[i]
            lo, hi, est = row[ci_lower_col], row[ci_upper_col], row["estimate"]
            # CI whisker: solid (significant) vs dashed (not significant)
            ax.plot([lo, hi], [i, i], color=c, linewidth=2.5,
                    linestyle="-" if sig else dash_style, alpha=0.9, zorder=2)
            # CI caps
            for x_cap in (lo, hi):
                ax.plot([x_cap, x_cap], [i - 0.12, i + 0.12],
                        color=c, linewidth=2.0, alpha=0.9, zorder=2)
            # Point: filled (significant) vs hollow (not significant)
            ax.plot(est, i, marker="o", markersize=10,
                    markerfacecolor=c if sig else "white",
                    markeredgecolor=c, markeredgewidth=2.2, zorder=3)
            p_fdr = row.get("p_fdr", np.nan)
            p_label = f"p={p_fdr:.3f}" if pd.notna(p_fdr) else ""
            star = "✱ " if sig else ""
            ax.text(
                hi, i + 0.22,
                f"{star}t={row['t_value']:.2f}  {p_label}",
                va="bottom", fontsize=8.5, color=c,
                fontweight="bold" if sig else "normal",
            )
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        # Colour each y-tick label by its dimension to reinforce the legend
        for tick, c in zip(ax.get_yticklabels(), colors):
            tick.set_color(c)
            tick.set_fontweight("bold")
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_ylim(-0.6, len(labels) - 0.4)

    _draw_panel(
        ax_add, add_plot, add_labels, add_colors, add_sig,
        ci_lower_col="conf_lower", ci_upper_col="conf_upper",
        xlabel="Coefficient (95% CI)",
        title="Additive Effects",
    )
    if mod_plot is not None:
        _draw_panel(
            ax_mod, mod_plot, mod_labels, mod_colors, mod_sig,
            ci_lower_col="ci_lower", ci_upper_col="ci_upper",
            xlabel="onoff × moderator coefficient (95% CI)",
            title="Moderation of onoff Effect",
        )

    # Shared legend — fill/line style encodes significance (colour = dimension)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="#444444",
                   markerfacecolor="#444444", markeredgecolor="#444444",
                   markersize=9, linewidth=2.5, linestyle="-",
                   label=f"Significant (p_fdr < {MCC_ALPHA})"),
            Line2D([0], [0], marker="o", color="#444444",
                   markerfacecolor="white", markeredgecolor="#444444",
                   markersize=9, linewidth=2.0, linestyle=dash_style,
                   label="Not significant"),
        ],
        fontsize=10, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"{MARKER_LABELS[marker]}: LMM Results  "
        f"(FDR-BH, α={MCC_ALPHA})",
        fontsize=14, fontweight="bold",
    )

    out_path = output_dir / f"{marker}_combined_forest{out_suffix}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def binned_within_subject(
    data: pd.DataFrame, predictor: str, marker: str, bin_width: float
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """Aggregate a predictor–marker relationship as a within-subject binscatter.

    The predictor is split into fixed-width bins of ``bin_width`` (e.g. 0-10,
    10-20, …). Within each subject the mean marker is computed per bin, then
    those per-subject bin means are averaged across subjects. Because each
    subject contributes one value per bin, the across-subject mean reflects the
    *within-subject* relationship — which is what the mixed model (random
    intercept per subject) estimates — rather than between-subject differences
    (removed here anyway by the within-subject z-scoring of the marker).

    Fixed-width (rather than quantile) bins keep the aggregated points evenly
    spaced along the x axis even when the predictor is bimodal (e.g. onoff,
    where ratings pile up near 0 and 100).

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level dataframe containing 'subject', ``predictor`` and ``marker``.
    predictor : str
        Continuous probe predictor (0-100 scale, or time_on_task).
    marker : str
        Objective marker column (within-subject z-scored upstream).
    bin_width : float
        Width of each fixed bin in predictor units.

    Returns
    -------
    agg : pd.DataFrame or None
        One row per non-empty bin with columns ``x`` (bin centre), ``y_mean``
        (mean of per-subject bin means), ``y_se`` (SE across subjects) and ``n``
        (number of contributing subjects). ``None`` if the predictor cannot be
        binned.
    per_sub : pd.DataFrame or None
        Per-subject per-bin means (columns 'subject', ``x_real`` = mean predictor
        in bin, ``y`` = mean marker), for the faint background cloud. ``None``
        when ``agg`` is ``None``.
    """
    sub = data[["subject", predictor, marker]].dropna().copy()
    if sub.empty:
        return None, None

    lo = np.floor(sub[predictor].min() / bin_width) * bin_width
    hi = np.ceil(sub[predictor].max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 2:  # cannot form a single bin
        return None, None

    sub["_bin"] = pd.cut(sub[predictor], bins=edges, include_lowest=True)

    per_sub = (
        sub.groupby(["subject", "_bin"], observed=True)
        .agg(x_real=(predictor, "mean"), y=(marker, "mean"))
        .reset_index()
        .dropna(subset=["y"])
    )
    # Bin centre (constant within a bin) used for the evenly-spaced aggregate dots
    per_sub["x_center"] = per_sub["_bin"].apply(lambda iv: float(iv.mid))

    agg = (
        per_sub.groupby("_bin", observed=True)
        .agg(x=("x_center", "first"), y_mean=("y", "mean"),
             y_std=("y", "std"), n=("y", "size"))
        .reset_index()
        .dropna(subset=["x", "y_mean"])
    )
    agg["y_se"] = agg["y_std"] / np.sqrt(agg["n"].clip(lower=1))
    return agg, per_sub


def marginal_lmm_line(
    results_df: pd.DataFrame, data: pd.DataFrame, predictor: str,
    x_range: np.ndarray,
    predictors: list[str] | None = None,
) -> np.ndarray:
    """LMM marginal prediction line for one predictor, others held at their mean.

    The plotted line must pass through the binned data cloud. Using the bare
    ``intercept + slope * x`` is wrong for a single-predictor panel because the
    multivariate intercept is the prediction when *every* predictor is 0, so the
    line is vertically offset by the (ignored) contribution of the other
    predictors. Here we hold the other predictors at their sample means:

        ŷ(x) = intercept + Σ_{k≠p} β_k · mean(x_k) + β_p · x

    Parameters
    ----------
    results_df : pd.DataFrame
        `fit_lmm` output (must contain 'predictor' and 'estimate', incl. the
        'Intercept' row).
    data : pd.DataFrame
        Data used for the fit, to compute predictor means.
    predictor : str
        The predictor on the x axis of this panel.
    x_range : np.ndarray
        X values at which to evaluate the line.
    predictors : list[str] | None
        Predictor set this model was fit with (used to sum the "others at
        their mean" offset). Defaults to the module-level `PREDICTORS`. Pass
        `SENSITIVITY_PREDICTORS_TIME_SQ` / `SENSITIVITY_PREDICTORS_VALENCE_SQ`
        for the square-effects sensitivity models.

    Returns
    -------
    np.ndarray
        Predicted marker values along ``x_range``.
    """
    if predictors is None:
        predictors = PREDICTORS
    coef = dict(zip(results_df["predictor"], results_df["estimate"]))
    intercept = coef.get("Intercept", 0.0)
    slope_p = coef.get(predictor, 0.0)
    others_offset = sum(
        coef.get(k, 0.0) * float(data[k].mean())
        for k in predictors
        if k != predictor and k in data.columns
    )
    return intercept + others_offset + slope_p * x_range


def plot_scatter_grid(
    data: pd.DataFrame, results_df: pd.DataFrame, marker: str, output_dir: Path,
    show_stats: bool = True,
    out_suffix: str = "",
    predictors: list[str] | None = None,
) -> Path:
    """Plotly grid of within-subject binscatter panels: one per predictor.

    Each panel shows, in the predictor's dimension colour:
    - Faint per-subject bin means (one light point per subject × bin).
    - The mean ± SE across subjects per fixed-width bin (bold points + error
      bars) — the colour points are the across-subject average.
    - The LMM marginal slope line, others held at their mean so it passes
      through the data (solid = significant after FDR, dashed = not).

    Plain per-subject means are avoided on purpose: the marker is z-scored
    within subject upstream, so a single mean per subject collapses to ~0 and
    erases the within-subject effect. See :func:`binned_within_subject`.

    Parameters
    ----------
    data : pd.DataFrame
        Merged probe-level dataframe (marker within-subject z-scored).
    results_df : pd.DataFrame
        Output of `fit_lmm` for this marker.
    marker : str
        Dependent variable name.
    output_dir : Path
        Output directory.
    show_stats : bool
        If True (default), panel titles include β and p_fdr. If False, only
        the dimension name is shown — useful when pairing with a forest plot.
    out_suffix : str
        Optional suffix appended to the output filename before the extension.
    predictors : list[str] | None
        Predictor set to plot (one panel per predictor). Defaults to the
        module-level `PREDICTORS`. Pass `SENSITIVITY_PREDICTORS_TIME_SQ` /
        `SENSITIVITY_PREDICTORS_VALENCE_SQ` for the square-effects sensitivity
        models.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if predictors is None:
        predictors = PREDICTORS

    n_cols = 3
    n_rows = int(np.ceil(len(predictors) / n_cols))

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        horizontal_spacing=0.08, vertical_spacing=0.18,
    )

    z_suffix = " (within-subj. z)" if APPLY_WITHIN_SUBJECT_Z else ""
    marker_label = f"{MARKER_LABELS[marker]}{z_suffix}"

    # Collect all per-subject cloud values to set a shared, robust y-range so the
    # noisy individual points do not dominate the across-subject mean points.
    cloud_y: list[float] = []

    for ax_idx, predictor in enumerate(predictors):
        row = ax_idx // n_cols + 1
        col = ax_idx % n_cols + 1
        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)

        pred_row = results_df[results_df["predictor"] == predictor]
        significant = bool(pred_row["significant_fdr"].values[0]) if len(pred_row) > 0 else False

        if predictor not in data.columns or marker not in data.columns:
            continue

        agg, per_sub = binned_within_subject(
            data, predictor, marker, SCATTER_BIN_WIDTH
        )
        if agg is None:
            continue

        cloud_y.extend(per_sub["y"].tolist())

        # Faint per-subject bin means (cloud)
        fig.add_trace(
            go.Scatter(
                x=per_sub["x_real"], y=per_sub["y"],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.18),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        # Across-subject mean ± SE per bin (foreground colour points)
        fig.add_trace(
            go.Scatter(
                x=agg["x"], y=agg["y_mean"],
                error_y=dict(type="data", array=agg["y_se"], visible=True,
                             thickness=1.6, width=4, color=color),
                mode="markers",
                marker=dict(color=color, size=9,
                            line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=(
                    f"{PREDICTOR_LABELS.get(predictor, predictor)}: "
                    "%{x:.0f}<br>mean: %{y:.3f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        # LMM marginal slope line (others at their mean → passes through data)
        x_line = np.linspace(float(agg["x"].min()), float(agg["x"].max()), 100)
        y_line = marginal_lmm_line(results_df, data, predictor, x_line, predictors=predictors)
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line,
                mode="lines",
                line=dict(color=color, width=3,
                          dash="solid" if significant else "dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.update_xaxes(title_text=PREDICTOR_LABELS.get(predictor, predictor),
                         row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=marker_label, row=row, col=col)

    # Add panel titles anchored to each subplot's own domain (xref/yref = 'x{n}
    # domain' / 'y{n} domain') so they sit just above their own panel at y=1.07
    # regardless of where the panel sits in the figure.  This avoids the plotly
    # subplot_titles overlap bug where row-2 titles land on row-1 x-axis labels.
    axis_idx = 1  # plotly numbers axes x1,y1 … x6,y6 for a 2×3 grid
    for ax_idx, predictor in enumerate(predictors):
        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)
        pred_row_ann = results_df[results_df["predictor"] == predictor]
        slope_ann = pred_row_ann["estimate"].values[0] if len(pred_row_ann) > 0 else 0.0
        p_fdr_ann = pred_row_ann["p_fdr"].values[0] if len(pred_row_ann) > 0 else np.nan
        sig_ann = bool(pred_row_ann["significant_fdr"].values[0]) if len(pred_row_ann) > 0 else False
        star_ann = " ✱" if sig_ann else ""
        p_text_ann = f"p_fdr={p_fdr_ann:.3f}" if not np.isnan(p_fdr_ann) else ""

        x_ref = f"x{axis_idx} domain" if axis_idx > 1 else "x domain"
        y_ref = f"y{axis_idx} domain" if axis_idx > 1 else "y domain"

        if show_stats:
            title_text = (
                f"<b>{PREDICTOR_LABELS.get(predictor, predictor)}{star_ann}</b>"
                f"  β={slope_ann:.3f}  {p_text_ann}"
            )
        else:
            title_text = f"<b>{PREDICTOR_LABELS.get(predictor, predictor)}</b>"

        fig.add_annotation(
            text=title_text,
            xref=x_ref, yref=y_ref,
            x=0.5, y=1.07,
            xanchor="center", yanchor="bottom",
            showarrow=False,
            font=dict(color=color, size=12),
        )
        axis_idx += 1

    # Shared robust y-range (2nd–98th pct of the cloud) so the across-subject
    # mean points and the LMM line stay readable instead of being squashed by
    # the noisiest individual per-subject bin means.
    if cloud_y:
        y_lo = float(np.percentile(cloud_y, 2))
        y_hi = float(np.percentile(cloud_y, 98))
        pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.5
        fig.update_yaxes(range=[y_lo - pad, y_hi + pad])

    if show_stats:
        title_text = (
            f"<b>{MARKER_LABELS[marker]}: LMM Coefficients by Predictor</b>"
            f"<br><sub>faint = per-subject bin means · points = mean ± SE "
            f"across subjects · line = LMM marginal slope · "
            f"bins of {SCATTER_BIN_WIDTH:g}</sub>"
        )
        top_margin = 110
    else:
        title_text = f"<b>{MARKER_LABELS[marker]}: LMM Coefficients by Predictor</b>"
        top_margin = 80

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title_text, x=0.5, xanchor="center"),
        width=420 * n_cols, height=430 * n_rows,
        margin=dict(t=top_margin, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    out_base = output_dir / f"{marker}_scatter_grid{out_suffix}"
    fig.write_html(f"{out_base}.html")
    fig.write_image(f"{out_base}.png", scale=2)
    fig.write_image(f"{out_base}.pdf")
    print(f"Saved: {out_base}.html / .png / .pdf")
    return Path(f"{out_base}.png")


def plot_combined_panel(
    data: pd.DataFrame,
    results_df: pd.DataFrame,
    moderation_df: pd.DataFrame | None,
    marker: str,
    output_dir: Path,
    width: int = 1700,
    height: int = 920,
    forest_col_weight: float = 1.6,
    predictors: list[str] | None = None,
) -> None:
    """Single plotly figure: scatter grid (left, 4 cols × N rows) + forest (right).

    Layout: N rows × 5 cols.  Cols 1-4 hold the binscatter panels (one per
    predictor in ``predictors``); col 5 holds the additive forest (row 1) and,
    if ``moderation_df`` is given, the moderation forest (row 2). The forest
    column is ``forest_col_weight`` times wider than each scatter column.
    Every predictor set (present, time_sq, valence_sq) now passes a
    ``moderation_df``, so row 2 is normally drawn; ``moderation_df=None``
    remains supported (row 2 left empty/hidden, additive-forest-only) for any
    future set with ``with_moderation=False``.

    Scatter panel titles show only dimension names (statistics are in the
    forest). Both halves share the same plotly template and font sizes so the
    figure looks like one coherent object, not two pasted images.

    Parameters
    ----------
    data, results_df, moderation_df, marker, output_dir : (see other plot fns)
    width, height : int
        Output figure size in CSS pixels (scale=2 doubles the export resolution).
    forest_col_weight : float
        Width of the forest column relative to one scatter column (default 1.6).
    predictors : list[str] | None
        Predictor set shown in the scatter grid + additive forest. Defaults to
        the module-level `PREDICTORS`. Pass `SENSITIVITY_PREDICTORS_TIME_SQ` /
        `SENSITIVITY_PREDICTORS_VALENCE_SQ` for the square-effects sensitivity
        models.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if predictors is None:
        predictors = PREDICTORS

    # ── Prepare forest data ────────────────────────────────────────────────────
    add_plot = (
        results_df[results_df["predictor"].isin(predictors)]
        .copy()
        .sort_values("estimate", ascending=True)
        .reset_index(drop=True)
    )
    if moderation_df is not None:
        mod_plot = (
            moderation_df[moderation_df["marker"] == marker]
            .copy()
            .sort_values("estimate", ascending=True)
            .reset_index(drop=True)
        )
        mod_plot["ci_lower"] = mod_plot["estimate"] - 1.96 * mod_plot["std_error"]
        mod_plot["ci_upper"] = mod_plot["estimate"] + 1.96 * mod_plot["std_error"]
    else:
        mod_plot = None

    n_scatter_cols = 4
    n_total_cols = n_scatter_cols + 1  # + forest column
    n_scatter_rows = int(np.ceil(len(predictors) / n_scatter_cols))
    min_rows = 2 if mod_plot is not None else 1
    total_rows = max(n_scatter_rows, min_rows)  # ≥2 rows when 2 forest panels are needed
    unit = 1.0
    col_widths = [unit] * n_scatter_cols + [unit * forest_col_weight]
    forest_col = n_total_cols  # rightmost column

    fig = make_subplots(
        rows=total_rows, cols=n_total_cols,
        column_widths=col_widths,
        horizontal_spacing=0.045,
        vertical_spacing=0.30 if total_rows == 2 else 0.22,
    )

    z_suffix = " (within-subj. z)" if APPLY_WITHIN_SUBJECT_Z else ""
    marker_label = f"{MARKER_LABELS[marker]}{z_suffix}"

    # ── Scatter panels (cols 1-3, rows 1-2) ───────────────────────────────────
    cloud_y_all: list[float] = []
    scatter_data: dict = {}  # cache per predictor so we don't bin twice

    for ax_idx, predictor in enumerate(predictors):
        if predictor not in data.columns or marker not in data.columns:
            continue
        agg, per_sub = binned_within_subject(data, predictor, marker, SCATTER_BIN_WIDTH)
        if agg is None:
            continue
        scatter_data[predictor] = (agg, per_sub)
        cloud_y_all.extend(per_sub["y"].tolist())

    y_lo, y_hi = None, None
    if cloud_y_all:
        y_lo = float(np.percentile(cloud_y_all, 2))
        y_hi = float(np.percentile(cloud_y_all, 98))

    for ax_idx, predictor in enumerate(predictors):
        s_row = ax_idx // n_scatter_cols + 1
        s_col = ax_idx % n_scatter_cols + 1
        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)

        pred_row_s = results_df[results_df["predictor"] == predictor]
        significant = bool(pred_row_s["significant_fdr"].values[0]) if len(pred_row_s) > 0 else False

        if predictor not in scatter_data:
            continue
        agg, per_sub = scatter_data[predictor]

        # Cloud
        fig.add_trace(go.Scatter(
            x=per_sub["x_real"], y=per_sub["y"],
            mode="markers",
            marker=dict(color=color, size=4, opacity=0.15),
            showlegend=False, hoverinfo="skip",
        ), row=s_row, col=s_col)

        # Mean ± SE per bin
        fig.add_trace(go.Scatter(
            x=agg["x"], y=agg["y_mean"],
            error_y=dict(type="data", array=agg["y_se"].tolist(), visible=True,
                         thickness=1.5, width=4, color=color),
            mode="markers",
            marker=dict(color=color, size=8, line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=(
                f"{PREDICTOR_LABELS.get(predictor, predictor)}: %{{x:.0f}}"
                f"<br>mean: %{{y:.3f}}<extra></extra>"
            ),
        ), row=s_row, col=s_col)

        # LMM marginal line (others @ mean)
        x_line = np.linspace(float(agg["x"].min()), float(agg["x"].max()), 100)
        y_line = marginal_lmm_line(results_df, data, predictor, x_line, predictors=predictors)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines",
            line=dict(color=color, width=2.5,
                      dash="solid" if significant else "dash"),
            showlegend=False, hoverinfo="skip",
        ), row=s_row, col=s_col)

        if y_lo is not None:
            pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.2
            fig.update_yaxes(range=[y_lo - pad, y_hi + pad], row=s_row, col=s_col)

        fig.update_xaxes(
            title_text=PREDICTOR_LABELS.get(predictor, predictor),
            title_font=dict(size=11), row=s_row, col=s_col,
        )
        if s_col == 1:
            fig.update_yaxes(
                title_text=marker_label,
                title_font=dict(size=10), row=s_row, col=s_col,
            )

    # ── Forest panels (rightmost column, rows 1 and 2) ────────────────────────
    def _add_forest_panel(
        plot_df: pd.DataFrame,
        ci_lo_col: str,
        ci_hi_col: str,
        f_row: int,
        predictor_col: str,
    ) -> None:
        labels = [PREDICTOR_LABELS.get(p, p) for p in plot_df[predictor_col]]
        colors = [PREDICTOR_COLORS.get(p, DEFAULT_PREDICTOR_COLOR) for p in plot_df[predictor_col]]
        sig_flags = [bool(s) for s in plot_df["significant_fdr"].tolist()]

        # Vertical reference at 0
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-0.6, len(labels) - 0.4],
            mode="lines",
            line=dict(color="#888", width=1.2, dash="dash"),
            showlegend=False, hoverinfo="skip",
        ), row=f_row, col=forest_col)

        for i, (_, r) in enumerate(plot_df.iterrows()):
            c = colors[i]
            sig = sig_flags[i]
            est, lo, hi = r["estimate"], r[ci_lo_col], r[ci_hi_col]
            p_fdr = r.get("p_fdr", np.nan)
            star = " ✱" if sig else ""
            p_text = f"p={p_fdr:.3f}" if pd.notna(p_fdr) else ""

            # CI line
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[i, i], mode="lines",
                line=dict(color=c, width=2.5, dash="solid" if sig else "dash"),
                showlegend=False, hoverinfo="skip",
            ), row=f_row, col=forest_col)

            # CI caps
            for x_cap in (lo, hi):
                fig.add_trace(go.Scatter(
                    x=[x_cap, x_cap], y=[i - 0.12, i + 0.12], mode="lines",
                    line=dict(color=c, width=1.8),
                    showlegend=False, hoverinfo="skip",
                ), row=f_row, col=forest_col)

            # Point
            fig.add_trace(go.Scatter(
                x=[est], y=[i], mode="markers",
                marker=dict(
                    color=c if sig else "white", size=11,
                    line=dict(color=c, width=2.2),
                ),
                showlegend=False,
                hovertemplate=f"{labels[i]}{star}<br>β={est:.5g}<br>{p_text}<extra></extra>",
            ), row=f_row, col=forest_col)

        # Y-axis: predictor names as tick labels, placed on the right side so
        # they extend into the right margin instead of bleeding into the
        # adjacent scatter column on the left (colour already encoded by the
        # CI line/point, so a single tick colour is sufficient).
        fig.update_yaxes(
            tickvals=list(range(len(labels))),
            ticktext=[f"<b>{label}</b>" for label in labels],
            tickfont=dict(size=10),
            side="right",
            automargin=True,
            range=[-0.6, len(labels) - 0.4],
            row=f_row, col=forest_col,
        )

    _add_forest_panel(add_plot, "conf_lower", "conf_upper", f_row=1,
                      predictor_col="predictor")
    if mod_plot is not None:
        _add_forest_panel(mod_plot, "ci_lower",   "ci_upper",   f_row=2,
                          predictor_col="moderator")

    # x-axis titles for forest encoded inside panel title (avoids gap overlap)
    fig.update_xaxes(title_text="", row=1, col=forest_col)
    if mod_plot is not None:
        fig.update_xaxes(title_text="", row=2, col=forest_col)

    # ── Panel titles (domain-anchored annotations) ─────────────────────────────
    for ax_idx, predictor in enumerate(predictors):
        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)
        s_row = ax_idx // n_scatter_cols + 1
        s_col = ax_idx % n_scatter_cols + 1
        ax_n = (s_row - 1) * n_total_cols + s_col
        xr = "x domain" if ax_n == 1 else f"x{ax_n} domain"
        yr = "y domain" if ax_n == 1 else f"y{ax_n} domain"
        fig.add_annotation(
            text=f"<b>{PREDICTOR_LABELS.get(predictor, predictor)}</b>",
            xref=xr, yref=yr,
            x=0.5, y=1.08, xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(color=color, size=10),
        )

    # Forest titles left-aligned (xanchor="left") so the text grows rightward
    # into the forest panel/margin rather than leftward over the col-4 title.
    forest_ax_row1 = n_total_cols          # row1, forest_col → axis n_total_cols
    forest_ax_row2 = n_total_cols * 2      # row2, forest_col → axis 2*n_total_cols
    fig.add_annotation(
        text="<b>Additive Effects</b>  <span style='font-size:9px;color:#777'>coef. (95% CI)</span>",
        xref=f"x{forest_ax_row1} domain", yref=f"y{forest_ax_row1} domain",
        x=0.0, y=1.08, xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(color="#333", size=12),
    )
    if mod_plot is not None:
        fig.add_annotation(
            text="<b>onoff Moderation</b>  <span style='font-size:9px;color:#777'>coef. (95% CI)</span>",
            xref=f"x{forest_ax_row2} domain", yref=f"y{forest_ax_row2} domain",
            x=0.0, y=1.08, xanchor="left", yanchor="bottom",
            showarrow=False, font=dict(color="#333", size=12),
        )

    # Hide empty grid cells (e.g. the leftover scatter cell when len(predictors)
    # is not a multiple of n_scatter_cols, or unused forest-col rows)
    used_cells = {(1, forest_col)}
    if mod_plot is not None:
        used_cells.add((2, forest_col))
    for ax_idx in range(len(predictors)):
        used_cells.add((ax_idx // n_scatter_cols + 1, ax_idx % n_scatter_cols + 1))
    for r in range(1, total_rows + 1):
        for c in range(1, n_total_cols + 1):
            if (r, c) not in used_cells:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)

    effective_height = height if total_rows == 2 else height * total_rows // 2

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>{MARKER_LABELS[marker]}</b>",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        width=width, height=effective_height,
        margin=dict(t=80, b=55, l=55, r=140),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)

    out_base = output_dir / f"{marker}_combined_panel"
    fig.write_html(f"{out_base}.html")
    fig.write_image(f"{out_base}.png", scale=2)
    fig.write_image(f"{out_base}.pdf")
    print(f"Saved combined panel: {out_base}.png")


# =============================================================================
# MODERATION ANALYSIS
# =============================================================================


def fit_moderation_lmm(
    data: pd.DataFrame,
    marker: str,
    moderator: str,
    method: str,
    maxiter: int,
) -> dict:
    """Fit a moderation LMM: marker ~ onoff * moderator + (1|subject).

    The key test is the interaction term `onoff:moderator`, which indicates
    whether the slope of `onoff` on the marker depends on the moderator.

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level merged dataframe.
    marker : str
        Dependent variable (objective marker).
    moderator : str
        Potential moderator variable (probe dimension).
    method : str
        Optimisation method for statsmodels.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    dict
        Keys: marker, moderator, interaction_term, estimate, std_error,
        t_value, p_value, n_obs, n_subjects, converged.
    """
    import re

    formula_full = f"{marker} ~ onoff * {moderator}"
    interaction_term = f"onoff:{moderator}"

    needed_cols = ["subject", marker, "onoff", moderator]
    model_data = (
        data[[c for c in needed_cols if c in data.columns]]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )
    model_data["subject"] = model_data["subject"].astype(str)

    print(f"  Moderator: {moderator}  |  N={len(model_data)}, "
          f"subjects={model_data['subject'].nunique()}")

    model = smf.mixedlm(formula_full, model_data, groups="subject")
    result = model.fit(method=method.upper(), maxiter=maxiter, reml=True, disp=False)

    # Extract the interaction term (statsmodels may name it differently)
    # Try both "onoff:moderator" and "moderator:onoff"
    term_candidates = [interaction_term, f"{moderator}:onoff"]
    found_term = next((t for t in term_candidates if t in result.params.index), None)

    if found_term is None:
        # Fallback: look for a term containing both variable names
        found_term = next(
            (t for t in result.params.index if "onoff" in t and moderator in t),
            None,
        )

    if found_term is None:
        return {
            "marker": marker, "moderator": moderator,
            "interaction_term": interaction_term,
            "estimate": np.nan, "std_error": np.nan,
            "t_value": np.nan, "p_value": np.nan,
            "n_obs": len(model_data),
            "n_subjects": model_data["subject"].nunique(),
            "converged": result.converged,
        }

    return {
        "marker": marker,
        "moderator": moderator,
        "interaction_term": found_term,
        "estimate": float(result.params[found_term]),
        "std_error": float(result.bse[found_term]),
        "t_value": float(result.tvalues[found_term]),
        "p_value": float(result.pvalues[found_term]),
        "n_obs": len(model_data),
        "n_subjects": model_data["subject"].nunique(),
        "converged": result.converged,
    }


def plot_moderation_forest(
    moderation_df: pd.DataFrame, marker: str, output_dir: Path
) -> None:
    """Forest plot of all interaction terms for one marker.

    Shows the `onoff:moderator` coefficient for each moderator,
    coloured by significance after FDR.

    Parameters
    ----------
    moderation_df : pd.DataFrame
        Output of `run_moderation_analysis`, filtered to this marker,
        with `p_fdr` and `significant_fdr` columns.
    marker : str
        Marker name.
    output_dir : Path
        Output directory.
    """
    plot_df = moderation_df[
        moderation_df["marker"] == marker
    ].copy().sort_values("estimate", ascending=True).reset_index(drop=True)

    if plot_df.empty:
        return

    labels = [PREDICTOR_LABELS.get(m, m) for m in plot_df["moderator"]]
    colors = [
        PALETTE_SIGNIFICANT if sig is True else PALETTE_NOT_SIGNIFICANT
        for sig in plot_df["significant_fdr"].tolist()
    ]

    # Compute 95% CI from estimate ± 1.96 * SE
    plot_df["ci_lower"] = plot_df["estimate"] - 1.96 * plot_df["std_error"]
    plot_df["ci_upper"] = plot_df["estimate"] + 1.96 * plot_df["std_error"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.7 + 1.5)))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.errorbar(
            x=row["estimate"], y=i,
            xerr=[[row["estimate"] - row["ci_lower"]],
                  [row["ci_upper"] - row["estimate"]]],
            fmt="D", markersize=8, color=colors[i],
            capsize=5, capthick=2, linewidth=2, elinewidth=2,
        )
        p_fdr = row["p_fdr"]
        p_label = f"p_fdr={p_fdr:.3f}" if not np.isnan(p_fdr) else ""
        ax.text(
            row["ci_upper"] + 0.001, i,
            f"t={row['t_value']:.2f}  {p_label}",
            va="center", fontsize=9, color=colors[i],
        )

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Interaction Coefficient: onoff × moderator (95% CI)",
                  fontsize=11, fontweight="bold")
    ax.set_title(
        f"{MARKER_LABELS[marker]}: Moderation of onoff Effect\n"
        f"(FDR-BH corrected across all marker × moderator tests, α={MCC_ALPHA})",
        fontsize=12, fontweight="bold", pad=10,
    )

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=PALETTE_SIGNIFICANT, markersize=9,
               label=f"Significant (p_fdr < {MCC_ALPHA})"),
        Line2D([0], [0], marker="D", color="w",
               markerfacecolor=PALETTE_NOT_SIGNIFICANT, markersize=9,
               label="Not significant"),
    ], fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"moderation_forest_{marker}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_interaction(
    data: pd.DataFrame,
    marker: str,
    moderator: str,
    output_dir: Path,
) -> None:
    """Interaction plot: onoff effect on marker at Low vs High moderator.

    The moderator is split at the within-subject median. Probes where the
    moderator is below the subject's median are 'Low'; above are 'High'.
    Each group's mean is shown as a function of onoff quintile bins.

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level data.
    marker : str
        Dependent variable.
    moderator : str
        Moderator variable.
    output_dir : Path
        Output directory.
    """
    df_plot = data[["subject", "onoff", moderator, marker]].dropna().copy()

    # Within-subject median split of the moderator
    subj_medians = df_plot.groupby("subject")[moderator].median()
    df_plot["mod_group"] = df_plot.apply(
        lambda row: "High" if row[moderator] >= subj_medians[row["subject"]] else "Low",
        axis=1,
    )

    # Bin onoff into 5 equal-width bins for visualisation
    df_plot["onoff_bin"] = pd.cut(
        df_plot["onoff"], bins=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    palette = {"Low": PALETTE_MOD_LOW, "High": PALETTE_MOD_HIGH}
    x_positions = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5}

    for group, color in palette.items():
        grp = df_plot[df_plot["mod_group"] == group]
        agg = grp.groupby("onoff_bin", observed=True)[marker].agg(["mean", "sem"])
        x_vals = [x_positions[b] for b in agg.index]
        ax.errorbar(
            x_vals, agg["mean"], yerr=agg["sem"],
            marker="o", linewidth=2.5, markersize=8,
            capsize=4, capthick=2, color=color,
            label=f"{PREDICTOR_LABELS.get(moderator, moderator)}: {group}",
        )

    mod_label = PREDICTOR_LABELS.get(moderator, moderator)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(["Q1\n(low onoff)", "Q2", "Q3", "Q4", "Q5\n(high onoff)"],
                       fontsize=10)
    ax.set_xlabel("On/Off Task (quintile bins)", fontsize=12, fontweight="bold")
    ax.set_ylabel(MARKER_LABELS[marker], fontsize=12, fontweight="bold")
    ax.set_title(
        f"{MARKER_LABELS[marker]}\n"
        f"Moderation by {mod_label} (Low vs High, within-subject median split)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"interaction_{marker}_{moderator}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_moderation_analysis(
    data: pd.DataFrame,
    markers: list[str],
    moderators: list[str],
    method: str,
    maxiter: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Run all moderation LMMs and produce a FDR-corrected summary.

    For each marker × moderator combination fits:
        marker ~ onoff * moderator + (1|subject)
    and extracts the `onoff:moderator` interaction coefficient.
    FDR-BH is applied across ALL marker × moderator tests together.

    Parameters
    ----------
    data : pd.DataFrame
        Probe-level data.
    markers : list[str]
        Objective markers to use as dependent variables.
    moderators : list[str]
        Variables to test as potential moderators.
    method : str
        LMM optimisation method.
    maxiter : int
        Maximum iterations.
    output_dir : Path
        Output directory for CSVs and figures.

    Returns
    -------
    pd.DataFrame
        One row per (marker, moderator) with interaction statistics and FDR.
    """
    print(f"\n{'='*60}")
    print("MODERATION ANALYSIS: onoff × moderator")
    print(f"Testing {len(markers)} markers × {len(moderators)} moderators "
          f"= {len(markers) * len(moderators)} models")
    print("="*60)

    rows = []
    for marker in markers:
        print(f"\n--- Marker: {MARKER_LABELS[marker]} ---")
        for moderator in moderators:
            row = fit_moderation_lmm(data, marker, moderator, method, maxiter)
            rows.append(row)

    moderation_df = pd.DataFrame(rows)

    # FDR-BH across ALL tests (marker × moderator)
    valid_mask = moderation_df["p_value"].notna()
    p_vals = moderation_df.loc[valid_mask, "p_value"].values
    _, p_fdr, _, _ = multipletests(p_vals, method=MCC_METHOD)

    moderation_df["p_fdr"] = np.nan
    moderation_df.loc[valid_mask, "p_fdr"] = p_fdr
    moderation_df["significant_fdr"] = moderation_df["p_fdr"] < MCC_ALPHA

    # Add readable labels
    moderation_df["marker_label"] = moderation_df["marker"].map(MARKER_LABELS)
    moderation_df["moderator_label"] = moderation_df["moderator"].map(
        lambda m: PREDICTOR_LABELS.get(m, m)
    )

    # Save summary CSV
    csv_path = output_dir / "moderation_summary.csv"
    moderation_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Print results table
    display_cols = ["marker_label", "moderator_label", "estimate",
                    "t_value", "p_value", "p_fdr", "significant_fdr"]
    print(f"\n{'='*60}")
    print("MODERATION RESULTS (onoff × moderator interaction term):")
    print("="*60)
    print(moderation_df[display_cols].to_string(index=False))

    # Interaction plots for significant pairs (or all if none significant)
    sig_pairs = moderation_df[moderation_df["significant_fdr"] == True]
    if len(sig_pairs) > 0:
        print(f"\n{len(sig_pairs)} significant moderation(s) — generating interaction plots...")
        for _, row in sig_pairs.iterrows():
            plot_interaction(data, row["marker"], row["moderator"], output_dir)
    else:
        print("\nNo significant moderations after FDR correction.")
        print("Generating interaction plots for all pairs (exploratory)...")
        for marker in markers:
            for moderator in moderators:
                plot_interaction(data, marker, moderator, output_dir)

    return moderation_df


# =============================================================================
# MAIN
# =============================================================================

def _fit_predictor_set(
    df: pd.DataFrame, set_name: str, set_cfg: dict, output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Fit one predictor set's LMMs and generate its full plot suite.

    For every marker in `OBJECTIVE_MARKERS`, fits the additive LMM given by
    `set_cfg["formula_template"]` / `set_cfg["predictors"]`, saves
    `<marker>_lmm_results.csv` and `lmm_summary_all_markers.csv`, runs
    `run_moderation_analysis` over `set_cfg["moderators"]` if
    `set_cfg["with_moderation"]`, and generates the scatter grid, combined
    forest and combined panel — all under
    `output_dir / set_cfg["output_subdir"]`.

    Parameters
    ----------
    df : pd.DataFrame
        Merged probe-level dataframe with all derived predictor columns
        (time_on_task, present, time_sq, valence_sq).
    set_name : str
        Key into `PREDICTOR_SETS` (used for log messages only).
    set_cfg : dict
        One entry of `PREDICTOR_SETS`.
    output_dir : Path
        Dataset output directory (`OUTPUT_BASE_DIR / dataset_name`).

    Returns
    -------
    dict[str, pd.DataFrame]
        Per-marker `fit_lmm` results for this predictor set, for use in
        cross-model comparisons by the caller.
    """
    predictors = set_cfg["predictors"]
    set_dir = output_dir / set_cfg["output_subdir"]
    set_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PREDICTOR SET: {set_name}  ->  {set_dir}")
    print("="*60)

    # --- Fit additive LMM per marker ---
    results: dict[str, pd.DataFrame] = {}
    for marker in OBJECTIVE_MARKERS:
        results_df = fit_lmm(
            data=df,
            marker=marker,
            formula_template=set_cfg["formula_template"],
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
            predictors=predictors,
        )
        results[marker] = results_df
        csv_path = set_dir / f"{marker}_lmm_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results: {csv_path}")

    # --- Summary table across markers ---
    summary_rows = []
    for marker, results_df in results.items():
        for _, row in results_df[results_df["predictor"].isin(predictors)].iterrows():
            summary_rows.append({
                "marker": MARKER_LABELS[marker],
                "predictor": PREDICTOR_LABELS.get(row["predictor"], row["predictor"]),
                "estimate": row["estimate"],
                "std_error": row["std_error"],
                "t_value": row["t_value"],
                "p_value": row["p_value"],
                "p_fdr": row.get("p_fdr", np.nan),
                "significant_fdr": row.get("significant_fdr", False),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = set_dir / "lmm_summary_all_markers.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved: {summary_path}")

    sig = summary_df[summary_df["significant_fdr"] == True]
    print(f"\nSIGNIFICANT ADDITIVE EFFECTS ({set_name}, FDR-BH corrected):")
    if len(sig) > 0:
        print(sig[["marker", "predictor", "estimate", "t_value", "p_value", "p_fdr"]].to_string(index=False))
    else:
        print("  None found after FDR correction.")

    # --- Moderation analysis ---
    moderation_df = None
    if set_cfg["with_moderation"]:
        moderation_df = run_moderation_analysis(
            data=df,
            markers=OBJECTIVE_MARKERS,
            moderators=set_cfg["moderators"],
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
            output_dir=set_dir,
        )

    # --- Plots ---
    print(f"\nGenerating plots for {set_name}...")
    for marker, results_df in results.items():
        plot_scatter_grid(df, results_df, marker, set_dir, predictors=predictors)
        plot_combined_forest(results_df, moderation_df, marker, set_dir, predictors=predictors)
        plot_combined_panel(df, results_df, moderation_df, marker, set_dir, predictors=predictors)

    return results


def run_pipeline_for_dataset(dataset_name: str, markers_path: str, output_base: Path) -> None:
    """Run the full LMM pipeline for a specific objective markers dataset."""
    print(f"\n\n{'#'*80}")
    print(f"### RUNNING PIPELINE FOR DATASET: {dataset_name.upper()}")
    print(f"{'#'*80}\n")

    np.random.seed(RANDOM_STATE)
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    print("Loading objective markers data...")
    df_markers = pd.read_csv(markers_path)
    df_markers["total_errors"] = (
        df_markers["omission_rate"] + df_markers["commission_rate"]
    )
    print(f"  {len(df_markers)} probe-level observations")

    print("Loading probe metadata...")
    df_probe = pd.read_csv(PROBE_DATA_PATH)
    print(f"  {len(df_probe)} probe observations with metadata")

    # --- Merge ---------------------------------------------------------------
    df_markers["subject"] = df_markers["subject"].astype(str)
    df_probe["subject_id"] = df_probe["subject_id"].astype(str)

    df_markers_renamed = df_markers.rename(columns={"sart": "task"})

    merge_cols_probe = [
        c for c in [
            "subject_id", "task", "probe_number", "group",
            "order (IE/EI)", "inclusion_exclusion",
            "onoff", "valence", "time", "selfother", "confidence",
        ]
        if c in df_probe.columns
    ]

    df = pd.merge(
        df_markers_renamed,
        df_probe[merge_cols_probe],
        left_on=["subject", "task", "probe_number"],
        right_on=["subject_id", "task", "probe_number"],
        how="inner",
    )
    print(f"\nMerged: {len(df)} observations from {df['subject'].nunique()} subjects")

    # --- Task normalisation & filter -----------------------------------------
    df["task"] = df["task"].apply(normalize_task_label)
    df = df[df["task"].isin(["Sart1", "Sart2", "Sart3", "Sart4"])].copy()

    # --- Derived variables ---------------------------------------------------
    df = add_time_on_task(df)
    df = compute_present_dimension(df)
    df = compute_time_squared(df)
    df = compute_valence_squared(df)

    # --- Within-subject z-scoring (same as objective_markers_analysis.py) ----
    if APPLY_WITHIN_SUBJECT_Z:
        print("\nApplying within-subject z-scoring to objective markers...")
        df = apply_within_subject_z_scoring(df, OBJECTIVE_MARKERS)

    # --- Fit + plot each predictor set ----------------------------------------
    results_by_set: dict[str, dict[str, pd.DataFrame]] = {}
    for set_name, set_cfg in PREDICTOR_SETS.items():
        results_by_set[set_name] = _fit_predictor_set(df, set_name, set_cfg, output_dir)

    # --- Sensitivity comparison: present vs time_sq (collinear swap) ----------
    # present and time_sq are both functions of |time-50| and become severely
    # collinear (r=-0.97, VIF~16) once `time` is in the model, so they are fit
    # as SEPARATE models (see SENSITIVITY_PREDICTORS_TIME_SQ); compare the two
    # estimates side by side.
    comparison_rows = []
    for marker in OBJECTIVE_MARKERS:
        present_row = results_by_set["present"][marker][
            results_by_set["present"][marker]["predictor"] == "present"
        ].iloc[0]
        time_sq_row = results_by_set["time_sq"][marker][
            results_by_set["time_sq"][marker]["predictor"] == "time_sq"
        ].iloc[0]
        comparison_rows.append({
            "marker": MARKER_LABELS[marker],
            "present_estimate": present_row["estimate"],
            "present_p_fdr": present_row["p_fdr"],
            "present_significant_fdr": present_row["significant_fdr"],
            "time_sq_estimate": time_sq_row["estimate"],
            "time_sq_p_fdr": time_sq_row["p_fdr"],
            "time_sq_significant_fdr": time_sq_row["significant_fdr"],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "present_vs_time_sq_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSensitivity comparison saved: {comparison_path}")
    print(comparison_df.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE FOR {dataset_name.upper()}")
    print(f"Results in: {output_dir.resolve()}")
    print("="*60)


def plots_only(predictor_set_names: list[str] | None = None) -> None:
    """Regenerate figures from existing CSVs — skips LMM fitting.

    Loads ``<marker>_lmm_results.csv`` (and, for predictor sets with
    ``with_moderation=True``, ``moderation_summary.csv``) from each dataset
    output directory and re-runs the plotting functions. Useful for rapid
    figure iteration without waiting for the LMM fits (~1-2 min). Requires a
    full pipeline run to have been completed first (so the relevant CSVs
    already exist for every requested predictor set).

    Parameters
    ----------
    predictor_set_names : list[str] | None
        Which entries of `PREDICTOR_SETS` to regenerate plots for. Defaults
        to all of them (`["present", "time_sq", "valence_sq"]`). E.g. pass
        `["present"]` to regenerate only the primary plots, or `["time_sq"]`
        / `["valence_sq"]` to regenerate only one
        `sensitivity_square_effects/<set>/` suite.
    """
    if predictor_set_names is None:
        predictor_set_names = list(PREDICTOR_SETS.keys())

    for dataset_name, markers_path in OBJECTIVE_MARKERS_PATHS.items():
        output_dir = OUTPUT_BASE_DIR / dataset_name
        print(f"\n### PLOTS-ONLY for {dataset_name.upper()} — loading CSVs from {output_dir}")

        # --- Reconstruct data (needed for binscatter and marginal line) -------
        df_markers = pd.read_csv(markers_path)
        df_markers["total_errors"] = (
            df_markers["omission_rate"] + df_markers["commission_rate"]
        )
        df_probe = pd.read_csv(PROBE_DATA_PATH)
        df_markers["subject"] = df_markers["subject"].astype(str)
        df_probe["subject_id"] = df_probe["subject_id"].astype(str)
        merge_cols = [c for c in [
            "subject_id", "task", "probe_number",
            "onoff", "valence", "time", "selfother", "confidence",
        ] if c in df_probe.columns]
        df = pd.merge(
            df_markers.rename(columns={"sart": "task"}),
            df_probe[merge_cols],
            left_on=["subject", "task", "probe_number"],
            right_on=["subject_id", "task", "probe_number"],
            how="inner",
        )
        df["task"] = df["task"].apply(normalize_task_label)
        df = df[df["task"].isin(["Sart1", "Sart2", "Sart3", "Sart4"])].copy()
        df = add_time_on_task(df)
        df = compute_present_dimension(df)
        df = compute_time_squared(df)
        df = compute_valence_squared(df)
        if APPLY_WITHIN_SUBJECT_Z:
            df = apply_within_subject_z_scoring(df, OBJECTIVE_MARKERS)

        # --- Regenerate plots for each requested predictor set ----------------
        for set_name in predictor_set_names:
            set_cfg = PREDICTOR_SETS[set_name]
            set_predictors = set_cfg["predictors"]
            set_dir = output_dir / set_cfg["output_subdir"]
            print(f"  --- Predictor set: {set_name} → {set_dir} ---")

            results_for_set: dict[str, pd.DataFrame] = {}
            for marker in OBJECTIVE_MARKERS:
                csv_path = set_dir / f"{marker}_lmm_results.csv"
                results_for_set[marker] = pd.read_csv(csv_path)
                print(f"    Loaded {csv_path}")

            moderation_df = None
            if set_cfg["with_moderation"]:
                moderation_df = pd.read_csv(set_dir / "moderation_summary.csv")
                print("    Loaded moderation_summary.csv")

            print("    Generating scatter grids...")
            for marker, results_df in results_for_set.items():
                plot_scatter_grid(df, results_df, marker, set_dir, predictors=set_predictors)

            print("    Generating forest plots...")
            for marker, results_df in results_for_set.items():
                plot_combined_forest(results_df, moderation_df, marker, set_dir,
                                     predictors=set_predictors)

            print("    Generating combined panels...")
            for marker, results_df in results_for_set.items():
                plot_combined_panel(df, results_df, moderation_df, marker, set_dir,
                                    predictors=set_predictors)

        print(f"  Done → {output_dir}")


def main() -> None:
    """Run the full LMM pipeline for all objective marker datasets."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip LMM fitting; regenerate figures from existing CSVs.")
    parser.add_argument(
        "--predictor-set", choices=list(PREDICTOR_SETS.keys()) + ["all"],
        default="all",
        help=(
            "With --plots-only, restrict regeneration to one predictor set: "
            "'present' (primary, with the `present` dimension), 'time_sq' "
            "or 'valence_sq' (square-effects sensitivity models, "
            "results/.../sensitivity_square_effects/<set>/), or 'all' "
            "(default, regenerates all three)."
        ),
    )
    args = parser.parse_args()

    if args.plots_only:
        set_names = None if args.predictor_set == "all" else [args.predictor_set]
        plots_only(predictor_set_names=set_names)
    else:
        for name, path in OBJECTIVE_MARKERS_PATHS.items():
            run_pipeline_for_dataset(name, path, OUTPUT_BASE_DIR)

if __name__ == "__main__":
    main()
