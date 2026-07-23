"""
LMM Analysis: Probe Dimensions → Objective Behavioral Markers

TWO ANALYSES:

1. ADDITIVE MODEL (one per marker):
    marker ~ onoff + valence + valence_sq + selfother + time + time_sq
             + confidence + time_on_task + (1|subject)
   Tests the independent contribution of each probe dimension, including
   curvature (quadratic) terms for valence and time.

2. MODERATION ANALYSIS (one per marker × moderator):
    marker ~ onoff * moderator + (1|subject)
   For each non-onoff probe dimension (incl. valence_sq, time_sq), tests
   whether it moderates the effect of onoff on the objective marker. The
   interaction term onoff:moderator is the key test. FDR-BH corrected across
   all marker × moderator tests. Visualised as an interaction plot: onoff
   effect at Low vs High moderator (split at within-subject median).

Note: This analysis intentionally does not discriminate between Controls and
Risk of Depression (RoD) groups. For the purposes of this pipeline, all subjects
are treated as a single group in the models and visualizations.

Preprocessing mirrors objective_markers_analysis.py:
  - Merge objective markers with probe-level metadata (subject/task/probe)
  - Derive time_on_task = probe_number + 15 * (sart_number - 1)
  - Within-subject z-scoring of objective markers (APPLY_WITHIN_SUBJECT_Z)

QUADRATIC TERMS (time_sq, valence_sq) — global orthogonalization:
  `time_sq = (time-50)^2/50` and `valence_sq = (valence-50)^2/50` (see
  compute_time_squared / compute_valence_squared) test whether `time` and
  `valence` have curvature (U-shaped) relationships with the markers, beyond
  their linear effects. Each is then GLOBALLY ORTHOGONALIZED against its own
  linear term (orthogonalize_quadratic): a single pooled OLS fit
  `quad ~ linear` is subtracted off, leaving only the curvature component —
  the same parametrization as R's `poly(x, 2)`. By the invariance of the
  highest-order polynomial term, this does NOT change the quadratic term's
  estimate/SE/t/p; it only removes its collinearity with the linear term
  (cleaning up the linear term's VIF/SE). The (intercept, slope) of each
  orthogonalization is saved to `<dataset>/quadratic_orthogonalization.csv`
  for transparency.

`present` REMOVED: a previous predictor `present = 50 - |time-50|` aimed to
capture "distance from now". It is a near-perfect monotonic function of
`|time-50|` (r≈-0.97 with `time_sq`) — the same information as `time_sq`,
just rescaled/flipped — so it cannot coexist with `time_sq` in one model
(severe collinearity). `time_sq` (orthogonalized) already captures this
construct, plus asymmetry, within the single unified model below, so
`present` is dropped entirely.

Outputs (saved under OUTPUT_BASE_DIR/<dataset>/):
  full_model/<marker>_lmm_results.csv       — additive model results + FDR
  full_model/<marker>_combined_forest.png    — additive + moderation forest plot
  full_model/<marker>_scatter_grid.{html,png,pdf} — scatter grid per predictor
  full_model/<marker>_combined_panel.{html,png,pdf} — scatter grid + forest, one figure
  full_model/moderation_summary.csv          — interaction term stats, FDR-corrected
  full_model/moderation_forest_<marker>.png  — forest of interaction coefficients
  full_model/interaction_<marker>_<mod>.png  — interaction plot per significant pair
  quadratic_orthogonalization.csv            — (intercept, slope) for time_sq/valence_sq (dataset root)

PREDICTOR_SETS ties the named predictor set (predictors, formula, output
subdir, moderators) together; `plots_only(predictor_set_names=...)` /
`--predictor-set` lets its plots be regenerated independently of refitting.

Author: Nicolas Bruno
"""

import os
import sys
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics import gaussian_residual_diagnostics, save_diagnostic_plots
from glmm_backend import fit_glmm, fit_moderation_glmm
from response_transforms import empirical_logit, load_glmm_config, log_transform

GLMM_CONFIG = load_glmm_config()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input data paths (same sources as objective_markers_analysis.py)
OBJECTIVE_MARKERS_PATHS: dict[str, str] = {
    "full_segment": "results/Behavior/objective_markers/objective_markers_per_probe.csv",
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
    # Old sum-of-two-rates definition, fitted in the Gaussian track only for
    # continuity with already-published output. See the design spec, section 4.
    "total_errors_legacy": "Total Errors (legacy sum-of-rates)",
}

# Probe-level predictors (fixed effects, continuous, 0-100 scale)
PREDICTORS: list[str] = [
    "onoff", "valence", "valence_sq", "selfother", "time", "time_sq",
    "confidence", "time_on_task",
]

PREDICTOR_LABELS: dict[str, str] = {
    "onoff": "On/Off Task",
    "valence": "Valence",
    "valence_sq": "Valence² (curvature)",
    "selfother": "Self/Other",
    "time": "Time (probe dim.)",
    "time_sq": "Time² (curvature)",
    "confidence": "Confidence",
    "time_on_task": "Time on Task",
}

# LMM formula template — marker name is substituted at runtime
FORMULA_TEMPLATE = (
    "{marker} ~ onoff + valence + valence_sq + selfother + time + time_sq"
    " + confidence + time_on_task + (1|subject)"
)

# Potential moderators of the onoff effect (all probe dimensions except onoff)
# Each will be tested in: marker ~ onoff * moderator + (1|subject)
MODERATORS: list[str] = [
    "valence", "valence_sq", "selfother", "time", "time_sq", "confidence",
    "time_on_task",
]

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

# Z-score all predictors (mean=0, SD=1) before LMM fitting so that β coefficients
# are in SD units and forest-plot CIs are directly comparable across linear and
# quadratic terms. The scatter visualisation always uses the original (un-z-scored)
# data; marginal lines are denormalised via predictor_sds before plotting.
STANDARDIZE_PREDICTORS: bool = True

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
    "onoff":        PALETTE["dimensions"]["onoff"],       # red          — On/Off-Task
    "valence":      PALETTE["dimensions"]["valence"],     # blue         — Valence
    "valence_sq":   PALETTE["quadratic"]["valence_sq"],   # light blue   — Valence² tint
    "selfother":    PALETTE["dimensions"]["selfother"],   # green        — Self/Other
    "time":         PALETTE["dimensions"]["time"],        # purple       — Time (probe dim.)
    "time_sq":      PALETTE["quadratic"]["time_sq"],      # light purple — Time² tint
    "confidence":   PALETTE["dimensions"]["confidence"],  # orange       — Confidence
    "time_on_task": PALETTE["neutral"]["covariate"],      # grey         — covariate
}
DEFAULT_PREDICTOR_COLOR: str = PALETTE["neutral"]["covariate"]

# --- Predictor sets ----------------------------------------------------------
# Single unified model: every probe dimension (incl. the orthogonalized
# quadratic terms time_sq / valence_sq, see orthogonalize_quadratic) in one
# additive model + one onoff-moderation forest. Kept as a dict (rather than a
# bare formula) so the full plot suite (scatter grid + additive forest +
# onoff-moderation forest + combined panel) and `plots_only(...)` /
# `--predictor-set` can stay generic across `_fit_predictor_set`.
PREDICTOR_SETS: dict[str, dict] = {
    "full_model": {
        "predictors": PREDICTORS,
        "formula_template": FORMULA_TEMPLATE,
        "output_subdir": "full_model",
        "moderators": MODERATORS,
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


def compute_time_squared(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a quadratic `time_sq` term from `time`.

    `time_sq = (time - 50)^2 / 50` lets the additive model fit a parabola
    whose vertex can fall anywhere in [0, 100] (vertex at
    `time = 50 - beta_time / (2 * beta_time_sq)`), capturing asymmetric
    U-shapes that the linear `time` term alone cannot. Centring on 50 before
    squaring reduces collinearity with the linear `time` term; the `/ 50`
    scaling puts `time_sq` on the same 0-50 range as `valence_sq`. Call
    `orthogonalize_quadratic(df, "time_sq", "time")` afterwards to fully
    remove the residual correlation (global poly-2 parametrization).

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
    curvature (U-shaped) relationship with the markers beyond its linear
    effect. Centring on 50 follows the same "neutral midpoint" convention as
    `time_sq` (compute_time_squared). The `/ 50` scaling puts `valence_sq` on
    the same 0-50 range as `time_sq`. Call
    `orthogonalize_quadratic(df, "valence_sq", "valence")` afterwards to fully
    remove the residual correlation (global poly-2 parametrization).

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


def orthogonalize_quadratic(
    df: pd.DataFrame, quad_col: str, linear_col: str
) -> tuple[pd.DataFrame, float, float]:
    """Globally orthogonalize a quadratic term against its linear counterpart.

    Fits a pooled OLS regression `quad_col ~ linear_col` across all rows and
    subtracts the fitted values, leaving only the curvature component. This is
    the poly(x, 2) parametrization used in R and the EEG/CBPT pipeline, and
    ensures that the quadratic term's coefficient is unconfounded by the linear
    term's range (lower VIF). By the invariance of the highest-order polynomial
    term under invertible linear reparametrizations of the fixed-effects design
    matrix, the quadratic term's LMM estimate, SE, t, and p are unchanged —
    only the linear term's SE is affected (reduced collinearity).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing `quad_col` and `linear_col`.
    quad_col : str
        Name of the quadratic column to orthogonalize in-place (e.g.
        ``"time_sq"`` or ``"valence_sq"``).
    linear_col : str
        Name of the linear column to regress out (e.g. ``"time"`` or
        ``"valence"``).

    Returns
    -------
    df : pd.DataFrame
        Input dataframe with `quad_col` replaced by the residualised version.
    intercept : float
        OLS intercept of `quad_col ~ linear_col` (saved for transparency).
    slope : float
        OLS slope of `quad_col ~ linear_col` (saved for transparency).
    """
    slope, intercept, *_ = stats.linregress(df[linear_col], df[quad_col])
    df[quad_col] = df[quad_col] - (intercept + slope * df[linear_col])
    return df, float(intercept), float(slope)


def standardize_predictors(
    df: pd.DataFrame, predictors: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Z-score predictor columns in-place (pooled across all rows: mean=0, SD=1).

    Called on a *copy* of the dataframe so the original remains in original units
    for scatter visualisation. The returned SDs allow the caller to denormalise
    LMM coefficients for marginal-line computation on the original-scale data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with predictor columns to standardise.
    predictors : list[str]
        Column names to z-score.

    Returns
    -------
    df : pd.DataFrame
        Same dataframe with predictor columns replaced by their z-scores.
    predictor_sds : dict[str, float]
        Mapping ``{predictor: pooled_sd}`` — used for β denormalisation.
    """
    predictor_sds: dict[str, float] = {}
    for col in predictors:
        if col not in df.columns:
            continue
        m = float(df[col].mean())
        s = float(df[col].std(ddof=0))
        predictor_sds[col] = s
        if s > 0:
            df[col] = (df[col] - m) / s
    return df, predictor_sds


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
    diagnostics_dir: Path | None = None,
    diagnostics_label: str | None = None,
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
        Defaults to the module-level `PREDICTORS`.

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

    # Residual diagnostics — the check this pipeline previously never performed.
    if diagnostics_dir is not None:
        label = diagnostics_label or f"gaussian_{marker}"
        save_diagnostic_plots(
            fitted=result.fittedvalues.values,
            residuals=result.resid.values,
            label=label,
            output_dir=Path(diagnostics_dir),
            n_bins=int(GLMM_CONFIG["diagnostics"]["n_residual_bins"]),
        )
        summary = gaussian_residual_diagnostics(
            result.resid.values, result.fittedvalues.values
        )
        summary.update({"model": label, "family": "gaussian"})
        pd.DataFrame([summary]).to_csv(
            Path(diagnostics_dir) / f"{label}_residual_summary.csv", index=False
        )

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


# Scale that coefficients live on, per link function. Empty for identity, where
# the coefficient is already in the marker's own units.
LINK_SCALE_LABELS: dict[str, str] = {
    "identity": "",
    "logit": "log-odds",
    "log": "log",
}


def coefficient_axis_label(link: str, standardized: bool) -> str:
    """Axis label stating the scale the plotted coefficients are on.

    A GLMM β is not in the marker's units: binomial coefficients are log-odds
    and Gamma coefficients are log. Labelling them as bare coefficients invites
    the reader to interpret a β of -0.49 as a half-unit drop in error rate,
    which it is not.

    Parameters
    ----------
    link : str
        ``"identity"``, ``"logit"`` or ``"log"``.
    standardized : bool
        Whether predictors were z-scored (β is then "per SD").

    Returns
    -------
    str
        Axis label.
    """
    scale = LINK_SCALE_LABELS[link]
    if standardized:
        return f"β ({scale} per SD, 95% CI)" if scale else "β (SD units, 95% CI)"
    return f"Coefficient ({scale}, 95% CI)" if scale else "Coefficient (95% CI)"


def _draw_forest_panel(
    ax: "plt.Axes",
    plot_df: pd.DataFrame,
    labels: list[str],
    colors: list[str],
    sig_flags: list[bool],
    ci_lower_col: str,
    ci_upper_col: str,
    xlabel: str,
    title: str,
    yticks_right: bool = False,
) -> None:
    """Draw one forest panel onto *ax*.

    Shared by ``plot_combined_forest`` (standalone) and
    ``plot_combined_panel`` (embedded). Significance encoded by fill
    (filled = significant, hollow = not) and line style (solid / dashed).

    Parameters
    ----------
    ax : plt.Axes
    plot_df : pd.DataFrame
        One row per predictor/moderator with 'estimate', 't_value', 'p_fdr',
        'significant_fdr', and the CI columns named by ``ci_lower_col`` /
        ``ci_upper_col``.
    labels, colors, sig_flags : pre-computed lists aligned to ``plot_df``.
    yticks_right : bool
        If True, place y-tick labels on the right side of the axis (used when
        the panel is in the rightmost column of the combined figure).
    """
    dash_style = (0, (4, 2))
    for i, (_, row) in enumerate(plot_df.iterrows()):
        c = colors[i]
        sig = sig_flags[i]
        lo, hi, est = float(row[ci_lower_col]), float(row[ci_upper_col]), float(row["estimate"])
        ax.plot([lo, hi], [i, i], color=c, linewidth=2.5,
                linestyle="-" if sig else dash_style, alpha=0.9, zorder=2)
        for x_cap in (lo, hi):
            ax.plot([x_cap, x_cap], [i - 0.12, i + 0.12],
                    color=c, linewidth=2.0, alpha=0.9, zorder=2)
        ax.plot(est, i, marker="o", markersize=10,
                markerfacecolor=c if sig else "white",
                markeredgecolor=c, markeredgewidth=2.2, zorder=3)
        p_fdr = row.get("p_fdr", np.nan)
        p_label = f"p={p_fdr:.3f}" if pd.notna(p_fdr) else ""
        star = "✱ " if sig else ""
        # Wald z for both backends: statsmodels MixedLM and lme4 glmer both
        # report a normal-approximation statistic, not a t.
        ax.text(
            hi, i + 0.22,
            f"{star}z={row['t_value']:.2f}  {p_label}",
            va="bottom", fontsize=8.5, color=c,
            fontweight="bold" if sig else "normal",
        )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    for tick, c in zip(ax.get_yticklabels(), colors):
        tick.set_color(c)
        tick.set_fontweight("bold")
    if yticks_right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.spines["top"].set_visible(False)


def plot_combined_forest(
    additive_df: pd.DataFrame,
    moderation_df: pd.DataFrame | None,
    marker: str,
    output_dir: Path,
    figsize_override: tuple[float, float] | None = None,
    out_suffix: str = "",
    predictors: list[str] | None = None,
    standardized: bool = False,
    link: str = "identity",
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
        module-level `PREDICTORS`.
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

    _draw_forest_panel(
        ax_add, add_plot, add_labels, add_colors, add_sig,
        ci_lower_col="conf_lower", ci_upper_col="conf_upper",
        xlabel=coefficient_axis_label(link, standardized),
        title="Additive Effects",
    )
    if mod_plot is not None:
        _draw_forest_panel(
            ax_mod, mod_plot, mod_labels, mod_colors, mod_sig,
            ci_lower_col="ci_lower", ci_upper_col="ci_upper",
            xlabel=(
                "onoff × moderator "
                f"({LINK_SCALE_LABELS[link]}, 95% CI)"
                if LINK_SCALE_LABELS[link]
                else "onoff × moderator coefficient (95% CI)"
            ),
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


def inverse_link(eta: np.ndarray, link: str) -> np.ndarray:
    """Map a linear predictor back to the scale of the observed response.

    GLMM coefficients live on the link scale (log-odds for binomial, log for
    Gamma), while the binscatter panels plot the marker in its original units.
    Drawing the marginal line without this transform puts it at, for example,
    log(0.25) = -1.39 while the data sit at 0.25 -- the line and the cloud end
    up on entirely different parts of the axis.

    Parameters
    ----------
    eta : np.ndarray
        Linear predictor.
    link : str
        One of ``"identity"`` (Gaussian tracks), ``"log"`` (Gamma) or
        ``"logit"`` (binomial).

    Returns
    -------
    np.ndarray
        Predicted values on the response scale.

    Raises
    ------
    ValueError
        If the link name is not recognised.
    """
    if link == "identity":
        return eta
    if link == "log":
        return np.exp(eta)
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-eta))
    raise ValueError(f"Unknown link: {link!r}")


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
    predictor_sds: dict[str, float] | None = None,
    link: str = "identity",
) -> np.ndarray:
    """LMM marginal prediction line for one predictor, others held at their mean.

    The plotted line must pass through the binned data cloud. Using the bare
    ``intercept + slope * x`` is wrong for a single-predictor panel because the
    multivariate intercept is the prediction when *every* predictor is 0, so the
    line is vertically offset by the (ignored) contribution of the other
    predictors.

    **Non-standardised model** (``predictor_sds`` is None / empty):
        ŷ(x) = intercept + Σ_{k≠p} β_k · mean(x_k) + β_p · x

    **Standardised model** (``predictor_sds`` provided, ``data`` in original
    units):
        The standardised model intercept β₀_std = β₀_orig + Σ_k β_k_orig · m_k,
        so holding all other predictors at their sample means collapses to:
            ŷ(x) = β₀_std + (β_p_std / SD_p) · (x − m_p)
        eliminating the explicit "others at mean" sum.

    Parameters
    ----------
    results_df : pd.DataFrame
        `fit_lmm` output (must contain 'predictor' and 'estimate', incl. the
        'Intercept' row).
    data : pd.DataFrame
        *Original*-scale data (used for predictor means and OLS re-fit for _sq
        distance conversion); NOT the z-scored fitting copy.
    predictor : str
        The predictor on the x axis of this panel.
    x_range : np.ndarray
        X values at which to evaluate the line. For ``_sq`` predictors this is
        in ``|base − 50|`` space (distance from neutral, 0–50).
    predictors : list[str] | None
        Predictor set this model was fit with. Defaults to `PREDICTORS`.
    predictor_sds : dict[str, float] | None
        Pooled SDs used for standardisation (from `standardize_predictors`).
        When provided, coefficients are treated as standardised and are
        denormalised before computing the marginal line.

    Returns
    -------
    np.ndarray
        Predicted marker values along ``x_range``.
    """
    if predictors is None:
        predictors = PREDICTORS
    if predictor_sds is None:
        predictor_sds = {}

    coef = dict(zip(results_df["predictor"], results_df["estimate"]))
    intercept = coef.get("Intercept", 0.0)

    # Every branch below builds eta, the LINEAR PREDICTOR. For a GLMM that is on
    # the link scale (log-odds / log), not the scale of the plotted data, so the
    # inverse link is applied once at the end.
    if predictor_sds:
        # --- Standardised model: β in SD units; data in original units ----------
        def β_orig(k: str) -> float:
            sd = predictor_sds.get(k, 1.0)
            return coef.get(k, 0.0) / (sd if sd > 0 else 1.0)

        if predictor.endswith("_sq"):
            # x_range is |base − 50| (original units, 0–50).
            # E[quad_orth | dist=d] = d²/50 − (a + b·50)
            base_pred = predictor[:-3]
            orig_quad = (data[base_pred] - 50.0) ** 2 / 50.0
            ols_slope, ols_intercept, *_ = stats.linregress(data[base_pred], orig_quad)
            quad_orth_at_d = x_range ** 2 / 50.0 - (ols_intercept + ols_slope * 50.0)
            m_base = float(data[base_pred].mean())
            eta = (
                intercept
                + β_orig(base_pred) * (50.0 - m_base)
                + β_orig(predictor) * quad_orth_at_d
            )
        else:
            m_p = float(data[predictor].mean())
            eta = intercept + β_orig(predictor) * (x_range - m_p)

    elif predictor.endswith("_sq"):
        # --- Non-standardised, quadratic: x_range is |base_pred − 50| space -----
        base_pred = predictor[:-3]
        orig_quad = (data[base_pred] - 50.0) ** 2 / 50.0
        ols_slope, ols_intercept, *_ = stats.linregress(data[base_pred], orig_quad)
        quad_orth_at_d = x_range ** 2 / 50.0 - (ols_intercept + ols_slope * 50.0)
        others_offset = sum(
            coef.get(k, 0.0) * float(data[k].mean())
            for k in predictors
            if k not in (predictor, base_pred) and k in data.columns
        )
        eta = (
            intercept + others_offset
            + coef.get(base_pred, 0.0) * 50.0
            + coef.get(predictor, 0.0) * quad_orth_at_d
        )

    else:
        # --- Non-standardised, linear -------------------------------------------
        slope_p = coef.get(predictor, 0.0)
        others_offset = sum(
            coef.get(k, 0.0) * float(data[k].mean())
            for k in predictors
            if k != predictor and k in data.columns
        )
        eta = intercept + others_offset + slope_p * x_range

    return inverse_link(eta, link)


def plot_scatter_grid(
    data: pd.DataFrame, results_df: pd.DataFrame, marker: str, output_dir: Path,
    show_stats: bool = True,
    out_suffix: str = "",
    predictors: list[str] | None = None,
    predictor_sds: dict[str, float] | None = None,
    link: str = "identity",
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
        module-level `PREDICTORS`.

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

    # Pre-compute |base - 50| columns for _sq predictors so binning uses a
    # meaningful, monotone x-axis (distance from the neutral midpoint, 0–50)
    # rather than the raw orthogonalized residual (sparse at extremes) or the
    # linear predictor (would duplicate that panel's scatter).
    data_plot = data.copy()
    for pred in predictors:
        if pred.endswith("_sq"):
            base = pred[:-3]
            data_plot[f"_dist_{base}"] = (data_plot[base] - 50.0).abs()

    cloud_y: list[float] = []

    for ax_idx, predictor in enumerate(predictors):
        row = ax_idx // n_cols + 1
        col = ax_idx % n_cols + 1
        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)

        if predictor.endswith("_sq"):
            base_pred = predictor[:-3]
            plot_col = f"_dist_{base_pred}"
            x_label = f"|{PREDICTOR_LABELS.get(base_pred, base_pred)} − 50|"
        else:
            plot_col = predictor
            x_label = PREDICTOR_LABELS.get(predictor, predictor)

        pred_row = results_df[results_df["predictor"] == predictor]
        significant = bool(pred_row["significant_fdr"].values[0]) if len(pred_row) > 0 else False

        if plot_col not in data_plot.columns or marker not in data_plot.columns:
            continue

        agg, per_sub = binned_within_subject(data_plot, plot_col, marker, SCATTER_BIN_WIDTH)
        if agg is None:
            continue

        cloud_y.extend(per_sub["y"].tolist())

        fig.add_trace(
            go.Scatter(
                x=per_sub["x_real"], y=per_sub["y"],
                mode="markers",
                marker=dict(color=color, size=4, opacity=0.18),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=agg["x"], y=agg["y_mean"],
                error_y=dict(type="data", array=agg["y_se"], visible=True,
                             thickness=1.6, width=4, color=color),
                mode="markers",
                marker=dict(color=color, size=9, line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"{x_label}: %{{x:.0f}}<br>mean: %{{y:.3f}}<extra></extra>",
            ),
            row=row, col=col,
        )

        # marginal_lmm_line handles _sq predictors in distance space;
        # pass predictor_sds so it can denormalise standardised β
        x_line = np.linspace(float(agg["x"].min()), float(agg["x"].max()), 100)
        y_line = marginal_lmm_line(
            results_df, data, predictor, x_line,
            predictors=predictors, predictor_sds=predictor_sds, link=link,
        )
        fig.add_trace(
            go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=color, width=3, dash="solid" if significant else "dash"),
                showlegend=False, hoverinfo="skip",
            ),
            row=row, col=col,
        )

        fig.update_xaxes(title_text=x_label, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=marker_label, row=row, col=col)

    # β is on the link scale for the GLMM tracks; say so next to the number.
    beta_unit = (
        f" {LINK_SCALE_LABELS[link]}" if LINK_SCALE_LABELS[link] else ""
    )

    axis_idx = 1
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
                f"  β={slope_ann:.3f}{beta_unit}  {p_text_ann}"
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
    fig.write_image(f"{out_base}.svg")
    print(f"Saved: {out_base}.html / .png / .svg")
    return Path(f"{out_base}.png")


def plot_combined_panel(
    data: pd.DataFrame,
    results_df: pd.DataFrame,
    moderation_df: pd.DataFrame | None,
    marker: str,
    output_dir: Path,
    predictors: list[str] | None = None,
    predictor_sds: dict[str, float] | None = None,
    link: str = "identity",
    # Legacy keyword args kept for call-site compatibility — ignored
    width: int = 1700,
    height: int = 920,
    forest_col_weight: float = 1.6,
) -> None:
    """Combined matplotlib figure: binscatter grid (left) + forest panels (right).

    Layout: N_scatter_rows × 5 columns. Columns 1-4 hold one binscatter panel
    per predictor (row-major order); column 5 holds the additive-effects forest
    (row 1) and, if ``moderation_df`` is given, the onoff-moderation forest
    (row 2). Scatter x-labels add a "(linear)" suffix when the predictor has a
    quadratic counterpart in the model, and use a short "Dim²" label for the
    quadratic panels (which plot |base − 50| on the x-axis).

    Parameters
    ----------
    data : pd.DataFrame
        Original-scale probe-level dataframe (scatter + marginal line).
    results_df : pd.DataFrame
        Output of `fit_lmm` for this marker.
    moderation_df : pd.DataFrame | None
        Output of `run_moderation_analysis`. ``None`` → one forest row.
    marker : str
        Dependent variable name.
    output_dir : Path
        Output directory.
    predictors : list[str] | None
        Predictor set. Defaults to `PREDICTORS`.
    predictor_sds : dict[str, float] | None
        Standardisation SDs for marginal-line denormalisation.
    """
    from matplotlib.gridspec import GridSpec

    if predictors is None:
        predictors = PREDICTORS

    n_scatter_cols = 4
    n_preds = len(predictors)
    n_scatter_rows = int(np.ceil(n_preds / n_scatter_cols))
    n_forest_rows = 2 if moderation_df is not None else 1
    n_total_rows = max(n_scatter_rows, n_forest_rows)

    # ── Figure & GridSpec ──────────────────────────────────────────────────────
    fig_w = 4.0 * n_scatter_cols + 4.8
    fig_h = 4.5 * n_total_rows + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        n_total_rows, n_scatter_cols + 1,
        figure=fig,
        width_ratios=[1] * n_scatter_cols + [1.3],
        hspace=0.60, wspace=0.38,
        left=0.06, right=0.98, top=0.92, bottom=0.08,
    )

    # ── x-label helper ────────────────────────────────────────────────────────
    def _scatter_xlabel(pred: str) -> str:
        """Short, unambiguous x-axis label for scatter panels.

        Linear predictors that have a _sq counterpart → "Dim (linear)".
        Quadratic predictors → "Dim²"  (axis is |base − 50|, 0-50 range).
        Others → standard PREDICTOR_LABELS entry.
        """
        if pred.endswith("_sq"):
            base = pred[:-3]
            return PREDICTOR_LABELS.get(base, base) + "²"
        if pred + "_sq" in predictors:
            return PREDICTOR_LABELS.get(pred, pred) + " (linear)"
        return PREDICTOR_LABELS.get(pred, pred)

    # ── Pre-compute |base − 50| distance columns for _sq predictors ───────────
    data_plot = data.copy()
    for pred in predictors:
        if pred.endswith("_sq"):
            base = pred[:-3]
            data_plot[f"_dist_{base}"] = (data_plot[base] - 50.0).abs()

    z_suffix = " (within-subj. z)" if APPLY_WITHIN_SUBJECT_Z else ""
    marker_label = f"{MARKER_LABELS[marker]}{z_suffix}"

    # ── Scatter panels ─────────────────────────────────────────────────────────
    for ax_idx, predictor in enumerate(predictors):
        s_row = ax_idx // n_scatter_cols
        s_col = ax_idx % n_scatter_cols
        ax = fig.add_subplot(gs[s_row, s_col])

        color = PREDICTOR_COLORS.get(predictor, DEFAULT_PREDICTOR_COLOR)
        x_label = _scatter_xlabel(predictor)

        pred_row = results_df[results_df["predictor"] == predictor]
        significant = (
            bool(pred_row["significant_fdr"].values[0]) if len(pred_row) > 0 else False
        )

        plot_col = (
            f"_dist_{predictor[:-3]}" if predictor.endswith("_sq") else predictor
        )
        if plot_col not in data_plot.columns or marker not in data_plot.columns:
            ax.set_visible(False)
            continue

        agg, per_sub = binned_within_subject(data_plot, plot_col, marker, SCATTER_BIN_WIDTH)
        if agg is None:
            ax.set_visible(False)
            continue

        # Background cloud (per-subject bin means)
        ax.scatter(
            per_sub["x_real"], per_sub["y"],
            color=color, alpha=0.15, s=12, zorder=1, linewidths=0,
        )
        # Across-subject mean ± SE per bin
        ax.errorbar(
            agg["x"], agg["y_mean"], yerr=agg["y_se"],
            fmt="o", color=color, markersize=7, capsize=3,
            linewidth=1.5, markeredgewidth=0.5, zorder=2,
        )
        # LMM marginal line (denormalised when standardised)
        x_line = np.linspace(float(agg["x"].min()), float(agg["x"].max()), 100)
        y_line = marginal_lmm_line(
            results_df, data, predictor, x_line,
            predictors=predictors, predictor_sds=predictor_sds, link=link,
        )
        ax.plot(
            x_line, y_line, color=color, lw=2.5,
            linestyle="-" if significant else "--", zorder=3,
        )

        ax.set_title(x_label, fontsize=11, color=color, fontweight="bold", pad=5)
        ax.set_xlabel(x_label, fontsize=10, color=color, fontweight="bold")
        if s_col == 0:
            ax.set_ylabel(marker_label, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide leftover empty scatter cells
    for ax_idx in range(n_preds, n_scatter_rows * n_scatter_cols):
        s_row = ax_idx // n_scatter_cols
        s_col = ax_idx % n_scatter_cols
        fig.add_subplot(gs[s_row, s_col]).set_visible(False)

    # ── Forest panels ──────────────────────────────────────────────────────────
    add_plot = (
        results_df[results_df["predictor"].isin(predictors)]
        .copy()
        .sort_values("estimate", ascending=True)
        .reset_index(drop=True)
    )
    add_labels = [PREDICTOR_LABELS.get(p, p) for p in add_plot["predictor"]]
    add_colors = [PREDICTOR_COLORS.get(p, DEFAULT_PREDICTOR_COLOR) for p in add_plot["predictor"]]
    add_sig = [bool(s) for s in add_plot["significant_fdr"]]
    coef_label = coefficient_axis_label(link, bool(predictor_sds))

    ax_add = fig.add_subplot(gs[0, -1])
    _draw_forest_panel(
        ax_add, add_plot, add_labels, add_colors, add_sig,
        ci_lower_col="conf_lower", ci_upper_col="conf_upper",
        xlabel=coef_label, title="Additive Effects",
        yticks_right=True,
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
        mod_labels = [PREDICTOR_LABELS.get(m, m) for m in mod_plot["moderator"]]
        mod_colors = [
            PREDICTOR_COLORS.get(m, DEFAULT_PREDICTOR_COLOR) for m in mod_plot["moderator"]
        ]
        mod_sig = [bool(s) for s in mod_plot["significant_fdr"]]

        ax_mod = fig.add_subplot(gs[1, -1])
        _draw_forest_panel(
            ax_mod, mod_plot, mod_labels, mod_colors, mod_sig,
            ci_lower_col="ci_lower", ci_upper_col="ci_upper",
            xlabel=(
                f"onoff × moderator ({LINK_SCALE_LABELS[link]}, 95% CI)"
                if LINK_SCALE_LABELS[link]
                else "onoff × moderator (95% CI)"
            ),
            title="onoff Moderation",
            yticks_right=True,
        )

    fig.suptitle(
        MARKER_LABELS[marker], fontsize=15, fontweight="bold", y=0.97,
    )

    out_base = output_dir / f"{marker}_combined_panel"
    fig.savefig(f"{out_base}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(f"{out_base}.svg", bbox_inches="tight")
    plt.close(fig)
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
    moderation_df: pd.DataFrame, marker: str, output_dir: Path,
    link: str = "identity",
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
    link_suffix = (
        f" ({LINK_SCALE_LABELS[link]})" if LINK_SCALE_LABELS[link] else ""
    )
    ax.set_xlabel(f"Interaction Coefficient: onoff × moderator{link_suffix} (95% CI)",
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
    backend: str = "gaussian",
) -> pd.DataFrame:
    """Run all moderation models and produce a FDR-corrected summary.

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
        LMM optimisation method (Gaussian backend only).
    maxiter : int
        Maximum iterations (Gaussian backend only).
    output_dir : Path
        Output directory for CSVs and figures.
    backend : str
        ``"glmm"`` fits binomial/Gamma models via lme4; anything else uses the
        Gaussian statsmodels backend. Keeping the estimand consistent with the
        additive model matters — mixing families within one paper section is
        not defensible.

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
            if backend == "glmm":
                row = fit_moderation_glmm(
                    data=data, marker=marker, moderator=moderator,
                    config=GLMM_CONFIG,
                    marker_spec=GLMM_CONFIG["markers"][marker],
                )
            else:
                row = fit_moderation_lmm(
                    data, marker, moderator, method, maxiter
                )
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

def build_track_specs() -> list[dict]:
    """Single definition of the four model tracks.

    Shared by ``run_pipeline_for_dataset`` and ``plots_only`` so the fitting
    and the figure-regeneration paths can never drift apart on which tracks
    exist, where they are written, or which markers each one contains.

    Returns
    -------
    list[dict]
        One entry per track with keys ``name``, ``output_subdir``,
        ``backend``, ``olre``, ``markers`` and ``transform``.
    """
    olre_markers = [
        m for m in OBJECTIVE_MARKERS
        if GLMM_CONFIG["markers"][m].get("olre", False)
    ]
    return [
        {"name": "glmm", "output_subdir": "glmm", "backend": "glmm",
         "olre": False, "markers": list(OBJECTIVE_MARKERS), "transform": False},
        {"name": "sensitivity_olre", "output_subdir": "sensitivity_olre",
         "backend": "glmm", "olre": True, "markers": olre_markers,
         "transform": False},
        {"name": "sensitivity_gaussian", "output_subdir": "sensitivity_gaussian",
         "backend": "gaussian", "olre": False,
         "markers": list(OBJECTIVE_MARKERS) + ["total_errors_legacy"],
         "transform": False},
        {"name": "sensitivity_transformed",
         "output_subdir": "sensitivity_transformed", "backend": "gaussian",
         "olre": False, "markers": list(OBJECTIVE_MARKERS), "transform": True},
    ]


def marker_link(marker: str, backend: str) -> str:
    """Link function used to plot this marker's marginal line.

    The Gaussian and transformed tracks are fitted with an identity link, so
    their coefficients are already on the plotted scale. The GLMM track is not:
    binomial coefficients are log-odds and Gamma coefficients are log, and the
    marginal line has to be mapped back before it can be drawn over the data.

    Parameters
    ----------
    marker : str
        Marker name.
    backend : str
        ``"glmm"`` or the Gaussian backend.

    Returns
    -------
    str
        ``"identity"``, ``"logit"`` or ``"log"``.
    """
    if backend != "glmm":
        return "identity"
    family = GLMM_CONFIG["markers"][marker]["family"]
    return "logit" if family == "binomial" else "log"


def apply_response_transforms(
    df: pd.DataFrame, markers: list[str], glmm_config: dict
) -> pd.DataFrame:
    """Return a copy of *df* with each marker replaced by its transformed form.

    Used by the ``sensitivity_transformed`` track: ``log`` for ``rtcv`` and the
    Haldane-corrected empirical logit for the rate markers, which stays finite
    at the 65% of probes with zero events.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level dataframe with raw markers and count columns.
    markers : list[str]
        Markers to transform.
    glmm_config : dict
        Parsed ``glmm_config.yaml``.

    Returns
    -------
    pd.DataFrame
        Copy with transformed marker columns.

    Raises
    ------
    ValueError
        If a transform type is not recognised.
    """
    out = df.copy()
    correction = glmm_config["empirical_logit_correction"]
    for marker in markers:
        spec = glmm_config["transforms"][marker]
        if spec["type"] == "empirical_logit":
            valid = out[spec["total_col"]] > 0
            out.loc[~valid, marker] = np.nan
            out.loc[valid, marker] = empirical_logit(
                out.loc[valid, spec["success_col"]].values,
                out.loc[valid, spec["total_col"]].values,
                correction=correction,
            )
        elif spec["type"] == "log":
            valid = out[spec["response_col"]] > 0
            out.loc[~valid, marker] = np.nan
            out.loc[valid, marker] = log_transform(
                out.loc[valid, spec["response_col"]].values
            )
        else:
            raise ValueError(f"Unknown transform type: {spec['type']}")
    return out


def build_model_comparison(
    track_results: dict[str, dict[str, pd.DataFrame]],
    predictors: list[str],
) -> pd.DataFrame:
    """Assemble the side-by-side specification comparison table.

    One row per marker x predictor, with the estimate, p-value and FDR
    significance from each track. This is the artefact that lets the manuscript
    state whether conclusions are robust to specification -- or show exactly
    where they are not. ``total_errors_legacy`` is excluded because its
    different marker definition makes it non-comparable.

    Parameters
    ----------
    track_results : dict[str, dict[str, pd.DataFrame]]
        Mapping ``{track_name: {marker: fit results}}``.
    predictors : list[str]
        Predictors to include.

    Returns
    -------
    pd.DataFrame
        Wide comparison table, one row per marker x predictor.
    """
    rows: list[dict] = []
    for track, per_marker in track_results.items():
        for marker, results_df in per_marker.items():
            if marker == "total_errors_legacy":
                continue
            subset = results_df[results_df["predictor"].isin(predictors)]
            for _, r in subset.iterrows():
                rows.append({
                    "marker": marker,
                    "predictor": r["predictor"],
                    "track": track,
                    "estimate": r["estimate"],
                    "p_value": r["p_value"],
                    "p_fdr": r.get("p_fdr", np.nan),
                    "significant_fdr": r.get("significant_fdr", False),
                })
    long_df = pd.DataFrame(rows)
    wide = long_df.pivot_table(
        index=["marker", "predictor"], columns="track",
        values=["estimate", "p_fdr", "significant_fdr"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}__{track}" for stat, track in wide.columns]
    return wide.reset_index()


def _fit_predictor_set(
    df: pd.DataFrame, set_name: str, set_cfg: dict, output_dir: Path,
    backend: str = "gaussian", olre: bool = False,
    markers: list[str] | None = None,
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
        (time_on_task, time_sq⊥, valence_sq⊥).
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

    # --- Standardise predictors for fitting (original df kept for plots) --------
    pred_sds: dict[str, float] = {}
    if STANDARDIZE_PREDICTORS:
        df_fit, pred_sds = standardize_predictors(df.copy(), predictors)
        sds_df = pd.DataFrame(
            [{"predictor": k, "sd": v} for k, v in pred_sds.items()]
        )
        sds_path = set_dir / "predictor_standardization.csv"
        sds_df.to_csv(sds_path, index=False)
        print(f"\nPredictor SDs saved: {sds_path}")
        print(sds_df.to_string(index=False))
    else:
        df_fit = df

    # --- Fit additive model per marker (backend-dispatched) ---
    # Diagnostics for every track land in one shared directory so the residual
    # behaviour of the competing specifications can be compared side by side.
    diagnostics_dir = output_dir / GLMM_CONFIG["diagnostics"]["output_subdir"]
    track_tag = set_cfg["output_subdir"]

    fit_markers = markers if markers is not None else OBJECTIVE_MARKERS
    results: dict[str, pd.DataFrame] = {}
    for marker in fit_markers:
        if backend == "glmm":
            results_df = fit_glmm(
                data=df_fit,
                marker=marker,
                predictors=predictors,
                config=GLMM_CONFIG,
                marker_spec=GLMM_CONFIG["markers"][marker],
                olre=olre,
                mcc_method=MCC_METHOD,
                mcc_alpha=MCC_ALPHA,
                diagnostics_dir=diagnostics_dir,
            )
        else:
            results_df = fit_lmm(
                data=df_fit,
                marker=marker,
                formula_template=set_cfg["formula_template"],
                method=LMM_METHOD,
                maxiter=LMM_MAXITER,
                predictors=predictors,
                diagnostics_dir=diagnostics_dir,
                diagnostics_label=f"{track_tag}_{marker}",
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
            data=df_fit,
            markers=fit_markers,
            moderators=set_cfg["moderators"],
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
            output_dir=set_dir,
            backend=backend,
        )

    # --- Plots (original-scale df; β denormalised via pred_sds) ---
    is_std = STANDARDIZE_PREDICTORS
    print(f"\nGenerating plots for {set_name}...")
    for marker, results_df in results.items():
        link = marker_link(marker, backend)
        plot_scatter_grid(
            df, results_df, marker, set_dir,
            predictors=predictors, predictor_sds=pred_sds, link=link,
        )
        plot_combined_forest(
            results_df, moderation_df, marker, set_dir,
            predictors=predictors, standardized=is_std, link=link,
        )
        plot_combined_panel(
            df, results_df, moderation_df, marker, set_dir,
            predictors=predictors, predictor_sds=pred_sds, link=link,
        )

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
    # New definition (all three tracks): a genuine proportion of all trials.
    # The old sum-of-two-rates mixed denominators (~27 go vs ~3 no-go trials)
    # and was a proportion of nothing. See the design spec, section 2.
    df_markers["_total_error_count"] = (
        df_markers["n_omissions"] + df_markers["n_commissions"]
    )
    df_markers["total_errors"] = (
        df_markers["_total_error_count"] / df_markers["n_trials_window"]
    )
    # Retained only for continuity with already-published Gaussian output.
    df_markers["total_errors_legacy"] = (
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
    df = compute_time_squared(df)
    df = compute_valence_squared(df)
    df, time_sq_intercept, time_sq_slope = orthogonalize_quadratic(df, "time_sq", "time")
    df, valence_sq_intercept, valence_sq_slope = orthogonalize_quadratic(df, "valence_sq", "valence")
    orth_df = pd.DataFrame([
        {"term": "time_sq", "linear_col": "time",
         "intercept": time_sq_intercept, "slope": time_sq_slope},
        {"term": "valence_sq", "linear_col": "valence",
         "intercept": valence_sq_intercept, "slope": valence_sq_slope},
    ])
    orth_path = output_dir / "quadratic_orthogonalization.csv"
    orth_df.to_csv(orth_path, index=False)
    print(f"\nQuadratic orthogonalization saved: {orth_path}")
    print(orth_df.to_string(index=False))

    # --- Within-subject z-scoring (same as objective_markers_analysis.py) ----
    if APPLY_WITHIN_SUBJECT_Z:
        print("\nApplying within-subject z-scoring to objective markers...")
        df = apply_within_subject_z_scoring(df, OBJECTIVE_MARKERS)

    # --- Fit + plot unified full_model ----------------------------------------
    track_results: dict[str, dict[str, pd.DataFrame]] = {}
    for set_name, set_cfg in PREDICTOR_SETS.items():
        for track in build_track_specs():
            track_df = (
                apply_response_transforms(df, OBJECTIVE_MARKERS, GLMM_CONFIG)
                if track["transform"] else df
            )
            track_cfg = dict(set_cfg, output_subdir=track["output_subdir"])
            track_results[track["name"]] = _fit_predictor_set(
                track_df, set_name, track_cfg, output_dir,
                backend=track["backend"], olre=track["olre"],
                markers=track["markers"],
            )

    comparison_path = output_dir / "model_comparison.csv"
    build_model_comparison(track_results, PREDICTORS).to_csv(
        comparison_path, index=False
    )
    print(f"\nModel comparison saved: {comparison_path}")

    used_config_path = output_dir / "used_config.yaml"
    yaml.safe_dump(GLMM_CONFIG, open(used_config_path, "w"), sort_keys=False)
    print(f"Config snapshot saved: {used_config_path}")

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
        to all of them (`["full_model"]`).
    """
    if predictor_set_names is None:
        predictor_set_names = list(PREDICTOR_SETS.keys())

    for dataset_name, markers_path in OBJECTIVE_MARKERS_PATHS.items():
        output_dir = OUTPUT_BASE_DIR / dataset_name
        print(f"\n### PLOTS-ONLY for {dataset_name.upper()} — loading CSVs from {output_dir}")

        # --- Reconstruct data (needed for binscatter and marginal line) -------
        df_markers = pd.read_csv(markers_path)
        # Must mirror run_pipeline_for_dataset exactly, or the binscatter cloud
        # would be built from a different marker than the one that was fitted.
        df_markers["_total_error_count"] = (
            df_markers["n_omissions"] + df_markers["n_commissions"]
        )
        df_markers["total_errors"] = (
            df_markers["_total_error_count"] / df_markers["n_trials_window"]
        )
        df_markers["total_errors_legacy"] = (
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
        df = compute_time_squared(df)
        df = compute_valence_squared(df)
        df, _, _ = orthogonalize_quadratic(df, "time_sq", "time")
        df, _, _ = orthogonalize_quadratic(df, "valence_sq", "valence")
        if APPLY_WITHIN_SUBJECT_Z:
            df = apply_within_subject_z_scoring(df, OBJECTIVE_MARKERS)

        # --- Regenerate plots for every track of each predictor set -----------
        df_transformed = apply_response_transforms(
            df, OBJECTIVE_MARKERS, GLMM_CONFIG
        )

        for set_name in predictor_set_names:
            set_cfg = PREDICTOR_SETS[set_name]
            set_predictors = set_cfg["predictors"]

            for track in build_track_specs():
                set_dir = output_dir / track["output_subdir"]
                if not set_dir.exists():
                    print(f"  --- skipping {track['name']}: {set_dir} absent ---")
                    continue
                print(f"  --- {set_name} / {track['name']} → {set_dir} ---")

                track_df = df_transformed if track["transform"] else df

                results_for_set: dict[str, pd.DataFrame] = {}
                for marker in track["markers"]:
                    csv_path = set_dir / f"{marker}_lmm_results.csv"
                    results_for_set[marker] = pd.read_csv(csv_path)

                moderation_df = None
                mod_path = set_dir / "moderation_summary.csv"
                if set_cfg["with_moderation"] and mod_path.exists():
                    moderation_df = pd.read_csv(mod_path)

                # Predictor SDs saved during fitting (for marginal-line denorm.)
                pred_sds: dict[str, float] = {}
                sds_path = set_dir / "predictor_standardization.csv"
                if sds_path.exists():
                    sds_df = pd.read_csv(sds_path)
                    pred_sds = dict(zip(sds_df["predictor"], sds_df["sd"]))

                is_std = bool(pred_sds)

                for marker, results_df in results_for_set.items():
                    link = marker_link(marker, track["backend"])
                    plot_scatter_grid(
                        track_df, results_df, marker, set_dir,
                        predictors=set_predictors, predictor_sds=pred_sds,
                        link=link,
                    )
                    plot_combined_forest(
                        results_df, moderation_df, marker, set_dir,
                        standardized=is_std, predictors=set_predictors,
                        link=link,
                    )
                    plot_combined_panel(
                        track_df, results_df, moderation_df, marker, set_dir,
                        predictors=set_predictors, predictor_sds=pred_sds,
                        link=link,
                    )

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
            "With --plots-only, restrict regeneration to one predictor set "
            "(currently only 'full_model'), or 'all' (default, regenerates "
            "all sets in PREDICTOR_SETS)."
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
