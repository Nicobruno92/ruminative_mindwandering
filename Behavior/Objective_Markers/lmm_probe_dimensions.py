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

Preprocessing mirrors objective_markers_analysis.py:
  - Merge objective markers with probe-level metadata (subject/task/probe)
  - Derive time_on_task = probe_number + 15 * (sart_number - 1)
  - Within-subject z-scoring of objective markers (APPLY_WITHIN_SUBJECT_Z)

Outputs (saved under OUTPUT_DIR):
  - <marker>_lmm_results.csv       — additive model results + FDR
  - <marker>_forest_plot.png        — coefficient forest plot
  - <marker>_scatter_grid.png       — scatter grid per predictor
  - moderation_summary.csv          — interaction term stats, FDR-corrected
  - moderation_forest_<marker>.png  — forest of interaction coefficients
  - interaction_<marker>_<mod>.png  — interaction plot per significant pair

Author: Nicolas Bruno
"""

import os
from pathlib import Path

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
OBJECTIVE_MARKERS_PATH = (
    "results/Behavior/objective_markers/objective_markers_per_probe.csv"
)
PROBE_DATA_PATH = (
    "results/Behavior/probe_data/probe_level_aggregated_data.csv"
)

# Output directory
OUTPUT_DIR = Path(
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
PREDICTORS: list[str] = ["onoff", "valence", "selfother", "time", "confidence", "time_on_task"]

PREDICTOR_LABELS: dict[str, str] = {
    "onoff": "On/Off Task",
    "valence": "Valence",
    "selfother": "Self/Other",
    "time": "Time (probe dim.)",
    "confidence": "Confidence",
    "time_on_task": "Time on Task",
}

# LMM formula template — marker name is substituted at runtime
FORMULA_TEMPLATE = (
    "{marker} ~ onoff + valence + selfother + time + confidence"
    " + time_on_task + (1|subject)"
)

# Potential moderators of the onoff effect (all probe dimensions except onoff)
# Each will be tested in: marker ~ onoff * moderator + (1|subject)
MODERATORS: list[str] = ["valence", "selfother", "time", "confidence", "time_on_task"]

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
APPLY_WITHIN_SUBJECT_Z: bool = True

# Figure settings
FIG_DPI: int = 300
PALETTE_SIGNIFICANT = "#E74C3C"   # red — significant after FDR
PALETTE_NOT_SIGNIFICANT = "#7F8C8D"  # grey — not significant
# Interaction plot colours: Low/High moderator
PALETTE_MOD_LOW = "#2E86AB"   # blue — low moderator
PALETTE_MOD_HIGH = "#F24236"  # red — high moderator

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

    Returns
    -------
    pd.DataFrame
        One row per fixed effect with columns:
        predictor, estimate, std_error, t_value, p_value, conf_lower,
        conf_upper, p_fdr, significant_fdr.
    """
    fixed_formula = parse_lmm_formula(formula_template, marker)

    # Columns needed for this model
    needed_cols = ["subject", marker] + PREDICTORS
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
    predictor_mask = results_df["predictor"].isin(PREDICTORS)
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
    moderation_df: pd.DataFrame,
    marker: str,
    output_dir: Path,
) -> None:
    """Combined forest plot: additive effects (left) + moderation terms (right).

    Left panel shows fixed-effect coefficients from the full additive model.
    Right panel shows the `onoff:moderator` interaction coefficient for each
    potential moderator. Both panels use the same colour coding:
    red = significant after FDR, grey = not significant.

    Parameters
    ----------
    additive_df : pd.DataFrame
        Output of `fit_lmm` for this marker (additive model results).
    moderation_df : pd.DataFrame
        Output of `run_moderation_analysis`, full table (all markers), with
        `p_fdr` and `significant_fdr` columns.
    marker : str
        Marker name (used in figure title and filename).
    output_dir : Path
        Output directory.
    """
    # --- Prepare additive panel data ---
    add_plot = additive_df[additive_df["predictor"].isin(PREDICTORS)].copy()
    add_plot = add_plot.sort_values("estimate", ascending=True).reset_index(drop=True)
    add_labels = [PREDICTOR_LABELS.get(p, p) for p in add_plot["predictor"]]
    add_colors = [
        PALETTE_SIGNIFICANT if sig is True else PALETTE_NOT_SIGNIFICANT
        for sig in add_plot["significant_fdr"].tolist()
    ]

    # --- Prepare moderation panel data ---
    mod_plot = moderation_df[
        moderation_df["marker"] == marker
    ].copy().sort_values("estimate", ascending=True).reset_index(drop=True)
    mod_plot["ci_lower"] = mod_plot["estimate"] - 1.96 * mod_plot["std_error"]
    mod_plot["ci_upper"] = mod_plot["estimate"] + 1.96 * mod_plot["std_error"]
    mod_labels = [PREDICTOR_LABELS.get(m, m) for m in mod_plot["moderator"]]
    mod_colors = [
        PALETTE_SIGNIFICANT if sig is True else PALETTE_NOT_SIGNIFICANT
        for sig in mod_plot["significant_fdr"].tolist()
    ]

    # --- Build figure (two panels, shared colour legend) ---
    fig, (ax_add, ax_mod) = plt.subplots(
        1, 2, figsize=(16, max(5, max(len(add_labels), len(mod_labels)) * 0.65 + 2.5)),
        constrained_layout=True,
    )

    # Helper to draw one panel
    def _draw_panel(ax, plot_df, labels, colors, ci_lower_col, ci_upper_col,
                    xlabel, title):
        for i, (_, row) in enumerate(plot_df.iterrows()):
            ax.errorbar(
                x=row["estimate"], y=i,
                xerr=[[row["estimate"] - row[ci_lower_col]],
                      [row[ci_upper_col] - row["estimate"]]],
                fmt="o", markersize=8, color=colors[i],
                capsize=5, capthick=2, linewidth=2, elinewidth=2,
            )
            p_fdr = row.get("p_fdr", np.nan)
            p_label = f"p={p_fdr:.3f}" if pd.notna(p_fdr) else ""
            ax.text(
                row[ci_upper_col],
                i + 0.25,
                f"t={row['t_value']:.2f}  {p_label}",
                va="bottom", fontsize=8.5, color=colors[i],
            )
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, axis="x", alpha=0.3)

    _draw_panel(
        ax_add, add_plot, add_labels, add_colors,
        ci_lower_col="conf_lower", ci_upper_col="conf_upper",
        xlabel="Coefficient (95% CI)",
        title="Additive Effects",
    )
    _draw_panel(
        ax_mod, mod_plot, mod_labels, mod_colors,
        ci_lower_col="ci_lower", ci_upper_col="ci_upper",
        xlabel="onoff × moderator coefficient (95% CI)",
        title="Moderation of onoff Effect",
    )

    # Shared legend
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE_SIGNIFICANT, markersize=9,
                   label=f"Significant (p_fdr < {MCC_ALPHA})"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=PALETTE_NOT_SIGNIFICANT, markersize=9,
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

    out_path = output_dir / f"{marker}_combined_forest.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scatter_grid(
    data: pd.DataFrame, results_df: pd.DataFrame, marker: str, output_dir: Path
) -> None:
    """Grid of scatter plots: one panel per predictor vs objective marker.

    Each panel shows:
    - One dot per subject (mean across probes) with colour per group
    - LMM-derived regression line (intercept + slope for that predictor)

    Parameters
    ----------
    data : pd.DataFrame
        Merged probe-level dataframe.
    results_df : pd.DataFrame
        Output of `fit_lmm` for this marker.
    marker : str
        Dependent variable name.
    output_dir : Path
        Output directory.
    """
    n_cols = 3
    n_rows = int(np.ceil(len(PREDICTORS) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes_flat = axes.flatten()

    # Subject-level means for scatter
    agg_cols = ["subject"] + [c for c in ["group"] + PREDICTORS + [marker] if c in data.columns]
    subj_df = data[agg_cols].groupby("subject").mean(numeric_only=True).reset_index()
    if "group" in data.columns:
        subj_df["group"] = data.groupby("subject")["group"].first().values

    # Colour by group if available
    group_colors_map = {}
    if "group" in subj_df.columns:
        unique_groups = subj_df["group"].dropna().unique()
        palette = ["#2E86AB", "#F24236"]
        group_colors_map = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}

    # Extract intercept from LMM results
    intercept_row = results_df[results_df["predictor"] == "Intercept"]
    intercept = intercept_row["estimate"].values[0] if len(intercept_row) > 0 else 0.0

    for ax_idx, predictor in enumerate(PREDICTORS):
        ax = axes_flat[ax_idx]
        pred_label = PREDICTOR_LABELS.get(predictor, predictor)

        # LMM slope for this predictor
        pred_row = results_df[results_df["predictor"] == predictor]
        slope = pred_row["estimate"].values[0] if len(pred_row) > 0 else 0.0
        p_fdr = pred_row["p_fdr"].values[0] if len(pred_row) > 0 else np.nan
        significant = pred_row["significant_fdr"].values[0] if len(pred_row) > 0 else False

        if predictor not in subj_df.columns or marker not in subj_df.columns:
            ax.set_visible(False)
            continue

        valid = subj_df[[predictor, marker]].dropna()

        # Scatter per group
        if group_colors_map:
            for group, color in group_colors_map.items():
                grp_mask = subj_df["group"] == group
                grp_valid = subj_df[grp_mask][[predictor, marker]].dropna()
                ax.scatter(
                    grp_valid[predictor], grp_valid[marker],
                    color=color, s=60, alpha=0.8, linewidths=0, label=group, zorder=3,
                )
        else:
            ax.scatter(valid[predictor], valid[marker],
                       color="#2E86AB", s=60, alpha=0.8, linewidths=0, zorder=3)

        # LMM regression line
        if len(valid) > 1:
            x_range = np.linspace(valid[predictor].min(), valid[predictor].max(), 100)
            y_line = intercept + slope * x_range
            line_color = PALETTE_SIGNIFICANT if significant else "#555555"
            ax.plot(x_range, y_line, color=line_color, linewidth=2.5, zorder=4)

        # Subplot annotation
        sig_marker = "✱" if significant else ""
        p_text = f"p_fdr={p_fdr:.3f}" if not np.isnan(p_fdr) else ""
        ax.set_title(
            f"{pred_label}{sig_marker}\n"
            f"β={slope:.3f}  {p_text}",
            fontsize=10, fontweight="bold" if significant else "normal",
        )
        ax.set_xlabel(pred_label, fontsize=9)
        ax.set_ylabel(MARKER_LABELS[marker], fontsize=9)
        ax.grid(True, alpha=0.25)

    # Remove empty axes
    for ax_idx in range(len(PREDICTORS), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Group legend (top-right of figure)
    if group_colors_map:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=g) for g, c in group_colors_map.items()]
        fig.legend(handles=handles, fontsize=10, loc="upper right", title="Group")

    fig.suptitle(
        f"{MARKER_LABELS[marker]}: LMM Coefficients by Predictor\n"
        f"(dots = subject means, line = LMM slope)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = output_dir / f"{marker}_scatter_grid.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


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

def main() -> None:
    """Run the full LMM pipeline for all objective markers."""
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data -----------------------------------------------------------
    print("Loading objective markers data...")
    df_markers = pd.read_csv(OBJECTIVE_MARKERS_PATH)
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

    # --- Within-subject z-scoring (same as objective_markers_analysis.py) ----
    if APPLY_WITHIN_SUBJECT_Z:
        print("\nApplying within-subject z-scoring to objective markers...")
        df = apply_within_subject_z_scoring(df, OBJECTIVE_MARKERS)

    # --- Run one LMM per marker ----------------------------------------------
    all_results: dict[str, pd.DataFrame] = {}

    for marker in OBJECTIVE_MARKERS:
        results_df = fit_lmm(
            data=df,
            marker=marker,
            formula_template=FORMULA_TEMPLATE,
            method=LMM_METHOD,
            maxiter=LMM_MAXITER,
        )
        all_results[marker] = results_df

        # Save CSV
        csv_path = OUTPUT_DIR / f"{marker}_lmm_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results: {csv_path}")

    # --- Plots ---------------------------------------------------------------
    print("\nGenerating scatter grids...")
    for marker, results_df in all_results.items():
        plot_scatter_grid(df, results_df, marker, OUTPUT_DIR)

    # --- Summary table -------------------------------------------------------
    # Collect significant predictors across markers for quick inspection
    summary_rows = []
    for marker, results_df in all_results.items():
        for _, row in results_df[results_df["predictor"].isin(PREDICTORS)].iterrows():
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
    summary_path = OUTPUT_DIR / "lmm_summary_all_markers.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary table saved: {summary_path}")

    # Print significant effects
    sig = summary_df[summary_df["significant_fdr"] == True]
    print(f"\n{'='*60}")
    print("SIGNIFICANT ADDITIVE EFFECTS (FDR-BH corrected):")
    print("="*60)
    if len(sig) > 0:
        print(sig[["marker", "predictor", "estimate", "t_value", "p_value", "p_fdr"]].to_string(index=False))
    else:
        print("  None found after FDR correction.")

    # --- Moderation analysis --------------------------------------------------
    moderation_df = run_moderation_analysis(
        data=df,
        markers=OBJECTIVE_MARKERS,
        moderators=MODERATORS,
        method=LMM_METHOD,
        maxiter=LMM_MAXITER,
        output_dir=OUTPUT_DIR,
    )

    # --- Combined forest plots (additive + moderation side-by-side) -----------
    print("\nGenerating combined forest plots...")
    for marker, results_df in all_results.items():
        plot_combined_forest(results_df, moderation_df, marker, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"Results in: {OUTPUT_DIR.resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
