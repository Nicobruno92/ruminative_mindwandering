#!/usr/bin/env python3
"""
Bidirectional Mood–Thoughts Analysis (Block Level)

This script tests whether mood (EVA scale, ~1–10) predicts thought dimensions
(0–100 probes) and whether thought content predicts mood changes across SART
blocks.

It builds a block-level dataset per subject × task (Sart1–Sart4) with:
- mean_<dim> : mean probe score per SART block
- mood_pre, mood_post : EVA before/after block (from EVA blocks)
- delta_mood : mood_post - mood_pre

Models (mixed-effects, random intercept per subject):
1) mood → thoughts   : mean_<dim>  ~ mood_pre + covariates
2) thoughts → mood   : mood_post   ~ mood_pre + mean_<dim> + covariates

Run this file directly; adjust paths and options in the CONFIG section below.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import statsmodels.formula.api as smf

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input data files
# Use PCA-augmented probe file so that PC1 can be treated as an extra
# thought-dimension variable. This CSV is produced by Behavior/PCA/pca_analysis.py.
PROBE_DATA_FILE: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/probe_data/pca_results.csv"
)
EVA_DATA_FILE: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Output directories
RESULTS_DIR: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/mood_thoughts_bidirectional"
)
PLOTS_DIR: str = os.path.join(RESULTS_DIR, "plots")

# Thought dimensions to analyse (0–100 scale in probes, plus PCA1)
THOUGHT_DIMENSIONS: List[str] = [
    "onoff",
    "valence",
    "time",
    "selfother",
    "confidence",
    "pca1",  # PC1 from PCA of [valence, selfother, time]
]

# EVA columns to use as mood indices (approx 1–10 scale)
# Options include: "EVAtense", "EVAfeel", "EVAmood", "EVAhurt", "EVAaverage", "total_score".
MOOD_EVA_DIMENSIONS: List[str] = ["EVAtense", "EVAfeel", "EVAmood", "EVAhurt", "EVAaverage", "total_score"]
# First element is kept for backward compatibility in labels where a single
# mood dimension is needed.
MOOD_EVA_DIMENSION: str = MOOD_EVA_DIMENSIONS[0]

# Plot aesthetics
GROUP_ORDER: List[str] = ["Controls", "Risk of Depression"]
GROUP_COLORS: List[str] = ["#2E86AB", "#F24236"]

# Optional ON/OFF filter at probe level
APPLY_ONOFF_FILTER: bool = True
ONOFF_MAX_EXCLUSIVE: float = 50.0

# Minimal number of blocks to fit a model
MIN_BLOCKS_PER_MODEL: int = 25

# Use a simplified covariate structure (only group as categorical factor).
# This reduces the risk of collinearity between task and inclusion_exclusion
# and makes mixed-model fitting more stable.
USE_SIMPLE_COVARIATES: bool = True

# When computing block-level means of thought dimensions, optionally use only
# the last N probes within each subject×task block (e.g., last 5 probes).
# This lets you ask whether the *late* probe content in a SART predicts mood
# change, instead of averaging across the entire block.
USE_LAST_N_PROBES_FOR_MEAN: bool = True
N_PROBES_PER_BLOCK_FOR_MEAN: int = 5
PROBE_INDEX_COLUMN: str = "probe_number"


# =============================================================================
# IO AND PREPROCESSING
# =============================================================================


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_probe_data() -> pd.DataFrame:
    """Load probe-level data and optionally filter on ON/OFF.

    Returns
    -------
    pd.DataFrame
        Probe-level dataframe with required columns.
    """
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe data file not found: {PROBE_DATA_FILE}")

    df = pd.read_csv(PROBE_DATA_FILE)
    print(f"Loaded {len(df)} probe rows")

    # If PCA components are present, expose PC1 as 'pca1' for downstream analyses
    if "PC1" in df.columns and "pca1" not in df.columns:
        df["pca1"] = df["PC1"]
        print("Created 'pca1' column from PC1 in PCA results")

    required = {"subject_id", "task", "group"} | set(THOUGHT_DIMENSIONS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Probe data missing columns: {sorted(missing)}")

    if APPLY_ONOFF_FILTER and "onoff" in df.columns:
        before = len(df)
        df = df[df["onoff"] < ONOFF_MAX_EXCLUSIVE].copy()
        after = len(df)
        print(f"Applied onoff < {ONOFF_MAX_EXCLUSIVE}: {before} -> {after} probes")

    return df


def load_eva_data() -> pd.DataFrame:
    """Load EVA mood scale blocks.

    Returns
    -------
    pd.DataFrame
        EVA dataframe with one row per EVA block.
    """
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA data file not found: {EVA_DATA_FILE}")

    df = pd.read_csv(EVA_DATA_FILE)
    print(f"Loaded {len(df)} EVA blocks")

    # Require all requested mood EVA dimensions to be present so that we can
    # iterate over them later.
    required = {"subject_id", "task", "block_number"} | set(MOOD_EVA_DIMENSIONS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"EVA data missing columns: {sorted(missing)}")

    # Keep metadata if present (group, inclusion_exclusion)
    return df


def build_block_level_dataset(
    df_probes: pd.DataFrame,
    df_eva: pd.DataFrame,
    mood_col: str,
) -> pd.DataFrame:
    """Aggregate probes and EVA to block level.

    Parameters
    ----------
    df_probes : pd.DataFrame
        Probe-level data.
    df_eva : pd.DataFrame
        EVA block-level data.

    Returns
    -------
    pd.DataFrame
        Block-level dataframe with mean thoughts and mood_pre/post.
    """
    # Optionally restrict to the last N probes within each subject×task block
    if USE_LAST_N_PROBES_FOR_MEAN and PROBE_INDEX_COLUMN in df_probes.columns:
        def _keep_last_n(group: pd.DataFrame) -> pd.DataFrame:
            # Sort by the probe index column and keep only the last N rows
            group_sorted = group.sort_values(PROBE_INDEX_COLUMN)
            return group_sorted.tail(N_PROBES_PER_BLOCK_FOR_MEAN)

        df_for_mean = (
            df_probes.groupby(["subject_id", "task"], group_keys=False)
            .apply(_keep_last_n)
        )
        print(
            f"Using last {N_PROBES_PER_BLOCK_FOR_MEAN} probes per subject×task "
            f"(based on '{PROBE_INDEX_COLUMN}') for block-level means"
        )
    else:
        if USE_LAST_N_PROBES_FOR_MEAN and PROBE_INDEX_COLUMN not in df_probes.columns:
            print(
                f"Warning: PROBE_INDEX_COLUMN='{PROBE_INDEX_COLUMN}' not found; "
                "using all probes per block for means instead."
            )
        df_for_mean = df_probes

    # Mean thought dimensions per subject × task
    agg_dict: Dict[str, str] = {dim: "mean" for dim in THOUGHT_DIMENSIONS}
    df_thought = (
        df_for_mean.groupby(["subject_id", "task", "group"], as_index=False)
        .agg(agg_dict)
        .rename(columns={dim: f"mean_{dim}" for dim in THOUGHT_DIMENSIONS})
    )

    # Carry inclusion_exclusion if present in probe data
    if "inclusion_exclusion" in df_probes.columns:
        ie_meta = (
            df_probes[["subject_id", "task", "inclusion_exclusion"]]
            .drop_duplicates()
        )
        df_thought = df_thought.merge(ie_meta, on=["subject_id", "task"], how="left")

    # Mood per subject × task: first/last EVA block for the chosen mood column

    def _eva_to_block(group: pd.DataFrame) -> pd.Series:
        group_sorted = group.sort_values("block_number")
        vals = group_sorted[mood_col].dropna()
        if vals.empty:
            mood_pre = np.nan
            mood_post = np.nan
        else:
            mood_pre = float(vals.iloc[0])
            mood_post = float(vals.iloc[-1])
        out = {"mood_pre": mood_pre, "mood_post": mood_post}
        # Metadata if present
        for col in ["group", "inclusion_exclusion"]:
            if col in group_sorted.columns:
                out[col] = group_sorted[col].iloc[0]
        return pd.Series(out)

    df_eva_block = (
        df_eva.groupby(["subject_id", "task"], as_index=False)
        .apply(_eva_to_block)
        .reset_index(drop=True)
    )
    df_eva_block["delta_mood"] = df_eva_block["mood_post"] - df_eva_block["mood_pre"]

    # Merge
    df_block = df_thought.merge(df_eva_block, on=["subject_id", "task"], how="inner")

    before = len(df_block)
    df_block = df_block.dropna(subset=["mood_pre", "mood_post"]).reset_index(drop=True)
    print(
        f"Block-level dataset: {before} -> {len(df_block)} blocks after removing missing mood"
    )

    return df_block


# =============================================================================
# MODELLING
# =============================================================================


def _build_covariate_rhs(df: pd.DataFrame, include_mood_pre: bool, mean_col: str | None) -> str:
    """Build RHS of formula depending on available covariates.

    Parameters
    ----------
    df : pd.DataFrame
        Block-level data.
    include_mood_pre : bool
        Whether to include mood_pre in the RHS.
    mean_col : str or None
        Mean thought column to include (for thoughts→mood model).
    """
    terms: List[str] = []

    # Core temporal / directional predictors
    if include_mood_pre:
        terms.append("mood_pre")
    if mean_col is not None:
        terms.append(mean_col)

    # Categorical covariates
    if USE_SIMPLE_COVARIATES:
        # Only adjust for group to keep models stable
        if "group" in df.columns:
            terms.append("C(group, Treatment('Controls'))")
    else:
        # Full specification: group + task + inclusion_exclusion (if present)
        if "group" in df.columns:
            terms.append("C(group, Treatment('Controls'))")
        if "task" in df.columns:
            terms.append("C(task)")
        if "inclusion_exclusion" in df.columns:
            terms.append("C(inclusion_exclusion)")

    return " + ".join(terms) if terms else "1"


def run_mixedlm(
    data: pd.DataFrame,
    dep_var: str,
    rhs: str,
    model_name: str,
    output_dir: str,
) -> pd.DataFrame:
    """Fit linear mixed model with random intercept for subject_id.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    dep_var : str
        Dependent variable name.
    rhs : str
        Right-hand side of the formula.
    model_name : str
        Label for outputs.
    output_dir : str
        Directory where CSV and summary will be saved.
    """
    print(f"\n=== {model_name} | {dep_var} ===")
    print(f"Formula: {dep_var} ~ {rhs}")

    # Identify required columns
    tokens = (
        rhs.replace("~", " ")
        .replace("+", " ")
        .replace("*", " ")
        .replace(":", " ")
        .replace("/", " ")
        .replace("C(", " ")
        .replace(")", " ")
        .replace(",", " ")
        .split()
    )
    # Keep only actual column names, ignore formula helpers such as Treatment(...)
    predictors = {
        t
        for t in tokens
        if t
        and t not in {"1"}
        and not t.startswith("Treatment(")
    }
    required = {dep_var, "subject_id"} | predictors
    missing = required - set(data.columns)
    if missing:
        print(f"Skipping {model_name}: missing columns {sorted(missing)}")
        return pd.DataFrame()

    model_df = data.loc[:, sorted(required)].copy()
    for cat in ["group", "task", "inclusion_exclusion"]:
        if cat in model_df.columns:
            model_df[cat] = model_df[cat].astype("category")

    model_df = model_df.dropna(axis=0, how="any").reset_index(drop=True)
    n_blocks = len(model_df)
    if n_blocks < MIN_BLOCKS_PER_MODEL:
        print(f"Skipping {model_name}: only {n_blocks} usable blocks")
        return pd.DataFrame()

    print(
        f"Fitting on {n_blocks} blocks from {model_df['subject_id'].nunique()} subjects"
    )

    formula = f"{dep_var} ~ {rhs}"
    model = smf.mixedlm(formula, model_df, groups=model_df["subject_id"])

    try:
        # Use LBFGS by default; if optimisation fails or the design is singular,
        # catch the error and skip this model instead of crashing the script.
        result = model.fit(method="lbfgs", maxiter=1000)
    except np.linalg.LinAlgError as exc:
        print(f"Skipping {model_name} ({dep_var}): linear algebra error: {exc}")
        return pd.DataFrame()
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Skipping {model_name} ({dep_var}): optimisation error: {exc}")
        return pd.DataFrame()

    params = result.params
    conf_int = result.conf_int()

    results_df = pd.DataFrame(
        {
            "parameter": params.index,
            "estimate": params.values,
            "std_error": result.bse.values,
            "t_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "conf_lower": conf_int.iloc[:, 0].values,
            "conf_upper": conf_int.iloc[:, 1].values,
            "aic": result.aic,
            "bic": result.bic,
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"{model_name}_{dep_var}.csv")
    out_txt = os.path.join(output_dir, f"{model_name}_{dep_var}_summary.txt")
    results_df.to_csv(out_csv, index=False)
    with open(out_txt, "w") as f:
        f.write(str(result.summary()))

    print(f"Saved results to: {out_csv}")
    return results_df


# =============================================================================
# PLOTTING
# =============================================================================


def plot_bidirectional_relationships(
    df_block: pd.DataFrame,
    thought_dim: str,
    mood_label: str,
    pdf: Optional[PdfPages],
    out_dir: str,
) -> None:
    """Plot mood→thoughts and thoughts→mood scatterplots with regression lines.

    Parameters
    ----------
    df_block : pd.DataFrame
        Block-level dataframe for a given mood dimension.
    thought_dim : str
        Name of the thought dimension (e.g., "onoff").
    mood_label : str
        EVA column used as mood index for this analysis.
    pdf : PdfPages or None
        If provided, each figure is added as a page to this PDF.
    out_dir : str
        Directory where PNG files will be saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    mean_col = f"mean_{thought_dim}"

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------------------------------------------------
    # Panel 1: mood_pre (x) → mean_thought (y)
    ax1 = axes[0]
    for g_idx, group in enumerate(GROUP_ORDER):
        if group not in df_block["group"].values:
            continue
        gdat = df_block[df_block["group"] == group]
        ax1.scatter(
            gdat["mood_pre"],
            gdat[mean_col],
            color=GROUP_COLORS[g_idx],
            alpha=0.5,
            s=30,
            label=group,
        )
        if len(gdat) > 5:
            z = np.polyfit(gdat["mood_pre"], gdat[mean_col], 1)
            x_line = np.linspace(gdat["mood_pre"].min(), gdat["mood_pre"].max(), 50)
            ax1.plot(x_line, np.polyval(z, x_line), color=GROUP_COLORS[g_idx])

    ax1.set_xlabel(f"Mood pre (EVA: {mood_label})")
    ax1.set_ylabel(f"Mean {thought_dim} (0–100)")
    ax1.set_title("Mood → Thoughts")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel 2: mean_thought (x) → delta_mood (y)
    ax2 = axes[1]
    for g_idx, group in enumerate(GROUP_ORDER):
        if group not in df_block["group"].values:
            continue
        gdat = df_block[df_block["group"] == group]
        ax2.scatter(
            gdat[mean_col],
            gdat["delta_mood"],
            color=GROUP_COLORS[g_idx],
            alpha=0.5,
            s=30,
            label=group,
        )
        if len(gdat) > 5:
            z = np.polyfit(gdat[mean_col], gdat["delta_mood"], 1)
            x_line = np.linspace(gdat[mean_col].min(), gdat[mean_col].max(), 50)
            ax2.plot(x_line, np.polyval(z, x_line), color=GROUP_COLORS[g_idx])

    ax2.set_xlabel(f"Mean {thought_dim} (0–100)")
    ax2.set_ylabel("ΔMood (post − pre)")
    ax2.set_title("Thoughts → Mood")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    plt.suptitle(f"Bidirectional Mood–Thoughts ({mood_label}): {thought_dim}")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    # Save to multi-page PDF if requested
    if pdf is not None:
        pdf.savefig(fig)

    # Also save individual PNG per thought dimension and mood
    safe_mood = mood_label.replace("/", "-")
    out_png = os.path.join(out_dir, f"{thought_dim}_bidirectional_{safe_mood}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_png}")


def plot_beta_heatmaps(
    combined_results: pd.DataFrame,
    plots_dir: Optional[str] = None,
    file_suffix: str = "",
    pdf: Optional[PdfPages] = None,
) -> None:
    """Create heatmaps of key beta slopes across thought dimensions and groups.

    Parameters
    ----------
    combined_results : pd.DataFrame
        Combined results table from all fitted models
        (output of concatenating per-dimension results).
    plots_dir : str or None
        Directory where the heatmap PNG will be saved. If None, uses PLOTS_DIR.
    file_suffix : str
        Optional suffix to append to the output filename (e.g., "_EVAmood").
    pdf : PdfPages or None
        If provided, the heatmap figure is also added as a page to this PDF.
    """
    if combined_results.empty:
        return

    dims = THOUGHT_DIMENSIONS

    # Significance threshold for colouring cells
    alpha = 0.05

    # ------------------------------------------------------------------
    # 1) Mood → Thoughts: raw betas and p-values for each predictor
    # ------------------------------------------------------------------
    predictors_m2t = [
        "mood_pre",
        "group_Risk",
        "mood_pre:group_Risk",
    ]
    beta_m2t = np.full((len(dims), len(predictors_m2t)), np.nan, dtype=float)
    p_m2t = np.full_like(beta_m2t, np.nan, dtype=float)

    for i, dim in enumerate(dims):
        df_dim = combined_results[
            (combined_results["direction"] == "mood_to_thoughts")
            & (combined_results["thought_dim"] == dim)
        ]
        if df_dim.empty:
            continue

        # mood_pre
        row = df_dim[df_dim["parameter"] == "mood_pre"]
        if not row.empty:
            beta_m2t[i, 0] = float(row["estimate"].iloc[0])
            p_m2t[i, 0] = float(row["p_value"].iloc[0])

        # group_Risk: C(group, Treatment('Controls'))[T.Risk of Depression]
        row = df_dim[
            df_dim["parameter"]
            == "C(group, Treatment('Controls'))[T.Risk of Depression]"
        ]
        if not row.empty:
            beta_m2t[i, 1] = float(row["estimate"].iloc[0])
            p_m2t[i, 1] = float(row["p_value"].iloc[0])

        # interaction mood_pre:group_Risk
        row = df_dim[df_dim["parameter"].str.startswith("mood_pre:C(")]
        if not row.empty:
            beta_m2t[i, 2] = float(row["estimate"].iloc[0])
            p_m2t[i, 2] = float(row["p_value"].iloc[0])

    beta_m2t_df = pd.DataFrame(beta_m2t, index=dims, columns=predictors_m2t)
    p_m2t_df = pd.DataFrame(p_m2t, index=dims, columns=predictors_m2t)

    # ------------------------------------------------------------------
    # 2) Thoughts → Mood: raw betas and p-values for each predictor
    # ------------------------------------------------------------------
    predictors_t2m = [
        "mood_pre",
        "mean_thought",
        "group_Risk",
        "mean_thought:group_Risk",
    ]
    beta_t2m = np.full((len(dims), len(predictors_t2m)), np.nan, dtype=float)
    p_t2m = np.full_like(beta_t2m, np.nan, dtype=float)

    for i, dim in enumerate(dims):
        mean_col = f"mean_{dim}"
        df_dim = combined_results[
            (combined_results["direction"] == "thoughts_to_mood")
            & (combined_results["thought_dim"] == dim)
        ]
        if df_dim.empty:
            continue

        # mood_pre
        row = df_dim[df_dim["parameter"] == "mood_pre"]
        if not row.empty:
            beta_t2m[i, 0] = float(row["estimate"].iloc[0])
            p_t2m[i, 0] = float(row["p_value"].iloc[0])

        # mean_thought (mean_<dim>)
        row = df_dim[df_dim["parameter"] == mean_col]
        if not row.empty:
            beta_t2m[i, 1] = float(row["estimate"].iloc[0])
            p_t2m[i, 1] = float(row["p_value"].iloc[0])

        # group_Risk main effect
        row = df_dim[
            df_dim["parameter"]
            == "C(group, Treatment('Controls'))[T.Risk of Depression]"
        ]
        if not row.empty:
            beta_t2m[i, 2] = float(row["estimate"].iloc[0])
            p_t2m[i, 2] = float(row["p_value"].iloc[0])

        # interaction mean_thought:group_Risk
        row = df_dim[df_dim["parameter"].str.startswith(f"{mean_col}:C(")]
        if not row.empty:
            beta_t2m[i, 3] = float(row["estimate"].iloc[0])
            p_t2m[i, 3] = float(row["p_value"].iloc[0])

    beta_t2m_df = pd.DataFrame(beta_t2m, index=dims, columns=predictors_t2m)
    p_t2m_df = pd.DataFrame(p_t2m, index=dims, columns=predictors_t2m)

    # ------------------------------------------------------------------
    # Build significance masks and annotations (asterisks for p-values)
    # ------------------------------------------------------------------
    sig_m2t = (p_m2t_df.values < alpha) & ~np.isnan(p_m2t_df.values)
    sig_t2m = (p_t2m_df.values < alpha) & ~np.isnan(p_t2m_df.values)

    # Use only significant betas to determine the colour range, to avoid
    # being dominated by extreme but non-significant estimates.
    all_sig_vals = np.concatenate([
        beta_m2t_df.values[sig_m2t],
        beta_t2m_df.values[sig_t2m],
    ])
    all_sig_vals = all_sig_vals[~np.isnan(all_sig_vals)]
    if all_sig_vals.size == 0:
        # No significant effects at all: nothing to colour meaningfully
        return

    # Percentile-based vmax to reduce the impact of extreme values
    vmax = float(np.nanpercentile(np.abs(all_sig_vals), 90))
    if vmax == 0:
        vmax = float(np.nanmax(np.abs(all_sig_vals)))
    if vmax == 0:
        # All effects are exactly zero
        return

    # Annotation matrices with asterisks encoding significance level
    def _build_annotations(beta_df: pd.DataFrame, p_df: pd.DataFrame) -> List[List[str]]:
        annot: List[List[str]] = []
        for i in range(beta_df.shape[0]):
            row_ann: List[str] = []
            for j in range(beta_df.shape[1]):
                beta_val = beta_df.iat[i, j]
                p_val = p_df.iat[i, j]
                if np.isnan(beta_val):
                    row_ann.append("")
                    continue
                text = f"{beta_val:.2f}"
                if not np.isnan(p_val):
                    if p_val < 0.001:
                        text += "***"
                    elif p_val < 0.01:
                        text += "**"
                    elif p_val < alpha:
                        text += "*"
                row_ann.append(text)
            annot.append(row_ann)
        return annot

    annot_m2t = _build_annotations(beta_m2t_df, p_m2t_df)
    annot_t2m = _build_annotations(beta_t2m_df, p_t2m_df)

    # Matrices used for colouring: keep true betas for significant cells,
    # but set non-significant cells to zero so they appear at the centre
    # of the colour map (almost white). Numeric annotations still show
    # the original beta values everywhere.
    beta_m2t_plot = beta_m2t_df.copy()
    beta_t2m_plot = beta_t2m_df.copy()
    beta_m2t_plot.values[~sig_m2t] = 0.0
    beta_t2m_plot.values[~sig_t2m] = 0.0

    # ------------------------------------------------------------------
    # Plot heatmaps of raw betas (significant cells coloured, numbers always shown)
    # ------------------------------------------------------------------
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    sns.heatmap(
        beta_m2t_plot,
        ax=axes[0],
        annot=annot_m2t,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Beta (significant cells coloured)"},
    )
    axes[0].set_title("Mood → Thoughts (betas)")
    axes[0].set_xlabel("Predictor")
    axes[0].set_ylabel("Thought dimension")

    sns.heatmap(
        beta_t2m_plot,
        ax=axes[1],
        annot=annot_t2m,
        fmt="",
        cmap="RdBu_r",
        center=0.0,
        vmin=-vmax,
        vmax=vmax,
        cbar_kws={"label": "Beta (significant cells coloured)"},
    )
    axes[1].set_title("Thoughts → Mood (betas)")
    axes[1].set_xlabel("Predictor")
    axes[1].set_ylabel("Thought dimension")

    plt.tight_layout()

    # Add page to PDF if requested
    if pdf is not None:
        pdf.savefig(fig)

    target_dir = plots_dir if plots_dir is not None else PLOTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    out_png = os.path.join(target_dir, f"beta_heatmaps_overview{file_suffix}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved beta heatmaps to: {out_png}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    """Run bidirectional mood–thoughts analysis for all configured mood scales."""
    print("=" * 70)
    print("BIDIRECTIONAL MOOD–THOUGHTS ANALYSIS (BLOCK LEVEL)")
    print("=" * 70)

    ensure_directories()

    df_probes = load_probe_data()
    df_eva = load_eva_data()

    # Global multi-page PDF with one heatmap page per mood dimension
    heatmap_pdf_path = os.path.join(PLOTS_DIR, "beta_heatmaps_overview_all_moods.pdf")
    with PdfPages(heatmap_pdf_path) as heatmap_pdf:
        for mood_col in MOOD_EVA_DIMENSIONS:
            if mood_col not in df_eva.columns:
                print(f"Skipping mood column '{mood_col}': not found in EVA data")
                continue

            print("\n" + "#" * 70)
            print(f"ANALYSIS FOR MOOD DIMENSION: {mood_col}")
            print("#" * 70)

            # Build block-level dataset for this mood dimension
            df_block = build_block_level_dataset(df_probes, df_eva, mood_col=mood_col)

            print("\nBlock-level summary:")
            print(f"  N blocks: {len(df_block)}")
            print(f"  N subjects: {df_block['subject_id'].nunique()}")
            print(f"  Tasks: {sorted(df_block['task'].unique())}")

            all_results: List[pd.DataFrame] = []

            mood_results_dir = os.path.join(RESULTS_DIR, mood_col)
            mood_plots_dir = os.path.join(PLOTS_DIR, mood_col)
            os.makedirs(mood_results_dir, exist_ok=True)
            os.makedirs(mood_plots_dir, exist_ok=True)

            # Multi-page PDF with one page per thought dimension for this mood
            pdf_path = os.path.join(mood_plots_dir, f"bidirectional_plots_{mood_col}.pdf")
            with PdfPages(pdf_path) as pdf:
                for dim in THOUGHT_DIMENSIONS:
                    mean_col = f"mean_{dim}"
                    if mean_col not in df_block.columns:
                        print(f"Skipping {dim}: column {mean_col} missing")
                        continue

                    print("\n" + "-" * 60)
                    print(f"ANALYSIS FOR THOUGHT DIMENSION: {dim.upper()}")
                    print("-" * 60)

                    dim_results_dir = os.path.join(mood_results_dir, dim)
                    os.makedirs(dim_results_dir, exist_ok=True)

                    # Mood → Thoughts: allow group to modulate the effect of mood_pre
                    if "group" in df_block.columns:
                        # mood_pre * group expands to mood_pre + C(group) + interaction term
                        rhs_mood_to_thoughts = "mood_pre * C(group, Treatment('Controls'))"
                    else:
                        rhs_mood_to_thoughts = "mood_pre"
                    res1 = run_mixedlm(
                        df_block,
                        dep_var=mean_col,
                        rhs=rhs_mood_to_thoughts,
                        model_name="mood_to_thoughts",
                        output_dir=dim_results_dir,
                    )
                    if not res1.empty:
                        res1.insert(0, "direction", "mood_to_thoughts")
                        res1.insert(1, "thought_dim", dim)
                        res1.insert(2, "mood_dim", mood_col)
                        all_results.append(res1)

                    # Thoughts → Mood (predict mood_post controlling for mood_pre)
                    # Here we want group to modulate the effect of mean_<dim> on mood_post.
                    if "group" in df_block.columns:
                        rhs_thoughts_to_mood = (
                            f"mood_pre + {mean_col} * C(group, Treatment('Controls'))"
                        )
                    else:
                        rhs_thoughts_to_mood = f"mood_pre + {mean_col}"
                    res2 = run_mixedlm(
                        df_block,
                        dep_var="mood_post",
                        rhs=rhs_thoughts_to_mood,
                        model_name="thoughts_to_mood",
                        output_dir=dim_results_dir,
                    )
                    if not res2.empty:
                        res2.insert(0, "direction", "thoughts_to_mood")
                        res2.insert(1, "thought_dim", dim)
                        res2.insert(2, "mood_dim", mood_col)
                        all_results.append(res2)

                    # Plots (one page per thought dimension in the PDF)
                    plot_bidirectional_relationships(
                        df_block,
                        thought_dim=dim,
                        mood_label=mood_col,
                        pdf=pdf,
                        out_dir=mood_plots_dir,
                    )

            if all_results:
                combined = pd.concat(all_results, ignore_index=True)
                combined_file = os.path.join(
                    mood_results_dir,
                    f"mood_thoughts_models_summary_{mood_col}.csv",
                )
                combined.to_csv(combined_file, index=False)
                print(f"\nSaved combined model summary to: {combined_file}")

                # Create heatmaps summarising key beta coefficients across dimensions
                plot_beta_heatmaps(
                    combined,
                    plots_dir=mood_plots_dir,
                    file_suffix=f"_{mood_col}",
                    pdf=heatmap_pdf,
                )

            print("\nMulti-page PDF of bidirectional plots saved to:")
            print(f"  {pdf_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results base dir: {RESULTS_DIR}")
    print(f"Plots base dir:   {PLOTS_DIR}")
    print("Combined heatmap PDF across moods:")
    print(f"  {heatmap_pdf_path}")


if __name__ == "__main__":
    main()
