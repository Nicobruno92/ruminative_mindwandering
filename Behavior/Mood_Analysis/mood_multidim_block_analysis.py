from __future__ import annotations

"""Block-Level Mood Analysis (LMM + Plots)

This script mirrors the multi-dimension probe analysis but for mood scales
measured at the block level (EVA data).

It:
- Builds a block-level dataset with one row per subject × task.
- Uses all EVA mood variables as dependent variables.
- Fits LMMs for group effects and inclusion/exclusion effects (baseline
  corrected across SART1→SART2 and SART3→SART4).
- Exports model results/metrics and generates plots and descriptive
  statistics per mood variable.

The design follows the structure of ``probe_analysis_multidim.py`` while
respecting the block-level nature of the mood measurements.

Usage
-----
Simply run this script. Adjust configuration variables in the CONFIGURATION
section as needed.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
import statsmodels.formula.api as smf


# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Input EVA block-level mood data
EVA_DATA_FILE: str = (
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Probe-level aggregated data (used to attach group / IE / order info)
PROBE_DATA_FILE: str = (
    "results/Behavior/probe_data/probe_level_aggregated_data.csv"
)

# Output directory for mood LMM analysis
RESULTS_DIR: str = "results/Behavior/mood"
PLOTS_DIR: str = RESULTS_DIR

# Mood dimensions to analyze (columns expected in EVA data)
MOOD_DIMENSIONS: List[str] = [
    "EVAtense",
    "EVAfeel",
    "EVAmood",
    "EVAhurt",
    "EVAaverage",
    "total_score",
]

# Plot aesthetics
GROUP_ORDER: List[str] = ["Controls", "Risk of Depression"]
GROUP_COLORS: List[str] = ["#2E86AB", "#F24236"]
IE_ORDER: List[str] = ["inclusion", "exclusion"]
IE_COLORS: List[str] = ["#A23B72", "#F18F01"]

# Minimum number of blocks required for analysis
MIN_BLOCKS: int = 20


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_block_level_mood_data() -> pd.DataFrame:
    """Load EVA and probe data and create block-level mood dataset.

    The resulting table has one row per subject × task with:

    - EVA mood scales (first non-NaN block per task).
    - ``group`` and ``inclusion_exclusion`` from probe data.
    - ``order (IE/EI)`` if available in probe data.

    Returns
    -------
    pd.DataFrame
        Block-level mood dataset.
    """
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA data not found: {EVA_DATA_FILE}")
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe data not found: {PROBE_DATA_FILE}")

    df_eva = pd.read_csv(EVA_DATA_FILE)
    df_probes = pd.read_csv(PROBE_DATA_FILE)

    required_eva = {"subject_id", "task", "block_number"} | set(MOOD_DIMENSIONS)
    missing_eva = required_eva - set(df_eva.columns)
    if missing_eva:
        raise ValueError(f"EVA data missing columns: {sorted(missing_eva)}")

    required_probe = {
        "subject_id",
        "task",
        "group",
        "inclusion_exclusion",
    }
    missing_probe = required_probe - set(df_probes.columns)
    if missing_probe:
        raise ValueError(f"Probe data missing columns: {sorted(missing_probe)}")

    # Aggregate EVA to one row per subject × task (first non-NaN block per mood)
    def _first_block_moods(group: pd.DataFrame) -> pd.Series:
        group_sorted = group.sort_values("block_number")
        out: Dict[str, float] = {}
        for dim in MOOD_DIMENSIONS:
            vals = group_sorted[dim].dropna()
            out[dim] = float(vals.iloc[0]) if not vals.empty else np.nan
        return pd.Series(out)

    df_mood = (
        df_eva.groupby(["subject_id", "task"], as_index=False)
        .apply(_first_block_moods)
        .reset_index(drop=True)
    )

    # Aggregate probe data to task level to attach group and IE labels
    agg_cols = {
        "group": "first",
        "inclusion_exclusion": "first",
    }
    if "order (IE/EI)" in df_probes.columns:
        agg_cols["order (IE/EI)"] = "first"

    df_task_info = (
        df_probes.groupby(["subject_id", "task"], as_index=False)
        .agg(agg_cols)
    )

    df_block = df_mood.merge(df_task_info, on=["subject_id", "task"], how="inner")

    # Drop rows with missing group or any mood measure
    before = len(df_block)
    df_block = df_block.dropna(subset=["group"] + MOOD_DIMENSIONS).reset_index(drop=True)
    after = len(df_block)
    if after < before:
        print(
            f"Dropped {before - after} block rows due to missing group or mood scores"
        )

    # Derive SART number from task label (e.g., Sart1 -> 1)
    if "task" in df_block.columns:
        df_block["sart_number"] = (
            df_block["task"].astype(str).str.extract(r"(\d+)").astype(float)
        )

    print("\nBLOCK-LEVEL MOOD DATASET SUMMARY")
    print("- N blocks:", len(df_block))
    print("- N subjects:", df_block["subject_id"].nunique())
    print("- Groups:")
    print(df_block["group"].value_counts())

    if len(df_block) < MIN_BLOCKS:
        raise ValueError(
            f"Insufficient data: {len(df_block)} blocks < {MIN_BLOCKS} minimum"
        )

    return df_block


def preprocess_mood_data(
    df_block: pd.DataFrame,
    mood_dimensions: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare datasets for group and inclusion/exclusion analyses.

    Steps
    -----
    1. Create baseline maps (Sart1 baseline for Sart2, Sart3 baseline for Sart4)
       per subject and mood dimension.
    2. Construct a dataset for inclusion/exclusion analysis with baseline-
       corrected mood scores for the post-manipulation tasks (Sart2, Sart4).

    Parameters
    ----------
    df_block : pd.DataFrame
        Block-level mood data (one row per subject × task).
    mood_dimensions : list of str
        Mood variables to analyze.

    Returns
    -------
    df_group : pd.DataFrame
        Dataset for group-level analyses (all blocks, raw mood).
    df_ie : pd.DataFrame
        Dataset for inclusion/exclusion analyses with normalized mood
        (Sart2 and Sart4 only).
    """
    df_group = df_block.copy()

    # -------------------------------------------------------------------------
    # Step 1: Compute baselines per subject and mood dimension
    # -------------------------------------------------------------------------
    baseline_means: Dict[Tuple[int, str, str], float] = {}

    for subject_id in df_block["subject_id"].unique():
        subj_data = df_block[df_block["subject_id"] == subject_id]

        sart1 = subj_data[subj_data["task"] == "Sart1"]
        sart3 = subj_data[subj_data["task"] == "Sart3"]

        for dim in mood_dimensions:
            if dim in sart1.columns and not sart1.empty:
                baseline_means[(subject_id, "Sart2", dim)] = float(sart1[dim].mean())
            if dim in sart3.columns and not sart3.empty:
                baseline_means[(subject_id, "Sart4", dim)] = float(sart3[dim].mean())

    print(
        f"Computed {len(baseline_means)} subject-task-dimension baseline entries"
    )

    # -------------------------------------------------------------------------
    # Step 2: Build inclusion/exclusion dataset (Sart2 and Sart4 only)
    # -------------------------------------------------------------------------
    df_ie = df_block.copy()

    # Keep only inclusion/exclusion blocks and tasks 2/4
    mask_ie = df_ie["inclusion_exclusion"].isin(IE_ORDER)
    mask_task = df_ie["task"].isin(["Sart2", "Sart4"])
    df_ie = df_ie[mask_ie & mask_task].copy()

    print("Inclusion/Exclusion subset:")
    print("- N rows:", len(df_ie))
    print("- N subjects:", df_ie["subject_id"].nunique())

    # Create normalized columns (post - baseline)
    for dim in mood_dimensions:
        norm_col = f"{dim}_normalized"

        def _normalize_row(row: pd.Series) -> float:
            key = (row["subject_id"], row["task"], dim)
            baseline = baseline_means.get(key, np.nan)
            if np.isnan(baseline):
                return np.nan
            return float(row[dim]) - baseline

        df_ie[norm_col] = df_ie.apply(_normalize_row, axis=1)

    # Drop rows where all normalized columns are NaN
    norm_cols = [f"{dim}_normalized" for dim in mood_dimensions]
    before = len(df_ie)
    df_ie = df_ie.dropna(subset=norm_cols, how="all").reset_index(drop=True)
    after = len(df_ie)

    if after < before:
        print(f"Removed {before - after} IE rows with missing baselines")

    return df_group, df_ie


def run_lmm_analysis(
    data: pd.DataFrame,
    dependent_var: str,
    formula_rhs: str,
    model_name: str,
    output_dir: str,
) -> Tuple[pd.DataFrame, object]:
    """Run linear mixed model analysis with random intercepts by subject.

    Parameters
    ----------
    data : pd.DataFrame
        Input data frame used for the model.
    dependent_var : str
        Dependent variable column name.
    formula_rhs : str
        Right-hand side of the formula (predictors and interactions).
    model_name : str
        Short name for the model used in filenames.
    output_dir : str
        Directory where results will be saved.

    Returns
    -------
    results_df : pd.DataFrame
        Fixed-effects table (estimates, SE, t, p, CIs).
    fitted_model : object
        Fitted statsmodels MixedLMResults object.
    """
    print(f"\n=== FITTING MODEL: {model_name} ({dependent_var}) ===")
    print(f"Formula: {dependent_var} ~ {formula_rhs}")

    tokens = (
        formula_rhs.replace("*", " ")
        .replace(":", " ")
        .replace("+", " ")
        .replace("/", " ")
        .replace("~", " ")
        .split()
    )
    predictor_cols = {t.strip() for t in tokens if t.strip()}
    required_cols = {dependent_var, "subject_id"} | predictor_cols

    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for model {model_name}: {sorted(missing)}"
        )

    model_df = data.loc[:, sorted(required_cols)].copy()

    for cat_col in ["group", "inclusion_exclusion"]:
        if cat_col in model_df.columns:
            model_df[cat_col] = model_df[cat_col].astype("category")

    # Drop rows with missing values
    before_n = len(model_df)
    model_df = (
        model_df.dropna(axis=0, how="any")
        .sort_values(["subject_id"])
        .reset_index(drop=True)
    )
    after_n = len(model_df)
    if after_n < before_n:
        print(f"Dropped {before_n - after_n} rows with missing data")

    os.makedirs(output_dir, exist_ok=True)

    full_formula = f"{dependent_var} ~ {formula_rhs}"
    model = smf.mixedlm(full_formula, model_df, groups="subject_id").fit()
    print("Model fitted successfully!")
    print(model.summary())

    results_df = pd.DataFrame(
        {
            "predictor": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "conf_lower": model.conf_int().iloc[:, 0].values,
            "conf_upper": model.conf_int().iloc[:, 1].values,
        }
    )
    results_df["significant_05"] = results_df["p_value"] < 0.05
    results_df["significant_01"] = results_df["p_value"] < 0.01

    results_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # Calculate and save model metrics
    n_groups = model_df['subject_id'].nunique()
    
    # Try to get AIC/BIC from model, calculate manually if not available
    aic_value = model.aic if hasattr(model, 'aic') and not pd.isna(model.aic) else None
    bic_value = model.bic if hasattr(model, 'bic') and not pd.isna(model.bic) else None
    
    if aic_value is None:
        k = len(model.params)
        aic_value = -2 * model.llf + 2 * k
    
    if bic_value is None:
        k = len(model.params)
        n = model.nobs
        bic_value = -2 * model.llf + k * np.log(n)
        
    model_metrics = {
        'aic': aic_value,
        'bic': bic_value,
        'log_likelihood': model.llf,
        'n_observations': model.nobs,
        'n_groups': n_groups,
        'n_parameters': len(model.params),
        'converged': model.converged,
    }
    
    metrics_df = pd.DataFrame([model_metrics])
    metrics_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")

    # Save summary text
    summary_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(str(model.summary()))
    print(f"Summary saved to: {summary_file}")

    return results_df, model


def plot_mood_dimension(
    df_group: pd.DataFrame,
    df_ie: pd.DataFrame,
    dep: str,
    out_dir: str,
) -> None:
    """Create plots for one mood dimension.

    Panels
    ------
    1. Raincloud: group comparison (all blocks, raw mood).
    2. Raincloud: inclusion vs exclusion (baseline-corrected mood).
    3. Line: group × inclusion/exclusion interaction (normalized).
    4. Line: SART trajectory by group & order (raw mood).

    Parameters
    ----------
    df_group : pd.DataFrame
        Block-level data for group analyses.
    df_ie : pd.DataFrame
        Inclusion/exclusion dataset with normalized mood.
    dep : str
        Mood variable to plot.
    out_dir : str
        Directory where figures are saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_ie.columns
    ylabel_suffix = " (Normalized)" if has_normalized else ""

    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ------------------------------------------------------------------
    # Panel 1: Group comparison (raw mood)
    # ------------------------------------------------------------------
    df_agg_group = df_group.groupby(["subject_id", "group"])[dep].mean().reset_index()
    n_participants_by_group = (
        df_agg_group.groupby("group")["subject_id"].nunique().to_dict()
    )

    pt.RainCloud(
        x="group",
        y=dep,
        data=df_agg_group,
        palette=GROUP_COLORS,
        order=GROUP_ORDER,
        bw=0.2,
        width_viol=0.6,
        alpha=0.7,
        dodge=True,
        pointplot=True,
        move=-0.1,
        ax=ax1,
    )
    ax1.set_title(f"{dep}: Group Effect", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Group", fontsize=12, fontweight="bold")
    ax1.set_ylabel(dep, fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    for i, group in enumerate(GROUP_ORDER):
        n = n_participants_by_group.get(group, 0)
        ax1.text(
            i,
            ax1.get_ylim()[1] * 0.95,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # ------------------------------------------------------------------
    # Panel 2: Inclusion/Exclusion (normalized mood)
    # ------------------------------------------------------------------
    if has_normalized and not df_ie.empty:
        df_agg_ie = (
            df_ie.groupby(["subject_id", "inclusion_exclusion"])[dep_normalized]
            .mean()
            .reset_index()
        )
        n_participants_by_ie = (
            df_agg_ie.groupby("inclusion_exclusion")["subject_id"]
            .nunique()
            .to_dict()
        )

        pt.RainCloud(
            x="inclusion_exclusion",
            y=dep_normalized,
            data=df_agg_ie,
            palette=IE_COLORS,
            order=IE_ORDER,
            bw=0.2,
            width_viol=0.6,
            alpha=0.7,
            dodge=True,
            pointplot=True,
            move=-0.1,
            ax=ax2,
        )
        ax2.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Baseline",
        )
        ax2.set_title(
            f"{dep}: Inclusion/Exclusion Effect (Baseline-Corrected)",
            fontsize=16,
            fontweight="bold",
        )
        ax2.set_xlabel("Condition", fontsize=12, fontweight="bold")
        ax2.set_ylabel(f"{dep}{ylabel_suffix}", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc="best")

        for i, condition in enumerate(IE_ORDER):
            n = n_participants_by_ie.get(condition, 0)
            ax2.text(
                i,
                ax2.get_ylim()[1] * 0.95,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.8,
                ),
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "No normalized IE data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )

    # ------------------------------------------------------------------
    # Panel 3: Group × Inclusion/Exclusion interaction (normalized)
    # ------------------------------------------------------------------
    if has_normalized and not df_ie.empty:
        for group in df_ie["group"].dropna().unique():
            group_data = df_ie[df_ie["group"] == group]
            if len(group_data) == 0:
                continue
            ie_means = (
                group_data.groupby("inclusion_exclusion")[dep_normalized]
                .agg(["mean", "sem"])
                .reindex(IE_ORDER)
                .reset_index()
            )
            n_participants_group = group_data["subject_id"].nunique()
            color = (
                GROUP_COLORS[0]
                if group == GROUP_ORDER[0]
                else GROUP_COLORS[1]
            )
            x_positions = [0, 1]
            ax3.errorbar(
                x_positions,
                ie_means["mean"],
                yerr=ie_means["sem"],
                marker="o",
                linewidth=3,
                markersize=10,
                capsize=5,
                capthick=2,
                label=f"{group} (n={n_participants_group})",
                color=color,
                alpha=0.9,
            )
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(IE_ORDER, fontsize=12, fontweight="bold")
        ax3.set_ylabel(f"{dep}{ylabel_suffix}", fontsize=12, fontweight="bold")
        ax3.set_title(
            f"{dep}: Group × Inclusion/Exclusion (Baseline-Corrected)",
            fontsize=16,
            fontweight="bold",
        )
        ax3.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        ax3.legend(fontsize=10, loc="best")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No normalized IE data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
        )

    # ------------------------------------------------------------------
    # Panel 4: SART trajectory by group and order (raw mood)
    # ------------------------------------------------------------------
    if "sart_number" in df_group.columns:
        sart_order = [1, 2, 3, 4]
        x_positions = sart_order
        order_styles = {"IE": "-", "EI": "--"}

        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in df_group["group"].values:
                continue
            group_data = df_group[df_group["group"] == group]
            color = GROUP_COLORS[group_idx]

            orders = [
                o
                for o in ["IE", "EI"]
                if "order (IE/EI)" in group_data.columns
                and o in group_data["order (IE/EI)"].values
            ]
            if not orders:
                orders = [None]

            for order in orders:
                if order is not None:
                    order_data = group_data[
                        group_data["order (IE/EI)"] == order
                    ]
                    label_prefix = f"{group} - {order}"
                    linestyle = order_styles.get(order, "-")
                else:
                    order_data = group_data
                    label_prefix = group
                    linestyle = "-"

                means = []
                sems = []
                for s in sart_order:
                    task_label = f"Sart{s}" if isinstance(s, int) else s
                    subset = order_data[order_data["task"] == task_label]
                    if len(subset) > 0:
                        means.append(subset[dep].mean())
                        sems.append(subset[dep].sem())
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)

                n_subjects = order_data["subject_id"].nunique()
                label = f"{label_prefix} (n={n_subjects})"

                ax4.errorbar(
                    x_positions,
                    means,
                    yerr=sems,
                    marker="o",
                    linewidth=2.5,
                    markersize=8,
                    linestyle=linestyle,
                    capsize=5,
                    capthick=2,
                    alpha=0.8,
                    color=color,
                    label=label,
                )

        ax4.set_xticks(x_positions)
        ax4.set_xticklabels([
            f"Sart{s}" for s in sart_order
        ], fontsize=11, fontweight="bold")
        ax4.set_xlabel("SART Task", fontsize=12, fontweight="bold")
        ax4.set_ylabel(dep, fontsize=12, fontweight="bold")
        ax4.set_title(
            f"{dep}: SART Trajectory by Group & Order",
            fontsize=16,
            fontweight="bold",
        )
        ax4.legend(fontsize=9, loc="best")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "No SART number information",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
        )

    plt.suptitle(
        f"Block-Level Mood Analysis: {dep}", fontsize=18, fontweight="bold"
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = os.path.join(out_dir, f"{dep}_mood_analysis.png")
    out_svg = os.path.join(out_dir, f"{dep}_mood_analysis.svg")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.close(fig)


def descriptive_statistics(
    df_group: pd.DataFrame,
    df_ie: pd.DataFrame,
    dep: str,
    out_dir: str,
) -> None:
    """Compute and save descriptive statistics for one mood dimension.

    Parameters
    ----------
    df_group : pd.DataFrame
        Block-level data for group analysis.
    df_ie : pd.DataFrame
        Inclusion/exclusion dataset with normalized mood.
    dep : str
        Mood variable.
    out_dir : str
        Output directory for CSV export.
    """
    os.makedirs(out_dir, exist_ok=True)

    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_ie.columns

    overall_stats = df_group[dep].describe()
    print(f"Overall {dep}: mean = {overall_stats['mean']:.3f}, SD = {overall_stats['std']:.3f}")

    print("\nBy group (raw):")
    group_stats = (
        df_group.groupby("group")[dep].agg(["count", "mean", "std"]).round(3)
    )
    print(group_stats)

    if not df_ie.empty:
        print("\nBy inclusion/exclusion (raw, post-manipulation blocks only):")
        ie_stats_raw = (
            df_ie.groupby("inclusion_exclusion")[dep]
            .agg(["count", "mean", "std"])
            .round(3)
        )
        print(ie_stats_raw)

        if has_normalized:
            print("\nBy inclusion/exclusion (normalized, post - baseline):")
            ie_stats_norm = (
                df_ie.groupby("inclusion_exclusion")[dep_normalized]
                .agg(["count", "mean", "std"])
                .round(3)
            )
            print(ie_stats_norm)

    # Cell-wise stats for IE subset (raw and normalized if available)
    stats_list: List[Dict[str, float]] = []
    if not df_ie.empty:
        for group in df_ie["group"].dropna().unique():
            for ie in df_ie["inclusion_exclusion"].dropna().unique():
                subset = df_ie[
                    (df_ie["group"] == group)
                    & (df_ie["inclusion_exclusion"] == ie)
                ]
                if len(subset) == 0:
                    continue
                entry: Dict[str, float] = {
                    "Group": group,
                    "Inclusion_Exclusion": ie,
                    "N": float(len(subset)),
                    "Mean_Raw": float(subset[dep].mean()),
                    "SD_Raw": float(subset[dep].std()),
                    "SE_Raw": float(subset[dep].sem()),
                }
                if has_normalized:
                    entry.update(
                        {
                            "Mean_Normalized": float(
                                subset[dep_normalized].mean()
                            ),
                            "SD_Normalized": float(
                                subset[dep_normalized].std()
                            ),
                            "SE_Normalized": float(
                                subset[dep_normalized].sem()
                            ),
                        }
                    )
                stats_list.append(entry)

    stats_df = pd.DataFrame(stats_list)
    stats_file = os.path.join(out_dir, f"{dep}_descriptive_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"Descriptive statistics saved to: {stats_file}")


def analyze_mood_dimension(
    df_group: pd.DataFrame,
    df_ie: pd.DataFrame,
    dep: str,
) -> None:
    """Run the full pipeline (LMMs + plots + descriptives) for one mood variable.

    Parameters
    ----------
    df_group : pd.DataFrame
        Block-level dataset for group analyses.
    df_ie : pd.DataFrame
        Inclusion/exclusion dataset with normalized mood.
    dep : str
        Mood variable name.
    """
    print("\n" + "=" * 60)
    print(f"BLOCK-LEVEL MOOD ANALYSIS FOR {dep}")
    print("=" * 60)

    dim_results_dir = os.path.join(RESULTS_DIR, dep)
    dim_plots_dir = os.path.join(PLOTS_DIR, dep)
    os.makedirs(dim_results_dir, exist_ok=True)
    os.makedirs(dim_plots_dir, exist_ok=True)

    # Model 1: Group effect (raw mood, all blocks)
    run_lmm_analysis(df_group, dep, "group", "group_effect", dim_results_dir)

    # Model 2: Inclusion/Exclusion effect (normalized, post-manip blocks)
    dep_normalized = f"{dep}_normalized"
    if dep_normalized in df_ie.columns:
        run_lmm_analysis(
            df_ie,
            dep_normalized,
            "inclusion_exclusion",
            "inclusion_exclusion_effect",
            dim_results_dir,
        )

        # Model 3: Group × Inclusion/Exclusion interaction (normalized)
        run_lmm_analysis(
            df_ie,
            dep_normalized,
            "group * inclusion_exclusion",
            "group_ie_interaction",
            dim_results_dir,
        )
    else:
        print(
            f"Skipping IE models for {dep} (no normalized column {dep_normalized})"
        )

    # Visualization and descriptives
    print("\n" + "=" * 60)
    print(f"CREATING VISUALIZATIONS FOR {dep}")
    print("=" * 60)
    plot_mood_dimension(df_group, df_ie, dep, dim_plots_dir)

    print("\n" + "=" * 60)
    print(f"DESCRIPTIVE STATISTICS FOR {dep}")
    print("=" * 60)
    descriptive_statistics(df_group, df_ie, dep, dim_results_dir)

    print(f"\nCompleted mood dimension: {dep}")
    print(f"Results saved to: {dim_results_dir}")
    print(f"Plots saved to:   {dim_plots_dir}")


def main() -> None:
    """Entry point for the block-level mood LMM analysis."""
    print("\n" + "=" * 60)
    print("BLOCK-LEVEL MOOD LMM ANALYSIS")
    print("=" * 60)

    ensure_directories()

    df_block = load_block_level_mood_data()
    df_group, df_ie = preprocess_mood_data(df_block, MOOD_DIMENSIONS)

    for dep in MOOD_DIMENSIONS:
        if dep in df_block.columns:
            analyze_mood_dimension(df_group, df_ie, dep)
        else:
            print(f"Skipping {dep}: column not found in block-level data")

    print("\n" + "=" * 60)
    print("MOOD ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results base dir: {RESULTS_DIR}")
    print(f"Plots base dir:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()
