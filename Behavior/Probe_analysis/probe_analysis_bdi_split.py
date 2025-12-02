#%%
from __future__ import annotations

"""
BDI-Split Multidimensional Probe Analysis (LMM + Plots + PCA)

This script replicates the functionality of ``probe_analysis_multidim.py`` but:

- Splits the original ``Controls`` group into two subgroups based on BDI
  (within-controls median) and keeps the ``Risk of Depression`` group, yielding
  three groups in total:
    - ``Controls Low BDI``
    - ``Controls High BDI``
    - ``Risk of Depression``
- Adds ``onoff`` as an additional dimension.
- Optionally runs analyses on the full onoff range and on the subset
  ``onoff < ONOFF_MAX_EXCLUSIVE``.
- Integrates principal components (PC1–PC3) from the PCA pipeline
  (``results/Behavior/probe_data/pca_results.csv``) using the same
  three-group structure.
- Performs pairwise group comparisons on subject-level means for each
  dependent variable when there is a group factor.

Design notes
------------
- Configuration lives at the top of the script. Adjust paths and settings there.
- This script is plug-and-play: modify configuration variables and run.
- The probe-level analyses use ``probe_level_aggregated_data.csv``.
- PCA-based analyses use ``pca_results.csv`` (already filtered to onoff < 50
  in the PCA script).
"""

import os
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import statsmodels.formula.api as smf
from scipy import stats as sp_stats


# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# Input probe-level aggregated data (without PCA components)
DATA_FILE: str = "../../results/Behavior/probe_data/probe_level_aggregated_data.csv"

# PCA LMM results file (contains PC1–PC3 and original dimensions for a subset)
PCA_RESULTS_FILE: str = "../../results/Behavior/probe_data/pca_results.csv"

# Dimensions to analyze from the probe-level file
BASE_DIMENSIONS: List[str] = ["valence", "time", "selfother", "confidence", "onoff"]

# PCA component dimensions to analyze from PCA_RESULTS_FILE
PCA_DIMENSIONS: List[str] = ["PC1", "PC2", "PC3"]

# Output base directories for results and plots (will be created if missing)
BASE_RESULTS_DIR: str = "../../results/Behavior/bdi_split/lmm_analysis_multidim"
BASE_PLOTS_DIR: str = "../../results/Behavior/bdi_split/lmm_plots_multidim"

# Plot aesthetics for 3-group design
GROUP_COLUMN: str = "group_bdi3"
GROUP_ORDER: List[str] = [
    "Controls Low BDI",
    "Controls High BDI",
    "Risk of Depression",
]
GROUP_COLORS: List[str] = ["#2E86AB", "#A3A1FB", "#F24236"]

IE_ORDER: List[str] = ["inclusion", "exclusion"]
IE_COLORS: List[str] = ["#A23B72", "#F18F01"]

# On/off filtering
RUN_FULL_ONOFF_RANGE: bool = True
RUN_ONOFF_LT50: bool = True
ONOFF_MAX_EXCLUSIVE: float = 50.0

# Optional filter: exclude baseline condition from inclusion/exclusion analysis
EXCLUDE_BASELINE: bool = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_directories(*dirs: str) -> None:
    """Create output directories if they do not exist.

    Parameters
    ----------
    *dirs : str
        One or more directory paths to create.
    """
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_probe_data(path: str) -> pd.DataFrame:
    """Load the probe-level aggregated CSV.

    Parameters
    ----------
    path : str
        Path to the probe-level aggregated CSV.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_FILE not found: {path}")
    return pd.read_csv(path)


def load_pca_results(path: str) -> pd.DataFrame:
    """Load the PCA results CSV (with PC1–PC3).

    Parameters
    ----------
    path : str
        Path to the PCA results CSV.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCA_RESULTS_FILE not found: {path}")
    return pd.read_csv(path)


def build_bdi_group_mapping(df: pd.DataFrame) -> Dict[int, str]:
    """Build mapping from ``subject_id`` to 3-level BDI group label.

    The mapping is constructed as follows:

    - Within subjects whose original ``group`` is ``Controls`` and who have a
      non-missing BDI value, compute the median BDI across subjects.
    - Controls with BDI <= median → ``Controls Low BDI``.
    - Controls with BDI > median → ``Controls High BDI``.
    - Subjects with original ``group`` == ``Risk of Depression`` →
      ``Risk of Depression``.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level dataframe with columns ``subject_id``, ``group`` and ``bdi``.

    Returns
    -------
    Dict[int, str]
        Mapping from subject_id to 3-level group label.
    """
    required_cols = {"subject_id", "group", "bdi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input dataframe missing required columns: {sorted(missing)}")

    # Subject-level BDI (assumes constant within subject)
    subj_bdi = (
        df.dropna(subset=["bdi"])
        .groupby("subject_id")["bdi"]
        .first()
    )

    subj_group = (
        df.groupby("subject_id")["group"]
        .first()
    )

    controls_mask = subj_group == "Controls"
    controls_bdi = subj_bdi[controls_mask].dropna()
    if controls_bdi.empty:
        raise ValueError("No controls with non-missing BDI available for median split.")

    bdi_median = controls_bdi.median()
    print("\nBDI median within Controls:", bdi_median)

    mapping: Dict[int, str] = {}
    for subj_id, grp in subj_group.items():
        if grp == "Controls":
            bdi_value = subj_bdi.get(subj_id, np.nan)
            if np.isnan(bdi_value):
                # Keep as unmapped; will be dropped later
                continue
            if bdi_value <= bdi_median:
                mapping[subj_id] = "Controls Low BDI"
            else:
                mapping[subj_id] = "Controls High BDI"
        elif grp == "Risk of Depression":
            mapping[subj_id] = "Risk of Depression"
        else:
            # If other groups exist, leave unmapped for now
            continue

    # Report group sizes
    group_counts: Dict[str, int] = {}
    for label in mapping.values():
        group_counts[label] = group_counts.get(label, 0) + 1

    print("Subject counts by BDI-split group (mapped subjects only):")
    for g in GROUP_ORDER:
        n = group_counts.get(g, 0)
        print(f"  {g}: {n} subjects")

    return mapping


def add_bdi_group_column(df: pd.DataFrame, mapping: Dict[int, str]) -> pd.DataFrame:
    """Add the 3-level BDI group column to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ``subject_id``.
    mapping : Dict[int, str]
        Mapping from subject_id to 3-level group label.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with the new ``GROUP_COLUMN``.
    """
    out = df.copy()
    out[GROUP_COLUMN] = out["subject_id"].map(mapping)
    # Drop rows without mapping (e.g., missing BDI or unexpected original group)
    before_rows = len(out)
    out = out.dropna(subset=[GROUP_COLUMN])
    after_rows = len(out)
    if before_rows > after_rows:
        print(
            f"Dropped {before_rows - after_rows} rows without BDI-split group label "
            f"({GROUP_COLUMN})."
        )
    return out


def validate_columns(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dimensions: Iterable[str],
) -> None:
    """Validate that required columns exist in the input data frames.

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data.
    dimensions : Iterable[str]
        Dependent variable names to verify.
    """
    lmm_required = {"subject_id", GROUP_COLUMN}
    ie_required = {"subject_id", GROUP_COLUMN, "inclusion_exclusion"}

    missing_lmm = lmm_required - set(df_lmm.columns)
    missing_ie = ie_required - set(df_lmm_ie.columns)
    if missing_lmm:
        raise ValueError(f"df_lmm is missing required columns: {sorted(missing_lmm)}")
    if missing_ie:
        raise ValueError(f"df_lmm_ie is missing required columns: {sorted(missing_ie)}")

    for dep in dimensions:
        if dep not in df_lmm.columns:
            raise ValueError(f"Dependent variable '{dep}' not in df_lmm columns")
        if dep not in df_lmm_ie.columns:
            raise ValueError(f"Dependent variable '{dep}' not in df_lmm_ie columns")


def run_lmm_analysis(
    data: pd.DataFrame,
    dependent_var: str,
    formula_rhs: str,
    model_name: str,
    output_dir: str,
    categorical_cols: Optional[List[str]] = None,
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
    categorical_cols : list of str, optional
        Columns to cast as categorical before fitting.

    Returns
    -------
    Tuple[pd.DataFrame, object]
        results_df, fitted_model
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== FITTING MODEL: {model_name} ({dependent_var}) ===")
    print(f"Formula: {dependent_var} ~ {formula_rhs}")
    print(
        f"Sample size: {len(data)} observations from {data['subject_id'].nunique()} subjects"
    )

    # Prepare clean data subset to avoid indexing issues
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
        raise ValueError(f"Missing required columns for model {model_name}: {sorted(missing)}")

    model_df = data.loc[:, sorted(required_cols)].copy()

    # Cast selected columns to category if present
    if categorical_cols is None:
        categorical_cols = []
    for cat_col in categorical_cols:
        if cat_col in model_df.columns:
            if cat_col == GROUP_COLUMN:
                # Ensure BDI-split group uses GROUP_ORDER with "Controls Low BDI" as reference
                model_df[cat_col] = pd.Categorical(
                    model_df[cat_col],
                    categories=GROUP_ORDER,
                    ordered=True,
                )
            else:
                model_df[cat_col] = model_df[cat_col].astype("category")

    # Identify and report rows with NA in required columns
    before_n = len(model_df)
    na_mask = model_df.isna().any(axis=1)
    dropped_df = model_df.loc[na_mask].copy()

    # Drop rows with NA in required columns
    model_df = (
        model_df.dropna(axis=0, how="any").sort_values(["subject_id"]).reset_index(drop=True)
    )
    after_n = len(model_df)
    num_dropped = before_n - after_n
    if num_dropped > 0:
        na_counts = dropped_df.isna().sum()
        subj_counts = dropped_df["subject_id"].value_counts().sort_index()

        print(
            f"Dropped {num_dropped} rows with missing data for model '{model_name}' "
            f"({dependent_var})."
        )
        print("- Missing by column (only >0 shown):")
        for col, cnt in na_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")
        print("- Dropped rows by subject_id:")
        for sid, cnt in subj_counts.items():
            print(f"  {sid}: {cnt}")

        dropped_path = os.path.join(
            output_dir, f"dropped_rows_{model_name}_{dependent_var}.csv"
        )
        dropped_df.to_csv(dropped_path, index=False)
        print(f"- Saved dropped rows to: {dropped_path}")

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

    # Extract model fit metrics
    n_groups = model_df["subject_id"].nunique()
    aic_value = model.aic if hasattr(model, "aic") and not pd.isna(model.aic) else None
    bic_value = model.bic if hasattr(model, "bic") and not pd.isna(model.bic) else None

    if aic_value is None:
        k = len(model.params)
        aic_value = -2 * model.llf + 2 * k
        print(f"Note: AIC calculated manually: {aic_value:.3f}")

    if bic_value is None:
        k = len(model.params)
        n = model.nobs
        bic_value = -2 * model.llf + k * np.log(n)
        print(f"Note: BIC calculated manually: {bic_value:.3f}")

    model_metrics = {
        "aic": aic_value,
        "bic": bic_value,
        "log_likelihood": model.llf,
        "log_likelihood_restricted": model.llf_fe if hasattr(model, "llf_fe") else None,
        "n_observations": model.nobs,
        "n_groups": n_groups,
        "n_parameters": len(model.params),
        "n_fixed_effects": len(model.fe_params) if hasattr(model, "fe_params") else None,
        "n_random_effects": len(model.cov_re) if hasattr(model, "cov_re") else None,
        "converged": model.converged,
        "scale": model.scale if hasattr(model, "scale") else None,
        "deviance": -2 * model.llf if model.llf is not None else None,
        "rsquared_within": None,
        "rsquared_between": None,
    }

    print("\n=== MODEL FIT METRICS ===")
    print(f"AIC: {model_metrics['aic']:.3f}")
    print(f"BIC: {model_metrics['bic']:.3f}")
    print(f"Log-Likelihood: {model_metrics['log_likelihood']:.3f}")
    print(f"N observations: {model_metrics['n_observations']}")
    print(f"N groups: {model_metrics['n_groups']}")
    print(f"Converged: {model_metrics['converged']}")
    if model_metrics["scale"] is not None:
        print(f"Scale: {model_metrics['scale']:.6f}")

    results_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_results.csv")
    results_df.to_csv(results_file, index=False)

    metrics_df = pd.DataFrame([model_metrics])
    metrics_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)

    summary_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(str(model.summary()))
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("MODEL FIT METRICS\n")
        f.write("=" * 60 + "\n")
        for key, value in model_metrics.items():
            if value is not None:
                if isinstance(value, float):
                    f.write(f"{key.upper()}: {value:.6f}\n")
                else:
                    f.write(f"{key.upper()}: {value}\n")
            else:
                f.write(f"{key.upper()}: Not available\n")

    print(f"Results saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Enhanced summary saved to: {summary_file}")

    return results_df, model


def pairwise_group_comparisons(
    data: pd.DataFrame,
    dep: str,
    group_col: str,
    out_dir: str,
    label: str,
) -> None:
    """Compute pairwise group comparisons on subject-level means.

    For each pair of groups present in ``group_col``, this function computes
    subject-level means of ``dep`` and then performs Welch's t-test.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing ``subject_id``, group column, and dependent var.
    dep : str
        Dependent variable name.
    group_col : str
        Name of the group factor column.
    out_dir : str
        Directory where results will be saved.
    label : str
        Short label for the analysis (e.g., "overall", "IE_normalized").
    """
    os.makedirs(out_dir, exist_ok=True)

    if group_col not in data.columns:
        print(f"Skipping pairwise comparisons for {dep}: '{group_col}' missing.")
        return

    # Subject-level aggregation
    agg = (
        data.groupby([group_col, "subject_id"])[dep]
        .mean()
        .reset_index()
        .dropna(subset=[dep])
    )

    unique_groups = [g for g in GROUP_ORDER if g in agg[group_col].unique()]
    if len(unique_groups) < 2:
        print(f"Not enough groups for pairwise comparisons for {dep}.")
        return

    rows: List[Dict[str, float]] = []
    for g1, g2 in combinations(unique_groups, 2):
        vals1 = agg.loc[agg[group_col] == g1, dep].values
        vals2 = agg.loc[agg[group_col] == g2, dep].values
        if len(vals1) < 2 or len(vals2) < 2:
            continue
        t_stat, p_val = sp_stats.ttest_ind(vals1, vals2, equal_var=False)

        rows.append(
            {
                "analysis_label": label,
                "dimension": dep,
                "group1": g1,
                "group2": g2,
                "n1": int(len(vals1)),
                "n2": int(len(vals2)),
                "mean1": float(np.mean(vals1)),
                "mean2": float(np.mean(vals2)),
                "sd1": float(np.std(vals1, ddof=1)),
                "sd2": float(np.std(vals2, ddof=1)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            }
        )

    if not rows:
        print(f"No valid pairwise comparisons computed for {dep}.")
        return

    results_df = pd.DataFrame(rows)
    # Bonferroni correction within this analysis
    m = len(results_df)
    results_df["p_bonferroni"] = np.minimum(results_df["p_value"] * m, 1.0)
    results_df["significant_05"] = results_df["p_value"] < 0.05
    results_df["significant_bonferroni"] = results_df["p_bonferroni"] < 0.05

    # Console summary for quick inspection
    display_cols = [
        "group1",
        "group2",
        "mean1",
        "mean2",
        "t_statistic",
        "p_value",
        "p_bonferroni",
        "significant_bonferroni",
    ]
    available_cols = [c for c in display_cols if c in results_df.columns]
    print("\nPAIRWISE GROUP COMPARISONS -", dep, f"({label})")
    print(results_df[available_cols].round(3).to_string(index=False))

    out_file = os.path.join(out_dir, f"pairwise_group_comparisons_{dep}_{label}.csv")
    results_df.to_csv(out_file, index=False)

    print(f"Pairwise group comparisons for {dep} ({label}) saved to: {out_file}")


# =============================================================================
# PLOTTING
# =============================================================================


def plot_dimension(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dep: str,
    out_dir: str,
) -> None:
    """Create a comprehensive 2x3 figure combining all key analyses.

    Grid layout:
    - Row 1: Raincloud (Group) | Raincloud (I/E) | Group × I/E Interaction
    - Row 2: Time by Group | Intervention Distance by I/E | SART Trajectories

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data.
    dep : str
        Dependent variable column to plot.
    out_dir : str
        Output directory for saved plots.
    """
    os.makedirs(out_dir, exist_ok=True)

    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    dep_ie = dep_normalized if has_normalized else dep
    ylabel_suffix = " (Normalized)" if has_normalized else ""

    plt.style.use("default")
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # ---------------------------------------------------------------------
    # Row 1: DISTRIBUTION COMPARISONS
    # ---------------------------------------------------------------------

    # Plot 1: Raincloud for group comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_agg_group = df_lmm.groupby(["subject_id", GROUP_COLUMN])[dep].mean().reset_index()
    n_participants_by_group = (
        df_agg_group.groupby(GROUP_COLUMN)["subject_id"].nunique().to_dict()
    )

    pt.RainCloud(
        x=GROUP_COLUMN,
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
    ax1.set_title(
        f"{dep.upper()}: Group Effect",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"{dep.title()} Score", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    for i, group in enumerate(GROUP_ORDER):
        n = n_participants_by_group.get(group, 0)
        ax1.text(
            i,
            ax1.get_ylim()[1] * 0.95,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Plot 2: Raincloud for inclusion/exclusion
    ax2 = fig.add_subplot(gs[0, 1])
    df_agg_ie = (
        df_lmm_ie.groupby(["subject_id", "inclusion_exclusion"])[dep_ie]
        .mean()
        .reset_index()
    )
    n_participants_by_ie = (
        df_agg_ie.groupby("inclusion_exclusion")["subject_id"].nunique().to_dict()
    )

    pt.RainCloud(
        x="inclusion_exclusion",
        y=dep_ie,
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
    title_suffix = " (Baseline-Corrected)" if has_normalized else ""
    ax2.set_title(
        f"{dep.upper()}: Inclusion/Exclusion Effect{title_suffix}",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax2.set_ylabel(f"{dep.title()}{ylabel_suffix}", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    if has_normalized:
        ax2.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Baseline",
        )
        ax2.legend(fontsize=10, loc="best")

    for i, condition in enumerate(IE_ORDER):
        n = n_participants_by_ie.get(condition, 0)
        ax2.text(
            i,
            ax2.get_ylim()[1] * 0.95,
            f"n={n}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Plot 3: Interaction plot (Group × I/E)
    ax3 = fig.add_subplot(gs[0, 2])
    for group in df_lmm_ie[GROUP_COLUMN].dropna().unique():
        group_data = df_lmm_ie[df_lmm_ie[GROUP_COLUMN] == group]
        if len(group_data) == 0:
            continue
        ie_means = (
            group_data.groupby("inclusion_exclusion")[dep_ie]
            .agg(["mean", "sem"])
            .reindex(IE_ORDER)
            .reset_index()
        )
        n_participants_group = group_data["subject_id"].nunique()
        color = GROUP_COLORS[GROUP_ORDER.index(group)] if group in GROUP_ORDER else "gray"
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
    ax3.set_xticklabels(IE_ORDER, fontsize=14, fontweight="bold")
    ax3.set_ylabel(f"{dep.title()}{ylabel_suffix}", fontsize=14, fontweight="bold")
    ax3.set_title(
        f"{dep.upper()}: Group × I/E Interaction{title_suffix}",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    if has_normalized:
        ax3.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Baseline",
        )
    ax3.legend(fontsize=12, title_fontsize=12, loc="best")
    ax3.grid(True, alpha=0.3)

    # ---------------------------------------------------------------------
    # Row 2: TIME-ON-TASK TRAJECTORIES
    # ---------------------------------------------------------------------

    # Plot 4: Time-on-task by group
    ax4 = fig.add_subplot(gs[1, 0])
    if "time_on_task" in df_lmm.columns:
        for i, group in enumerate(GROUP_ORDER):
            if group in df_lmm[GROUP_COLUMN].values:
                group_data = df_lmm[df_lmm[GROUP_COLUMN] == group]
                time_group_agg = (
                    group_data.groupby("time_on_task")[dep]
                    .agg(["mean", "sem"])
                    .reset_index()
                )
                color = GROUP_COLORS[i]
                ax4.errorbar(
                    time_group_agg["time_on_task"],
                    time_group_agg["mean"],
                    yerr=time_group_agg["sem"],
                    marker="o",
                    linewidth=2.5,
                    markersize=6,
                    capsize=3,
                    alpha=0.8,
                    color=color,
                    label=group,
                )
        ax4.set_xlabel("Time on Task (Probe Number)", fontsize=14, fontweight="bold")
        ax4.set_ylabel(f"{dep.title()} Score", fontsize=14, fontweight="bold")
        ax4.set_title(
            f"{dep.upper()}: Time-on-Task by Group",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax4.legend(fontsize=12, loc="best")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(
            0.5,
            0.5,
            "Time-on-task data not available",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=14,
        )

    # Plot 5: Intervention distance by I/E (probe_number within block)
    ax5 = fig.add_subplot(gs[1, 1])
    if "probe_number" in df_lmm_ie.columns:
        for i, condition in enumerate(IE_ORDER):
            if condition in df_lmm_ie["inclusion_exclusion"].values:
                ie_data = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == condition]
                probe_ie_agg = (
                    ie_data.groupby("probe_number")[dep_ie]
                    .agg(["mean", "sem"])
                    .reset_index()
                )
                color = IE_COLORS[i]
                ax5.errorbar(
                    probe_ie_agg["probe_number"],
                    probe_ie_agg["mean"],
                    yerr=probe_ie_agg["sem"],
                    marker="s",
                    linewidth=2.5,
                    markersize=6,
                    capsize=3,
                    alpha=0.8,
                    color=color,
                    label=condition.capitalize(),
                )
        ax5.set_xlabel(
            "Intervention Distance (Probe Number)",
            fontsize=14,
            fontweight="bold",
        )
        ax5.set_ylabel(f"{dep.title()}{ylabel_suffix}", fontsize=14, fontweight="bold")
        ax5.set_title(
            f"{dep.upper()}: Intervention Distance by I/E{title_suffix}",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        if has_normalized:
            ax5.axhline(
                y=0,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Baseline",
            )
        ax5.legend(fontsize=12, loc="best")
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, 15.5)
    else:
        ax5.text(
            0.5,
            0.5,
            "Probe number data not available",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=14,
        )

    # Plot 6: SART mean trajectories by group and order
    ax6 = fig.add_subplot(gs[1, 2])
    if "task" in df_lmm.columns and "order (IE/EI)" in df_lmm.columns:
        sart_order = ["Sart1", "Sart2", "Sart3", "Sart4"]
        x_positions = [1, 2, 3, 4]
        order_styles = {"IE": "-", "EI": "--"}

        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in df_lmm[GROUP_COLUMN].values:
                continue
            group_data = df_lmm[df_lmm[GROUP_COLUMN] == group]
            color = GROUP_COLORS[group_idx]

            for order in ["IE", "EI"]:
                if order not in group_data["order (IE/EI)"].values:
                    continue
                order_data = group_data[group_data["order (IE/EI)"] == order]
                n_subjects = order_data["subject_id"].nunique()

                means: List[float] = []
                sems: List[float] = []
                for sart in sart_order:
                    sart_data = order_data[order_data["task"] == sart]
                    if len(sart_data) > 0:
                        means.append(sart_data[dep].mean())
                        sems.append(sart_data[dep].sem())
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)

                linestyle = order_styles[order]
                label = f"{group} - {order} (n={n_subjects})"

                line = ax6.errorbar(
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

        ax6.set_xticks(x_positions)
        ax6.set_xticklabels(sart_order, fontsize=12, fontweight="bold")
        ax6.set_xlabel("SART Task", fontsize=14, fontweight="bold")
        ax6.set_ylabel(f"{dep.title()} Score", fontsize=14, fontweight="bold")
        ax6.set_title(
            f"{dep.upper()}: SART Trajectory by Group & Order\n(Solid=IE, Dashed=EI)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        handles, labels = ax6.get_legend_handles_labels()
        new_handles = []
        for handle in handles:
            if hasattr(handle, "get_children"):
                lines = [child for child in handle.get_children() if hasattr(child, "set_marker")]
                if lines:
                    line = lines[0]
                    line.set_marker("")
                    line.set_markersize(0)
                    new_handles.append(line)
            else:
                handle.set_marker("")
                handle.set_markersize(0)
                new_handles.append(handle)
        ax6.legend(new_handles, labels, fontsize=11, loc="best")
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0.5, 4.5)
    else:
        ax6.text(
            0.5,
            0.5,
            "Order data not available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )

    # ---------------------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------------------
    plt.suptitle(
        f"Comprehensive Analysis: {dep.upper()}",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )

    out_png = os.path.join(out_dir, f"{dep}_comprehensive_analysis.png")
    out_svg = os.path.join(out_dir, f"{dep}_comprehensive_analysis.svg")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_svg, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================


def descriptive_statistics(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dep: str,
    out_dir: str,
) -> None:
    """Compute and save descriptive statistics for a dependent variable.

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data.
    dep : str
        Dependent variable column.
    out_dir : str
        Output directory for CSV export.
    """
    os.makedirs(out_dir, exist_ok=True)

    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    dep_ie = dep_normalized if has_normalized else dep

    overall_stats = df_lmm[dep].describe()
    print(f"Overall: Mean = {overall_stats['mean']:.3f}, SD = {overall_stats['std']:.3f}")

    print("\nBy Group (BDI-split):")
    group_stats = (
        df_lmm.groupby(GROUP_COLUMN)[dep]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    print(group_stats)

    print("\nBy Inclusion/Exclusion (Raw):")
    ie_stats = (
        df_lmm_ie.groupby("inclusion_exclusion")[dep]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    print(ie_stats)

    if has_normalized:
        print("\nBy Inclusion/Exclusion (Baseline-Corrected):")
        ie_stats_norm = (
            df_lmm_ie.groupby("inclusion_exclusion")[dep_ie]
            .agg(["count", "mean", "std"])
            .round(3)
        )
        print(ie_stats_norm)

    print("\nBy Group × Inclusion/Exclusion (Raw):")
    interaction_stats = (
        df_lmm_ie.groupby([GROUP_COLUMN, "inclusion_exclusion"])[dep]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    print(interaction_stats)

    if has_normalized:
        print("\nBy Group × Inclusion/Exclusion (Baseline-Corrected):")
        interaction_stats_norm = (
            df_lmm_ie.groupby([GROUP_COLUMN, "inclusion_exclusion"])[dep_ie]
            .agg(["count", "mean", "std"])
            .round(3)
        )
        print(interaction_stats_norm)

    stats_list: List[Dict[str, float]] = []
    for group in df_lmm_ie[GROUP_COLUMN].dropna().unique():
        for ie in df_lmm_ie["inclusion_exclusion"].dropna().unique():
            subset = df_lmm_ie[
                (df_lmm_ie[GROUP_COLUMN] == group)
                & (df_lmm_ie["inclusion_exclusion"] == ie)
            ]
            if len(subset) > 0:
                stats_dict = {
                    "Group": group,
                    "Inclusion_Exclusion": ie,
                    "N": int(len(subset)),
                    "Mean_Raw": float(subset[dep].mean()),
                    "SD_Raw": float(subset[dep].std()),
                    "SE_Raw": float(subset[dep].sem()),
                }
                if has_normalized:
                    stats_dict.update(
                        {
                            "Mean_Normalized": float(subset[dep_ie].mean()),
                            "SD_Normalized": float(subset[dep_ie].std()),
                            "SE_Normalized": float(subset[dep_ie].sem()),
                        }
                    )
                stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    stats_file = os.path.join(out_dir, f"{dep}_descriptive_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    suffix_msg = " (raw and normalized)" if has_normalized else ""
    print(f"Descriptive statistics{suffix_msg} saved to: {stats_file}")


# =============================================================================
# MAIN ANALYSIS PIPELINE FOR A SET OF DIMENSIONS
# =============================================================================


def baseline_normalize_ie(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dimensions: Iterable[str],
) -> pd.DataFrame:
    """Apply baseline normalization to the IE subset for given dimensions.

    Normalization is defined per subject and dimension as:

    - For Sart2: value - Sart1 mean
    - For Sart4: value - Sart3 mean

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Full probe-level dataframe (used to compute baselines).
    df_lmm_ie : pd.DataFrame
        IE subset where normalized columns will be added.
    dimensions : Iterable[str]
        List of dimensions to normalize.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_lmm_ie`` with ``<dim>_normalized`` columns added.
    """
    df_ie = df_lmm_ie.copy()

    print("\n" + "=" * 60)
    print("APPLYING BASELINE NORMALIZATION")
    print("=" * 60)

    baseline_means: Dict[Tuple[int, str, str], float] = {}
    for subject in df_lmm["subject_id"].unique():
        subject_data = df_lmm[df_lmm["subject_id"] == subject]

        sart1_data = subject_data[subject_data["task"] == "Sart1"]
        if len(sart1_data) > 0:
            for dim in dimensions:
                if dim in sart1_data.columns:
                    baseline_means[(subject, "Sart1", dim)] = sart1_data[dim].mean()

        sart3_data = subject_data[subject_data["task"] == "Sart3"]
        if len(sart3_data) > 0:
            for dim in dimensions:
                if dim in sart3_data.columns:
                    baseline_means[(subject, "Sart3", dim)] = sart3_data[dim].mean()

    print(
        "Calculated baseline means for",
        len(baseline_means),
        "subject-SART-dimension combinations",
    )

    def _normalize_row(row: pd.Series, dimension: str) -> float:
        subject = row["subject_id"]
        task = row["task"]
        if task == "Sart2":
            baseline_key = (subject, "Sart1", dimension)
        elif task == "Sart4":
            baseline_key = (subject, "Sart3", dimension)
        else:
            return np.nan

        baseline_mean = baseline_means.get(baseline_key, np.nan)
        if pd.isna(baseline_mean):
            return np.nan
        return float(row[dimension] - baseline_mean)

    for dim in dimensions:
        if dim in df_ie.columns:
            norm_col = f"{dim}_normalized"
            df_ie[norm_col] = df_ie.apply(lambda row: _normalize_row(row, dim), axis=1)

    # Note: We do NOT drop rows here - that's done in step 5 of run_analysis_for_dataset
    # This allows the filtering to happen in the correct order (after baseline normalization)
    normalized_cols = [f"{dim}_normalized" for dim in dimensions if f"{dim}_normalized" in df_ie.columns]
    
    # Report statistics
    print(f"Normalized IE subset: {len(df_ie)} observations")
    for dim in dimensions:
        col = f"{dim}_normalized"
        if col in df_ie.columns:
            print(
                f"  {col} range: [{df_ie[col].min():.2f}, {df_ie[col].max():.2f}]",
            )

    return df_ie


def analyze_dimension(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dep: str,
    results_root: str,
    plots_root: str,
) -> None:
    """Run the full pipeline (LMMs + plots + descriptives) for one dimension.

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data (with normalized columns if needed).
    dep : str
        Dependent variable to analyze.
    results_root : str
        Base directory for LMM results for this analysis set.
    plots_root : str
        Base directory for plots for this analysis set.
    """
    print("\n" + "=" * 60)
    print(f"LINEAR MIXED MODEL ANALYSIS FOR {dep.upper()}")
    print("=" * 60)

    dim_results_dir = os.path.join(results_root, dep)
    dim_plots_dir = os.path.join(plots_root, dep)
    ensure_directories(dim_results_dir, dim_plots_dir)

    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    dep_ie = dep_normalized if has_normalized else dep

    # Model 1: Group effect (BDI-split, raw values)
    group_results_dir = dim_results_dir
    run_lmm_analysis(
        df_lmm,
        dep,
        GROUP_COLUMN,
        "group_effect",
        group_results_dir,
        categorical_cols=[GROUP_COLUMN],
    )

    # Pairwise group comparisons (subject-level means)
    pairwise_group_comparisons(
        df_lmm,
        dep,
        GROUP_COLUMN,
        group_results_dir,
        label="overall",
    )

    # Model 2: Inclusion/Exclusion effect
    run_lmm_analysis(
        df_lmm_ie,
        dep_ie,
        "inclusion_exclusion",
        "inclusion_exclusion_effect",
        dim_results_dir,
        categorical_cols=["inclusion_exclusion"],
    )

    # One-sample tests vs zero for normalized IE effects
    if has_normalized:
        print("\n" + "=" * 60)
        print(f"ONE-SAMPLE TESTS FOR {dep.upper()} (H0: effect = 0)")
        print("=" * 60)

        inclusion_data = (
            df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == "inclusion"][dep_ie].dropna()
        )
        t_incl, p_incl = sp_stats.ttest_1samp(inclusion_data, 0)
        mean_incl = inclusion_data.mean()
        se_incl = inclusion_data.sem()
        n_incl = len(inclusion_data)

        print("\n1. INCLUSION vs Baseline (zero):")
        print(f"   N = {n_incl}")
        print(f"   Mean = {mean_incl:.4f} (SE = {se_incl:.4f})")
        print(f"   t({n_incl-1}) = {t_incl:.4f}, p = {p_incl:.4f}")

        exclusion_data = (
            df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == "exclusion"][dep_ie].dropna()
        )
        t_excl, p_excl = sp_stats.ttest_1samp(exclusion_data, 0)
        mean_excl = exclusion_data.mean()
        se_excl = exclusion_data.sem()
        n_excl = len(exclusion_data)

        print("\n2. EXCLUSION vs Baseline (zero):")
        print(f"   N = {n_excl}")
        print(f"   Mean = {mean_excl:.4f} (SE = {se_excl:.4f})")
        print(f"   t({n_excl-1}) = {t_excl:.4f}, p = {p_excl:.4f}")

        p_bonf_incl = min(p_incl * 2, 1.0)
        p_bonf_excl = min(p_excl * 2, 1.0)
        print("\n3. Bonferroni-corrected p-values (2 tests):")
        print(f"   Inclusion: p_corrected = {p_bonf_incl:.4f}")
        print(f"   Exclusion: p_corrected = {p_bonf_excl:.4f}")

        one_sample_results = pd.DataFrame(
            [
                {
                    "Dimension": dep,
                    "Condition": "Inclusion",
                    "N": n_incl,
                    "Mean": mean_incl,
                    "SE": se_incl,
                    "t_statistic": t_incl,
                    "p_value": p_incl,
                    "p_bonferroni": p_bonf_incl,
                },
                {
                    "Dimension": dep,
                    "Condition": "Exclusion",
                    "N": n_excl,
                    "Mean": mean_excl,
                    "SE": se_excl,
                    "t_statistic": t_excl,
                    "p_value": p_excl,
                    "p_bonferroni": p_bonf_excl,
                },
            ]
        )
        one_sample_file = os.path.join(
            dim_results_dir, f"{dep}_one_sample_tests_vs_baseline.csv"
        )
        one_sample_results.to_csv(one_sample_file, index=False)
        print(f"One-sample test results saved to: {one_sample_file}")

    # Model 3: Group × I/E interaction
    run_lmm_analysis(
        df_lmm_ie,
        dep_ie,
        f"{GROUP_COLUMN} * inclusion_exclusion",
        "group_ie_interaction",
        dim_results_dir,
        categorical_cols=[GROUP_COLUMN, "inclusion_exclusion"],
    )

    # Time-on-task models (if available)
    if "time_on_task" in df_lmm.columns:
        run_lmm_analysis(
            df_lmm,
            dep,
            f"{GROUP_COLUMN} + time_on_task",
            "group_time_additive",
            dim_results_dir,
            categorical_cols=[GROUP_COLUMN],
        )
        run_lmm_analysis(
            df_lmm,
            dep,
            f"{GROUP_COLUMN} * time_on_task",
            "group_time_interaction",
            dim_results_dir,
            categorical_cols=[GROUP_COLUMN],
        )

        # Intervention distance models (probe_number within block)
        run_lmm_analysis(
            df_lmm_ie,
            dep_ie,
            "inclusion_exclusion + probe_number",
            "ie_intervention_distance",
            dim_results_dir,
            categorical_cols=["inclusion_exclusion"],
        )
        run_lmm_analysis(
            df_lmm_ie,
            dep_ie,
            f"{GROUP_COLUMN} * inclusion_exclusion + probe_number",
            "group_ie_plus_distance",
            dim_results_dir,
            categorical_cols=[GROUP_COLUMN, "inclusion_exclusion"],
        )
        run_lmm_analysis(
            df_lmm_ie,
            dep_ie,
            "inclusion_exclusion * probe_number",
            "ie_distance_interaction",
            dim_results_dir,
            categorical_cols=["inclusion_exclusion"],
        )
        run_lmm_analysis(
            df_lmm_ie,
            dep_ie,
            f"{GROUP_COLUMN} * inclusion_exclusion * probe_number",
            "three_way_with_distance",
            dim_results_dir,
            categorical_cols=[GROUP_COLUMN, "inclusion_exclusion"],
        )
        run_lmm_analysis(
            df_lmm_ie,
            dep_ie,
            f"{GROUP_COLUMN} * probe_number + inclusion_exclusion",
            "group_distance_plus_ie",
            dim_results_dir,
            categorical_cols=[GROUP_COLUMN, "inclusion_exclusion"],
        )
    else:
        print("Skipping time-on-task models (time_on_task column not available)")

    print("\n" + "=" * 60)
    print(f"CREATING VISUALIZATIONS FOR {dep.upper()}")
    print("=" * 60)
    plot_dimension(df_lmm, df_lmm_ie, dep, dim_plots_dir)

    print("\n" + "=" * 60)
    print(f"DESCRIPTIVE STATISTICS FOR {dep.upper()}")
    print("=" * 60)
    descriptive_statistics(df_lmm, df_lmm_ie, dep, dim_results_dir)

    print(f"\nCompleted dimension: {dep}")
    print(f"Results saved to: {dim_results_dir}")
    print(f"Plots saved to:   {dim_plots_dir}")


def run_analysis_for_dataset(
    df: pd.DataFrame,
    dimensions: Iterable[str],
    results_dir: str,
    plots_dir: str,
    label: str,
    apply_onoff_filter: bool = False,
    onoff_threshold: float = 50.0,
) -> None:
    """Run the full multi-dimension analysis for a dataset.
    
    Preprocessing order (to preserve participants):
    1. Create time-on-task variables
    2. Apply baseline normalization (BEFORE filtering)
    3. Apply onoff filtering (if requested)
    4. Exclude baseline condition from IE analysis
    5. Clean up rows with missing normalized values

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing all relevant columns.
    dimensions : Iterable[str]
        Dependent variables to analyze.
    results_dir : str
        Root directory for LMM results.
    plots_dir : str
        Root directory for plots.
    label : str
        Text label for logging (e.g. "full_onoff", "onoff_lt50", "pca").
    apply_onoff_filter : bool
        Whether to filter rows where onoff < threshold.
    onoff_threshold : float
        Threshold for onoff filtering.
    """
    print("\n" + "=" * 60)
    print(f"RUNNING ANALYSIS SET: {label}")
    print("=" * 60)

    ensure_directories(results_dir, plots_dir)

    df_lmm = df.copy()
    df_lmm_ie = df_lmm.copy()

    validate_columns(df_lmm, df_lmm_ie, dimensions)

    # =========================================================================
    # STEP 1: Create time_on_task if not already present
    # =========================================================================
    print("\nStep 1: Creating time-on-task variables...")
    if "task" in df_lmm.columns and "probe_number" in df_lmm.columns:
        if "time_on_task" not in df_lmm.columns:
            df_lmm["sart_number"] = df_lmm["task"].str.extract(r"(\d+)").astype(int)
            df_lmm["time_on_task"] = df_lmm["probe_number"] + (
                15 * (df_lmm["sart_number"] - 1)
            )
            df_lmm["relative_time_on_task"] = df_lmm["probe_number"]

        if "time_on_task" not in df_lmm_ie.columns:
            df_lmm_ie["sart_number"] = df_lmm_ie["task"].str.extract(r"(\d+)").astype(int)
            df_lmm_ie["time_on_task"] = df_lmm_ie["probe_number"] + (
                15 * (df_lmm_ie["sart_number"] - 1)
            )
            df_lmm_ie["relative_time_on_task"] = df_lmm_ie["probe_number"]

        print(
            f"  - time_on_task range: {df_lmm['time_on_task'].min()} to {df_lmm['time_on_task'].max()}"
        )
    else:
        print("  - Skipping ('task' or 'probe_number' column not found)")

    # =========================================================================
    # STEP 2: Baseline normalization (BEFORE filtering)
    # =========================================================================
    print("\nStep 2: Applying baseline normalization for IE analysis...")
    print("  Note: Using ALL available data (before filtering) to compute baselines")
    df_lmm_ie = baseline_normalize_ie(df_lmm, df_lmm_ie, dimensions)

    # =========================================================================
    # STEP 3: Apply onoff filtering (AFTER baseline normalization)
    # =========================================================================
    if apply_onoff_filter:
        print(f"\nStep 3: Applying onoff filter (onoff < {onoff_threshold})...")
        
        if "onoff" not in df_lmm.columns or "onoff" not in df_lmm_ie.columns:
            raise ValueError(
                "apply_onoff_filter is True but 'onoff' column is missing in the data."
            )
        
        before_n_lmm = len(df_lmm)
        before_s_lmm = df_lmm["subject_id"].nunique()
        before_n_ie = len(df_lmm_ie)
        before_s_ie = df_lmm_ie["subject_id"].nunique()
        
        df_lmm = df_lmm[df_lmm["onoff"] < onoff_threshold].copy()
        df_lmm_ie = df_lmm_ie[df_lmm_ie["onoff"] < onoff_threshold].copy()
        
        after_n_lmm = len(df_lmm)
        after_s_lmm = df_lmm["subject_id"].nunique()
        after_n_ie = len(df_lmm_ie)
        after_s_ie = df_lmm_ie["subject_id"].nunique()
        
        print(f"  - Probe-level: {before_n_lmm} rows/{before_s_lmm} subjects -> {after_n_lmm} rows/{after_s_lmm} subjects")
        print(f"  - IE-level:    {before_n_ie} rows/{before_s_ie} subjects -> {after_n_ie} rows/{after_s_ie} subjects")
    else:
        print("\nStep 3: Skipping onoff filter (not requested)")

    # =========================================================================
    # STEP 4: Exclude baseline condition from IE analysis (AFTER filtering)
    # =========================================================================
    if EXCLUDE_BASELINE:
        print("\nStep 4: Excluding baseline condition from IE analysis...")
        
        if "inclusion_exclusion" not in df_lmm_ie.columns:
            raise ValueError(
                "EXCLUDE_BASELINE is True but 'inclusion_exclusion' column is missing."
            )
        
        before_n_ie = len(df_lmm_ie)
        before_s_ie = df_lmm_ie["subject_id"].nunique()
        
        df_lmm_ie = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] != "baseline"].copy()
        
        after_n_ie = len(df_lmm_ie)
        after_s_ie = df_lmm_ie["subject_id"].nunique()
        
        print(f"  - IE-level: {before_n_ie} rows/{before_s_ie} subjects -> {after_n_ie} rows/{after_s_ie} subjects")
    else:
        print("\nStep 4: Keeping baseline condition in IE analysis")

    # =========================================================================
    # STEP 5: Clean up rows with missing normalized values
    # =========================================================================
    print("\nStep 5: Cleaning up rows with missing normalized values...")
    
    before_dropna = len(df_lmm_ie)
    before_subjects = df_lmm_ie["subject_id"].nunique()
    
    normalized_cols = [f"{dim}_normalized" for dim in dimensions if f"{dim}_normalized" in df_lmm_ie.columns]
    if normalized_cols:
        df_lmm_ie = df_lmm_ie.dropna(subset=normalized_cols)
    
    after_dropna = len(df_lmm_ie)
    after_subjects = df_lmm_ie["subject_id"].nunique()
    
    if before_dropna > after_dropna:
        print(f"  - Removed {before_dropna - after_dropna} rows with missing baseline data")
        print(f"  - Subjects: {before_subjects} -> {after_subjects}")
    else:
        print("  - No rows removed (all baselines available)")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final probe-level dataset: {len(df_lmm)} rows, {df_lmm['subject_id'].nunique()} subjects")
    print(f"Final IE-level dataset:    {len(df_lmm_ie)} rows, {df_lmm_ie['subject_id'].nunique()} subjects")

    # Run per-dimension pipeline
    for dep in dimensions:
        analyze_dimension(df_lmm, df_lmm_ie, dep, results_dir, plots_dir)

    print("\n" + "=" * 60)
    print(f"ANALYSIS SET COMPLETE: {label}")
    print("=" * 60)
    print(f"Results base dir: {results_dir}")
    print(f"Plots base dir:   {plots_dir}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """Entry point to run BDI-split multidimensional analyses.

    This function runs three blocks:

    1. Probe-level analyses on the full onoff range (if enabled).
    2. Probe-level analyses restricted to ``onoff < ONOFF_MAX_EXCLUSIVE``
       (if enabled).
    3. PCA-based analyses (PC1–PC3) using ``pca_results.csv`` (always on the
       subset provided by that file).
    """
    ensure_directories(BASE_RESULTS_DIR, BASE_PLOTS_DIR)

    # ------------------------------------------------------------------
    # Load data and construct BDI-split group mapping
    # ------------------------------------------------------------------
    df_probe = load_probe_data(DATA_FILE)
    bdi_mapping = build_bdi_group_mapping(df_probe)

    # Probe-level data with BDI-split group
    df_probe_bdi = add_bdi_group_column(df_probe, bdi_mapping)

    # ------------------------------------------------------------------
    # 1) Full onoff range analyses
    # ------------------------------------------------------------------
    if RUN_FULL_ONOFF_RANGE:
        results_dir_full = os.path.join(BASE_RESULTS_DIR, "onoff_full_range")
        plots_dir_full = os.path.join(BASE_PLOTS_DIR, "onoff_full_range")
        run_analysis_for_dataset(
            df_probe_bdi,
            BASE_DIMENSIONS,
            results_dir_full,
            plots_dir_full,
            label="probe_full_onoff",
        )

    # ------------------------------------------------------------------
    # 2) onoff < ONOFF_MAX_EXCLUSIVE analyses
    # ------------------------------------------------------------------
    if RUN_ONOFF_LT50:
        if "onoff" not in df_probe_bdi.columns:
            raise ValueError(
                "RUN_ONOFF_LT50 is True but 'onoff' column is missing in the data."
            )
        # Note: filtering is now done INSIDE run_analysis_for_dataset AFTER baseline normalization
        # This preserves participants by computing baselines from full data first
        results_dir_lt50 = os.path.join(BASE_RESULTS_DIR, "onoff_lt50")
        plots_dir_lt50 = os.path.join(BASE_PLOTS_DIR, "onoff_lt50")
        run_analysis_for_dataset(
            df_probe_bdi,  # Pass full data, filtering happens after baseline normalization
            BASE_DIMENSIONS,
            results_dir_lt50,
            plots_dir_lt50,
            label="probe_onoff_lt50",
            apply_onoff_filter=True,
            onoff_threshold=ONOFF_MAX_EXCLUSIVE,
        )

    # ------------------------------------------------------------------
    # 3) PCA-based analyses (PC1–PC3) with BDI-split groups
    # ------------------------------------------------------------------
    df_pca = load_pca_results(PCA_RESULTS_FILE)
    df_pca_bdi = add_bdi_group_column(df_pca, bdi_mapping)

    # Some PCA results files already contain time_on_task; keep as is.
    results_dir_pca = os.path.join(BASE_RESULTS_DIR, "pca_components")
    plots_dir_pca = os.path.join(BASE_PLOTS_DIR, "pca_components")
    run_analysis_for_dataset(
        df_pca_bdi,
        PCA_DIMENSIONS,
        results_dir_pca,
        plots_dir_pca,
        label="pca_components",
    )

    print("\n" + "=" * 60)
    print("ALL BDI-SPLIT ANALYSES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

# %%
