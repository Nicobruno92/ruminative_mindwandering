# %%
"""
Multi-dimension Probe Analysis (LMM + Plots)

This script repeats the On/Off linear mixed-model analysis across multiple
dimensions: valence, time, selfother, and confidence. It fits three models per
dimension (group effect, inclusion/exclusion effect, and their interaction),
exports model results, and generates plots mirroring the On/Off analysis.

Design notes
------------
- Configuration lives at the top of the script. Adjust paths and settings there.
- The script expects a single tidy aggregated CSV with at least these columns:
  [    # Plot 6 (1,2): SART Mean Trajectories by Group and Order (4 points per line)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'order (IE/EI)' in df_lmm.columns:
        # Calculate mean per SART for each group and order combination
        sart_order = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        x_positions = [1, 2, 3, 4]  # X-axis positions for each SART
        
        # Define line styles for orders
        order_styles = {'IE': '-', 'EI': '--'}  # IE = Inclusion-Exclusion, EI = Exclusion-Inclusion
        
        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in df_lmm["group"].values:
                continue
            group_data = df_lmm[df_lmm["group"] == group]
            color = GROUP_COLORS[group_idx]
            
            for order in ['IE', 'EI']:
                if order not in group_data['order (IE/EI)'].values:
                    continue
                order_data = group_data[group_data['order (IE/EI)'] == order]', 'inclusion_exclusion', <dimension>]. We reuse this
  same table for both the probe-level and inclusion/exclusion analyses.
- For each dimension in DIMENSIONS, we run the same set of models and plots.

Usage
-----
Simply run this script. Modify CONFIG variables as needed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
import statsmodels.formula.api as smf
from scipy import stats


# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# Input tidy aggregated data file. Provide CSV path that contains the required columns.
# If your data are in other formats (e.g., parquet/pickle), change the loader accordingly.
DATA_FILE: str = "../../results/Behavior/probe_data/probe_level_aggregated_data.csv"

# Dimensions to analyze as dependent variables. Columns must exist in the data.
DIMENSIONS: List[str] = ["valence", "time", "selfother", "confidence"]

# Output base directories for results and plots (will be created if missing)
RESULTS_DIR: str = "../../results/Behavior/probe_data/lmm_analysis_multidim"
PLOTS_DIR: str = "../../results/Behavior/probe_data/lmm_analysis_multidim"

# Plot aesthetics
GROUP_ORDER: List[str] = ["Controls", "Risk of Depression"]
IE_ORDER: List[str] = ["inclusion", "exclusion"]
GROUP_COLORS: List[str] = ["#2E86AB", "#F24236"]
IE_COLORS: List[str] = ["#A23B72", "#F18F01"]

# Optional filter: analyze only rows where onoff < threshold
APPLY_ONOFF_FILTER: bool = True
ONOFF_MAX_EXCLUSIVE: float = 62

# Optional filter: exclude baseline condition from inclusion/exclusion analysis
EXCLUDE_BASELINE: bool = False  # Set to True to only analyze inclusion vs exclusion

#%%
# =============================================================================


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load input dataframes for LMM and Inclusion/Exclusion analyses.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        df_lmm, df_lmm_ie
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"DATA_FILE not found: {DATA_FILE}. Update the path in the CONFIG section."
        )

    df = pd.read_csv(DATA_FILE)
    # We reuse the same aggregated table for both uses
    return df, df


def validate_columns(df_lmm: pd.DataFrame, df_lmm_ie: pd.DataFrame, dimensions: List[str]) -> None:
    """
    Validate that required columns exist in the input data frames.

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data.
    dimensions : List[str]
        Dependent variable names to verify.
    """
    lmm_required = {"subject_id", "group"}
    ie_required = {"subject_id", "group", "inclusion_exclusion"}

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
) -> Tuple[pd.DataFrame, object]:
    """
    Run linear mixed model analysis with random intercepts by subject.

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
    Tuple[pd.DataFrame, object]
        results_df, fitted_model
    """
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
    # Cast factors to category if present
    for cat_col in ["group", "inclusion_exclusion"]:
        if cat_col in model_df.columns:
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
        # Column-wise NA counts
        na_counts = dropped_df.isna().sum()
        # Subject-wise counts
        subj_counts = dropped_df["subject_id"].value_counts().sort_index()
        # Basic distributions for key factors if present
        group_counts = (
            dropped_df["group"].value_counts().sort_index() if "group" in dropped_df.columns else None
        )
        ie_counts = (
            dropped_df["inclusion_exclusion"].value_counts().sort_index()
            if "inclusion_exclusion" in dropped_df.columns
            else None
        )

        print(
            f"Dropped {num_dropped} rows with missing data for model '{model_name}' ({dependent_var})."
        )
        print("- Missing by column (only >0 shown):")
        for col, cnt in na_counts.items():
            if cnt > 0:
                print(f"  {col}: {cnt}")
        print("- Dropped rows by subject_id:")
        for sid, cnt in subj_counts.items():
            print(f"  {sid}: {cnt}")
        if group_counts is not None:
            print("- Dropped rows by group:")
            for g, cnt in group_counts.items():
                print(f"  {g}: {cnt}")
        if ie_counts is not None:
            print("- Dropped rows by inclusion_exclusion:")
            for cond, cnt in ie_counts.items():
                print(f"  {cond}: {cnt}")

        # Save the dropped rows for audit
        os.makedirs(output_dir, exist_ok=True)
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
    # Calculate n_groups from the data since model doesn't have this attribute
    n_groups = model_df['subject_id'].nunique()
    
    # Try to get AIC/BIC from model, calculate manually if not available
    aic_value = model.aic if hasattr(model, 'aic') and not pd.isna(model.aic) else None
    bic_value = model.bic if hasattr(model, 'bic') and not pd.isna(model.bic) else None
    
    # Manual calculation if not available from model
    if aic_value is None:
        # AIC = -2*log-likelihood + 2*k (where k is number of parameters)
        k = len(model.params)
        aic_value = -2 * model.llf + 2 * k
        print(f"Note: AIC calculated manually: {aic_value:.3f}")
    
    if bic_value is None:
        # BIC = -2*log-likelihood + k*ln(n)
        k = len(model.params)
        n = model.nobs
        bic_value = -2 * model.llf + k * np.log(n)
        print(f"Note: BIC calculated manually: {bic_value:.3f}")
    
    model_metrics = {
        'aic': aic_value,
        'bic': bic_value,
        'log_likelihood': model.llf,
        'log_likelihood_restricted': model.llf_fe if hasattr(model, 'llf_fe') else None,
        'n_observations': model.nobs,
        'n_groups': n_groups,
        'n_parameters': len(model.params),
        'n_fixed_effects': len(model.fe_params) if hasattr(model, 'fe_params') else None,
        'n_random_effects': len(model.cov_re) if hasattr(model, 'cov_re') else None,
        'converged': model.converged,
        'scale': model.scale if hasattr(model, 'scale') else None,
        'deviance': -2 * model.llf if model.llf is not None else None,
        'rsquared_within': None,
        'rsquared_between': None
    }
    
    # Print model fit metrics
    print("\n=== MODEL FIT METRICS ===")
    print(f"AIC: {model_metrics['aic']:.3f}")
    print(f"BIC: {model_metrics['bic']:.3f}")
    print(f"Log-Likelihood: {model_metrics['log_likelihood']:.3f}")
    print(f"N observations: {model_metrics['n_observations']}")
    print(f"N groups: {model_metrics['n_groups']}")
    print(f"Converged: {model_metrics['converged']}")
    if model_metrics['scale'] is not None:
        print(f"Scale: {model_metrics['scale']:.6f}")

    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Save model metrics
    metrics_df = pd.DataFrame([model_metrics])
    metrics_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save enhanced model summary with metrics
    summary_file = os.path.join(output_dir, f"{model_name}_{dependent_var}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(str(model.summary()))
        f.write("\n\n" + "="*60 + "\n")
        f.write("MODEL FIT METRICS\n")
        f.write("="*60 + "\n")
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


def plot_dimension(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dep: str,
    out_dir: str,
) -> None:
    """
    Create a comprehensive 2x3 figure combining all key analyses.
    
    Grid layout:
    - Row 1: Raincloud (Group) | Raincloud (I/E) | Interaction
    - Row 2: Time by Group | Time by I/E | SART Trajectories

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
    
    # Determine if we have normalized version for IE plots
    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    dep_ie = dep_normalized if has_normalized else dep
    ylabel_suffix = " (Normalized)" if has_normalized else ""

    plt.style.use("default")
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # =========================================================================
    # ROW 1: DISTRIBUTION COMPARISONS
    # =========================================================================

    # Plot 1 (0,0): Raincloud for group comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_agg_group = df_lmm.groupby(["subject_id", "group"])[dep].mean().reset_index()
    n_participants_by_group = df_agg_group.groupby("group")["subject_id"].nunique().to_dict()
    
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
    ax1.set_title(
        f"{dep.upper()}: Group Effect",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"{dep.title()} Score", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Add sample sizes
    for i, group in enumerate(GROUP_ORDER):
        n = n_participants_by_group.get(group, 0)
        ax1.text(
            i, ax1.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 2 (0,1): Raincloud for inclusion/exclusion (NORMALIZED)
    ax2 = fig.add_subplot(gs[0, 1])
    df_agg_ie = df_lmm_ie.groupby(["subject_id", "inclusion_exclusion"])[dep_ie].mean().reset_index()
    n_participants_by_ie = df_agg_ie.groupby("inclusion_exclusion")["subject_id"].nunique().to_dict()
    
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
    ax2.set_ylabel(f"{dep.title()} Score{ylabel_suffix}", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Add baseline reference line for normalized plots
    if has_normalized:
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax2.legend(fontsize=10, loc='best')
    
    # Add sample sizes
    for i, condition in enumerate(IE_ORDER):
        n = n_participants_by_ie.get(condition, 0)
        ax2.text(
            i, ax2.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 3 (0,2): Interaction plot (Group × I/E) (NORMALIZED)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Store data for one-sample t-tests against 0 (baseline comparison)
    baseline_tests = []
    
    for group in df_lmm_ie["group"].dropna().unique():
        group_data = df_lmm_ie[df_lmm_ie["group"] == group]
        if len(group_data) == 0:
            continue
        ie_means = (
            group_data.groupby("inclusion_exclusion")[dep_ie]
            .agg(["mean", "sem"])
            .reindex(IE_ORDER)
            .reset_index()
        )
        n_participants_group = group_data["subject_id"].nunique()
        color = GROUP_COLORS[0] if group == GROUP_ORDER[0] else GROUP_COLORS[1]
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
        
        # One-sample t-tests against 0 (baseline) for each condition
        if has_normalized:
            for i, condition in enumerate(IE_ORDER):
                cond_data = group_data[group_data["inclusion_exclusion"] == condition]
                # Aggregate per subject first
                subj_means = cond_data.groupby("subject_id")[dep_ie].mean()
                if len(subj_means) > 1:
                    t_stat, p_val = stats.ttest_1samp(subj_means, 0)
                    baseline_tests.append({
                        'group': group,
                        'condition': condition,
                        't_stat': t_stat,
                        'p_val': p_val,
                        'mean': subj_means.mean(),
                        'n': len(subj_means),
                        'x_pos': x_positions[i],
                        'color': color
                    })
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(IE_ORDER, fontsize=14, fontweight="bold")
    ax3.set_ylabel(f"{dep.title()} Score{ylabel_suffix}", fontsize=14, fontweight="bold")
    ax3.set_title(
        f"{dep.upper()}: Group × I/E Interaction{title_suffix}",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    # Add baseline reference line for normalized plots
    if has_normalized:
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        
        # Add significance markers for tests against baseline (0)
        y_min, y_max = ax3.get_ylim()
        y_range = y_max - y_min
        
        for test in baseline_tests:
            # Determine significance level
            if test['p_val'] < 0.001:
                sig_marker = '***'
            elif test['p_val'] < 0.01:
                sig_marker = '**'
            elif test['p_val'] < 0.05:
                sig_marker = '*'
            else:
                sig_marker = ''
            
            if sig_marker:
                # Position marker above or below the point based on mean value
                y_offset = 0.08 * y_range if test['mean'] > 0 else -0.08 * y_range
                x_offset = -0.15 if test['group'] == GROUP_ORDER[0] else 0.15
                ax3.text(
                    test['x_pos'] + x_offset, 
                    test['mean'] + y_offset,
                    sig_marker,
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color=test['color']
                )
        
        # Print baseline comparison results
        print(f"\n  {dep.upper()} - One-sample t-tests against baseline (0):")
        for test in baseline_tests:
            sig = '*' if test['p_val'] < 0.05 else ''
            print(f"    {test['group']} - {test['condition']}: t({test['n']-1})={test['t_stat']:.3f}, p={test['p_val']:.4f} {sig}")
    
    ax3.legend(fontsize=12, title_fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # ROW 2: TIME-ON-TASK TRAJECTORIES
    # =========================================================================

    # Plot 4 (1,0): Time-on-task by group
    ax4 = fig.add_subplot(gs[1, 0])
    if 'time_on_task' in df_lmm.columns:
        for i, group in enumerate(GROUP_ORDER):
            if group in df_lmm["group"].values:
                group_data = df_lmm[df_lmm["group"] == group]
                time_group_agg = group_data.groupby("time_on_task")[dep].agg(["mean", "sem"]).reset_index()
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
        ax4.legend(fontsize=12, loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Time-on-task data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)

    # Plot 5 (1,1): Intervention distance by inclusion/exclusion (probe 1-15 within block) (NORMALIZED)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'probe_number' in df_lmm_ie.columns:
        for i, condition in enumerate(IE_ORDER):
            if condition in df_lmm_ie["inclusion_exclusion"].values:
                ie_data = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == condition]
                # Use probe_number (1-15 within each manipulation block)
                probe_ie_agg = ie_data.groupby("probe_number")[dep_ie].agg(["mean", "sem"]).reset_index()
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
        ax5.set_xlabel("Intervention Distance (Probe Number)", fontsize=14, fontweight="bold")
        ax5.set_ylabel(f"{dep.title()} Score{ylabel_suffix}", fontsize=14, fontweight="bold")
        ax5.set_title(
            f"{dep.upper()}: Intervention Distance by I/E{title_suffix}",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        # Add baseline reference line for normalized plots
        if has_normalized:
            ax5.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
        ax5.legend(fontsize=12, loc='best')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0.5, 15.5)  # Set x-axis limits for 1-15 probes
    else:
        ax5.text(0.5, 0.5, 'Probe number data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)

    # Plot 6 (1,2): SART Mean Trajectories by Group and Order (4 points per line)
    ax6 = fig.add_subplot(gs[1, 2])
    if 'task' in df_lmm.columns and 'order (IE/EI)' in df_lmm.columns:
        # Calculate mean per SART for each group and order combination
        sart_order = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        x_positions = [1, 2, 3, 4]  # X-axis positions for each SART
        
        # Define line styles for orders
        order_styles = {'IE': '-', 'EI': '--'}  # IE = Inclusion-Exclusion, EI = Exclusion-Inclusion
        
        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in df_lmm["group"].values:
                continue
            group_data = df_lmm[df_lmm["group"] == group]
            color = GROUP_COLORS[group_idx]
            
            for order in ['IE', 'EI']:
                if order not in group_data['order (IE/EI)'].values:
                    continue
                order_data = group_data[group_data['order (IE/EI)'] == order]
                
                # Calculate n for this group-order combination
                n_subjects = order_data['subject_id'].nunique()
                
                means = []
                sems = []
                for sart in sart_order:
                    sart_data = order_data[order_data['task'] == sart]
                    if len(sart_data) > 0:
                        means.append(sart_data[dep].mean())
                        sems.append(sart_data[dep].sem())
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)
                
                # Plot line with error bars
                linestyle = order_styles[order]
                label = f"{group} - {order} (n={n_subjects})"
                
                line = ax6.errorbar(
                    x_positions,
                    means,
                    yerr=sems,
                    marker='o',
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
        ax6.set_xticklabels(sart_order, fontsize=12, fontweight='bold')
        ax6.set_xlabel("SART Task", fontsize=14, fontweight="bold")
        ax6.set_ylabel(f"{dep.title()} Score", fontsize=14, fontweight="bold")
        ax6.set_title(
            f"{dep.upper()}: SART Trajectory by Group & Order\n(Solid=IE, Dashed=EI)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        # Create legend with only lines (no markers)
        handles, labels = ax6.get_legend_handles_labels()
        # Remove markers from legend handles - errorbar returns containers, need to access the line
        new_handles = []
        for handle in handles:
            if hasattr(handle, 'get_children'):
                # ErrorbarContainer - get the line component
                lines = [child for child in handle.get_children() if hasattr(child, 'set_marker')]
                if lines:
                    line = lines[0]
                    line.set_marker('')
                    line.set_markersize(0)
                    new_handles.append(line)
            else:
                # Regular line object
                handle.set_marker('')
                handle.set_markersize(0)
                new_handles.append(handle)
        ax6.legend(new_handles, labels, fontsize=11, loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0.5, 4.5)
    else:
        ax6.text(0.5, 0.5, 'Order data not available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)

    # =========================================================================
    # SAVE AND DISPLAY
    # =========================================================================

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
    # plt.show()  # Removed to prevent segmentation fault
    plt.close(fig)


def descriptive_statistics(df_lmm: pd.DataFrame, df_lmm_ie: pd.DataFrame, dep: str, out_dir: str) -> None:
    """
    Compute and save descriptive statistics mirroring the On/Off block.

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
    
    # Determine if we have normalized version
    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    dep_ie = dep_normalized if has_normalized else dep

    overall_stats = df_lmm[dep].describe()
    print(f"Overall: Mean = {overall_stats['mean']:.3f}, SD = {overall_stats['std']:.3f}")

    print("\nBy Group:")
    group_stats = df_lmm.groupby("group")[dep].agg(["count", "mean", "std"]).round(3)
    print(group_stats)

    print("\nBy Inclusion/Exclusion (Raw):")
    ie_stats = (
        df_lmm_ie.groupby("inclusion_exclusion")[dep].agg(["count", "mean", "std"]).round(3)
    )
    print(ie_stats)
    
    if has_normalized:
        print("\nBy Inclusion/Exclusion (Baseline-Corrected):")
        ie_stats_norm = (
            df_lmm_ie.groupby("inclusion_exclusion")[dep_ie].agg(["count", "mean", "std"]).round(3)
        )
        print(ie_stats_norm)

    print("\nBy Group × Inclusion/Exclusion (Raw):")
    interaction_stats = (
        df_lmm_ie.groupby(["group", "inclusion_exclusion"])[dep]
        .agg(["count", "mean", "std"])
        .round(3)
    )
    print(interaction_stats)
    
    if has_normalized:
        print("\nBy Group × Inclusion/Exclusion (Baseline-Corrected):")
        interaction_stats_norm = (
            df_lmm_ie.groupby(["group", "inclusion_exclusion"])[dep_ie]
            .agg(["count", "mean", "std"])
            .round(3)
        )
        print(interaction_stats_norm)

    # Save cell-wise stats (both raw and normalized if available)
    stats_list: List[Dict[str, float]] = []
    for group in df_lmm_ie["group"].dropna().unique():
        for ie in df_lmm_ie["inclusion_exclusion"].dropna().unique():
            subset = df_lmm_ie[
                (df_lmm_ie["group"] == group) & (df_lmm_ie["inclusion_exclusion"] == ie)
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
                    stats_dict.update({
                        "Mean_Normalized": float(subset[dep_ie].mean()),
                        "SD_Normalized": float(subset[dep_ie].std()),
                        "SE_Normalized": float(subset[dep_ie].sem()),
                    })
                stats_list.append(stats_dict)
    stats_df = pd.DataFrame(stats_list)
    stats_file = os.path.join(out_dir, f"{dep}_descriptive_statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    suffix_msg = " (raw and normalized)" if has_normalized else ""
    print(f"Descriptive statistics{suffix_msg} saved to: {stats_file}")


def analyze_dimension(df_lmm: pd.DataFrame, df_lmm_ie: pd.DataFrame, dep: str) -> None:
    """
    Run the full pipeline (3 LMMs + plots + descriptives) for one dimension.

    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data.
    df_lmm_ie : pd.DataFrame
        Inclusion/Exclusion-level data.
    dep : str
        Dependent variable to analyze.
    """
    print("\n" + "=" * 60)
    print(f"LINEAR MIXED MODEL ANALYSIS FOR {dep.upper()}")
    print("=" * 60)

    dim_results_dir = os.path.join(RESULTS_DIR, dep)
    dim_plots_dir = os.path.join(PLOTS_DIR, dep)
    os.makedirs(dim_results_dir, exist_ok=True)
    os.makedirs(dim_plots_dir, exist_ok=True)

    # Determine if we have normalized version
    dep_normalized = f"{dep}_normalized"
    has_normalized = dep_normalized in df_lmm_ie.columns
    
    # Use normalized version for IE analyses if available
    dep_ie = dep_normalized if has_normalized else dep
    
    # Model 1: Group effect (uses raw values)
    run_lmm_analysis(df_lmm, dep, "group", "group_effect", dim_results_dir)

    # Model 2: Inclusion/Exclusion effect (NORMALIZED)
    run_lmm_analysis(
        df_lmm_ie, dep_ie, "inclusion_exclusion", "inclusion_exclusion_effect", dim_results_dir
    )
    
    # =========================================================================
    # ONE-SAMPLE TESTS: Are inclusion/exclusion effects different from zero?
    # =========================================================================
    if has_normalized:
        print("\n" + "=" * 60)
        print(f"ONE-SAMPLE TESTS FOR {dep.upper()} (H0: effect = 0)")
        print("=" * 60)
        
        from scipy import stats as sp_stats
        
        # Test 1: Is INCLUSION effect different from zero?
        inclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'inclusion'][dep_ie].dropna()
        t_incl, p_incl = sp_stats.ttest_1samp(inclusion_data, 0)
        mean_incl = inclusion_data.mean()
        se_incl = inclusion_data.sem()
        n_incl = len(inclusion_data)
        
        print(f"\n1. INCLUSION vs Baseline (zero):")
        print(f"   N = {n_incl}")
        print(f"   Mean = {mean_incl:.4f} (SE = {se_incl:.4f})")
        print(f"   t({n_incl-1}) = {t_incl:.4f}, p = {p_incl:.4f}")
        if p_incl < 0.05:
            print(f"   *** SIGNIFICANT: Inclusion {'increases' if mean_incl > 0 else 'decreases'} relative to baseline")
        else:
            print(f"   Not significant")
        
        # Test 2: Is EXCLUSION effect different from zero?
        exclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'exclusion'][dep_ie].dropna()
        t_excl, p_excl = sp_stats.ttest_1samp(exclusion_data, 0)
        mean_excl = exclusion_data.mean()
        se_excl = exclusion_data.sem()
        n_excl = len(exclusion_data)
        
        print(f"\n2. EXCLUSION vs Baseline (zero):")
        print(f"   N = {n_excl}")
        print(f"   Mean = {mean_excl:.4f} (SE = {se_excl:.4f})")
        print(f"   t({n_excl-1}) = {t_excl:.4f}, p = {p_excl:.4f}")
        if p_excl < 0.05:
            print(f"   *** SIGNIFICANT: Exclusion {'increases' if mean_excl > 0 else 'decreases'} relative to baseline")
        else:
            print(f"   Not significant")
        
        # Bonferroni correction
        p_bonf_incl = min(p_incl * 2, 1.0)
        p_bonf_excl = min(p_excl * 2, 1.0)
        
        print(f"\n3. Bonferroni-corrected p-values (2 tests):")
        print(f"   Inclusion: p_corrected = {p_bonf_incl:.4f} {'***' if p_bonf_incl < 0.05 else ''}")
        print(f"   Exclusion: p_corrected = {p_bonf_excl:.4f} {'***' if p_bonf_excl < 0.05 else ''}")
        
        # Save results
        one_sample_results = pd.DataFrame([
            {
                'Dimension': dep,
                'Condition': 'Inclusion',
                'N': n_incl,
                'Mean': mean_incl,
                'SE': se_incl,
                't_statistic': t_incl,
                'p_value': p_incl,
                'p_bonferroni': p_bonf_incl,
                'Significant_uncorrected': p_incl < 0.05,
                'Significant_bonferroni': p_bonf_incl < 0.05
            },
            {
                'Dimension': dep,
                'Condition': 'Exclusion',
                'N': n_excl,
                'Mean': mean_excl,
                'SE': se_excl,
                't_statistic': t_excl,
                'p_value': p_excl,
                'p_bonferroni': p_bonf_excl,
                'Significant_uncorrected': p_excl < 0.05,
                'Significant_bonferroni': p_bonf_excl < 0.05
            }
        ])
        
        one_sample_file = os.path.join(dim_results_dir, f"{dep}_one_sample_tests_vs_baseline.csv")
        one_sample_results.to_csv(one_sample_file, index=False)
        print(f"\nOne-sample test results saved to: {one_sample_file}")

    # Model 3: Interaction effect (NORMALIZED)
    run_lmm_analysis(
        df_lmm_ie, dep_ie, "group * inclusion_exclusion", "group_ie_interaction", dim_results_dir
    )
    
    # Model 4: Time on task effect (if available)
    if 'time_on_task' in df_lmm.columns:
        # Model 5: Group + Time on task
        run_lmm_analysis(
            df_lmm, dep, "group + time_on_task", "group_time_additive", dim_results_dir
        )
        
        # Model 6: Group × Time on task interaction
        run_lmm_analysis(
            df_lmm, dep, "group * time_on_task", "group_time_interaction", dim_results_dir
        )
        
        # =====================================================================
        # INTERVENTION DISTANCE MODELS (probe_number within block: 1-15)
        # These test how the effect evolves within each manipulation block
        # =====================================================================
        
        # Model 7: Inclusion/Exclusion + Intervention distance (NORMALIZED)
        run_lmm_analysis(
            df_lmm_ie, dep_ie, "inclusion_exclusion + probe_number", "ie_intervention_distance", dim_results_dir
        )
        
        # Model 8: Group * Inclusion/Exclusion + Intervention distance (NORMALIZED)
        run_lmm_analysis(
            df_lmm_ie, dep_ie, "group * inclusion_exclusion + probe_number", "group_ie_plus_distance", dim_results_dir
        )
        
        # Model 9: Inclusion/Exclusion × Intervention distance interaction (NORMALIZED)
        run_lmm_analysis(
            df_lmm_ie, dep_ie, "inclusion_exclusion * probe_number", "ie_distance_interaction", dim_results_dir
        )
        
        # Model 10: Group × Inclusion/Exclusion × Intervention distance (three-way) (NORMALIZED)
        run_lmm_analysis(
            df_lmm_ie, dep_ie, "group * inclusion_exclusion * probe_number", "three_way_with_distance", dim_results_dir
        )
        
        # Model 11: Group × Intervention distance + Inclusion/Exclusion (NORMALIZED)
        run_lmm_analysis(
            df_lmm_ie, dep_ie, "group * probe_number + inclusion_exclusion", "group_distance_plus_ie", dim_results_dir
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


def preprocess_data(
    df_lmm: pd.DataFrame,
    df_lmm_ie: pd.DataFrame,
    dimensions: List[str],
    apply_onoff_filter: bool = False,
    onoff_threshold: float = 50.0,
    exclude_baseline: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data with the following steps (in order):
    1. Create time-on-task variables
    2. Apply onoff filtering FIRST (if requested)
    3. Compute combined baseline (SART1 + SART3) from FILTERED off-task data
    4. Apply baseline normalization using the combined baseline
    5. Exclude baseline condition from IE analysis (if requested)
    
    This order ensures that:
    - Baseline is computed only from off-task thoughts (onoff < threshold)
    - Combining SART1 + SART3 maximizes chances of having baseline data per subject
    - The normalized zero represents the subject's off-task baseline level
    
    Parameters
    ----------
    df_lmm : pd.DataFrame
        Probe-level data for global analysis.
    df_lmm_ie : pd.DataFrame
        Data for inclusion/exclusion analysis.
    dimensions : List[str]
        List of dependent variables to normalize.
    apply_onoff_filter : bool
        Whether to filter rows where onoff < threshold.
    onoff_threshold : float
        Threshold for onoff filtering.
    exclude_baseline : bool
        Whether to exclude baseline condition from IE analysis.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Preprocessed df_lmm, df_lmm_ie
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # =========================================================================
    # STEP 1: Create time-on-task variables
    # =========================================================================
    if 'task' in df_lmm.columns and 'probe_number' in df_lmm.columns:
        print("\nStep 1: Creating time-on-task variables...")
        df_lmm['sart_number'] = df_lmm['task'].str.extract(r'(\d+)').astype(int)
        df_lmm['time_on_task'] = df_lmm['probe_number'] + (15 * (df_lmm['sart_number'] - 1))
        df_lmm['relative_time_on_task'] = df_lmm['probe_number']
        
        df_lmm_ie['sart_number'] = df_lmm_ie['task'].str.extract(r'(\d+)').astype(int)
        df_lmm_ie['time_on_task'] = df_lmm_ie['probe_number'] + (15 * (df_lmm_ie['sart_number'] - 1))
        df_lmm_ie['relative_time_on_task'] = df_lmm_ie['probe_number']
        
        print(f"  - time_on_task range: {df_lmm['time_on_task'].min()} to {df_lmm['time_on_task'].max()}")
    else:
        print("\nStep 1: Skipping time-on-task creation ('task' column not found)")
    
    # =========================================================================
    # STEP 2: Apply onoff filtering FIRST
    # =========================================================================
    if apply_onoff_filter:
        print(f"\nStep 2: Applying onoff filter FIRST (onoff < {onoff_threshold})...")
        
        if "onoff" not in df_lmm.columns or "onoff" not in df_lmm_ie.columns:
            raise ValueError(
                "apply_onoff_filter is True but 'onoff' column is missing in the data."
            )
        
        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["onoff"] < onoff_threshold].copy()
        
        before_n_lmm = len(df_lmm)
        before_s_lmm = df_lmm["subject_id"].nunique()
        before_n_ie = len(df_lmm_ie)
        before_s_ie = df_lmm_ie["subject_id"].nunique()
        
        df_lmm = _filter(df_lmm)
        df_lmm_ie = _filter(df_lmm_ie)
        
        after_n_lmm = len(df_lmm)
        after_s_lmm = df_lmm["subject_id"].nunique()
        after_n_ie = len(df_lmm_ie)
        after_s_ie = df_lmm_ie["subject_id"].nunique()
        
        print(f"  - Probe-level: {before_n_lmm} rows/{before_s_lmm} subjects -> {after_n_lmm} rows/{after_s_lmm} subjects")
        print(f"  - IE-level:    {before_n_ie} rows/{before_s_ie} subjects -> {after_n_ie} rows/{after_s_ie} subjects")
    else:
        print("\nStep 2: Skipping onoff filter (not requested)")
    
    # =========================================================================
    # STEP 3: Compute COMBINED baseline (SART1 + SART3) from FILTERED data
    # =========================================================================
    print("\nStep 3: Computing COMBINED baseline (SART1 + SART3) from filtered off-task data...")
    print("  Note: Combining both baseline blocks to maximize data availability")
    
    # Get baseline data (SART1 and SART3) from the FILTERED dataset
    baseline_data = df_lmm[df_lmm['task'].isin(['Sart1', 'Sart3'])].copy()
    
    # Calculate combined baseline mean per subject for each dimension
    baseline_means_combined = {}
    subjects_with_baseline = set()
    subjects_missing_baseline = set()
    
    for subject in df_lmm['subject_id'].unique():
        subject_baseline = baseline_data[baseline_data['subject_id'] == subject]
        
        if len(subject_baseline) > 0:
            subjects_with_baseline.add(subject)
            for dim in dimensions:
                if dim in subject_baseline.columns:
                    baseline_means_combined[(subject, dim)] = subject_baseline[dim].mean()
        else:
            subjects_missing_baseline.add(subject)
    
    print(f"  - Subjects with combined baseline data: {len(subjects_with_baseline)}")
    print(f"  - Subjects missing baseline data: {len(subjects_missing_baseline)}")
    if subjects_missing_baseline:
        print(f"    Missing subjects: {sorted(subjects_missing_baseline)}")
    
    # Diagnostic: trials per subject in baseline
    if len(subjects_with_baseline) > 0:
        baseline_counts = baseline_data.groupby('subject_id').size()
        print(f"  - Baseline trials per subject: min={baseline_counts.min()}, "
              f"median={baseline_counts.median():.0f}, max={baseline_counts.max()}")
    
    # =========================================================================
    # STEP 4: Apply baseline normalization using COMBINED baseline
    # =========================================================================
    print("\nStep 4: Applying baseline normalization (combined SART1+SART3)...")
    
    def normalize_by_combined_baseline(row, dimension):
        """Normalize by subtracting the combined SART1+SART3 baseline mean."""
        subject = row['subject_id']
        task = row['task']
        
        # Only normalize SART2 and SART4 (post-manipulation blocks)
        if task not in ['Sart2', 'Sart4']:
            return np.nan
        
        baseline_key = (subject, dimension)
        baseline_mean = baseline_means_combined.get(baseline_key, np.nan)
        
        if pd.isna(baseline_mean):
            return np.nan
        
        return row[dimension] - baseline_mean
    
    # Create normalized columns for each dimension
    for dim in dimensions:
        if dim in df_lmm_ie.columns:
            df_lmm_ie[f'{dim}_normalized'] = df_lmm_ie.apply(
                lambda row, d=dim: normalize_by_combined_baseline(row, d), axis=1
            )
    
    # Report normalization statistics
    normalized_cols = [f'{dim}_normalized' for dim in dimensions if f'{dim}_normalized' in df_lmm_ie.columns]
    if normalized_cols:
        print(f"  - Created {len(normalized_cols)} normalized columns")
        for col in normalized_cols:
            non_nan_count = df_lmm_ie[col].notna().sum()
            print(f"    {col}: {non_nan_count} non-NaN values")
    
    # =========================================================================
    # STEP 5: Exclude baseline condition from IE analysis
    # =========================================================================
    if exclude_baseline:
        print("\nStep 5: Excluding baseline condition from IE analysis...")
        
        if "inclusion_exclusion" not in df_lmm_ie.columns:
            raise ValueError(
                "exclude_baseline is True but 'inclusion_exclusion' column is missing."
            )
        
        before_n_ie = len(df_lmm_ie)
        before_s_ie = df_lmm_ie["subject_id"].nunique()
        
        df_lmm_ie = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] != "baseline"].copy()
        
        after_n_ie = len(df_lmm_ie)
        after_s_ie = df_lmm_ie["subject_id"].nunique()
        
        print(f"  - IE-level: {before_n_ie} rows/{before_s_ie} subjects -> {after_n_ie} rows/{after_s_ie} subjects")
    else:
        print("\nStep 5: Keeping baseline condition in IE analysis")
    
    # =========================================================================
    # STEP 6: Remove rows with missing normalized values
    # =========================================================================
    print("\nStep 6: Cleaning up rows with missing normalized values...")
    
    before_dropna = len(df_lmm_ie)
    before_subjects = df_lmm_ie["subject_id"].nunique()
    
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
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Final probe-level dataset: {len(df_lmm)} rows, {df_lmm['subject_id'].nunique()} subjects")
    print(f"Final IE-level dataset:    {len(df_lmm_ie)} rows, {df_lmm_ie['subject_id'].nunique()} subjects")
    
    if normalized_cols:
        print("\nNormalized dimension ranges (IE analysis):")
        for dim in dimensions:
            col = f'{dim}_normalized'
            if col in df_lmm_ie.columns and len(df_lmm_ie) > 0:
                print(f"  {col}: [{df_lmm_ie[col].min():.2f}, {df_lmm_ie[col].max():.2f}]")
    
    return df_lmm, df_lmm_ie


def main() -> None:
    """Entry point to run the multi-dimension analysis loop."""
    ensure_directories()
    df_lmm, df_lmm_ie = load_dataframes()
    
    # Update output directories based on filtering options
    global RESULTS_DIR, PLOTS_DIR
    if APPLY_ONOFF_FILTER:
        RESULTS_DIR = os.path.join(RESULTS_DIR, f"onoff_lt{int(ONOFF_MAX_EXCLUSIVE)}_combined_baseline")
        PLOTS_DIR = os.path.join(PLOTS_DIR, f"onoff_lt{int(ONOFF_MAX_EXCLUSIVE)}_combined_baseline")
    ensure_directories()
    
    # Validate columns before preprocessing
    validate_columns(df_lmm, df_lmm_ie, DIMENSIONS)
    
    # Preprocess data (filter first, then compute combined SART1+SART3 baseline)
    df_lmm, df_lmm_ie = preprocess_data(
        df_lmm=df_lmm,
        df_lmm_ie=df_lmm_ie,
        dimensions=DIMENSIONS,
        apply_onoff_filter=APPLY_ONOFF_FILTER,
        onoff_threshold=ONOFF_MAX_EXCLUSIVE,
        exclude_baseline=EXCLUDE_BASELINE,
    )

    # Run analysis for each dimension
    for dep in DIMENSIONS:
        analyze_dimension(df_lmm, df_lmm_ie, dep)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"All results base dir: {RESULTS_DIR}")
    print(f"All plots base dir:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()

#%%
