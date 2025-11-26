#!/usr/bin/env python3
"""
Multilevel Mediation Analysis: Group → Mood → Thought Content

Tests whether Depression Risk Group affects Thought Content through Mood
using Linear Mixed Models with Monte Carlo simulation for indirect effects.

Author: Senior Data Scientist - Computational Psychiatry
"""

from __future__ import annotations

import os
from typing import List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input data files
PROBE_DATA_FILE: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/probe_data/pca_results.csv"
)
EVA_DATA_FILE: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Output directory
RESULTS_DIR: str = (
    "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/"
    "results/Behavior/mediation_analysis/multilevel_mediation"
)

# Thought dimensions to analyze (0-100 scale)
THOUGHT_DIMENSIONS: List[str] = [
    "valence",
    "time",
    "selfother",
    "onoff",
    "confidence",
    "pca1",
]

# Mood dimension from EVA scale
MOOD_COLUMN: str = "EVAaverage"  # Options: EVAtense, EVAfeel, EVAmood, EVAhurt, EVAaverage, "total_score"

# Monte Carlo simulation parameters
N_SIMULATIONS: int = 20000
RANDOM_SEED: int = 42
CONFIDENCE_LEVEL: float = 0.95

# Group coding (Controls = 0, Risk = 1)
CONTROL_GROUP: str = "Controls"
RISK_GROUP: str = "Risk of Depression"

# Minimum blocks required for analysis
MIN_BLOCKS: int = 25

# Optional: Filter ON-task probes only
APPLY_ONOFF_FILTER: bool = True
ONOFF_MAX_EXCLUSIVE: float = 50.0

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def ensure_directories() -> None:
    """Create output directories if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")


def load_and_merge_data() -> pd.DataFrame:
    """
    Load probe and EVA data, merge to create block-level dataset.
    
    Returns
    -------
    pd.DataFrame
        Block-level data with mean thought scores and mood_pre per block.
    """
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    # Load probe data
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe data not found: {PROBE_DATA_FILE}")
    
    df_probes = pd.read_csv(PROBE_DATA_FILE)
    print(f"Loaded {len(df_probes)} probe-level observations")
    
    # Handle PCA column naming
    if "PC1" in df_probes.columns and "pca1" not in df_probes.columns:
        df_probes["pca1"] = df_probes["PC1"]
    
    # Validate required columns
    required_probe_cols = {"subject_id", "task", "group"} | set(THOUGHT_DIMENSIONS)
    missing = required_probe_cols - set(df_probes.columns)
    if missing:
        raise ValueError(f"Missing probe columns: {sorted(missing)}")
    
    # Optional: Filter ON-task probes
    if APPLY_ONOFF_FILTER and "onoff" in df_probes.columns:
        before = len(df_probes)
        df_probes = df_probes[df_probes["onoff"] < ONOFF_MAX_EXCLUSIVE].copy()
        print(f"Applied ON-task filter (onoff < {ONOFF_MAX_EXCLUSIVE}): {before} → {len(df_probes)} probes")
    
    # Load EVA data
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA data not found: {EVA_DATA_FILE}")
    
    df_eva = pd.read_csv(EVA_DATA_FILE)
    print(f"Loaded {len(df_eva)} EVA block observations")
    
    required_eva_cols = {"subject_id", "task", "block_number", MOOD_COLUMN}
    missing = required_eva_cols - set(df_eva.columns)
    if missing:
        raise ValueError(f"Missing EVA columns: {sorted(missing)}")
    
    # Aggregate probes to block level (mean per subject × task)
    print("\nAggregating to block level...")
    agg_dict = {dim: "mean" for dim in THOUGHT_DIMENSIONS}
    df_block_thoughts = (
        df_probes.groupby(["subject_id", "task", "group"], as_index=False)
        .agg(agg_dict)
        .rename(columns={dim: f"mean_{dim}" for dim in THOUGHT_DIMENSIONS})
    )
    
    # Extract mood_pre (first EVA block per subject × task)
    def get_mood_pre(group: pd.DataFrame) -> pd.Series:
        group_sorted = group.sort_values("block_number")
        mood_vals = group_sorted[MOOD_COLUMN].dropna()
        if mood_vals.empty:
            return pd.Series({"mood_pre": np.nan})
        return pd.Series({"mood_pre": float(mood_vals.iloc[0])})
    
    df_mood = (
        df_eva.groupby(["subject_id", "task"], as_index=False)
        .apply(get_mood_pre)
        .reset_index(drop=True)
    )
    
    # Merge thoughts and mood
    df_block = df_block_thoughts.merge(
        df_mood, on=["subject_id", "task"], how="inner"
    )
    
    # Remove missing data
    before = len(df_block)
    df_block = df_block.dropna(subset=["mood_pre"]).reset_index(drop=True)
    print(f"Block-level dataset: {before} → {len(df_block)} blocks after removing missing mood")
    
    print(f"\nFinal dataset:")
    print(f"  N blocks: {len(df_block)}")
    print(f"  N subjects: {df_block['subject_id'].nunique()}")
    print(f"  N Controls: {(df_block['group'] == CONTROL_GROUP).sum()}")
    print(f"  N Risk: {(df_block['group'] == RISK_GROUP).sum()}")
    
    # Create binary group indicator (0 = Controls, 1 = Risk)
    df_block["group_binary"] = (df_block["group"] == RISK_GROUP).astype(int)
    
    return df_block


# =============================================================================
# MEDIATION ANALYSIS
# =============================================================================


def fit_path_a(data: pd.DataFrame) -> Tuple[float, float, bool]:
    """
    Fit Path a: Mood ~ Group + (1|Subject)
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    
    Returns
    -------
    tuple
        (coefficient, standard_error, converged)
    """
    formula = "mood_pre ~ group_binary"
    
    try:
        model = smf.mixedlm(formula, data, groups=data["subject_id"])
        result = model.fit(method="lbfgs", maxiter=1000)
        
        coef = result.params["group_binary"]
        se = result.bse["group_binary"]
        
        return float(coef), float(se), True
        
    except Exception as e:
        warnings.warn(f"Path a failed to converge: {e}")
        return np.nan, np.nan, False


def fit_path_b(data: pd.DataFrame, thought_dim: str) -> Tuple[float, float, bool]:
    """
    Fit Path b: Thought ~ Mood + Group + (1|Subject)
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    thought_dim : str
        Thought dimension to analyze.
    
    Returns
    -------
    tuple
        (coefficient, standard_error, converged)
    """
    mean_col = f"mean_{thought_dim}"
    formula = f"{mean_col} ~ mood_pre + group_binary"
    
    try:
        model = smf.mixedlm(formula, data, groups=data["subject_id"])
        result = model.fit(method="lbfgs", maxiter=1000)
        
        coef = result.params["mood_pre"]
        se = result.bse["mood_pre"]
        
        return float(coef), float(se), True
        
    except Exception as e:
        warnings.warn(f"Path b failed for {thought_dim}: {e}")
        return np.nan, np.nan, False


def monte_carlo_indirect_effect(
    a: float,
    se_a: float,
    b: float,
    se_b: float,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float, bool]:
    """
    Compute indirect effect (a × b) using Monte Carlo simulation.
    
    Parameters
    ----------
    a : float
        Path a coefficient.
    se_a : float
        Path a standard error.
    b : float
        Path b coefficient.
    se_b : float
        Path b standard error.
    n_sims : int
        Number of Monte Carlo draws.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    tuple
        (mean_indirect, ci_lower, ci_upper, is_significant)
    """
    np.random.seed(seed)
    
    # Draw from normal distributions
    a_samples = np.random.normal(a, se_a, n_sims)
    b_samples = np.random.normal(b, se_b, n_sims)
    
    # Compute indirect effect distribution
    ab_samples = a_samples * b_samples
    
    # Calculate statistics
    mean_ab = float(np.mean(ab_samples))
    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = float(np.percentile(ab_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(ab_samples, 100 * (1 - alpha / 2)))
    
    # Significant if CI excludes zero
    is_significant = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
    
    return mean_ab, ci_lower, ci_upper, is_significant


def run_mediation_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run mediation analysis for all thought dimensions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    
    Returns
    -------
    pd.DataFrame
        Results table with indirect effects and CIs.
    """
    print("\n" + "=" * 70)
    print("MEDIATION ANALYSIS")
    print("=" * 70)
    print(f"Model: Group → Mood → Thought Content")
    print(f"Mood measure: {MOOD_COLUMN}")
    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")
    print(f"Confidence level: {CONFIDENCE_LEVEL * 100:.0f}%")
    
    # Fit Path a once (same for all thought dimensions)
    print("\n--- Path a: Mood ~ Group + (1|Subject) ---")
    a_coef, a_se, a_converged = fit_path_a(data)
    
    if not a_converged:
        raise RuntimeError("Path a failed to converge. Cannot proceed with mediation.")
    
    print(f"Group effect on Mood: β = {a_coef:.4f}, SE = {a_se:.4f}")
    
    # Fit Path b for each thought dimension
    results = []
    
    for dim in THOUGHT_DIMENSIONS:
        mean_col = f"mean_{dim}"
        
        if mean_col not in data.columns:
            print(f"\nSkipping {dim}: column {mean_col} not found")
            continue
        
        print(f"\n--- Path b: {dim} ~ Mood + Group + (1|Subject) ---")
        
        b_coef, b_se, b_converged = fit_path_b(data, dim)
        
        if not b_converged:
            print(f"  Path b failed to converge for {dim}")
            results.append({
                "dimension": dim,
                "a_coeff": a_coef,
                "a_se": a_se,
                "b_coeff": np.nan,
                "b_se": np.nan,
                "ab_effect": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "is_significant": False,
            })
            continue
        
        print(f"  Mood effect on {dim}: β = {b_coef:.4f}, SE = {b_se:.4f}")
        
        # Compute indirect effect
        ab_mean, ci_low, ci_high, is_sig = monte_carlo_indirect_effect(
            a_coef, a_se, b_coef, b_se
        )
        
        print(f"  Indirect effect (a×b): {ab_mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Significant: {'YES' if is_sig else 'NO'}")
        
        results.append({
            "dimension": dim,
            "a_coeff": a_coef,
            "a_se": a_se,
            "b_coeff": b_coef,
            "b_se": b_se,
            "ab_effect": ab_mean,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "is_significant": is_sig,
        })
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_mediation_forest(results: pd.DataFrame, output_path: str) -> None:
    """
    Create combined figure with forest plot, Path A, and Path B details.
    
    Parameters
    ----------
    results : pd.DataFrame
        Mediation results table.
    output_path : str
        Path to save the figure.
    """
    print("\n" + "=" * 70)
    print("CREATING COMBINED MEDIATION FIGURE")
    print("=" * 70)
    
    # Filter out dimensions with missing results
    plot_data = results[results["ab_effect"].notna()].copy()
    
    if plot_data.empty:
        print("No valid results to plot.")
        return
    
    # Sort by effect size for better visualization
    plot_data = plot_data.sort_values("ab_effect").reset_index(drop=True)
    
    # Setup combined figure
    import seaborn as sns
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1.2], hspace=0.3, wspace=0.3)
    
    # ========== PANEL 1: FOREST PLOT (TOP, SPANS BOTH COLUMNS) ==========
    ax_forest = fig.add_subplot(gs[0, :])
    
    # Add zebra stripes
    y_positions = np.arange(len(plot_data))
    for y in y_positions:
        if y % 2 == 0:
            ax_forest.axhspan(y - 0.5, y + 0.5, color='gray', alpha=0.1, zorder=0, linewidth=0)
    
    # Plot each dimension with significance styling
    for idx, row in plot_data.iterrows():
        is_sig = row["is_significant"]
        color = "#D32F2F" if is_sig else "#757575"
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        marker_face = color if is_sig else 'white'
        
        # Confidence interval
        ax_forest.plot(
            [row["ci_lower"], row["ci_upper"]],
            [idx, idx],
            color=color,
            linewidth=2,
            linestyle=linestyle,
            alpha=alpha,
            zorder=2,
        )
        
        # Point estimate
        ax_forest.plot(
            row["ab_effect"],
            idx,
            marker="o",
            markersize=9,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=2,
            alpha=alpha,
            zorder=3,
        )
    
    # Formatting
    ax_forest.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3, zorder=1)
    ax_forest.set_yticks(range(len(plot_data)))
    ax_forest.set_yticklabels(plot_data["dimension"], fontweight='bold', fontsize=12)
    ax_forest.set_xlabel("Indirect Effect (a × b)", fontsize=13, fontweight="bold")
    ax_forest.set_title(
        f"Mediation: Group → {MOOD_COLUMN} → Thought Content (Indirect Effects)\n"
        f"(Monte Carlo: {N_SIMULATIONS:,} simulations, {CONFIDENCE_LEVEL*100:.0f}% CI)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    ax_forest.grid(axis="x", alpha=0.3, linestyle="--")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="black", label="Significant",
               markerfacecolor="black", linestyle="-"),
        Line2D([0], [0], marker="o", color="black", label="Non-significant",
               markerfacecolor="white", markeredgecolor="black", linestyle=":", alpha=0.6),
    ]
    ax_forest.legend(handles=legend_elements, loc="best", frameon=True, fontsize=10)
    
    # ========== PANEL 2: PATH A (Group -> Mood) ==========
    ax_path_a = fig.add_subplot(gs[1, 0])
    
    # Path A is the same for all dimensions (single mood scale)
    a_coef = plot_data.iloc[0]["a_coeff"]
    a_se = plot_data.iloc[0]["a_se"]
    a_ci = a_se * 1.96
    a_sig = np.abs(a_coef / a_se) > 1.96
    
    COLOR_RISK = "#F24236"
    COLOR_CONTROL = "#2E86AB"
    color = COLOR_RISK if a_coef < 0 else COLOR_CONTROL
    alpha = 1.0 if a_sig else 0.4
    linestyle = '-' if a_sig else ':'
    marker_face = color if a_sig else 'white'
    
    # Plot Path A
    ax_path_a.plot([a_coef - a_ci, a_coef + a_ci], [0, 0], 
                   color=color, lw=2, linestyle=linestyle, alpha=alpha)
    ax_path_a.plot(a_coef, 0, 'o', color=color, markersize=12,
                   markerfacecolor=marker_face, markeredgecolor=color,
                   markeredgewidth=2, alpha=alpha)
    ax_path_a.text(a_coef, 0.15, f"β={a_coef:.2f}", ha='center', va='bottom',
                   fontsize=10, color=color, fontweight='bold', alpha=alpha)
    
    ax_path_a.set_yticks([0])
    ax_path_a.set_yticklabels([MOOD_COLUMN], fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coefficient (Group Effect on Mood)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Group → Mood", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.set_ylim(-0.5, 0.5)
    
    # ========== PANEL 3: PATH B (Mood -> Thoughts) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    # Create bar plot for Path B coefficients
    dims = plot_data["dimension"].values
    b_coefs = plot_data["b_coeff"].values
    b_ses = plot_data["b_se"].values
    b_sigs = np.abs(b_coefs / b_ses) > 1.96
    
    y_pos = np.arange(len(dims))
    colors = [COLOR_RISK if b < 0 else COLOR_CONTROL for b in b_coefs]
    alphas = [1.0 if sig else 0.4 for sig in b_sigs]
    
    for i, (dim, b, se, sig, col, alph) in enumerate(zip(dims, b_coefs, b_ses, b_sigs, colors, alphas)):
        ci = se * 1.96
        linestyle = '-' if sig else ':'
        marker_face = col if sig else 'white'
        
        ax_path_b.plot([b - ci, b + ci], [i, i], color=col, lw=2, linestyle=linestyle, alpha=alph)
        ax_path_b.plot(b, i, 'o', color=col, markersize=9,
                       markerfacecolor=marker_face, markeredgecolor=col,
                       markeredgewidth=2, alpha=alph)
        ax_path_b.text(b, i + 0.15, f"{b:.2f}{'*' if sig else ''}", 
                       ha='center', va='bottom', fontsize=9, color=col, 
                       fontweight='bold', alpha=alph)
    
    ax_path_b.set_yticks(y_pos)
    ax_path_b.set_yticklabels(dims, fontweight='bold')
    ax_path_b.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_b.set_xlabel(f"Path B Coefficient ({MOOD_COLUMN} → Thought)", fontweight='bold', fontsize=11)
    ax_path_b.set_title(f"Path B: {MOOD_COLUMN} → Thoughts", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.grid(True, axis='x', alpha=0.2, linestyle=':')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Combined figure saved: {output_path}")


def create_summary_table(results: pd.DataFrame, output_path: str) -> None:
    """
    Create formatted summary table for publication.
    
    Parameters
    ----------
    results : pd.DataFrame
        Mediation results.
    output_path : str
        Path to save the table.
    """
    # Format for readability
    summary = results.copy()
    
    # Round numeric columns
    numeric_cols = ["a_coeff", "a_se", "b_coeff", "b_se", "ab_effect", "ci_lower", "ci_upper"]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(4)
    
    # Add formatted CI column
    summary["95% CI"] = summary.apply(
        lambda row: f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
        if pd.notna(row["ci_lower"]) else "NA",
        axis=1,
    )
    
    # Reorder columns
    summary = summary[[
        "dimension",
        "a_coeff",
        "b_coeff",
        "ab_effect",
        "95% CI",
        "is_significant",
    ]]
    
    summary.columns = [
        "Thought Dimension",
        "Path a (Group→Mood)",
        "Path b (Mood→Thought)",
        "Indirect Effect (a×b)",
        "95% CI",
        "Significant",
    ]
    
    summary.to_csv(output_path, index=False)
    print(f"\nSummary table saved: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main() -> None:
    """Execute multilevel mediation analysis pipeline."""
    print("\n" + "=" * 70)
    print("MULTILEVEL MEDIATION ANALYSIS")
    print("Group → Mood → Thought Content")
    print("=" * 70)
    
    # Setup
    ensure_directories()
    
    # Load and prepare data
    df_block = load_and_merge_data()
    
    # Check minimum sample size
    if len(df_block) < MIN_BLOCKS:
        raise ValueError(
            f"Insufficient data: {len(df_block)} blocks < {MIN_BLOCKS} minimum"
        )
    
    # Run mediation analysis
    results = run_mediation_analysis(df_block)
    
    # Save results
    results_csv = os.path.join(RESULTS_DIR, "mediation_results.csv")
    results.to_csv(results_csv, index=False)
    print(f"\nDetailed results saved: {results_csv}")
    
    # Create summary table
    summary_csv = os.path.join(RESULTS_DIR, "mediation_summary_table.csv")
    create_summary_table(results, summary_csv)
    
    # Create forest plot
    forest_plot = os.path.join(RESULTS_DIR, "plots", "mediation_forest_plot.png")
    plot_mediation_forest(results, forest_plot)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"\nSignificant indirect effects:")
    sig_results = results[results["is_significant"]]
    if sig_results.empty:
        print("  None")
    else:
        for _, row in sig_results.iterrows():
            print(f"  - {row['dimension']}: {row['ab_effect']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
