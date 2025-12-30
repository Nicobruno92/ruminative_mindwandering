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
    "results/Behavior/probe_data/pca_results.csv"
)
EVA_DATA_FILE: str = (
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Output directory
RESULTS_DIR: str = (
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

# All mood dimensions from EVA scale to analyze
MOOD_COLUMNS: List[str] = [
    "EVAaverage",
    "EVAmood",
    "EVAfeel",
    "EVAtense",
    "EVAhurt",
]

# Colors will be generated using husl palette (matching reverse_mediation style)

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
ONOFF_MAX_EXCLUSIVE: float = 62.0

# Standardization
# If True, all thought dimensions (including pca1) are z-scored
# at the block level before fitting the mediation models for comparability.
STANDARDIZE_THOUGHTS: bool = True

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
        Block-level data with mean thought scores and all mood_pre columns per block.
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
    
    required_eva_cols = {"subject_id", "task", "block_number"} | set(MOOD_COLUMNS)
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
    
    # Extract mood_pre for ALL mood columns (first EVA block per subject × task)
    def get_all_mood_pre(group: pd.DataFrame) -> pd.Series:
        group_sorted = group.sort_values("block_number")
        result = {}
        for mood_col in MOOD_COLUMNS:
            mood_vals = group_sorted[mood_col].dropna()
            if mood_vals.empty:
                result[f"mood_pre_{mood_col}"] = np.nan
            else:
                result[f"mood_pre_{mood_col}"] = float(mood_vals.iloc[0])
        return pd.Series(result)
    
    df_mood = (
        df_eva.groupby(["subject_id", "task"], as_index=False)
        .apply(get_all_mood_pre, include_groups=False)
        .reset_index(drop=True)
    )
    
    # Merge thoughts and mood
    df_block = df_block_thoughts.merge(
        df_mood, on=["subject_id", "task"], how="inner"
    )
    
    # Remove rows missing ALL mood data
    mood_pre_cols = [f"mood_pre_{m}" for m in MOOD_COLUMNS]
    before = len(df_block)
    df_block = df_block.dropna(subset=mood_pre_cols, how="all").reset_index(drop=True)
    print(f"Block-level dataset: {before} → {len(df_block)} blocks after removing missing mood")
    
    # Optional: Standardize thought dimensions for comparability
    if STANDARDIZE_THOUGHTS:
        thought_cols = [f"mean_{d}" for d in THOUGHT_DIMENSIONS if f"mean_{d}" in df_block.columns]
        for col in thought_cols:
            df_block[col] = stats.zscore(df_block[col], nan_policy="omit")
        print(f"Standardized {len(thought_cols)} thought dimensions (z-scored)")
    
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


def fit_path_a(data: pd.DataFrame, mood_col: str) -> Tuple[float, float, bool]:
    """
    Fit Path a: Mood ~ Group + (1|Subject)
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    mood_col : str
        Mood column name (e.g., 'mood_pre_EVAaverage').
    
    Returns
    -------
    tuple
        (coefficient, standard_error, converged)
    """
    formula = f"{mood_col} ~ group_binary"
    
    try:
        model = smf.mixedlm(formula, data, groups=data["subject_id"])
        result = model.fit(method="lbfgs", maxiter=1000)
        
        coef = result.params["group_binary"]
        se = result.bse["group_binary"]
        
        return float(coef), float(se), True
        
    except Exception as e:
        warnings.warn(f"Path a failed to converge for {mood_col}: {e}")
        return np.nan, np.nan, False


def fit_path_b(data: pd.DataFrame, thought_dim: str, mood_col: str) -> Tuple[float, float, bool]:
    """
    Fit Path b: Thought ~ Mood + Group + (1|Subject)
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    thought_dim : str
        Thought dimension to analyze.
    mood_col : str
        Mood column name (e.g., 'mood_pre_EVAaverage').
    
    Returns
    -------
    tuple
        (coefficient, standard_error, converged)
    """
    mean_col = f"mean_{thought_dim}"
    formula = f"{mean_col} ~ {mood_col} + group_binary"
    
    try:
        model = smf.mixedlm(formula, data, groups=data["subject_id"])
        result = model.fit(method="lbfgs", maxiter=1000)
        
        coef = result.params[mood_col]
        se = result.bse[mood_col]
        
        return float(coef), float(se), True
        
    except Exception as e:
        warnings.warn(f"Path b failed for {thought_dim} ~ {mood_col}: {e}")
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
    Run mediation analysis for all thought dimensions × all mood dimensions.
    
    Parameters
    ----------
    data : pd.DataFrame
        Block-level data.
    
    Returns
    -------
    pd.DataFrame
        Results table with indirect effects and CIs for all combinations.
    """
    print("\n" + "=" * 70)
    print("MEDIATION ANALYSIS")
    print("=" * 70)
    print(f"Model: Group → Mood → Thought Content")
    print(f"Mood measures: {MOOD_COLUMNS}")
    print(f"Monte Carlo simulations: {N_SIMULATIONS:,}")
    print(f"Confidence level: {CONFIDENCE_LEVEL * 100:.0f}%")
    
    results = []
    
    # Iterate over all mood dimensions
    for mood_name in MOOD_COLUMNS:
        mood_col = f"mood_pre_{mood_name}"
        
        if mood_col not in data.columns:
            print(f"\nSkipping {mood_name}: column {mood_col} not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"MOOD DIMENSION: {mood_name}")
        print(f"{'='*60}")
        
        # Fit Path a for this mood dimension
        print(f"\n--- Path a: {mood_name} ~ Group + (1|Subject) ---")
        a_coef, a_se, a_converged = fit_path_a(data, mood_col)
        
        if not a_converged:
            print(f"  Path a failed to converge for {mood_name}")
            for dim in THOUGHT_DIMENSIONS:
                results.append({
                    "mood": mood_name,
                    "dimension": dim,
                    "a_coeff": np.nan,
                    "a_se": np.nan,
                    "b_coeff": np.nan,
                    "b_se": np.nan,
                    "ab_effect": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "is_significant": False,
                })
            continue
        
        print(f"  Group effect on {mood_name}: β = {a_coef:.4f}, SE = {a_se:.4f}")
        
        # Fit Path b for each thought dimension
        for dim in THOUGHT_DIMENSIONS:
            mean_col = f"mean_{dim}"
            
            if mean_col not in data.columns:
                print(f"\n  Skipping {dim}: column {mean_col} not found")
                continue
            
            print(f"\n--- Path b: {dim} ~ {mood_name} + Group + (1|Subject) ---")
            
            b_coef, b_se, b_converged = fit_path_b(data, dim, mood_col)
            
            if not b_converged:
                print(f"    Path b failed to converge for {dim}")
                results.append({
                    "mood": mood_name,
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
            
            print(f"    {mood_name} effect on {dim}: β = {b_coef:.4f}, SE = {b_se:.4f}")
            
            # Compute indirect effect
            ab_mean, ci_low, ci_high, is_sig = monte_carlo_indirect_effect(
                a_coef, a_se, b_coef, b_se
            )
            
            print(f"    Indirect effect (a×b): {ab_mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
            print(f"    Significant: {'YES' if is_sig else 'NO'}")
            
            results.append({
                "mood": mood_name,
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
    Create combined figure with forest plot showing all mood dimensions.
    Style matches reverse_mediation_analysis.py for consistency.
    
    Parameters
    ----------
    results : pd.DataFrame
        Mediation results table with columns: mood, dimension, a_coeff, etc.
    output_path : str
        Path to save the figure.
    """
    print("\n" + "=" * 70)
    print("CREATING COMBINED MEDIATION FIGURE")
    print("=" * 70)
    
    from matplotlib.lines import Line2D
    
    # Filter out dimensions with missing results
    plot_data = results[results["ab_effect"].notna()].copy()
    
    if plot_data.empty:
        print("No valid results to plot.")
        return
    
    # Setup combined figure (matching reverse_mediation style)
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1, 1.2], hspace=0.35, wspace=0.3)
    
    # Create color palette for moods (husl like reverse_mediation)
    palette = sns.color_palette("husl", n_colors=len(MOOD_COLUMNS))
    mood_colors = dict(zip(MOOD_COLUMNS, palette))
    
    # ========== PANEL 1: FOREST PLOT (TOP, SPANS BOTH COLUMNS) ==========
    ax_forest = fig.add_subplot(gs[0, :])
    
    # Use consistent order for dimensions and moods
    dims = THOUGHT_DIMENSIONS
    moods = MOOD_COLUMNS
    
    y_positions = np.arange(len(dims))
    height_per_group = 0.8
    bar_height = height_per_group / len(moods)
    
    # Add background "cells" for each dimension
    for y in y_positions:
        if y % 2 == 0:
            ax_forest.axhspan(y - 0.5, y + 0.5, color='gray', alpha=0.1, zorder=0, linewidth=0)
    
    # Loop to plot each mood's effect
    for i, mood in enumerate(moods):
        subset = plot_data[plot_data["mood"] == mood]
        
        # Align data to dimensions order
        subset_indexed = subset.set_index("dimension").reindex(dims)
        
        # Y-offsets to dodge points
        offsets = y_positions + (i - (len(moods)-1)/2) * bar_height * 0.8
        
        # Iterate through each dimension to plot individually
        for j, dim in enumerate(dims):
            if dim not in subset_indexed.index:
                continue
                
            row = subset_indexed.loc[dim]
            if pd.isna(row["ab_effect"]):
                continue
                
            y_pos = offsets[j]
            x = row["ab_effect"]
            ci_low = row["ci_lower"]
            ci_high = row["ci_upper"]
            is_sig = row["is_significant"]
            
            # Styling based on significance
            color = mood_colors[mood]
            alpha = 1.0 if is_sig else 0.4
            linestyle = '-' if is_sig else ':'
            marker_face_color = color if is_sig else 'white'
            
            # Error bar
            ax_forest.plot([ci_low, ci_high], [y_pos, y_pos], 
                    color=color, alpha=alpha, linestyle=linestyle, linewidth=2, zorder=2)
            
            # Point estimate
            ax_forest.plot(x, y_pos, 
                    marker='o', markersize=9, 
                    markeredgecolor=color, markerfacecolor=marker_face_color,
                    markeredgewidth=2, alpha=alpha, zorder=3)

    # Aesthetics
    ax_forest.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels(dims, fontweight='bold', fontsize=12)
    ax_forest.set_ylim(-0.5, len(dims) - 0.5)
    
    ax_forest.set_xlabel("Indirect Effect (a × b)", fontweight='bold', fontsize=13)
    ax_forest.set_title(
        f"Mediation: Group → Mood → Thought Content (Indirect Effects)\n"
        f"(Monte Carlo: {N_SIMULATIONS:,} simulations, {CONFIDENCE_LEVEL*100:.0f}% CI)",
        pad=20, fontweight='bold', fontsize=13
    )
    ax_forest.invert_yaxis()
    
    # Custom Legend (matching reverse_mediation style)
    legend_elements = [
        Line2D([0], [0], color='gray', label='Mood Mediators:', linewidth=0)
    ]
    for mood in moods:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=mood,
                   markerfacecolor=mood_colors[mood], markeredgecolor=mood_colors[mood], markersize=10)
        )
    
    legend_elements.append(Line2D([0], [0], color='gray', label='', linewidth=0))
    legend_elements.append(Line2D([0], [0], color='gray', label='Significance:', linewidth=0))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='black', label='Significant',
               markerfacecolor='black', linestyle='-')
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='black', label='Non-significant',
               markerfacecolor='white', markeredgecolor='black', linestyle=':', alpha=0.6)
    )
    
    ax_forest.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
    ax_forest.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # ========== PANEL 2: PATH A (Group -> Mood) ==========
    ax_path_a = fig.add_subplot(gs[1, 0])
    
    # Get unique path A coefficients per mood - reindex to standard order
    path_a_raw = (
        plot_data.drop_duplicates(subset=["mood"])
        .set_index("mood")
        .reindex(MOOD_COLUMNS)
    )
    path_a_data = path_a_raw.reset_index().dropna(subset=["a_coeff", "a_se"]).copy()
    path_a_data["a_significant"] = np.abs(path_a_data["a_coeff"] / path_a_data["a_se"]) > 1.96
    
    y_pos = np.arange(len(path_a_data))
    
    for idx, (_, row) in enumerate(path_a_data.iterrows()):
        val = row["a_coeff"]
        err = row["a_se"] * 1.96
        is_sig = row["a_significant"]
        mood_name = row["mood"]
        
        COLOR_RISK = "#F24236"
        COLOR_CONTROL = "#2E86AB"
        color = COLOR_RISK if val < 0 else COLOR_CONTROL
        
        # Styling based on significance
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        marker_face = color if is_sig else 'white'
        
        ax_path_a.plot([val - err, val + err], [idx, idx], 
                       color=color, lw=2, linestyle=linestyle, alpha=alpha)
        ax_path_a.plot(val, idx, 'o', color=color, markersize=10,
                       markerfacecolor=marker_face, markeredgecolor=color,
                       markeredgewidth=2, alpha=alpha)
        ax_path_a.text(val, idx + 0.18, f"β={val:.2f}", ha='center', va='bottom',
                       fontsize=13, color=color, fontweight='bold', alpha=alpha)
    
    ax_path_a.set_yticks(y_pos)
    ax_path_a.set_yticklabels(path_a_data["mood"], fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coef (Group → Mood)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Group → Mood", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.invert_yaxis()
    
    # ========== PANEL 3: PATH B HEATMAP (Mood -> Thoughts) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    # Pivot: Filas=Thoughts (Outcome), Cols=Moods (Mediator)
    pivot_b = plot_data.pivot(index="dimension", columns="mood", values="b_coeff")
    pivot_b = pivot_b.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_COLUMNS)
    
    pivot_sig = plot_data.pivot(index="dimension", columns="mood", values="is_significant")
    pivot_sig = pivot_sig.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_COLUMNS)
    
    # Crear datos coloreados solo para celdas significativas
    colored_data = pivot_b.copy()
    colored_data[~pivot_sig] = np.nan
    
    # Heatmap solo con celdas significativas coloreadas
    sns.heatmap(colored_data, cmap="RdBu_r", center=0,
                ax=ax_path_b, cbar_kws={'label': 'Beta Coefficient'},
                linewidths=1, linecolor='white')
    
    # Añadir manualmente todas las anotaciones (incluyendo celdas blancas)
    for i in range(pivot_b.shape[0]):
        for j in range(pivot_b.shape[1]):
            val = pivot_b.iloc[i, j]
            if not pd.isna(val):
                text = f"{val:.2f}"
                if pivot_sig.iloc[i, j]:
                    text += "*"
                # Color del texto: negro para celdas blancas, blanco para celdas coloreadas
                text_color = 'black' if not pivot_sig.iloc[i, j] else 'white'
                ax_path_b.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontsize=14,
                    fontweight='bold',
                )
    
    ax_path_b.set_title("Path B: Mood → Thoughts (Predictive Power)", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.set_ylabel("Thought Dimension", fontweight='bold', fontsize=11)
    ax_path_b.set_xlabel("Mood Mediator", fontweight='bold', fontsize=11)
    
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
    
    # Reorder columns (now includes mood column)
    cols_to_keep = ["mood", "dimension", "a_coeff", "b_coeff", "ab_effect", "95% CI", "is_significant"]
    summary = summary[[c for c in cols_to_keep if c in summary.columns]]
    
    summary.columns = [
        "Mood Dimension",
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
            print(f"  - {row['mood']} → {row['dimension']}: {row['ab_effect']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
