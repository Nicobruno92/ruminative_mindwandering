#!/usr/bin/env python3
"""
Comparative Multilevel Mediation Analysis: Mood Specificity & Time Control

Comparing different mood scales as mediators between Depression Risk and Thought Content,
controlling for time-on-task effects.

Author: Senior Data Scientist - Computational Psychiatry
"""

from __future__ import annotations

import os
import warnings
from typing import List, Dict, Tuple, Optional

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
    "results/Behavior/mediation_analysis/multilevel_mediation_comparative"
)

# Thought dimensions to analyze# Thought dimensions (Outcomes) - Standardized order
THOUGHT_DIMENSIONS: List[str] = [
    "onoff",
    "valence", 
    "time",
    "selfother",
    "confidence",
    "pca1",
]

# Mood scales to compare (Mediators) - Standardized order
MOOD_SCALES: List[str] = [
    "EVAaverage",
    "EVAmood",
    "EVAfeel", 
    "EVAtense",
    "EVAhurt",
]

# Analysis parameters
N_SIMULATIONS: int = 10000
RANDOM_SEED: int = 42
CONFIDENCE_LEVEL: float = 0.95
MIN_BLOCKS: int = 25

# Standardization
# If True, all thought dimensions in THOUGHT_DIMENSIONS (including pca1)
# are z-scored at the block level before fitting the mediation models.
STANDARDIZE_THOUGHTS: bool = True

# Group coding
CONTROL_GROUP: str = "Controls"
RISK_GROUP: str = "Risk of Depression"

# Filtering
APPLY_ONOFF_FILTER: bool = True
ONOFF_MAX_EXCLUSIVE: float = 50.0


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def ensure_directories() -> None:
    """Create output directories."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Output directory: {RESULTS_DIR}")


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load, merge, and preprocess data for analysis.
    
    Returns
    -------
    pd.DataFrame
        Block-level data with standardized time covariate.
    """
    print("\n" + "=" * 70)
    print("DATA LOADING & PREPROCESSING")
    print("=" * 70)

    # 1. Load Probe Data
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe file not found: {PROBE_DATA_FILE}")
    df_probes = pd.read_csv(PROBE_DATA_FILE)
    print(f"Loaded {len(df_probes)} probe trials.")

    # Filter ON-task if requested
    if APPLY_ONOFF_FILTER and "onoff" in df_probes.columns:
        n_orig = len(df_probes)
        df_probes = df_probes[df_probes["onoff"] < ONOFF_MAX_EXCLUSIVE].copy()
        print(f"Applied ON-task filter (onoff < {ONOFF_MAX_EXCLUSIVE}): {n_orig} -> {len(df_probes)} trials")

    # 2. Aggregate Probes to Block Level (Subject x Task)
    # We assume 'task' (e.g., Sart1, Sart2) represents the block unit
    print("Aggregating probes to block level...")
    agg_cols = [d for d in THOUGHT_DIMENSIONS if d in df_probes.columns]
    if "pca1" not in agg_cols and "PC1" in df_probes.columns:
        df_probes["pca1"] = df_probes["PC1"]
        agg_cols.append("pca1")
    
    df_blocks = (
        df_probes.groupby(["subject_id", "task", "group"], as_index=False)[agg_cols]
        .mean()
    )

    # 3. Load EVA Data (Mood)
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA file not found: {EVA_DATA_FILE}")
    df_eva = pd.read_csv(EVA_DATA_FILE)
    print(f"Loaded {len(df_eva)} EVA entries.")

    # Select the relevant mood measurement per block
    # We take the first measurement (lowest block_number) within each task as 'mood_pre'
    # This represents the mood state entering the task
    print("Extracting pre-task mood scores...")
    
    def get_pre_task_mood(group):
        # Sort by internal block number to get the first one
        g_sorted = group.sort_values("block_number")
        if g_sorted.empty:
            return None
        # Return all mood columns for the first entry
        first_row = g_sorted.iloc[0]
        return first_row[MOOD_SCALES]

    df_mood = df_eva.groupby(["subject_id", "task"])[MOOD_SCALES].apply(
        lambda x: x.sort_index().iloc[0] if not x.empty else None
    ).reset_index()

    # 4. Merge Data
    df_merged = df_blocks.merge(df_mood, on=["subject_id", "task"], how="inner")

    # Optional standardization of thought dimensions (for comparability across mediators)
    if STANDARDIZE_THOUGHTS:
        thought_cols = [col for col in THOUGHT_DIMENSIONS if col in df_merged.columns]
        for col in thought_cols:
            df_merged[col] = stats.zscore(df_merged[col], nan_policy="omit")
    
    # 5. Create Covariates
    # Extract numeric block number from task string (Sart1 -> 1)
    df_merged["block_num"] = df_merged["task"].str.extract(r'(\d+)').astype(float)
    
    # Z-score block number (Time-on-task control)
    df_merged["block_z"] = stats.zscore(df_merged["block_num"], nan_policy='omit')
    
    # Binary Group (Controls=0, Risk=1)
    df_merged["group_bin"] = (df_merged["group"] == RISK_GROUP).astype(int)

    # Drop missing values
    df_merged = df_merged.dropna(subset=MOOD_SCALES + ["block_z", "group_bin"])
    
    print(f"Final Dataset: {len(df_merged)} blocks, {df_merged['subject_id'].nunique()} subjects.")
    print(f"Covariate 'block_z' created (Mean={df_merged['block_z'].mean():.2f}, Std={df_merged['block_z'].std():.2f})")
    
    return df_merged


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def fit_lmm(formula: str, data: pd.DataFrame, group_col: str = "subject_id") -> Tuple[float, float, bool]:
    """Fit LMM and return coef, se, and convergence status."""
    try:
        model = smf.mixedlm(formula, data, groups=data[group_col])
        result = model.fit(method="lbfgs", maxiter=2000, disp=False)
        
        # Extract parameter of interest (first predictor after Intercept usually, 
        # but we look it up by name in the calling function logic)
        return result, True
    except Exception as e:
        # warnings.warn(f"LMM Failed: {e}")
        return None, False


def monte_carlo_indirect(a, se_a, b, se_b, n_sims=10000):
    """Calculate indirect effect using Monte Carlo simulation."""
    np.random.seed(RANDOM_SEED)
    a_dist = np.random.normal(a, se_a, n_sims)
    b_dist = np.random.normal(b, se_b, n_sims)
    ab_dist = a_dist * b_dist
    
    mean_ab = np.mean(ab_dist)
    alpha = 1 - CONFIDENCE_LEVEL
    ci_low = np.percentile(ab_dist, 100 * alpha / 2)
    ci_high = np.percentile(ab_dist, 100 * (1 - alpha / 2))
    
    is_sig = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
    return mean_ab, ci_low, ci_high, is_sig


def run_comparative_mediation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run mediation analysis for every combination of Mood Scale and Thought Dimension.
    """
    print("\n" + "=" * 70)
    print("RUNNING ITERATIVE MEDIATION MODELS")
    print("=" * 70)
    print(f"Controlling for Time-on-Task (block_z)")
    
    results = []
    
    total_models = len(MOOD_SCALES) * len(THOUGHT_DIMENSIONS)
    count = 0
    
    for mood in MOOD_SCALES:
        # --- Path A: Mood ~ Group + Time + (1|Subj) ---
        # Only needs to be run once per mood scale
        formula_a = f"{mood} ~ group_bin + block_z"
        res_a, conv_a = fit_lmm(formula_a, data)
        
        if not conv_a:
            print(f"Skipping {mood}: Path A failed to converge.")
            continue
            
        a_coef = res_a.params["group_bin"]
        a_se = res_a.bse["group_bin"]
        
        # COFACTOR 1: Efecto del Tiempo sobre el Mood
        time_on_mood = res_a.params["block_z"]
        time_on_mood_se = res_a.bse["block_z"]
        
        for thought in THOUGHT_DIMENSIONS:
            count += 1
            if thought not in data.columns:
                continue
                
            print(f"[{count}/{total_models}] Mediator: {mood} -> Outcome: {thought} ...", end="\r")
            
            # --- Path B: Thought ~ Mood + Group + Time + (1|Subj) ---
            formula_b = f"{thought} ~ {mood} + group_bin + block_z"
            res_b, conv_b = fit_lmm(formula_b, data)
            
            if not conv_b:
                continue
                
            b_coef = res_b.params[mood]
            b_se = res_b.bse[mood]
            
            # COFACTOR 2: Efecto del Tiempo sobre los Pensamientos
            time_on_thought = res_b.params["block_z"]
            time_on_thought_se = res_b.bse["block_z"]
            
            # Indirect Effect
            ab, ci_lo, ci_hi, sig = monte_carlo_indirect(
                a_coef, a_se, b_coef, b_se, N_SIMULATIONS
            )
            
            results.append({
                "mood_scale": mood,
                "thought_dim": thought,
                "a_coef": a_coef,
                "a_se": a_se,
                "b_coef": b_coef,
                "b_se": b_se,
                "ab_effect": ab,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "is_significant": sig,
                # --- Guardamos los Cofactores ---
                "time_on_mood": time_on_mood,
                "time_on_mood_se": time_on_mood_se,
                "time_on_thought": time_on_thought,
                "time_on_thought_se": time_on_thought_se
            })
            
    print(f"\nCompleted {len(results)} successful mediation models.")
    return pd.DataFrame(results)


def plot_comparative_covariates(results: pd.DataFrame):
    """
    Dashboard para los Cofactores del Modelo Comparativo:
    1. Time -> Mood (Fatiga emocional?)
    2. Time -> Thoughts (Fatiga mental?)
    """
    print("\nGenerating Covariates Dashboard...")
    sns.set_theme(style="whitegrid", context="talk")
    COLOR_COV = "#7f8c8d"  # Gris Concreto (Neutro)
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2)
    
    # --- PANEL 1: Time -> Mood ---
    ax1 = fig.add_subplot(gs[0])
    # Agrupamos por mood (promedio de los modelos) y reordenamos según MOOD_SCALES
    df_time_mood = (
        results.groupby("mood_scale", as_index=True)[["time_on_mood", "time_on_mood_se"]]
        .mean(numeric_only=True)
        .reindex(MOOD_SCALES)
    )
    
    y_pos = np.arange(len(df_time_mood))
    for idx, (_, row) in enumerate(df_time_mood.iterrows()):
        val = row["time_on_mood"]
        err = row["time_on_mood_se"] * 1.96
        ax1.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2)
        ax1.plot(val, idx, 'o', color=COLOR_COV, markersize=8)
        
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_time_mood["mood_scale"], fontweight='bold')
    ax1.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax1.set_title("Effect of Time on Mood", fontweight='bold')
    ax1.set_xlabel("Beta (Time -> Mood)")
    ax1.invert_yaxis()

    # --- PANEL 2: Time -> Thoughts ---
    ax2 = fig.add_subplot(gs[1])
    # Agrupamos por pensamiento (promedio de los modelos) y reordenamos según THOUGHT_DIMENSIONS
    df_time_thought = (
        results.groupby("thought_dim", as_index=True)[["time_on_thought", "time_on_thought_se"]]
        .mean(numeric_only=True)
        .reindex(THOUGHT_DIMENSIONS)
    )
    
    y_pos2 = np.arange(len(df_time_thought))
    for idx, (_, row) in enumerate(df_time_thought.iterrows()):
        val = row["time_on_thought"]
        err = row["time_on_thought_se"] * 1.96
        ax2.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2)
        ax2.plot(val, idx, 'o', color=COLOR_COV, markersize=8)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(df_time_thought["thought_dim"], fontweight='bold')
    ax2.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax2.set_title("Effect of Time on Thoughts", fontweight='bold')
    ax2.set_xlabel("Beta (Time -> Thought)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "comparative_covariates_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved covariates plot: {out_path}")
    plt.close()


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_detailed_results(results: pd.DataFrame, data: pd.DataFrame) -> None:
    """
    Save detailed model results to text files.
    
    Parameters
    ----------
    results : pd.DataFrame
        Mediation results table.
    data : pd.DataFrame
        Original dataset for descriptive statistics.
    """
    print("\nSaving detailed model results...")
    
    # Create detailed results text file
    txt_path = os.path.join(RESULTS_DIR, "detailed_model_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPARATIVE MULTILEVEL MEDIATION ANALYSIS - DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Analysis: Group (Depression Risk) → Mood → Thought Content\n")
        f.write(f"Controlling for: Time-on-Task (block_z)\n")
        f.write(f"Monte Carlo Simulations: {N_SIMULATIONS:,}\n")
        f.write(f"Confidence Level: {CONFIDENCE_LEVEL * 100:.0f}%\n\n")
        
        # Dataset summary
        f.write("-" * 80 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Blocks: {len(data)}\n")
        f.write(f"Total Subjects: {data['subject_id'].nunique()}\n")
        f.write(f"Controls: {(data['group_bin'] == 0).sum()} blocks\n")
        f.write(f"Risk Group: {(data['group_bin'] == 1).sum()} blocks\n\n")
        
        # Results by mood scale
        for mood in MOOD_SCALES:
            mood_results = results[results["mood_scale"] == mood]
            if mood_results.empty:
                continue
                
            f.write("=" * 80 + "\n")
            f.write(f"MEDIATOR: {mood}\n")
            f.write("=" * 80 + "\n\n")
            
            # Path A (same for all thought dimensions with this mood)
            first_row = mood_results.iloc[0]
            f.write("PATH A: Group → Mood\n")
            f.write(f"  Model: {mood} ~ group_bin + block_z + (1|subject_id)\n")
            f.write(f"  Coefficient (a): {first_row['a_coef']:.6f}\n")
            f.write(f"  Standard Error: {first_row['a_se']:.6f}\n")
            f.write(f"  t-value: {first_row['a_coef']/first_row['a_se']:.3f}\n\n")
            
            # Path B for each thought dimension
            f.write("PATH B: Mood → Thought (controlling for Group and Time)\n")
            f.write("-" * 80 + "\n")
            
            for _, row in mood_results.iterrows():
                thought = row["thought_dim"]
                f.write(f"\n  Outcome: {thought}\n")
                f.write(f"    Model: {thought} ~ {mood} + group_bin + block_z + (1|subject_id)\n")
                f.write(f"    Coefficient (b): {row['b_coef']:.6f}\n")
                f.write(f"    Standard Error: {row['b_se']:.6f}\n")
                f.write(f"    t-value: {row['b_coef']/row['b_se']:.3f}\n")
                f.write(f"\n    INDIRECT EFFECT (a × b): {row['ab_effect']:.6f}\n")
                f.write(f"    95% CI: [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]\n")
                f.write(f"    Significant: {'YES' if row['is_significant'] else 'NO'}\n")
                f.write("    " + "-" * 70 + "\n")
            
            f.write("\n")
        
        # Summary table of significant effects
        f.write("=" * 80 + "\n")
        f.write("SUMMARY: SIGNIFICANT INDIRECT EFFECTS\n")
        f.write("=" * 80 + "\n\n")
        
        sig_results = results[results["is_significant"]].sort_values("ab_effect", ascending=False)
        
        if sig_results.empty:
            f.write("No significant indirect effects found.\n\n")
        else:
            f.write(f"Found {len(sig_results)} significant indirect effects:\n\n")
            f.write(f"{'Mood Scale':<15} {'Thought Dim':<15} {'Effect (a×b)':<15} {'95% CI':<30}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in sig_results.iterrows():
                ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                f.write(f"{row['mood_scale']:<15} {row['thought_dim']:<15} {row['ab_effect']:>12.6f}   {ci_str:<30}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Detailed results saved: {txt_path}")
    
    # Also save a summary table by thought dimension
    summary_path = os.path.join(RESULTS_DIR, "results_by_dimension.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTS ORGANIZED BY THOUGHT DIMENSION\n")
        f.write("=" * 80 + "\n\n")
        
        for thought in THOUGHT_DIMENSIONS:
            thought_results = results[results["thought_dim"] == thought]
            if thought_results.empty:
                continue
                
            f.write("-" * 80 + "\n")
            f.write(f"THOUGHT DIMENSION: {thought}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"{'Mood Mediator':<15} {'Effect (a×b)':<15} {'95% CI':<30} {'Significant':<12}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in thought_results.iterrows():
                ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                sig_str = "YES" if row['is_significant'] else "NO"
                f.write(f"{row['mood_scale']:<15} {row['ab_effect']:>12.6f}   {ci_str:<30} {sig_str:<12}\n")
            
            f.write("\n")
    
    print(f"Results by dimension saved: {summary_path}")


def plot_combined_mediation_figure(results: pd.DataFrame):
    """
    Generate combined figure with:
    - Top: Forest plot (indirect effects)
    - Middle Left: Path A (Group -> Mood)
    - Middle Right: Path B Heatmap (Mood -> Thoughts)
    - Bottom Left: Time -> Mood (covariate)
    - Bottom Right: Time -> Thoughts (covariate)
    """
    print("\nGenerating combined mediation visualization...")
    
    # Setup Plot
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 0.8], width_ratios=[1, 1.5], hspace=0.3, wspace=0.3)
    
    # ========== PANEL 1: FOREST PLOT (TOP, SPANS BOTH COLUMNS) ==========
    ax_forest = fig.add_subplot(gs[0, :])
    
    # Create color palette for moods
    palette = sns.color_palette("husl", n_colors=len(MOOD_SCALES))
    mood_colors = dict(zip(MOOD_SCALES, palette))
    
    # Use consistent order for dimensions and moods
    dims = THOUGHT_DIMENSIONS
    moods = MOOD_SCALES
    
    y_positions = np.arange(len(dims))
    height_per_group = 0.8
    bar_height = height_per_group / len(moods)
    
    # Add background "cells" for each dimension
    for y in y_positions:
        if y % 2 == 0:
            ax_forest.axhspan(y - 0.5, y + 0.5, color='gray', alpha=0.1, zorder=0, linewidth=0)
    
    # Loop to plot each mood's effect
    for i, mood in enumerate(moods):
        subset = results[results["mood_scale"] == mood]
        
        # Align data to dimensions order
        subset_indexed = subset.set_index("thought_dim").reindex(dims)
        
        # Y-offsets to dodge points
        offsets = y_positions + (i - (len(moods)-1)/2) * bar_height * 0.8
        
        # Iterate through each dimension to plot individually (for significance styling)
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
    ax_forest.set_title("Forward Mediation: Group → Mood → Thoughts (Indirect Effects)\n(Controlled for Time-on-Task)", 
                 pad=20, fontweight='bold', fontsize=13)
    ax_forest.invert_yaxis()
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', label='Mood Scales:', linewidth=0)
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
    
    # Path A es igual para todos los thoughts, quitamos duplicados - use standardized order
    path_a_data = results.drop_duplicates(subset=["mood_scale"]).set_index("mood_scale").reindex(MOOD_SCALES).dropna().reset_index()
    
    # Check significance for Path A (t-test: |coef/se| > 1.96)
    path_a_data["a_significant"] = np.abs(path_a_data["a_coef"] / path_a_data["a_se"]) > 1.96
    
    y_pos = np.arange(len(path_a_data))
    
    for idx, (_, row) in enumerate(path_a_data.iterrows()):
        val = row["a_coef"]
        err = row["a_se"] * 1.96
        is_sig = row["a_significant"]
        
        # Color based on direction
        COLOR_RISK = "#F24236"
        COLOR_CONTROL = "#2E86AB"
        color = COLOR_RISK if val < 0 else COLOR_CONTROL
        
        # Styling based on significance (matching forest plot)
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        marker_face = color if is_sig else 'white'
        
        # Error bar
        ax_path_a.plot([val - err, val + err], [idx, idx], 
                       color=color, lw=2, linestyle=linestyle, alpha=alpha)
        # Point
        ax_path_a.plot(val, idx, 'o', color=color, markersize=10, 
                       markerfacecolor=marker_face, markeredgecolor=color,
                       markeredgewidth=2, alpha=alpha)
        
        # Annotate value (even larger font for publication-ready readability)
        ax_path_a.text(
            val,
            idx + 0.18,
            f"β={val:.2f}",
            ha='center',
            va='bottom',
            fontsize=13,
            color=color,
            fontweight='bold',
            alpha=alpha,
        )

    ax_path_a.set_yticks(y_pos)
    ax_path_a.set_yticklabels(path_a_data["mood_scale"], fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coefficient (Group Effect on Mood)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Group → Mood", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.invert_yaxis()
    
    # ========== PANEL 3: PATH B HEATMAP (Mood -> Thought) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    # Pivot para Heatmap: Filas=Thoughts, Cols=Moods - ensure consistent order
    pivot_b = results.pivot(index="thought_dim", columns="mood_scale", values="b_coef")
    pivot_b = pivot_b.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
    # Máscara de significancia
    pivot_sig = results.pivot(index="thought_dim", columns="mood_scale", values="is_significant")
    pivot_sig = pivot_sig.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
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
    
    ax_path_b.set_title("Path B: Mood → Thoughts (Intrinsic Coupling)", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.set_ylabel("Thought Dimension", fontweight='bold', fontsize=11)
    ax_path_b.set_xlabel("Mood Scale", fontweight='bold', fontsize=11)

    # ========== PANEL 4: TIME -> MOOD (COVARIATE) ==========
    ax_time_mood = fig.add_subplot(gs[2, 0])
    COLOR_COV = "#7f8c8d"  # Gris Concreto (Neutro)
    
    # Agrupamos por mood (promedio de los modelos) - ensure consistent order
    df_time_mood = results.drop_duplicates(subset=["mood_scale"]).set_index("mood_scale").reindex(MOOD_SCALES).dropna()
    
    y_pos = np.arange(len(df_time_mood))
    for idx, (_, row) in enumerate(df_time_mood.iterrows()):
        val = row["time_on_mood"]
        err = row["time_on_mood_se"] * 1.96
        
        # Significance styling for covariates
        is_sig = abs(val) > 1.96 * row["time_on_mood_se"]
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        
        ax_time_mood.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2, alpha=alpha, linestyle=linestyle)
        ax_time_mood.plot(val, idx, 'o', color=COLOR_COV, markersize=8, alpha=alpha)
        
    ax_time_mood.set_yticks(y_pos)
    ax_time_mood.set_yticklabels(df_time_mood.index, fontweight='bold')
    ax_time_mood.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax_time_mood.set_title("Time → Mood (Fatigue)", fontweight='bold', pad=12, fontsize=12)
    ax_time_mood.set_xlabel("Beta (Time Effect)", fontweight='bold', fontsize=11)
    ax_time_mood.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_time_mood.invert_yaxis()

    # ========== PANEL 5: TIME -> THOUGHTS (COVARIATE) ==========
    ax_time_thought = fig.add_subplot(gs[2, 1])
    
    # Agrupamos por pensamiento (promedio de los modelos) - ensure consistent order
    df_time_thought = results.drop_duplicates(subset=["thought_dim"]).set_index("thought_dim").reindex(THOUGHT_DIMENSIONS).dropna()
    
    y_pos2 = np.arange(len(df_time_thought))
    for idx, (_, row) in enumerate(df_time_thought.iterrows()):
        val = row["time_on_thought"]
        err = row["time_on_thought_se"] * 1.96
        
        # Significance styling for covariates
        is_sig = abs(val) > 1.96 * row["time_on_thought_se"]
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        
        ax_time_thought.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2, alpha=alpha, linestyle=linestyle)
        ax_time_thought.plot(val, idx, 'o', color=COLOR_COV, markersize=8, alpha=alpha)

    ax_time_thought.set_yticks(y_pos2)
    ax_time_thought.set_yticklabels(df_time_thought.index, fontweight='bold')
    ax_time_thought.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax_time_thought.set_title("Time → Thoughts (Fatigue)", fontweight='bold', pad=12, fontsize=12)
    ax_time_thought.set_xlabel("Beta (Time Effect)", fontweight='bold', fontsize=11)
    ax_time_thought.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_time_thought.invert_yaxis()

    # Save combined figure
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "forward_mediation_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved: {out_path}")
    plt.close()


def main():
    ensure_directories()
    
    # 1. Load
    df = load_and_preprocess_data()
    
    # 2. Analyze
    if len(df) < MIN_BLOCKS:
        print("Insufficient data for analysis.")
        return
        
    results_df = run_comparative_mediation(df)
    
    # 3. Save Results
    csv_path = os.path.join(RESULTS_DIR, "comparative_mediation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")
    
    # 4. Save detailed text reports
    save_detailed_results(results_df, df)
    
    # 5. Generate combined visualization (now includes covariates)
    plot_combined_mediation_figure(results_df)
    
    print("\nDone.")

if __name__ == "__main__":
    main()
