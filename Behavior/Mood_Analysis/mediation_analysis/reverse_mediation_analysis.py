#!/usr/bin/env python3
"""
Reverse Multilevel Mediation Analysis: Testing the Vicious Cycle Hypothesis

Tests whether Depression Risk Group affects Mood Post-Block through Thought Content.
Direction: Group → Thoughts → Mood Change (controlling for baseline mood)

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
    "results/Behavior/probe_data/pca_results.csv"
)
EVA_DATA_FILE: str = (
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Output directory
RESULTS_DIR: str = (
    "results/Behavior/mediation_analysis/reverse_mediation"
)

# Thought dimensions (MEDIATORS in reverse analysis) - Standardized order
THOUGHT_DIMENSIONS: List[str] = [
    "onoff",
    "valence", 
    "time",
    "selfother",
    "confidence",
    "pca1",
]

# Mood scales (OUTCOMES in reverse analysis) - Standardized order
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
# are z-scored at the block level before fitting the reverse mediation models.
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
    Load, merge, and preprocess data for reverse mediation analysis.
    
    Returns
    -------
    pd.DataFrame
        Block-level data with mood_pre, mood_post, and thought dimensions.
    """
    print("\n" + "=" * 70)
    print("DATA LOADING & PREPROCESSING - REVERSE MEDIATION")
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

    # 2. Aggregate Probes to Block Level
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

    # 4. Extract Mood Pre and Post per block
    print("Extracting mood_pre (current task) and mood_post (next task)...")
    
    # Take the first measurement within each task as the mood for that task
    df_mood_task = df_eva.sort_values(["subject_id", "task", "block_number"]).groupby(
        ["subject_id", "task"], as_index=False
    ).first()
    
    # Keep only mood columns
    mood_cols = ["subject_id", "task"] + MOOD_SCALES
    df_mood_task = df_mood_task[mood_cols]
    
    # Create mood_pre (current task) and mood_post (next task)
    # Sort by subject and task to ensure proper ordering
    df_mood_task["task_num"] = df_mood_task["task"].str.extract(r'(\d+)').astype(int)
    df_mood_task = df_mood_task.sort_values(["subject_id", "task_num"])
    
    # For each task, get its mood as "pre" and the next task's mood as "post"
    df_mood_pre = df_mood_task.copy()
    df_mood_post = df_mood_task.copy()
    
    # Shift mood_post by one task forward within each subject
    for mood in MOOD_SCALES:
        df_mood_post[f"{mood}_post"] = df_mood_post.groupby("subject_id")[mood].shift(-1)
        df_mood_pre[f"{mood}_pre"] = df_mood_pre[mood]
    
    # Merge pre and post
    merge_cols = ["subject_id", "task"]
    pre_cols = merge_cols + [f"{m}_pre" for m in MOOD_SCALES]
    post_cols = merge_cols + [f"{m}_post" for m in MOOD_SCALES]
    
    df_mood = df_mood_pre[pre_cols].merge(
        df_mood_post[post_cols], on=merge_cols, how="inner"
    )
    
    # Remove blocks without post data (last task per subject)
    df_mood = df_mood.dropna(subset=[f"{m}_post" for m in MOOD_SCALES])

    # 5. Merge Data
    df_merged = df_blocks.merge(df_mood, on=["subject_id", "task"], how="inner")

    # Optional standardization of thought dimensions (for comparability across mediators)
    if STANDARDIZE_THOUGHTS:
        thought_cols = [col for col in THOUGHT_DIMENSIONS if col in df_merged.columns]
        for col in thought_cols:
            df_merged[col] = stats.zscore(df_merged[col], nan_policy="omit")
    
    # 6. Create Covariates
    # Extract numeric block number from task string (Sart1 -> 1)
    df_merged["block_num"] = df_merged["task"].str.extract(r'(\d+)').astype(float)
    
    # Z-score block number (Time-on-task control)
    df_merged["block_z"] = stats.zscore(df_merged["block_num"], nan_policy='omit')
    
    # Binary Group (Controls=0, Risk=1)
    df_merged["group_bin"] = (df_merged["group"] == RISK_GROUP).astype(int)

    # Drop missing values
    required_cols = [f"{m}_pre" for m in MOOD_SCALES] + [f"{m}_post" for m in MOOD_SCALES]
    required_cols += ["block_z", "group_bin"]
    df_merged = df_merged.dropna(subset=required_cols)
    
    print(f"Final Dataset: {len(df_merged)} blocks, {df_merged['subject_id'].nunique()} subjects.")
    print(f"Covariate 'block_z' created (Mean={df_merged['block_z'].mean():.2f}, Std={df_merged['block_z'].std():.2f})")
    
    return df_merged


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def fit_lmm(formula: str, data: pd.DataFrame, group_col: str = "subject_id") -> Tuple[Optional[object], bool]:
    """Fit LMM and return result object and convergence status."""
    try:
        model = smf.mixedlm(formula, data, groups=data[group_col])
        result = model.fit(method="lbfgs", maxiter=2000, disp=False)
        return result, True
    except Exception as e:
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


def run_reverse_mediation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run reverse mediation analysis: Group → Thought → Mood Change.
    
    For each combination of Thought Dimension (Mediator) and Mood Scale (Outcome):
    - Path A: Thought ~ Group + block_z
    - Path B: Mood_Post ~ Thought + Group + Mood_Pre + block_z
    """
    print("\n" + "=" * 70)
    print("RUNNING REVERSE MEDIATION MODELS")
    print("=" * 70)
    print(f"Testing: Group → Thoughts → Mood Change")
    print(f"Controlling for: Baseline Mood + Time-on-Task")
    
    results = []
    
    total_models = len(THOUGHT_DIMENSIONS) * len(MOOD_SCALES)
    count = 0
    
    for thought in THOUGHT_DIMENSIONS:
        if thought not in data.columns:
            continue
        
        # --- Path A: Group → Thought ---
        # Only needs to be run once per thought dimension
        formula_a = f"{thought} ~ group_bin + block_z"
        res_a, conv_a = fit_lmm(formula_a, data)
        
        if not conv_a:
            print(f"Skipping {thought}: Path A failed to converge.")
            continue
            
        a_coef = res_a.params["group_bin"]
        a_se = res_a.bse["group_bin"]
        
        # COFACTOR 1: Efecto del Tiempo sobre el Pensamiento
        time_on_thought = res_a.params["block_z"]
        time_on_thought_se = res_a.bse["block_z"]
        
        for mood in MOOD_SCALES:
            count += 1
            mood_pre = f"{mood}_pre"
            mood_post = f"{mood}_post"
            
            if mood_pre not in data.columns or mood_post not in data.columns:
                continue
                
            print(f"[{count}/{total_models}] Mediator: {thought} -> Outcome: {mood} ...", end="\r")
            
            # --- Path B: Thought → Mood_Post (controlling for Mood_Pre) ---
            # This is the ANCOVA approach to measure mood CHANGE
            formula_b = f"{mood_post} ~ {thought} + group_bin + {mood_pre} + block_z"
            res_b, conv_b = fit_lmm(formula_b, data)
            
            if not conv_b:
                continue
                
            b_coef = res_b.params[thought]
            b_se = res_b.bse[thought]
            
            # COFACTOR 2: Efecto del Tiempo sobre Mood Post
            time_on_mood = res_b.params["block_z"]
            time_on_mood_se = res_b.bse["block_z"]
            
            # COFACTOR 3: Efecto del Baseline (Autocorrelación)
            baseline_effect = res_b.params[mood_pre]
            baseline_se = res_b.bse[mood_pre]
            
            # Indirect Effect
            ab, ci_lo, ci_hi, sig = monte_carlo_indirect(
                a_coef, a_se, b_coef, b_se, N_SIMULATIONS
            )
            
            results.append({
                "thought_dim": thought,
                "mood_scale": mood,
                "a_coef": a_coef,
                "a_se": a_se,
                "b_coef": b_coef,
                "b_se": b_se,
                "ab_effect": ab,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "is_significant": sig,
                # --- Guardamos los Cofactores ---
                "time_on_thought": time_on_thought,
                "time_on_thought_se": time_on_thought_se,
                "time_on_mood": time_on_mood,
                "time_on_mood_se": time_on_mood_se,
                "baseline_effect": baseline_effect,
                "baseline_se": baseline_se
            })
            
    print(f"\nCompleted {len(results)} successful mediation models.")
    return pd.DataFrame(results)


# =============================================================================
# REPORTING
# =============================================================================

def save_detailed_results(results: pd.DataFrame, data: pd.DataFrame) -> None:
    """Save detailed model results to text files."""
    print("\nSaving detailed model results...")
    
    # Detailed results file
    txt_path = os.path.join(RESULTS_DIR, "detailed_model_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("REVERSE MULTILEVEL MEDIATION ANALYSIS - DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hypothesis: Vicious Cycle - Group → Thoughts → Mood Worsening\n")
        f.write("Analysis: Group (Depression Risk) → Thought Content → Mood Post-Block\n")
        f.write(f"Controlling for: Baseline Mood (mood_pre) + Time-on-Task (block_z)\n")
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
        
        # Results by thought dimension (mediator)
        for thought in THOUGHT_DIMENSIONS:
            thought_results = results[results["thought_dim"] == thought]
            if thought_results.empty:
                continue
                
            f.write("=" * 80 + "\n")
            f.write(f"MEDIATOR: {thought}\n")
            f.write("=" * 80 + "\n\n")
            
            # Path A (same for all mood outcomes with this thought)
            first_row = thought_results.iloc[0]
            f.write("PATH A: Group → Thought\n")
            f.write(f"  Model: {thought} ~ group_bin + block_z + (1|subject_id)\n")
            f.write(f"  Coefficient (a): {first_row['a_coef']:.6f}\n")
            f.write(f"  Standard Error: {first_row['a_se']:.6f}\n")
            f.write(f"  t-value: {first_row['a_coef']/first_row['a_se']:.3f}\n\n")
            
            # Path B for each mood outcome
            f.write("PATH B: Thought → Mood Change (controlling for Baseline Mood, Group, Time)\n")
            f.write("-" * 80 + "\n")
            
            for _, row in thought_results.iterrows():
                mood = row["mood_scale"]
                f.write(f"\n  Outcome: {mood}_post\n")
                f.write(f"    Model: {mood}_post ~ {thought} + group_bin + {mood}_pre + block_z + (1|subject_id)\n")
                f.write(f"    Coefficient (b): {row['b_coef']:.6f}\n")
                f.write(f"    Standard Error: {row['b_se']:.6f}\n")
                f.write(f"    t-value: {row['b_coef']/row['b_se']:.3f}\n")
                f.write(f"\n    INDIRECT EFFECT (a × b): {row['ab_effect']:.6f}\n")
                f.write(f"    95% CI: [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}]\n")
                f.write(f"    Significant: {'YES' if row['is_significant'] else 'NO'}\n")
                f.write("    " + "-" * 70 + "\n")
            
            f.write("\n")
        
        # Summary of significant effects
        f.write("=" * 80 + "\n")
        f.write("SUMMARY: SIGNIFICANT INDIRECT EFFECTS\n")
        f.write("=" * 80 + "\n\n")
        
        sig_results = results[results["is_significant"]].sort_values("ab_effect", ascending=False)
        
        if sig_results.empty:
            f.write("No significant indirect effects found.\n\n")
        else:
            f.write(f"Found {len(sig_results)} significant indirect effects:\n\n")
            f.write(f"{'Thought (M)':<15} {'Mood (Y)':<15} {'Effect (a×b)':<15} {'95% CI':<30}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in sig_results.iterrows():
                ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                f.write(f"{row['thought_dim']:<15} {row['mood_scale']:<15} {row['ab_effect']:>12.6f}   {ci_str:<30}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Detailed results saved: {txt_path}")
    
    # Summary by mood outcome
    summary_path = os.path.join(RESULTS_DIR, "results_by_mood_outcome.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTS ORGANIZED BY MOOD OUTCOME\n")
        f.write("=" * 80 + "\n\n")
        
        for mood in MOOD_SCALES:
            mood_results = results[results["mood_scale"] == mood]
            if mood_results.empty:
                continue
                
            f.write("-" * 80 + "\n")
            f.write(f"MOOD OUTCOME: {mood}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"{'Thought Mediator':<15} {'Effect (a×b)':<15} {'95% CI':<30} {'Significant':<12}\n")
            f.write("-" * 80 + "\n")
            
            for _, row in mood_results.iterrows():
                ci_str = f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                sig_str = "YES" if row['is_significant'] else "NO"
                f.write(f"{row['thought_dim']:<15} {row['ab_effect']:>12.6f}   {ci_str:<30} {sig_str:<12}\n")
            
            f.write("\n")
    
    print(f"Results by mood outcome saved: {summary_path}")


def plot_reverse_covariates(results: pd.DataFrame):
    """
    Dashboard para los Cofactores del Modelo Reverso:
    1. Time -> Thoughts (Fatiga mental?)
    2. Time -> Mood Post (Empeoramiento por tiempo?)
    3. Mood Pre -> Mood Post (Inercia emocional)
    """
    print("\nGenerating Covariates Dashboard...")
    sns.set_theme(style="whitegrid", context="talk")
    COLOR_COV = "#7f8c8d"  # Gris Concreto (Neutro)
    
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
    
    # --- PANEL 1: Time -> Thoughts (Path A Control) ---
    ax1 = fig.add_subplot(gs[0])
    # Agrupamos por pensamiento (promedio de los modelos) y fijamos el orden
    df_time_thought = (
        results.groupby("thought_dim", as_index=True)[["time_on_thought", "time_on_thought_se"]]
        .mean(numeric_only=True)
        .reindex(THOUGHT_DIMENSIONS)
    )
    
    y_pos = np.arange(len(df_time_thought))
    for idx, (_, row) in enumerate(df_time_thought.iterrows()):
        val = row["time_on_thought"]
        err = row["time_on_thought_se"] * 1.96
        ax1.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2)
        ax1.plot(val, idx, 'o', color=COLOR_COV, markersize=8)
        
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_time_thought["thought_dim"], fontweight='bold')
    ax1.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax1.set_title("Effect of Time on Thoughts", fontweight='bold')
    ax1.set_xlabel("Beta (Time -> Thought)")
    ax1.invert_yaxis()

    # --- PANEL 2: Time -> Mood Post (Path B Control) ---
    ax2 = fig.add_subplot(gs[1])
    # Agrupamos por Mood y fijamos el orden según MOOD_SCALES
    df_time_mood = (
        results.groupby("mood_scale", as_index=True)[["time_on_mood", "time_on_mood_se"]]
        .mean(numeric_only=True)
        .reindex(MOOD_SCALES)
    )
    
    y_pos2 = np.arange(len(df_time_mood))
    for idx, (_, row) in enumerate(df_time_mood.iterrows()):
        val = row["time_on_mood"]
        err = row["time_on_mood_se"] * 1.96
        ax2.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2)
        ax2.plot(val, idx, 'o', color=COLOR_COV, markersize=8)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(df_time_mood["mood_scale"], fontweight='bold')
    ax2.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax2.set_title("Effect of Time on Mood Change", fontweight='bold')
    ax2.set_xlabel("Beta (Time -> Mood Post)")
    ax2.invert_yaxis()

    # --- PANEL 3: Mood Pre -> Mood Post (Autocorrelation) ---
    ax3 = fig.add_subplot(gs[2])
    df_base = (
        results.groupby("mood_scale", as_index=True)[["baseline_effect", "baseline_se"]]
        .mean(numeric_only=True)
        .reindex(MOOD_SCALES)
    )
    
    y_pos3 = np.arange(len(df_base))
    for idx, (_, row) in enumerate(df_base.iterrows()):
        val = row["baseline_effect"]
        err = row["baseline_se"] * 1.96
        # Este suele ser muy fuerte y positivo, quizás usar otro tono de gris más oscuro
        ax3.plot([val-err, val+err], [idx, idx], color="#34495e", lw=3)
        ax3.plot(val, idx, 'D', color="#34495e", markersize=8) # Diamante para baseline
        ax3.text(val, idx+0.1, f"{val:.2f}", ha='center', fontsize=9, color="#34495e")

    ax3.set_yticks(y_pos3)
    ax3.set_yticklabels(df_base["mood_scale"], fontweight='bold')
    ax3.set_title("Emotional Inertia\n(Pre -> Post)", fontweight='bold')
    ax3.set_xlabel("Beta (Autocorrelation)")
    ax3.invert_yaxis()
    
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "reverse_covariates_dashboard.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved covariates plot: {out_path}")
    plt.close()


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_combined_reverse_mediation_figure(results: pd.DataFrame):
    """
    Generate combined figure with:
    - Top: Forest plot (indirect effects)
    - Middle Left: Path A (Group -> Thoughts)
    - Middle Right: Path B Heatmap (Thoughts -> Mood Change)
    - Bottom Left: Time -> Thoughts (covariate)
    - Bottom Middle: Time -> Mood Post (covariate)
    - Bottom Right: Mood Pre -> Mood Post (covariate)
    """
    print("\nGenerating combined reverse mediation visualization...")
    
    # Setup Plot
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(24, 18))
    # Más espacio vertical para los paneles inferiores para evitar que se encimen
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.3, 1.0, 1.1],
        width_ratios=[1, 1, 0.8],
        hspace=0.45,
        wspace=0.3,
    )
    
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
    ax_forest.set_title("Reverse Mediation: Group → Thoughts → Mood Change (Indirect Effects)\n(Controlled for Baseline Mood + Time-on-Task)", 
                 pad=20, fontweight='bold', fontsize=13)
    ax_forest.invert_yaxis()
    
    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', label='Mood Outcomes:', linewidth=0)
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
    
    # ========== PANEL 2: PATH A (Group -> Thoughts) ==========
    ax_path_a = fig.add_subplot(gs[1, 0])
    
    # Path A: Efecto del Grupo sobre los Pensamientos (Mediador) - use standardized order
    # Reindex a todas las dimensiones y solo filtramos por las columnas necesarias para Path A
    path_a_raw = (
        results.drop_duplicates(subset=["thought_dim"])  # una fila por pensamiento
        .set_index("thought_dim")
        .reindex(THOUGHT_DIMENSIONS)
    )
    path_a_data = path_a_raw.reset_index().dropna(subset=["a_coef", "a_se"]).copy()
    
    # Check significance for Path A
    path_a_data["a_significant"] = np.abs(path_a_data["a_coef"] / path_a_data["a_se"]) > 1.96
    
    y_pos = np.arange(len(path_a_data))
    
    for idx, (_, row) in enumerate(path_a_data.iterrows()):
        val = row["a_coef"]
        err = row["a_se"] * 1.96
        is_sig = row["a_significant"]
        
        COLOR_RISK = "#F24236"
        COLOR_CONTROL = "#2E86AB"
        color = COLOR_RISK if val < 0 else COLOR_CONTROL
        
        # Styling based on significance
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        marker_face = color if is_sig else 'white'
        
        ax_path_a.plot([val - err, val + err], [idx, idx], 
                       color=color, lw=2, linestyle=linestyle, alpha=alpha)
        ax_path_a.plot(
            val,
            idx,
            'o',
            color=color,
            markersize=10,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=2,
            alpha=alpha,
        )
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
    ax_path_a.set_yticklabels(path_a_data["thought_dim"], fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coef (Group -> Thoughts)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Group → Thoughts", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.invert_yaxis()
    
    # ========== PANEL 3: PATH B HEATMAP (Thoughts -> Mood Change) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    # Pivot: Filas=Thoughts (Mediator), Cols=Moods (Outcome)
    # This matches the forward mediation layout - ensure consistent order
    pivot_b = results.pivot(index="thought_dim", columns="mood_scale", values="b_coef")
    pivot_b = pivot_b.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
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
    
    ax_path_b.set_title("Path B: Thoughts → Mood Change (Predictive Power)", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.set_ylabel("Thought Dimension", fontweight='bold', fontsize=11)
    ax_path_b.set_xlabel("Mood Outcome", fontweight='bold', fontsize=11)

    # ========== PANEL 4: TIME -> THOUGHTS (COVARIATE) ==========
    ax_time_thought = fig.add_subplot(gs[2, 0])
    COLOR_COV = "#7f8c8d"  # Gris Concreto (Neutro)
    
    # Agrupamos por pensamiento (promedio de los modelos) - ensure consistent order
    # Usamos promedio por thought_dim para evitar escoger filas con covariados NaN
    df_time_thought = (
        results.groupby("thought_dim", as_index=True)[["time_on_thought", "time_on_thought_se"]]
        .mean(numeric_only=True)
        .reindex(THOUGHT_DIMENSIONS)
    )
    
    y_pos = np.arange(len(df_time_thought))
    for idx, (_, row) in enumerate(df_time_thought.iterrows()):
        val = row["time_on_thought"]
        se_val = row["time_on_thought_se"]
        if pd.isna(val) or pd.isna(se_val):
            continue
        err = se_val * 1.96
        
        # Significance styling for covariates
        is_sig = abs(val) > 1.96 * row["time_on_thought_se"]
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        
        ax_time_thought.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2, alpha=alpha, linestyle=linestyle)
        ax_time_thought.plot(val, idx, 'o', color=COLOR_COV, markersize=8, alpha=alpha)
        
    ax_time_thought.set_yticks(y_pos)
    ax_time_thought.set_yticklabels(df_time_thought.index, fontweight='bold')
    ax_time_thought.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax_time_thought.set_title("Time → Thoughts (Fatigue)", fontweight='bold', pad=12, fontsize=12)
    ax_time_thought.set_xlabel("Beta (Time Effect)", fontweight='bold', fontsize=11)
    ax_time_thought.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_time_thought.invert_yaxis()

    # ========== PANEL 5: TIME -> MOOD POST (COVARIATE) ==========
    ax_time_mood = fig.add_subplot(gs[2, 1])
    
    # Agrupamos por Mood - ensure consistent order
    df_time_mood = (
        results.groupby("mood_scale", as_index=True)[["time_on_mood", "time_on_mood_se"]]
        .mean(numeric_only=True)
        .reindex(MOOD_SCALES)
    )
    
    y_pos2 = np.arange(len(df_time_mood))
    for idx, (_, row) in enumerate(df_time_mood.iterrows()):
        val = row["time_on_mood"]
        se_val = row["time_on_mood_se"]
        if pd.isna(val) or pd.isna(se_val):
            continue
        err = se_val * 1.96
        
        # Significance styling for covariates
        is_sig = abs(val) > 1.96 * row["time_on_mood_se"]
        alpha = 1.0 if is_sig else 0.4
        linestyle = '-' if is_sig else ':'
        
        ax_time_mood.plot([val-err, val+err], [idx, idx], color=COLOR_COV, lw=2, alpha=alpha, linestyle=linestyle)
        ax_time_mood.plot(val, idx, 'o', color=COLOR_COV, markersize=8, alpha=alpha)

    ax_time_mood.set_yticks(y_pos2)
    ax_time_mood.set_yticklabels(df_time_mood.index, fontweight='bold')
    ax_time_mood.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax_time_mood.set_title("Time → Mood Post (Fatigue)", fontweight='bold', pad=12, fontsize=12)
    ax_time_mood.set_xlabel("Beta (Time Effect)", fontweight='bold', fontsize=11)
    ax_time_mood.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_time_mood.invert_yaxis()

    # ========== PANEL 6: MOOD PRE -> MOOD POST (COVARIATE) ==========
    ax_baseline = fig.add_subplot(gs[2, 2])
    # Agrupamos por Mood para la inercia emocional (baseline) - un valor por escala
    df_base = (
        results.groupby("mood_scale", as_index=True)[["baseline_effect", "baseline_se"]]
        .mean(numeric_only=True)
        .reindex(MOOD_SCALES)
    )
    
    y_pos3 = np.arange(len(df_base))
    for idx, (_, row) in enumerate(df_base.iterrows()):
        val = row["baseline_effect"]
        se_val = row["baseline_se"]
        if pd.isna(val):
            continue
        # Si no tenemos SE, dibujamos sin barra de error
        if pd.isna(se_val):
            err = 0.0
        else:
            err = se_val * 1.96
        
        # Significance styling for covariates (solo si hay SE)
        if pd.isna(se_val):
            is_sig = False
        else:
            is_sig = abs(val) > 1.96 * se_val
        alpha = 1.0 if is_sig else 0.7
        linestyle = '-' if is_sig else ':'
        
        # Usar diamante para baseline y color más oscuro
        ax_baseline.plot(
            [val - err, val + err],
            [idx, idx],
            color="#34495e",
            lw=3,
            alpha=alpha,
            linestyle=linestyle,
        )
        ax_baseline.plot(val, idx, 'D', color="#34495e", markersize=8, alpha=alpha)
        ax_baseline.text(
            val,
            idx + 0.1,
            f"{val:.2f}",
            ha='center',
            fontsize=13,
            color="#34495e",
            alpha=alpha,
        )

    ax_baseline.set_yticks(y_pos3)
    ax_baseline.set_yticklabels(df_base.index, fontweight='bold')
    ax_baseline.axvline(0, color='k', linestyle=':', alpha=0.3)
    ax_baseline.set_title("Emotional Inertia\n(Pre → Post)", fontweight='bold', pad=12, fontsize=12)
    ax_baseline.set_xlabel("Beta (Autocorrelation)", fontweight='bold', fontsize=11)
    ax_baseline.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_baseline.invert_yaxis()

    # Save combined figure
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "reverse_mediation_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved: {out_path}")
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    ensure_directories()
    
    # 1. Load and preprocess
    df = load_and_preprocess_data()
    
    # 2. Check minimum sample size
    if len(df) < MIN_BLOCKS:
        print("Insufficient data for analysis.")
        return
        
    # 3. Run reverse mediation analysis
    results_df = run_reverse_mediation(df)
    
    # 4. Save results
    csv_path = os.path.join(RESULTS_DIR, "reverse_mediation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")
    
    # 5. Save detailed text reports
    save_detailed_results(results_df, df)
    
    # 6. Generate combined visualization (now includes covariates)
    plot_combined_reverse_mediation_figure(results_df)
    
    print("\n" + "=" * 70)
    print("REVERSE MEDIATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {RESULTS_DIR}")
    
    # Print summary
    sig_results = results_df[results_df["is_significant"]]
    print(f"\nSignificant indirect effects found: {len(sig_results)}")
    if not sig_results.empty:
        print("\nTop effects:")
        for _, row in sig_results.nlargest(5, "ab_effect").iterrows():
            print(f"  {row['thought_dim']} → {row['mood_scale']}: {row['ab_effect']:.4f} "
                  f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
