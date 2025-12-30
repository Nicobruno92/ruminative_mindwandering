#!/usr/bin/env python3
"""
Cyberball Moderated Mediation Analysis (Hayes Model 7/8)

Tests whether the Indirect Effect of Cyberball Condition on Thoughts via Mood
is conditional on the Group (Controls vs Risk of Depression).

Model:
    X: condition_bin (0=Inclusion, 1=Exclusion)
    Y: thought_valence (Outcome)
    M: mood_post (Mediator)
    W: group_bin (Moderator: 0=Control, 1=Risk)
    Cov: mood_pre (Baseline)

Hypothesis: The Risk Group shows stronger emotional reactivity to exclusion,
leading to a larger indirect effect (Condition → Mood → Thoughts) compared
to Controls.

Author: Senior Data Scientist - Computational Psychiatry
"""

from __future__ import annotations

import os
import warnings
from typing import List, Tuple, Optional, Dict

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
    "results/Behavior/probe_data/probe_level_aggregated_data.csv"
)
EVA_DATA_FILE: str = (
    "results/Behavior/scales_data/eva_aggregated_data.csv"
)

# Output directory
RESULTS_DIR: str = (
    "results/Behavior/mediation_analysis/cyberball_moderated_mediation"
)

# Thought dimensions to analyze (0-100 scale)
THOUGHT_DIMENSIONS: List[str] = [
    "valence",
    "time",
    "selfother",
    "onoff",
    "confidence",
]

# Mood scales from EVA
MOOD_SCALES: List[str] = [
    "EVAaverage",
    "EVAmood",
    "EVAfeel",
    "EVAtense",
    "EVAhurt",
]

# Monte Carlo simulation parameters
N_SIMULATIONS: int = 20000
RANDOM_SEED: int = 42
CONFIDENCE_LEVEL: float = 0.95

# Group coding
CONTROL_GROUP: str = "Controls"
RISK_GROUP: str = "Risk of Depression"

# Cyberball conditions to analyze (exclude baseline)
# Sart2 and Sart4 are the manipulation blocks
CYBERBALL_TASKS: List[str] = ["Sart2", "Sart4"]

# Filtering
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


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load probe and EVA data, merge to create block-level dataset for
    Cyberball moderated mediation analysis.
    
    Returns
    -------
    pd.DataFrame
        Block-level data with:
        - condition_bin: 0=Inclusion, 1=Exclusion
        - group_bin: 0=Controls, 1=Risk
        - mood_pre: Mood at start of task (from baseline block)
        - mood_post: Mood at end of task
        - thought dimensions: Mean per block
    """
    print("\n" + "=" * 70)
    print("DATA LOADING & PREPROCESSING - CYBERBALL MODERATED MEDIATION")
    print("=" * 70)

    # 1. Load Probe Data
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe file not found: {PROBE_DATA_FILE}")
    df_probes = pd.read_csv(PROBE_DATA_FILE)
    print(f"Loaded {len(df_probes)} probe trials.")

    # Filter to Cyberball manipulation blocks only (Sart2 and Sart4)
    df_probes = df_probes[df_probes["task"].isin(CYBERBALL_TASKS)].copy()
    print(f"Filtered to Cyberball blocks (Sart2, Sart4): {len(df_probes)} trials")

    # Filter ON-task if requested
    if APPLY_ONOFF_FILTER and "onoff" in df_probes.columns:
        n_orig = len(df_probes)
        df_probes = df_probes[df_probes["onoff"] < ONOFF_MAX_EXCLUSIVE].copy()
        print(f"Applied ON-task filter (onoff < {ONOFF_MAX_EXCLUSIVE}): {n_orig} -> {len(df_probes)} trials")

    # 2. Aggregate Probes to Block Level
    print("Aggregating probes to block level...")
    agg_cols = [d for d in THOUGHT_DIMENSIONS if d in df_probes.columns]
    
    df_blocks = (
        df_probes.groupby(["subject_id", "task", "group", "inclusion_exclusion"], as_index=False)[agg_cols]
        .mean()
    )
    print(f"Block-level aggregation: {len(df_blocks)} blocks")

    # 3. Load EVA Data (Mood)
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA file not found: {EVA_DATA_FILE}")
    df_eva = pd.read_csv(EVA_DATA_FILE)
    print(f"Loaded {len(df_eva)} EVA entries.")

    # 4. Extract Mood Pre and Post per block
    # mood_pre: From the baseline block before the manipulation (Sart1 for Sart2, Sart3 for Sart4)
    # mood_post: From the manipulation block itself
    print("Extracting mood_pre (baseline) and mood_post (manipulation block)...")
    
    # Get first EVA measurement per task as mood for that task
    df_mood_task = df_eva.sort_values(["subject_id", "task", "block_number"]).groupby(
        ["subject_id", "task"], as_index=False
    ).first()
    
    # Create mapping: Sart2 uses Sart1 as baseline, Sart4 uses Sart3 as baseline
    baseline_map = {"Sart2": "Sart1", "Sart4": "Sart3"}
    
    mood_data_list = []
    for _, row in df_blocks.iterrows():
        subj = row["subject_id"]
        task = row["task"]
        baseline_task = baseline_map.get(task)
        
        # Get mood_post from current task
        mood_post_row = df_mood_task[
            (df_mood_task["subject_id"] == subj) & 
            (df_mood_task["task"] == task)
        ]
        
        # Get mood_pre from baseline task
        mood_pre_row = df_mood_task[
            (df_mood_task["subject_id"] == subj) & 
            (df_mood_task["task"] == baseline_task)
        ]
        
        mood_entry = {"subject_id": subj, "task": task}
        
        for mood in MOOD_SCALES:
            mood_entry[f"{mood}_post"] = (
                float(mood_post_row[mood].iloc[0]) if not mood_post_row.empty and mood in mood_post_row.columns 
                else np.nan
            )
            mood_entry[f"{mood}_pre"] = (
                float(mood_pre_row[mood].iloc[0]) if not mood_pre_row.empty and mood in mood_pre_row.columns 
                else np.nan
            )
        
        mood_data_list.append(mood_entry)
    
    df_mood = pd.DataFrame(mood_data_list)

    # 5. Merge Data
    df_merged = df_blocks.merge(df_mood, on=["subject_id", "task"], how="inner")

    # 6. Create Binary Variables
    # condition_bin: 0=Inclusion, 1=Exclusion
    df_merged["condition_bin"] = (df_merged["inclusion_exclusion"] == "exclusion").astype(int)
    
    # group_bin: 0=Controls, 1=Risk
    df_merged["group_bin"] = (df_merged["group"] == RISK_GROUP).astype(int)

    # 7. Drop missing values
    required_cols = [f"{m}_pre" for m in MOOD_SCALES] + [f"{m}_post" for m in MOOD_SCALES]
    required_cols += ["condition_bin", "group_bin"]
    df_merged = df_merged.dropna(subset=required_cols)
    
    print(f"\nFinal Dataset: {len(df_merged)} blocks, {df_merged['subject_id'].nunique()} subjects.")
    print(f"  Controls: {(df_merged['group_bin'] == 0).sum()} blocks")
    print(f"  Risk: {(df_merged['group_bin'] == 1).sum()} blocks")
    print(f"  Inclusion: {(df_merged['condition_bin'] == 0).sum()} blocks")
    print(f"  Exclusion: {(df_merged['condition_bin'] == 1).sum()} blocks")
    
    return df_merged


# =============================================================================
# STATISTICAL ANALYSIS - MODERATED MEDIATION
# =============================================================================

def fit_lmm(formula: str, data: pd.DataFrame, group_col: str = "subject_id") -> Tuple[Optional[object], bool]:
    """Fit LMM and return result object and convergence status."""
    try:
        model = smf.mixedlm(formula, data, groups=data[group_col])
        result = model.fit(method="lbfgs", maxiter=2000, disp=False)
        return result, True
    except Exception as e:
        warnings.warn(f"Model failed: {e}")
        return None, False


def monte_carlo_moderated_mediation(
    a_base: float, a_base_se: float,
    a_int: float, a_int_se: float,
    b: float, b_se: float,
    cov_a_base_a_int: float = 0.0,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED
) -> Dict[str, Tuple[float, float, float, bool]]:
    """
    Monte Carlo simulation for moderated mediation (Hayes Model 7).
    
    Parameters
    ----------
    a_base : float
        Path a coefficient for condition (effect on Controls)
    a_base_se : float
        Standard error for a_base
    a_int : float
        Interaction coefficient (condition × group)
    a_int_se : float
        Standard error for a_int
    b : float
        Path b coefficient (mood → thought)
    b_se : float
        Standard error for b
    cov_a_base_a_int : float
        Covariance between a_base and a_int (default 0)
    n_sims : int
        Number of Monte Carlo draws
    seed : int
        Random seed
    
    Returns
    -------
    dict
        Results for Controls, Risk, and Index of Moderated Mediation
    """
    np.random.seed(seed)
    
    # Draw from multivariate normal for a_base and a_int
    mean_a = [a_base, a_int]
    cov_a = [[a_base_se**2, cov_a_base_a_int], 
             [cov_a_base_a_int, a_int_se**2]]
    a_samples = np.random.multivariate_normal(mean_a, cov_a, n_sims)
    a_base_dist = a_samples[:, 0]
    a_int_dist = a_samples[:, 1]
    
    # Draw b independently
    b_dist = np.random.normal(b, b_se, n_sims)
    
    # Calculate indirect effects
    # For Controls (W=0): IE = a_base × b
    ie_controls = a_base_dist * b_dist
    
    # For Risk (W=1): IE = (a_base + a_int) × b
    ie_risk = (a_base_dist + a_int_dist) * b_dist
    
    # Index of Moderated Mediation: a_int × b
    index_mm = a_int_dist * b_dist
    
    alpha = 1 - CONFIDENCE_LEVEL
    
    def summarize(dist: np.ndarray) -> Tuple[float, float, float, bool]:
        mean_val = float(np.mean(dist))
        ci_low = float(np.percentile(dist, 100 * alpha / 2))
        ci_high = float(np.percentile(dist, 100 * (1 - alpha / 2)))
        is_sig = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        return mean_val, ci_low, ci_high, is_sig
    
    return {
        "controls": summarize(ie_controls),
        "risk": summarize(ie_risk),
        "index_mm": summarize(index_mm),
    }


def run_moderated_mediation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run moderated mediation analysis: Condition → Mood → Thoughts (moderated by Group).
    
    For each combination of Thought Dimension (Outcome) and Mood Scale (Mediator):
    - Mediator Model: mood_post ~ condition_bin * group_bin + mood_pre
    - Outcome Model: thought ~ mood_post + condition_bin * group_bin + mood_pre
    """
    print("\n" + "=" * 70)
    print("RUNNING MODERATED MEDIATION MODELS (Hayes Model 7)")
    print("=" * 70)
    print(f"Testing: Condition → Mood → Thoughts (Group moderates Path a)")
    print(f"Controlling for: Baseline Mood (mood_pre)")
    
    results = []
    total_models = len(THOUGHT_DIMENSIONS) * len(MOOD_SCALES)
    count = 0
    
    for mood in MOOD_SCALES:
        mood_pre = f"{mood}_pre"
        mood_post = f"{mood}_post"
        
        if mood_pre not in data.columns or mood_post not in data.columns:
            print(f"Skipping {mood}: columns not found")
            continue
        
        # --- Mediator Model: Path a is moderated ---
        # mood_post ~ condition_bin * group_bin + mood_pre
        formula_m = f"{mood_post} ~ condition_bin * group_bin + {mood_pre}"
        res_m, conv_m = fit_lmm(formula_m, data)
        
        if not conv_m:
            print(f"Skipping {mood}: Mediator model failed to converge.")
            continue
        
        # Extract coefficients
        a_base = res_m.params["condition_bin"]  # Effect of exclusion on Controls
        a_base_se = res_m.bse["condition_bin"]
        
        a_int = res_m.params["condition_bin:group_bin"]  # Interaction: extra effect for Risk
        a_int_se = res_m.bse["condition_bin:group_bin"]
        
        # Get covariance if available
        try:
            cov_matrix = res_m.cov_params()
            cov_a_base_a_int = cov_matrix.loc["condition_bin", "condition_bin:group_bin"]
        except Exception:
            cov_a_base_a_int = 0.0
        
        for thought in THOUGHT_DIMENSIONS:
            count += 1
            
            if thought not in data.columns:
                continue
            
            print(f"[{count}/{total_models}] Mediator: {mood} -> Outcome: {thought} ...", end="\r")
            
            # --- Outcome Model: Path b and c' ---
            # thought ~ mood_post + condition_bin * group_bin + mood_pre
            formula_y = f"{thought} ~ {mood_post} + condition_bin * group_bin + {mood_pre}"
            res_y, conv_y = fit_lmm(formula_y, data)
            
            if not conv_y:
                continue
            
            b = res_y.params[mood_post]  # Path b: mood → thought
            b_se = res_y.bse[mood_post]
            
            c_base = res_y.params["condition_bin"]  # Direct effect on Controls
            c_base_se = res_y.bse["condition_bin"]
            
            c_int = res_y.params["condition_bin:group_bin"]  # Direct interaction
            c_int_se = res_y.bse["condition_bin:group_bin"]
            
            # Monte Carlo simulation for indirect effects
            mc_results = monte_carlo_moderated_mediation(
                a_base, a_base_se,
                a_int, a_int_se,
                b, b_se,
                cov_a_base_a_int,
                N_SIMULATIONS,
                RANDOM_SEED
            )
            
            results.append({
                "mood_scale": mood,
                "thought_dim": thought,
                # Path a (moderated)
                "a_base": a_base,
                "a_base_se": a_base_se,
                "a_int": a_int,
                "a_int_se": a_int_se,
                # Path b
                "b": b,
                "b_se": b_se,
                # Direct effects
                "c_base": c_base,
                "c_base_se": c_base_se,
                "c_int": c_int,
                "c_int_se": c_int_se,
                # Indirect effect for Controls
                "ie_controls": mc_results["controls"][0],
                "ie_controls_ci_low": mc_results["controls"][1],
                "ie_controls_ci_high": mc_results["controls"][2],
                "ie_controls_sig": mc_results["controls"][3],
                # Indirect effect for Risk
                "ie_risk": mc_results["risk"][0],
                "ie_risk_ci_low": mc_results["risk"][1],
                "ie_risk_ci_high": mc_results["risk"][2],
                "ie_risk_sig": mc_results["risk"][3],
                # Index of Moderated Mediation
                "index_mm": mc_results["index_mm"][0],
                "index_mm_ci_low": mc_results["index_mm"][1],
                "index_mm_ci_high": mc_results["index_mm"][2],
                "index_mm_sig": mc_results["index_mm"][3],
            })
    
    print(f"\nCompleted {len(results)} successful moderated mediation models.")
    return pd.DataFrame(results)


def run_path_a_delta_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    """Quick diagnostic for Path A using mood deltas (post - pre).

    Tests the interaction ``condition_bin * group_bin`` on delta mood for each
    EVA scale using a simple LMM with random intercepts by subject.
    """
    print("\n" + "=" * 70)
    print("QUICK DIAGNOSTIC: PATH A USING DELTA MOOD")
    print("=" * 70)

    df = data.copy()

    # 1) Compute delta mood for each EVA scale
    for mood in MOOD_SCALES:
        post_col = f"{mood}_post"
        pre_col = f"{mood}_pre"
        if post_col not in df.columns or pre_col not in df.columns:
            continue
        df[f"{mood}_delta"] = df[post_col] - df[pre_col]

    # 2) Test interaction on delta mood
    print("TESTING PATH A: Does Exclusion cause a bigger MOOD DROP in Risk Group?")
    results: list[dict] = []

    for mood in MOOD_SCALES:
        delta_col = f"{mood}_delta"
        if delta_col not in df.columns:
            continue

        formula = f"{delta_col} ~ condition_bin * group_bin"
        res, conv = fit_lmm(formula, df)
        if not conv:
            continue

        # Interaction term: additional effect of exclusion in Risk vs Controls
        beta_int = res.params.get("condition_bin:group_bin", np.nan)
        pval_int = res.pvalues.get("condition_bin:group_bin", np.nan)

        sig_flag = (pd.notna(pval_int) and pval_int < 0.05)
        sig_str = "**" if sig_flag else ""
        if pd.notna(beta_int) and pd.notna(pval_int):
            print(f"{mood}: Beta_Interaction = {beta_int:.3f}, p = {pval_int:.3f} {sig_str}")
        else:
            print(f"{mood}: interaction term not estimable")

        results.append(
            {
                "mood_scale": mood,
                "beta_interaction": float(beta_int) if pd.notna(beta_int) else np.nan,
                "pval_interaction": float(pval_int) if pd.notna(pval_int) else np.nan,
                "significant_0.05": bool(sig_flag),
            }
        )

    # 3) Simple diagnostic plot for EVAaverage delta
    if "EVAaverage_delta" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.pointplot(
            data=df,
            x="condition_bin",
            y="EVAaverage_delta",
            hue="group_bin",
            dodge=True,
            ci=95,
        )
        plt.title("Change in EVAaverage Mood (Post - Pre) by Condition and Group")
        plt.ylabel("Mood Change (Delta)")
        plt.xlabel("Cyberball Condition")
        plt.xticks([0, 1], ["Inclusion", "Exclusion"])
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "plots", "path_a_delta_EVAaverage.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Path A delta diagnostic plot saved: {out_path}")
        plt.close()

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_moderated_mediation_figure(results: pd.DataFrame) -> None:
    """
    Generate combined figure with:
    - Top: Forest plot of Indirect Effects by Group
    - Middle Left: Path A (Condition → Mood, moderated by Group)
    - Middle Right: Path B Heatmap (Mood → Thoughts)
    - Bottom: Index of Moderated Mediation
    """
    print("\nGenerating moderated mediation visualization...")
    
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(
        4, 2,
        height_ratios=[1.3, 1.0, 1.0, 0.8],
        width_ratios=[1, 1],
        hspace=0.4,
        wspace=0.3,
    )
    
    # Color scheme
    COLOR_CONTROL = "#2E86AB"
    COLOR_RISK = "#F24236"
    
    # ========== PANEL 1: FOREST PLOT - INDIRECT EFFECTS BY GROUP ==========
    ax_forest = fig.add_subplot(gs[0, :])
    
    # Create color palette for moods
    palette = sns.color_palette("husl", n_colors=len(MOOD_SCALES))
    mood_colors = dict(zip(MOOD_SCALES, palette))
    
    # Use consistent order for dimensions
    dims = THOUGHT_DIMENSIONS
    y_positions = np.arange(len(dims))
    
    # Add background stripes
    for y in y_positions:
        if y % 2 == 0:
            ax_forest.axhspan(y - 0.5, y + 0.5, color='gray', alpha=0.1, zorder=0, linewidth=0)
    
    # Plot indirect effects for both groups
    bar_height = 0.35
    
    for i, mood in enumerate(MOOD_SCALES):
        subset = results[results["mood_scale"] == mood]
        subset_indexed = subset.set_index("thought_dim").reindex(dims)
        
        for j, dim in enumerate(dims):
            if dim not in subset_indexed.index:
                continue
            
            row = subset_indexed.loc[dim]
            if pd.isna(row["ie_controls"]):
                continue
            
            # Controls (left offset)
            y_ctrl = j - bar_height/2
            ie_ctrl = row["ie_controls"]
            ci_low_ctrl = row["ie_controls_ci_low"]
            ci_high_ctrl = row["ie_controls_ci_high"]
            sig_ctrl = row["ie_controls_sig"]
            
            alpha_ctrl = 1.0 if sig_ctrl else 0.4
            linestyle_ctrl = '-' if sig_ctrl else ':'
            
            ax_forest.plot([ci_low_ctrl, ci_high_ctrl], [y_ctrl, y_ctrl],
                          color=COLOR_CONTROL, alpha=alpha_ctrl, linestyle=linestyle_ctrl, linewidth=2, zorder=2)
            ax_forest.plot(ie_ctrl, y_ctrl, marker='o', markersize=8,
                          markeredgecolor=COLOR_CONTROL, 
                          markerfacecolor=COLOR_CONTROL if sig_ctrl else 'white',
                          markeredgewidth=2, alpha=alpha_ctrl, zorder=3)
            
            # Risk (right offset)
            y_risk = j + bar_height/2
            ie_risk = row["ie_risk"]
            ci_low_risk = row["ie_risk_ci_low"]
            ci_high_risk = row["ie_risk_ci_high"]
            sig_risk = row["ie_risk_sig"]
            
            alpha_risk = 1.0 if sig_risk else 0.4
            linestyle_risk = '-' if sig_risk else ':'
            
            ax_forest.plot([ci_low_risk, ci_high_risk], [y_risk, y_risk],
                          color=COLOR_RISK, alpha=alpha_risk, linestyle=linestyle_risk, linewidth=2, zorder=2)
            ax_forest.plot(ie_risk, y_risk, marker='s', markersize=8,
                          markeredgecolor=COLOR_RISK,
                          markerfacecolor=COLOR_RISK if sig_risk else 'white',
                          markeredgewidth=2, alpha=alpha_risk, zorder=3)
    
    ax_forest.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, zorder=1)
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels(dims, fontweight='bold', fontsize=12)
    ax_forest.set_xlabel("Indirect Effect (a × b)", fontweight='bold', fontsize=13)
    ax_forest.set_title(
        "Moderated Mediation: Cyberball Condition → Mood → Thoughts\n"
        "(Indirect Effects by Group, Averaged Across Mood Scales)",
        pad=20, fontweight='bold', fontsize=14
    )
    ax_forest.invert_yaxis()
    ax_forest.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=COLOR_CONTROL, label='Controls',
               markerfacecolor=COLOR_CONTROL, markersize=10, linestyle='-'),
        Line2D([0], [0], marker='s', color=COLOR_RISK, label='Risk of Depression',
               markerfacecolor=COLOR_RISK, markersize=10, linestyle='-'),
        Line2D([0], [0], color='gray', label='Significant', linestyle='-', linewidth=2),
        Line2D([0], [0], color='gray', label='Non-significant', linestyle=':', linewidth=2, alpha=0.5),
    ]
    ax_forest.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10)
    
    # ========== PANEL 2: PATH A - CONDITION → MOOD (MODERATED) ==========
    ax_path_a = fig.add_subplot(gs[1, 0])
    
    # Get unique path a values per mood scale
    path_a_data = results.drop_duplicates(subset=["mood_scale"]).set_index("mood_scale").reindex(MOOD_SCALES)
    
    y_pos = np.arange(len(MOOD_SCALES))
    
    for idx, mood in enumerate(MOOD_SCALES):
        if mood not in path_a_data.index:
            continue
        row = path_a_data.loc[mood]
        
        # a_base (Controls)
        a_base = row["a_base"]
        a_base_se = row["a_base_se"]
        sig_base = abs(a_base / a_base_se) > 1.96
        
        ax_path_a.plot([a_base - 1.96*a_base_se, a_base + 1.96*a_base_se], [idx - 0.15, idx - 0.15],
                      color=COLOR_CONTROL, lw=2, alpha=1.0 if sig_base else 0.4,
                      linestyle='-' if sig_base else ':')
        ax_path_a.plot(a_base, idx - 0.15, 'o', color=COLOR_CONTROL, markersize=10,
                      markerfacecolor=COLOR_CONTROL if sig_base else 'white',
                      markeredgewidth=2, alpha=1.0 if sig_base else 0.4)
        
        # a_base + a_int (Risk)
        a_risk = a_base + row["a_int"]
        a_int_se = row["a_int_se"]
        # Combined SE (approximate)
        a_risk_se = np.sqrt(a_base_se**2 + a_int_se**2)
        sig_risk = abs(a_risk / a_risk_se) > 1.96
        
        ax_path_a.plot([a_risk - 1.96*a_risk_se, a_risk + 1.96*a_risk_se], [idx + 0.15, idx + 0.15],
                      color=COLOR_RISK, lw=2, alpha=1.0 if sig_risk else 0.4,
                      linestyle='-' if sig_risk else ':')
        ax_path_a.plot(a_risk, idx + 0.15, 's', color=COLOR_RISK, markersize=10,
                      markerfacecolor=COLOR_RISK if sig_risk else 'white',
                      markeredgewidth=2, alpha=1.0 if sig_risk else 0.4)
    
    ax_path_a.set_yticks(y_pos)
    ax_path_a.set_yticklabels(MOOD_SCALES, fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coefficient (Exclusion Effect on Mood)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Condition → Mood\n(Moderated by Group)", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.invert_yaxis()
    
    # ========== PANEL 3: PATH B HEATMAP (MOOD → THOUGHTS) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    pivot_b = results.pivot(index="thought_dim", columns="mood_scale", values="b")
    pivot_b = pivot_b.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
    # Significance mask
    pivot_b_se = results.pivot(index="thought_dim", columns="mood_scale", values="b_se")
    pivot_b_se = pivot_b_se.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    pivot_sig = (np.abs(pivot_b / pivot_b_se) > 1.96)
    
    # Color only significant cells
    colored_data = pivot_b.copy()
    colored_data[~pivot_sig] = np.nan
    
    sns.heatmap(colored_data, cmap="RdBu_r", center=0,
                ax=ax_path_b, cbar_kws={'label': 'Beta Coefficient'},
                linewidths=1, linecolor='white')
    
    # Add annotations
    for i in range(pivot_b.shape[0]):
        for j in range(pivot_b.shape[1]):
            val = pivot_b.iloc[i, j]
            if not pd.isna(val):
                text = f"{val:.2f}"
                if pivot_sig.iloc[i, j]:
                    text += "*"
                text_color = 'white' if pivot_sig.iloc[i, j] else 'black'
                ax_path_b.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                              color=text_color, fontsize=12, fontweight='bold')
    
    ax_path_b.set_title("Path B: Mood → Thoughts", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.set_ylabel("Thought Dimension", fontweight='bold', fontsize=11)
    ax_path_b.set_xlabel("Mood Scale", fontweight='bold', fontsize=11)
    
    # ========== PANEL 4: INDEX OF MODERATED MEDIATION ==========
    ax_index = fig.add_subplot(gs[2, :])
    
    # Pivot for heatmap
    pivot_index = results.pivot(index="thought_dim", columns="mood_scale", values="index_mm")
    pivot_index = pivot_index.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
    pivot_index_sig = results.pivot(index="thought_dim", columns="mood_scale", values="index_mm_sig")
    pivot_index_sig = pivot_index_sig.reindex(index=THOUGHT_DIMENSIONS, columns=MOOD_SCALES)
    
    # Color only significant cells
    colored_index = pivot_index.copy()
    colored_index[~pivot_index_sig] = np.nan
    
    sns.heatmap(colored_index, cmap="PuOr", center=0,
                ax=ax_index, cbar_kws={'label': 'Index of Moderated Mediation'},
                linewidths=1, linecolor='white')
    
    # Add annotations
    for i in range(pivot_index.shape[0]):
        for j in range(pivot_index.shape[1]):
            val = pivot_index.iloc[i, j]
            if not pd.isna(val):
                text = f"{val:.2f}"
                if pivot_index_sig.iloc[i, j]:
                    text += "*"
                text_color = 'white' if pivot_index_sig.iloc[i, j] else 'black'
                ax_index.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                             color=text_color, fontsize=12, fontweight='bold')
    
    ax_index.set_title(
        "Index of Moderated Mediation (a_int × b)\n"
        "Positive = Risk Group has STRONGER indirect effect than Controls",
        fontweight='bold', pad=12, fontsize=12
    )
    ax_index.set_ylabel("Thought Dimension", fontweight='bold', fontsize=11)
    ax_index.set_xlabel("Mood Scale", fontweight='bold', fontsize=11)
    
    # ========== PANEL 5: SUMMARY BAR CHART ==========
    ax_summary = fig.add_subplot(gs[3, :])
    
    # Average across mood scales for each thought dimension
    summary_data = results.groupby("thought_dim").agg({
        "ie_controls": "mean",
        "ie_risk": "mean",
        "index_mm": "mean",
    }).reindex(THOUGHT_DIMENSIONS)
    
    x = np.arange(len(THOUGHT_DIMENSIONS))
    width = 0.25
    
    bars1 = ax_summary.bar(x - width, summary_data["ie_controls"], width, 
                           label='IE Controls', color=COLOR_CONTROL, alpha=0.8)
    bars2 = ax_summary.bar(x, summary_data["ie_risk"], width,
                           label='IE Risk', color=COLOR_RISK, alpha=0.8)
    bars3 = ax_summary.bar(x + width, summary_data["index_mm"], width,
                           label='Index MM', color='#9B59B6', alpha=0.8)
    
    ax_summary.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(THOUGHT_DIMENSIONS, fontweight='bold', fontsize=11)
    ax_summary.set_ylabel("Effect Size", fontweight='bold', fontsize=11)
    ax_summary.set_title("Summary: Average Effects Across Mood Scales", fontweight='bold', pad=12, fontsize=12)
    ax_summary.legend(loc='best', fontsize=10)
    ax_summary.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Save
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plots", "cyberball_moderated_mediation_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved: {out_path}")
    plt.close()


def save_detailed_results(results: pd.DataFrame, data: pd.DataFrame) -> None:
    """Save detailed model results to text files."""
    print("\nSaving detailed model results...")
    
    txt_path = os.path.join(RESULTS_DIR, "detailed_model_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CYBERBALL MODERATED MEDIATION ANALYSIS - DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hypothesis: The Risk Group shows stronger emotional reactivity to exclusion,\n")
        f.write("leading to a larger indirect effect (Condition → Mood → Thoughts) compared\n")
        f.write("to Controls.\n\n")
        
        f.write("Model: Hayes Model 7 (Moderated Mediation)\n")
        f.write("  X: Cyberball Condition (0=Inclusion, 1=Exclusion)\n")
        f.write("  M: Mood (EVA scales)\n")
        f.write("  Y: Thought Dimensions\n")
        f.write("  W: Group (0=Controls, 1=Risk)\n")
        f.write("  Cov: Baseline Mood (mood_pre)\n\n")
        
        f.write(f"Monte Carlo Simulations: {N_SIMULATIONS:,}\n")
        f.write(f"Confidence Level: {CONFIDENCE_LEVEL * 100:.0f}%\n\n")
        
        # Dataset summary
        f.write("-" * 80 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Blocks: {len(data)}\n")
        f.write(f"Total Subjects: {data['subject_id'].nunique()}\n")
        f.write(f"Controls: {(data['group_bin'] == 0).sum()} blocks\n")
        f.write(f"Risk Group: {(data['group_bin'] == 1).sum()} blocks\n")
        f.write(f"Inclusion: {(data['condition_bin'] == 0).sum()} blocks\n")
        f.write(f"Exclusion: {(data['condition_bin'] == 1).sum()} blocks\n\n")
        
        # Summary of significant effects
        f.write("=" * 80 + "\n")
        f.write("SUMMARY: SIGNIFICANT EFFECTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Significant Index of Moderated Mediation
        sig_index = results[results["index_mm_sig"]].sort_values("index_mm", ascending=False)
        f.write("Significant Index of Moderated Mediation (Group difference in indirect effect):\n")
        if sig_index.empty:
            f.write("  None\n\n")
        else:
            for _, row in sig_index.iterrows():
                f.write(f"  {row['mood_scale']} → {row['thought_dim']}: "
                       f"Index = {row['index_mm']:.4f} "
                       f"[{row['index_mm_ci_low']:.4f}, {row['index_mm_ci_high']:.4f}]\n")
            f.write("\n")
        
        # Significant indirect effects for Risk group
        sig_risk = results[results["ie_risk_sig"]].sort_values("ie_risk", ascending=False)
        f.write("Significant Indirect Effects for Risk Group:\n")
        if sig_risk.empty:
            f.write("  None\n\n")
        else:
            for _, row in sig_risk.iterrows():
                f.write(f"  {row['mood_scale']} → {row['thought_dim']}: "
                       f"IE = {row['ie_risk']:.4f} "
                       f"[{row['ie_risk_ci_low']:.4f}, {row['ie_risk_ci_high']:.4f}]\n")
            f.write("\n")
        
        # Significant indirect effects for Controls
        sig_ctrl = results[results["ie_controls_sig"]].sort_values("ie_controls", ascending=False)
        f.write("Significant Indirect Effects for Controls:\n")
        if sig_ctrl.empty:
            f.write("  None\n\n")
        else:
            for _, row in sig_ctrl.iterrows():
                f.write(f"  {row['mood_scale']} → {row['thought_dim']}: "
                       f"IE = {row['ie_controls']:.4f} "
                       f"[{row['ie_controls_ci_low']:.4f}, {row['ie_controls_ci_high']:.4f}]\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Detailed results saved: {txt_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    ensure_directories()
    
    # 1. Load and preprocess
    df = load_and_preprocess_data()
    
    # 2. Quick diagnostic: Path A using delta mood (post - pre)
    path_a_delta_df = run_path_a_delta_diagnostic(df)
    diag_csv = os.path.join(RESULTS_DIR, "path_a_delta_results.csv")
    path_a_delta_df.to_csv(diag_csv, index=False)
    print(f"Path A delta diagnostic results saved: {diag_csv}")
    
    # 3. Run moderated mediation analysis
    results_df = run_moderated_mediation(df)
    
    # 4. Save results
    csv_path = os.path.join(RESULTS_DIR, "cyberball_moderated_mediation_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")
    
    # 5. Save detailed text reports
    save_detailed_results(results_df, df)
    
    # 6. Generate visualization
    plot_moderated_mediation_figure(results_df)
    
    print("\n" + "=" * 70)
    print("CYBERBALL MODERATED MEDIATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults directory: {RESULTS_DIR}")
    
    # Print summary
    sig_index = results_df[results_df["index_mm_sig"]]
    print(f"\nSignificant Index of Moderated Mediation: {len(sig_index)}")
    if not sig_index.empty:
        print("\nTop effects (Group difference in indirect effect):")
        for _, row in sig_index.nlargest(5, "index_mm").iterrows():
            print(f"  {row['mood_scale']} → {row['thought_dim']}: "
                  f"Index = {row['index_mm']:.4f} "
                  f"[{row['index_mm_ci_low']:.4f}, {row['index_mm_ci_high']:.4f}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
