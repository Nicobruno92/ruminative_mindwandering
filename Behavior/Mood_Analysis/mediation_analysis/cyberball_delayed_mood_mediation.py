#!/usr/bin/env python3
"""
Cyberball Delayed Mood Mediation Analysis (Hayes Model 7/8)

Tests whether the Indirect Effect of Cyberball Condition on Delayed Mood 
(measured after SART) via Thoughts (during SART) is conditional on Group.

Model:
    X: condition_bin (0=Inclusion, 1=Exclusion)
    M: thoughts (Mediator - during SART)
    Y: mood_delayed (Outcome - measured after completing SART)
    W: group_bin (Moderator: 0=Control, 1=Risk)
    Cov: mood_pre (Baseline mood before Cyberball)

Hypothesis: Cyberball manipulation affects thoughts during SART, which in turn
influence delayed mood. The Risk Group may show stronger thought-driven mood
changes compared to Controls.

Procedure Reference:
    Sart1 → Cyberball (Inc/Exc) → Sart2 → [mood_delayed measured here]
    Sart3 → Cyberball (Exc/Inc) → Sart4 → [mood_delayed measured here]

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    "results/Behavior/mediation_analysis/cyberball_delayed_mood_mediation"
)

# Thought dimensions to analyze (Mediator - 0-100 scale)
# PC1 is computed from valence, selfother, time via PCA
THOUGHT_DIMENSIONS: List[str] = [
    "PC1",
    "valence",
    "time",
    "selfother",
    "onoff",
    "confidence",
]

# Variables used for PCA computation
PCA_VARIABLES: List[str] = ["valence", "selfother", "time"]

# Mood scales from EVA (Outcome)
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

# Cyberball conditions to analyze (post-manipulation blocks)
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


def compute_pc1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PC1 from valence, selfother, time using PCA.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with valence, selfother, time columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with PC1 column added
    """
    df = df.copy()
    
    # Check if PC1 already exists
    if "PC1" in df.columns:
        print("PC1 already exists in data, using existing values.")
        return df
    
    # Check required columns
    missing = [v for v in PCA_VARIABLES if v not in df.columns]
    if missing:
        print(f"Cannot compute PC1: missing columns {missing}")
        return df
    
    # Get rows with valid PCA data
    valid_mask = df[PCA_VARIABLES].notna().all(axis=1)
    pca_data = df.loc[valid_mask, PCA_VARIABLES]
    
    if len(pca_data) < 3:
        print("Insufficient data to compute PC1")
        return df
    
    # Standardize and compute PCA
    scaler = StandardScaler()
    pca_scaled = scaler.fit_transform(pca_data)
    
    pca = PCA(n_components=1)
    pc1_values = pca.fit_transform(pca_scaled).flatten()
    
    # Add PC1 to dataframe
    df["PC1"] = np.nan
    df.loc[valid_mask, "PC1"] = pc1_values
    
    explained_var = pca.explained_variance_ratio_[0] * 100
    print(f"Computed PC1 from {PCA_VARIABLES} (explains {explained_var:.1f}% variance)")
    
    return df


def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load probe and EVA data, merge to create block-level dataset for
    Cyberball → Thoughts → Delayed Mood mediation analysis.
    
    Returns
    -------
    pd.DataFrame
        Block-level data with:
        - condition_bin: 0=Inclusion, 1=Exclusion
        - group_bin: 0=Controls, 1=Risk
        - mood_pre: Baseline mood (Sart1 for Sart2, Sart3 for Sart4)
        - mood_delayed: Mood at end of SART (last EVA block)
        - thought dimensions: Mean per block (during SART)
    """
    print("\n" + "=" * 70)
    print("DATA LOADING & PREPROCESSING - DELAYED MOOD MEDIATION")
    print("=" * 70)

    # 1. Load Probe Data (Thoughts during SART)
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

    # Compute PC1 if not present
    df_probes = compute_pc1(df_probes)

    # 2. Aggregate Probes to Block Level (Thoughts = Mediator)
    print("Aggregating probes to block level (thoughts = mediator)...")
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

    # 4. Extract Mood Pre (baseline) and Mood Delayed (after SART) per block
    # mood_pre: From baseline block (Sart1 for Sart2, Sart3 for Sart4)
    # mood_delayed: LAST EVA measurement of the manipulation block (after completing SART)
    print("Extracting mood_pre (baseline) and mood_delayed (after SART)...")
    
    # Get FIRST EVA measurement per task as baseline mood
    df_mood_first = df_eva.sort_values(["subject_id", "task", "block_number"]).groupby(
        ["subject_id", "task"], as_index=False
    ).first()
    
    # Get LAST EVA measurement per task as delayed mood (after completing SART)
    df_mood_last = df_eva.sort_values(["subject_id", "task", "block_number"]).groupby(
        ["subject_id", "task"], as_index=False
    ).last()
    
    # Create mapping: Sart2 uses Sart1 as baseline, Sart4 uses Sart3 as baseline
    baseline_map = {"Sart2": "Sart1", "Sart4": "Sart3"}
    
    mood_data_list = []
    for _, row in df_blocks.iterrows():
        subj = row["subject_id"]
        task = row["task"]
        baseline_task = baseline_map.get(task)
        
        # Get mood_delayed from current task (LAST block - after completing SART)
        mood_delayed_row = df_mood_last[
            (df_mood_last["subject_id"] == subj) & 
            (df_mood_last["task"] == task)
        ]
        
        # Get mood_pre from baseline task (FIRST block)
        mood_pre_row = df_mood_first[
            (df_mood_first["subject_id"] == subj) & 
            (df_mood_first["task"] == baseline_task)
        ]
        
        mood_entry = {"subject_id": subj, "task": task}
        
        for mood in MOOD_SCALES:
            mood_entry[f"{mood}_delayed"] = (
                float(mood_delayed_row[mood].iloc[0]) if not mood_delayed_row.empty and mood in mood_delayed_row.columns 
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
    required_cols = [f"{m}_pre" for m in MOOD_SCALES] + [f"{m}_delayed" for m in MOOD_SCALES]
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
        result = model.fit(method="powell", maxiter=2000, disp=False)
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
        Path b coefficient (thoughts → mood_delayed)
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
    Run moderated mediation analysis: Condition → Thoughts → Delayed Mood (moderated by Group).
    
    For each combination of Mood Scale (Outcome) and Thought Dimension (Mediator):
    - Mediator Model: thought ~ condition_bin * group_bin + mood_pre
    - Outcome Model: mood_delayed ~ thought + condition_bin * group_bin + mood_pre
    """
    print("\n" + "=" * 70)
    print("RUNNING MODERATED MEDIATION MODELS (Hayes Model 7)")
    print("=" * 70)
    print(f"Testing: Condition → Thoughts → Delayed Mood (Group moderates Path a)")
    print(f"Controlling for: Baseline Mood (mood_pre)")
    
    results = []
    total_models = len(THOUGHT_DIMENSIONS) * len(MOOD_SCALES)
    count = 0
    
    for thought in THOUGHT_DIMENSIONS:
        if thought not in data.columns:
            print(f"Skipping {thought}: column not found")
            continue
        
        # --- Mediator Model: Path a is moderated ---
        # thought ~ condition_bin * group_bin + mood_pre (using EVAaverage_pre as baseline)
        mood_pre = "EVAaverage_pre"  # Use average mood as baseline covariate
        
        formula_m = f"{thought} ~ condition_bin * group_bin + {mood_pre}"
        res_m, conv_m = fit_lmm(formula_m, data)
        
        if not conv_m:
            print(f"Skipping {thought}: Mediator model failed to converge.")
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
        
        for mood in MOOD_SCALES:
            count += 1
            mood_delayed = f"{mood}_delayed"
            mood_pre_specific = f"{mood}_pre"
            
            if mood_delayed not in data.columns:
                continue
            
            print(f"[{count}/{total_models}] Mediator: {thought} -> Outcome: {mood} ...", end="\r")
            
            # --- Outcome Model: Path b and c' ---
            # mood_delayed ~ thought + condition_bin * group_bin + mood_pre
            formula_y = f"{mood_delayed} ~ {thought} + condition_bin * group_bin + {mood_pre_specific}"
            res_y, conv_y = fit_lmm(formula_y, data)
            
            if not conv_y:
                continue
            
            b = res_y.params[thought]  # Path b: thought → mood_delayed
            b_se = res_y.bse[thought]
            
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
                "thought_dim": thought,
                "mood_scale": mood,
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


def run_path_a_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    """Quick diagnostic for Path A: Does Cyberball affect thoughts differently by group?

    Tests the interaction ``condition_bin * group_bin`` on each thought dimension.
    """
    print("\n" + "=" * 70)
    print("QUICK DIAGNOSTIC: PATH A (Condition → Thoughts)")
    print("=" * 70)

    df = data.copy()
    mood_pre = "EVAaverage_pre"

    print("TESTING PATH A: Does Exclusion affect Thoughts differently in Risk Group?")
    results: list[dict] = []

    for thought in THOUGHT_DIMENSIONS:
        if thought not in df.columns:
            continue

        formula = f"{thought} ~ condition_bin * group_bin + {mood_pre}"
        res, conv = fit_lmm(formula, df)
        if not conv:
            continue

        # Interaction term: additional effect of exclusion in Risk vs Controls
        beta_int = res.params.get("condition_bin:group_bin", np.nan)
        pval_int = res.pvalues.get("condition_bin:group_bin", np.nan)

        sig_flag = (pd.notna(pval_int) and pval_int < 0.05)
        sig_str = "**" if sig_flag else ""
        if pd.notna(beta_int) and pd.notna(pval_int):
            print(f"{thought}: Beta_Interaction = {beta_int:.3f}, p = {pval_int:.3f} {sig_str}")
        else:
            print(f"{thought}: interaction term not estimable")

        results.append(
            {
                "thought_dim": thought,
                "beta_interaction": float(beta_int) if pd.notna(beta_int) else np.nan,
                "pval_interaction": float(pval_int) if pd.notna(pval_int) else np.nan,
                "significant_0.05": bool(sig_flag),
            }
        )

    # Simple diagnostic plot for valence
    if "valence" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.pointplot(
            data=df,
            x="condition_bin",
            y="valence",
            hue="group_bin",
            dodge=True,
            errorbar=("ci", 95),
        )
        plt.title("Thought Valence by Cyberball Condition and Group")
        plt.ylabel("Thought Valence (0-100)")
        plt.xlabel("Cyberball Condition")
        plt.xticks([0, 1], ["Inclusion", "Exclusion"])
        plt.legend(title="Group", labels=["Controls", "Risk"])
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "plots", "path_a_valence_diagnostic.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Path A diagnostic plot saved: {out_path}")
        plt.close()

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_moderated_mediation_figure(results: pd.DataFrame) -> None:
    """
    Generate combined figure with:
    - Top: Forest plot of Indirect Effects by Group
    - Middle Left: Path A (Condition → Thoughts, moderated by Group)
    - Middle Right: Path B Heatmap (Thoughts → Delayed Mood)
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
    
    # ========== PANEL 1: INDIRECT EFFECTS HEATMAPS BY GROUP ==========
    # Two side-by-side heatmaps showing indirect effects for Controls and Risk
    ax_ie_ctrl = fig.add_subplot(gs[0, 0])
    ax_ie_risk = fig.add_subplot(gs[0, 1])
    
    # Pivot data for heatmaps: rows=mood, cols=thought
    pivot_ie_ctrl = results.pivot(index="mood_scale", columns="thought_dim", values="ie_controls")
    pivot_ie_ctrl = pivot_ie_ctrl.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    pivot_ie_risk = results.pivot(index="mood_scale", columns="thought_dim", values="ie_risk")
    pivot_ie_risk = pivot_ie_risk.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    # Significance masks
    pivot_sig_ctrl = results.pivot(index="mood_scale", columns="thought_dim", values="ie_controls_sig")
    pivot_sig_ctrl = pivot_sig_ctrl.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    pivot_sig_risk = results.pivot(index="mood_scale", columns="thought_dim", values="ie_risk_sig")
    pivot_sig_risk = pivot_sig_risk.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    # Get common color scale
    all_ie_ctrl = pivot_ie_ctrl.values.flatten()
    all_ie_risk = pivot_ie_risk.values.flatten()
    all_ie = np.concatenate([all_ie_ctrl[~np.isnan(all_ie_ctrl)], 
                              all_ie_risk[~np.isnan(all_ie_risk)]])
    if len(all_ie) > 0:
        vmax = max(abs(all_ie.min()), abs(all_ie.max()))
    else:
        vmax = 1.0
    
    # Controls heatmap (color only significant)
    colored_ctrl = pivot_ie_ctrl.copy()
    colored_ctrl[~pivot_sig_ctrl.fillna(False)] = np.nan
    
    sns.heatmap(colored_ctrl, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax_ie_ctrl, cbar_kws={'label': 'Indirect Effect'},
                linewidths=1, linecolor='white')
    
    # Add annotations for Controls
    for i in range(pivot_ie_ctrl.shape[0]):
        for j in range(pivot_ie_ctrl.shape[1]):
            val = pivot_ie_ctrl.iloc[i, j]
            if not pd.isna(val):
                text = f"{val:.2f}"
                is_sig = pivot_sig_ctrl.iloc[i, j] if not pd.isna(pivot_sig_ctrl.iloc[i, j]) else False
                if is_sig:
                    text += "*"
                text_color = 'white' if is_sig else 'black'
                ax_ie_ctrl.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                               color=text_color, fontsize=10, fontweight='bold')
    
    ax_ie_ctrl.set_title(f"Indirect Effects: CONTROLS\n(Cyberball → Thought → Mood)", 
                         fontweight='bold', pad=12, fontsize=13, color=COLOR_CONTROL)
    ax_ie_ctrl.set_ylabel("Mood Scale (Outcome)", fontweight='bold', fontsize=11)
    ax_ie_ctrl.set_xlabel("Thought Dimension (Mediator)", fontweight='bold', fontsize=11)
    
    # Risk heatmap (color only significant)
    colored_risk = pivot_ie_risk.copy()
    colored_risk[~pivot_sig_risk.fillna(False)] = np.nan
    
    sns.heatmap(colored_risk, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                ax=ax_ie_risk, cbar_kws={'label': 'Indirect Effect'},
                linewidths=1, linecolor='white')
    
    # Add annotations for Risk
    for i in range(pivot_ie_risk.shape[0]):
        for j in range(pivot_ie_risk.shape[1]):
            val = pivot_ie_risk.iloc[i, j]
            if not pd.isna(val):
                text = f"{val:.2f}"
                is_sig = pivot_sig_risk.iloc[i, j] if not pd.isna(pivot_sig_risk.iloc[i, j]) else False
                if is_sig:
                    text += "*"
                text_color = 'white' if is_sig else 'black'
                ax_ie_risk.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                               color=text_color, fontsize=10, fontweight='bold')
    
    ax_ie_risk.set_title(f"Indirect Effects: RISK GROUP\n(Cyberball → Thought → Mood)", 
                         fontweight='bold', pad=12, fontsize=13, color=COLOR_RISK)
    ax_ie_risk.set_ylabel("Mood Scale (Outcome)", fontweight='bold', fontsize=11)
    ax_ie_risk.set_xlabel("Thought Dimension (Mediator)", fontweight='bold', fontsize=11)
    
    # ========== PANEL 2: PATH A - CONDITION → THOUGHTS (MODERATED) ==========
    ax_path_a = fig.add_subplot(gs[1, 0])
    
    # Get unique path a values per thought dimension
    path_a_data = results.drop_duplicates(subset=["thought_dim"]).set_index("thought_dim").reindex(THOUGHT_DIMENSIONS)
    
    y_pos = np.arange(len(THOUGHT_DIMENSIONS))
    
    for idx, thought in enumerate(THOUGHT_DIMENSIONS):
        if thought not in path_a_data.index:
            continue
        row = path_a_data.loc[thought]
        
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
    ax_path_a.set_yticklabels(THOUGHT_DIMENSIONS, fontweight='bold')
    ax_path_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_path_a.set_xlabel("Path A Coefficient (Exclusion Effect on Thoughts)", fontweight='bold', fontsize=11)
    ax_path_a.set_title("Path A: Condition → Thoughts\n(Moderated by Group)", fontweight='bold', pad=12, fontsize=12)
    ax_path_a.grid(True, axis='x', alpha=0.2, linestyle=':')
    ax_path_a.invert_yaxis()
    
    # ========== PANEL 3: PATH B HEATMAP (THOUGHTS → DELAYED MOOD) ==========
    ax_path_b = fig.add_subplot(gs[1, 1])
    
    pivot_b = results.pivot(index="mood_scale", columns="thought_dim", values="b")
    pivot_b = pivot_b.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    # Significance mask
    pivot_b_se = results.pivot(index="mood_scale", columns="thought_dim", values="b_se")
    pivot_b_se = pivot_b_se.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
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
    
    ax_path_b.set_title("Path B: Thoughts → Delayed Mood", fontweight='bold', pad=12, fontsize=12)
    ax_path_b.set_ylabel("Mood Scale (Outcome)", fontweight='bold', fontsize=11)
    ax_path_b.set_xlabel("Thought Dimension (Mediator)", fontweight='bold', fontsize=11)
    
    # ========== PANEL 4: INDEX OF MODERATED MEDIATION ==========
    ax_index = fig.add_subplot(gs[2, :])
    
    # Pivot for heatmap
    pivot_index = results.pivot(index="mood_scale", columns="thought_dim", values="index_mm")
    pivot_index = pivot_index.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
    pivot_index_sig = results.pivot(index="mood_scale", columns="thought_dim", values="index_mm_sig")
    pivot_index_sig = pivot_index_sig.reindex(index=MOOD_SCALES, columns=THOUGHT_DIMENSIONS)
    
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
    ax_index.set_ylabel("Mood Scale (Outcome)", fontweight='bold', fontsize=11)
    ax_index.set_xlabel("Thought Dimension (Mediator)", fontweight='bold', fontsize=11)
    
    # ========== PANEL 5: SUMMARY BAR CHART ==========
    ax_summary = fig.add_subplot(gs[3, :])
    
    # Average across thought dimensions for each mood scale
    summary_data = results.groupby("mood_scale").agg({
        "ie_controls": "mean",
        "ie_risk": "mean",
        "index_mm": "mean",
    }).reindex(MOOD_SCALES)
    
    x = np.arange(len(MOOD_SCALES))
    width = 0.25
    
    bars1 = ax_summary.bar(x - width, summary_data["ie_controls"], width, 
                           label='IE Controls', color=COLOR_CONTROL, alpha=0.8)
    bars2 = ax_summary.bar(x, summary_data["ie_risk"], width,
                           label='IE Risk', color=COLOR_RISK, alpha=0.8)
    bars3 = ax_summary.bar(x + width, summary_data["index_mm"], width,
                           label='Index MM', color='#9B59B6', alpha=0.8)
    
    ax_summary.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax_summary.set_xticks(x)
    ax_summary.set_xticklabels(MOOD_SCALES, fontweight='bold', fontsize=11)
    ax_summary.set_ylabel("Effect Size", fontweight='bold', fontsize=11)
    ax_summary.set_title("Summary: Average Effects Across Thought Dimensions", fontweight='bold', pad=12, fontsize=12)
    ax_summary.legend(loc='best', fontsize=10)
    ax_summary.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Save
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "plots", "cyberball_delayed_mood_mediation_combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved: {out_path}")
    plt.close()


def save_detailed_results(results: pd.DataFrame, data: pd.DataFrame) -> None:
    """Save detailed model results to text files."""
    print("\nSaving detailed model results...")
    
    txt_path = os.path.join(RESULTS_DIR, "detailed_model_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CYBERBALL DELAYED MOOD MEDIATION ANALYSIS - DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Hypothesis: Cyberball manipulation affects thoughts during SART,\n")
        f.write("which in turn influence delayed mood (measured after completing SART).\n")
        f.write("The Risk Group may show stronger thought-driven mood changes.\n\n")
        
        f.write("Model: Hayes Model 7 (Moderated Mediation)\n")
        f.write("  X: Cyberball Condition (0=Inclusion, 1=Exclusion)\n")
        f.write("  M: Thought Dimensions (during SART)\n")
        f.write("  Y: Mood Delayed (after completing SART)\n")
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
        if len(sig_index) > 0:
            f.write("SIGNIFICANT INDEX OF MODERATED MEDIATION:\n")
            f.write("-" * 50 + "\n")
            for _, row in sig_index.iterrows():
                f.write(f"  {row['thought_dim']} → {row['mood_scale']}: ")
                f.write(f"Index = {row['index_mm']:.3f} [{row['index_mm_ci_low']:.3f}, {row['index_mm_ci_high']:.3f}]\n")
            f.write("\n")
        else:
            f.write("No significant Index of Moderated Mediation effects found.\n\n")
        
        # Significant indirect effects for Controls
        sig_controls = results[results["ie_controls_sig"]].sort_values("ie_controls", ascending=False)
        if len(sig_controls) > 0:
            f.write("SIGNIFICANT INDIRECT EFFECTS - CONTROLS:\n")
            f.write("-" * 50 + "\n")
            for _, row in sig_controls.iterrows():
                f.write(f"  {row['thought_dim']} → {row['mood_scale']}: ")
                f.write(f"IE = {row['ie_controls']:.3f} [{row['ie_controls_ci_low']:.3f}, {row['ie_controls_ci_high']:.3f}]\n")
            f.write("\n")
        
        # Significant indirect effects for Risk
        sig_risk = results[results["ie_risk_sig"]].sort_values("ie_risk", ascending=False)
        if len(sig_risk) > 0:
            f.write("SIGNIFICANT INDIRECT EFFECTS - RISK GROUP:\n")
            f.write("-" * 50 + "\n")
            for _, row in sig_risk.iterrows():
                f.write(f"  {row['thought_dim']} → {row['mood_scale']}: ")
                f.write(f"IE = {row['ie_risk']:.3f} [{row['ie_risk_ci_low']:.3f}, {row['ie_risk_ci_high']:.3f}]\n")
            f.write("\n")
    
    print(f"Detailed results saved: {txt_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    """Main entry point for the delayed mood mediation analysis."""
    print("\n" + "=" * 70)
    print("CYBERBALL DELAYED MOOD MEDIATION ANALYSIS")
    print("Path: Cyberball Condition → Thoughts (SART) → Delayed Mood")
    print("=" * 70)
    
    ensure_directories()
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Run diagnostics
    path_a_results = run_path_a_diagnostic(data)
    path_a_results.to_csv(os.path.join(RESULTS_DIR, "path_a_diagnostic.csv"), index=False)
    
    # Run moderated mediation
    results = run_moderated_mediation(data)
    
    # Save results
    results.to_csv(os.path.join(RESULTS_DIR, "mediation_results.csv"), index=False)
    print(f"\nResults saved: {os.path.join(RESULTS_DIR, 'mediation_results.csv')}")
    
    # Generate visualization
    plot_moderated_mediation_figure(results)
    
    # Save detailed results
    save_detailed_results(results, data)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
