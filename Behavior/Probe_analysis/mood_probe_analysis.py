#!/usr/bin/env python3
"""
Mood (EVA) as Predictor of Thought Dimensions Analysis

This script analyzes how mood state (measured by EVA scale) predicts the dimensions
of thought-probes during SART tasks. We use the EVA measured at the beginning of 
each SART as a predictor of thought dimensions during that SART.

Design
------
- EVA data: Mood scale with 4 questions (tense, feel, mood, hurt) + total score
- Probe data: Thought dimensions (onoff, valence, time, selfother, confidence)
- Analysis: Linear mixed models with mood predicting each thought dimension
- Controls: Group (Controls vs Risk of Depression), task, subject random effects

EVA Timing Structure:
- Sart1: EVA at end only (~1180s)
- Sart2: EVA at end only (~1170s)  
- Sart3: EVA at beginning (~2s) AND end (~1135s)
- Sart4: EVA at end only (~1180s)

Strategy:
- Use Sart3 block 1 (beginning) as predictor for Sart3 probes
- For other SARTs, we could use previous SART's end EVA or same SART's end EVA

Author: Analysis Assistant
Date: 2025-11-13
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================
PROBE_DATA_FILE = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/probe_data/probe_level_aggregated_data.csv"
EVA_DATA_FILE = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/scales_data/eva_aggregated_data.csv"

# Output directories
RESULTS_DIR = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/mood_probe_analysis"
PLOTS_DIR = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/mood_probe_analysis/plots"

# Thought dimensions to analyze as dependent variables
THOUGHT_DIMENSIONS = ["onoff", "valence", "time", "selfother", "confidence"]

# EVA dimensions to use as predictors
EVA_DIMENSIONS = ["EVAtense", "EVAfeel", "EVAmood", "EVAhurt", "total_score"]

# Plot aesthetics
GROUP_ORDER = ["Controls", "Risk of Depression"]
GROUP_COLORS = ["#2E86AB", "#F24236"]

# Optional filter for onoff dimension
APPLY_ONOFF_FILTER = True
ONOFF_MAX_EXCLUSIVE = 50

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_and_merge_data() -> pd.DataFrame:
    """
    Load probe and EVA data, then merge them appropriately.
    
    Strategy:
    - For Sart3: Use EVA block 1 (beginning, onset ~0-5s) as predictor
    - For Sart1, Sart2, Sart4: Use EVA at end of same SART
    
    Returns
    -------
    pd.DataFrame
        Merged dataframe with probe data and corresponding EVA values
    """
    # Load probe data
    if not os.path.exists(PROBE_DATA_FILE):
        raise FileNotFoundError(f"Probe data file not found: {PROBE_DATA_FILE}")
    df_probes = pd.read_csv(PROBE_DATA_FILE)
    print(f"Loaded {len(df_probes)} probe observations")
    
    # Load EVA data
    if not os.path.exists(EVA_DATA_FILE):
        raise FileNotFoundError(f"EVA data file not found: {EVA_DATA_FILE}")
    df_eva = pd.read_csv(EVA_DATA_FILE)
    print(f"Loaded {len(df_eva)} EVA blocks")
    
    # Apply onoff filter if requested
    if APPLY_ONOFF_FILTER:
        before = len(df_probes)
        df_probes = df_probes[df_probes["onoff"] < ONOFF_MAX_EXCLUSIVE].copy()
        after = len(df_probes)
        print(f"Applied onoff filter (<{ONOFF_MAX_EXCLUSIVE}): {before} -> {after} probes")
    
    # Prepare EVA data for merging
    # For Sart3: use block 1 (beginning)
    # For others: use the only block available (end)
    df_eva_for_merge = df_eva.copy()
    
    # For Sart3, keep only block 1 (beginning)
    sart3_eva = df_eva_for_merge[
        (df_eva_for_merge['task'] == 'Sart3') & 
        (df_eva_for_merge['block_number'] == 1)
    ].copy()
    
    # For other SARTs, keep all (they only have 1 block anyway)
    other_eva = df_eva_for_merge[df_eva_for_merge['task'] != 'Sart3'].copy()
    
    # Combine
    df_eva_for_merge = pd.concat([sart3_eva, other_eva], ignore_index=True)
    
    print(f"\nEVA blocks for merging: {len(df_eva_for_merge)}")
    print("  Distribution by task:")
    print(df_eva_for_merge.groupby('task').size())
    
    # Rename EVA columns to add 'eva_' prefix to avoid conflicts
    eva_cols_to_rename = {col: f'eva_{col}' for col in EVA_DIMENSIONS}
    df_eva_for_merge = df_eva_for_merge.rename(columns=eva_cols_to_rename)
    
    # Merge on subject_id and task
    df_merged = pd.merge(
        df_probes,
        df_eva_for_merge[['subject_id', 'task'] + [f'eva_{col}' for col in EVA_DIMENSIONS]],
        on=['subject_id', 'task'],
        how='left'
    )
    
    # Check merge success
    n_with_eva = df_merged['eva_total_score'].notna().sum()
    print(f"\nMerge results:")
    print(f"  Total probes: {len(df_merged)}")
    print(f"  Probes with EVA data: {n_with_eva} ({100*n_with_eva/len(df_merged):.1f}%)")
    print(f"  Probes without EVA: {len(df_merged) - n_with_eva}")
    
    # Remove probes without EVA data
    df_merged = df_merged[df_merged['eva_total_score'].notna()].copy()
    print(f"\nFinal dataset: {len(df_merged)} probes with EVA data")
    
    return df_merged


def fit_multivariate_mood_model(df: pd.DataFrame, thought_dim: str) -> Dict:
    """
    Fit linear mixed model with ALL EVA dimensions as predictors:
    thought_dimension ~ EVAtense + EVAfeel + EVAmood + EVAhurt + group + (1|subject_id)
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with probe and EVA data
    thought_dim : str
        Thought dimension to predict (e.g., 'onoff', 'valence')
        
    Returns
    -------
    Dict
        Dictionary with model results including all EVA coefficients
    """
    # Prepare data with all EVA dimensions
    eva_cols = [f'eva_{dim}' for dim in ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']]
    cols_needed = [thought_dim] + eva_cols + ['group', 'subject_id']
    df_model = df[cols_needed].dropna().copy()
    
    if len(df_model) < 50:
        print(f"Warning: Only {len(df_model)} observations for {thought_dim}")
        return None
    
    # Center EVA predictors for interpretability
    for eva_col in eva_cols:
        df_model[f'{eva_col}_c'] = df_model[eva_col] - df_model[eva_col].mean()
    
    # Build formula with all centered EVA predictors
    eva_terms = ' + '.join([f'{eva_col}_c' for eva_col in eva_cols])
    formula = f"{thought_dim} ~ {eva_terms} + C(group, Treatment('Controls'))"
    
    try:
        model = smf.mixedlm(
            formula,
            data=df_model,
            groups=df_model['subject_id'],
            re_formula='1'
        )
        result = model.fit(method='lbfgs', maxiter=1000)
        
        # Extract key statistics
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()
        
        results_dict = {
            'thought_dim': thought_dim,
            'n_obs': len(df_model),
            'n_subjects': df_model['subject_id'].nunique(),
            'aic': result.aic,
            'bic': result.bic,
            'converged': result.converged,
            'group_coef': params.get("C(group, Treatment('Controls'))[T.Risk of Depression]", np.nan),
            'group_pval': pvalues.get("C(group, Treatment('Controls'))[T.Risk of Depression]", np.nan),
        }
        
        # Add coefficients for each EVA dimension
        for eva_dim in ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']:
            eva_col_c = f'eva_{eva_dim}_c'
            results_dict[f'{eva_dim}_coef'] = params.get(eva_col_c, np.nan)
            results_dict[f'{eva_dim}_pval'] = pvalues.get(eva_col_c, np.nan)
            if eva_col_c in conf_int.index:
                results_dict[f'{eva_dim}_ci_lower'] = conf_int.loc[eva_col_c, 0]
                results_dict[f'{eva_dim}_ci_upper'] = conf_int.loc[eva_col_c, 1]
            else:
                results_dict[f'{eva_dim}_ci_lower'] = np.nan
                results_dict[f'{eva_dim}_ci_upper'] = np.nan
        
        return results_dict
        
    except Exception as e:
        print(f"Error fitting model for {thought_dim}: {e}")
        return None


def fit_mood_model(df: pd.DataFrame, thought_dim: str, eva_dim: str) -> Dict:
    """
    Fit linear mixed model: thought_dimension ~ eva_dimension + group + (1|subject_id)
    
    This is kept for backward compatibility but not used in main analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe with probe and EVA data
    thought_dim : str
        Thought dimension to predict (e.g., 'onoff', 'valence')
    eva_dim : str
        EVA dimension to use as predictor (e.g., 'eva_total_score')
        
    Returns
    -------
    Dict
        Dictionary with model results
    """
    # Prepare data
    df_model = df[[thought_dim, eva_dim, 'group', 'subject_id']].dropna().copy()
    
    if len(df_model) < 50:
        print(f"Warning: Only {len(df_model)} observations for {thought_dim} ~ {eva_dim}")
        return None
    
    # Center EVA predictor for interpretability
    df_model[f'{eva_dim}_centered'] = df_model[eva_dim] - df_model[eva_dim].mean()
    
    # Fit model
    formula = f"{thought_dim} ~ {eva_dim}_centered + C(group, Treatment('Controls'))"
    
    try:
        model = smf.mixedlm(
            formula,
            data=df_model,
            groups=df_model['subject_id'],
            re_formula='1'
        )
        result = model.fit(method='lbfgs', maxiter=1000)
        
        # Extract key statistics
        params = result.params
        pvalues = result.pvalues
        conf_int = result.conf_int()
        
        results_dict = {
            'thought_dim': thought_dim,
            'eva_dim': eva_dim,
            'n_obs': len(df_model),
            'n_subjects': df_model['subject_id'].nunique(),
            'eva_coef': params.get(f'{eva_dim}_centered', np.nan),
            'eva_pval': pvalues.get(f'{eva_dim}_centered', np.nan),
            'eva_ci_lower': conf_int.loc[f'{eva_dim}_centered', 0] if f'{eva_dim}_centered' in conf_int.index else np.nan,
            'eva_ci_upper': conf_int.loc[f'{eva_dim}_centered', 1] if f'{eva_dim}_centered' in conf_int.index else np.nan,
            'group_coef': params.get("C(group, Treatment('Controls'))[T.Risk of Depression]", np.nan),
            'group_pval': pvalues.get("C(group, Treatment('Controls'))[T.Risk of Depression]", np.nan),
            'aic': result.aic,
            'bic': result.bic,
            'converged': result.converged
        }
        
        return results_dict
        
    except Exception as e:
        print(f"Error fitting model {thought_dim} ~ {eva_dim}: {e}")
        return None


def analyze_multivariate_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit multivariate models with ALL EVA dimensions as predictors for each thought dimension.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe
        
    Returns
    -------
    pd.DataFrame
        Results table with all model fits
    """
    results_list = []
    
    for thought_dim in THOUGHT_DIMENSIONS:
        print(f"\nFitting multivariate model: {thought_dim} ~ EVAtense + EVAfeel + EVAmood + EVAhurt + group")
        result = fit_multivariate_mood_model(df, thought_dim)
        if result is not None:
            results_list.append(result)
    
    df_results = pd.DataFrame(results_list)
    
    # Add significance stars for each EVA dimension
    def add_stars(pval):
        if pd.isna(pval):
            return ''
        elif pval < 0.001:
            return '***'
        elif pval < 0.01:
            return '**'
        elif pval < 0.05:
            return '*'
        else:
            return ''
    
    for eva_dim in ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']:
        df_results[f'{eva_dim}_sig'] = df_results[f'{eva_dim}_pval'].apply(add_stars)
    
    df_results['group_sig'] = df_results['group_pval'].apply(add_stars)
    
    return df_results


def analyze_all_combinations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit models for all combinations of thought dimensions and EVA predictors.
    
    This is kept for backward compatibility but not used in main analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe
        
    Returns
    -------
    pd.DataFrame
        Results table with all model fits
    """
    results_list = []
    
    for thought_dim in THOUGHT_DIMENSIONS:
        for eva_dim in [f'eva_{d}' for d in EVA_DIMENSIONS]:
            print(f"\nFitting: {thought_dim} ~ {eva_dim}")
            result = fit_mood_model(df, thought_dim, eva_dim)
            if result is not None:
                results_list.append(result)
    
    df_results = pd.DataFrame(results_list)
    
    # Add significance stars
    def add_stars(pval):
        if pd.isna(pval):
            return ''
        elif pval < 0.001:
            return '***'
        elif pval < 0.01:
            return '**'
        elif pval < 0.05:
            return '*'
        else:
            return ''
    
    df_results['eva_sig'] = df_results['eva_pval'].apply(add_stars)
    df_results['group_sig'] = df_results['group_pval'].apply(add_stars)
    
    return df_results


def plot_mood_correlations(df: pd.DataFrame) -> None:
    """
    Create correlation plots between EVA dimensions and thought dimensions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Merged dataframe
    """
    fig, axes = plt.subplots(len(THOUGHT_DIMENSIONS), len(EVA_DIMENSIONS), 
                             figsize=(20, 16))
    
    for i, thought_dim in enumerate(THOUGHT_DIMENSIONS):
        for j, eva_dim in enumerate(EVA_DIMENSIONS):
            ax = axes[i, j]
            eva_col = f'eva_{eva_dim}'
            
            # Scatter plot with regression line
            df_plot = df[[thought_dim, eva_col, 'group']].dropna()
            
            if len(df_plot) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_xlabel(eva_dim)
                ax.set_ylabel(thought_dim)
                continue
            
            # Plot by group
            for group_idx, group in enumerate(GROUP_ORDER):
                if group not in df_plot['group'].values:
                    continue
                group_data = df_plot[df_plot['group'] == group]
                color = GROUP_COLORS[group_idx]
                
                # Scatter
                ax.scatter(group_data[eva_col], group_data[thought_dim],
                          alpha=0.3, s=20, color=color, label=group)
                
                # Regression line
                if len(group_data) > 5:
                    z = np.polyfit(group_data[eva_col], group_data[thought_dim], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(group_data[eva_col].min(), 
                                        group_data[eva_col].max(), 100)
                    ax.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.7)
            
            # Compute overall correlation
            r, p = stats.pearsonr(df_plot[eva_col], df_plot[thought_dim])
            
            ax.set_xlabel(eva_dim, fontsize=10)
            ax.set_ylabel(thought_dim, fontsize=10)
            ax.set_title(f'r={r:.2f}, p={p:.3f}', fontsize=9)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'mood_thought_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved correlation plot: {PLOTS_DIR}/mood_thought_correlations.png")


def plot_coefficient_heatmap(df_results: pd.DataFrame) -> None:
    """
    Create heatmap of EVA coefficients predicting thought dimensions.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Results table from analyze_multivariate_models
    """
    # Extract coefficient and p-value matrices
    eva_dims = ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']
    thought_dims = df_results['thought_dim'].values
    
    # Create matrices
    coef_matrix = np.zeros((len(thought_dims), len(eva_dims)))
    pval_matrix = np.zeros((len(thought_dims), len(eva_dims)))
    
    for i, thought_dim in enumerate(thought_dims):
        for j, eva_dim in enumerate(eva_dims):
            coef_matrix[i, j] = df_results.loc[df_results['thought_dim'] == thought_dim, f'{eva_dim}_coef'].values[0]
            pval_matrix[i, j] = df_results.loc[df_results['thought_dim'] == thought_dim, f'{eva_dim}_pval'].values[0]
    
    # Create DataFrame for seaborn
    coef_df = pd.DataFrame(coef_matrix, index=thought_dims, columns=eva_dims)
    pval_df = pd.DataFrame(pval_matrix, index=thought_dims, columns=eva_dims)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Heatmap
    sns.heatmap(coef_df, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, cbar_kws={'label': 'Coefficient'},
                ax=ax, vmin=-1.0, vmax=3.5)
    
    # Add significance stars
    for i in range(len(thought_dims)):
        for j in range(len(eva_dims)):
            pval = pval_df.iloc[i, j]
            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else:
                stars = ''
            
            if stars:
                ax.text(j + 0.5, i + 0.7, stars, 
                       ha='center', va='center', 
                       color='black', fontsize=12, fontweight='bold')
    
    ax.set_title('Multivariate EVA Predictors of Thought Dimensions\n(LMM Coefficients - All EVA dimensions in same model)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('EVA Dimension (Predictor)', fontsize=12)
    ax.set_ylabel('Thought Dimension (Outcome)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'eva_coefficients_heatmap_multivariate.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {PLOTS_DIR}/eva_coefficients_heatmap_multivariate.png")


def main():
    """Main execution function."""
    print("=" * 70)
    print("MOOD (EVA) AS PREDICTOR OF THOUGHT DIMENSIONS")
    print("=" * 70)
    
    # Setup
    ensure_directories()
    
    # Load and merge data
    print("\n--- Loading and merging data ---")
    df = load_and_merge_data()
    
    # Descriptive statistics
    print("\n--- Descriptive Statistics ---")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Groups: {df['group'].value_counts()}")
    
    print("\nEVA dimensions (mean ± std):")
    for eva_dim in EVA_DIMENSIONS:
        col = f'eva_{eva_dim}'
        print(f"  {eva_dim}: {df[col].mean():.2f} ± {df[col].std():.2f}")
    
    print("\nThought dimensions (mean ± std):")
    for thought_dim in THOUGHT_DIMENSIONS:
        print(f"  {thought_dim}: {df[thought_dim].mean():.2f} ± {df[thought_dim].std():.2f}")
    
    # Fit multivariate models (all EVA dimensions together)
    print("\n--- Fitting Multivariate Linear Mixed Models ---")
    print("Each model includes ALL EVA dimensions as simultaneous predictors")
    df_results = analyze_multivariate_models(df)
    
    # Save results
    results_file = os.path.join(RESULTS_DIR, 'mood_probe_lmm_multivariate_results.csv')
    df_results.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")
    
    # Display results for each thought dimension
    print("\n" + "=" * 70)
    print("MULTIVARIATE MODEL RESULTS")
    print("=" * 70)
    
    for thought_dim in THOUGHT_DIMENSIONS:
        result = df_results[df_results['thought_dim'] == thought_dim]
        if len(result) == 0:
            continue
        
        print(f"\n{thought_dim.upper()}")
        print("-" * 50)
        print(f"N observations: {result['n_obs'].values[0]}")
        print(f"N subjects: {result['n_subjects'].values[0]}")
        print(f"AIC: {result['aic'].values[0]:.2f}, BIC: {result['bic'].values[0]:.2f}")
        print(f"\nEVA Coefficients:")
        
        for eva_dim in ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']:
            coef = result[f'{eva_dim}_coef'].values[0]
            pval = result[f'{eva_dim}_pval'].values[0]
            sig = result[f'{eva_dim}_sig'].values[0]
            ci_lower = result[f'{eva_dim}_ci_lower'].values[0]
            ci_upper = result[f'{eva_dim}_ci_upper'].values[0]
            print(f"  {eva_dim:12s}: β={coef:6.3f} [{ci_lower:6.3f}, {ci_upper:6.3f}], p={pval:.4f} {sig}")
        
        group_coef = result['group_coef'].values[0]
        group_pval = result['group_pval'].values[0]
        group_sig = result['group_sig'].values[0]
        print(f"\nGroup effect: β={group_coef:6.3f}, p={group_pval:.4f} {group_sig}")
    
    # Summary of significant predictors
    print("\n" + "=" * 70)
    print("SUMMARY: Significant EVA Predictors (p < 0.05)")
    print("=" * 70)
    
    sig_found = False
    for thought_dim in THOUGHT_DIMENSIONS:
        result = df_results[df_results['thought_dim'] == thought_dim]
        if len(result) == 0:
            continue
        
        sig_predictors = []
        for eva_dim in ['EVAtense', 'EVAfeel', 'EVAmood', 'EVAhurt']:
            pval = result[f'{eva_dim}_pval'].values[0]
            if pval < 0.05:
                coef = result[f'{eva_dim}_coef'].values[0]
                sig = result[f'{eva_dim}_sig'].values[0]
                sig_predictors.append(f"{eva_dim} (β={coef:.3f}, {sig})")
        
        if sig_predictors:
            sig_found = True
            print(f"\n{thought_dim}: {', '.join(sig_predictors)}")
    
    if not sig_found:
        print("\nNo significant predictors found.")
    
    # Create plots
    print("\n--- Creating Plots ---")
    plot_mood_correlations(df)
    plot_coefficient_heatmap(df_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
