"""
Diagnostic script to identify causes of "bubbly" (spatially incoherent) t-statistics.

This script checks for common issues that cause fragmented spatial patterns:
1. Multicollinearity between predictors
2. Insufficient within-subject variance
3. Channel-specific convergence issues
4. Predictor correlation with spatial structure

Run this BEFORE your main analysis to diagnose data issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from reader import load_all_probe_data, prepare_data_for_lmm
from helpers import extract_all_formula_variables


def check_multicollinearity(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    """
    Check for multicollinearity between predictors using correlation matrix.
    
    High correlations (|r| > 0.7) can cause variance to be split unpredictably
    across predictors, leading to spatially incoherent effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data
    predictors : list
        List of predictor variable names
        
    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix between predictors
    """
    print("\n" + "="*70)
    print("MULTICOLLINEARITY CHECK")
    print("="*70)
    
    # Compute correlation matrix
    corr_matrix = df[predictors].corr()
    
    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))
    
    # Check for high correlations (excluding diagonal)
    high_corr = []
    for i, pred1 in enumerate(predictors):
        for j, pred2 in enumerate(predictors):
            if i < j:  # Only upper triangle
                corr_val = corr_matrix.loc[pred1, pred2]
                if abs(corr_val) > 0.7:
                    high_corr.append((pred1, pred2, corr_val))
    
    if high_corr:
        print("\n⚠ HIGH CORRELATIONS DETECTED (|r| > 0.7):")
        for pred1, pred2, corr_val in high_corr:
            print(f"  {pred1} <-> {pred2}: r = {corr_val:.3f}")
        print("\n  → This can cause variance to split unpredictably between predictors")
        print("  → Consider removing one predictor from each highly correlated pair")
    else:
        print("\n✓ No high correlations detected")
    
    return corr_matrix


def check_within_subject_variance(df: pd.DataFrame, predictor: str) -> dict:
    """
    Check if predictor has sufficient within-subject variance.
    
    LMM needs within-subject variation to estimate effects. If most variance
    is between-subjects, the effect may be spatially fragmented.
    
    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data with 'subject' column
    predictor : str
        Predictor variable name
        
    Returns
    -------
    variance_stats : dict
        Within and between subject variance statistics
    """
    print("\n" + "="*70)
    print(f"WITHIN-SUBJECT VARIANCE CHECK: {predictor}")
    print("="*70)
    
    # Compute within and between subject variance
    grand_mean = df[predictor].mean()
    
    # Between-subject variance (variance of subject means)
    subject_means = df.groupby('subject')[predictor].mean()
    between_var = np.var(subject_means)
    
    # Within-subject variance (average variance within subjects)
    within_vars = []
    for subj in df['subject'].unique():
        subj_data = df[df['subject'] == subj][predictor]
        if len(subj_data) > 1:
            within_vars.append(np.var(subj_data))
    within_var = np.mean(within_vars) if within_vars else 0.0
    
    # Total variance
    total_var = np.var(df[predictor])
    
    # Intraclass correlation (ICC): proportion of variance between subjects
    icc = between_var / total_var if total_var > 0 else 0.0
    
    print(f"\nVariance decomposition:")
    print(f"  Total variance:   {total_var:.4f}")
    print(f"  Between-subject:  {between_var:.4f} ({100*between_var/total_var:.1f}%)")
    print(f"  Within-subject:   {within_var:.4f} ({100*within_var/total_var:.1f}%)")
    print(f"  ICC (between/total): {icc:.3f}")
    
    if icc > 0.8:
        print("\n⚠ WARNING: Very high ICC (>0.8)")
        print("  → Most variance is between subjects, not within")
        print("  → LMM may struggle to estimate within-subject effects")
        print("  → This can cause spatially fragmented results")
    elif icc > 0.5:
        print("\n⚠ Moderate ICC (0.5-0.8)")
        print("  → Substantial between-subject variance")
        print("  → Consider subject-level normalization")
    else:
        print("\n✓ Good within-subject variance (ICC < 0.5)")
    
    return {
        'total_var': total_var,
        'between_var': between_var,
        'within_var': within_var,
        'icc': icc
    }


def check_predictor_distributions(df: pd.DataFrame, predictors: list):
    """
    Check distributions of predictors for outliers and skewness.
    
    Extreme values or skewed distributions can cause spatially incoherent
    effects by giving undue weight to specific observations.
    """
    print("\n" + "="*70)
    print("PREDICTOR DISTRIBUTION CHECK")
    print("="*70)
    
    for pred in predictors:
        print(f"\n{pred}:")
        print(f"  Mean: {df[pred].mean():.3f}")
        print(f"  Std:  {df[pred].std():.3f}")
        print(f"  Min:  {df[pred].min():.3f}")
        print(f"  Max:  {df[pred].max():.3f}")
        print(f"  Skewness: {df[pred].skew():.3f}")
        
        # Check for extreme skewness
        if abs(df[pred].skew()) > 2:
            print(f"  ⚠ High skewness (|skew| > 2) - consider transformation")
        
        # Check for outliers (>3 SD from mean)
        z_scores = np.abs((df[pred] - df[pred].mean()) / df[pred].std())
        n_outliers = np.sum(z_scores > 3)
        if n_outliers > 0:
            pct_outliers = 100 * n_outliers / len(df)
            print(f"  ⚠ {n_outliers} outliers (>3 SD): {pct_outliers:.1f}% of data")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path):
    """Plot correlation heatmap between predictors."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=1)
    plt.title('Predictor Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Correlation heatmap saved to: {output_path}")
    plt.close()


def plot_within_subject_variance(df: pd.DataFrame, predictor: str, output_path: Path):
    """Plot within-subject variance for predictor."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Subject means
    subject_means = df.groupby('subject')[predictor].mean().sort_values()
    axes[0].bar(range(len(subject_means)), subject_means.values)
    axes[0].axhline(df[predictor].mean(), color='r', linestyle='--', label='Grand mean')
    axes[0].set_xlabel('Subject (sorted by mean)')
    axes[0].set_ylabel(f'{predictor} (mean)')
    axes[0].set_title('Between-Subject Variance')
    axes[0].legend()
    
    # Plot 2: Within-subject distributions (box plots for first 20 subjects)
    subjects = df['subject'].unique()[:20]
    data_to_plot = [df[df['subject'] == subj][predictor].values for subj in subjects]
    axes[1].boxplot(data_to_plot, labels=[str(s) for s in subjects])
    axes[1].axhline(df[predictor].mean(), color='r', linestyle='--', label='Grand mean')
    axes[1].set_xlabel('Subject')
    axes[1].set_ylabel(predictor)
    axes[1].set_title('Within-Subject Variance (first 20 subjects)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Within-subject variance plot saved to: {output_path}")
    plt.close()


def main():
    """Run diagnostic checks."""
    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override cluster paths with local paths
    config['project']['features_root'] = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/features/"
    
    # Create output directory
    output_dir = Path(__file__).parent / "diagnostics" / "bubbly_tstats"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("DIAGNOSTIC: Identifying causes of 'bubbly' t-statistics")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    
    # Load probe data
    print("\nLoading data...")
    df_all = load_all_probe_data(
        features_root=config['project']['features_root'],
        subjects=config['project']['subjects'],
        tasks=config['project']['tasks'],
        marker_types=['evoked'],  # Just use evoked for diagnosis
        verbose=True
    )
    
    if df_all is None or len(df_all) == 0:
        print("ERROR: No data loaded. Check your config.yaml paths.")
        return
    
    # Get unique markers and use first one
    available_markers = df_all['marker'].unique()
    first_marker = available_markers[0]
    
    print(f"\nUsing marker: {first_marker}")
    print(f"Available markers: {list(available_markers)[:5]}...")
    
    # Prepare data for this marker
    power_data, df_behavioral, channels = prepare_data_for_lmm(
        df=df_all,
        marker_name=first_marker,
        formula=config['lmm']['formula']
    )
    
    print(f"Data shape: {len(df_behavioral)} observations, {df_behavioral['subject'].nunique()} subjects")
    print(f"Channels: {len(channels)}")
    
    # Extract predictors from formula
    formula = config['lmm']['formula']
    predictor_of_interest = config['lmm']['predictor_of_interest']
    
    print(f"\nFormula: {formula}")
    print(f"Predictor of interest: {predictor_of_interest}")
    
    # Get all predictors (excluding 'power' and random effects)
    all_vars = extract_all_formula_variables(formula)
    predictors = [v for v in all_vars if v != 'power']
    
    print(f"\nPredictors to check: {predictors}")
    
    # Run checks
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTIC CHECKS")
    print("="*70)
    
    # 1. Multicollinearity
    corr_matrix = check_multicollinearity(df_behavioral, predictors)
    plot_correlation_heatmap(corr_matrix, output_dir / "predictor_correlations.png")
    
    # 2. Within-subject variance for predictor of interest
    variance_stats = check_within_subject_variance(df_behavioral, predictor_of_interest)
    plot_within_subject_variance(df_behavioral, predictor_of_interest, 
                                 output_dir / f"within_subject_variance_{predictor_of_interest}.png")
    
    # 3. Distribution checks
    check_predictor_distributions(df_behavioral, predictors)
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*70)
    
    # Check for critical issues
    critical_issues = []
    
    # High multicollinearity
    high_corr_pairs = []
    for i, pred1 in enumerate(predictors):
        for j, pred2 in enumerate(predictors):
            if i < j and abs(corr_matrix.loc[pred1, pred2]) > 0.7:
                high_corr_pairs.append((pred1, pred2, corr_matrix.loc[pred1, pred2]))
    
    if high_corr_pairs:
        critical_issues.append("High multicollinearity detected")
        print("\n⚠ CRITICAL: High multicollinearity")
        print("  Recommendation: Remove one predictor from each highly correlated pair:")
        for pred1, pred2, corr_val in high_corr_pairs:
            print(f"    - Either {pred1} or {pred2} (r={corr_val:.3f})")
    
    # High ICC
    if variance_stats['icc'] > 0.8:
        critical_issues.append("Very high ICC (most variance between subjects)")
        print("\n⚠ CRITICAL: Very high ICC")
        print("  Recommendation: Enable channel-wise normalization:")
        print("    preprocessing:")
        print("      normalize_by_subject: true")
        print("      channel_wise: true")
    
    if not critical_issues:
        print("\n✓ No critical issues detected")
        print("\nPossible causes of 'bubbly' t-stats:")
        print("  1. Effect is genuinely heterogeneous across channels")
        print("  2. Insufficient statistical power (increase n_permutations)")
        print("  3. Model complexity fragmenting spatial patterns")
        print("     → Try simpler model: 'power ~ onoff + (1|subject)'")
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
