"""
Analyze t-values from existing results to diagnose physiological coherence issues.

This script reads the saved pickle files from completed analyses and examines:
1. T-value distributions across channels and markers
2. Spatial patterns and coherence
3. Comparison across different model configurations
4. Identification of potential issues (extreme values, inconsistent patterns)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Results directory
RESULTS_DIR = Path("/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff_time_on_task_valence_selfother_time")

# Output directory
OUTPUT_DIR = Path(__file__).parent / "diagnostics" / "t_value_analysis_from_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Marker types to analyze
MARKER_TYPES = ["state", "evoked"]

# Sample markers to analyze in detail
SAMPLE_MARKERS = {
    "state": ["state_power_alpha", "state_power_beta", "state_power_theta"],
    "evoked": ["evoked_power_alpha", "evoked_power_beta", "evoked_power_theta"]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_marker_results(marker_dir):
    """Load results from a marker directory."""
    results_file = marker_dir / "results.pkl"
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"  ✗ Failed to load {results_file}: {e}")
        return None


def analyze_t_statistics(t_stats, marker_name, threshold=2.5):
    """Analyze t-statistic distribution for a single marker."""
    # Remove NaN values
    t_valid = t_stats[~np.isnan(t_stats)]
    
    analysis = {
        'marker': marker_name,
        'n_channels': len(t_stats),
        'n_valid': len(t_valid),
        'n_nan': np.sum(np.isnan(t_stats)),
        'pct_nan': 100 * np.sum(np.isnan(t_stats)) / len(t_stats)
    }
    
    if len(t_valid) == 0:
        return analysis
    
    # Descriptive statistics
    analysis.update({
        'mean': float(t_valid.mean()),
        'median': float(np.median(t_valid)),
        'std': float(t_valid.std()),
        'min': float(t_valid.min()),
        'max': float(t_valid.max()),
        'q25': float(np.percentile(t_valid, 25)),
        'q75': float(np.percentile(t_valid, 75)),
        'iqr': float(np.percentile(t_valid, 75) - np.percentile(t_valid, 25))
    })
    
    # Threshold analysis
    above_thresh = np.abs(t_valid) > threshold
    analysis.update({
        'n_above_threshold': int(np.sum(above_thresh)),
        'pct_above_threshold': 100 * np.sum(above_thresh) / len(t_valid),
        'n_positive': int(np.sum(t_valid > threshold)),
        'n_negative': int(np.sum(t_valid < -threshold)),
        'pct_positive': 100 * np.sum(t_valid > threshold) / len(t_valid),
        'pct_negative': 100 * np.sum(t_valid < -threshold) / len(t_valid)
    })
    
    # Extreme values
    extreme = np.abs(t_valid) > 10
    analysis.update({
        'n_extreme': int(np.sum(extreme)),
        'pct_extreme': 100 * np.sum(extreme) / len(t_valid),
        'max_abs': float(np.max(np.abs(t_valid)))
    })
    
    # Sign distribution
    analysis.update({
        'n_positive_all': int(np.sum(t_valid > 0)),
        'n_negative_all': int(np.sum(t_valid < 0)),
        'n_zero': int(np.sum(t_valid == 0)),
        'pct_positive_all': 100 * np.sum(t_valid > 0) / len(t_valid),
        'pct_negative_all': 100 * np.sum(t_valid < 0) / len(t_valid)
    })
    
    # Skewness and kurtosis
    analysis.update({
        'skewness': float(stats.skew(t_valid)),
        'kurtosis': float(stats.kurtosis(t_valid))
    })
    
    return analysis


def plot_t_distribution(t_stats, marker_name, output_path, threshold=2.5):
    """Plot t-statistic distribution for a single marker."""
    t_valid = t_stats[~np.isnan(t_stats)]
    
    if len(t_valid) == 0:
        print(f"  ⚠ No valid t-values for {marker_name}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    ax = axes[0]
    ax.hist(t_valid, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=1, label=f'Threshold (±{threshold})')
    ax.axvline(-threshold, color='orange', linestyle='--', linewidth=1)
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution\n(mean={t_valid.mean():.2f}, SD={t_valid.std():.2f})', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot (check normality)
    ax = axes[1]
    stats.probplot(t_valid, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot\n(check for normality)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Box plot
    ax = axes[2]
    bp = ax.boxplot(t_valid, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axhline(threshold, color='orange', linestyle='--', linewidth=1, label=f'Threshold')
    ax.axhline(-threshold, color='orange', linestyle='--', linewidth=1)
    ax.set_ylabel('T-statistic', fontsize=12)
    ax.set_title('Box Plot\n(outliers and spread)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{marker_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_markers(analyses_df, output_dir):
    """Compare t-statistics across markers."""
    if len(analyses_df) == 0:
        print("  ⚠ No analyses to compare")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Mean t-values
    ax = axes[0, 0]
    analyses_df_sorted = analyses_df.sort_values('mean')
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['mean'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Mean T-statistic')
    ax.set_title('Mean T-values by Marker')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Standard deviation
    ax = axes[0, 1]
    analyses_df_sorted = analyses_df.sort_values('std')
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['std'], color='coral', alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.set_xlabel('SD of T-statistics')
    ax.set_title('Variability by Marker')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Percentage above threshold
    ax = axes[0, 2]
    analyses_df_sorted = analyses_df.sort_values('pct_above_threshold', ascending=False)
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['pct_above_threshold'], color='green', alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.set_xlabel('% Channels Above Threshold')
    ax.set_title('Significant Channels by Marker')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Extreme values
    ax = axes[1, 0]
    analyses_df_sorted = analyses_df.sort_values('pct_extreme', ascending=False)
    colors = ['red' if x > 5 else 'orange' if x > 1 else 'green' for x in analyses_df_sorted['pct_extreme']]
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['pct_extreme'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.set_xlabel('% Extreme Values (|t| > 10)')
    ax.set_title('Extreme T-values (Potential Issues)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. Skewness
    ax = axes[1, 1]
    analyses_df_sorted = analyses_df.sort_values('skewness')
    colors = ['red' if abs(x) > 2 else 'orange' if abs(x) > 1 else 'green' for x in analyses_df_sorted['skewness']]
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['skewness'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Skewness')
    ax.set_title('Distribution Asymmetry')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 6. Max absolute t-value
    ax = axes[1, 2]
    analyses_df_sorted = analyses_df.sort_values('max_abs', ascending=False)
    colors = ['red' if x > 20 else 'orange' if x > 10 else 'green' for x in analyses_df_sorted['max_abs']]
    ax.barh(range(len(analyses_df_sorted)), analyses_df_sorted['max_abs'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(analyses_df_sorted)))
    ax.set_yticklabels(analyses_df_sorted['marker'], fontsize=8)
    ax.set_xlabel('Max |T-statistic|')
    ax.set_title('Maximum Absolute T-value')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'marker_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir / 'marker_comparison.png'}")
    plt.close()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run comprehensive t-value analysis on existing results."""
    print("\n" + "="*70)
    print("T-VALUE ANALYSIS FROM EXISTING RESULTS")
    print("="*70)
    print(f"\nResults directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Collect all analyses
    all_analyses = []
    
    for marker_type in MARKER_TYPES:
        print(f"\n{'='*70}")
        print(f"ANALYZING {marker_type.upper()} MARKERS")
        print(f"{'='*70}")
        
        # Find all marker directories
        marker_dirs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{marker_type}_")])
        
        print(f"Found {len(marker_dirs)} {marker_type} markers")
        
        for marker_dir in marker_dirs:
            marker_name = marker_dir.name
            print(f"\n  Processing: {marker_name}")
            
            # Load results
            results = load_marker_results(marker_dir)
            if results is None:
                print(f"    ⚠ No results found")
                continue
            
            # Extract t-statistics
            if 'observed_t_stats' not in results:
                print(f"    ⚠ No t-statistics in results")
                continue
            
            t_stats = results['observed_t_stats']
            print(f"    ✓ Loaded t-statistics: {len(t_stats)} channels")
            
            # Analyze
            analysis = analyze_t_statistics(t_stats, marker_name)
            all_analyses.append(analysis)
            
            # Print summary
            print(f"    Mean t: {analysis['mean']:.3f} ± {analysis['std']:.3f}")
            print(f"    Range: [{analysis['min']:.3f}, {analysis['max']:.3f}]")
            print(f"    Above threshold: {analysis['n_above_threshold']}/{analysis['n_valid']} ({analysis['pct_above_threshold']:.1f}%)")
            
            if analysis['pct_extreme'] > 0:
                print(f"    ⚠ EXTREME VALUES: {analysis['n_extreme']}/{analysis['n_valid']} ({analysis['pct_extreme']:.1f}%)")
            
            # Plot if in sample markers
            if marker_name in SAMPLE_MARKERS.get(marker_type, []):
                plot_path = OUTPUT_DIR / f"{marker_name}_distribution.png"
                plot_t_distribution(t_stats, marker_name, plot_path)
                print(f"    ✓ Saved plot: {plot_path.name}")
    
    # Create summary DataFrame
    if len(all_analyses) > 0:
        print(f"\n{'='*70}")
        print("CREATING SUMMARY")
        print(f"{'='*70}")
        
        df_analyses = pd.DataFrame(all_analyses)
        
        # Save summary
        summary_file = OUTPUT_DIR / 'summary_statistics.csv'
        df_analyses.to_csv(summary_file, index=False)
        print(f"\n✓ Saved summary: {summary_file}")
        
        # Create comparison plots
        print(f"\nGenerating comparison plots...")
        compare_markers(df_analyses, OUTPUT_DIR)
        
        # Print overall statistics
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Total markers analyzed: {len(df_analyses)}")
        print(f"\nMean t-statistics:")
        print(f"  Overall mean: {df_analyses['mean'].mean():.3f}")
        print(f"  Overall SD: {df_analyses['std'].mean():.3f}")
        print(f"  Range: [{df_analyses['mean'].min():.3f}, {df_analyses['mean'].max():.3f}]")
        
        print(f"\nExtreme values:")
        n_markers_with_extremes = (df_analyses['pct_extreme'] > 0).sum()
        print(f"  Markers with extreme values: {n_markers_with_extremes}/{len(df_analyses)} ({100*n_markers_with_extremes/len(df_analyses):.1f}%)")
        if n_markers_with_extremes > 0:
            print(f"  ⚠ WARNING: {n_markers_with_extremes} markers have extreme t-values (|t| > 10)")
            print(f"  This may indicate:")
            print(f"    - Convergence issues in LMM")
            print(f"    - Scaling/normalization problems")
            print(f"    - Overfitting")
            print(f"    - Data quality issues")
        
        print(f"\nSignificant channels (|t| > 2.5):")
        print(f"  Mean across markers: {df_analyses['pct_above_threshold'].mean():.1f}%")
        print(f"  Range: [{df_analyses['pct_above_threshold'].min():.1f}%, {df_analyses['pct_above_threshold'].max():.1f}%]")
        
        # Identify problematic markers
        print(f"\n{'='*70}")
        print("PROBLEMATIC MARKERS")
        print(f"{'='*70}")
        
        # Extreme values
        extreme_markers = df_analyses[df_analyses['pct_extreme'] > 5].sort_values('pct_extreme', ascending=False)
        if len(extreme_markers) > 0:
            print(f"\nMarkers with >5% extreme values:")
            for _, row in extreme_markers.iterrows():
                print(f"  - {row['marker']}: {row['pct_extreme']:.1f}% extreme, max |t|={row['max_abs']:.1f}")
        
        # High skewness
        skewed_markers = df_analyses[df_analyses['skewness'].abs() > 2].sort_values('skewness', key=abs, ascending=False)
        if len(skewed_markers) > 0:
            print(f"\nMarkers with high skewness (|skew| > 2):")
            for _, row in skewed_markers.iterrows():
                print(f"  - {row['marker']}: skewness={row['skewness']:.2f}")
        
        # Very high SD
        high_sd_markers = df_analyses[df_analyses['std'] > 5].sort_values('std', ascending=False)
        if len(high_sd_markers) > 0:
            print(f"\nMarkers with very high SD (>5):")
            for _, row in high_sd_markers.iterrows():
                print(f"  - {row['marker']}: SD={row['std']:.2f}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nRecommendations:")
    print("  1. Review markers with extreme t-values (|t| > 10)")
    print("  2. Check for convergence issues in LMM diagnostics")
    print("  3. Consider adjusting normalization strategy")
    print("  4. Verify predictor scaling and variance")
    print("  5. Check for outliers in behavioral data")


if __name__ == "__main__":
    main()
