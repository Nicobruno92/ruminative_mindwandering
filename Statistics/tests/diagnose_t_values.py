"""
Diagnostic script to analyze t-value distributions and identify issues.

This script tests different configurations to understand why t-values may be
physiologically incoherent (e.g., too extreme, wrong sign, spatially inconsistent).

Key diagnostics:
1. T-value distributions across channels
2. Effect of normalization on t-values
3. Model convergence issues
4. Predictor variance and scaling
5. Comparison with simple t-tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import warnings
from scipy import stats
import sys

# Add Statistics directory to path
sys.path.insert(0, str(Path(__file__).parent))

from reader import load_all_probe_data, prepare_data_for_lmm, filter_subjects_by_variability
from lmm_model import run_lmm_per_channel
from helpers import normalize_by_subject, extract_all_formula_variables, apply_preprocessing

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to config file
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Test configurations
TEST_CONFIGS = {
    "baseline": {
        "normalize": False,
        "formula": "power ~ onoff + (1|subject)",
        "description": "No normalization, single predictor"
    },
    "normalized_zscore": {
        "normalize": True,
        "norm_method": "zscore",
        "channel_wise": False,
        "formula": "power ~ onoff + (1|subject)",
        "description": "Z-score normalization (global), single predictor"
    },
    "normalized_robust": {
        "normalize": True,
        "norm_method": "robust",
        "channel_wise": False,
        "formula": "power ~ onoff + (1|subject)",
        "description": "Robust normalization (global), single predictor"
    },
    "multiple_predictors": {
        "normalize": True,
        "norm_method": "zscore",
        "channel_wise": False,
        "formula": "power ~ onoff + time_on_task + valence + selfother + time + (1|subject)",
        "description": "Z-score normalization, multiple predictors"
    },
    "channel_wise": {
        "normalize": True,
        "norm_method": "zscore",
        "channel_wise": True,
        "formula": "power ~ onoff + (1|subject)",
        "description": "Z-score normalization (channel-wise), single predictor"
    }
}

# Marker to test (use a representative one)
TEST_MARKER = "EEG_psd_bands_spectralpower_alpha"  # Change if needed

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_simple_ttest(power_data, df_behavioral, predictor):
    """
    Compute simple t-test for comparison (ignoring repeated measures).
    
    This provides a baseline to compare against LMM t-values.
    """
    n_channels = power_data.shape[1]
    t_stats_simple = np.zeros(n_channels)
    p_values_simple = np.ones(n_channels)
    
    # Check if predictor is binary or continuous
    unique_vals = df_behavioral[predictor].dropna().unique()
    
    if len(unique_vals) == 2:
        # Binary predictor: independent t-test
        mask1 = df_behavioral[predictor] == unique_vals[0]
        mask2 = df_behavioral[predictor] == unique_vals[1]
        
        for ch_idx in range(n_channels):
            group1 = power_data[mask1, ch_idx]
            group2 = power_data[mask2, ch_idx]
            
            # Remove NaN
            group1 = group1[~np.isnan(group1)]
            group2 = group2[~np.isnan(group2)]
            
            if len(group1) > 1 and len(group2) > 1:
                t_stat, p_val = stats.ttest_ind(group1, group2)
                t_stats_simple[ch_idx] = t_stat
                p_values_simple[ch_idx] = p_val
    else:
        # Continuous predictor: correlation-based t-test
        predictor_vals = df_behavioral[predictor].values
        
        for ch_idx in range(n_channels):
            channel_vals = power_data[:, ch_idx]
            
            # Remove NaN
            valid_mask = ~(np.isnan(channel_vals) | np.isnan(predictor_vals))
            if np.sum(valid_mask) < 5:
                continue
            
            channel_clean = channel_vals[valid_mask]
            predictor_clean = predictor_vals[valid_mask]
            
            if np.std(channel_clean) > 0 and np.std(predictor_clean) > 0:
                # Pearson correlation
                r, p_val = stats.pearsonr(channel_clean, predictor_clean)
                # Convert to t-statistic: t = r * sqrt((n-2)/(1-r^2))
                n = len(channel_clean)
                if abs(r) < 0.9999:  # Avoid division by zero
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    t_stats_simple[ch_idx] = t_stat
                    p_values_simple[ch_idx] = p_val
    
    return t_stats_simple, p_values_simple


def analyze_predictor_distribution(df_behavioral, predictor):
    """Analyze predictor variable distribution."""
    print(f"\n{'='*70}")
    print(f"PREDICTOR ANALYSIS: {predictor}")
    print(f"{'='*70}")
    
    pred_vals = df_behavioral[predictor].dropna()
    
    print(f"  N observations: {len(pred_vals)}")
    print(f"  N unique values: {pred_vals.nunique()}")
    print(f"  Range: [{pred_vals.min():.3f}, {pred_vals.max():.3f}]")
    print(f"  Mean ± SD: {pred_vals.mean():.3f} ± {pred_vals.std():.3f}")
    print(f"  Median (IQR): {pred_vals.median():.3f} ({pred_vals.quantile(0.25):.3f}, {pred_vals.quantile(0.75):.3f})")
    
    # Check for outliers
    q1, q3 = pred_vals.quantile(0.25), pred_vals.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = pred_vals[(pred_vals < lower_bound) | (pred_vals > upper_bound)]
    print(f"  Outliers (IQR method): {len(outliers)} ({100*len(outliers)/len(pred_vals):.1f}%)")
    
    # Within-subject variance
    print(f"\n  Within-subject variance:")
    subject_stds = df_behavioral.groupby('subject')[predictor].std()
    print(f"    Mean within-subject SD: {subject_stds.mean():.3f}")
    print(f"    Range: [{subject_stds.min():.3f}, {subject_stds.max():.3f}]")
    print(f"    Subjects with SD < 5: {(subject_stds < 5).sum()}/{len(subject_stds)}")
    print(f"    Subjects with SD < 1: {(subject_stds < 1).sum()}/{len(subject_stds)}")


def analyze_t_distribution(t_stats, label, threshold=2.5):
    """Analyze t-statistic distribution."""
    # Remove NaN values
    t_valid = t_stats[~np.isnan(t_stats)]
    
    print(f"\n{'='*70}")
    print(f"T-STATISTIC DISTRIBUTION: {label}")
    print(f"{'='*70}")
    
    print(f"  N channels: {len(t_stats)}")
    print(f"  N valid (non-NaN): {len(t_valid)}")
    print(f"  N NaN: {np.sum(np.isnan(t_stats))}")
    
    if len(t_valid) == 0:
        print("  ⚠ WARNING: All t-values are NaN!")
        return
    
    print(f"\n  Descriptive statistics:")
    print(f"    Mean: {t_valid.mean():.3f}")
    print(f"    Median: {np.median(t_valid):.3f}")
    print(f"    SD: {t_valid.std():.3f}")
    print(f"    Range: [{t_valid.min():.3f}, {t_valid.max():.3f}]")
    print(f"    IQR: [{np.percentile(t_valid, 25):.3f}, {np.percentile(t_valid, 75):.3f}]")
    
    print(f"\n  Threshold analysis (|t| > {threshold}):")
    above_thresh = np.abs(t_valid) > threshold
    print(f"    N channels: {np.sum(above_thresh)} ({100*np.sum(above_thresh)/len(t_valid):.1f}%)")
    print(f"    Positive: {np.sum((t_valid > threshold))} ({100*np.sum(t_valid > threshold)/len(t_valid):.1f}%)")
    print(f"    Negative: {np.sum((t_valid < -threshold))} ({100*np.sum(t_valid < -threshold)/len(t_valid):.1f}%)")
    
    print(f"\n  Extreme values (|t| > 10):")
    extreme = np.abs(t_valid) > 10
    print(f"    N channels: {np.sum(extreme)} ({100*np.sum(extreme)/len(t_valid):.1f}%)")
    if np.sum(extreme) > 0:
        print(f"    Max |t|: {np.max(np.abs(t_valid)):.3f}")
        print(f"    ⚠ WARNING: Extreme t-values detected! May indicate:")
        print(f"      - Convergence issues")
        print(f"      - Scaling problems")
        print(f"      - Overfitting")
    
    print(f"\n  Sign distribution:")
    print(f"    Positive: {np.sum(t_valid > 0)} ({100*np.sum(t_valid > 0)/len(t_valid):.1f}%)")
    print(f"    Negative: {np.sum(t_valid < 0)} ({100*np.sum(t_valid < 0)/len(t_valid):.1f}%)")
    print(f"    Zero: {np.sum(t_valid == 0)} ({100*np.sum(t_valid == 0)/len(t_valid):.1f}%)")


def plot_t_distributions(results_dict, output_dir):
    """Plot t-statistic distributions for all configurations."""
    n_configs = len(results_dict)
    fig, axes = plt.subplots(2, (n_configs + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (config_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        t_stats = result['t_stats']
        t_valid = t_stats[~np.isnan(t_stats)]
        
        if len(t_valid) > 0:
            ax.hist(t_valid, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            ax.axvline(2.5, color='orange', linestyle='--', linewidth=1, label='Threshold')
            ax.axvline(-2.5, color='orange', linestyle='--', linewidth=1)
            
            ax.set_xlabel('T-statistic')
            ax.set_ylabel('Frequency')
            ax.set_title(f"{config_name}\n(mean={t_valid.mean():.2f}, SD={t_valid.std():.2f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid t-values', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
    
    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 't_distributions_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 't_distributions_comparison.png'}")
    plt.close()


def plot_t_comparison(results_dict, output_dir):
    """Plot comparison of t-values across configurations."""
    # Get baseline config
    if 'baseline' not in results_dict:
        print("⚠ Baseline configuration not found, skipping comparison plot")
        return
    
    baseline_t = results_dict['baseline']['t_stats']
    
    fig, axes = plt.subplots(1, len(results_dict) - 1, figsize=(15, 5))
    if len(results_dict) == 2:
        axes = [axes]
    
    plot_idx = 0
    for config_name, result in results_dict.items():
        if config_name == 'baseline':
            continue
        
        ax = axes[plot_idx]
        t_stats = result['t_stats']
        
        # Remove NaN from both
        valid_mask = ~(np.isnan(baseline_t) | np.isnan(t_stats))
        if np.sum(valid_mask) > 0:
            ax.scatter(baseline_t[valid_mask], t_stats[valid_mask], alpha=0.5, s=20)
            
            # Add diagonal line
            min_val = min(baseline_t[valid_mask].min(), t_stats[valid_mask].min())
            max_val = max(baseline_t[valid_mask].max(), t_stats[valid_mask].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
            
            # Compute correlation
            r, p = stats.pearsonr(baseline_t[valid_mask], t_stats[valid_mask])
            ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3e}', 
                   transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Baseline t-statistic')
            ax.set_ylabel(f'{config_name} t-statistic')
            ax.set_title(config_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No valid comparison', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(config_name)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_dir / 't_comparison_vs_baseline.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 't_comparison_vs_baseline.png'}")
    plt.close()


def plot_lmm_vs_ttest(t_lmm, t_simple, output_dir, config_name):
    """Compare LMM t-values with simple t-test."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Remove NaN
    valid_mask = ~(np.isnan(t_lmm) | np.isnan(t_simple))
    
    if np.sum(valid_mask) > 0:
        # Scatter plot
        ax = axes[0]
        ax.scatter(t_simple[valid_mask], t_lmm[valid_mask], alpha=0.5, s=20)
        
        # Add diagonal
        min_val = min(t_simple[valid_mask].min(), t_lmm[valid_mask].min())
        max_val = max(t_simple[valid_mask].max(), t_lmm[valid_mask].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        
        # Correlation
        r, p = stats.pearsonr(t_simple[valid_mask], t_lmm[valid_mask])
        ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3e}', 
               transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Simple t-test')
        ax.set_ylabel('LMM t-statistic')
        ax.set_title('LMM vs Simple T-test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Difference plot (Bland-Altman style)
        ax = axes[1]
        mean_t = (t_simple[valid_mask] + t_lmm[valid_mask]) / 2
        diff_t = t_lmm[valid_mask] - t_simple[valid_mask]
        
        ax.scatter(mean_t, diff_t, alpha=0.5, s=20)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.axhline(diff_t.mean(), color='blue', linestyle='-', linewidth=2, label=f'Mean diff={diff_t.mean():.3f}')
        ax.axhline(diff_t.mean() + 1.96*diff_t.std(), color='blue', linestyle='--', linewidth=1, label='±1.96 SD')
        ax.axhline(diff_t.mean() - 1.96*diff_t.std(), color='blue', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Mean t-statistic')
        ax.set_ylabel('Difference (LMM - Simple)')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'lmm_vs_ttest_{config_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'lmm_vs_ttest_{config_name}.png'}")
    plt.close()


# ============================================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================================

def run_diagnostics():
    """Run comprehensive diagnostics on t-value computation."""
    print("\n" + "="*70)
    print("T-VALUE DIAGNOSTIC ANALYSIS")
    print("="*70)
    
    # Load config
    config = load_config(CONFIG_PATH)
    
    # Create output directory (use local path)
    output_dir = Path(__file__).parent / 'diagnostics' / 't_value_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data for test marker
    print(f"\nLoading data for marker: {TEST_MARKER}")
    print(f"  Marker type: state")
    
    try:
        # Load all probe data
        df_all = load_all_probe_data(
            features_root=config['project']['features_root'],
            subjects=config['project']['subjects'],
            tasks=config['project']['tasks'],
            marker_types=['state']
        )
        
        # Filter by predictor variability if specified
        predictor = config['lmm']['predictor_of_interest']
        min_var = config['project'].get('min_predictor_variability')
        if min_var:
            df_all = filter_subjects_by_variability(
                df=df_all,
                predictor_column=predictor,
                min_variability=min_var,
                subject_column='subject'
            )
        
        # Prepare data for LMM
        formula = config['lmm']['formula']
        power_data, df_behavioral, ch_names = prepare_data_for_lmm(
            df=df_all,
            marker_name=TEST_MARKER,
            formula=formula
        )
        
        print(f"✓ Data loaded successfully")
        print(f"  Shape: {power_data.shape}")
        print(f"  Channels: {len(ch_names)}")
        print(f"  Observations: {len(df_behavioral)}")
        print(f"  Subjects: {df_behavioral['subject'].nunique()}")
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze predictor distribution
    predictor = config['lmm']['predictor_of_interest']
    analyze_predictor_distribution(df_behavioral, predictor)
    
    # Store results for all configurations
    results = {}
    
    # Test each configuration
    for config_name, test_config in TEST_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"TESTING CONFIGURATION: {config_name}")
        print(f"Description: {test_config['description']}")
        print(f"{'='*70}")
        
        # Prepare data
        power_test = power_data.copy()
        df_test = df_behavioral.copy()
        
        # Apply normalization if specified
        if test_config.get('normalize', False):
            print(f"\nApplying normalization:")
            print(f"  Method: {test_config['norm_method']}")
            print(f"  Channel-wise: {test_config['channel_wise']}")
            
            power_test = normalize_by_subject(
                power_data=power_test,
                df_behavioral=df_test,
                method=test_config['norm_method'],
                subject_col='subject',
                channel_wise=test_config['channel_wise']
            )
            print(f"  ✓ Normalization applied")
        
        # Run LMM
        print(f"\nRunning LMM:")
        print(f"  Formula: {test_config['formula']}")
        
        try:
            t_stats, p_values, diagnostics = run_lmm_per_channel(
                power_data=power_test,
                df_behavioral=df_test,
                formula=test_config['formula'],
                predictor_of_interest=predictor,
                method=config['lmm']['method'],
                maxiter=config['lmm']['maxiter'],
                random_state=config['lmm']['random_state'],
                return_diagnostics=True
            )
            
            print(f"  ✓ LMM completed")
            
            # Analyze t-distribution
            analyze_t_distribution(t_stats, config_name)
            
            # Compute simple t-test for comparison (only for baseline)
            if config_name == 'baseline':
                print(f"\nComputing simple t-test for comparison...")
                t_simple, p_simple = compute_simple_ttest(power_test, df_test, predictor)
                analyze_t_distribution(t_simple, "Simple T-test")
                
                # Plot comparison
                plot_lmm_vs_ttest(t_stats, t_simple, output_dir, config_name)
            
            # Store results
            results[config_name] = {
                't_stats': t_stats,
                'p_values': p_values,
                'diagnostics': diagnostics,
                'config': test_config
            }
            
        except Exception as e:
            print(f"  ✗ LMM failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Plot comparisons
    if len(results) > 0:
        print(f"\n{'='*70}")
        print("GENERATING COMPARISON PLOTS")
        print(f"{'='*70}")
        
        plot_t_distributions(results, output_dir)
        plot_t_comparison(results, output_dir)
    
    # Save summary
    summary_file = output_dir / 'diagnostic_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("T-VALUE DIAGNOSTIC SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Marker: {TEST_MARKER}\n")
        f.write(f"Predictor: {predictor}\n")
        f.write(f"N configurations tested: {len(results)}\n\n")
        
        for config_name, result in results.items():
            t_stats = result['t_stats']
            t_valid = t_stats[~np.isnan(t_stats)]
            
            f.write(f"\n{config_name}:\n")
            f.write(f"  Description: {result['config']['description']}\n")
            if len(t_valid) > 0:
                f.write(f"  Mean t: {t_valid.mean():.3f}\n")
                f.write(f"  SD t: {t_valid.std():.3f}\n")
                f.write(f"  Range: [{t_valid.min():.3f}, {t_valid.max():.3f}]\n")
                f.write(f"  |t| > 2.5: {np.sum(np.abs(t_valid) > 2.5)} ({100*np.sum(np.abs(t_valid) > 2.5)/len(t_valid):.1f}%)\n")
                f.write(f"  |t| > 10: {np.sum(np.abs(t_valid) > 10)} ({100*np.sum(np.abs(t_valid) > 10)/len(t_valid):.1f}%)\n")
            else:
                f.write(f"  No valid t-values\n")
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    print(f"\n{'='*70}")
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}")
    print("\nRecommendations:")
    print("  1. Check t-value distributions for extreme values (|t| > 10)")
    print("  2. Compare normalized vs non-normalized results")
    print("  3. Check if LMM t-values correlate with simple t-test")
    print("  4. Look for signs of convergence issues in diagnostics")
    print("  5. Consider using robust normalization if outliers present")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    run_diagnostics()
