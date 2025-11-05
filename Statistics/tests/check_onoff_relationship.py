"""
Quick diagnostic script to check the relationship between onoff and power.
This helps understand why LMM t-values don't match High-Low differences.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from reader import load_all_probe_data, prepare_data_for_lmm

def check_onoff_relationship():
    """Check relationship between onoff and power for a specific marker."""
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    df_all = load_all_probe_data(
        features_root=config['project']['features_root'],
        subjects=config['project']['subjects'],
        tasks=config['project']['tasks'],
        marker_types=config['project']['marker_types'],
        qa_exclusions=None,
        verbose=True
    )
    
    # No PCA needed for this diagnostic
    pca_data = None
    
    # Select a marker to analyze (use power_normalized_delta from your example)
    marker_name = 'power_normalized_gamma'
    marker_type = 'state'
    
    print(f"\nAnalyzing marker: {marker_name} ({marker_type})")
    
    # Filter for this marker
    df_marker = df_all[
        (df_all['marker_type'] == marker_type) & 
        (df_all['marker'] == marker_name)
    ]
    
    # Prepare data
    power_data, df_behavioral, channels = prepare_data_for_lmm(
        df=df_marker,
        marker_name=marker_name,
        formula=config['lmm']['formula'],
        include_channels=None,
        exclude_channels=None,
        pca_data=pca_data,
        onoff_max_value=config['project'].get('onoff_max_value')
    )
    
    print(f"Data shape: {power_data.shape}")
    print(f"Channels: {len(channels)}")
    print(f"Observations: {len(df_behavioral)}")
    
    # Get onoff values
    onoff = df_behavioral['onoff'].values
    
    print(f"\nOnoff statistics:")
    print(f"  Range: [{np.min(onoff):.1f}, {np.max(onoff):.1f}]")
    print(f"  Mean: {np.mean(onoff):.1f}")
    print(f"  Median: {np.median(onoff):.1f}")
    print(f"  Std: {np.std(onoff):.1f}")
    
    # Check distribution
    median_onoff = np.median(onoff)
    high_mask = onoff >= median_onoff
    low_mask = onoff < median_onoff
    
    print(f"\nHigh/Low split:")
    print(f"  High group: {np.sum(high_mask)} observations (onoff >= {median_onoff:.1f})")
    print(f"  Low group: {np.sum(low_mask)} observations (onoff < {median_onoff:.1f})")
    
    # Select a few representative channels
    example_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # Different spatial locations
    
    rows = np.ceil(len(example_channels) / 5).astype(int)
    columns = min(5, len(example_channels))
    
    fig, axes = plt.subplots(rows, columns, figsize=(10 * columns, 5 * rows))
    axes = np.ravel(axes)[:len(example_channels)]
    
    for idx, ch_idx in enumerate(example_channels):
        if ch_idx >= len(channels):
            continue
            
        ax = axes[idx]
        ch_name = channels[ch_idx]
        power_ch = power_data[:, ch_idx]
        
        # Scatter plot
        ax.scatter(onoff, power_ch, alpha=0.3, s=10)
        
        # Fit linear regression
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(onoff, power_ch)
        
        # Plot regression line
        onoff_range = np.array([np.min(onoff), np.max(onoff)])
        ax.plot(onoff_range, slope * onoff_range + intercept, 
                'r-', linewidth=2, label=f'slope={slope:.3f}')
        
        # Add median split lines
        high_mean = np.mean(power_ch[high_mask])
        low_mean = np.mean(power_ch[low_mask])
        
        ax.axvline(median_onoff, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(high_mean, color='blue', linestyle='--', alpha=0.5, 
                   label=f'High mean={high_mean:.2f}')
        ax.axhline(low_mean, color='orange', linestyle='--', alpha=0.5,
                   label=f'Low mean={low_mean:.2f}')
        
        ax.set_xlabel('onoff')
        ax.set_ylabel('Power (normalized)')
        ax.set_title(f'{ch_name}\nr={r_value:.2f}, p={p_value:.3f}\n'
                     f'High-Low diff={high_mean-low_mean:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('results/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{marker_name}_onoff_relationship.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to {output_path}")
    
    # Compute correlations for all channels
    correlations = []
    high_low_diffs = []
    
    for ch_idx in range(len(channels)):
        power_ch = power_data[:, ch_idx]
        
        # Linear correlation (what LMM captures)
        corr, p = np.corrcoef(onoff, power_ch)[0, 1], 0
        correlations.append(corr)
        
        # High-Low difference (what raw comparison shows)
        high_mean = np.mean(power_ch[high_mask])
        low_mean = np.mean(power_ch[low_mask])
        high_low_diffs.append(high_mean - low_mean)
    
    correlations = np.array(correlations)
    high_low_diffs = np.array(high_low_diffs)
    
    print(f"\nCorrelation between LMM approach and High-Low approach:")
    print(f"  Correlation (onoff-power): range [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
    print(f"  High-Low differences: range [{np.min(high_low_diffs):.3f}, {np.max(high_low_diffs):.3f}]")
    print(f"  Correlation between approaches: {np.corrcoef(correlations, high_low_diffs)[0,1]:.3f}")
    
    # Check if they have same sign
    same_sign = np.sum(np.sign(correlations) == np.sign(high_low_diffs))
    print(f"  Channels with same sign: {same_sign}/{len(channels)} ({100*same_sign/len(channels):.1f}%)")
    
    return correlations, high_low_diffs, channels


if __name__ == '__main__':
    correlations, high_low_diffs, channels = check_onoff_relationship()
