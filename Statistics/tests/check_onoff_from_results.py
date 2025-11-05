"""
Quick diagnostic to understand why LMM t-values look uniform.
Loads from existing results instead of reprocessing data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def check_results():
    """Load existing results and analyze the relationship."""
    
    # Path to your existing results
    results_path = Path('/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff/state_power_normalized_delta/results.pkl')
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("\nAvailable result directories:")
        base_path = Path('/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster/onoff')
        if base_path.exists():
            for d in sorted(base_path.iterdir()):
                if d.is_dir():
                    print(f"  - {d.name}")
        return
    
    print(f"Loading results from: {results_path}")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    # Extract key information
    t_stats = results['t_stats']
    ch_names = results['ch_names']
    clusters = results['clusters']
    cluster_p_values = results['cluster_p_values']
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Marker: {results['marker_name']} ({results['marker_type']})")
    print(f"Formula: {results['formula']}")
    print(f"Predictor: {results['predictor_of_interest']}")
    print(f"N subjects: {results['n_subjects']}")
    print(f"N observations: {results['n_observations']}")
    print(f"N channels: {len(ch_names)}")
    
    print(f"\n{'='*60}")
    print("T-STATISTICS DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Range: [{np.min(t_stats):.3f}, {np.max(t_stats):.3f}]")
    print(f"Mean: {np.mean(t_stats):.3f}")
    print(f"Median: {np.median(t_stats):.3f}")
    print(f"Std: {np.std(t_stats):.3f}")
    
    # Count by sign
    n_positive = np.sum(t_stats > 0)
    n_negative = np.sum(t_stats < 0)
    n_near_zero = np.sum(np.abs(t_stats) < 0.5)
    
    print(f"\nSign distribution:")
    print(f"  Positive: {n_positive}/{len(t_stats)} ({100*n_positive/len(t_stats):.1f}%)")
    print(f"  Negative: {n_negative}/{len(t_stats)} ({100*n_negative/len(t_stats):.1f}%)")
    print(f"  Near zero (|t| < 0.5): {n_near_zero}/{len(t_stats)} ({100*n_near_zero/len(t_stats):.1f}%)")
    
    # Magnitude distribution
    threshold = results.get('threshold', 2.0)
    n_above_threshold = np.sum(np.abs(t_stats) > threshold)
    print(f"\nMagnitude:")
    print(f"  |t| > {threshold}: {n_above_threshold}/{len(t_stats)} ({100*n_above_threshold/len(t_stats):.1f}%)")
    
    print(f"\n{'='*60}")
    print("CLUSTER RESULTS")
    print(f"{'='*60}")
    print(f"N clusters: {len(clusters)}")
    print(f"N significant (p < 0.05): {np.sum(cluster_p_values < 0.05)}")
    
    if len(clusters) > 0:
        print(f"\nCluster sizes:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {len(cluster)} channels (p={cluster_p_values[i]:.4f})")
            if len(cluster) <= 10:
                cluster_names = [ch_names[idx] for idx in cluster]
                print(f"    Channels: {', '.join(cluster_names)}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of t-values
    ax = axes[0, 0]
    ax.hist(t_stats, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold (±{threshold})')
    ax.axvline(-threshold, color='orange', linestyle='--', linewidth=2)
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of T-statistics', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. T-values by channel (sorted)
    ax = axes[0, 1]
    sorted_idx = np.argsort(t_stats)
    ax.plot(range(len(t_stats)), t_stats[sorted_idx], 'o-', markersize=4)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.axhline(threshold, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(-threshold, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Channel (sorted by t-value)', fontsize=12)
    ax.set_ylabel('T-statistic', fontsize=12)
    ax.set_title('T-statistics (sorted)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Channels with highest |t|
    ax = axes[1, 0]
    abs_t = np.abs(t_stats)
    top_idx = np.argsort(abs_t)[-15:][::-1]  # Top 15
    
    y_pos = np.arange(len(top_idx))
    colors = ['blue' if t_stats[i] > 0 else 'red' for i in top_idx]
    ax.barh(y_pos, t_stats[top_idx], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([ch_names[i] for i in top_idx], fontsize=9)
    ax.set_xlabel('T-statistic', fontsize=12)
    ax.set_title('Top 15 Channels by |t|', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Text summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
INTERPRETATION:

Why do t-values look "uniform"?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MOSTLY POSITIVE ({n_positive}/{len(t_stats)} channels)
   → Positive relationship between onoff and power
   → Higher onoff → Higher power (on average)

2. RANGE: {np.min(t_stats):.2f} to {np.max(t_stats):.2f}
   → Not actually uniform!
   → But mostly in same direction

3. GIANT CLUSTER ({len(clusters[0]) if len(clusters) > 0 else 0} channels)
   → Many adjacent channels > threshold
   → With only 50 permutations, hard to reject
   → Need 5000+ permutations for reliable results

4. LMM vs HIGH-LOW DIFFERENCE:
   → LMM: Linear trend across all onoff values
   → High-Low: Difference between extremes
   → Can show different patterns!

SOLUTION:
✓ Use TFCE (threshold-free)
✓ Increase permutations to 5000
✓ Both already fixed in your config!
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('results/diagnostics')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{results['marker_name']}_t_statistics_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plot saved to {output_path}")
    
    plt.show()
    
    return results


if __name__ == '__main__':
    results = check_results()
