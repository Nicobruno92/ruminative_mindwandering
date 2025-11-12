"""
Simplified diagnostic for "bubbly" t-statistics.

This version works with already-computed results to diagnose spatial coherence issues.
Run this on your local machine after downloading results from the cluster.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle


def analyze_tstat_spatial_coherence(t_stats: np.ndarray, adjacency, ch_names: list, 
                                     marker_name: str, output_dir: Path):
    """
    Analyze spatial coherence of t-statistics.
    
    Checks:
    1. Correlation between neighboring electrodes
    2. Spatial autocorrelation
    3. Sign consistency in neighborhoods
    """
    print("\n" + "="*70)
    print(f"SPATIAL COHERENCE ANALYSIS: {marker_name}")
    print("="*70)
    
    n_channels = len(t_stats)
    
    # Remove NaN values
    valid_mask = ~np.isnan(t_stats)
    n_valid = np.sum(valid_mask)
    
    print(f"\nValid channels: {n_valid}/{n_channels} ({100*n_valid/n_channels:.1f}%)")
    
    if n_valid < 10:
        print("⚠ Too few valid channels for spatial analysis")
        return
    
    # Convert adjacency to dense if sparse
    from scipy.sparse import issparse
    if issparse(adjacency):
        adj_dense = adjacency.toarray()
    else:
        adj_dense = adjacency
    
    # 1. Neighbor correlation analysis
    print("\n" + "-"*70)
    print("NEIGHBOR CORRELATION")
    print("-"*70)
    
    neighbor_correlations = []
    sign_agreements = []
    
    for i in range(n_channels):
        if not valid_mask[i]:
            continue
        
        # Find neighbors
        neighbors = np.where(adj_dense[i, :] > 0)[0]
        valid_neighbors = neighbors[valid_mask[neighbors]]
        
        if len(valid_neighbors) > 0:
            # Correlation with neighbors
            neighbor_t = t_stats[valid_neighbors]
            my_t = t_stats[i]
            
            # Simple correlation: do neighbors have similar values?
            for neighbor_val in neighbor_t:
                neighbor_correlations.append(my_t * neighbor_val)  # Product (positive if same sign)
                sign_agreements.append(np.sign(my_t) == np.sign(neighbor_val))
    
    if neighbor_correlations:
        avg_neighbor_corr = np.mean(neighbor_correlations)
        pct_sign_agreement = 100 * np.mean(sign_agreements)
        
        print(f"\nAverage neighbor correlation: {avg_neighbor_corr:.3f}")
        print(f"Sign agreement with neighbors: {pct_sign_agreement:.1f}%")
        
        if avg_neighbor_corr < 0:
            print("\n⚠ CRITICAL: Negative neighbor correlation!")
            print("  → Neighboring electrodes have OPPOSITE effects")
            print("  → This is the 'bubbly' pattern - spatially incoherent")
        elif avg_neighbor_corr < 0.3:
            print("\n⚠ WARNING: Low neighbor correlation")
            print("  → Weak spatial coherence")
        else:
            print("\n✓ Good spatial coherence")
        
        if pct_sign_agreement < 60:
            print("\n⚠ CRITICAL: Low sign agreement!")
            print("  → Checkerboard pattern - neighboring electrodes flip signs")
    
    # 2. Global spatial autocorrelation (Moran's I approximation)
    print("\n" + "-"*70)
    print("SPATIAL AUTOCORRELATION (Moran's I)")
    print("-"*70)
    
    # Simple Moran's I calculation
    valid_t = t_stats[valid_mask]
    t_mean = np.mean(valid_t)
    t_centered = valid_t - t_mean
    
    # Get adjacency for valid channels only
    valid_indices = np.where(valid_mask)[0]
    adj_valid = adj_dense[np.ix_(valid_indices, valid_indices)]
    
    # Moran's I = (N/W) * sum(w_ij * (x_i - mean) * (x_j - mean)) / sum((x_i - mean)^2)
    W = np.sum(adj_valid)  # Total weight
    numerator = 0
    for i in range(len(valid_t)):
        for j in range(len(valid_t)):
            numerator += adj_valid[i, j] * t_centered[i] * t_centered[j]
    
    denominator = np.sum(t_centered ** 2)
    
    if denominator > 0 and W > 0:
        morans_i = (len(valid_t) / W) * (numerator / denominator)
        print(f"\nMoran's I: {morans_i:.3f}")
        print("  Range: [-1, 1]")
        print("  -1: Perfect dispersion (checkerboard)")
        print("   0: Random spatial pattern")
        print("  +1: Perfect clustering (smooth)")
        
        if morans_i < -0.2:
            print("\n⚠ CRITICAL: Negative spatial autocorrelation!")
            print("  → Checkerboard/bubbly pattern confirmed")
        elif morans_i < 0.2:
            print("\n⚠ WARNING: Low spatial autocorrelation")
            print("  → Weak spatial structure")
        else:
            print("\n✓ Positive spatial autocorrelation - good coherence")
    
    # 3. Visualize
    print("\n" + "-"*70)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("-"*70)
    
    # Plot histogram of neighbor correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(neighbor_correlations, bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--', label='Zero correlation')
    axes[0].axvline(np.mean(neighbor_correlations), color='g', linestyle='-', 
                    linewidth=2, label=f'Mean: {np.mean(neighbor_correlations):.3f}')
    axes[0].set_xlabel('Neighbor correlation (t_i × t_neighbor)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Neighbor Correlations')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot t-stat distribution
    axes[1].hist(valid_t, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', label='Zero')
    axes[1].axvline(np.mean(valid_t), color='g', linestyle='-', 
                    linewidth=2, label=f'Mean: {np.mean(valid_t):.3f}')
    axes[1].set_xlabel('T-statistic')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of T-statistics')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f"{marker_name}_spatial_coherence.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved: {plot_path}")
    plt.close()


def main():
    """
    Analyze spatial coherence from saved results.
    
    Usage:
    1. Download your results pickle file from cluster
    2. Update the path below
    3. Run this script
    """
    print("="*70)
    print("SPATIAL COHERENCE DIAGNOSTIC")
    print("="*70)
    
    # UPDATE THIS PATH to your results file
    results_dir = Path("/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/lmm_cluster")
    
    # Find a results pickle file
    pickle_files = list(results_dir.glob("**/results.pkl"))
    
    if not pickle_files:
        print("\n⚠ No results files found!")
        print(f"   Searched in: {results_dir}")
        print("\nTo use this diagnostic:")
        print("1. Download a results .pkl file from the cluster")
        print("2. Update the 'results_dir' path in this script")
        print("3. Run again")
        return
    
    print(f"\nFound {len(pickle_files)} result files")
    
    # Analyze multiple markers to compare
    markers_to_analyze = [
        'power_normalized_delta',
        'power_normalized_theta', 
        'power_normalized_alpha',
        'power_normalized_beta',
        'power_normalized_gamma',
        'wsmi_alpha',
        'wsmi_beta',
        'wsmi_gamma',
        'wsmi_theta'
    ]
    
    analyzed_count = 0
    for results_file in pickle_files:
        # Check if this is one of the markers we want
        marker_match = any(marker in str(results_file) for marker in markers_to_analyze)
        if not marker_match:
            continue
        
        analyzed_count += 1
        if analyzed_count > 5:  # Limit to 5 markers
            break
            
        print(f"\n{'='*70}")
        print(f"Analyzing: {results_file.parent.name}/{results_file.name}")
        print(f"{'='*70}")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        # Extract data
        t_stats = results['t_stats']
        ch_names = results['ch_names']
        marker_name = results.get('marker_name', 'unknown')
        
        # Get adjacency from MNE info object
        import mne
        info = results['info']
        adjacency, _ = mne.channels.find_ch_adjacency(info, ch_type='eeg')
        
        print(f"  T-stats shape: {t_stats.shape}")
        print(f"  Channels: {len(ch_names)}")
        print(f"  Adjacency: {adjacency.shape}")
        
        # Create output directory
        output_dir = Path(__file__).parent / "diagnostics" / "spatial_coherence"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        analyze_tstat_spatial_coherence(t_stats, adjacency, ch_names, marker_name, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
