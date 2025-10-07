import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.stats import spatio_temporal_cluster_test
from scipy import stats
import os
import sys
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(script_dir)


def prepare_data_for_cluster_test(df, marker, condition_col='onoff_label', 
                                 condition_high='high', condition_low='low'):
    """
    Prepare data for cluster-based permutation testing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Aggregated marker data
    marker : str
        The marker to analyze
    condition_col : str
        The condition column name
    condition_high : str
        Label for high condition
    condition_low : str
        Label for low condition
        
    Returns
    -------
    X : array, shape (n_subjects, n_channels, n_times)
        Data array for cluster test (n_times=1 for our case)
    ch_names : list
        Channel names
    high_mean : array
        Mean values for high condition
    low_mean : array
        Mean values for low condition
    """
    # Filter data for the specific marker
    marker_data = df[df['marker'] == marker].copy()
    
    if marker_data.empty:
        return None, None, None, None
    
    # Separate conditions
    high_data = marker_data[marker_data[condition_col] == condition_high]
    low_data = marker_data[marker_data[condition_col] == condition_low]
    
    if high_data.empty or low_data.empty:
        return None, None, None, None
    
    # Get common channels
    high_channels = set(high_data['channel'].unique())
    low_channels = set(low_data['channel'].unique())
    common_channels = sorted(list(high_channels.intersection(low_channels)))
    
    if len(common_channels) == 0:
        return None, None, None, None
    
    # Get common subjects
    high_subjects = set(high_data['subject_id'].unique())
    low_subjects = set(low_data['subject_id'].unique())
    common_subjects = sorted(list(high_subjects.intersection(low_subjects)))
    
    if len(common_subjects) == 0:
        return None, None, None, None
    
    # Prepare data arrays
    n_subjects = len(common_subjects)
    n_channels = len(common_channels)
    
    # Initialize arrays
    high_array = np.zeros((n_subjects, n_channels, 1))  # 1 time point
    low_array = np.zeros((n_subjects, n_channels, 1))
    
    # Initialize arrays for mean values
    high_mean = np.zeros(n_channels)
    low_mean = np.zeros(n_channels)
    
    # Fill arrays
    for i, subject in enumerate(common_subjects):
        for j, channel in enumerate(common_channels):
            # High condition
            high_val = high_data[
                (high_data['subject_id'] == subject) & 
                (high_data['channel'] == channel)
            ]['mean'].values
            
            # Low condition
            low_val = low_data[
                (low_data['subject_id'] == subject) & 
                (low_data['channel'] == channel)
            ]['mean'].values
            
            if len(high_val) > 0:
                high_array[i, j, 0] = high_val[0]
            if len(low_val) > 0:
                low_array[i, j, 0] = low_val[0]
    
    # Calculate mean values across subjects
    high_mean = np.mean(high_array, axis=0).squeeze()
    low_mean = np.mean(low_array, axis=0).squeeze()
    
    # Combine into single array for cluster test
    X = [high_array, low_array]
    
    return X, common_channels, high_mean, low_mean


def run_cluster_permutation_test(X, ch_names, adjacency=None, n_permutations=1000,
                                threshold=None, tail=0, alpha=0.05):
    """
    Run cluster-based permutation test.
    
    Parameters
    ----------
    X : list of arrays
        Data for each condition [condition1, condition2]
    ch_names : list
        Channel names
    adjacency : array or None
        Channel adjacency matrix
    n_permutations : int
        Number of permutations
    threshold : float or None
        Threshold for clustering
    tail : int
        Tail for test (-1, 0, 1)
    alpha : float
        Significance level
        
    Returns
    -------
    T_obs : array
        Observed test statistics
    clusters : list
        Found clusters
    cluster_pv : array
        P-values for clusters
    H0 : array
        Null distribution
    """
    try:
        # Run cluster test
        T_obs, clusters, cluster_pv, H0 = spatio_temporal_cluster_test(
            X, adjacency=adjacency, n_permutations=n_permutations,
            threshold=threshold, tail=tail, n_jobs=1, verbose=False
        )
        
        return T_obs, clusters, cluster_pv, H0
        
    except Exception as e:
        print(f"Error in cluster test: {e}")
        return None, None, None, None


def get_channel_adjacency(ch_names, montage_name='standard_1020'):
    """
    Get channel adjacency matrix based on spatial proximity.
    
    Parameters
    ----------
    ch_names : list
        Channel names
    montage_name : str
        MNE montage name
        
    Returns
    -------
    adjacency : array
        Adjacency matrix
    """
    try:
        # Create info object
        info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
        montage = mne.channels.make_standard_montage(montage_name)
        info.set_montage(montage, on_missing='ignore')
        
        # Get adjacency matrix
        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(info, ch_type='eeg')
        
        return adjacency
        
    except Exception as e:
        print(f"Could not create adjacency matrix: {e}")
        return None


def plot_cluster_results(T_obs, clusters, cluster_pv, ch_names, marker,
                        condition_high, condition_low, alpha=0.05,
                        save_path=None, figsize=(15, 15), 
                        high_values=None, low_values=None):
    """
    Plot cluster-based permutation test results.
    
    Parameters
    ----------
    T_obs : array
        Observed test statistics
    clusters : list
        Found clusters
    cluster_pv : array
        P-values for clusters
    ch_names : list
        Channel names
    marker : str
        Marker name
    condition_high : str
        High condition label
    condition_low : str
        Low condition label
    alpha : float
        Significance level
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    high_values : array
        Mean values for high condition
    low_values : array
        Mean values for low condition
        
    Returns
    -------
    fig : matplotlib.Figure
        Figure object
    """
    # Find significant clusters
    significant_clusters = []
    for i, pv in enumerate(cluster_pv):
        if pv < alpha:
            significant_clusters.append((i, pv, clusters[i]))
    
    # Create figure with a 3x3 grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    
    # Main title
    fig.suptitle(f'Cluster-based Permutation Test: {marker}\n'
                 f'{condition_high} vs {condition_low}', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 0:
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Filter T_obs for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            T_obs_filtered = T_obs[ch_indices, 0] if len(ch_indices) > 0 else T_obs[:, 0]
            
            # Calculate difference for available channels
            if high_values is not None and low_values is not None:
                high_filtered = np.array([high_values[ch_names.index(ch)] for ch in available_channels if ch in ch_names])
                low_filtered = np.array([low_values[ch_names.index(ch)] for ch in available_channels if ch in ch_names])
                diff_filtered = high_filtered - low_filtered
            else:
                high_filtered = None
                low_filtered = None
                diff_filtered = None
            
            # First row: Raw values - high, low, and difference
            if high_filtered is not None and low_filtered is not None:
                # Plot high condition (ON-task)
                ax_high = fig.add_subplot(gs[0, 0])
                vmin_high = np.min(high_filtered)
                vmax_high = np.max(high_filtered)
                im_high, _ = mne.viz.plot_topomap(high_filtered, info, show=False, axes=ax_high,
                                             cmap='viridis', contours=6, sensors=True,
                                             names=available_channels, outlines='head')
                ax_high.set_title(f'{condition_high} (ON-task) Raw Values')
                ax_high.text(0.02, 0.98, f'Range: {vmin_high:.3f} to {vmax_high:.3f}',
                        transform=ax_high.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                cbar_high = plt.colorbar(im_high, ax=ax_high, shrink=0.8)
                cbar_high.set_label(f'{marker} value')
                
                # Plot low condition (OFF-task)
                ax_low = fig.add_subplot(gs[0, 1])
                vmin_low = np.min(low_filtered)
                vmax_low = np.max(low_filtered)
                im_low, _ = mne.viz.plot_topomap(low_filtered, info, show=False, axes=ax_low,
                                             cmap='viridis', contours=6, sensors=True,
                                             names=available_channels, outlines='head')
                ax_low.set_title(f'{condition_low} (OFF-task) Raw Values')
                ax_low.text(0.02, 0.98, f'Range: {vmin_low:.3f} to {vmax_low:.3f}',
                        transform=ax_low.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                cbar_low = plt.colorbar(im_low, ax=ax_low, shrink=0.8)
                cbar_low.set_label(f'{marker} value')
                
                # Plot difference (ON-task minus OFF-task)
                ax_diff = fig.add_subplot(gs[0, 2])
                # Use symmetric scale for difference
                diff_abs_max = max(abs(np.min(diff_filtered)), abs(np.max(diff_filtered)))
                im_diff, _ = mne.viz.plot_topomap(diff_filtered, info, show=False, axes=ax_diff,
                                             cmap='RdBu_r', contours=6, sensors=True,
                                             names=available_channels, outlines='head')
                ax_diff.set_title(f'Difference ({condition_high} minus {condition_low})')
                ax_diff.text(0.02, 0.98, f'Range: {np.min(diff_filtered):.3f} to {np.max(diff_filtered):.3f}',
                        transform=ax_diff.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                cbar_diff = plt.colorbar(im_diff, ax=ax_diff, shrink=0.8)
                cbar_diff.set_label(f'Difference in {marker} value')
                try:
                    cbar_diff.ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                except:
                    pass
                
                # Add legend for difference plot
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', label='ON-task > OFF-task'),
                    Patch(facecolor='blue', label='OFF-task > ON-task')
                ]
                ax_diff.legend(handles=legend_elements, loc='lower right')
            
            # Second row: T-statistics and significant clusters
            # Plot T-statistics topomap
            ax_t = fig.add_subplot(gs[1, 0:2])  # Span two columns
            # Use symmetric scale for T-statistics
            t_max = max(abs(T_obs_filtered.min()), abs(T_obs_filtered.max()))
            im_t, _ = mne.viz.plot_topomap(T_obs_filtered, info, show=False, axes=ax_t,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_t.set_title('T-statistics (ON-task vs OFF-task)', fontsize=14, fontweight='bold')
            
            # Add text showing T-statistic range
            ax_t.text(0.02, 0.98, f'T-range: {T_obs_filtered.min():.3f} to {T_obs_filtered.max():.3f}',
                     transform=ax_t.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar for T-statistics
            cbar_t = plt.colorbar(im_t, ax=ax_t, shrink=0.8)
            cbar_t.set_label('T-statistic', fontsize=12)
            try:
                cbar_t.ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            except:
                pass
            
            # Add legend for T-statistics
            legend_elements = [
                Patch(facecolor='red', label='ON-task > OFF-task'),
                Patch(facecolor='blue', label='OFF-task > ON-task')
            ]
            ax_t.legend(handles=legend_elements, loc='lower right', fontsize=10)
            
            # Plot significant clusters topomap
            ax_sig = fig.add_subplot(gs[1, 2])
            cluster_mask = np.zeros_like(T_obs_filtered, dtype=bool)
            
            for i, pv, cluster in significant_clusters:
                # Handle different cluster array shapes
                if len(cluster[0].shape) > 1:
                    cluster_indices = cluster[0][:, 0]
                else:
                    cluster_indices = cluster[0]
                    
                cluster_channels = [ch_names[idx] for idx in cluster_indices 
                                  if idx < len(ch_names) and ch_names[idx] in available_channels]
                cluster_indices = [available_channels.index(ch) for ch in cluster_channels 
                                 if ch in available_channels]
                if cluster_indices:
                    cluster_mask[cluster_indices] = True
            
            # Create masked T-statistics
            T_masked = T_obs_filtered.copy()
            T_masked[~cluster_mask] = 0
            
            im_sig, _ = mne.viz.plot_topomap(T_masked, info, show=False, axes=ax_sig,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_sig.set_title(f'Significant Clusters (p < {alpha})')
            
            # Add text showing number of significant clusters
            n_sig_clusters = len(significant_clusters)
            ax_sig.text(0.02, 0.98, f'Significant clusters: {n_sig_clusters}',
                     transform=ax_sig.transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add colorbar for significant clusters
            cbar_sig = plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
            cbar_sig.set_label('T-statistic\n(significant only)')
            try:
                cbar_sig.ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            except:
                pass
    
    except Exception as e:
        print(f"Could not create topoplots: {e}")
        # Create simple bar plots instead
        # First row: Raw values
        if high_values is not None and low_values is not None:
            ax_high = fig.add_subplot(gs[0, 0])
            ax_high.bar(range(len(high_values)), high_values, color='green')
            ax_high.set_title(f'{condition_high} (ON-task) Raw Values')
            ax_high.set_xlabel('Channel Index')
            ax_high.set_ylabel(f'{marker} value')
            
            ax_low = fig.add_subplot(gs[0, 1])
            ax_low.bar(range(len(low_values)), low_values, color='orange')
            ax_low.set_title(f'{condition_low} (OFF-task) Raw Values')
            ax_low.set_xlabel('Channel Index')
            ax_low.set_ylabel(f'{marker} value')
            
            ax_diff = fig.add_subplot(gs[0, 2])
            diff_values = high_values - low_values
            ax_diff.bar(range(len(diff_values)), diff_values,
                   color=['red' if d > 0 else 'blue' for d in diff_values])
            ax_diff.set_title(f'Difference ({condition_high} minus {condition_low})')
            ax_diff.set_xlabel('Channel Index')
            ax_diff.set_ylabel(f'Difference in {marker} value')
            ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Second row: T-statistics
        ax_t = fig.add_subplot(gs[1, 0:2])
        bars = ax_t.bar(range(len(T_obs[:, 0])), T_obs[:, 0], 
                       color=['red' if t > 0 else 'blue' for t in T_obs[:, 0]])
        ax_t.set_title('T-statistics by Channel', fontsize=14, fontweight='bold')
        ax_t.set_xlabel('Channel Index')
        ax_t.set_ylabel('T-statistic')
        ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add value range text
        ax_t.text(0.02, 0.98, f'T-range: {T_obs[:, 0].min():.3f} to {T_obs[:, 0].max():.3f}',
                 transform=ax_t.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_sig = fig.add_subplot(gs[1, 2])
        ax_sig.text(0.5, 0.5, 'Topoplot not available\n(channel positioning issue)', 
                ha='center', va='center', transform=ax_sig.transAxes)
        ax_sig.set_title('Significant Clusters')
        
        # Add legend for bar colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='ON-task > OFF-task'),
            Patch(facecolor='blue', label='OFF-task > ON-task')
        ]
        ax_t.legend(handles=legend_elements, loc='upper right')
    
    # Plot cluster statistics
    ax_stats = fig.add_subplot(gs[2, :])
    
    if len(significant_clusters) > 0:
        cluster_info = []
        for i, pv, cluster in significant_clusters:
            cluster_size = len(cluster[0])
            max_t = np.max(np.abs(T_obs[cluster[0][:, 0], 0]))
            cluster_channels = [ch_names[idx] for idx in cluster[0][:, 0] if idx < len(ch_names)]
            
            cluster_info.append({
                'Cluster': i + 1,
                'P-value': pv,
                'Size': cluster_size,
                'Max |T|': max_t,
                'Channels': ', '.join(cluster_channels[:5]) + ('...' if len(cluster_channels) > 5 else '')
            })
        
        # Create table
        cluster_df = pd.DataFrame(cluster_info)
        
        # Plot as table
        ax_stats.axis('tight')
        ax_stats.axis('off')
        table = ax_stats.table(cellText=cluster_df.values,
                         colLabels=cluster_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax_stats.set_title('Significant Clusters Summary', pad=20)
        
    else:
        ax_stats.text(0.5, 0.5, f'No significant clusters found (α = {alpha})', 
                ha='center', va='center', transform=ax_stats.transAxes, fontsize=14)
        ax_stats.set_title('Cluster Results')
        ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster results plot saved to: {save_path}")
    
    return fig


def save_cluster_results(T_obs, clusters, cluster_pv, ch_names, marker,
                        condition_high, condition_low, save_path, alpha=0.05):
    """
    Save cluster test results to CSV files.
    
    Parameters
    ----------
    T_obs : array
        Observed test statistics
    clusters : list
        Found clusters
    cluster_pv : array
        P-values for clusters
    ch_names : list
        Channel names
    marker : str
        Marker name
    condition_high : str
        High condition label
    condition_low : str
        Low condition label
    save_path : str
        Base path for saving results
    alpha : float
        Significance level
    """
    # Save T-statistics for all channels
    t_stats_df = pd.DataFrame({
        'channel': ch_names,
        'T_statistic': T_obs[:, 0],
        'marker': marker,
        'comparison': f'{condition_high}_vs_{condition_low}'
    })
    
    t_stats_file = save_path.replace('.csv', '_t_statistics.csv')
    t_stats_df.to_csv(t_stats_file, index=False)
    print(f"T-statistics saved to: {t_stats_file}")
    
    # Save cluster information
    if len(clusters) > 0:
        cluster_results = []
        
        for i, (cluster, pv) in enumerate(zip(clusters, cluster_pv)):
            # Handle different cluster array shapes
            if len(cluster[0].shape) > 1:
                cluster_indices = cluster[0][:, 0]
            else:
                cluster_indices = cluster[0]
            
            cluster_channels = [ch_names[idx] for idx in cluster_indices if idx < len(ch_names)]
            max_t = np.max(np.abs(T_obs[cluster_indices, 0]))
            mean_t = np.mean(T_obs[cluster_indices, 0])
            
            cluster_results.append({
                'cluster_id': i + 1,
                'p_value': pv,
                'significant': pv < alpha,
                'cluster_size': len(cluster[0]),
                'max_abs_t': max_t,
                'mean_t': mean_t,
                'channels': '; '.join(cluster_channels),
                'marker': marker,
                'comparison': f'{condition_high}_vs_{condition_low}'
            })
        
        cluster_df = pd.DataFrame(cluster_results)
        cluster_file = save_path.replace('.csv', '_clusters.csv')
        cluster_df.to_csv(cluster_file, index=False)
        print(f"Cluster results saved to: {cluster_file}")
        
        return len([c for c in cluster_results if c['significant']])
    
    return 0


def analyze_marker_clusters(df, marker, condition_col='onoff_label',
                           condition_high='high', condition_low='low',
                           n_permutations=1000, alpha=0.05, output_dir=None):
    """
    Analyze a single marker using cluster-based permutation testing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Aggregated marker data
    marker : str
        Marker to analyze
    condition_col : str
        Condition column name
    condition_high : str
        High condition label
    condition_low : str
        Low condition label
    n_permutations : int
        Number of permutations
    alpha : float
        Significance level
    output_dir : str
        Output directory for results
        
    Returns
    -------
    dict
        Results summary
    """
    print(f"Analyzing marker: {marker}")
    
    # Prepare data
    X, ch_names, high_values, low_values = prepare_data_for_cluster_test(
        df, marker, condition_col, condition_high, condition_low
    )
    
    if X is None or ch_names is None:
        print(f"  ✗ Could not prepare data for {marker}")
        return None
    
    print(f"  Data prepared: {len(ch_names)} channels, {X[0].shape[0]} subjects")
    
    # Get adjacency matrix
    adjacency = get_channel_adjacency(ch_names)
    
    # Run cluster test
    T_obs, clusters, cluster_pv, H0 = run_cluster_permutation_test(
        X, ch_names, adjacency=adjacency, n_permutations=n_permutations,
        alpha=alpha
    )
    
    if T_obs is None:
        print(f"  ✗ Cluster test failed for {marker}")
        return None
    
    # Count significant clusters
    n_significant = sum(pv < alpha for pv in cluster_pv)
    print(f"  Found {len(clusters)} clusters, {n_significant} significant (α = {alpha})")
    
    # Save results if output directory provided
    if output_dir:
        # Save statistical results
        results_file = os.path.join(output_dir, f"{marker}_cluster_results.csv")
        n_sig_saved = save_cluster_results(
            T_obs, clusters, cluster_pv, ch_names, marker,
            condition_high, condition_low, results_file, alpha
        )
        
        # Create and save plot
        plot_file = os.path.join(output_dir, f"{marker}_cluster_plot.png")
        fig = plot_cluster_results(
            T_obs, clusters, cluster_pv, ch_names, marker,
            condition_high, condition_low, alpha=alpha, save_path=plot_file,
            high_values=high_values, low_values=low_values
        )
        plt.close(fig)
    
    # Return summary
    return {
        'marker': marker,
        'n_channels': len(ch_names),
        'n_subjects': X[0].shape[0],
        'n_clusters': len(clusters),
        'n_significant_clusters': n_significant,
        'significant_p_values': [pv for pv in cluster_pv if pv < alpha],
        'min_p_value': min(cluster_pv) if len(cluster_pv) > 0 else None
    }


def main():
    """
    Main function to run cluster-based permutation tests for all markers.
    """
    # Define file paths
    csv_file_path = ('./results/aggregated_mne_markers/'
                     'aggregated_mne_markers_onoff_5trials_go_correct_outliers_removed_iqr_probe.csv')
    
    
    print("="*70)
    print("CLUSTER-BASED PERMUTATION TESTING FOR EEG MARKERS")
    print("="*70)
    print("Loading aggregated marker data...")
    
    try:
        # Load data
        df = pd.read_csv(csv_file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Get markers
        available_markers = df['marker'].unique()
        markers = [m for m in available_markers if m != 'marker']
        print(f"Found {len(markers)} markers to analyze")
        
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create output directory
    output_dir = './results/cluster_permutation_tests'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Analysis parameters
    condition_col = 'onoff_label'
    condition_high = 'high'  # on-task
    condition_low = 'low'    # off-task
    n_permutations = 1000
    alpha = 0.05
    
    print(f"\nAnalysis parameters:")
    print(f"  Condition: {condition_col}")
    print(f"  Comparison: {condition_high} vs {condition_low}")
    print(f"  Permutations: {n_permutations}")
    print(f"  Alpha level: {alpha}")
    print("-"*70)
    
    # Run analysis for each marker
    all_results = []
    successful_analyses = 0
    failed_analyses = 0
    
    for i, marker in enumerate(markers, 1):
        print(f"[{i}/{len(markers)}] Processing: {marker}")
        
        try:
            result = analyze_marker_clusters(
                df, marker, condition_col, condition_high, condition_low,
                n_permutations, alpha, output_dir
            )
            
            if result:
                all_results.append(result)
                successful_analyses += 1
                print(f"  ✓ Completed successfully")
            else:
                failed_analyses += 1
                print(f"  ✗ Analysis failed")
                
        except Exception as e:
            failed_analyses += 1
            print(f"  ✗ Error: {e}")
        
        print()
    
    # Save summary results
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_file = os.path.join(output_dir, 'cluster_analysis_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")
        
        # Print final summary
        print("="*70)
        print("ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total markers analyzed: {len(markers)}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {failed_analyses}")
        
        if successful_analyses > 0:
            total_sig_clusters = sum(r['n_significant_clusters'] for r in all_results)
            markers_with_sig = sum(1 for r in all_results if r['n_significant_clusters'] > 0)
            
            print(f"\nStatistical Results:")
            print(f"  Markers with significant clusters: {markers_with_sig}/{successful_analyses}")
            print(f"  Total significant clusters found: {total_sig_clusters}")
            
            if markers_with_sig > 0:
                print(f"\nMarkers with significant differences:")
                for result in all_results:
                    if result['n_significant_clusters'] > 0:
                        min_p = result['min_p_value']
                        print(f"  - {result['marker']}: {result['n_significant_clusters']} "
                              f"clusters (min p = {min_p:.4f})")
        
        print(f"\n🎉 Analysis complete! Results saved to: {output_dir}")
    
    else:
        print("⚠️  No successful analyses completed.")


if __name__ == "__main__":
    main() 