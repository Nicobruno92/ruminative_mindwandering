#!/usr/bin/env python3
"""
Cluster-based permutation test for PC1 regression analysis

Adapted from cluster_mixedlm_permutation.py to use continuous PC1 as predictor
instead of categorical onoff_label comparisons.

Key adaptations:
1. Uses continuous PC1 as fixed effect in LMM
2. Random slope for PC1 per participant  
3. Z-scoring PC1 within participant option
4. Permutes PC1 values instead of categorical labels
5. Tests PC1 slope coefficients instead of group differences
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from statsmodels.formula.api import mixedlm
from tqdm import tqdm
from scipy import stats
import warnings
import time
import json
from pathlib import Path

# Silence convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
CSV_FILE = os.path.join(
    'results/aggregated_mne_markers/merged_pca_eeg_markers.csv'
)
OUT_DIR = os.path.join(project_root, 'results/cluster_permutation_tests_pc1')
N_PERM = 500
ALPHA = 0.05
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# PC1-specific configuration
Z_SCORE_PC1 = False  # Keep raw PC1 values – no z-scoring
REQUIRE_PC1_DATA = True  # Only analyze rows with PC1 data
MIN_PC1_OBSERVATIONS = 10  # Minimum PC1 observations per channel

# Statistical improvements
MIN_PERMUTATIONS = 200
EARLY_STOP_THRESHOLD = 0.001
MAX_PERMUTATIONS = 1000
CONVERGENCE_CHECK_INTERVAL = 50
MIN_CHANNELS_FOR_CLUSTER = 4
MIN_EFFECT_SIZE = 0.1

# For testing
TEST_MARKERS = None

# Markers that benefit from log transformation before z-scoring (typically spectral power measures)
LOG_TRANSFORM_MARKERS = [
    'a', 'b', 'd', 'g', 't', 'rms', 'var', 'se', 'sef90', 'sef95', 'msf'
]

# -----------------------------------------------------------------------------
# PC1-SPECIFIC HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def z_score_pc1_by_participant(df_marker: pd.DataFrame) -> pd.DataFrame:
    """Z-score PC1 values within each participant."""
    df_zscore = df_marker.copy()
    
    if 'PC1' not in df_zscore.columns or 'subject_id' not in df_zscore.columns:
        print("   WARNING: Missing required columns for PC1 z-scoring")
        return df_zscore
    
    z_scored_groups = []
    for subject_id, group in df_zscore.groupby('subject_id'):
        group_copy = group.copy()
        pc1_valid = group_copy['PC1'].dropna()
        
        if len(pc1_valid) < 2:
            z_scored_groups.append(group_copy)
            continue
            
        pc1_mean = pc1_valid.mean()
        pc1_std = pc1_valid.std()
        
        if pc1_std > 0:
            group_copy['PC1'] = (group_copy['PC1'] - pc1_mean) / pc1_std
        
        z_scored_groups.append(group_copy)
    
    return pd.concat(z_scored_groups, ignore_index=True)


def compute_pc1_lmm_tmap(df_marker: pd.DataFrame, ch_names, use_zscore_pc1=True, rng=None):
    """
    Return a 1-D array of t-values for PC1 slope coefficients.
    
    Parameters
    ----------
    df_marker : pd.DataFrame
        Data for the marker with PC1 and mean values
    ch_names : list
        Channel names
    use_zscore_pc1 : bool
        Whether to use z-scored PC1 values
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    t_vec : array
        T-statistics for PC1 effect per channel
    """
    t_vec = np.full(len(ch_names), np.nan)
    
    # Choose PC1 variable
    pc1_var = 'PC1' if not use_zscore_pc1 or 'PC1_z' not in df_marker.columns else 'PC1_z'
    
    # Choose dependent variable (EEG marker). Prefer within-subject z-scored values when available
    dv = 'mean_z' if 'mean_z' in df_marker.columns else 'mean'
    
    # Pre-filter channels with sufficient PC1 data
    valid_channels = []
    for i, ch in enumerate(ch_names):
        ch_data = df_marker[df_marker['channel'] == ch]
        if not ch_data.empty:
            pc1_valid_count = ch_data[pc1_var].notna().sum()
            if pc1_valid_count >= MIN_PC1_OBSERVATIONS:
                valid_channels.append((i, ch))
    
    print(f"    Processing {len(valid_channels)}/{len(ch_names)} channels with sufficient PC1 data")
    
    for i, ch in valid_channels:
        d_ch = df_marker[df_marker['channel'] == ch].copy()
        d_ch = d_ch.dropna(subset=[pc1_var])  # Only keep rows with PC1 data
        
        if len(d_ch) < MIN_PC1_OBSERVATIONS:
            continue
            
        try:
            # Fit LMM with PC1 as predictor and random slope (using chosen dependent variable)
            formula = f'{dv} ~ {pc1_var}'
            re_formula = f'~1'
            
            model = mixedlm(formula, d_ch, groups=d_ch['subject_id'], re_formula=re_formula)
            res = model.fit(method='lbfgs', reml=False, maxiter=100)
            
            # Extract t-value for PC1 effect
            if pc1_var in res.params:
                t_vec[i] = res.tvalues[pc1_var]
                
        except Exception as e:
            continue
            
    return t_vec


def permute_pc1_values(df, rng=None):
    """
    Shuffle PC1 values within each subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    df_perm : pd.DataFrame
        DataFrame with shuffled PC1 values
    """
    df_perm = df.copy()
    
    for subject_id in df_perm['subject_id'].unique():
        idx = df_perm['subject_id'] == subject_id
        pc1_values = df_perm.loc[idx, 'PC1'].values
        
        if rng is not None:
            shuffled_values = rng.permutation(pc1_values)
        else:
            shuffled_values = np.random.permutation(pc1_values)
            
        df_perm.loc[idx, 'PC1'] = shuffled_values
        
        # Also shuffle z-scored PC1 if available
        if 'PC1_z' in df_perm.columns:
            pc1_z_values = df_perm.loc[idx, 'PC1_z'].values
            if rng is not None:
                shuffled_z_values = rng.permutation(pc1_z_values)
            else:
                shuffled_z_values = np.random.permutation(pc1_z_values)
            df_perm.loc[idx, 'PC1_z'] = shuffled_z_values
    
    return df_perm


# Import remaining utility functions from original script
def get_channel_adjacency(ch_names, montage_name='standard_1020'):
    """Get channel adjacency matrix."""
    try:
        info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
        montage = mne.channels.make_standard_montage(montage_name)
        info.set_montage(montage, on_missing='ignore')
        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(info, ch_type='eeg')
        
        if adjacency.shape[0] != len(ch_names):
            print(f"WARNING: Adjacency matrix size mismatch")
            return None
            
        return adjacency
        
    except Exception as e:
        print(f"ERROR: Could not create adjacency matrix: {e}")
        return None


def create_pseudo_adjacency(ch_names, max_distance=3):
    """Create pseudo-adjacency matrix based on alphabetical proximity."""
    n_channels = len(ch_names)
    adjacency = np.zeros((n_channels, n_channels))
    sorted_indices = np.argsort(ch_names)
    
    for i, idx_i in enumerate(sorted_indices):
        for j, idx_j in enumerate(sorted_indices):
            if abs(i - j) <= max_distance and i != j:
                adjacency[idx_i, idx_j] = 1
                
    return adjacency


def find_clusters(data, threshold, adjacency=None, min_channels=MIN_CHANNELS_FOR_CLUSTER):
    """Find clusters in data above threshold."""
    above_thresh = np.abs(data) > threshold
    mask = ~np.isnan(data)
    above_thresh = above_thresh & mask
    
    if not np.any(above_thresh):
        return [], []
    
    if adjacency is None:
        clusters = [[i] for i in np.where(above_thresh)[0]]
        cluster_stats = [data[i] for i in np.where(above_thresh)[0]]
        valid_clusters = [c for c in clusters if len(c) >= min_channels]
        valid_stats = np.array([data[c[0]] for c in valid_clusters])
        return valid_clusters, valid_stats
    
    # Convert adjacency to dense if needed
    if hasattr(adjacency, 'toarray'):
        adj_dense = adjacency.toarray()
    else:
        adj_dense = adjacency
    
    # Find connected components
    clusters = []
    visited = set()
    
    for i in np.where(above_thresh)[0]:
        if i in visited:
            continue
        
        cluster = []
        to_visit = [i]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            cluster.append(current)
            
            neighbors = np.where(adj_dense[current] > 0)[0]
            for neighbor in neighbors:
                if above_thresh[neighbor] and neighbor not in visited:
                    to_visit.append(neighbor)
        
        if len(cluster) >= min_channels:
            clusters.append(sorted(cluster))
    
    cluster_stats = np.array([np.sum(data[cluster]) for cluster in clusters])
    return clusters, cluster_stats


def run_pc1_cluster_test(df, marker, n_perm=N_PERM, alpha=ALPHA, rng=None):
    """
    Run cluster-based permutation test for PC1 regression.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing PC1 and EEG markers
    marker : str
        EEG marker to analyze
    n_perm : int
        Number of permutations
    alpha : float
        Significance level
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    dict
        Results dictionary
    """
    print(f"\n=== Marker: {marker} (PC1 CLUSTER ANALYSIS) ===")
    df_m = df[df['marker'] == marker].copy()
    
    if df_m.empty:
        print("   No data – skipping.")
        return None

    # Filter for PC1 data if required
    if REQUIRE_PC1_DATA:
        initial_rows = len(df_m)
        df_m = df_m.dropna(subset=['PC1'])
        print(f"   Filtered for PC1 data: {len(df_m)}/{initial_rows} rows")
        
        if df_m.empty:
            print("   No PC1 data – skipping.")
            return None

    # ------------------------------------------------------------------
    # Normalize EEG marker values (mean): optional log transform + within-subject z-scoring
    # ------------------------------------------------------------------
    if marker in LOG_TRANSFORM_MARKERS:
        print(f"   Applying log transformation for marker {marker}")
        # Ensure non-negative before log; add 1 to avoid log(0)
        df_m['mean'] = np.log1p(np.maximum(df_m['mean'], 0))

    # Within-subject z-scoring of marker values
    print("   Applying within-subject z-scoring of marker values…")
    df_m['mean_z'] = (
        df_m.groupby('subject_id')['mean']
             .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std() > 0 else 0)
    )

    # Fallback for zero variance cases
    fallback_mask = df_m['mean_z'].isna() | np.isinf(df_m['mean_z'])
    if fallback_mask.any():
        df_m.loc[fallback_mask, 'mean_z'] = df_m.loc[fallback_mask, 'mean']
        print(f"   Applied fallback for {fallback_mask.sum()} observations with zero variance")

    # Use z-scored values for subsequent analysis
    df_m['mean'] = df_m['mean_z']

    ch_names = sorted(df_m['channel'].unique())
    print(f"   Channels: {len(ch_names)}  |  Rows: {df_m.shape[0]}")

    # Get connectivity matrix
    conn = get_channel_adjacency(ch_names)
    if conn is None:
        print("   ⚠️  Could not create spatial adjacency matrix.")
        conn = create_pseudo_adjacency(ch_names)
        if conn is not None:
            print("   ✓ Using pseudo-adjacency matrix")
        else:
            print("   ❌ Aborting cluster test")
            return None

    # Compute observed t-map for PC1 effects
    t_obs = compute_pc1_lmm_tmap(df_m, ch_names, use_zscore_pc1=Z_SCORE_PC1, rng=rng)
    
    # Check if we have valid t-statistics
    valid_t_count = np.sum(~np.isnan(t_obs))
    if valid_t_count == 0:
        print("   No valid t-statistics computed – skipping.")
        return None
    
    print(f"   Valid t-statistics: {valid_t_count}/{len(ch_names)}")

    # Permutation testing
    print("   Running permutation testing...")
    t_perm = np.zeros((n_perm, len(ch_names)))
    
    for p in tqdm(range(n_perm), desc="    Permutations"):
        df_p = permute_pc1_values(df_m, rng=rng)
        t_perm[p] = compute_pc1_lmm_tmap(df_p, ch_names, use_zscore_pc1=Z_SCORE_PC1, rng=rng)

    # Determine cluster-forming threshold
    valid_perm_values = t_perm[~np.isnan(t_perm)]
    if len(valid_perm_values) == 0:
        print("   No valid permutation values found – aborting")
        return None
        
    thresh = np.percentile(np.abs(valid_perm_values), 100 * (1 - alpha / 2))
    print(f"   Cluster forming threshold (|t|): {thresh:.3f}")

    # Find clusters in observed data
    clusters_obs, stats_obs = find_clusters(t_obs, thresh, conn, min_channels=MIN_CHANNELS_FOR_CLUSTER)
    print(f"   Found {len(clusters_obs)} clusters in observed data")

    # Build null distribution
    max_stats = np.zeros(n_perm)
    for p in range(n_perm):
        clusts, stats_p = find_clusters(t_perm[p], thresh, conn, min_channels=MIN_CHANNELS_FOR_CLUSTER)
        max_stats[p] = np.max(np.abs(stats_p)) if len(stats_p) else 0

    # Compute cluster p-values
    p_vals = np.ones(len(stats_obs))
    for i, s in enumerate(np.abs(stats_obs)):
        p_vals[i] = (np.sum(max_stats >= s) + 1) / (n_perm + 1)

    return {
        'marker': marker,
        'ch_names': ch_names,
        't_obs': t_obs,
        'clusters': clusters_obs,
        'cluster_stats': stats_obs,
        'cluster_pvals': p_vals,
        'thresh': thresh,
        'df_marker': df_m,
        'n_permutations': n_perm,
        'z_scored_pc1': Z_SCORE_PC1,
        'min_cluster_size': MIN_CHANNELS_FOR_CLUSTER,
        'valid_perm_values': len(valid_perm_values),
        'total_perm_values': t_perm.size
    }


def save_pc1_cluster_results(results, save_path, alpha=0.05):
    """Save PC1 cluster test results to CSV files."""
    # Save T-statistics
    t_stats_df = pd.DataFrame({
        'channel': results['ch_names'],
        'PC1_t_statistic': results['t_obs'],
        'marker': results['marker']
    })
    
    t_stats_file = save_path.replace('.csv', '_t_statistics.csv')
    t_stats_df.to_csv(t_stats_file, index=False)
    print(f"   T-statistics saved to: {t_stats_file}")
    
    # Save cluster information
    if len(results['clusters']) > 0:
        cluster_results = []
        
        for i, (cluster, pv) in enumerate(zip(results['clusters'], results['cluster_pvals'])):
            cluster_channels = [results['ch_names'][idx] for idx in cluster if idx < len(results['ch_names'])]
            max_t = np.max(np.abs(results['t_obs'][cluster]))
            mean_t = np.mean(results['t_obs'][cluster])
            
            cluster_results.append({
                'cluster_id': i + 1,
                'p_value': pv,
                'significant': pv < alpha,
                'cluster_size': len(cluster),
                'max_abs_t': max_t,
                'mean_t': mean_t,
                'channels': '; '.join(cluster_channels),
                'marker': results['marker'],
                'analysis_type': 'PC1_regression'
            })
        
        cluster_df = pd.DataFrame(cluster_results)
        cluster_file = save_path.replace('.csv', '_clusters.csv')
        cluster_df.to_csv(cluster_file, index=False)
        print(f"   Cluster results saved to: {cluster_file}")
        
        return len([c for c in cluster_results if c['significant']])
    
    return 0


def plot_pc1_cluster_results(results, save_path=None, alpha=0.05, figsize=(15, 15)):
    """Plot PC1 cluster results."""
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
    marker = results['marker']
    ch_names = results['ch_names']
    t_obs = results['t_obs']
    clusters = results['clusters']
    cluster_pvals = results['cluster_pvals']
    
    # Find significant clusters
    significant_clusters = []
    for i, pv in enumerate(cluster_pvals):
        if pv < alpha:
            significant_clusters.append((i, pv, clusters[i]))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, height_ratios=[1, 1, 0.5])
    
    fig.suptitle(f'PC1 Cluster-based Permutation Test: {marker}\n'
                 f'PC1 Regression Analysis', 
                 fontsize=16, fontweight='bold')
    
    try:
        # Create montage for topoplots
        montage = mne.channels.make_standard_montage('standard_1020')
        available_channels = [ch for ch in ch_names if ch in montage.ch_names]
        
        if len(available_channels) > 10:
            info = mne.create_info(ch_names=available_channels, sfreq=250., ch_types='eeg')
            info.set_montage(montage, on_missing='ignore')
            
            # Filter data for available channels
            ch_indices = [ch_names.index(ch) for ch in available_channels if ch in ch_names]
            t_obs_filtered = t_obs[ch_indices]
            t_obs_filtered = np.nan_to_num(t_obs_filtered, nan=0.0)
            
            # Plot PC1 t-statistics
            ax_t = fig.add_subplot(gs[0, 0:2])
            im_t, _ = mne.viz.plot_topomap(t_obs_filtered, info, show=False, axes=ax_t,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_t.set_title('PC1 T-statistics', fontsize=14, fontweight='bold')
            plt.colorbar(im_t, ax=ax_t, shrink=0.8)
            
            # Plot significant clusters
            ax_sig = fig.add_subplot(gs[1, 0:2])
            sig_mask = np.zeros_like(t_obs_filtered, dtype=bool)
            
            for i, pv, cluster in significant_clusters:
                cluster_channels = [ch_names[idx] for idx in cluster
                                  if idx < len(ch_names) and ch_names[idx] in available_channels]
                cluster_indices = [available_channels.index(ch) for ch in cluster_channels 
                                 if ch in available_channels]
                if cluster_indices:
                    sig_mask[cluster_indices] = True
            
            t_masked = t_obs_filtered.copy()
            t_masked[~sig_mask] = 0
            
            im_sig, _ = mne.viz.plot_topomap(t_masked, info, show=False, axes=ax_sig,
                                          cmap='RdBu_r', contours=6, sensors=True,
                                          names=available_channels, outlines='head')
            ax_sig.set_title(f'Significant PC1 Clusters (p < {alpha})')
            plt.colorbar(im_sig, ax=ax_sig, shrink=0.8)
    
    except Exception as e:
        print(f"Could not create topoplots: {e}")
        # Fallback to bar plots
        ax_t = fig.add_subplot(gs[1, 0:2])
        bars = ax_t.bar(range(len(t_obs)), np.nan_to_num(t_obs),
                       color=['red' if t > 0 else 'blue' for t in np.nan_to_num(t_obs)])
        
        # Highlight significant channels
        for i, pv, cluster in significant_clusters:
            for idx in cluster:
                if idx < len(bars):
                    bars[idx].set_edgecolor('yellow')
                    bars[idx].set_linewidth(3)
        
        ax_t.set_title('PC1 T-statistics by Channel')
        ax_t.set_xlabel('Channel Index')
        ax_t.set_ylabel('T-statistic')
        ax_t.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Summary statistics
    ax_stats = fig.add_subplot(gs[2, :])
    
    n_sig_clusters = len(significant_clusters)
    summary_text = [
        f"Marker: {marker}",
        f"Analysis: PC1 regression",
        f"Z-scored PC1: {results['z_scored_pc1']}",
        f"Significant clusters: {n_sig_clusters}",
        f"Total clusters: {len(clusters)}",
        f"Min cluster size: {results['min_cluster_size']}",
        f"Permutations: {results['n_permutations']}",
        f"PC1 effect range: {np.nanmin(t_obs):.3f} to {np.nanmax(t_obs):.3f}"
    ]
    
    ax_stats.text(0.05, 0.95, '\n'.join(summary_text),
                 transform=ax_stats.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label='Significant cluster'),
        Patch(facecolor='red', label='Positive PC1 effect'),
        Patch(facecolor='blue', label='Negative PC1 effect')
    ]
    ax_stats.legend(handles=legend_elements, loc='center right')
    ax_stats.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path + '.svg', dpi=300, bbox_inches='tight')
        print(f"   PC1 cluster plot saved to: {save_path}")
    
    return fig


def main():
    """Main function for PC1 cluster permutation testing."""
    print("🧠 PC1 CLUSTER-BASED PERMUTATION TEST 🧠")
    print("Adaptations for PC1 regression:")
    print("  ✓ Continuous PC1 as predictor")
    print("  ✓ Random slope for PC1 per participant")
    print("  ✓ PC1 z-scoring within participant")
    print("  ✓ PC1 value permutation (within subject)")
    print("  ✓ PC1 slope t-statistics")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"Merged CSV not found: {CSV_FILE}")
        print("Please run the merge script first:")
        print("python Stats/phenomenological_regression/merge_pca_eeg_data.py")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Initialize random number generator
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"Random seed: {RANDOM_STATE}")

    print("Loading merged data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded data with shape: {df.shape}")
    
    # Check PC1 availability
    pc1_available = df['PC1'].notna().sum()
    print(f"Rows with PC1 data: {pc1_available}/{len(df)} ({100*pc1_available/len(df):.1f}%)")
    
    if pc1_available == 0:
        print("No PC1 data found!")
        sys.exit(1)

    markers = sorted(df['marker'].unique())
    print(f"Available markers: {markers}")
    
    if TEST_MARKERS:
        markers_to_process = [m for m in TEST_MARKERS if m in markers]
        if not markers_to_process:
            print("None of the test markers found!")
            sys.exit(1)
        print(f"Processing test markers: {markers_to_process}")
    else:
        markers_to_process = markers

    all_results = []
    total_start_time = time.time()
    
    for marker in markers_to_process:
        try:
            print(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            marker_start_time = time.time()
            
            res = run_pc1_cluster_test(df, marker, n_perm=N_PERM, alpha=ALPHA, rng=rng)
            
            if res is None:
                print(f"No results for marker {marker}")
                continue
            
            marker_elapsed = time.time() - marker_start_time
            res['elapsed_time'] = marker_elapsed
            all_results.append(res)
            
            # Save results
            results_file = os.path.join(OUT_DIR, f"{marker}_pc1_cluster_results.csv")
            n_sig_saved = save_pc1_cluster_results(res, results_file, ALPHA)
            
            # Create plot
            plot_path = os.path.join(OUT_DIR, f"{marker}_pc1_cluster_plot")
            try:
                fig = plot_pc1_cluster_results(res, plot_path, ALPHA)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create plot for marker {marker}: {e}")
            
            print(f"   → Analysis completed in {marker_elapsed:.1f} seconds")
            
        except Exception as e:
            print(f"ERROR processing marker {marker}: {e}")
            continue

    # Final summary
    print("\n" + "="*60)
    print("PC1 CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    
    if all_results:
        sig_total = 0
        for res in all_results:
            sig = np.sum(res['cluster_pvals'] < ALPHA)
            sig_total += sig
            print(f"{res['marker']:>15}: {sig} significant PC1 cluster(s)")
        
        print(f"--------------------------------------------------------------")
        total_elapsed = time.time() - total_start_time
        avg_time = sum(res.get('elapsed_time', 0) for res in all_results) / len(all_results)
        
        print(f"Total markers analyzed: {len(all_results)}")
        print(f"Total significant PC1 clusters: {sig_total}")
        print(f"Z-scored PC1: {Z_SCORE_PC1}")
        print(f"Min cluster size: {MIN_CHANNELS_FOR_CLUSTER}")
        print(f"Total time: {total_elapsed:.1f} seconds")
        print(f"Average time per marker: {avg_time:.1f} seconds")
        print(f"Results in: {OUT_DIR}")
    else:
        print("No results generated!")


if __name__ == '__main__':
    main() 