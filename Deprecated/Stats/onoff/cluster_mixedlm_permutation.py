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
from scipy import ndimage
import time
import json  # Added for saving parameters

# Silence convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Get the script's directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Import the plotting function from the original script
sys.path.append(script_dir)
try:
    from cluster_permutation_test import plot_cluster_results
except ImportError:
    print("Warning: Could not import plot_cluster_results, using built-in plotting function")
    plot_cluster_results = None

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Use the subject-level aggregated file which is much smaller
CSV_FILE = os.path.join(
    'results/aggregated_mne_markers/aggregated_mne_markers_onoff_valence_confidence_time_selfother_5trials_go_correct_iqr_probe.csv'
)
OUT_DIR = os.path.join(project_root, 'results/cluster_permutation_tests_mixedlm')
N_PERM = 500          # reduced from 500 to make it faster
ALPHA = 0.05          # cluster-level alpha
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# For testing, limit to these markers (set to None to process all markers)
TEST_MARKERS = None  # Process all markers by default

# -----------------------------------------------------------------------------
# STATISTICAL IMPROVEMENTS CONFIGURATION
# -----------------------------------------------------------------------------
# Adaptive permutation testing parameters
MIN_PERMUTATIONS = 200  # Increased from 100 - minimum before early stopping
EARLY_STOP_THRESHOLD = 0.001  # Stop early if p-value precision is sufficient
MAX_PERMUTATIONS = 2000  # Increased for better precision
CONVERGENCE_CHECK_INTERVAL = 50  # Check convergence every N permutations

# High-precision permutation testing for promising results
HIGH_PRECISION_THRESHOLD = 0.10  # If any p < 0.10, use high precision
HIGH_PRECISION_MIN_PERMS = 2000  # Minimum permutations for high precision
HIGH_PRECISION_MAX_PERMS = 5000  # Maximum permutations for high precision

# Data quality requirements
MIN_OBS_PER_COMBINATION = 5  # Minimum observations per condition combination
MIN_CHANNELS_FOR_CLUSTER = 4  # Increased from 2 - minimum cluster size

# Effect size validation
MIN_EFFECT_SIZE_UV = 0.1  # Minimum effect size in µV (or µV²/Hz) to consider reliable

# Log-transform markers that benefit from it (typical spectral/power markers)
LOG_TRANSFORM_MARKERS = [
    'a', 'b', 'd', 'g', 't', 'rms', 'var', 'se', 'sef90', 'sef95', 'msf'
]

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def compute_lmm_tmap(df_marker: pd.DataFrame, ch_names, use_zscore=True, rng=None):
    """
    Return a 1-D array of t-values for the fixed effect cond (high vs low).
    
    Statistical improvements:
    - Initialize with NaN values for invalid channels
    - Pre-filter channels with sufficient data
    - Use z-scored data when available
    - Optimize LMM fitting parameters with robust standard errors
    - Better error handling
    """
    # Initialize with NaN values instead of zeros (Improvement #1)
    t_vec = np.full(len(ch_names), np.nan)
    
    # Choose dependent variable
    dv = 'mean_z' if use_zscore and 'mean_z' in df_marker.columns else 'mean'
    
    # Pre-filter channels with sufficient data
    valid_channels = []
    for i, ch in enumerate(ch_names):
        d_ch = df_marker[df_marker['channel'] == ch]
        if not d_ch.empty:
            # Check minimum observations per condition
            combo_counts = d_ch.groupby('onoff_label').size()
            if combo_counts.min() >= MIN_OBS_PER_COMBINATION:
                valid_channels.append((i, ch))
    
    print(f"    Processing {len(valid_channels)}/{len(ch_names)} channels with sufficient data")
    
    for i, ch in valid_channels:
        d_ch = df_marker[df_marker['channel'] == ch].copy()
        
        # Ensure categorical variable with proper ordering
        d_ch['cond'] = pd.Categorical(d_ch['onoff_label'], categories=['low', 'high'])
        
        try:
            # Optimized LMM fitting with robust standard errors (Improvement #6)
            model = mixedlm(f'{dv} ~ cond', d_ch, groups=d_ch['subject_id'], re_formula=f'~1 + cond')
            res = model.fit(
                method='lbfgs', 
                reml=False, 
                maxiter=100,
                cov_type='cluster',  # Robust standard errors
                cov_kwds={'groups': d_ch['subject_id']}  # Cluster by subject
            )
            
            # Extract t-value for condition effect
            param_name = [k for k in res.params.keys() if k.startswith('cond')][0]
            t_vec[i] = res.tvalues[param_name]
            
        except Exception as e:
            # Keep NaN for problematic channels (don't set to 0)
            continue
            
    return t_vec


def restricted_permutation(df, label_col='onoff_label', rng=None):
    """
    Shuffle labels within each subject (returns new Series).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    label_col : str
        Column to shuffle
    rng : np.random.Generator
        Random number generator for reproducible results
        
    Returns
    -------
    df_perm : pd.DataFrame
        DataFrame with shuffled labels
    """
    df_perm = df.copy()
    for sbj in df_perm['subject_id'].unique():
        idx = df_perm['subject_id'] == sbj
        # Use reproducible random state (Improvement #8)
        if rng is not None:
            shuffled_values = rng.permutation(df_perm.loc[idx, label_col].values)
        else:
            shuffled_values = df_perm.loc[idx, label_col].sample(frac=1).values
        df_perm.loc[idx, label_col] = shuffled_values
    return df_perm


def get_channel_adjacency(ch_names, montage_name='standard_1020'):
    """
    Get channel adjacency matrix based on spatial proximity.
    
    Statistical improvements:
    - More robust connectivity matrix creation
    - Proper error handling for adjacency matrix failures
    
    Parameters
    ----------
    ch_names : list
        Channel names
    montage_name : str
        MNE montage name
        
    Returns
    -------
    adjacency : array or None
        Adjacency matrix (sparse) or None if failed
    """
    try:
        # Create info object
        info = mne.create_info(ch_names=ch_names, sfreq=250., ch_types='eeg')
        montage = mne.channels.make_standard_montage(montage_name)
        info.set_montage(montage, on_missing='ignore')
        
        # Get adjacency matrix
        adjacency, ch_names_adj = mne.channels.find_ch_adjacency(info, ch_type='eeg')
        
        # Verify adjacency matrix is valid
        if adjacency.shape[0] != len(ch_names):
            print(f"WARNING: Adjacency matrix size mismatch ({adjacency.shape[0]} vs {len(ch_names)})")
            return None
            
        return adjacency
        
    except Exception as e:
        print(f"ERROR: Could not create adjacency matrix: {e}")
        # Return None instead of falling back to full connectivity (Improvement #3)
        return None


def create_pseudo_adjacency(ch_names, max_distance=3):
    """
    Create a pseudo-adjacency matrix based on alphabetical proximity.
    This is a fallback when spatial adjacency cannot be determined.
    
    Parameters
    ----------
    ch_names : list
        Channel names
    max_distance : int
        Maximum distance for considering channels as neighbors
        
    Returns
    -------
    adjacency : np.ndarray
        Pseudo-adjacency matrix
    """
    n_channels = len(ch_names)
    adjacency = np.zeros((n_channels, n_channels))
    
    # Sort channels alphabetically and create adjacency based on proximity
    sorted_indices = np.argsort(ch_names)
    
    for i, idx_i in enumerate(sorted_indices):
        for j, idx_j in enumerate(sorted_indices):
            if abs(i - j) <= max_distance and i != j:
                adjacency[idx_i, idx_j] = 1
                
    return adjacency


def find_clusters(data, threshold, adjacency=None, min_channels=MIN_CHANNELS_FOR_CLUSTER):
    """
    Find clusters in data above threshold with minimum size requirements.
    
    Statistical improvements:
    - Minimum cluster size requirement
    - More efficient connected components algorithm
    - Better handling of edge cases
    
    Parameters
    ----------
    data : array
        Data array (1D)
    threshold : float
        Threshold for cluster formation
    adjacency : scipy.sparse.csr_matrix or None
        Adjacency matrix
    min_channels : int
        Minimum cluster size
        
    Returns
    -------
    clusters : list
        List of arrays with indices for each cluster
    cluster_stats : array
        Sum of values within each cluster
    """
    # Apply threshold (two-sided)
    above_thresh = np.abs(data) > threshold
    
    # Handle NaN values
    mask = ~np.isnan(data)
    above_thresh = above_thresh & mask
    
    if not np.any(above_thresh):
        return [], []  # No clusters found
    
    # If no adjacency provided, each point is its own cluster
    if adjacency is None:
        clusters = [[i] for i in np.where(above_thresh)[0]]
        cluster_stats = [data[i] for i in np.where(above_thresh)[0]]
        # Apply minimum size filter
        valid_clusters = [c for c in clusters if len(c) >= min_channels]
        valid_stats = np.array([data[c[0]] for c in valid_clusters])
        return valid_clusters, valid_stats
    
    # Convert adjacency to dense if needed
    if hasattr(adjacency, 'toarray'):
        adj_dense = adjacency.toarray()
    else:
        adj_dense = adjacency
    
    # Find connected components using more efficient algorithm
    clusters = []
    visited = set()
    
    for i in np.where(above_thresh)[0]:
        if i in visited:
            continue
        
        # Start a new cluster using breadth-first search
        cluster = []
        to_visit = [i]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            cluster.append(current)
            
            # Find neighbors
            neighbors = np.where(adj_dense[current] > 0)[0]
            for neighbor in neighbors:
                if above_thresh[neighbor] and neighbor not in visited:
                    to_visit.append(neighbor)
        
        # Only keep clusters that meet minimum size requirement
        if len(cluster) >= min_channels:
            clusters.append(sorted(cluster))
    
    # Calculate cluster statistics (sum of values)
    cluster_stats = np.array([np.sum(data[cluster]) for cluster in clusters])
    
    return clusters, cluster_stats


def compute_permutation_pvalues_adaptive(t_obs, df_m, ch_names, 
                                        max_perm=MAX_PERMUTATIONS, 
                                        min_perm=MIN_PERMUTATIONS,
                                        alpha=ALPHA,
                                        rng=None):
    """
    Compute p-values with adaptive early stopping and automatic high-precision testing.
    
    Statistical improvements:
    - Adaptive early stopping based on convergence
    - Monitors p-value stability
    - Automatic high-precision testing for promising results
    - Reduces computational time while maintaining accuracy
    - Only use valid (non-NaN) values for threshold calculation
    
    Parameters
    ----------
    t_obs : array
        Observed t-statistics
    df_m : DataFrame
        Marker data
    ch_names : list
        Channel names
    max_perm : int
        Maximum permutations
    min_perm : int
        Minimum permutations before early stopping
    alpha : float
        Significance level
    rng : np.random.Generator
        Random number generator
        
    Returns
    -------
    t_perm : array
        Permutation t-statistics
    actual_perms : int
        Actual number of permutations performed
    needs_high_precision : bool
        Whether results suggest need for high-precision testing
    """
    n_channels = len(ch_names)
    t_perm = np.zeros((max_perm, n_channels))
    
    # Track convergence
    p_estimates = np.ones(n_channels)
    converged = np.zeros(n_channels, dtype=bool)
    
    actual_perms = 0
    
    print(f"    Starting adaptive permutation testing (max {max_perm}, min {min_perm})")
    
    for p in tqdm(range(max_perm), desc="    Permutations"):
        df_p = restricted_permutation(df_m, rng=rng)
        t_perm[p] = compute_lmm_tmap(df_p, ch_names, use_zscore=True, rng=rng)
        actual_perms += 1
        
        # Check convergence every CONVERGENCE_CHECK_INTERVAL permutations
        if p >= min_perm and (p + 1) % CONVERGENCE_CHECK_INTERVAL == 0:
            # Compute current p-value estimates
            for ch in range(n_channels):
                if not np.isnan(t_obs[ch]):
                    # Only count valid permutations (non-NaN)
                    valid_perm_values = t_perm[:p+1, ch][~np.isnan(t_perm[:p+1, ch])]
                    if len(valid_perm_values) > 0:
                        n_greater = np.sum(np.abs(valid_perm_values) >= np.abs(t_obs[ch]))
                        p_estimates[ch] = (n_greater + 1) / (len(valid_perm_values) + 1)
                    else:
                        p_estimates[ch] = 1.0
                    
                    # Check if converged (p-value stable enough)
                    if p > min_perm + CONVERGENCE_CHECK_INTERVAL:
                        prev_valid_values = t_perm[:p+1-CONVERGENCE_CHECK_INTERVAL, ch][~np.isnan(t_perm[:p+1-CONVERGENCE_CHECK_INTERVAL, ch])]
                        if len(prev_valid_values) > 0:
                            prev_n_greater = np.sum(np.abs(prev_valid_values) >= np.abs(t_obs[ch]))
                            prev_p = (prev_n_greater + 1) / (len(prev_valid_values) + 1)
                            
                            # Converged if change in p-value is small
                            if abs(p_estimates[ch] - prev_p) < EARLY_STOP_THRESHOLD:
                                converged[ch] = True
            
            # Early stopping if most channels converged or clearly non-significant
            if np.mean(converged) > 0.8 or np.all(p_estimates > alpha * 5):  # 5x alpha threshold
                print(f"    Early stopping at {p+1} permutations ({np.mean(converged)*100:.1f}% converged)")
                break
    
    # Trim the unused permutations
    t_perm = t_perm[:actual_perms]
    
    # Check if high-precision testing is needed (Improvement #5)
    needs_high_precision = (
        np.any(p_estimates < HIGH_PRECISION_THRESHOLD) and 
        actual_perms < HIGH_PRECISION_MIN_PERMS
    )
    
    if needs_high_precision:
        print(f"    → Found promising results (min p = {np.min(p_estimates):.3f})")
        print(f"    → Recommending high-precision testing with {HIGH_PRECISION_MIN_PERMS}-{HIGH_PRECISION_MAX_PERMS} permutations")
    
    return t_perm, actual_perms, needs_high_precision


def run_mixedlm_cluster_test(df, marker, n_perm=N_PERM, alpha=ALPHA, use_adaptive=True, rng=None):
    print(f"\n=== Marker: {marker} (IMPROVED STATISTICAL PIPELINE v4.0) ===")
    df_m = df[df['marker'] == marker].copy()
    if df_m.empty:
        print("   No data – skipping.")
        return None

    # ------------------------------------------------------------------
    # Statistical improvements: data preprocessing
    # ------------------------------------------------------------------
    
    # Apply log transformation if beneficial for this marker
    if marker in LOG_TRANSFORM_MARKERS:
        print(f"   Applying log transformation for marker {marker}")
        df_m['mean'] = np.log1p(np.maximum(df_m['mean'], 0))
    
    # Store original values for effect size calculation (Improvement #7)
    original_means = df_m.groupby(['onoff_label', 'channel'])['mean'].mean().unstack(level=0)
    if 'high' in original_means.columns and 'low' in original_means.columns:
        mean_difference_original = (original_means['high'] - original_means['low']).mean()
        print(f"   Mean difference (ON-OFF) in original units: {mean_difference_original:.4f}")
        
        # Check if effect size is meaningful (Improvement #7)
        if abs(mean_difference_original) < MIN_EFFECT_SIZE_UV:
            print(f"   ⚠️  WARNING: Effect size ({abs(mean_difference_original):.4f}) < {MIN_EFFECT_SIZE_UV} - potentially unreliable")
            effect_size_warning = True
        else:
            effect_size_warning = False
    else:
        mean_difference_original = np.nan
        effect_size_warning = False
    
    # Apply within-subject z-scoring for better statistical power
    print("   Applying within-subject z-scoring...")
    df_m['mean_z'] = (
        df_m.groupby('subject_id')['mean']
        .transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std() > 0 else 0)
    )
    
    # Handle zero variance cases (fallback to raw values)
    fallback_mask = df_m['mean_z'].isna() | np.isinf(df_m['mean_z'])
    if fallback_mask.any():
        df_m.loc[fallback_mask, 'mean_z'] = df_m.loc[fallback_mask, 'mean']
        print(f"   Applied fallback for {fallback_mask.sum()} observations with zero variance")

    # channels present for this marker
    ch_names = sorted(df_m['channel'].unique())
    print(f"   Channels: {len(ch_names)}  |  Rows: {df_m.shape[0]}")

    # connectivity matrix (for clustering) - Improvement #3
    conn = get_channel_adjacency(ch_names)
    if conn is None:
        print("   ⚠️  Could not create spatial adjacency matrix.")
        print("   → Attempting pseudo-adjacency matrix...")
        conn = create_pseudo_adjacency(ch_names)
        if conn is not None:
            print("   ✓ Using pseudo-adjacency matrix based on alphabetical proximity")
        else:
            print("   ❌ Aborting cluster test - no valid adjacency matrix available")
            return None

    # ------------------------------------------------------------------
    # observed t-map (using improved LMM computation)
    # ------------------------------------------------------------------
    t_obs = compute_lmm_tmap(df_m, ch_names, use_zscore=True, rng=rng)

    # ------------------------------------------------------------------
    # null distribution – adaptive or fixed permutation testing
    # ------------------------------------------------------------------
    needs_high_precision = False
    if use_adaptive:
        t_perm, actual_perms, needs_high_precision = compute_permutation_pvalues_adaptive(
            t_obs, df_m, ch_names, max_perm=n_perm, alpha=alpha, rng=rng
        )
        n_perm_used = actual_perms
        print(f"   Used {actual_perms} permutations (adaptive)")
        
        # Automatic high-precision relaunching (Improvement #5)
        if needs_high_precision:
            print(f"   🚀 Launching high-precision testing...")
            t_perm_hp, actual_perms_hp, _ = compute_permutation_pvalues_adaptive(
                t_obs, df_m, ch_names, 
                max_perm=HIGH_PRECISION_MAX_PERMS, 
                min_perm=HIGH_PRECISION_MIN_PERMS,
                alpha=alpha, rng=rng
            )
            # Use high-precision results
            t_perm = t_perm_hp
            n_perm_used = actual_perms_hp
            print(f"   ✓ High-precision testing completed with {actual_perms_hp} permutations")
    else:
        print("   Using fixed permutation testing")
        t_perm = np.zeros((n_perm, len(ch_names)))
        print("   Permutations:")
        for p in tqdm(range(n_perm)):
            df_p = restricted_permutation(df_m, rng=rng)
            t_perm[p] = compute_lmm_tmap(df_p, ch_names, use_zscore=True, rng=rng)
        n_perm_used = n_perm

    # ------------------------------------------------------------------
    # Determine cluster-forming threshold - Improvement #2
    # two-tailed → use absolute values, 97.5th percentile of permuted |t|
    # Only use valid (non-NaN) values for threshold calculation
    valid_perm_values = t_perm[~np.isnan(t_perm)]
    if len(valid_perm_values) == 0:
        print("   ❌ No valid permutation values found - aborting")
        return None
        
    thresh = np.percentile(np.abs(valid_perm_values), 100 * (1 - alpha / 2))
    print(f"   Cluster forming threshold (|t|): {thresh:.3f} (based on {len(valid_perm_values)} valid values)")

    # ------------------------------------------------------------------
    # Find clusters in observed map (with minimum size requirement)
    # ------------------------------------------------------------------
    clusters_obs, stats_obs = find_clusters(t_obs, thresh, conn, min_channels=MIN_CHANNELS_FOR_CLUSTER)
    print(f"   Found {len(clusters_obs)} clusters in observed data (min size: {MIN_CHANNELS_FOR_CLUSTER})")

    # ------------------------------------------------------------------
    # Build permutation distribution: max cluster statistic per perm
    # ------------------------------------------------------------------
    max_stats = np.zeros(n_perm_used)
    for p in range(n_perm_used):
        # Find clusters in permuted data
        clusts, stats_p = find_clusters(t_perm[p], thresh, conn, min_channels=MIN_CHANNELS_FOR_CLUSTER)
        max_stats[p] = np.max(np.abs(stats_p)) if len(stats_p) else 0

    # Compute p-values for observed clusters
    p_vals = np.ones(len(stats_obs))
    for i, s in enumerate(np.abs(stats_obs)):
        p_vals[i] = (np.sum(max_stats >= s) + 1) / (n_perm_used + 1)

    return {
        'marker': marker,
        'ch_names': ch_names,
        't_obs': t_obs,
        'clusters': clusters_obs,
        'cluster_stats': stats_obs,
        'cluster_pvals': p_vals,
        'thresh': thresh,
        'df_marker': df_m,  # keep for plotting means
        'n_permutations': n_perm_used,  # actual permutations used
        'log_transformed': marker in LOG_TRANSFORM_MARKERS,
        'adaptive_testing': use_adaptive,
        'min_cluster_size': MIN_CHANNELS_FOR_CLUSTER,
        'high_precision_used': needs_high_precision,
        'effect_size_original': mean_difference_original,
        'effect_size_warning': effect_size_warning,
        'valid_perm_values': len(valid_perm_values),
        'total_perm_values': t_perm.size
    }


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
        'T_statistic': T_obs,
        'marker': marker,
        'comparison': f'{condition_high}_vs_{condition_low}'
    })
    
    t_stats_file = save_path.replace('.csv', '_t_statistics.csv')
    t_stats_df.to_csv(t_stats_file, index=False)
    print(f"   T-statistics saved to: {t_stats_file}")
    
    # Save cluster information
    if len(clusters) > 0:
        cluster_results = []
        
        for i, (cluster, pv) in enumerate(zip(clusters, cluster_pv)):
            cluster_channels = [ch_names[idx] for idx in cluster if idx < len(ch_names)]
            max_t = np.max(np.abs(T_obs[cluster]))
            mean_t = np.mean(T_obs[cluster])
            
            cluster_results.append({
                'cluster_id': i + 1,
                'p_value': pv,
                'significant': pv < alpha,
                'cluster_size': len(cluster),
                'max_abs_t': max_t,
                'mean_t': mean_t,
                'channels': '; '.join(cluster_channels),
                'marker': marker,
                'comparison': f'{condition_high}_vs_{condition_low}'
            })
        
        cluster_df = pd.DataFrame(cluster_results)
        cluster_file = save_path.replace('.csv', '_clusters.csv')
        cluster_df.to_csv(cluster_file, index=False)
        print(f"   Cluster results saved to: {cluster_file}")
        
        return len([c for c in cluster_results if c['significant']])
    
    return 0


def plot_cluster_results(T_obs, clusters, cluster_pv, ch_names, marker,
                        condition_high, condition_low, alpha=0.05,
                        save_path=None, figsize=(15, 15), 
                        high_values=None, low_values=None):
    """
    Plot cluster-based permutation test results.
    COPIED FROM cluster_permutation_test.py to avoid import issues.
    
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
    import pandas as pd
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    
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

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("🚀 IMPROVED STATISTICAL PIPELINE (v4.0) 🚀")
    print("Statistical improvements:")
    print("  ✓ NaN handling for invalid channels")
    print("  ✓ Threshold based only on valid permutation values")
    print("  ✓ Robust connectivity matrix with fallbacks")
    print("  ✓ Minimum cluster size: 4 channels")
    print("  ✓ Enhanced permutation testing with automatic high-precision")
    print("  ✓ Robust standard errors in LMM")
    print("  ✓ Effect size validation")
    print("  ✓ Reproducible random number generation")
    print("  ✓ Parameter documentation in JSON")
    print("=" * 60)
    
    if not os.path.exists(CSV_FILE):
        print(f"CSV not found: {CSV_FILE}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Initialize random number generator for reproducibility (Improvement #8)
    rng = np.random.default_rng(RANDOM_STATE)
    print(f"Random seed: {RANDOM_STATE}")

    print("Loading data …")
    
    # Check file size before loading
    file_size_mb = os.path.getsize(CSV_FILE) / (1024 * 1024)
    print(f"CSV file size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 500 and TEST_MARKERS:
        print(f"Large file detected. Loading only data for test markers: {TEST_MARKERS}")
        # Use chunking to filter large files
        chunks = pd.read_csv(CSV_FILE, chunksize=1000000)
        df_list = []
        for chunk in tqdm(chunks, desc="Processing chunks"):
            if TEST_MARKERS:
                # Filter chunk by markers
                filtered_chunk = chunk[chunk['marker'].isin(TEST_MARKERS)]
                if not filtered_chunk.empty:
                    df_list.append(filtered_chunk)
            else:
                df_list.append(chunk)
        
        df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded filtered data with shape: {df.shape}")
    else:
        # Load entire file
        df = pd.read_csv(CSV_FILE)
        print(f"Loaded full data with shape: {df.shape}")
    
    # Print column names for debugging
    print(f"Available columns: {df.columns.tolist()}")
    
    # Ensure expected columns
    for col in ['subject_id', 'marker', 'channel', 'onoff_label', 'mean']:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from CSV")

    markers = sorted(df['marker'].unique())
    print(f"Available markers: {markers}")
    
    # Determine which markers to process
    if TEST_MARKERS:
        # Filter to only include test markers that are actually in the data
        markers_to_process = [m for m in TEST_MARKERS if m in markers]
        if not markers_to_process:
            print("None of the test markers found in data!")
            sys.exit(1)
        print(f"Processing test markers: {markers_to_process}")
    else:
        markers_to_process = markers
        print(f"Processing all {len(markers_to_process)} markers")

    all_results = []
    total_start_time = time.time()
    
    for marker in markers_to_process:
        try:
            print(f"\n{'='*60}\nProcessing marker: {marker}\n{'='*60}")
            marker_start_time = time.time()
            res = run_mixedlm_cluster_test(df, marker, n_perm=N_PERM, alpha=ALPHA, rng=rng)
            marker_elapsed = time.time() - marker_start_time
            
            if res is None:
                print(f"No results for marker {marker} - skipping")
                continue
                
            res['elapsed_time'] = marker_elapsed
            all_results.append(res)
            print(f"   → Analysis completed in {marker_elapsed:.1f} seconds")
            
            # ------------------------------------------------------------------
            # Save results in the same format as the original script
            # ------------------------------------------------------------------
            # Reshape t_obs to (n_channels, 1) to match the format expected by save_cluster_results
            t_mat = res['t_obs'].reshape(-1, 1)
            
            # Save statistical results
            results_file = os.path.join(OUT_DIR, f"{marker}_cluster_results.csv")
            n_sig_saved = save_cluster_results(
                res['t_obs'], res['clusters'], res['cluster_pvals'], res['ch_names'], 
                marker, 'high', 'low', results_file, ALPHA
            )
            
            # Save analysis parameters to JSON (Improvement #9)
            save_analysis_parameters(
                OUT_DIR, marker,
                alpha=ALPHA,
                n_perm=N_PERM,
                n_perm_actual=res.get('n_permutations', N_PERM),
                random_seed=RANDOM_STATE,
                min_cluster_size=res.get('min_cluster_size', MIN_CHANNELS_FOR_CLUSTER),
                log_transformed=res.get('log_transformed', False),
                adaptive_testing=res.get('adaptive_testing', True),
                high_precision_used=res.get('high_precision_used', False),
                effect_size_original=res.get('effect_size_original', np.nan),
                effect_size_warning=res.get('effect_size_warning', False),
                valid_perm_values=res.get('valid_perm_values', 'unknown'),
                total_perm_values=res.get('total_perm_values', 'unknown'),
                cluster_forming_threshold=res['thresh'],
                significant_clusters=n_sig_saved
            )

            # ------------------------------------------------------------------
            # Create plot matching legacy layout
            # ------------------------------------------------------------------
            plot_path = os.path.join(OUT_DIR, f"{marker}_cluster_plot.png")

            try:
                # mean per channel for each condition
                df_m = res['df_marker']
                means = (
                    df_m.groupby(['onoff_label', 'channel'])['mean']
                    .mean()
                    .unstack(level=0)
                )
                if 'high' in means.columns and 'low' in means.columns:
                    high_vals = means['high'].reindex(res['ch_names']).values
                    low_vals = means['low'].reindex(res['ch_names']).values
                else:
                    high_vals = low_vals = None

                # reshape t_obs to (n_channels, 1) to satisfy plot helper
                t_mat = res['t_obs'].reshape(-1, 1)

                # wrap clusters so cluster[0] pattern matches legacy code
                clusters_wrapped = []
                for cl in res['clusters']:
                    # Create a structure that matches what the original script expects
                    cl_array = np.zeros((len(cl), 1), dtype=int)
                    for i, idx in enumerate(cl):
                        cl_array[i, 0] = idx
                    clusters_wrapped.append((cl_array,))

                fig = plot_cluster_results(
                    t_mat,
                    clusters_wrapped,
                    res['cluster_pvals'],
                    res['ch_names'],
                    marker,
                    'high',
                    'low',
                    alpha=ALPHA,
                    save_path=plot_path,
                    high_values=high_vals,
                    low_values=low_vals,
                )
                plt.close(fig)
                print(f"   → plot saved {plot_path}")
            except Exception as e:
                print(f"ERROR creating plot for marker {marker}: {e}")
                # Try a simpler plot approach as fallback
                try:
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 1, 1)
                    plt.bar(range(len(res['t_obs'])), res['t_obs'])
                    plt.axhline(y=0, color='k', linestyle='-')
                    plt.title(f"T-statistics for marker {marker}")
                    plt.xlabel("Channel index")
                    plt.ylabel("T-value")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"   → fallback plot saved {plot_path}")
                except Exception as e2:
                    print(f"ERROR creating fallback plot for marker {marker}: {e2}")

                # mean per channel for each condition
                df_m = res['df_marker']
                means = (
                    df_m.groupby(['onoff_label', 'channel'])['mean']
                    .mean()
                    .unstack(level=0)
                )
                if 'high' in means.columns and 'low' in means.columns:
                    high_vals = means['high'].reindex(res['ch_names']).values
                    low_vals = means['low'].reindex(res['ch_names']).values
                else:
                    high_vals = low_vals = None

                if plot_cluster_results:
                    # Use the original plotting function if available
                    # reshape t_obs to (n_channels, 1) to satisfy plot helper
                    t_mat = res['t_obs'].reshape(-1, 1)

                    # wrap clusters so cluster[0] pattern matches legacy code
                    clusters_wrapped = []
                    for cl in res['clusters']:
                        # Create a structure that matches what the original script expects
                        cl_array = np.zeros((len(cl), 1), dtype=int)
                        for i, idx in enumerate(cl):
                            cl_array[i, 0] = idx
                        clusters_wrapped.append((cl_array,))

                    fig = plot_cluster_results(
                        t_mat,
                        clusters_wrapped,
                        res['cluster_pvals'],
                        res['ch_names'],
                        marker,
                        'high',
                        'low',
                        alpha=ALPHA,
                        save_path=plot_path,
                        high_values=high_vals,
                        low_values=low_vals,
                    )
                    plt.close(fig)
                else:
                    # Use the built-in plot_cluster_results function
                    # reshape t_obs to (n_channels, 1) to match expected format
                    t_mat = res['t_obs'].reshape(-1, 1)
                    
                    # wrap clusters so cluster[0] pattern matches expected format
                    clusters_wrapped = []
                    for cl in res['clusters']:
                        # Create a structure that matches what the original script expects
                        cl_array = np.zeros((len(cl), 1), dtype=int)
                        for i, idx in enumerate(cl):
                            cl_array[i, 0] = idx
                        clusters_wrapped.append((cl_array,))
                    
                    fig = plot_cluster_results(
                        t_mat,
                        clusters_wrapped,
                        res['cluster_pvals'],
                        res['ch_names'],
                        marker,
                        'high',
                        'low',
                        alpha=ALPHA,
                        save_path=plot_path,
                        high_values=high_vals,
                        low_values=low_vals,
                    )
                    plt.close(fig)
                
                print(f"   → plot saved {plot_path}")
        except Exception as e:
            print(f"ERROR processing marker {marker}: {e}")
            continue

    print("\n=========================== SUMMARY ===========================")
    sig_total = 0
    total_perms = 0
    adaptive_count = 0
    log_transformed_count = 0
    
    for res in all_results:
        sig = np.sum(res['cluster_pvals'] < ALPHA)
        sig_total += sig
        total_perms += res.get('n_permutations', N_PERM)
        if res.get('adaptive_testing', False):
            adaptive_count += 1
        if res.get('log_transformed', False):
            log_transformed_count += 1
            
        marker_info = f"{res['marker']:>10}: {sig} significant cluster(s)"
        if res.get('adaptive_testing', False):
            marker_info += f" [{res.get('n_permutations', N_PERM)} perms]"
        if res.get('log_transformed', False):
            marker_info += " [log-transformed]"
        print(marker_info)
        
    print(f"--------------------------------------------------------------")
    total_elapsed = time.time() - total_start_time
    avg_time_per_marker = sum(res.get('elapsed_time', 0) for res in all_results) / len(all_results) if all_results else 0
    
    print(f"Total markers analysed: {len(all_results)}")
    print(f"Total significant clusters: {sig_total}")
    print(f"Average permutations per marker: {total_perms / len(all_results):.0f}")
    print(f"Markers using adaptive testing: {adaptive_count}/{len(all_results)}")
    print(f"Markers with log transformation: {log_transformed_count}/{len(all_results)}")
    print(f"Minimum cluster size: {MIN_CHANNELS_FOR_CLUSTER} channels")
    print(f"Total analysis time: {total_elapsed:.1f} seconds")
    print(f"Average time per marker: {avg_time_per_marker:.1f} seconds")
    print(f"Results in: {OUT_DIR}\n")

def save_analysis_parameters(out_dir, marker, **params):
    """
    Save analysis parameters to JSON file for documentation (Improvement #9).
    
    Parameters
    ----------
    out_dir : str
        Output directory
    marker : str
        Marker name
    **params : dict
        Analysis parameters to save
    """
    params_dict = {
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'marker': marker,
        'alpha': params.get('alpha', ALPHA),
        'n_permutations_requested': params.get('n_perm', N_PERM),
        'n_permutations_actual': params.get('n_perm_actual', 'unknown'),
        'random_seed': params.get('random_seed', RANDOM_STATE),
        'min_cluster_size': params.get('min_cluster_size', MIN_CHANNELS_FOR_CLUSTER),
        'min_permutations': params.get('min_permutations', MIN_PERMUTATIONS),
        'max_permutations': params.get('max_permutations', MAX_PERMUTATIONS),
        'high_precision_threshold': params.get('high_precision_threshold', HIGH_PRECISION_THRESHOLD),
        'high_precision_min_perms': params.get('high_precision_min_perms', HIGH_PRECISION_MIN_PERMS),
        'high_precision_max_perms': params.get('high_precision_max_perms', HIGH_PRECISION_MAX_PERMS),
        'min_effect_size_uv': params.get('min_effect_size_uv', MIN_EFFECT_SIZE_UV),
        'log_transformed': params.get('log_transformed', False),
        'adaptive_testing': params.get('adaptive_testing', True),
        'high_precision_used': params.get('high_precision_used', False),
        'effect_size_original': params.get('effect_size_original', np.nan),
        'effect_size_warning': params.get('effect_size_warning', False),
        'valid_perm_values': params.get('valid_perm_values', 'unknown'),
        'total_perm_values': params.get('total_perm_values', 'unknown'),
        'cluster_forming_threshold': params.get('cluster_forming_threshold', 'unknown'),
        'significant_clusters': params.get('significant_clusters', 'unknown')
    }
    
    # Convert numpy types to Python types for JSON serialization
    for key, value in params_dict.items():
        if isinstance(value, np.integer):
            params_dict[key] = int(value)
        elif isinstance(value, np.floating):
            if np.isnan(value):
                params_dict[key] = None
            else:
                params_dict[key] = float(value)
        elif isinstance(value, np.ndarray):
            params_dict[key] = value.tolist()
    
    # Save to JSON file
    params_file = os.path.join(out_dir, f"{marker}_analysis_params.json")
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    print(f"   📋 Analysis parameters saved to: {params_file}")
    return params_file

if __name__ == '__main__':
    main() 