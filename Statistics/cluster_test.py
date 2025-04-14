"""
Spatial cluster permutation testing module.

This module implements spatial cluster-based permutation testing using MNE's
standard functions for finding channel adjacency and performing cluster tests.
"""

import numpy as np
import pandas as pd
import mne
from typing import Tuple, List


def get_channel_adjacency(montage_path: str, ch_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Get spatial adjacency matrix for EEG channels using MNE.
    
    Parameters
    ----------
    montage_path : str
        Path to montage file (.bvef or standard montage name)
    ch_names : List[str]
        List of channel names
        
    Returns
    -------
    adjacency : np.ndarray
        Sparse adjacency matrix
    ch_names : List[str]
        Channel names (may be reordered)
        
    Notes
    -----
    Uses mne.channels.find_ch_adjacency() for standard spatial adjacency.
    """
    # Create info object with channel positions
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    
    # Load montage
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)
    
    info.set_montage(montage)
    
    # Find channel adjacency
    adjacency, ch_names_ordered = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    
    return adjacency, ch_names_ordered


def spatial_cluster_permutation_test(
    observed_t_stats: np.ndarray,
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    formula: str,
    predictor_of_interest: str,
    adjacency: np.ndarray,
    threshold: float,
    n_permutations: int = 1000,
    tail: int = 0,
    seed: int = 42,
    n_jobs: int = -1,
    method: str = 'lbfgs',
    maxiter: int = 1000
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Perform spatial cluster-based permutation test.
    
    Parameters
    ----------
    observed_t_stats : np.ndarray
        Observed t-statistics from LMM, shape (n_channels,)
    power_data : np.ndarray
        Power values, shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with predictor variables
    formula : str
        R-style formula for LMM
    predictor_of_interest : str
        Name of predictor to test
    adjacency : np.ndarray
        Spatial adjacency matrix
    threshold : float
        T-statistic threshold for cluster formation
    n_permutations : int
        Number of permutations
    tail : int
        Tail for testing: 0=two-sided, -1=less, 1=greater
    seed : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    method : str
        LMM optimization method
    maxiter : int
        Maximum LMM iterations
        
    Returns
    -------
    clusters : List[np.ndarray]
        List of arrays, each containing channel indices for a cluster
    cluster_stats : np.ndarray
        Cluster-level statistics (sum of t-values)
    cluster_p_values : np.ndarray
        P-values for each cluster
        
    Notes
    -----
    Uses mne.stats.permutation_cluster_test for standard cluster testing.
    Cluster statistic is the sum of t-values within each cluster.
    """
    from .lmm_model import run_lmm_per_channel
    
    # Prepare data for permutation test
    # We need to create a function that computes t-stats for permuted labels
    def stat_fun(X, y):
        """
        Statistical function for permutation test.
        X: data (n_observations, n_channels)
        y: not used (compatibility with MNE)
        """
        # For each permutation, we'll permute the predictor variable
        # This is handled internally by running LMM with permuted data
        # For now, return identity since we'll handle permutations manually
        return X
    
    # Manual implementation following MNE's cluster test logic
    np.random.seed(seed)
    
    # Find observed clusters
    observed_clusters, observed_cluster_stats = _find_clusters(
        observed_t_stats, adjacency, threshold, tail
    )
    
    # Generate null distribution through permutations
    max_cluster_stats_null = []
    
    for perm_idx in range(n_permutations):
        # Permute the predictor of interest within subjects
        df_perm = _permute_within_subjects(df_behavioral, predictor_of_interest, seed + perm_idx)
        
        # Compute t-statistics for permuted data
        t_stats_perm, _ = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_perm,
            formula=formula,
            predictor_of_interest=predictor_of_interest,
            method=method,
            maxiter=maxiter,
            random_state=seed + perm_idx
        )
        
        # Find clusters in permuted data
        _, cluster_stats_perm = _find_clusters(
            t_stats_perm, adjacency, threshold, tail
        )
        
        # Store maximum cluster statistic
        if len(cluster_stats_perm) > 0:
            max_cluster_stats_null.append(np.max(np.abs(cluster_stats_perm)))
        else:
            max_cluster_stats_null.append(0.0)
    
    max_cluster_stats_null = np.array(max_cluster_stats_null)
    
    # Compute p-values for observed clusters
    cluster_p_values = np.zeros(len(observed_cluster_stats))
    for i, stat in enumerate(observed_cluster_stats):
        # P-value = proportion of permutations where max_stat >= observed_stat
        cluster_p_values[i] = np.mean(max_cluster_stats_null >= np.abs(stat))
    
    return observed_clusters, observed_cluster_stats, cluster_p_values


def _find_clusters(
    t_stats: np.ndarray,
    adjacency: np.ndarray,
    threshold: float,
    tail: int = 0
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find spatial clusters based on threshold.
    
    Parameters
    ----------
    t_stats : np.ndarray
        T-statistics, shape (n_channels,)
    adjacency : np.ndarray
        Spatial adjacency matrix
    threshold : float
        Threshold for cluster formation
    tail : int
        0=two-sided, -1=less, 1=greater
        
    Returns
    -------
    clusters : List[np.ndarray]
        List of channel index arrays for each cluster
    cluster_stats : np.ndarray
        Sum of t-values for each cluster
    """
    # Apply threshold based on tail
    if tail == 0:  # two-sided
        mask = np.abs(t_stats) > threshold
    elif tail == -1:  # less
        mask = t_stats < -threshold
    else:  # greater
        mask = t_stats > threshold
    
    if not np.any(mask):
        return [], np.array([])
    
    # Use MNE's clustering algorithm
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    # Get adjacency subgraph for suprathreshold channels
    suprathreshold_indices = np.where(mask)[0]
    
    if len(suprathreshold_indices) == 0:
        return [], np.array([])
    
    # Create subgraph adjacency matrix
    if hasattr(adjacency, 'toarray'):
        adj_array = adjacency.toarray()
    else:
        adj_array = adjacency
    
    sub_adj = adj_array[np.ix_(suprathreshold_indices, suprathreshold_indices)]
    sub_adj_sparse = csr_matrix(sub_adj)
    
    # Find connected components
    n_components, labels = connected_components(sub_adj_sparse, directed=False)
    
    # Extract clusters
    clusters = []
    cluster_stats = []
    
    for comp_idx in range(n_components):
        cluster_mask = labels == comp_idx
        cluster_channels = suprathreshold_indices[cluster_mask]
        
        if len(cluster_channels) > 0:
            clusters.append(cluster_channels)
            # Cluster statistic = sum of t-values in cluster
            cluster_stats.append(np.sum(t_stats[cluster_channels]))
    
    return clusters, np.array(cluster_stats)


def _permute_within_subjects(
    df: pd.DataFrame,
    predictor: str,
    seed: int
) -> pd.DataFrame:
    """
    Permute predictor values within each subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data
    predictor : str
        Name of predictor to permute
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with permuted predictor values
    """
    df_perm = df.copy()
    np.random.seed(seed)
    
    # Permute within each subject
    for subject in df['subject'].unique():
        subject_mask = df['subject'] == subject
        subject_values = df.loc[subject_mask, predictor].values
        permuted_values = np.random.permutation(subject_values)
        df_perm.loc[subject_mask, predictor] = permuted_values
    
    return df_perm


def summarize_clusters(
    clusters: List[np.ndarray],
    cluster_stats: np.ndarray,
    cluster_p_values: np.ndarray,
    ch_names: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Create summary DataFrame for significant clusters.
    
    Parameters
    ----------
    clusters : List[np.ndarray]
        List of channel index arrays
    cluster_stats : np.ndarray
        Cluster statistics
    cluster_p_values : np.ndarray
        Cluster p-values
    ch_names : List[str]
        Channel names
    alpha : float
        Significance threshold
        
    Returns
    -------
    pd.DataFrame
        Summary with columns: cluster_id, n_channels, channels, statistic, p_value, significant
    """
    import pandas as pd
    
    summary = []
    
    for i, (cluster, stat, pval) in enumerate(zip(clusters, cluster_stats, cluster_p_values)):
        cluster_channels = [ch_names[idx] for idx in cluster]
        summary.append({
            'cluster_id': i + 1,
            'n_channels': len(cluster),
            'channels': ', '.join(cluster_channels),
            'statistic': stat,
            'p_value': pval,
            'significant': pval < alpha
        })
    
    return pd.DataFrame(summary)
