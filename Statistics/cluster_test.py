"""
Spatial cluster permutation testing module.

This module implements spatial cluster-based permutation testing using MNE's
standard functions for finding channel adjacency and performing cluster tests.
Updated to work with aggregated probe marker data from the Junifer pipeline,
specifically analyzing continuous onoff values (mind-wandering levels).
"""

import numpy as np
import pandas as pd
import mne
from typing import Tuple, List


def get_channel_adjacency(montage_path: str, ch_names: List[str]) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Get spatial adjacency matrix for EEG channels using the actual montage file.
    
    This function ensures that the adjacency matrix is calculated based on the actual
    channel positions defined in the CACS-64_REF.bvef montage file.
    
    Parameters
    ----------
    montage_path : str
        Path to montage file (.bvef or standard montage name)
    ch_names : List[str]
        List of channel names
        
    Returns
    -------
    adjacency : np.ndarray
        Sparse adjacency matrix based on actual montage positions
    ch_names : List[str]
        Channel names that were used (subset of input that exists in montage)
    channel_indices : List[int]
        Indices of the channels in the original ch_names list
        
    Notes
    -----
    Uses mne.channels.find_ch_adjacency() with the actual montage file.
    Ensures adjacency is calculated from real channel positions.
    """
    # Load montage first to check available channels
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
        print(f"Loaded custom montage: {montage_path}")
    else:
        montage = mne.channels.make_standard_montage(montage_path)
        print(f"Loaded standard montage: {montage_path}")
    
    # Check which channels from our data are available in the montage
    montage_ch_names = set(montage.ch_names)
    data_ch_names = set(ch_names)
    common_ch_names = sorted(list(data_ch_names & montage_ch_names))
    
    if len(common_ch_names) == 0:
        raise ValueError(f"No common channels between data ({ch_names}) and montage ({montage.ch_names})")
    
    print(f"Found {len(common_ch_names)} common channels: {common_ch_names}")
    print(f"Data has {len(ch_names)} channels, montage has {len(montage.ch_names)} channels")
    
    # Create info object with only the channels that exist in the montage
    info = mne.create_info(ch_names=common_ch_names, sfreq=250, ch_types='eeg')
    
    # Set montage to info object - this should work now since we only use common channels
    info.set_montage(montage, on_missing='ignore')
    
    # Validate montage was set correctly
    if info.get_montage() is None:
        raise ValueError(f"Failed to set montage from {montage_path}")
    
    # Find channel adjacency based on actual montage positions
    adjacency, ch_names_ordered = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    
    # Validate adjacency matrix
    if adjacency.shape[0] != len(ch_names_ordered):
        raise ValueError(f"Adjacency matrix shape {adjacency.shape} doesn't match channel count {len(ch_names_ordered)}")
    
    print(f"Adjacency matrix computed: {adjacency.shape[0]}x{adjacency.shape[1]} channels")
    print(f"Adjacent connections: {adjacency.nnz}")
    
    # Create mapping from common channels back to original channel indices
    channel_indices = [ch_names.index(ch) for ch in common_ch_names]
    
    return adjacency, common_ch_names, channel_indices


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
    Implements spatial cluster permutation testing for LMM results.
    Permutes predictor values within subjects to preserve subject-level structure.
    Cluster statistic is the sum of t-values within each cluster.
    Works with aggregated probe marker data (onoff, valence, confidence, etc.).
    """
    from lmm_model import run_lmm_per_channel
    
    # Validate data for cluster testing
    if not validate_cluster_data(observed_t_stats, df_behavioral, predictor_of_interest):
        raise ValueError("Data validation failed for cluster testing")
    
    # Validate adjacency matrix
    n_channels = len(observed_t_stats)
    if adjacency.shape[0] != n_channels or adjacency.shape[1] != n_channels:
        raise ValueError(f"Adjacency matrix shape {adjacency.shape} does not match number of channels {n_channels}")
    
    # Manual implementation following MNE's cluster test logic
    np.random.seed(seed)
    
    # Find observed clusters
    observed_clusters, observed_cluster_stats = _find_clusters(
        observed_t_stats, adjacency, threshold, tail
    )
    
    # Generate null distribution through permutations
    print(f"Running {n_permutations} permutations for cluster testing...")
    max_cluster_stats_null = []
    
    for perm_idx in range(n_permutations):
        if perm_idx % 100 == 0 and perm_idx > 0:
            print(f"  Completed {perm_idx}/{n_permutations} permutations")
        
        # Permute the predictor of interest within subjects (deterministic)
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
        
        # Store maximum cluster statistic (deterministic)
        if len(cluster_stats_perm) > 0:
            max_cluster_stats_null.append(float(np.max(np.abs(cluster_stats_perm))))
        else:
            max_cluster_stats_null.append(0.0)
    
    max_cluster_stats_null = np.array(max_cluster_stats_null)
    
    # Compute p-values for observed clusters (deterministic)
    cluster_p_values = np.zeros(len(observed_cluster_stats))
    for i, stat in enumerate(observed_cluster_stats):
        # P-value = proportion of permutations where max_stat >= observed_stat
        cluster_p_values[i] = float(np.mean(max_cluster_stats_null >= np.abs(stat)))
    
    print(f"Cluster testing completed. Found {len(observed_clusters)} clusters.")
    
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
    
    This function preserves the subject-level structure by permuting the predictor
    variable within each subject separately. This is important for mixed-effects
    models where we want to break the association between the predictor and outcome
    while preserving the within-subject correlation structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data with aggregated probe marker information
    predictor : str
        Name of predictor to permute (e.g., 'onoff', 'valence')
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with permuted predictor values
    """
    df_perm = df.copy()
    np.random.seed(seed)
    
    # Permute within each subject to preserve subject-level structure
    for subject in df['subject'].unique():
        subject_mask = df['subject'] == subject
        subject_indices = df.loc[subject_mask].index
        
        # Get current values for this subject
        subject_values = df.loc[subject_mask, predictor].values
        
        # Permute the values
        permuted_values = np.random.permutation(subject_values)
        
        # Assign permuted values back
        df_perm.loc[subject_indices, predictor] = permuted_values
    
    return df_perm


def validate_cluster_data(
    observed_t_stats: np.ndarray,
    df_behavioral: pd.DataFrame,
    predictor_of_interest: str
) -> bool:
    """
    Validate data for cluster permutation testing.
    
    Parameters
    ----------
    observed_t_stats : np.ndarray
        Observed t-statistics from LMM
    df_behavioral : pd.DataFrame
        Behavioral data from aggregated probe markers
    predictor_of_interest : str
        Name of predictor variable
        
    Returns
    -------
    bool
        True if data is valid for cluster testing
    """
    # Check if predictor exists
    if predictor_of_interest not in df_behavioral.columns:
        print(f"Predictor '{predictor_of_interest}' not found in data")
        return False
    
    # Check for sufficient variation in predictor
    unique_vals = df_behavioral[predictor_of_interest].unique()
    if len(unique_vals) < 2:
        print(f"Predictor '{predictor_of_interest}' has insufficient variation")
        return False
    
    # Check for sufficient subjects
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 3:
        print(f"Need at least 3 subjects for cluster testing, found {n_subjects}")
        return False
    
    # Check t-statistics
    if np.all(np.isnan(observed_t_stats)):
        print("All t-statistics are NaN")
        return False
    
    # Check for reasonable t-statistics range
    valid_t_stats = observed_t_stats[~np.isnan(observed_t_stats)]
    if len(valid_t_stats) == 0:
        print("No valid t-statistics found")
        return False
    
    t_range = np.max(np.abs(valid_t_stats))
    if t_range < 0.1:
        print(f"T-statistics are very small (max |t| = {t_range:.3f})")
        return False
    
    return True


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
