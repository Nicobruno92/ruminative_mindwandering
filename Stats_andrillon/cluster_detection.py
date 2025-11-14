"""
Cluster Detection Module - Andrillon 2020 Implementation

This module implements the cluster-based permutation testing from Andrillon et al. (2020).

Key steps:
1. Identify candidate clusters (electrodes with p < cluster_alpha)
2. Group neighboring electrodes into spatial clusters
3. Calculate cluster statistic (sum of t-values)
4. Compare against null distribution from permutations
5. Compute Monte Carlo p-values

Reference:
    Andrillon et al. (2020). Lines 102-115:
    "Candidate clusters were defined as neighbouring electrodes with a p-value 
    below a threshold (called cluster alpha) of 0.025. For each candidate cluster, 
    we computed the sum of the t-values for all the electrodes belonging to the 
    cluster (which we will refer to as the cluster statistics)."
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from scipy import sparse

logger = logging.getLogger(__name__)


class ClusterResult:
    """Container for cluster detection results."""
    
    def __init__(
        self,
        cluster_type: str,
        electrodes: List[int],
        cluster_stat: float,
        p_value: float,
    ):
        """
        Parameters
        ----------
        cluster_type : str
            'positive' or 'negative'
        electrodes : list of int
            Electrode IDs in the cluster
        cluster_stat : float
            Cluster statistic (sum of t-values)
        p_value : float
            Monte Carlo p-value
        """
        self.cluster_type = cluster_type
        self.electrodes = electrodes
        self.cluster_stat = cluster_stat
        self.p_value = p_value
    
    def __repr__(self):
        return (
            f"ClusterResult(type={self.cluster_type}, "
            f"n_electrodes={len(self.electrodes)}, "
            f"stat={self.cluster_stat:.2f}, p={self.p_value:.4f})"
        )


def find_clusters_andrillon(
    real_stats: np.ndarray,
    perm_stats: np.ndarray,
    adjacency: sparse.csr_matrix,
    cluster_alpha: float = 0.025,
    montecarlo_alpha: float = 0.05,
    n_permutations: int = 1000,
) -> List[ClusterResult]:
    """
    Detect significant clusters using Andrillon 2020 methodology.
    
    Parameters
    ----------
    real_stats : np.ndarray, shape (n_electrodes, 4)
        Real model statistics: [electrode_id, beta, t_value, p_value]
    perm_stats : np.ndarray, shape (n_electrodes * n_permutations, 5)
        Permuted statistics: [electrode_id, beta, t_value, p_value, perm_id]
    adjacency : scipy.sparse.csr_matrix, shape (n_electrodes, n_electrodes)
        Adjacency matrix defining electrode neighbors
        adjacency[i, j] = 1 if electrodes i and j are neighbors
    cluster_alpha : float, default=0.025
        P-value threshold for candidate clusters
        Andrillon uses 0.025
    montecarlo_alpha : float, default=0.05
        Threshold for Monte Carlo p-value
        Clusters with p_MC < montecarlo_alpha are significant
    n_permutations : int, default=1000
        Number of permutations (for validation)
        
    Returns
    -------
    significant_clusters : list of ClusterResult
        List of significant clusters with their statistics
        
    Notes
    -----
    Algorithm (Andrillon 2020):
    1. Identify candidate electrodes: p < cluster_alpha
    2. Separate by sign (positive vs negative t-values)
    3. Group neighboring electrodes into clusters
    4. Calculate cluster statistic = sum(t-values)
    5. Repeat for each permutation
    6. Compare real clusters against null distribution
    7. Compute Monte Carlo p-value
    
    Positive clusters: p_MC = proportion(real_stat > perm_stats)
    Negative clusters: p_MC = proportion(real_stat < perm_stats)
    
    Examples
    --------
    >>> # Assuming you have real_stats, perm_stats, and adjacency
    >>> clusters = find_clusters_andrillon(
    ...     real_stats, perm_stats, adjacency,
    ...     cluster_alpha=0.025,
    ...     montecarlo_alpha=0.05
    ... )
    >>> for cluster in clusters:
    ...     print(cluster)
    """
    logger.info("Starting cluster detection (Andrillon 2020 method)...")
    
    # Validate inputs
    n_electrodes = real_stats.shape[0]
    expected_perm_rows = n_electrodes * n_permutations
    
    if perm_stats.shape[0] != expected_perm_rows:
        logger.warning(
            f"Expected {expected_perm_rows} permutation rows, "
            f"got {perm_stats.shape[0]}"
        )
    
    # Step 1: Find real clusters
    logger.info("Finding candidate clusters in real data...")
    real_clusters_pos, real_clusters_neg = _find_candidate_clusters(
        real_stats, adjacency, cluster_alpha
    )
    
    logger.info(f"Found {len(real_clusters_pos)} positive candidate clusters")
    logger.info(f"Found {len(real_clusters_neg)} negative candidate clusters")
    
    # Step 2: Find clusters in permuted data
    logger.info("Finding clusters in permuted data...")
    perm_cluster_stats_pos, perm_cluster_stats_neg = _get_permutation_null_distribution(
        perm_stats, adjacency, cluster_alpha, n_permutations
    )
    
    logger.info(f"Generated null distribution with {len(perm_cluster_stats_pos)} positive samples")
    logger.info(f"Generated null distribution with {len(perm_cluster_stats_neg)} negative samples")
    
    # Step 3: Compute Monte Carlo p-values for real clusters
    logger.info("Computing Monte Carlo p-values...")
    significant_clusters = []
    
    # Positive clusters
    for cluster_electrodes, cluster_stat in real_clusters_pos:
        p_mc = _compute_montecarlo_pvalue(
            cluster_stat, perm_cluster_stats_pos, tail='positive'
        )
        
        if p_mc < montecarlo_alpha:
            significant_clusters.append(
                ClusterResult('positive', cluster_electrodes, cluster_stat, p_mc)
            )
            logger.info(
                f"  Significant positive cluster: {len(cluster_electrodes)} electrodes, "
                f"stat={cluster_stat:.2f}, p={p_mc:.4f}"
            )
    
    # Negative clusters
    for cluster_electrodes, cluster_stat in real_clusters_neg:
        p_mc = _compute_montecarlo_pvalue(
            cluster_stat, perm_cluster_stats_neg, tail='negative'
        )
        
        if p_mc < montecarlo_alpha:
            significant_clusters.append(
                ClusterResult('negative', cluster_electrodes, cluster_stat, p_mc)
            )
            logger.info(
                f"  Significant negative cluster: {len(cluster_electrodes)} electrodes, "
                f"stat={cluster_stat:.2f}, p={p_mc:.4f}"
            )
    
    logger.info(f"Cluster detection complete! Found {len(significant_clusters)} significant clusters")
    
    return significant_clusters


def _find_candidate_clusters(
    stats: np.ndarray,
    adjacency: sparse.csr_matrix,
    cluster_alpha: float,
) -> Tuple[List[Tuple[List[int], float]], List[Tuple[List[int], float]]]:
    """
    Find candidate clusters in a single dataset (real or permuted).
    
    Parameters
    ----------
    stats : np.ndarray, shape (n_electrodes, 4)
        [electrode_id, beta, t_value, p_value]
    adjacency : sparse.csr_matrix
        Electrode adjacency matrix
    cluster_alpha : float
        P-value threshold
        
    Returns
    -------
    clusters_pos : list of (electrodes, cluster_stat)
        Positive clusters
    clusters_neg : list of (electrodes, cluster_stat)
        Negative clusters
    """
    # Identify significant electrodes
    sig_mask = stats[:, 3] < cluster_alpha
    
    # Separate by sign
    pos_mask = sig_mask & (stats[:, 2] > 0)
    neg_mask = sig_mask & (stats[:, 2] < 0)
    
    # Get electrode indices and t-values
    pos_electrodes = np.where(pos_mask)[0]
    pos_tvalues = stats[pos_mask, 2]
    
    neg_electrodes = np.where(neg_mask)[0]
    neg_tvalues = stats[neg_mask, 2]
    
    # Group into spatial clusters
    clusters_pos = _group_into_spatial_clusters(
        pos_electrodes, pos_tvalues, adjacency
    )
    
    clusters_neg = _group_into_spatial_clusters(
        neg_electrodes, neg_tvalues, adjacency
    )
    
    return clusters_pos, clusters_neg


def _group_into_spatial_clusters(
    electrode_indices: np.ndarray,
    t_values: np.ndarray,
    adjacency: sparse.csr_matrix,
) -> List[Tuple[List[int], float]]:
    """
    Group electrodes into spatially connected clusters.
    
    Parameters
    ----------
    electrode_indices : np.ndarray
        Indices of significant electrodes
    t_values : np.ndarray
        T-values for these electrodes
    adjacency : sparse.csr_matrix
        Full adjacency matrix
        
    Returns
    -------
    clusters : list of (electrodes, cluster_stat)
        Each cluster is a tuple of (electrode_list, sum_of_tvalues)
    """
    if len(electrode_indices) == 0:
        return []
    
    # Build subgraph for significant electrodes
    n_sig = len(electrode_indices)
    visited = np.zeros(n_sig, dtype=bool)
    clusters = []
    
    for i in range(n_sig):
        if visited[i]:
            continue
        
        # Start new cluster with BFS/DFS
        cluster_indices = []
        cluster_tvalues = []
        stack = [i]
        
        while stack:
            current_idx = stack.pop()
            
            if visited[current_idx]:
                continue
            
            visited[current_idx] = True
            cluster_indices.append(electrode_indices[current_idx])
            cluster_tvalues.append(t_values[current_idx])
            
            # Find neighbors
            current_electrode = electrode_indices[current_idx]
            
            for j in range(n_sig):
                if not visited[j]:
                    neighbor_electrode = electrode_indices[j]
                    
                    # Check if they are adjacent
                    if adjacency[current_electrode, neighbor_electrode] > 0:
                        stack.append(j)
        
        # Calculate cluster statistic (sum of t-values)
        cluster_stat = np.sum(cluster_tvalues)
        clusters.append((cluster_indices, cluster_stat))
    
    return clusters


def _get_permutation_null_distribution(
    perm_stats: np.ndarray,
    adjacency: sparse.csr_matrix,
    cluster_alpha: float,
    n_permutations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build null distribution from permuted data.
    
    For each permutation, find the maximum cluster statistic.
    This forms the null distribution for Monte Carlo testing.
    
    Parameters
    ----------
    perm_stats : np.ndarray, shape (n_electrodes * n_permutations, 5)
        [electrode_id, beta, t_value, p_value, perm_id]
    adjacency : sparse.csr_matrix
        Adjacency matrix
    cluster_alpha : float
        P-value threshold
    n_permutations : int
        Number of permutations
        
    Returns
    -------
    null_dist_pos : np.ndarray, shape (n_permutations,)
        Maximum positive cluster statistic per permutation
    null_dist_neg : np.ndarray, shape (n_permutations,)
        Minimum negative cluster statistic per permutation
    """
    null_dist_pos = []
    null_dist_neg = []
    
    for perm_id in range(n_permutations):
        # Get stats for this permutation
        perm_mask = perm_stats[:, 4] == perm_id
        perm_data = perm_stats[perm_mask, :4]
        
        # Find clusters
        clusters_pos, clusters_neg = _find_candidate_clusters(
            perm_data, adjacency, cluster_alpha
        )
        
        # Get maximum cluster statistic
        if clusters_pos:
            max_pos_stat = max(stat for _, stat in clusters_pos)
            null_dist_pos.append(max_pos_stat)
        else:
            null_dist_pos.append(0.0)
        
        if clusters_neg:
            min_neg_stat = min(stat for _, stat in clusters_neg)
            null_dist_neg.append(min_neg_stat)
        else:
            null_dist_neg.append(0.0)
    
    return np.array(null_dist_pos), np.array(null_dist_neg)


def _compute_montecarlo_pvalue(
    observed_stat: float,
    null_distribution: np.ndarray,
    tail: str = 'positive',
) -> float:
    """
    Compute Monte Carlo p-value.
    
    Parameters
    ----------
    observed_stat : float
        Observed cluster statistic
    null_distribution : np.ndarray
        Null distribution from permutations
    tail : str
        'positive' or 'negative'
        
    Returns
    -------
    p_value : float
        Monte Carlo p-value
        
    Notes
    -----
    Andrillon 2020:
    - Positive cluster: p = proportion(observed > null)
    - Negative cluster: p = proportion(observed < null)
    
    More precisely:
    - Positive: p = mean(observed_stat < null_distribution)
    - Negative: p = mean(observed_stat > null_distribution)
    
    This gives the proportion of permutations with more extreme statistics.
    """
    if tail == 'positive':
        # For positive clusters, we want to know how often permutations
        # produce larger statistics
        p_value = np.mean(observed_stat < null_distribution)
    elif tail == 'negative':
        # For negative clusters, we want to know how often permutations
        # produce smaller (more negative) statistics
        p_value = np.mean(observed_stat > null_distribution)
    else:
        raise ValueError(f"Invalid tail: {tail}. Must be 'positive' or 'negative'")
    
    return p_value


def apply_bonferroni_correction(
    clusters: List[ClusterResult],
    n_comparisons: int,
) -> List[ClusterResult]:
    """
    Apply Bonferroni correction to cluster p-values.
    
    Andrillon 2020: "In cases where several cluster-permutations were 
    performed in the same analysis (Fig. 5 and 6), we corrected the 
    Monte-Carlo p-values of the real clusters with the Bonferroni method."
    
    Parameters
    ----------
    clusters : list of ClusterResult
        Uncorrected clusters
    n_comparisons : int
        Number of comparisons (e.g., number of markers tested)
        
    Returns
    -------
    corrected_clusters : list of ClusterResult
        Clusters with Bonferroni-corrected p-values
        Only clusters surviving correction are returned
    """
    corrected_alpha = 0.05 / n_comparisons
    
    corrected_clusters = []
    for cluster in clusters:
        if cluster.p_value < corrected_alpha:
            corrected_clusters.append(cluster)
    
    logger.info(
        f"Bonferroni correction: {len(corrected_clusters)}/{len(clusters)} "
        f"clusters survive (alpha={corrected_alpha:.4f})"
    )
    
    return corrected_clusters


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_electrodes = 64
    n_permutations = 100
    
    # Create adjacency matrix (simple grid)
    adjacency = sparse.lil_matrix((n_electrodes, n_electrodes))
    for i in range(n_electrodes - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    adjacency = adjacency.tocsr()
    
    # Generate real stats with a cluster effect
    real_stats = np.zeros((n_electrodes, 4))
    real_stats[:, 0] = np.arange(n_electrodes)  # electrode IDs
    real_stats[:, 1] = np.random.randn(n_electrodes) * 0.1  # betas
    real_stats[:, 2] = np.random.randn(n_electrodes)  # t-values
    real_stats[:, 3] = np.random.rand(n_electrodes)  # p-values
    
    # Create a cluster: electrodes 10-15 with strong positive effect
    real_stats[10:16, 2] = np.random.randn(6) + 3.0  # High t-values
    real_stats[10:16, 3] = 0.01  # Low p-values
    
    # Generate permuted stats (null)
    perm_stats = np.zeros((n_electrodes * n_permutations, 5))
    for perm_id in range(n_permutations):
        start_idx = perm_id * n_electrodes
        end_idx = (perm_id + 1) * n_electrodes
        perm_stats[start_idx:end_idx, 0] = np.arange(n_electrodes)
        perm_stats[start_idx:end_idx, 1] = np.random.randn(n_electrodes) * 0.1
        perm_stats[start_idx:end_idx, 2] = np.random.randn(n_electrodes)
        perm_stats[start_idx:end_idx, 3] = np.random.rand(n_electrodes)
        perm_stats[start_idx:end_idx, 4] = perm_id
    
    # Run cluster detection
    print("\n=== Testing cluster detection ===")
    clusters = find_clusters_andrillon(
        real_stats, perm_stats, adjacency,
        cluster_alpha=0.025,
        montecarlo_alpha=0.05,
        n_permutations=n_permutations
    )
    
    print(f"\nFound {len(clusters)} significant clusters:")
    for cluster in clusters:
        print(f"  {cluster}")
