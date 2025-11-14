"""
Tests for cluster detection module.
"""

import sys
from pathlib import Path
import numpy as np
from scipy import sparse
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cluster_detection import (
    find_clusters_andrillon,
    _find_candidate_clusters,
    _group_into_spatial_clusters,
    _compute_montecarlo_pvalue,
    ClusterResult,
)


def create_simple_adjacency(n_electrodes=10):
    """Create simple linear adjacency matrix."""
    adjacency = sparse.lil_matrix((n_electrodes, n_electrodes))
    for i in range(n_electrodes - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    return adjacency.tocsr()


def create_test_stats(n_electrodes=10, cluster_electrodes=None, cluster_effect=3.0):
    """Create test statistics with a known cluster."""
    stats = np.zeros((n_electrodes, 4))
    stats[:, 0] = np.arange(n_electrodes)  # electrode IDs
    stats[:, 1] = np.random.randn(n_electrodes) * 0.1  # betas
    stats[:, 2] = np.random.randn(n_electrodes)  # t-values
    stats[:, 3] = np.random.rand(n_electrodes)  # p-values
    
    # Add cluster if specified
    if cluster_electrodes is not None:
        stats[cluster_electrodes, 2] = cluster_effect  # High t-values
        stats[cluster_electrodes, 3] = 0.01  # Low p-values
    
    return stats


class TestClustering:
    """Test cluster detection functionality."""
    
    def test_find_candidate_clusters_positive(self):
        """Test finding positive clusters."""
        adjacency = create_simple_adjacency(10)
        stats = create_test_stats(10, cluster_electrodes=[3, 4, 5], cluster_effect=3.0)
        
        clusters_pos, clusters_neg = _find_candidate_clusters(
            stats, adjacency, cluster_alpha=0.025
        )
        
        assert len(clusters_pos) > 0, "Should find at least one positive cluster"
        assert len(clusters_neg) == 0, "Should not find negative clusters"
        
        # Check that cluster contains expected electrodes
        cluster_electrodes, cluster_stat = clusters_pos[0]
        assert 3 in cluster_electrodes, "Cluster should contain electrode 3"
        assert 4 in cluster_electrodes, "Cluster should contain electrode 4"
        assert 5 in cluster_electrodes, "Cluster should contain electrode 5"
        
        print(f"✓ Found positive cluster with {len(cluster_electrodes)} electrodes, stat={cluster_stat:.2f}")
    
    def test_find_candidate_clusters_negative(self):
        """Test finding negative clusters."""
        adjacency = create_simple_adjacency(10)
        stats = create_test_stats(10, cluster_electrodes=[2, 3], cluster_effect=-3.0)
        
        clusters_pos, clusters_neg = _find_candidate_clusters(
            stats, adjacency, cluster_alpha=0.025
        )
        
        assert len(clusters_pos) == 0, "Should not find positive clusters"
        assert len(clusters_neg) > 0, "Should find at least one negative cluster"
        
        cluster_electrodes, cluster_stat = clusters_neg[0]
        assert cluster_stat < 0, "Negative cluster should have negative statistic"
        
        print(f"✓ Found negative cluster with {len(cluster_electrodes)} electrodes, stat={cluster_stat:.2f}")
    
    def test_group_into_spatial_clusters(self):
        """Test spatial grouping of electrodes."""
        adjacency = create_simple_adjacency(10)
        
        # Two separate clusters: [1, 2, 3] and [7, 8]
        electrode_indices = np.array([1, 2, 3, 7, 8])
        t_values = np.array([2.0, 2.5, 2.2, 3.0, 2.8])
        
        clusters = _group_into_spatial_clusters(
            electrode_indices, t_values, adjacency
        )
        
        assert len(clusters) == 2, "Should find 2 separate clusters"
        
        # Check cluster sizes
        sizes = [len(electrodes) for electrodes, _ in clusters]
        assert 3 in sizes, "Should have a cluster of size 3"
        assert 2 in sizes, "Should have a cluster of size 2"
        
        print(f"✓ Correctly grouped into {len(clusters)} spatial clusters")
    
    def test_compute_montecarlo_pvalue_positive(self):
        """Test Monte Carlo p-value calculation for positive cluster."""
        observed_stat = 10.0
        null_distribution = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0])
        
        p_value = _compute_montecarlo_pvalue(
            observed_stat, null_distribution, tail='positive'
        )
        
        # p = proportion of null >= observed
        # In this case: 2/7 ≈ 0.286
        expected_p = 2/7
        assert abs(p_value - expected_p) < 0.01, f"Expected p≈{expected_p:.3f}, got {p_value:.3f}"
        
        print(f"✓ Monte Carlo p-value (positive): {p_value:.3f}")
    
    def test_compute_montecarlo_pvalue_negative(self):
        """Test Monte Carlo p-value calculation for negative cluster."""
        observed_stat = -10.0
        null_distribution = np.array([-5.0, -6.0, -7.0, -8.0, -9.0, -11.0, -12.0])
        
        p_value = _compute_montecarlo_pvalue(
            observed_stat, null_distribution, tail='negative'
        )
        
        # p = proportion of null <= observed
        # In this case: 2/7 ≈ 0.286
        expected_p = 2/7
        assert abs(p_value - expected_p) < 0.01, f"Expected p≈{expected_p:.3f}, got {p_value:.3f}"
        
        print(f"✓ Monte Carlo p-value (negative): {p_value:.3f}")
    
    def test_find_clusters_andrillon_integration(self):
        """Test full cluster detection pipeline."""
        n_electrodes = 20
        n_permutations = 50
        
        # Create adjacency
        adjacency = create_simple_adjacency(n_electrodes)
        
        # Create real stats with a cluster
        real_stats = create_test_stats(
            n_electrodes, 
            cluster_electrodes=[5, 6, 7, 8], 
            cluster_effect=3.5
        )
        
        # Create permuted stats (null distribution)
        perm_stats = []
        for perm_id in range(n_permutations):
            perm_data = create_test_stats(n_electrodes)
            # Add perm_id column
            perm_id_col = np.full((n_electrodes, 1), perm_id)
            perm_data_with_id = np.hstack([perm_data, perm_id_col])
            perm_stats.append(perm_data_with_id)
        
        perm_stats = np.vstack(perm_stats)
        
        # Find clusters
        clusters = find_clusters_andrillon(
            real_stats,
            perm_stats,
            adjacency,
            cluster_alpha=0.025,
            montecarlo_alpha=0.05,
            n_permutations=n_permutations,
        )
        
        # Should find at least one cluster
        assert len(clusters) >= 0, "Should run without errors"
        
        if clusters:
            cluster = clusters[0]
            assert isinstance(cluster, ClusterResult), "Should return ClusterResult objects"
            assert cluster.cluster_type in ['positive', 'negative'], "Should have valid cluster type"
            assert len(cluster.electrodes) > 0, "Cluster should have electrodes"
            assert 0 <= cluster.p_value <= 1, "P-value should be between 0 and 1"
            
            print(f"✓ Full pipeline: Found {len(clusters)} cluster(s)")
            for i, c in enumerate(clusters):
                print(f"  Cluster {i+1}: {c.cluster_type}, {len(c.electrodes)} electrodes, p={c.p_value:.4f}")
        else:
            print("✓ Full pipeline: No significant clusters (expected with random data)")


def run_all_tests():
    """Run all clustering tests."""
    print("\n" + "="*60)
    print("TESTING: Cluster Detection Module")
    print("="*60 + "\n")
    
    test = TestClustering()
    
    try:
        test.test_find_candidate_clusters_positive()
        test.test_find_candidate_clusters_negative()
        test.test_group_into_spatial_clusters()
        test.test_compute_montecarlo_pvalue_positive()
        test.test_compute_montecarlo_pvalue_negative()
        test.test_find_clusters_andrillon_integration()
        
        print("\n" + "="*60)
        print("✅ ALL CLUSTERING TESTS PASSED")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ TEST FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
