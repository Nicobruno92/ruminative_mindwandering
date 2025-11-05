"""
Test to verify that the adjacency fix prevents non-adjacent clusters.

This test creates a simple scenario where the old subgraph approach would
incorrectly merge non-adjacent channels into a single cluster.
"""

import numpy as np
import sys
sys.path.insert(0, '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics')
from cluster_test import _find_clusters_single_tail, _get_cluster_extents


def test_non_adjacent_channels():
    """
    Test that non-adjacent channels don't form a cluster.
    
    Setup: 5 channels in a line: A - B - C - D - E
    Scenario: Channels A, C, E are suprathreshold (B and D are not)
    
    OLD BUG: Would create subgraph with A-C and C-E connections
             (because in subgraph, A and C are adjacent indices 0 and 1)
    CORRECT: Should create 3 separate clusters (A alone, C alone, E alone)
    """
    print("\n" + "="*70)
    print("TEST: Non-adjacent channels should NOT form a cluster")
    print("="*70)
    
    # Create adjacency for 5 channels in a line: A-B-C-D-E
    adjacency = np.array([
        [0, 1, 0, 0, 0],  # A connects to B
        [1, 0, 1, 0, 0],  # B connects to A and C
        [0, 1, 0, 1, 0],  # C connects to B and D
        [0, 0, 1, 0, 1],  # D connects to C and E
        [0, 0, 0, 1, 0],  # E connects to D
    ])
    
    print("\nAdjacency structure: A-B-C-D-E")
    print("Adjacency matrix:")
    print(adjacency)
    
    # Create t-statistics where A, C, E are suprathreshold
    # B and D are below threshold
    t_stats = np.array([3.0, 1.0, 3.0, 1.0, 3.0])  # A, C, E above threshold
    threshold = 2.0
    
    print(f"\nT-statistics: {t_stats}")
    print(f"Threshold: {threshold}")
    print(f"Suprathreshold channels: A (idx=0), C (idx=2), E (idx=4)")
    print(f"Below threshold: B (idx=1), D (idx=3)")
    
    # Create threshold mask
    threshold_mask = t_stats > threshold
    
    # Run cluster finding
    clusters, cluster_stats = _find_clusters_single_tail(
        t_stats, adjacency, threshold_mask, stat_fun='sum', t_power=1.0
    )
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Number of clusters found: {len(clusters)}")
    
    for i, cluster in enumerate(clusters):
        channel_names = ['A', 'B', 'C', 'D', 'E']
        cluster_names = [channel_names[idx] for idx in cluster]
        print(f"  Cluster {i+1}: channels {cluster} ({', '.join(cluster_names)}), "
              f"statistic={cluster_stats[i]:.3f}")
    
    # Verify correctness
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    
    if len(clusters) == 3:
        print("✓ PASS: Found 3 separate clusters (correct)")
        print("  Non-adjacent channels are correctly separated")
        
        # Check each cluster has only 1 channel
        all_single = all(len(c) == 1 for c in clusters)
        if all_single:
            print("✓ PASS: Each cluster contains exactly 1 channel (correct)")
        else:
            print("✗ FAIL: Some clusters have multiple channels (incorrect)")
            return False
            
        return True
    else:
        print(f"✗ FAIL: Found {len(clusters)} clusters, expected 3")
        print("  Non-adjacent channels were incorrectly merged!")
        return False


def test_adjacent_channels():
    """
    Test that adjacent channels DO form a cluster.
    
    Setup: 5 channels in a line: A - B - C - D - E
    Scenario: Channels A, B, C are suprathreshold (D and E are not)
    
    EXPECTED: Should create 1 cluster containing A, B, C
    """
    print("\n" + "="*70)
    print("TEST: Adjacent channels SHOULD form a cluster")
    print("="*70)
    
    # Create adjacency for 5 channels in a line: A-B-C-D-E
    adjacency = np.array([
        [0, 1, 0, 0, 0],  # A connects to B
        [1, 0, 1, 0, 0],  # B connects to A and C
        [0, 1, 0, 1, 0],  # C connects to B and D
        [0, 0, 1, 0, 1],  # D connects to C and E
        [0, 0, 0, 1, 0],  # E connects to D
    ])
    
    print("\nAdjacency structure: A-B-C-D-E")
    
    # Create t-statistics where A, B, C are suprathreshold
    t_stats = np.array([3.0, 3.0, 3.0, 1.0, 1.0])  # A, B, C above threshold
    threshold = 2.0
    
    print(f"\nT-statistics: {t_stats}")
    print(f"Threshold: {threshold}")
    print(f"Suprathreshold channels: A (idx=0), B (idx=1), C (idx=2)")
    print(f"Below threshold: D (idx=3), E (idx=4)")
    
    # Create threshold mask
    threshold_mask = t_stats > threshold
    
    # Run cluster finding
    clusters, cluster_stats = _find_clusters_single_tail(
        t_stats, adjacency, threshold_mask, stat_fun='sum', t_power=1.0
    )
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Number of clusters found: {len(clusters)}")
    
    for i, cluster in enumerate(clusters):
        channel_names = ['A', 'B', 'C', 'D', 'E']
        cluster_names = [channel_names[idx] for idx in cluster]
        print(f"  Cluster {i+1}: channels {cluster} ({', '.join(cluster_names)}), "
              f"statistic={cluster_stats[i]:.3f}")
    
    # Verify correctness
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    
    if len(clusters) == 1:
        print("✓ PASS: Found 1 cluster (correct)")
        
        # Check cluster contains A, B, C
        expected_channels = {0, 1, 2}
        actual_channels = set(clusters[0])
        
        if actual_channels == expected_channels:
            print("✓ PASS: Cluster contains channels A, B, C (correct)")
            
            # Check statistic
            expected_stat = 9.0  # 3 + 3 + 3
            if abs(cluster_stats[0] - expected_stat) < 0.001:
                print(f"✓ PASS: Cluster statistic is {cluster_stats[0]:.3f} (correct)")
                return True
            else:
                print(f"✗ FAIL: Cluster statistic is {cluster_stats[0]:.3f}, expected {expected_stat}")
                return False
        else:
            print(f"✗ FAIL: Cluster contains {actual_channels}, expected {expected_channels}")
            return False
    else:
        print(f"✗ FAIL: Found {len(clusters)} clusters, expected 1")
        return False


def test_tfce_extents():
    """
    Test that TFCE cluster extents are computed correctly.
    
    Same scenario as test_non_adjacent_channels but for TFCE.
    """
    print("\n" + "="*70)
    print("TEST: TFCE cluster extents for non-adjacent channels")
    print("="*70)
    
    # Create adjacency for 5 channels in a line: A-B-C-D-E
    adjacency = np.array([
        [0, 1, 0, 0, 0],  # A connects to B
        [1, 0, 1, 0, 0],  # B connects to A and C
        [0, 1, 0, 1, 0],  # C connects to B and D
        [0, 0, 1, 0, 1],  # D connects to C and E
        [0, 0, 0, 1, 0],  # E connects to D
    ])
    
    print("\nAdjacency structure: A-B-C-D-E")
    
    # Create mask where A, C, E are suprathreshold
    mask = np.array([True, False, True, False, True])
    
    print(f"Suprathreshold mask: {mask}")
    print(f"Suprathreshold channels: A (idx=0), C (idx=2), E (idx=4)")
    
    # Get cluster extents
    extents = _get_cluster_extents(mask, adjacency)
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Cluster extents: {extents}")
    
    # Verify correctness
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")
    
    # Each suprathreshold channel should have extent=1 (isolated)
    expected_extents = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    
    if np.allclose(extents, expected_extents):
        print("✓ PASS: Each non-adjacent channel has extent=1 (correct)")
        print("  Channels A, C, E are correctly identified as separate clusters")
        return True
    else:
        print(f"✗ FAIL: Extents are {extents}, expected {expected_extents}")
        print("  Non-adjacent channels were incorrectly merged!")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING ADJACENCY FIX FOR CLUSTERING")
    print("="*70)
    print("\nThis test verifies that the clustering algorithm correctly")
    print("identifies spatially contiguous clusters and does NOT merge")
    print("non-adjacent channels.")
    
    # Run tests
    test1_pass = test_non_adjacent_channels()
    test2_pass = test_adjacent_channels()
    test3_pass = test_tfce_extents()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Test 1 (Non-adjacent channels): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (Adjacent channels): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    print(f"Test 3 (TFCE extents): {'PASS ✓' if test3_pass else 'FAIL ✗'}")
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✓ ALL TESTS PASSED - Adjacency fix is working correctly!")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED - There may still be issues with the fix")
        sys.exit(1)
