"""
Verify that our clustering fix matches MNE's approach.

This test compares our implementation against MNE's _get_components logic
to ensure we're correctly handling adjacency matrices.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_array
from scipy.sparse.csgraph import connected_components
import sys
sys.path.insert(0, '/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/Statistics')
from cluster_test import _find_clusters_single_tail


def mne_get_components(x_in, adjacency):
    """
    MNE's approach to finding connected components.
    
    Adapted from MNE-Python's _get_components function.
    """
    # Filter adjacency to only keep edges where both endpoints are in x_in
    mask = np.logical_and(x_in[adjacency.row], x_in[adjacency.col])
    data = adjacency.data[mask]
    row = adjacency.row[mask]
    col = adjacency.col[mask]
    shape = adjacency.shape
    
    # Add self-loops for all suprathreshold vertices
    idx = np.where(x_in)[0]
    row = np.concatenate((row, idx))
    col = np.concatenate((col, idx))
    data = np.concatenate((data, np.ones(len(idx), dtype=data.dtype)))
    
    # Create filtered adjacency matrix
    adjacency_filtered = coo_array((data, (row, col)), shape=shape)
    
    # Find connected components
    _, components = connected_components(adjacency_filtered)
    
    # Extract clusters
    clusters = []
    for comp_idx in range(np.max(components) + 1):
        cluster = np.where(components == comp_idx)[0]
        # Only include clusters with suprathreshold vertices
        cluster = cluster[x_in[cluster]]
        if len(cluster) > 0:
            clusters.append(cluster)
    
    return clusters


def test_equivalence():
    """
    Test that our implementation produces the same clusters as MNE's approach.
    """
    print("\n" + "="*70)
    print("VERIFICATION: Our Fix vs MNE's Approach")
    print("="*70)
    
    # Create test adjacency: A-B-C-D-E (5 channels in a line)
    adjacency_dense = np.array([
        [0, 1, 0, 0, 0],  # A connects to B
        [1, 0, 1, 0, 0],  # B connects to A and C
        [0, 1, 0, 1, 0],  # C connects to B and D
        [0, 0, 1, 0, 1],  # D connects to C and E
        [0, 0, 0, 1, 0],  # E connects to D
    ])
    
    # Convert to sparse COO format (like MNE uses)
    adjacency_sparse = coo_array(adjacency_dense)
    
    print("\nTest Case 1: Non-adjacent channels (A, C, E suprathreshold)")
    print("="*70)
    
    # Test case 1: A, C, E are suprathreshold (non-adjacent)
    t_stats = np.array([3.0, 1.0, 3.0, 1.0, 3.0])
    threshold = 2.0
    threshold_mask = t_stats > threshold
    
    print(f"Suprathreshold channels: {np.where(threshold_mask)[0]} (A=0, C=2, E=4)")
    
    # Get clusters using MNE's approach
    mne_clusters = mne_get_components(threshold_mask, adjacency_sparse)
    mne_clusters = sorted([sorted(c.tolist()) for c in mne_clusters])
    
    # Get clusters using our approach
    our_clusters, _ = _find_clusters_single_tail(
        t_stats, adjacency_dense, threshold_mask, stat_fun='sum', t_power=1.0
    )
    our_clusters = sorted([sorted(c.tolist()) for c in our_clusters])
    
    print(f"\nMNE clusters: {mne_clusters}")
    print(f"Our clusters: {our_clusters}")
    
    if mne_clusters == our_clusters:
        print("✓ PASS: Clusters match MNE's implementation")
        test1_pass = True
    else:
        print("✗ FAIL: Clusters differ from MNE's implementation")
        test1_pass = False
    
    print("\n" + "="*70)
    print("Test Case 2: Adjacent channels (A, B, C suprathreshold)")
    print("="*70)
    
    # Test case 2: A, B, C are suprathreshold (adjacent)
    t_stats = np.array([3.0, 3.0, 3.0, 1.0, 1.0])
    threshold_mask = t_stats > threshold
    
    print(f"Suprathreshold channels: {np.where(threshold_mask)[0]} (A=0, B=1, C=2)")
    
    # Get clusters using MNE's approach
    mne_clusters = mne_get_components(threshold_mask, adjacency_sparse)
    mne_clusters = sorted([sorted(c.tolist()) for c in mne_clusters])
    
    # Get clusters using our approach
    our_clusters, _ = _find_clusters_single_tail(
        t_stats, adjacency_dense, threshold_mask, stat_fun='sum', t_power=1.0
    )
    our_clusters = sorted([sorted(c.tolist()) for c in our_clusters])
    
    print(f"\nMNE clusters: {mne_clusters}")
    print(f"Our clusters: {our_clusters}")
    
    if mne_clusters == our_clusters:
        print("✓ PASS: Clusters match MNE's implementation")
        test2_pass = True
    else:
        print("✗ FAIL: Clusters differ from MNE's implementation")
        test2_pass = False
    
    print("\n" + "="*70)
    print("Test Case 3: Complex pattern (A, B, D, E suprathreshold)")
    print("="*70)
    
    # Test case 3: A-B and D-E are two separate clusters
    t_stats = np.array([3.0, 3.0, 1.0, 3.0, 3.0])
    threshold_mask = t_stats > threshold
    
    print(f"Suprathreshold channels: {np.where(threshold_mask)[0]} (A=0, B=1, D=3, E=4)")
    print("Expected: Two clusters [A,B] and [D,E]")
    
    # Get clusters using MNE's approach
    mne_clusters = mne_get_components(threshold_mask, adjacency_sparse)
    mne_clusters = sorted([sorted(c.tolist()) for c in mne_clusters])
    
    # Get clusters using our approach
    our_clusters, _ = _find_clusters_single_tail(
        t_stats, adjacency_dense, threshold_mask, stat_fun='sum', t_power=1.0
    )
    our_clusters = sorted([sorted(c.tolist()) for c in our_clusters])
    
    print(f"\nMNE clusters: {mne_clusters}")
    print(f"Our clusters: {our_clusters}")
    
    if mne_clusters == our_clusters:
        print("✓ PASS: Clusters match MNE's implementation")
        test3_pass = True
    else:
        print("✗ FAIL: Clusters differ from MNE's implementation")
        test3_pass = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Test 1 (Non-adjacent): {'PASS ✓' if test1_pass else 'FAIL ✗'}")
    print(f"Test 2 (Adjacent): {'PASS ✓' if test2_pass else 'FAIL ✗'}")
    print(f"Test 3 (Complex): {'PASS ✓' if test3_pass else 'FAIL ✗'}")
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✓ ALL TESTS PASSED")
        print("Our implementation correctly matches MNE's clustering logic!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = test_equivalence()
    sys.exit(0 if success else 1)
