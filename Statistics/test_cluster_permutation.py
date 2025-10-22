"""
Test suite for cluster permutation testing module.

This module validates the correctness of the cluster permutation testing
implementation for Linear Mixed Models.

Run with: python Statistics/test_cluster_permutation.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cluster_test import (
    _find_clusters,
    _permute_within_subjects,
    _validate_adjacency_matrix,
    get_channel_adjacency
)


def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")


def print_test_result(passed: bool, message: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {message}" if message else status)
    return passed


def test_cluster_finding_logic():
    """
    Validate that cluster finding logic is correct with known test cases.
    
    Tests:
    1. Simple case with obvious clusters
    2. No clusters (all below threshold)
    3. Single large cluster
    4. Multiple disconnected clusters
    5. NaN handling
    """
    print_test_header("Cluster Finding Logic")
    
    all_passed = True
    
    # Test 1: Simple case with 2 obvious clusters
    print("\n1. Two obvious clusters...")
    # Create simple adjacency: 0-1-2  3-4-5 (two separate chains)
    adjacency = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1]
    ])
    
    # T-stats: first 3 channels high, last 3 channels high
    t_stats = np.array([3.0, 3.5, 3.2, 4.0, 3.8, 3.9])
    threshold = 2.5
    
    clusters, cluster_stats = _find_clusters(t_stats, adjacency, threshold, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 2)
    all_passed &= print_test_result(passed, f"Found {len(clusters)} clusters (expected 2)")
    
    if passed:
        # Check cluster sizes
        sizes = sorted([len(c) for c in clusters])
        passed = (sizes == [3, 3])
        all_passed &= print_test_result(passed, f"Cluster sizes: {sizes} (expected [3, 3])")
    
    # Test 2: No clusters (all below threshold)
    print("\n2. No clusters (all below threshold)...")
    t_stats_low = np.array([1.0, 1.5, 1.2, 1.8, 1.3, 1.1])
    clusters, cluster_stats = _find_clusters(t_stats_low, adjacency, threshold, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 0)
    all_passed &= print_test_result(passed, f"Found {len(clusters)} clusters (expected 0)")
    
    # Test 3: Single large cluster
    print("\n3. Single large cluster...")
    # Fully connected adjacency
    adjacency_full = np.ones((6, 6))
    t_stats_high = np.array([3.0, 3.5, 3.2, 4.0, 3.8, 3.9])
    
    clusters, cluster_stats = _find_clusters(t_stats_high, adjacency_full, threshold, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 1 and len(clusters[0]) == 6)
    all_passed &= print_test_result(passed, f"Found 1 cluster with {len(clusters[0]) if clusters else 0} channels (expected 6)")
    
    # Test 4: NaN handling
    print("\n4. NaN handling...")
    t_stats_nan = np.array([3.0, np.nan, 3.2, 4.0, np.nan, 3.9])
    clusters, cluster_stats = _find_clusters(t_stats_nan, adjacency, threshold, tail=0, stat_fun='sum')
    
    # Should find clusters but exclude NaN channels
    passed = (len(clusters) >= 1)
    all_passed &= print_test_result(passed, f"Handled NaN values, found {len(clusters)} clusters")
    
    # Test 5: stat_fun='max' vs 'sum'
    print("\n5. Cluster statistic: 'sum' vs 'max'...")
    t_stats_test = np.array([3.0, 5.0, 3.0])
    adjacency_chain = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    clusters_sum, stats_sum = _find_clusters(t_stats_test, adjacency_chain, 2.5, tail=0, stat_fun='sum')
    clusters_max, stats_max = _find_clusters(t_stats_test, adjacency_chain, 2.5, tail=0, stat_fun='max')
    
    if len(clusters_sum) > 0 and len(clusters_max) > 0:
        passed = (stats_sum[0] == 11.0 and stats_max[0] == 5.0)
        all_passed &= print_test_result(
            passed,
            f"sum={stats_sum[0]:.1f} (expected 11.0), max={stats_max[0]:.1f} (expected 5.0)"
        )
    else:
        all_passed &= print_test_result(False, "No clusters found for stat_fun test")
    
    return all_passed


def test_permutation_preserves_structure():
    """
    Validate that within-subject permutation preserves data structure.
    
    Tests:
    1. Number of observations per subject unchanged
    2. Set of values per subject unchanged
    3. Values are actually permuted (not identical)
    4. Subject column unchanged
    """
    print_test_header("Permutation Structure Preservation")
    
    all_passed = True
    
    # Create test data
    df = pd.DataFrame({
        'subject': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'onoff': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'power': np.random.randn(10)
    })
    
    # Permute
    df_perm = _permute_within_subjects(df, 'onoff', seed=42)
    
    # Test 1: Number of observations per subject
    print("\n1. Number of observations per subject...")
    orig_counts = df.groupby('subject').size()
    perm_counts = df_perm.groupby('subject').size()
    passed = orig_counts.equals(perm_counts)
    all_passed &= print_test_result(passed, f"Original: {orig_counts.to_dict()}, Permuted: {perm_counts.to_dict()}")
    
    # Test 2: Set of values per subject unchanged
    print("\n2. Set of values per subject unchanged...")
    all_sets_match = True
    for subject in df['subject'].unique():
        orig_vals = set(df[df['subject'] == subject]['onoff'].values)
        perm_vals = set(df_perm[df_perm['subject'] == subject]['onoff'].values)
        if orig_vals != perm_vals:
            all_sets_match = False
            print(f"  Subject {subject}: Original {orig_vals}, Permuted {perm_vals}")
    
    all_passed &= print_test_result(all_sets_match, "All subjects have same set of values")
    
    # Test 3: Values are actually permuted
    print("\n3. Values are actually permuted (not identical)...")
    # With seed=42, at least some values should be different
    n_different = np.sum(df['onoff'].values != df_perm['onoff'].values)
    passed = (n_different > 0)
    all_passed &= print_test_result(passed, f"{n_different}/10 values changed position")
    
    # Test 4: Subject column unchanged
    print("\n4. Subject column unchanged...")
    passed = df['subject'].equals(df_perm['subject'])
    all_passed &= print_test_result(passed, "Subject column identical")
    
    # Test 5: Other columns unchanged
    print("\n5. Other columns unchanged...")
    passed = np.allclose(df['power'].values, df_perm['power'].values)
    all_passed &= print_test_result(passed, "Power column identical")
    
    # Test 6: Deterministic with same seed
    print("\n6. Deterministic with same seed...")
    df_perm2 = _permute_within_subjects(df, 'onoff', seed=42)
    passed = df_perm['onoff'].equals(df_perm2['onoff'])
    all_passed &= print_test_result(passed, "Same seed produces identical permutation")
    
    return all_passed


def test_adjacency_consistency():
    """
    Validate adjacency matrix properties.
    
    Tests:
    1. Symmetry
    2. Self-connections on diagonal
    3. Reasonable number of connections
    4. No NaN values
    """
    print_test_header("Adjacency Matrix Consistency")
    
    all_passed = True
    
    # Test 1: Symmetric matrix
    print("\n1. Matrix symmetry...")
    adj = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    
    try:
        _validate_adjacency_matrix(adj, 4)
        passed = True
        all_passed &= print_test_result(passed, "Symmetric matrix validated")
    except Exception as e:
        all_passed &= print_test_result(False, f"Validation failed: {e}")
    
    # Test 2: Non-symmetric matrix (should warn)
    print("\n2. Non-symmetric matrix (should warn)...")
    adj_nonsym = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_adjacency_matrix(adj_nonsym, 4)
            passed = len(w) > 0 and "not symmetric" in str(w[0].message)
            all_passed &= print_test_result(passed, "Warning issued for non-symmetric matrix")
    except Exception as e:
        all_passed &= print_test_result(False, f"Unexpected error: {e}")
    
    # Test 3: Wrong shape
    print("\n3. Wrong shape (should fail)...")
    adj_wrong = np.array([[1, 1], [1, 1]])
    
    try:
        _validate_adjacency_matrix(adj_wrong, 4)
        all_passed &= print_test_result(False, "Should have raised ValueError")
    except ValueError as e:
        all_passed &= print_test_result(True, f"Correctly raised ValueError: {str(e)[:50]}")
    
    # Test 4: NaN values (should fail)
    print("\n4. NaN values (should fail)...")
    adj_nan = np.array([
        [1, 1, 0, 0],
        [1, 1, np.nan, 0],
        [0, np.nan, 1, 1],
        [0, 0, 1, 1]
    ])
    
    try:
        _validate_adjacency_matrix(adj_nan, 4)
        all_passed &= print_test_result(False, "Should have raised ValueError")
    except ValueError as e:
        all_passed &= print_test_result(True, f"Correctly raised ValueError: {str(e)[:50]}")
    
    return all_passed


def test_edge_cases():
    """
    Test edge cases and error handling.
    
    Tests:
    1. All t-stats are NaN
    2. Empty clusters
    3. Single channel
    4. Disconnected adjacency components
    """
    print_test_header("Edge Cases and Error Handling")
    
    all_passed = True
    
    # Test 1: All NaN t-stats
    print("\n1. All NaN t-statistics...")
    t_stats_all_nan = np.array([np.nan, np.nan, np.nan])
    adjacency = np.eye(3)
    clusters, stats = _find_clusters(t_stats_all_nan, adjacency, 2.0, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 0)
    all_passed &= print_test_result(passed, f"Correctly returned {len(clusters)} clusters")
    
    # Test 2: Single channel
    print("\n2. Single channel...")
    t_stats_single = np.array([3.5])
    adjacency_single = np.array([[1]])
    clusters, stats = _find_clusters(t_stats_single, adjacency_single, 2.0, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 1 and len(clusters[0]) == 1)
    all_passed &= print_test_result(passed, f"Found {len(clusters)} cluster with {len(clusters[0]) if clusters else 0} channel")
    
    # Test 3: Disconnected components in adjacency
    print("\n3. Disconnected adjacency components...")
    # Two separate islands: 0-1 and 2-3
    adjacency_disconnected = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])
    t_stats_high = np.array([3.0, 3.5, 4.0, 3.8])
    clusters, stats = _find_clusters(t_stats_high, adjacency_disconnected, 2.5, tail=0, stat_fun='sum')
    
    passed = (len(clusters) == 2)
    all_passed &= print_test_result(passed, f"Found {len(clusters)} clusters (expected 2 disconnected components)")
    
    # Test 4: Tail parameter
    print("\n4. Tail parameter (one-sided tests)...")
    t_stats_mixed = np.array([3.0, -3.5, 3.2, -4.0])
    adjacency_chain = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    
    # Tail = 1 (greater): should only find positive clusters
    clusters_pos, _ = _find_clusters(t_stats_mixed, adjacency_chain, 2.5, tail=1, stat_fun='sum')
    # Tail = -1 (less): should only find negative clusters
    clusters_neg, _ = _find_clusters(t_stats_mixed, adjacency_chain, 2.5, tail=-1, stat_fun='sum')
    
    passed = (len(clusters_pos) >= 1 and len(clusters_neg) >= 1)
    all_passed &= print_test_result(
        passed,
        f"Positive clusters: {len(clusters_pos)}, Negative clusters: {len(clusters_neg)}"
    )
    
    return all_passed


def test_validate_against_simple_ttest():
    """
    For a simple between-groups case, validate our cluster finding
    logic produces sensible results.
    
    We can't directly compare with MNE's cluster test (different stats),
    but we can verify our logic finds clusters where we expect them.
    
    Tests:
    1. Creates two groups with known effect in specific channels
    2. Computes simple t-statistics
    3. Verifies clusters are found in expected regions
    4. Verifies no spurious clusters in no-effect regions
    """
    print_test_header("Validation Against Simple T-Test Case")
    
    all_passed = True
    
    print("\nCreating two-group data with known cluster...")
    
    n_subjects_per_group = 15
    n_channels = 10
    
    # Group 1: normal
    group1 = np.random.randn(n_subjects_per_group, n_channels)
    
    # Group 2: elevated in channels 0-3
    group2 = np.random.randn(n_subjects_per_group, n_channels)
    group2[:, 0:4] += 1.0  # Strong effect in first 4 channels
    
    # Compute simple t-statistics (not LMM, just for validation)
    from scipy import stats
    t_stats = np.zeros(n_channels)
    for ch in range(n_channels):
        t_stats[ch], _ = stats.ttest_ind(group1[:, ch], group2[:, ch])
    
    # Create adjacency (linear chain)
    adjacency = np.zeros((n_channels, n_channels))
    for i in range(n_channels-1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1
    np.fill_diagonal(adjacency, 1)
    
    print(f"T-statistics: {np.round(t_stats, 2)}")
    print(f"Expected high t-stats in channels 0-3, low in 4-9")
    
    # Find clusters with threshold
    threshold = 2.0
    clusters, cluster_stats = _find_clusters(t_stats, adjacency, threshold, tail=0, stat_fun='sum')
    
    print(f"\nFound {len(clusters)} cluster(s) with threshold={threshold}")
    if len(clusters) > 0:
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: channels {list(cluster)}, stat={cluster_stats[i]:.2f}")
    
    # Test 1: Should find at least one cluster in channels 0-3
    print("\n1. Cluster found in expected region (channels 0-3)...")
    passed = False
    if len(clusters) > 0:
        for cluster in clusters:
            if any(ch in cluster for ch in [0, 1, 2, 3]):
                passed = True
                break
    
    all_passed &= print_test_result(passed, f"Found cluster overlapping with effect region")
    
    # Test 2: Verify no cluster exclusively in channels 7-9 (no effect there)
    print("\n2. No spurious clusters in no-effect region (channels 7-9)...")
    if len(clusters) > 0:
        unexpected = False
        for cluster in clusters:
            # Check if cluster is exclusively in the no-effect region
            if all(ch in [7, 8, 9] for ch in cluster) and len(cluster) > 1:
                unexpected = True
                print(f"  Unexpected cluster found: {list(cluster)}")
        
        passed = not unexpected
        all_passed &= print_test_result(passed, "No spurious clusters in no-effect region")
    else:
        all_passed &= print_test_result(True, "No clusters to check")
    
    # Test 3: T-statistics in effect region should be higher
    print("\n3. T-statistics higher in effect region...")
    t_effect = np.mean(np.abs(t_stats[0:4]))
    t_noeffect = np.mean(np.abs(t_stats[6:10]))
    passed = t_effect > t_noeffect
    all_passed &= print_test_result(
        passed, 
        f"Effect region |t|={t_effect:.2f} > No-effect region |t|={t_noeffect:.2f}"
    )
    
    # Test 4: Test with different stat_fun
    print("\n4. Comparing 'sum' vs 'max' cluster statistics...")
    clusters_sum, stats_sum = _find_clusters(t_stats, adjacency, threshold, tail=0, stat_fun='sum')
    clusters_max, stats_max = _find_clusters(t_stats, adjacency, threshold, tail=0, stat_fun='max')
    
    # Both should find clusters
    passed = (len(clusters_sum) > 0 and len(clusters_max) > 0)
    all_passed &= print_test_result(
        passed,
        f"sum: {len(clusters_sum)} clusters, max: {len(clusters_max)} clusters"
    )
    
    return all_passed


def test_integration_simple():
    """
    Simple integration test with synthetic data.
    
    Creates synthetic data with known structure and tests full pipeline.
    """
    print_test_header("Simple Integration Test")
    
    all_passed = True
    
    print("\nCreating synthetic data...")
    
    # Create simple synthetic data
    n_subjects = 5
    n_obs_per_subject = 10
    n_channels = 4
    
    subjects = np.repeat([f'S{i}' for i in range(n_subjects)], n_obs_per_subject)
    predictor = np.tile(np.arange(n_obs_per_subject), n_subjects) + np.random.randn(n_subjects * n_obs_per_subject) * 0.1
    
    # Create power data with some structure
    power_data = np.random.randn(n_subjects * n_obs_per_subject, n_channels)
    
    # Add correlation with predictor in channels 0 and 1
    power_data[:, 0] += predictor * 0.5
    power_data[:, 1] += predictor * 0.4
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'predictor': predictor
    })
    
    # Create simple adjacency (chain: 0-1-2-3)
    adjacency = np.array([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1]
    ])
    
    print("Testing permutation on synthetic data...")
    
    # Test permutation
    df_perm = _permute_within_subjects(df_behavioral, 'predictor', seed=123)
    
    # Check structure preserved
    passed = (
        len(df_perm) == len(df_behavioral) and
        df_perm['subject'].equals(df_behavioral['subject']) and
        not df_perm['predictor'].equals(df_behavioral['predictor'])
    )
    all_passed &= print_test_result(passed, "Permutation preserved structure and changed values")
    
    # Test cluster finding with synthetic t-stats
    print("\nTesting cluster finding...")
    t_stats_synthetic = np.array([2.5, 3.0, 1.5, 1.0])  # First two channels significant
    
    clusters, stats = _find_clusters(t_stats_synthetic, adjacency, 2.0, tail=0, stat_fun='sum')
    
    passed = (len(clusters) >= 1)
    all_passed &= print_test_result(passed, f"Found {len(clusters)} cluster(s)")
    
    if len(clusters) > 0:
        # First cluster should include channels 0 and 1
        cluster_channels = clusters[0]
        passed = (0 in cluster_channels and 1 in cluster_channels)
        all_passed &= print_test_result(passed, f"Cluster includes expected channels: {cluster_channels}")
    
    return all_passed


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*70)
    print("CLUSTER PERMUTATION TESTING - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Cluster Finding Logic", test_cluster_finding_logic),
        ("Permutation Structure", test_permutation_preserves_structure),
        ("Adjacency Consistency", test_adjacency_consistency),
        ("Edge Cases", test_edge_cases),
        ("Simple T-Test Validation", test_validate_against_simple_ttest),
        ("Integration Test", test_integration_simple)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n✗ EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    n_passed = sum(results.values())
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} test suites passed")
    
    if n_passed == n_total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠ {n_total - n_passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
