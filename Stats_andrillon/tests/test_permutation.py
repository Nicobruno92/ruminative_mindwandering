"""
Tests for LMM permutation module.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lmm_permutation import (
    fit_lmm_with_permutations,
    _permute_predictor,
    _fit_single_lmm,
    format_results_for_clustering,
)


def generate_test_data(n_subjects=10, n_tasks=2, n_probes_per_task=20, effect_size=0.05):
    """Generate synthetic data for testing."""
    np.random.seed(42)
    
    data_list = []
    for subj in range(n_subjects):
        for task in range(1, n_tasks + 1):
            for probe in range(n_probes_per_task):
                onoff = np.random.randint(0, 100)
                # Add effect: higher onoff -> higher power
                power = 10 + effect_size * onoff + np.random.randn() * 2
                
                data_list.append({
                    'subject': subj,
                    'task': task,
                    'onoff': onoff,
                    'power': power,
                })
    
    return pd.DataFrame(data_list)


class TestPermutation:
    """Test permutation functionality."""
    
    def test_permute_predictor_preserves_groups(self):
        """Test that permutation preserves within-group structure."""
        data = generate_test_data(n_subjects=5, n_tasks=2, n_probes_per_task=10)
        
        # Get original values per group
        original_groups = {}
        for (subj, task), group in data.groupby(['subject', 'task']):
            original_groups[(subj, task)] = sorted(group['onoff'].values)
        
        # Permute
        data_perm = _permute_predictor(data.copy(), 'onoff', ['subject', 'task'])
        
        # Check that each group has same values (just reordered)
        for (subj, task), group in data_perm.groupby(['subject', 'task']):
            permuted_values = sorted(group['onoff'].values)
            original_values = original_groups[(subj, task)]
            
            assert len(permuted_values) == len(original_values)
            np.testing.assert_array_equal(permuted_values, original_values)
        
        print("✓ Permutation preserves within-group values")
    
    def test_permute_predictor_changes_order(self):
        """Test that permutation actually changes the order."""
        data = generate_test_data(n_subjects=5, n_tasks=2, n_probes_per_task=20)
        
        original_values = data['onoff'].values.copy()
        
        # Permute multiple times
        changed = False
        for _ in range(10):
            data_perm = _permute_predictor(data.copy(), 'onoff', ['subject', 'task'])
            permuted_values = data_perm['onoff'].values
            
            if not np.array_equal(original_values, permuted_values):
                changed = True
                break
        
        assert changed, "Permutation should change order"
        print("✓ Permutation changes order")
    
    def test_fit_single_lmm(self):
        """Test single LMM fitting."""
        data = generate_test_data(n_subjects=10, n_tasks=2, n_probes_per_task=30)
        
        stats = _fit_single_lmm(
            data,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            method="powell",
            maxiter=5000,
        )
        
        assert stats.shape == (3,), "Should return [beta, t, p]"
        assert not np.isnan(stats[0]), "Beta should not be NaN"
        assert not np.isnan(stats[1]), "T-value should not be NaN"
        assert not np.isnan(stats[2]), "P-value should not be NaN"
        assert 0 <= stats[2] <= 1, "P-value should be between 0 and 1"
        
        print(f"✓ LMM fitting works: beta={stats[0]:.4f}, t={stats[1]:.4f}, p={stats[2]:.4f}")
    
    def test_fit_lmm_with_permutations(self):
        """Test LMM with permutations."""
        data = generate_test_data(n_subjects=10, n_tasks=2, n_probes_per_task=30)
        
        real_stats, perm_stats = fit_lmm_with_permutations(
            data,
            formula="power ~ onoff + (1|subject)",
            predictor_of_interest="onoff",
            n_permutations=10,  # Small number for speed
            permutation_within=['subject', 'task'],
            random_state=42,
        )
        
        # Check real stats (returns 1D array of shape (3,))
        assert real_stats.shape == (3,), "Real stats should be (3,)"
        assert not np.isnan(real_stats[0]), "Real beta should not be NaN"
        
        # Check permuted stats
        assert perm_stats.shape == (10, 4), "Perm stats should be (n_perm, 4)"
        assert np.all(perm_stats[:, 3] == np.arange(10)), "Perm IDs should be correct"
        
        # Check that permuted t-values are different from real
        real_t = real_stats[1]
        perm_t = perm_stats[:, 1]
        
        # At least some permutations should have different t-values
        assert not np.all(perm_t == real_t), "Permutations should produce different t-values"
        
        print(f"✓ Permutations work: real_t={real_t:.4f}, mean_perm_t={np.nanmean(perm_t):.4f}")
    
    def test_format_results_for_clustering(self):
        """Test formatting results for clustering."""
        # Create mock results
        real_results = {
            0: np.array([0.5, 3.2, 0.001]),
            1: np.array([0.3, 2.1, 0.03]),
            2: np.array([-0.4, -2.5, 0.01]),
        }
        
        perm_results = {
            0: np.array([[0.1, 1.2, 0.2, 0], [0.2, 1.5, 0.15, 1]]),
            1: np.array([[0.05, 0.8, 0.4, 0], [0.15, 1.1, 0.25, 1]]),
            2: np.array([[-0.1, -1.0, 0.3, 0], [-0.2, -1.3, 0.2, 1]]),
        }
        
        real_array, perm_array = format_results_for_clustering(real_results, perm_results)
        
        # Check shapes
        assert real_array.shape == (3, 4), "Real array should be (n_electrodes, 4)"
        assert perm_array.shape == (6, 5), "Perm array should be (n_electrodes * n_perm, 5)"
        
        # Check electrode IDs
        assert np.array_equal(real_array[:, 0], [0, 1, 2]), "Electrode IDs should be correct"
        
        print("✓ Result formatting works")


def run_all_tests():
    """Run all permutation tests."""
    print("\n" + "="*60)
    print("TESTING: LMM Permutation Module")
    print("="*60 + "\n")
    
    test = TestPermutation()
    
    try:
        test.test_permute_predictor_preserves_groups()
        test.test_permute_predictor_changes_order()
        test.test_fit_single_lmm()
        test.test_fit_lmm_with_permutations()
        test.test_format_results_for_clustering()
        
        print("\n" + "="*60)
        print("✅ ALL PERMUTATION TESTS PASSED")
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
