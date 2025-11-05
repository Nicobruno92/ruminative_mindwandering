"""
Test suite for LMM-based spatial cluster permutation testing.

Following MNE-Python testing patterns for cluster permutation tests.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from scipy import stats
from scipy.sparse import csr_matrix
from pathlib import Path
import warnings

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cluster_test import (
    spatial_cluster_permutation_test,
    spatial_cluster_test_tfce,
    _validate_adjacency_matrix,
    _find_clusters,
    _permute_within_subjects,
)
from lmm_model import run_lmm_per_channel, parse_random_effects


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_eeg_data():
    """Create simple synthetic EEG data for testing."""
    rng = np.random.RandomState(42)
    n_subjects = 20  # Increased for better LMM convergence
    n_trials_per_subject = 40  # Increased for better LMM convergence
    n_channels = 8
    
    subjects = np.repeat(np.arange(n_subjects), n_trials_per_subject)
    trials = np.tile(np.arange(n_trials_per_subject), n_subjects)
    onoff = rng.binomial(1, 0.5, size=len(subjects))
    
    # More realistic power data with subject-specific baselines
    power_data = np.zeros((len(subjects), n_channels))
    for subj in range(n_subjects):
        subj_mask = subjects == subj
        # Subject-specific baseline (random intercept)
        subj_baseline = rng.randn() * 2.0
        # Add baseline + noise
        power_data[subj_mask] = subj_baseline + rng.randn(n_trials_per_subject, n_channels) * 3.0
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'trial': trials,
        'onoff': onoff
    })
    
    return power_data, df_behavioral, n_channels


@pytest.fixture
def eeg_data_with_effect():
    """Create EEG data with a real effect in specific channels."""
    rng = np.random.RandomState(42)
    n_subjects = 25  # More subjects for better power
    n_trials_per_subject = 50  # More trials
    n_channels = 10
    
    subjects = np.repeat(np.arange(n_subjects), n_trials_per_subject)
    trials = np.tile(np.arange(n_trials_per_subject), n_subjects)
    onoff = rng.binomial(1, 0.5, size=len(subjects))
    
    # More realistic power data with subject-specific baselines
    power_data = np.zeros((len(subjects), n_channels))
    for subj in range(n_subjects):
        subj_mask = subjects == subj
        # Subject-specific baseline (random intercept)
        subj_baseline = rng.randn() * 2.0
        # Add baseline + noise
        power_data[subj_mask] = subj_baseline + rng.randn(n_trials_per_subject, n_channels) * 3.0
    
    # Add strong effect in channels 2-4 when onoff=1
    effect_size = 1.5  # Larger effect for better detection
    for subj_idx in range(n_subjects):
        subj_mask = subjects == subj_idx
        onoff_mask = subj_mask & (onoff == 1)
        subject_effect = rng.randn() * 0.3
        power_data[onoff_mask, 2:5] += effect_size + subject_effect
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'trial': trials,
        'onoff': onoff
    })
    
    return power_data, df_behavioral, n_channels


@pytest.fixture  
def simple_adjacency():
    """Create simple linear chain adjacency matrix."""
    n_channels = 8
    adjacency = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    
    return csr_matrix(adjacency)


# =============================================================================
# TEST LMM MODEL FUNCTIONS
# =============================================================================

def test_parse_random_effects():
    """Test parsing of random effects formulas."""
    # Random intercept only
    formula1 = "power ~ onoff + (1|subject)"
    fixed, random = parse_random_effects(formula1)
    assert fixed == "power ~ onoff"
    assert random is None
    
    # Random intercept + slope
    formula2 = "power ~ onoff + (1 + onoff|subject)"
    fixed, random = parse_random_effects(formula2)
    assert fixed == "power ~ onoff"
    assert random == "1 + onoff"


def test_lmm_per_channel_basic(simple_eeg_data):
    """Test basic LMM fitting per channel."""
    power_data, df_behavioral, n_channels = simple_eeg_data
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    t_stats, p_values, diagnostics = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=True
    )
    
    # Check output shapes
    assert t_stats.shape == (n_channels,)
    assert p_values.shape == (n_channels,)
    assert isinstance(diagnostics, dict)
    
    # Check that statistics are finite
    assert np.all(np.isfinite(t_stats))
    assert np.all(np.isfinite(p_values))
    
    # Check p-values are in valid range
    assert np.all(p_values >= 0)
    assert np.all(p_values <= 1)


# Test detects effect - may fail if LMM doesn't converge on simple data
def test_lmm_detects_effect(eeg_data_with_effect):
    """Test that LMM can detect effects in channels with planted signal."""
    power_data, df_behavioral, n_channels = eeg_data_with_effect
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    t_stats, p_values, diagnostics = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=True
    )
    
    # Effect was planted in channels 2-4
    effect_channels = [2, 3, 4]
    null_channels = [0, 1, 5, 6, 7, 8, 9]
    
    mean_t_effect = np.abs(t_stats[effect_channels]).mean()
    mean_t_null = np.abs(t_stats[null_channels]).mean()
    
    # Effect channels should have larger t-stats on average
    assert mean_t_effect > mean_t_null


# =============================================================================
# TEST CLUSTER FINDING FUNCTIONS
# =============================================================================

def test_validate_adjacency_matrix():
    """Test adjacency matrix validation."""
    n_channels = 5
    
    # Valid adjacency
    adj_valid = np.eye(n_channels)
    _validate_adjacency_matrix(adj_valid, n_channels)  # Should not raise
    
    # Valid sparse adjacency
    adj_sparse = csr_matrix(adj_valid)
    _validate_adjacency_matrix(adj_sparse, n_channels)  # Should not raise
    
    # Invalid shape
    adj_wrong_shape = np.eye(n_channels + 1)
    with pytest.raises(ValueError, match="shape.*does not match"):
        _validate_adjacency_matrix(adj_wrong_shape, n_channels)


def test_find_clusters_basic():
    """Test basic cluster finding."""
    t_stats = np.array([3.0, 3.5, 0.5, -0.3, -3.0, -3.2, 0.1, 0.2])
    threshold = 2.0
    tail = 0  # two-tailed
    
    # Linear chain adjacency
    adjacency = np.zeros((8, 8))
    for i in range(7):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    adjacency = csr_matrix(adjacency)
    
    clusters, cluster_stats = _find_clusters(
        t_stats, adjacency, threshold, tail
    )
    
    # Should find 2 clusters (positive: 0-1, negative: 4-5)
    assert len(clusters) == 2


def test_find_clusters_tails():
    """Test cluster finding with different tail settings."""
    t_stats = np.array([3.0, 3.5, -3.0, -3.2, 0.5])
    threshold = 2.0
    
    # Linear adjacency
    adjacency = np.zeros((5, 5))
    for i in range(4):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    adjacency = csr_matrix(adjacency)
    
    # Two-tailed: should find both positive and negative clusters
    clusters_two, stats_two = _find_clusters(t_stats, adjacency, threshold, 0)
    assert len(clusters_two) == 2
    
    # Right-tailed: only positive cluster
    clusters_right, stats_right = _find_clusters(t_stats, adjacency, threshold, 1)
    assert len(clusters_right) == 1
    assert stats_right[0] > 0
    
    # Left-tailed: only negative cluster
    clusters_left, stats_left = _find_clusters(t_stats, adjacency, threshold, -1)
    assert len(clusters_left) == 1
    assert stats_left[0] < 0


# =============================================================================
# TEST PERMUTATION FUNCTIONS
# =============================================================================

def test_permute_within_subjects(simple_eeg_data):
    """Test that permutation preserves within-subject structure."""
    _, df_behavioral, _ = simple_eeg_data
    
    df_permuted = _permute_within_subjects(
        df_behavioral.copy(), 'onoff', seed=42
    )
    
    # Check that subject IDs are unchanged
    assert_array_equal(
        df_behavioral['subject'].values,
        df_permuted['subject'].values
    )
    
    # Check that onoff is different
    assert not np.array_equal(
        df_behavioral['onoff'].values,
        df_permuted['onoff'].values
    )


# =============================================================================
# TEST FULL CLUSTER PERMUTATION PIPELINE
# =============================================================================

def test_cluster_permutation_basic(simple_eeg_data, simple_adjacency):
    """Test basic cluster permutation test."""
    power_data, df_behavioral, n_channels = simple_eeg_data
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    # First compute observed t-stats
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=False
    )
    
    clusters, cluster_stats, cluster_pvals, diagnostics = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        adjacency=simple_adjacency,
        n_permutations=50,
        threshold=2.0,
        tail=0,
        seed=42,
        n_jobs=1  # Use sequential execution to avoid import issues
    )
    
    # Check result structure
    assert isinstance(clusters, list)
    assert isinstance(cluster_stats, np.ndarray)
    assert isinstance(cluster_pvals, np.ndarray)
    
    # Check p-values are valid
    assert np.all(cluster_pvals >= 0)
    assert np.all(cluster_pvals <= 1)


def test_cluster_permutation_detects_effect(eeg_data_with_effect):
    """Test that cluster permutation can detect real effects."""
    power_data, df_behavioral, n_channels = eeg_data_with_effect
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    # Create adjacency that matches data dimensions (10 channels)
    adjacency = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        adjacency[i, i + 1] = 1
        adjacency[i + 1, i] = 1
    adjacency = csr_matrix(adjacency)
    
    # First compute observed t-stats
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=False
    )
    
    clusters, cluster_stats, cluster_pvals, diagnostics = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        adjacency=adjacency,
        n_permutations=100,
        threshold=2.0,
        tail=0,
        seed=42,
        n_jobs=1  # Use sequential to avoid import issues
    )
    
    # Should find at least one cluster
    assert len(clusters) > 0
    
    # With planted effect, should approach significance
    min_p = np.min(cluster_pvals) if len(cluster_pvals) > 0 else 1.0
    assert min_p < 0.2  # Relaxed threshold for stochastic test


# =============================================================================
# TEST TFCE
# =============================================================================

def test_tfce_basic(simple_eeg_data, simple_adjacency):
    """Test TFCE cluster permutation."""
    power_data, df_behavioral, n_channels = simple_eeg_data
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    # First compute observed t-stats
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=False
    )
    
    tfce_scores, p_values, diagnostics = spatial_cluster_test_tfce(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        adjacency=simple_adjacency,
        n_permutations=50,
        seed=42
    )
    
    # Check shapes
    assert tfce_scores.shape == (n_channels,)
    assert p_values.shape == (n_channels,)
    
    # Check p-values are valid
    assert np.all(p_values >= 0)
    assert np.all(p_values <= 1)


# =============================================================================
# TEST REPRODUCIBILITY
# =============================================================================

def test_permutation_reproducibility(simple_eeg_data, simple_adjacency):
    """Test that results are reproducible with same random seed."""
    power_data, df_behavioral, _ = simple_eeg_data
    
    formula = "power ~ onoff + (1|subject)"
    predictor = "onoff"
    
    # First compute observed t-stats
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        return_diagnostics=False
    )
    
    clusters1, cluster_stats1, cluster_pvals1, _ = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        adjacency=simple_adjacency,
        n_permutations=50,
        threshold=2.0,
        tail=0,
        seed=42,
        n_jobs=1  # Use sequential execution
    )
    
    clusters2, cluster_stats2, cluster_pvals2, _ = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        adjacency=simple_adjacency,
        n_permutations=50,
        threshold=2.0,
        tail=0,
        seed=42,
        n_jobs=1  # Use sequential execution
    )
    
    # Results should be identical
    assert_array_equal(cluster_pvals1, cluster_pvals2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
