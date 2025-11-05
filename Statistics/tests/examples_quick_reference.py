#!/usr/bin/env python
"""
Quick test examples demonstrating the LMM cluster permutation test API.

Run this file to see how to use the testing functions.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cluster_test import (
    spatial_cluster_permutation_test,
    spatial_cluster_test_tfce,
    _find_clusters,
    _validate_adjacency_matrix
)
from lmm_model import run_lmm_per_channel, parse_random_effects


def example_1_basic_lmm():
    """Example 1: Basic LMM per channel."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic LMM per Channel")
    print("="*60)
    
    # Create simple data
    rng = np.random.RandomState(42)
    n_subjects = 10
    n_trials = 20
    n_channels = 5
    
    subjects = np.repeat(np.arange(n_subjects), n_trials)
    onoff = rng.binomial(1, 0.5, size=len(subjects))
    power_data = rng.randn(len(subjects), n_channels)
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'onoff': onoff
    })
    
    # Run LMM
    formula = "power ~ onoff + (1|subject)"
    t_stats, p_values, diagnostics = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest="onoff",
        return_diagnostics=True
    )
    
    print(f"T-statistics shape: {t_stats.shape}")
    print(f"T-statistics: {t_stats}")
    print(f"P-values: {p_values}")
    print(f"Converged channels: {diagnostics['n_converged']}/{n_channels}")


def example_2_cluster_test():
    """Example 2: Full cluster permutation test."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Cluster Permutation Test")
    print("="*60)
    
    # Create data
    rng = np.random.RandomState(42)
    n_subjects = 12
    n_trials = 25
    n_channels = 8
    
    subjects = np.repeat(np.arange(n_subjects), n_trials)
    onoff = rng.binomial(1, 0.5, size=len(subjects))
    power_data = rng.randn(len(subjects), n_channels)
    
    # Add small effect in channels 2-3
    for subj in range(n_subjects):
        mask = (subjects == subj) & (onoff == 1)
        power_data[mask, 2:4] += 0.3
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'onoff': onoff
    })
    
    # Create linear adjacency
    adjacency = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1
    adjacency = csr_matrix(adjacency)
    
    # Compute observed t-statistics
    formula = "power ~ onoff + (1|subject)"
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest="onoff",
        return_diagnostics=False
    )
    
    print(f"Observed t-statistics: {t_stats}")
    
    # Run cluster test
    clusters, cluster_pvals, null_dist, diagnostics = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest="onoff",
        adjacency=adjacency,
        threshold=2.0,
        n_permutations=100,
        tail=0,
        seed=42,
        verbose=False
    )
    
    print(f"\nNumber of clusters found: {len(clusters)}")
    for i, (cluster, pval) in enumerate(zip(clusters, cluster_pvals)):
        print(f"  Cluster {i+1}: channels {cluster}, p={pval:.4f}")


def example_3_tfce():
    """Example 3: TFCE (Threshold-Free Cluster Enhancement)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: TFCE Test")
    print("="*60)
    
    # Create simple data
    rng = np.random.RandomState(42)
    n_subjects = 10
    n_trials = 20
    n_channels = 6
    
    subjects = np.repeat(np.arange(n_subjects), n_trials)
    onoff = rng.binomial(1, 0.5, size=len(subjects))
    power_data = rng.randn(len(subjects), n_channels)
    
    df_behavioral = pd.DataFrame({
        'subject': subjects,
        'onoff': onoff
    })
    
    # Create adjacency
    adjacency = np.zeros((n_channels, n_channels))
    for i in range(n_channels - 1):
        adjacency[i, i+1] = 1
        adjacency[i+1, i] = 1
    adjacency = csr_matrix(adjacency)
    
    # Compute observed t-statistics
    formula = "power ~ onoff + (1|subject)"
    t_stats, _, _ = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest="onoff",
        return_diagnostics=False
    )
    
    # Run TFCE
    tfce_scores, p_values, diagnostics = spatial_cluster_test_tfce(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest="onoff",
        adjacency=adjacency,
        n_permutations=50,
        seed=42,
        verbose=False
    )
    
    print(f"TFCE scores: {tfce_scores}")
    print(f"P-values: {p_values}")
    print(f"Significant channels (p<0.05): {np.sum(p_values < 0.05)}")


def example_4_formula_parsing():
    """Example 4: Formula parsing."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Formula Parsing")
    print("="*60)
    
    formulas = [
        "power ~ onoff + (1|subject)",
        "power ~ onoff + (1 + onoff|subject)",
        "power ~ onoff + age + (1|subject)",
        "power ~ onoff * condition + (1 + onoff|subject)",
    ]
    
    for formula in formulas:
        fixed, random = parse_random_effects(formula)
        print(f"\nOriginal:     {formula}")
        print(f"Fixed:        {fixed}")
        print(f"Random:       {random}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LMM CLUSTER PERMUTATION TEST - QUICK EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate the basic API of the LMM cluster")
    print("permutation testing pipeline, following MNE-Python patterns.")
    
    try:
        example_1_basic_lmm()
        example_2_cluster_test()
        example_3_tfce()
        example_4_formula_parsing()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nFor full test suite, run:")
        print("  pytest test_lmm_cluster_permutation.py -v")
        print("\nFor documentation, see:")
        print("  TEST_SUITE_README.md")
        print("  MNE_STYLE_TESTS_SUMMARY.md")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
