#!/usr/bin/env python3
"""
Quick verification script to check if the bug fix resolved the issues.

Run this after re-running the pipeline to verify:
1. Permutations completed successfully
2. Null distribution is valid (non-zero)
3. Results are spatially specific (not whole-scalp)

Usage:
    python Statistics/verify_fix.py
"""

import pickle
import numpy as np
from pathlib import Path
import yaml

def verify_results(results_dir: str):
    """
    Verify that results are valid after bug fix.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory (e.g., results/lmm_cluster/onoff/evoked_P3a)
    """
    results_path = Path(results_dir)
    
    print("="*70)
    print("VERIFICATION REPORT")
    print("="*70)
    print(f"Results directory: {results_path}")
    print()
    
    # Load results
    results_file = results_path / "results.pkl"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return False
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # Check 1: Verify permutations completed
    print("CHECK 1: Permutation Completion")
    print("-" * 70)
    
    if 'cluster_diagnostics' in results:
        diag = results['cluster_diagnostics']
        
        if 'null_distribution' in diag:
            null_dist = diag['null_distribution']
            
            # Check if null distribution has valid values
            if isinstance(null_dist, dict):
                mean_val = null_dist.get('mean', 0)
                std_val = null_dist.get('std', 0)
            else:
                mean_val = np.mean(null_dist)
                std_val = np.std(null_dist)
            
            if mean_val == 0 and std_val == 0:
                print("❌ FAILED: Null distribution is all zeros")
                print("   This indicates permutations failed to run properly")
                return False
            else:
                print(f"✓ PASSED: Null distribution has valid values")
                print(f"   Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        else:
            print("⚠ WARNING: No null_distribution found in diagnostics")
    else:
        print("⚠ WARNING: No cluster_diagnostics found in results")
    
    print()
    
    # Check 2: Verify spatial specificity
    print("CHECK 2: Spatial Specificity")
    print("-" * 70)
    
    n_channels = len(results.get('t_stats', []))
    
    if 'tfce_p_values' in results:
        # TFCE results
        p_values = results['tfce_p_values']
        sig_channels = np.sum(p_values < 0.05)
        pct_sig = 100 * sig_channels / n_channels if n_channels > 0 else 0
        
        print(f"Significant channels: {sig_channels}/{n_channels} ({pct_sig:.1f}%)")
        
        if pct_sig > 90:
            print("❌ FAILED: >90% of channels are significant (likely whole-scalp artifact)")
            return False
        elif pct_sig > 0:
            print(f"✓ PASSED: Results show spatial specificity")
        else:
            print("⚠ INFO: No significant channels found")
    
    elif 'clusters' in results:
        # Threshold-based results
        clusters = results['clusters']
        cluster_p_values = results.get('cluster_p_values', [])
        
        if len(clusters) > 0:
            sig_clusters = np.sum(np.array(cluster_p_values) < 0.05)
            
            # Check largest cluster size
            cluster_sizes = [len(c) for c in clusters]
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            pct_max = 100 * max_cluster_size / n_channels if n_channels > 0 else 0
            
            print(f"Total clusters: {len(clusters)}")
            print(f"Significant clusters: {sig_clusters}")
            print(f"Largest cluster: {max_cluster_size}/{n_channels} channels ({pct_max:.1f}%)")
            
            if pct_max > 90:
                print("❌ FAILED: Largest cluster covers >90% of scalp (likely artifact)")
                return False
            else:
                print(f"✓ PASSED: Clusters show spatial specificity")
        else:
            print("⚠ INFO: No clusters found")
    
    print()
    
    # Check 3: Model quality
    print("CHECK 3: Model Quality")
    print("-" * 70)
    
    if 'lmm_diagnostics' in results:
        lmm_diag = results['lmm_diagnostics']
        conv_rate = lmm_diag.get('convergence_rate', 0)
        n_converged = lmm_diag.get('n_converged', 0)
        n_failed = lmm_diag.get('n_failed', 0)
        
        print(f"Convergence rate: {100*conv_rate:.1f}%")
        print(f"Converged: {n_converged}, Failed: {n_failed}")
        
        if conv_rate < 0.8:
            print("⚠ WARNING: Low convergence rate (<80%)")
        else:
            print("✓ PASSED: Good convergence rate")
    
    print()
    print("="*70)
    print("OVERALL: Verification completed")
    print("="*70)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Default to most recent P3a results
    default_dir = "results/lmm_cluster/onoff/evoked_P3a"
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = default_dir
        print(f"Using default results directory: {results_dir}")
        print(f"To specify different directory: python verify_fix.py <path/to/results>")
        print()
    
    verify_results(results_dir)
