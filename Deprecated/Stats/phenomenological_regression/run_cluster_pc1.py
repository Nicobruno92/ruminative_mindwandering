#!/usr/bin/env python3
"""
Wrapper script for running PC1 cluster permutation tests on individual markers.
Called by the SLURM array job script.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add the script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the main cluster testing function
from cluster_pc1_permutation import run_pc1_cluster_test, save_pc1_cluster_results, plot_pc1_cluster_results
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Run PC1 cluster permutation test for a single marker')
    parser.add_argument('--marker', type=str, required=True,
                       help='EEG marker to analyze')
    parser.add_argument('--csv-file', type=str,
                       default='results/aggregated_mne_markers/merged_pca_eeg_markers.csv',
                       help='Path to merged CSV file')
    parser.add_argument('--out-dir', type=str,
                       default='results/cluster_permutation_tests_pc1',
                       help='Output directory')
    parser.add_argument('--n-perm', type=int, default=500,
                       help='Number of permutations')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--z-score-pc1', action='store_true', default=True,
                       help='Z-score PC1 within participants')
    parser.add_argument('--min-cluster-size', type=int, default=4,
                       help='Minimum cluster size')
    
    args = parser.parse_args()
    
    print(f"🧠 PC1 CLUSTER TEST: {args.marker} 🧠")
    print(f"CSV file: {args.csv_file}")
    print(f"Output directory: {args.out_dir}")
    print(f"Permutations: {args.n_perm}")
    print(f"Alpha: {args.alpha}")
    print(f"Random seed: {args.random_seed}")
    print(f"Z-score PC1: {args.z_score_pc1}")
    print(f"Min cluster size: {args.min_cluster_size}")
    
    # Check if input file exists
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        print("Please run the merge script first:")
        print("python Stats/phenomenological_regression/merge_pca_eeg_data.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)
    
    # Load data
    print(f"Loading data for marker {args.marker}...")
    df = pd.read_csv(args.csv_file)
    
    # Filter for the specific marker
    df_marker = df[df['marker'] == args.marker]
    if df_marker.empty:
        print(f"ERROR: No data found for marker {args.marker}")
        sys.exit(1)
    
    print(f"Loaded {len(df_marker)} rows for marker {args.marker}")
    
    # Check PC1 availability
    pc1_available = df_marker['PC1'].notna().sum()
    print(f"Rows with PC1 data: {pc1_available}/{len(df_marker)}")
    
    if pc1_available == 0:
        print(f"ERROR: No PC1 data for marker {args.marker}")
        sys.exit(1)
    
    # Run cluster test
    try:
        results = run_pc1_cluster_test(
            df, args.marker, 
            n_perm=args.n_perm, 
            alpha=args.alpha, 
            rng=rng
        )
        
        if results is None:
            print(f"No results generated for marker {args.marker}")
            sys.exit(1)
        
        # Save results
        results_file = os.path.join(args.out_dir, f"{args.marker}_pc1_cluster_results.csv")
        n_sig = save_pc1_cluster_results(results, results_file, args.alpha)
        
        print(f"Found {n_sig} significant clusters for marker {args.marker}")
        
        # Create plot
        plot_path = os.path.join(args.out_dir, f"{args.marker}_pc1_cluster_plot.png")
        try:
            fig = plot_pc1_cluster_results(results, plot_path, args.alpha)
            plt.close(fig)
            print(f"Plot saved: {plot_path}")
        except Exception as e:
            print(f"Warning: Could not create plot for marker {args.marker}: {e}")
        
        print(f"✓ Successfully completed PC1 cluster test for marker {args.marker}")
        
    except Exception as e:
        print(f"ERROR: Failed to process marker {args.marker}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 