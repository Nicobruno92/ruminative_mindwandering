"""
Main pipeline executor for LMM-based spatial cluster permutation testing.

This script orchestrates the complete analysis pipeline:
1. Load data and configuration
2. Run LMM for each channel
3. Perform spatial cluster permutation test
4. Generate visualizations and save results
"""

import pickle
import argparse
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from datetime import datetime

# Import pipeline modules
from reader import load_data, validate_formula_variables
from lmm_model import run_lmm_per_channel
from cluster_test import (
    get_channel_adjacency,
    spatial_cluster_permutation_test,
    summarize_clusters
)
from plot_results import create_results_report


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path: str = "Statistics/config.yaml") -> None:
    """
    Execute complete LMM-based spatial cluster permutation pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    print("="*80)
    print("LMM-BASED SPATIAL CLUSTER PERMUTATION TESTING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Extract parameters
    data_path = config['project']['data_path']
    output_path = config['project']['output_path']
    montage_path = config['project']['montage_path']
    
    formula = config['lmm']['formula']
    predictor_of_interest = config['lmm']['predictor_of_interest']
    method = config['lmm']['method']
    maxiter = config['lmm']['maxiter']
    random_state = config['lmm']['random_state']
    
    threshold = config['clustering']['threshold']
    n_permutations = config['clustering']['n_permutations']
    alpha = config['clustering']['alpha']
    tail = config['clustering']['tail']
    seed = config['clustering']['seed']
    n_jobs = config['clustering']['n_jobs']
    
    save_pickle = config['output']['save_pickle']
    save_csv = config['output']['save_csv']
    save_figures = config['output']['save_figures']
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n" + "-"*80)
    print("STEP 1: Loading data")
    print("-"*80)
    print(f"Data path: {data_path}")
    
    power_data, df_behavioral = load_data(data_path)
    n_observations, n_channels = power_data.shape
    
    print(f"✓ Loaded power data: {n_observations} observations × {n_channels} channels")
    print(f"✓ Loaded behavioral data: {len(df_behavioral)} rows × {len(df_behavioral.columns)} columns")
    print(f"  Behavioral columns: {', '.join(df_behavioral.columns)}")
    print(f"  Number of subjects: {df_behavioral['subject'].nunique()}")
    
    # Validate formula variables
    validate_formula_variables(df_behavioral, formula)
    print(f"✓ Formula validated: {formula}")
    
    # Step 2: Get channel information and adjacency
    print("\n" + "-"*80)
    print("STEP 2: Setting up channel information")
    print("-"*80)
    
    # Create dummy channel names if not available
    # In practice, these should come from the data
    ch_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    print(f"Loading montage from: {montage_path}")
    adjacency, ch_names_ordered = get_channel_adjacency(montage_path, ch_names)
    print(f"✓ Adjacency matrix computed: {adjacency.shape}")
    
    # Reorder power_data to match channel order
    ch_order_idx = [ch_names.index(ch) for ch in ch_names_ordered]
    power_data = power_data[:, ch_order_idx]
    ch_names = ch_names_ordered
    
    # Create MNE Info object for visualization
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types='eeg')
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)
    info.set_montage(montage)
    
    # Step 3: Run LMM for each channel
    print("\n" + "-"*80)
    print("STEP 3: Running Linear Mixed Models")
    print("-"*80)
    print(f"Formula: {formula}")
    print(f"Predictor of interest: {predictor_of_interest}")
    print(f"Method: {method}, Max iterations: {maxiter}")
    
    t_stats, p_values = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor_of_interest,
        method=method,
        maxiter=maxiter,
        random_state=random_state
    )
    
    print(f"✓ LMM completed for {n_channels} channels")
    print(f"  T-statistics range: [{np.min(t_stats):.3f}, {np.max(t_stats):.3f}]")
    print(f"  Channels with |t| > {threshold}: {np.sum(np.abs(t_stats) > threshold)}")
    
    # Step 4: Spatial cluster permutation test
    print("\n" + "-"*80)
    print("STEP 4: Spatial Cluster Permutation Test")
    print("-"*80)
    print(f"Threshold: {threshold}")
    print(f"Permutations: {n_permutations}")
    print(f"Alpha: {alpha}")
    print("Running permutation test (this may take several minutes)...")
    
    clusters, cluster_stats, cluster_p_values = spatial_cluster_permutation_test(
        observed_t_stats=t_stats,
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor_of_interest,
        adjacency=adjacency,
        threshold=threshold,
        n_permutations=n_permutations,
        tail=tail,
        seed=seed,
        n_jobs=n_jobs,
        method=method,
        maxiter=maxiter
    )
    
    n_clusters = len(clusters)
    n_sig_clusters = np.sum(cluster_p_values < alpha)
    
    print("✓ Cluster analysis completed")
    print(f"  Total clusters found: {n_clusters}")
    print(f"  Significant clusters (α={alpha}): {n_sig_clusters}")
    
    if n_clusters > 0:
        print("\nCluster summary:")
        cluster_summary = summarize_clusters(
            clusters, cluster_stats, cluster_p_values, ch_names, alpha
        )
        print(cluster_summary.to_string(index=False))
    
    # Step 5: Save results
    print("\n" + "-"*80)
    print("STEP 5: Saving results")
    print("-"*80)
    
    # Save pickle
    if save_pickle:
        results_dict = {
            'config': config,
            't_stats': t_stats,
            'p_values': p_values,
            'clusters': clusters,
            'cluster_stats': cluster_stats,
            'cluster_p_values': cluster_p_values,
            'ch_names': ch_names,
            'info': info,
            'power_data_shape': power_data.shape,
            'n_subjects': df_behavioral['subject'].nunique()
        }
        
        pickle_path = output_dir / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"✓ Results saved to {pickle_path}")
    
    # Save CSV summary
    if save_csv and n_clusters > 0:
        csv_path = output_dir / "cluster_summary.csv"
        cluster_summary.to_csv(csv_path, index=False)
        print(f"✓ Cluster summary saved to {csv_path}")
    
    # Save t-statistics per channel
    if save_csv:
        t_stats_df = pd.DataFrame({
            'channel': ch_names,
            't_statistic': t_stats,
            'p_value': p_values
        })
        t_stats_path = output_dir / "t_statistics.csv"
        t_stats_df.to_csv(t_stats_path, index=False)
        print(f"✓ T-statistics saved to {t_stats_path}")
    
    # Generate visualizations
    if save_figures:
        print("\nGenerating visualizations...")
        create_results_report(
            t_stats=t_stats,
            clusters=clusters,
            cluster_stats=cluster_stats,
            cluster_p_values=cluster_p_values,
            info=info,
            threshold=threshold,
            alpha=alpha,
            output_dir=str(output_dir)
        )
        print(f"✓ Figures saved to {output_dir}")
    
    # Print completion message
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {output_dir.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LMM-based spatial cluster permutation testing pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics/config.yaml",
        help="Path to configuration YAML file"
    )
    
    args = parser.parse_args()
    
    main(config_path=args.config)
