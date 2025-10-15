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
from reader import (
    load_all_probe_data, 
    prepare_data_for_lmm, 
    validate_formula_variables,
    get_channel_names,
    get_available_markers
)
from lmm_model import run_lmm_per_channel
from cluster_test import (
    get_channel_adjacency,
    spatial_cluster_permutation_test,
    summarize_clusters
)
from plot_results import create_results_report


def create_marker_type_summaries(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create organized summaries by marker type for better analysis.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe with all results
    output_dir : Path
        Output directory for saving summaries
    """
    # Group by marker type
    for marker_type in summary_df['marker_type'].unique():
        type_df = summary_df[summary_df['marker_type'] == marker_type].copy()
        
        # Save type-specific summary
        type_summary_path = output_dir / f"summary_{marker_type}_markers.csv"
        type_df.to_csv(type_summary_path, index=False)
        
        # Create analysis summary for this type
        analysis_summary = {
            'marker_type': marker_type,
            'n_markers': len(type_df),
            'n_significant_markers': len(type_df[type_df['n_sig_clusters'] > 0]),
            'total_clusters': type_df['n_clusters'].sum(),
            'total_sig_clusters': type_df['n_sig_clusters'].sum(),
            'avg_t_stat': type_df['t_stat_mean'].mean(),
            'max_t_stat': type_df['t_stat_max'].max(),
            'min_t_stat': type_df['t_stat_min'].min(),
            'n_subjects_avg': type_df['n_subjects'].mean(),
            'analysis_date': datetime.now().isoformat()
        }
        
        # Save analysis summary
        analysis_df = pd.DataFrame([analysis_summary])
        analysis_path = output_dir / f"analysis_summary_{marker_type}.csv"
        analysis_df.to_csv(analysis_path, index=False)
        
        print(f"✓ {marker_type} marker summary saved to {type_summary_path}")
        print(f"✓ {marker_type} analysis summary saved to {analysis_path}")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_single_marker(marker_name: str, df_all: pd.DataFrame, config: dict, 
                         ch_names: list, adjacency: np.ndarray, info: mne.Info) -> dict:
    """
    Process a single marker through the complete LMM cluster pipeline.
    
    Parameters
    ----------
    marker_name : str
        Name of the marker to process
    df_all : pd.DataFrame
        Combined dataframe with all probe data
    config : dict
        Configuration dictionary
    ch_names : list
        Channel names
    adjacency : np.ndarray
        Channel adjacency matrix
    info : mne.Info
        MNE Info object for visualization
        
    Returns
    -------
    dict
        Results dictionary for this marker
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING MARKER: {marker_name}")
    print(f"{'='*60}")
    
    try:
        # Extract parameters
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
        output_path = config['project']['output_path']
        
        # Prepare data for this marker
        print(f"Preparing data for marker: {marker_name}")
        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=df_all,
            marker_name=marker_name,
            formula=formula,
            include_channels=None,
            exclude_channels=None
        )
        
        # Ensure deterministic channel ordering - data channels must match montage channels
        # Find intersection of data channels and montage channels
        data_channels_set = set(channels)
        montage_channels_set = set(ch_names)
        
        # Get channels that exist in both data and montage
        common_channels = sorted(list(data_channels_set & montage_channels_set))
        
        if len(common_channels) == 0:
            raise ValueError(f"No common channels between data ({len(channels)} channels) and montage ({len(ch_names)} channels)")
        
        if len(common_channels) < len(channels) * 0.8:  # Less than 80% overlap
            missing_data = sorted(list(data_channels_set - montage_channels_set))
            missing_montage = sorted(list(montage_channels_set - data_channels_set))
            raise ValueError(f"Insufficient channel overlap. Missing from montage: {missing_data[:5]}... Missing from data: {missing_montage[:5]}...")
        
        # Create reordering indices
        data_to_common = [channels.index(ch) for ch in common_channels]
        common_to_montage = [ch_names.index(ch) for ch in common_channels]
        
        # Reorder power data to match common channels
        power_data_filtered = power_data[:, data_to_common]
        
        # Create final channel names in montage order
        ch_names_final = [ch_names[i] for i in common_to_montage]
        
        # Reorder power data to match montage order
        power_data = power_data_filtered[:, common_to_montage]
        n_observations, n_channels = power_data.shape
        
        print(f"✓ Data prepared: {n_observations} observations × {n_channels} channels")
        
        # Validate formula variables
        validate_formula_variables(df_behavioral, formula)
        
        # Run LMM for each channel
        print(f"Running LMM for {n_channels} channels...")
        t_stats, p_values = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=formula,
            predictor_of_interest=predictor_of_interest,
            method=method,
            maxiter=maxiter,
            random_state=random_state
        )
        
        print(f"✓ LMM completed")
        print(f"  T-statistics range: [{np.min(t_stats):.3f}, {np.max(t_stats):.3f}]")
        print(f"  Channels with |t| > {threshold}: {np.sum(np.abs(t_stats) > threshold)}")
        
        # Spatial cluster permutation test
        print("Running spatial cluster permutation test...")
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
        
        print(f"✓ Cluster analysis completed")
        print(f"  Total clusters: {n_clusters}")
        print(f"  Significant clusters (α={alpha}): {n_sig_clusters}")
        
        # Create output directory for this marker with type-specific naming
        output_dir = Path(output_path)
        
        # Get marker type for organized saving
        marker_type = df_behavioral['marker_type'].iloc[0] if 'marker_type' in df_behavioral.columns else 'unknown'
        
        # Create safe marker name with type information
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        safe_marker_name = f"{marker_type}_{safe_marker_name}"
        
        # Save results with comprehensive metadata
        results_dict = {
            'config': config,
            'marker_name': marker_name,
            'marker_type': marker_type,
            't_stats': t_stats,
            'p_values': p_values,
            'clusters': clusters,
            'cluster_stats': cluster_stats,
            'cluster_p_values': cluster_p_values,
            'ch_names': ch_names_final,
            'info': info,
            'power_data_shape': power_data.shape,
            'n_subjects': df_behavioral['subject'].nunique(),
            'n_observations': n_observations,
            'formula': formula,
            'predictor_of_interest': predictor_of_interest,
            'n_clusters': n_clusters,
            'n_sig_clusters': n_sig_clusters,
            'analysis_timestamp': datetime.now().isoformat(),
            'threshold': threshold,
            'alpha': alpha,
            'n_permutations': n_permutations
        }
        
        # Save pickle
        if save_pickle:
            pickle_path = output_dir / f"results_{safe_marker_name}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(results_dict, f)
            print(f"✓ Results saved to {pickle_path}")
        
        # Save CSV summary
        if save_csv and n_clusters > 0:
            cluster_summary = summarize_clusters(
                clusters, cluster_stats, cluster_p_values, ch_names_final, alpha
            )
            # Add metadata to cluster summary
            cluster_summary['marker_name'] = marker_name
            cluster_summary['marker_type'] = marker_type
            cluster_summary['analysis_timestamp'] = datetime.now().isoformat()
            
            csv_path = output_dir / f"cluster_summary_{safe_marker_name}.csv"
            cluster_summary.to_csv(csv_path, index=False)
            print(f"✓ Cluster summary saved to {csv_path}")
        
        # Save t-statistics per channel
        if save_csv:
            t_stats_df = pd.DataFrame({
                'channel': ch_names_final,
                't_statistic': t_stats,
                'p_value': p_values,
                'marker_name': marker_name,
                'marker_type': marker_type,
                'predictor_of_interest': predictor_of_interest,
                'formula': formula,
                'analysis_timestamp': datetime.now().isoformat()
            })
            t_stats_path = output_dir / f"t_statistics_{safe_marker_name}.csv"
            t_stats_df.to_csv(t_stats_path, index=False)
            print(f"✓ T-statistics saved to {t_stats_path}")
        
        # Generate visualizations
        if save_figures:
            print("Generating visualizations...")
            create_results_report(
                t_stats=t_stats,
                clusters=clusters,
                cluster_stats=cluster_stats,
                cluster_p_values=cluster_p_values,
                info=info,
                threshold=threshold,
                alpha=alpha,
                marker_name=marker_name,
                output_dir=str(output_dir)
            )
            print(f"✓ Figures saved to {output_dir}")
        
        return results_dict
        
    except Exception as e:
        print(f"✗ Error processing marker '{marker_name}': {e}")
        return None


def main(config_path: str = "Statistics/config.yaml", 
         run_all_markers: bool = False,
         specific_markers: list = None) -> None:
    """
    Execute complete LMM-based spatial cluster permutation pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    run_all_markers : bool
        If True, run analysis for all available markers
    specific_markers : list
        List of specific marker names to analyze (overrides run_all_markers)
    """
    print("="*80)
    print("LMM-BASED SPATIAL CLUSTER PERMUTATION TESTING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Extract parameters
    features_root = config['project']['features_root']
    output_path = config['project']['output_path']
    montage_path = config['project']['montage_path']
    
    # Data filtering parameters
    subjects = config['project'].get('subjects', None)
    tasks = config['project'].get('tasks', None)
    marker_types = config['project'].get('marker_types', None)
    marker_name = config['project'].get('marker_name', None)  # Make optional
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    print("\n" + "-"*80)
    print("STEP 1: Loading data")
    print("-"*80)
    print(f"Features root: {features_root}")
    
    # Load all aggregated probe data
    print("Loading all aggregated probe data...")
    df_all = load_all_probe_data(
        features_root=features_root,
        subjects=subjects,
        tasks=tasks,
        marker_types=marker_types,
        verbose=True
    )
    
    # Determine which markers to process
    if specific_markers is not None:
        markers_to_process = specific_markers
        print(f"Processing specific markers: {markers_to_process}")
    elif run_all_markers:
        # Get all available markers (state and evoked kept separate)
        available_markers_dict = get_available_markers(features_root, marker_types)
        markers_to_process = []
        for marker_type, markers in available_markers_dict.items():
            markers_to_process.extend(markers)
            print(f"  {marker_type} markers: {len(markers)}")
        print(f"Processing all available markers: {len(markers_to_process)} markers total")
    elif marker_name is not None:
        markers_to_process = [marker_name]
        print(f"Processing single marker: {marker_name}")
    else:
        print("No markers specified. Use --all-markers or --markers or set marker_name in config.")
        return
    
    # Step 2: Set up channel information (do this once for all markers)
    print("\n" + "-"*80)
    print("STEP 2: Setting up channel information")
    print("-"*80)
    
    # Get channel names from the data (deterministic)
    data_channels = get_channel_names(features_root)
    print(f"Data channels: {len(data_channels)} channels")
    
    # Load montage and get adjacency matrix
    print(f"Loading montage from: {montage_path}")
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)
    
    montage_channels = montage.ch_names
    print(f"Montage channels: {len(montage_channels)} channels")
    
    # Find common channels between data and montage
    common_channels = sorted(list(set(data_channels) & set(montage_channels)))
    print(f"Common channels: {len(common_channels)} channels")
    
    if len(common_channels) < len(data_channels) * 0.8:
        raise ValueError(f"Insufficient channel overlap: {len(common_channels)}/{len(data_channels)} channels")
    
    # Create adjacency matrix for common channels only
    adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, common_channels)
    print(f"✓ Adjacency matrix computed: {adjacency.shape}")
    
    # Verify adjacency matrix matches our channels
    if not all(ch in ch_names_ordered for ch in common_channels):
        missing = set(common_channels) - set(ch_names_ordered)
        raise ValueError(f"Adjacency matrix missing channels: {missing}")
    
    # Create MNE Info object for visualization
    info = mne.create_info(ch_names=ch_names_ordered, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    
    # Step 3: Process each marker
    print("\n" + "-"*80)
    print("STEP 3: Processing markers")
    print("-"*80)
    
    successful_results = []
    failed_markers = []
    
    for i, current_marker in enumerate(markers_to_process):
        print(f"\nProcessing marker {i+1}/{len(markers_to_process)}: {current_marker}")
        
        result = process_single_marker(
            marker_name=current_marker,
            df_all=df_all,
            config=config,
            ch_names=ch_names_ordered,
            adjacency=adjacency,
            info=info
        )
        
        if result is not None:
            successful_results.append(result)
            print(f"✓ Successfully processed {current_marker}")
        else:
            failed_markers.append(current_marker)
            print(f"✗ Failed to process {current_marker}")
    
    # Step 4: Create summary report
    print("\n" + "-"*80)
    print("STEP 4: Creating summary report")
    print("-"*80)
    
    # Create summary of all results with marker type information
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'marker_name': result['marker_name'],
            'marker_type': result.get('marker_type', 'unknown'),
            'n_observations': result['power_data_shape'][0],
            'n_channels': result['power_data_shape'][1],
            'n_subjects': result['n_subjects'],
            'n_clusters': result['n_clusters'],
            'n_sig_clusters': result['n_sig_clusters'],
            't_stat_min': np.min(result['t_stats']),
            't_stat_max': np.max(result['t_stats']),
            't_stat_mean': np.mean(np.abs(result['t_stats'])),
            'predictor_of_interest': result['predictor_of_interest'],
            'formula': result['formula'],
            'analysis_timestamp': result.get('analysis_timestamp', 'unknown')
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "pipeline_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Pipeline summary saved to {summary_path}")
        
        # Create type-specific summaries for better organization
        create_marker_type_summaries(summary_df, output_dir)
        
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    
    # Print completion message
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {len(successful_results)} markers")
    print(f"Failed markers: {len(failed_markers)}")
    if failed_markers:
        print(f"Failed markers: {failed_markers}")
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
    parser.add_argument(
        "--all-markers",
        action="store_true",
        help="Run analysis for all available markers"
    )
    parser.add_argument(
        "--markers",
        nargs="+",
        help="List of specific marker names to analyze"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        run_all_markers=args.all_markers,
        specific_markers=args.markers
    )
