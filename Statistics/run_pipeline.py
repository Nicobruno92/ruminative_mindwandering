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
    spatial_cluster_permutation_test
)
from plot_results import create_results_report
from helpers import (
    parse_formula_components,
    extract_fixed_effects_from_formula,
    load_qa_summary,
    get_qa_exclusion_list,
    apply_preprocessing,
    load_pca_data,
    summarize_clusters
)
from multiple_comparisons import (
    correct_cluster_p_values,
    create_correction_summary
)
from generate_summary_report import generate_summary_report


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
        
        # Determine which cluster count to use (corrected if available)
        if 'n_sig_clusters_corrected' in type_df.columns:
            sig_clusters_col = 'n_sig_clusters_corrected'
        else:
            sig_clusters_col = 'n_sig_clusters_uncorrected'
        
        # Create analysis summary for this type
        analysis_summary = {
            'marker_type': marker_type,
            'n_markers': len(type_df),
            'n_significant_markers_uncorrected': len(type_df[type_df['n_sig_clusters_uncorrected'] > 0]),
            'n_significant_markers_corrected': len(type_df[type_df[sig_clusters_col] > 0]),
            'total_clusters': type_df['n_clusters'].sum(),
            'total_sig_clusters_uncorrected': type_df['n_sig_clusters_uncorrected'].sum(),
            'total_sig_clusters_corrected': type_df[sig_clusters_col].sum(),
            'correction_method': type_df['correction_method'].iloc[0] if len(type_df) > 0 else 'none',
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


def process_single_marker(marker_spec: tuple, df_all: pd.DataFrame, config: dict, 
                         ch_names: list, adjacency: np.ndarray, info: mne.Info,
                         pca_data: pd.DataFrame = None,
                         qa_exclusions_dict: dict = None) -> dict:
    """
    Process a single marker through the complete LMM cluster pipeline.
    
    Parameters
    ----------
    marker_spec : tuple
        Tuple of (marker_name, marker_type) to process
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
    marker_name, marker_type = marker_spec
    print(f"\n{'='*60}")
    print(f"PROCESSING MARKER: {marker_name} ({marker_type})")
    print(f"{'='*60}")
    
    try:
        # Extract parameters
        formula = config['lmm']['formula']
        predictor_of_interest = config['lmm'].get('predictor_of_interest')
        
        method = config['lmm']['method']
        maxiter = config['lmm']['maxiter']
        random_state = config['lmm']['random_state']
        
        threshold = config['clustering']['threshold']
        n_permutations = config['clustering']['n_permutations']
        alpha = config['clustering']['alpha']
        tail = config['clustering']['tail']
        seed = config['clustering']['seed']
        n_jobs = config['clustering']['n_jobs']
        permutation_method = config['clustering'].get('permutation_method')
        
        save_pickle = config['output']['save_pickle']
        save_csv = config['output']['save_csv']
        save_figures = config['output']['save_figures']
        output_path = config['project']['output_path']
        
        # Filter data for this specific marker type
        print(f"Preparing data for marker: {marker_name} ({marker_type})")
        df_marker_type = df_all[df_all['marker_type'] == marker_type]
        
        # Get onoff filter from config
        onoff_max_value = config['project'].get('onoff_max_value', None)
        
        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=df_marker_type,
            marker_name=marker_name,
            formula=formula,
            include_channels=None,
            exclude_channels=None,
            pca_data=pca_data,
            onoff_max_value=onoff_max_value
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
        
        # Apply preprocessing (e.g., subject-level normalization)
        power_data, preprocessing_info = apply_preprocessing(
            power_data=power_data,
            df_behavioral=df_behavioral,
            config=config,
            verbose=True
        )
        
        # Validate formula variables
        validate_formula_variables(df_behavioral, formula)
        
        # Run LMM for each channel
        print(f"Running LMM for {n_channels} channels...")
        t_stats, p_values, lmm_diagnostics = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_behavioral,
            formula=formula,
            predictor_of_interest=predictor_of_interest,
            method=method,
            maxiter=maxiter,
            random_state=random_state,
            return_diagnostics=True
        )
        
        # Print LMM convergence summary
        print(f"✓ LMM completed: {lmm_diagnostics['n_converged']}/{n_channels} channels converged ({100*lmm_diagnostics['convergence_rate']:.1f}%)")
        if lmm_diagnostics['n_failed'] > 0:
            print(f"  Warning: {lmm_diagnostics['n_failed']} channels failed to converge")
        
        print(f"✓ LMM completed")
        print(f"  T-statistics range: [{np.min(t_stats):.3f}, {np.max(t_stats):.3f}]")
        print(f"  Channels with |t| > {threshold}: {np.sum(np.abs(t_stats) > threshold)}")
        
        # Spatial cluster permutation test
        print("Running spatial cluster permutation test...")
        clusters, cluster_stats, cluster_p_values, cluster_diagnostics = spatial_cluster_permutation_test(
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
            maxiter=maxiter,
            permutation_method=permutation_method
        )
        
        n_clusters = len(clusters)
        n_sig_clusters = np.sum(cluster_p_values < alpha)
        
        print(f"✓ Cluster analysis completed")
        print(f"  Total clusters: {n_clusters}")
        print(f"  Significant clusters (α={alpha}): {n_sig_clusters}")
        
        # Create output directory structure: base / model_folder / marker_folder
        base_output_dir = Path(output_path)
        
        # Extract fixed effects from formula to create model folder
        model_folder_name = extract_fixed_effects_from_formula(formula)
        
        # Create safe marker name for folder
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        marker_folder_name = f"{marker_type}_{safe_marker_name}"
        
        # Create full output directory: base / model / marker
        output_dir = base_output_dir / model_folder_name / marker_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model: {model_folder_name}")
        print(f"Output directory: {output_dir}")
        
        # Save config file to marker-specific directory for reproducibility
        config_save_path = output_dir / "config.yaml"
        import yaml
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Config saved to {config_save_path}")
        
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
            'n_permutations': n_permutations,
            'preprocessing_info': preprocessing_info,
            'qa_filtering_applied': qa_exclusions_dict is not None and len(qa_exclusions_dict) > 0,
            'qa_exclusions_count': len(qa_exclusions_dict.get(marker_type, set())) if qa_exclusions_dict else 0,
            'lmm_diagnostics': lmm_diagnostics
        }
        
        # Save pickle (simplified filename since we're in marker-specific folder)
        if save_pickle:
            pickle_path = output_dir / "results.pkl"
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
            
            csv_path = output_dir / "cluster_summary.csv"
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
            t_stats_path = output_dir / "t_statistics.csv"
            t_stats_df.to_csv(t_stats_path, index=False)
            print(f"✓ T-statistics saved to {t_stats_path}")
            
            # Save model quality metrics per channel
            if lmm_diagnostics['n_converged'] > 0:
                model_quality_df = pd.DataFrame({
                    'channel': [ch_names_final[i] for i in range(len(lmm_diagnostics['aic']))],
                    'aic': lmm_diagnostics['aic'],
                    'bic': lmm_diagnostics['bic'],
                    'log_likelihood': lmm_diagnostics['log_likelihood'],
                    'conditional_r2': lmm_diagnostics['conditional_r2'],
                    'shapiro_p_value': lmm_diagnostics['shapiro_p_values'],
                    'breusch_pagan_p': lmm_diagnostics['breusch_pagan_p'],
                    'residual_variance': lmm_diagnostics['residual_variance'],
                    'marker_name': marker_name,
                    'marker_type': marker_type,
                    'analysis_timestamp': datetime.now().isoformat()
                })
                model_quality_path = output_dir / "model_quality.csv"
                model_quality_df.to_csv(model_quality_path, index=False)
                print(f"✓ Model quality metrics saved to {model_quality_path}")
                
                # Save summary of model diagnostics
                diagnostics_summary = {
                    'marker_name': marker_name,
                    'marker_type': marker_type,
                    'n_channels': n_channels,
                    'n_converged': lmm_diagnostics['n_converged'],
                    'n_failed': lmm_diagnostics['n_failed'],
                    'convergence_rate': lmm_diagnostics['convergence_rate'],
                    'aic_mean': lmm_diagnostics.get('aic_mean', np.nan),
                    'bic_mean': lmm_diagnostics.get('bic_mean', np.nan),
                    'log_likelihood_mean': lmm_diagnostics.get('log_likelihood_mean', np.nan),
                    'conditional_r2_mean': lmm_diagnostics.get('conditional_r2_mean', np.nan),
                    'conditional_r2_median': lmm_diagnostics.get('conditional_r2_median', np.nan),
                    'shapiro_p_mean': lmm_diagnostics.get('shapiro_p_mean', np.nan),
                    'n_normality_violations': lmm_diagnostics.get('n_normality_violations', 0),
                    'pct_normality_violations': lmm_diagnostics.get('pct_normality_violations', np.nan),
                    'breusch_pagan_p_mean': lmm_diagnostics.get('breusch_pagan_p_mean', np.nan),
                    'n_heteroscedasticity': lmm_diagnostics.get('n_heteroscedasticity', 0),
                    'pct_heteroscedasticity': lmm_diagnostics.get('pct_heteroscedasticity', np.nan),
                    'analysis_timestamp': datetime.now().isoformat()
                }
                diagnostics_summary_df = pd.DataFrame([diagnostics_summary])
                diagnostics_summary_path = output_dir / "lmm_diagnostics_summary.csv"
                diagnostics_summary_df.to_csv(diagnostics_summary_path, index=False)
                print(f"✓ LMM diagnostics summary saved to {diagnostics_summary_path}")
        
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
         marker_index: int = None) -> None:
    """
    Execute complete LMM-based spatial cluster permutation pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    marker_index : int, optional
        Index of specific marker to process (for SLURM array jobs)
        If None, processes all markers configured in config.yaml
    """
    print("="*80)
    print("LMM-BASED SPATIAL CLUSTER PERMUTATION TESTING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    
    # Validate and display formula configuration
    formula = config['lmm']['formula']
    predictor_config = config['lmm'].get('predictor_of_interest', 'auto')
    
    print("\n" + "-"*80)
    print("FORMULA CONFIGURATION")
    print("-"*80)
    print(f"Formula: {formula}")
    
    # Parse formula to show components
    formula_components = parse_formula_components(formula)
    print(f"  Response variable: {formula_components['response']}")
    print(f"  Fixed effects: {', '.join(formula_components['fixed_effects'])}")
    print(f"  Random effects: {', '.join(formula_components['random_effects'])}")
    if formula_components['has_interaction']:
        print(f"  ⚠ Formula contains interactions (*)")
    if formula_components['has_random_slopes']:
        print(f"  ⚠ Formula contains random slopes")
    
    # Extract parameters
    features_root = config['project']['features_root']
    output_path = config['project']['output_path']
    montage_path = config['project']['montage_path']
    
    # Data filtering parameters
    subjects = config['project'].get('subjects', None)
    tasks = config['project'].get('tasks', None)
    marker_types = config['project'].get('marker_types', None)
    markers_config = config['project'].get('markers', 'all')  # Read markers from config
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config file to base output directory for reference
    import yaml
    config_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_backup_path = output_dir / f"config_{config_timestamp}.yaml"
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Config saved to {config_backup_path}")
    
    # Step 1: Load data
    print("\n" + "-"*80)
    print("STEP 1: Loading data")
    print("-"*80)
    print(f"Features root: {features_root}")
    
    # Load QA summary if configured
    qa_exclusions_dict = {}
    qa_summary_info = None
    qa_summary_path = config['project'].get('qa_summary_path', None)
    exclude_failed_qa = config['project'].get('exclude_failed_qa', False)
    
    if qa_summary_path and exclude_failed_qa:
        print("\nLoading QA summary...")
        qa_df = load_qa_summary(qa_summary_path, verbose=True)
        
        # Get exclusion lists for each marker type
        if marker_types:
            for marker_type in marker_types:
                exclusion_set, exclusion_info = get_qa_exclusion_list(
                    qa_df, marker_type, verbose=True
                )
                qa_exclusions_dict[marker_type] = exclusion_set
        
        qa_summary_info = {
            'qa_summary_path': qa_summary_path,
            'exclusions_by_type': {
                mtype: len(excl_set)
                for mtype, excl_set in qa_exclusions_dict.items()
            }
        }
        print("✓ QA filtering enabled")
    elif qa_summary_path and not exclude_failed_qa:
        print("\nQA summary path provided but exclude_failed_qa is False")
        print("Skipping QA filtering...")
    else:
        print("\nNo QA filtering configured")
    
    # Load all aggregated probe data
    print("\nLoading all aggregated probe data...")
    df_all = load_all_probe_data(
        features_root=features_root,
        subjects=subjects,
        tasks=tasks,
        marker_types=marker_types,
        qa_exclusions=qa_exclusions_dict if qa_exclusions_dict else None,
        verbose=True
    )
    
    # Load PCA data if path is provided in config
    pca_data = None
    pca_results_path = config['project'].get('pca_results_path', None)
    if pca_results_path:
        print("\nLoading PCA data...")
        pca_data = load_pca_data(pca_results_path, verbose=True)
        print("✓ PCA data loaded successfully")
    else:
        print("\nNo PCA data path provided in config. Skipping PCA data loading.")
    
    # Determine which markers to process based on config
    # Create list of (marker_name, marker_type) tuples to handle markers in both evoked and state
    if isinstance(markers_config, str) and markers_config.lower() == 'all':
        # Get all available markers (state and evoked kept separate)
        available_markers_dict = get_available_markers(features_root, marker_types)
        markers_to_process = []
        for marker_type, markers in available_markers_dict.items():
            # Create tuples of (marker_name, marker_type)
            markers_to_process.extend([(m, marker_type) for m in markers])
            print(f"  {marker_type} markers: {len(markers)}")
        print(f"Processing all available markers: {len(markers_to_process)} markers total")
    elif isinstance(markers_config, list):
        # Process specific markers from config
        # For each marker, check which types it exists in
        available_markers_dict = get_available_markers(features_root, marker_types)
        markers_to_process = []
        for marker_name in markers_config:
            for marker_type, type_markers in available_markers_dict.items():
                if marker_name in type_markers:
                    markers_to_process.append((marker_name, marker_type))
        print(f"Processing {len(markers_to_process)} marker-type combinations from config")
        if len(markers_to_process) == 0:
            print(f"Warning: No markers found matching config: {markers_config}")
            return
    else:
        print(f"Invalid markers configuration: {markers_config}")
        print("Set project.markers to 'all' or a list of marker names")
        return
    
    # If marker_index is provided (SLURM array job), process only that marker
    if marker_index is not None:
        if marker_index < 0 or marker_index >= len(markers_to_process):
            print(f"Error: marker_index {marker_index} out of range (0-{len(markers_to_process)-1})")
            return
        selected_marker = markers_to_process[marker_index]
        markers_to_process = [selected_marker]
        print(f"SLURM array mode: Processing marker {marker_index}: {selected_marker[0]} ({selected_marker[1]})")
    
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
    
    for i, marker_spec in enumerate(markers_to_process):
        marker_name, marker_type = marker_spec
        print(f"\nProcessing marker {i+1}/{len(markers_to_process)}: {marker_name} ({marker_type})")
        
        result = process_single_marker(
            marker_spec=marker_spec,
            df_all=df_all,
            config=config,
            ch_names=ch_names_ordered,
            adjacency=adjacency,
            info=info,
            pca_data=pca_data,
            qa_exclusions_dict=qa_exclusions_dict
        )
        
        if result is not None:
            successful_results.append(result)
            print(f"✓ Successfully processed {marker_name} ({marker_type})")
        else:
            failed_markers.append(marker_spec)
            print(f"✗ Failed to process {marker_name} ({marker_type})")
    
    # Step 3.5: Apply multiple comparisons correction
    print("\n" + "-"*80)
    print("STEP 3.5: Multiple comparisons correction")
    print("-"*80)
    
    # Get correction settings from config
    mcc_config = config.get('multiple_comparisons', {})
    mcc_alpha = mcc_config.get('alpha', config['clustering']['alpha'])
    
    # Apply correction separately for evoked and state markers
    for marker_type in ['evoked', 'state']:
        # Get correction method for this marker type
        correction_method = mcc_config.get(marker_type, False)
        
        # Filter results for this marker type
        type_results = [r for r in successful_results if r.get('marker_type') == marker_type]
        
        if len(type_results) == 0:
            print(f"\nNo {marker_type} markers to correct")
            continue
        
        print(f"\n{marker_type.upper()} markers:")
        print(f"  Number of markers: {len(type_results)}")
        
        # Apply correction
        corrected_results = correct_cluster_p_values(
            results_list=type_results,
            correction_method=correction_method,
            alpha=mcc_alpha,
            verbose=True
        )
        
        # Update the results in successful_results
        for corrected in corrected_results:
            marker_name = corrected['marker_name']
            for i, result in enumerate(successful_results):
                if result['marker_name'] == marker_name and result.get('marker_type') == marker_type:
                    successful_results[i] = corrected
                    break
    
    # Save correction summary
    if successful_results:
        base_output_dir = Path(config['project']['output_path'])
        formula = config['lmm']['formula']
        model_folder_name = extract_fixed_effects_from_formula(formula)
        model_output_dir = base_output_dir / model_folder_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config file to model directory for easy reference
        model_config_path = model_output_dir / "config.yaml"
        with open(model_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Config saved to model directory: {model_config_path}")
        
        correction_summary_path = model_output_dir / "multiple_comparisons_summary.csv"
        correction_summary = create_correction_summary(
            successful_results,
            output_path=str(correction_summary_path)
        )
        print(f"\n✓ Multiple comparisons summary:")
        print(correction_summary.to_string(index=False))
    
    # Step 4: Create summary report
    print("\n" + "-"*80)
    print("STEP 4: Creating summary report")
    print("-"*80)
    
    # Create summary of all results with marker type information
    summary_data = []
    for result in successful_results:
        # Count significant clusters (corrected if available)
        if 'cluster_p_values_corrected' in result:
            alpha = result.get('correction_alpha', mcc_alpha)
            n_sig_corrected = np.sum(result['cluster_p_values_corrected'] <= alpha)
        else:
            n_sig_corrected = result['n_sig_clusters']
        
        # Extract LMM diagnostics if available
        lmm_diag = result.get('lmm_diagnostics', {})
        
        summary_data.append({
            'marker_name': result['marker_name'],
            'marker_type': result.get('marker_type', 'unknown'),
            'n_observations': result['power_data_shape'][0],
            'n_channels': result['power_data_shape'][1],
            'n_subjects': result['n_subjects'],
            'n_clusters': result['n_clusters'],
            'n_sig_clusters_uncorrected': result['n_sig_clusters'],
            'n_sig_clusters_corrected': n_sig_corrected,
            'correction_method': result.get('correction_method', 'none'),
            't_stat_min': np.min(result['t_stats']),
            't_stat_max': np.max(result['t_stats']),
            't_stat_mean': np.mean(np.abs(result['t_stats'])),
            'predictor_of_interest': result['predictor_of_interest'],
            'formula': result['formula'],
            'qa_filtering_applied': result.get('qa_filtering_applied', False),
            'qa_exclusions_count': result.get('qa_exclusions_count', 0),
            # Model quality metrics
            'lmm_converged': lmm_diag.get('n_converged', 0),
            'lmm_convergence_rate': lmm_diag.get('convergence_rate', np.nan),
            'aic_mean': lmm_diag.get('aic_mean', np.nan),
            'bic_mean': lmm_diag.get('bic_mean', np.nan),
            'conditional_r2_mean': lmm_diag.get('conditional_r2_mean', np.nan),
            'conditional_r2_median': lmm_diag.get('conditional_r2_median', np.nan),
            'shapiro_p_mean': lmm_diag.get('shapiro_p_mean', np.nan),
            'pct_normality_violations': lmm_diag.get('pct_normality_violations', np.nan),
            'pct_heteroscedasticity': lmm_diag.get('pct_heteroscedasticity', np.nan),
            'analysis_timestamp': result.get('analysis_timestamp', 'unknown')
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Extract model folder name from formula
        formula = config['lmm']['formula']
        model_folder_name = extract_fixed_effects_from_formula(formula)
        
        # Save summary to model-specific directory
        base_output_dir = Path(config['project']['output_path'])
        model_output_dir = base_output_dir / model_folder_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = model_output_dir / "pipeline_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Pipeline summary saved to {summary_path}")
        
        # Create type-specific summaries for better organization
        create_marker_type_summaries(summary_df, model_output_dir)
        
        print("\nSummary:")
        print(summary_df.to_string(index=False))
    
    # Step 5: Generate comprehensive summary report (only if not processing single marker)
    if marker_index is None and successful_results:
        print("\n" + "-"*80)
        print("STEP 5: Generating comprehensive summary report")
        print("-"*80)
        
        formula = config['lmm']['formula']
        model_folder_name = extract_fixed_effects_from_formula(formula)
        base_output_dir = Path(config['project']['output_path'])
        model_output_dir = base_output_dir / model_folder_name
        
        try:
            # Generate report with all topoplots and tables
            generate_summary_report(
                model_dir=model_output_dir,
                alpha=config['clustering']['alpha'],
                use_corrected=True,
                verbose=True
            )
        except Exception as e:
            print(f"⚠ Warning: Failed to generate summary report: {e}")
            print("  Individual marker results are still available.")
    
    # Print completion message
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {len(successful_results)} markers")
    print(f"Failed markers: {len(failed_markers)}")
    if failed_markers:
        print(f"Failed markers: {failed_markers}")
    
    # Show QA filtering summary if applied
    if qa_exclusions_dict:
        print("\nQA Filtering Summary:")
        for mtype, excl_set in qa_exclusions_dict.items():
            print(f"  {mtype}: {len(excl_set)} files excluded")
    
    # Show model-specific results directory
    formula = config['lmm']['formula']
    model_folder_name = extract_fixed_effects_from_formula(formula)
    base_output_dir = Path(config['project']['output_path'])
    model_output_dir = base_output_dir / model_folder_name
    print(f"Model: {model_folder_name}")
    print(f"Results directory: {model_output_dir.absolute()}")
    
    if marker_index is None and successful_results:
        print("\n" + "="*80)
        print("SUMMARY REPORT FILES")
        print("="*80)
        print("Check the model directory for:")
        print("  - SUMMARY_REPORT_*.csv      : Complete results table")
        print("  - SUMMARY_TOPOPLOTS_*.pdf   : All topoplots for comparison")
        print("  - SUMMARY_DETAILED_*.xlsx   : Detailed Excel with multiple sheets")
        print("="*80)


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
        "--marker-index",
        type=int,
        default=None,
        help="Index of marker to process (for SLURM array jobs)"
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        marker_index=args.marker_index
    )
