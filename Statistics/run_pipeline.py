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
    filter_subjects_by_variability,
    validate_formula_variables,
    get_channel_names,
    get_available_markers
)
from lmm_model import run_lmm_per_channel
from cluster_test import (
    get_channel_adjacency,
    spatial_cluster_permutation_test,
    spatial_cluster_test_tfce
)
from scipy.sparse import issparse
from plot_results import create_results_report, create_raw_topography_report
from helpers import (
    parse_formula_components,
    extract_fixed_effects_from_formula,
    get_model_folder_name,
    load_qa_summary,
    get_qa_exclusion_list,
    apply_preprocessing,
    load_pca_data,
    summarize_clusters,
    normalize_predictors
)
from generate_summary_report import generate_summary_report, generate_pipeline_qa_html_report


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
                        adjacency: np.ndarray, info: mne.Info,
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
    adjacency : np.ndarray
        Channel adjacency matrix
    info : mne.Info
        MNE Info object for visualization (contains canonical channel order)
        
    Returns
    -------
    dict
        Results dictionary for this marker
    """
    marker_name, marker_type = marker_spec
    print(f"\n{'='*60}")
    print(f"PROCESSING MARKER: {marker_name} ({marker_type})")
    print(f"{'='*60}")
    
    if True:
        # Extract parameters
        formula = config['lmm']['formula']
        predictor_of_interest = config['lmm'].get('predictor_of_interest')
        
        method = config['lmm']['method']
        maxiter = config['lmm']['maxiter']
        random_state = config['lmm']['random_state']
        
        clustering_method = config['clustering'].get('method', 'threshold')
        threshold = config['clustering']['threshold']
        n_permutations = config['clustering']['n_permutations']
        alpha = config['clustering']['alpha']
        tail = config['clustering']['tail']
        seed = config['clustering']['seed']
        n_jobs = config['clustering']['n_jobs']
        permutation_method = config['clustering'].get('permutation_method')
        t_power = config['clustering'].get('t_power', 1.0)  # Default to 1.0 if not specified
        
        # TFCE parameters (only used if method='tfce')
        tfce_E = config['clustering'].get('tfce', {}).get('E', 0.5)
        tfce_H = config['clustering'].get('tfce', {}).get('H', 2.0)
        tfce_n_steps = config['clustering'].get('tfce', {}).get('n_steps', 100)
        
        save_pickle = config['output']['save_pickle']
        save_csv = config['output']['save_csv']
        save_figures = config['output']['save_figures']
        output_path = config['project']['output_path']
        
        # Filter data for this specific marker type
        print(f"Preparing data for marker: {marker_name} ({marker_type})")
        df_marker_type = df_all[df_all['marker_type'] == marker_type]
        
        # Get filtering parameters from config
        onoff_max_value = config['project'].get('onoff_max_value', None)
        min_predictor_variability = config['project'].get('min_predictor_variability', None)
        min_minority_ratio = config['project'].get('min_minority_ratio', None)
        
        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=df_marker_type,
            marker_name=marker_name,
            formula=formula,
            include_channels=None,
            exclude_channels=None,
            pca_data=pca_data,
            onoff_max_value=onoff_max_value,
            min_predictor_variability=min_predictor_variability,
            min_minority_ratio=min_minority_ratio,
            predictor_of_interest=predictor_of_interest
        )
        
        # CRITICAL: Project data to info order (canonical order established in main())
        # The info object contains the canonical channel order that must be preserved
        info_order = [ch for ch in info['ch_names'] if ch in channels]
        
        if len(info_order) == 0:
            raise ValueError(f"No common channels between data ({len(channels)} channels) and info ({len(info['ch_names'])} channels)")
        
        if len(info_order) < len(channels) * 0.8:  # Less than 80% overlap
            missing_data = sorted(list(set(channels) - set(info['ch_names'])))
            missing_info = sorted(list(set(info['ch_names']) - set(channels)))
            raise ValueError(f"Insufficient channel overlap. Missing from info: {missing_data[:5]}... Missing from data: {missing_info[:5]}...")
        
        # Project data to canonical info order
        idx_in_data = [channels.index(ch) for ch in info_order]
        power_data = power_data[:, idx_in_data]
        ch_names_final = info_order  # Use info order as final order
        n_observations, n_channels = power_data.shape
        
        # CRITICAL ASSERTION: Ensure data and info are aligned
        assert list(ch_names_final) == list(info['ch_names']), \
            f"Channel order mismatch between data and info. Data: {ch_names_final[:5]}..., Info: {info['ch_names'][:5]}..."
        assert power_data.shape[1] == len(info['ch_names']), \
            f"Data shape mismatch: power_data has {power_data.shape[1]} channels, info has {len(info['ch_names'])} channels"
        
        print(f"✓ Data prepared: {n_observations} observations × {n_channels} channels")
        
        # Apply preprocessing (e.g., subject-level normalization)
        power_data, preprocessing_info = apply_preprocessing(
            power_data=power_data,
            df_behavioral=df_behavioral,
            config=config,
            verbose=True
        )
        
        # Apply predictor normalization if enabled
        # This normalizes independent variables (like onoff, confidence) within subjects
        predictor_norm_config = config['preprocessing'].get('predictor_normalization', {})
        if predictor_norm_config.get('enabled', False):
            # Apply normalization to the behavioral dataframe
            # This is done INPLACE on the copy or returns a new df
            df_behavioral = normalize_predictors(
                df=df_behavioral,
                method=predictor_norm_config.get('method', 'zscore'),
                subject_col='subject',
                predictors=predictor_norm_config.get('predictors', 'all'),
                verbose=True
            )
            
            # Log this step
            preprocessing_info['steps_applied'].append(
                f"normalize_predictors_{predictor_norm_config.get('method', 'zscore')}"
            )
        
        # Create output directory structure: base / model_folder / marker_folder
        base_output_dir = Path(output_path)
        
        # Build model folder name: fixed effects + active predictor of interest
        model_folder_name = get_model_folder_name(formula, predictor_of_interest)
        
        # Create safe marker name for folder
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        marker_folder_name = f"{marker_type}_{safe_marker_name}"
        
        # Create full output directory: base / model / marker
        output_dir = base_output_dir / model_folder_name / marker_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate raw topography report (before statistical analysis)
        print("\nCreating raw topography report...")
        create_raw_topography_report(
            power_data=power_data,
            df_behavioral=df_behavioral,
            ch_names=ch_names_final,
            info=info,
            marker_name=marker_name,
            output_dir=str(output_dir),
            subject_col='subject',
            predictor_of_interest=predictor_of_interest
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
        if clustering_method == 'threshold':
            print(f"  Channels with |t| > {threshold}: {np.sum(np.abs(t_stats) > threshold)}")
        
        # ========== CREATE EXCLUDE MASK (if configured) ==========
        # This prevents boundary artifacts from edge channels
        exclude_mask = None
        exclude_config = config['clustering'].get('exclude_channels', {})
        if exclude_config.get('enabled', False):
            exclude_method = exclude_config.get('method', 'manual')
            
            if exclude_method == 'manual':
                # Exclude specific channel names
                channel_names_to_exclude = exclude_config.get('channel_names', [])
                if channel_names_to_exclude:
                    exclude_mask = np.zeros(n_channels, dtype=bool)
                    for ch_name in channel_names_to_exclude:
                        if ch_name in ch_names_final:
                            ch_idx = ch_names_final.index(ch_name)
                            exclude_mask[ch_idx] = True
                    
                    n_excluded = np.sum(exclude_mask)
                    print(f"\n✓ Excluding {n_excluded}/{n_channels} channels from clustering (manual method)")
                    print(f"  Excluded channels: {[ch for ch in channel_names_to_exclude if ch in ch_names_final]}")
                    
            elif exclude_method == 'auto_position':
                # Exclude outermost channels by position
                percentile = exclude_config.get('auto_percentile', 10)
                from mne.channels import find_layout
                if True:
                    layout = find_layout(info)
                    pos = layout.pos[:, :2]  # x, y positions
                    
                    # Calculate distance from center
                    center = pos.mean(axis=0)
                    distances = np.linalg.norm(pos - center, axis=1)
                    
                    # Channels in outer percentile
                    threshold_dist = np.percentile(distances, 100 - percentile)
                    exclude_mask = distances >= threshold_dist
                    
                    n_excluded = np.sum(exclude_mask)
                    excluded_names = [ch_names_final[i] for i in range(n_channels) if exclude_mask[i]]
                    print(f"\n✓ Excluding {n_excluded}/{n_channels} channels from clustering (auto_position method, {percentile}th percentile)")
                    print(f"  Excluded channels: {excluded_names}")
        
        # Spatial cluster permutation test - choose method
        if clustering_method == 'tfce':
            print(f"Running TFCE-based spatial permutation test...")
            print(f"  TFCE parameters: E={tfce_E}, H={tfce_H}, n_steps={tfce_n_steps}")
            
            # TFCE returns channel-wise results (not clusters)
            tfce_map, tfce_p_values, cluster_diagnostics = spatial_cluster_test_tfce(
                observed_t_stats=t_stats,
                power_data=power_data,
                df_behavioral=df_behavioral,
                formula=formula,
                predictor_of_interest=predictor_of_interest,
                adjacency=adjacency,
                n_permutations=n_permutations,
                E=tfce_E,
                H=tfce_H,
                n_tfce_steps=tfce_n_steps,
                seed=seed,
                n_jobs=n_jobs,
                method=method,
                maxiter=maxiter,
                verbose=True,
                return_diagnostics=True,
                exclude=exclude_mask
            )
            
            # For TFCE, create spatially-connected clusters from significant channels
            sig_channels = np.where(tfce_p_values < alpha)[0]
            if len(sig_channels) > 0:
                # Find spatially connected components using adjacency matrix
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import connected_components
                
                # CRITICAL FIX: Mask the full adjacency matrix instead of creating subgraph
                # This preserves true spatial adjacency relationships
                # Convert to dense if sparse
                if issparse(adjacency):
                    adj_array = adjacency.toarray()
                else:
                    adj_array = adjacency.copy()
                
                # Create masked adjacency: zero out rows/cols for non-significant channels
                masked_adj = np.zeros_like(adj_array)
                masked_adj[np.ix_(sig_channels, sig_channels)] = \
                    adj_array[np.ix_(sig_channels, sig_channels)]
                
                # Find connected components on the masked adjacency
                n_components, labels = connected_components(
                    csgraph=csr_matrix(masked_adj),
                    directed=False,
                    return_labels=True
                )
                
                # Create clusters (one per connected component)
                # Note: labels has length n_channels (full size), not just sig_channels
                clusters = []
                cluster_stats = []
                cluster_p_values = []
                
                for comp_idx in range(n_components):
                    # Get all channels in this component
                    comp_channels = np.where(labels == comp_idx)[0]
                    
                    # Filter to only significant channels
                    sig_mask = np.isin(comp_channels, sig_channels)
                    comp_channels = comp_channels[sig_mask]
                    
                    if len(comp_channels) > 0:
                        clusters.append(comp_channels)
                        cluster_stats.append(np.sum(tfce_map[comp_channels]))
                        cluster_p_values.append(np.min(tfce_p_values[comp_channels]))
                
                cluster_stats = np.array(cluster_stats)
                cluster_p_values = np.array(cluster_p_values)
            else:
                clusters = []
                cluster_stats = np.array([])
                cluster_p_values = np.array([])
                
        else:  # threshold-based clustering
            print("Running threshold-based spatial cluster permutation test...")
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
                t_power=t_power,
                permutation_method=permutation_method,
                exclude=exclude_mask
            )
        
        n_clusters = len(clusters)
        n_sig_clusters = np.sum(cluster_p_values < alpha)
        
        print(f"✓ Cluster analysis completed")
        print(f"  Total clusters: {n_clusters}")
        print(f"  Significant clusters (α={alpha}): {n_sig_clusters}")
        
        # Output directory already created above
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
            'target_variable': predictor_of_interest,  # Explicitly highlight target variable
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
            'clustering_method': clustering_method,
            'threshold': threshold if clustering_method == 'threshold' else None,
            'tfce_params': {'E': tfce_E, 'H': tfce_H, 'n_steps': tfce_n_steps} if clustering_method == 'tfce' else None,
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
        
        # Save CSV summary (UNCORRECTED p-values)
        if save_csv and n_clusters > 0:
            cluster_summary = summarize_clusters(
                clusters, cluster_stats, cluster_p_values, ch_names_final, alpha
            )
            
            # Add metadata columns
            cluster_summary['target_variable'] = predictor_of_interest
            cluster_summary['marker_name'] = marker_name
            cluster_summary['marker_type'] = marker_type
            
            # Reorder columns to highlight the target variable
            cols = cluster_summary.columns.tolist()
            if 'p_value' in cols:
                # Move key metadata/results up so the target variable is visually prominent
                base_cols = ['target_variable', 'marker_name', 'marker_type', 'cluster_id', 'p_value', 'statistic']
                other_cols = [c for c in cols if c not in base_cols]
                cluster_summary = cluster_summary[base_cols + other_cols]
            
            csv_path = output_dir / "cluster_summary_uncorrected.csv"
            cluster_summary.to_csv(csv_path, index=False)
            print(f"✓ Cluster summary (uncorrected) saved to {csv_path}")
        
        # Save t-statistics per channel
        if save_csv:
            t_stats_df = pd.DataFrame({
                'target_variable': predictor_of_interest,
                'marker_name': marker_name,
                'marker_type': marker_type,
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
                output_dir=str(output_dir),
                config=config
            )
            print(f"✓ Figures saved to {output_dir}")
        
        return {'success': True, 'result': results_dict}


def main(config_path: str = "Statistics/config.yaml",
         marker_index: int = None,
         predictor_of_interest: str = None) -> None:
    """
    Execute complete LMM-based spatial cluster permutation pipeline.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
    marker_index : int, optional
        Index of specific marker to process (for SLURM array jobs).
        If None, processes all markers configured in config.yaml.
    predictor_of_interest : str, optional
        Override for ``lmm.predictor_of_interest`` in config.yaml.
        Required when the config entry is a list (multi-predictor loop mode).
        When provided, this value is written back into the config dict so all
        downstream functions (including process_single_marker) use it.
    """
    print("="*80)
    print("LMM-BASED SPATIAL CLUSTER PERMUTATION TESTING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    # ------------------------------------------------------------------
    # Resolve predictor_of_interest
    # The config value may be a single string or a list (multi-predictor
    # loop mode).  A CLI override always takes precedence.  When the
    # config holds a list and no override is provided, raise immediately
    # so the user knows they must use submit_predictor_loop.sh.
    # ------------------------------------------------------------------
    poi_in_config = config['lmm'].get('predictor_of_interest', 'auto')
    if predictor_of_interest is not None:
        # CLI / programmatic override
        config['lmm']['predictor_of_interest'] = predictor_of_interest
    elif isinstance(poi_in_config, list):
        raise ValueError(
            f"config.yaml 'lmm.predictor_of_interest' is a list {poi_in_config}. "
            "When running directly, specify a single predictor via "
            "'--predictor-of-interest <name>'.  To submit one SLURM job per "
            "predictor automatically, use: "
            "bash Statistics/submit_predictor_loop.sh"
        )
    # else: config already has a single string — nothing to do

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
    selected_markers = config.get('selected_markers', {})
    feature_families = config.get('feature_families', {})
    
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
        if selected_markers:
            for marker_type in selected_markers.keys():
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
        marker_types=list(selected_markers.keys()) if selected_markers else None,
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
    
    # Determine which markers to process based on configured selected_markers
    markers_to_process = []
    
    assert selected_markers, "No selected_markers section found in config."
    
    available_markers_dict = get_available_markers(features_root)
    
    print("\n--- Marker resolution (feature_families) ---")
    for marker_type, family_list in selected_markers.items():
        if marker_type not in available_markers_dict:
            print(f"  Warning: epoch type '{marker_type}' not found in data — skipping.")
            continue
        
        available_type_markers = available_markers_dict[marker_type]
        
        # Special case: ["all"] → include every marker for this epoch type
        if isinstance(family_list, list) and len(family_list) == 1 and \
                str(family_list[0]).lower() == 'all':
            markers_to_process.extend([(m, marker_type) for m in available_type_markers])
            print(f"  {marker_type}: all ({len(available_type_markers)} markers)")
            continue
        
        # Resolve family names → fragments → filter by substring
        assert isinstance(family_list, list), \
            f"selected_markers['{marker_type}'] must be a list, got {type(family_list)}"
        
        resolved_fragments = []
        for family_name in family_list:
            assert family_name in feature_families, (
                f"Family '{family_name}' referenced in selected_markers['{marker_type}'] "
                f"is not defined in feature_families. "
                f"Available families: {list(feature_families.keys())}"
            )
            resolved_fragments.extend(feature_families[family_name])
        
        # Filter available markers: include if any fragment is a substring of the marker name
        matched = [
            m for m in available_type_markers
            if any(frag in m for frag in resolved_fragments)
        ]
        
        markers_to_process.extend([(m, marker_type) for m in matched])
        print(
            f"  {marker_type}: families={family_list} → "
            f"{len(matched)} markers matched "
            f"(out of {len(available_type_markers)} available)"
        )
    
    print(f"\nTotal markers to process: {len(markers_to_process)}")
    assert len(markers_to_process) > 0, \
        "No markers matched the configured feature_families / selected_markers. " \
        "Check that marker names in the data contain the configured fragments."
    
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
    
    # Find common channels respecting montage order (not sorted)
    common_channels = [ch for ch in montage_channels if ch in data_channels]
    print(f"Common channels: {len(common_channels)} channels (montage order preserved)")
    
    if len(common_channels) < len(data_channels) * 0.8:
        raise ValueError(f"Insufficient channel overlap: {len(common_channels)}/{len(data_channels)} channels")
    
    # Create adjacency matrix for common channels only
    adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, common_channels)
    print(f"✓ Adjacency matrix computed: {adjacency.shape}")
    
    # Verify adjacency matrix matches our channels
    if not all(ch in ch_names_ordered for ch in common_channels):
        missing = set(common_channels) - set(ch_names_ordered)
        raise ValueError(f"Adjacency matrix missing channels: {missing}")
    
    # Create MNE Info object for visualization using the canonical order from adjacency
    info = mne.create_info(ch_names=ch_names_ordered, sfreq=250, ch_types='eeg')
    info.set_montage(montage)
    
    # CRITICAL VALIDATION: Ensure adjacency and info are perfectly aligned
    assert adjacency.shape[0] == adjacency.shape[1] == len(info['ch_names']), \
        f"Adjacency ({adjacency.shape}) no coincide con #canales de info ({len(info['ch_names'])})"
    assert list(ch_names_ordered) == list(info['ch_names']), \
        f"Channel order mismatch: adjacency returned {ch_names_ordered[:5]}..., info has {info['ch_names'][:5]}..."
    
    # CRITICAL: ch_names_ordered is now the canonical channel order for the entire pipeline
    print(f"✓ Canonical channel order established: {len(ch_names_ordered)} channels")
    print(f"✓ Adjacency matrix validated: {adjacency.shape} matches {len(info['ch_names'])} channels")
    
    # Save channel order for audit trail
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_order_path = output_dir / "channel_order.txt"
    with open(channel_order_path, 'w') as f:
        f.write("# Canonical channel order for this analysis\n")
        f.write(f"# Total channels: {len(ch_names_ordered)}\n")
        f.write(f"# Analysis timestamp: {datetime.now().isoformat()}\n\n")
        for i, ch in enumerate(ch_names_ordered):
            f.write(f"{i:3d}: {ch}\n")
    print(f"✓ Channel order saved for audit: {channel_order_path}")
    
    # Step 3: Process each marker
    print("\n" + "-"*80)
    print("STEP 3: Processing markers")
    print("-"*80)
    
    successful_results = []
    failed_markers = []  # List of dicts with 'marker' and 'error' keys
    
    for i, marker_spec in enumerate(markers_to_process):
        marker_name, marker_type = marker_spec
        print(f"\nProcessing marker {i+1}/{len(markers_to_process)}: {marker_name} ({marker_type})")
        
        outcome = process_single_marker(
            marker_spec=marker_spec,
            df_all=df_all,
            config=config,
            adjacency=adjacency,
            info=info,
            pca_data=pca_data,
            qa_exclusions_dict=qa_exclusions_dict
        )
        
        if outcome is not None and outcome.get('success', False):
            successful_results.append(outcome['result'])
            print(f"✓ Successfully processed {marker_name} ({marker_type})")
        else:
            error_msg = outcome.get('error', 'Unknown error') if outcome else 'Unknown error'
            failed_markers.append({
                'marker': f"{marker_name} ({marker_type})",
                'error': error_msg
            })
            print(f"✗ Failed to process {marker_name} ({marker_type})")
    
    # Step 4: Create summary report
    # Note: Multiple comparisons correction is applied separately via:
    #   - SLURM workflow: apply_mcc_postprocessing.py (automatic)
    #   - Manual: bash Statistics/apply_mcc_manual.sh <model_dir>
    print("\n" + "-"*80)
    print("STEP 4: Creating summary report")
    print("-"*80)
    
    # Create summary of all results with marker type information
    summary_data = []
    for result in successful_results:
        # Count significant clusters (corrected if available)
        # Note: Corrected values are added by apply_mcc_postprocessing.py
        if 'cluster_rejected' in result:
            n_sig_corrected = np.sum(result['cluster_rejected'])
        elif 'cluster_p_values_corrected' in result:
            alpha = result.get('correction_alpha', config['clustering']['alpha'])
            n_sig_corrected = np.sum(result['cluster_p_values_corrected'] <= alpha)
        else:
            # No correction applied yet
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
        
        # Build model folder name: fixed effects + active predictor of interest
        formula = config['lmm']['formula']
        model_folder_name = get_model_folder_name(formula, config['lmm']['predictor_of_interest'])
        
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
        model_folder_name = get_model_folder_name(formula, config['lmm']['predictor_of_interest'])
        base_output_dir = Path(config['project']['output_path'])
        model_output_dir = base_output_dir / model_folder_name
        
        if True:
            # Generate report with all topoplots and tables
            generate_summary_report(
                model_dir=model_output_dir,
                alpha=config['clustering']['alpha'],
                use_corrected=True,
                verbose=True
            )
        
        # Generate HTML QA report
        if True:
            successful_marker_names = [r.get('marker_name', 'unknown') for r in successful_results]
            html_path = generate_pipeline_qa_html_report(
                model_dir=model_output_dir,
                successful_markers=successful_marker_names,
                failed_markers=failed_markers,
                config=config,
                output_filename="pipeline_qa_summary.html"
            )
            print(f"✓ Pipeline QA HTML report saved to {html_path}")
    
    # Print completion message
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successfully processed: {len(successful_results)} markers")
    print(f"Failed markers: {len(failed_markers)}")
    if failed_markers:
        for fail_info in failed_markers:
            print(f"  - {fail_info['marker']}: {fail_info['error'][:100]}..." if len(fail_info['error']) > 100 else f"  - {fail_info['marker']}: {fail_info['error']}")
    
    # Show QA filtering summary if applied
    if qa_exclusions_dict:
        print("\nQA Filtering Summary:")
        for mtype, excl_set in qa_exclusions_dict.items():
            print(f"  {mtype}: {len(excl_set)} files excluded")
    
    # Show model-specific results directory
    formula = config['lmm']['formula']
    model_folder_name = get_model_folder_name(formula, config['lmm']['predictor_of_interest'])
    base_output_dir = Path(config['project']['output_path'])
    model_output_dir = base_output_dir / model_folder_name
    print(f"Model: {model_folder_name}")
    print(f"Results directory: {model_output_dir.absolute()}")
    
    if marker_index is None and successful_results:
        print("\n" + "="*80)
        print("SUMMARY REPORT FILES")
        print("="*80)
        print("Check the model directory for:")
        print("  - pipeline_qa_summary.html  : Pipeline execution status (HTML)")
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
    parser.add_argument(
        "--predictor-of-interest",
        type=str,
        default=None,
        dest="predictor_of_interest",
        help=(
            "Override lmm.predictor_of_interest from config.yaml. "
            "Required when the config value is a list (multi-predictor loop mode). "
            "Example: --predictor-of-interest valence"
        )
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        marker_index=args.marker_index,
        predictor_of_interest=args.predictor_of_interest
    )
