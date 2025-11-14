"""
Andrillon 2020 Pipeline - Main Workflow

Complete pipeline implementing Andrillon et al. (2020) cluster-permutation methodology.

Workflow:
1. Load configuration
2. Load marker data (reuse reader.py from current pipeline)
3. Preprocess data (normalization)
4. For each marker:
   a. Fit LMM per electrode with permutations
   b. Detect clusters
   c. Apply multiple comparisons correction
5. Save results
6. Generate visualizations
"""

import sys
import os
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import sparse

 # Add Statistics directory to path to import existing modules
 stats_dir = Path(__file__).parent.parent / "Statistics"
 sys.path.insert(0, str(stats_dir))

 # Import ALL functions from the proven Statistics pipeline
 from reader import load_all_probe_data, prepare_data_for_lmm, get_available_markers
 from cluster_test import get_channel_adjacency, find_clusters
 from lmm_model import run_lmm_per_channel
 # from visualization import plot_cluster_results  # For later use - module not yet created

 # Import from local Stats_andrillon modules
 from cluster_detection import apply_bonferroni_correction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_marker_list(config: Dict) -> List[str]:
    """
    Get list of markers to analyze based on config.
    
    Returns list of marker names based on config.project.markers setting.
    Uses get_available_markers from Statistics/reader.py (proven implementation).
    """
    markers_setting = config['project']['markers']
    features_root = config['project']['features_root']
    subjects = config['project']['subjects']
    tasks = config['project']['tasks']
    marker_types = config['project']['marker_types']
    
    if markers_setting == 'all':
        # Auto-discover all available markers using proven function
        markers = get_available_markers(
            features_root=features_root,
            marker_types=marker_types
        )
        
        # Format as "marker_type/marker_name" for consistency
        formatted_markers = []
        for marker_type, marker_names in markers.items():
            for marker_name in marker_names:
                formatted_markers.append(f"{marker_type}/{marker_name}")
        
        logger.info(f"Auto-discovered {len(formatted_markers)} markers across {len(markers)} types")
        return formatted_markers
    else:
        # Use specified markers from config
        return markers_setting


def run_marker_analysis(
    marker_name: str,
    config: Dict,
    montage_path: str,
) -> Dict:
    """
    Run complete analysis for a single marker.
    
    Parameters
    ----------
    marker_name : str
        Name of the marker to analyze
    config : dict
        Configuration dictionary
    montage_path : str
        Path to montage file for building adjacency matrix
        
    Returns
    -------
    results : dict
        Analysis results including clusters and statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing marker: {marker_name}")
    logger.info(f"{'='*80}\n")
    
    # 1. Load data
    logger.info("Step 1: Loading marker data...")
    try:
        # Parse marker_name to extract type and name
        # Format: "marker_type/marker_name" (e.g., "evoked/wsmi_theta")
        if '/' in marker_name:
            marker_type, marker_base_name = marker_name.split('/', 1)
        else:
            # Fallback: assume no type prefix
            marker_type = None
            marker_base_name = marker_name
        
        logger.info(f"  Marker type: {marker_type}")
        logger.info(f"  Marker name: {marker_base_name}")
        
        # Load all probe data
        df_all = load_all_probe_data(
            features_root=config['project']['features_root'],
            subjects=config['project'].get('subjects'),
            tasks=config['project'].get('tasks'),
            marker_types=config['project']['marker_types'],
        )
        
        # Filter by marker type if specified
        if marker_type:
            logger.info(f"  Filtering data for marker_type: {marker_type}")
            df_filtered = df_all[df_all['marker_type'] == marker_type]
            logger.info(f"  Filtered from {len(df_all)} to {len(df_filtered)} rows")
        else:
            df_filtered = df_all
        
        # Prepare data for this specific marker (using base name without type prefix)
        power_data, df_behavioral, channels = prepare_data_for_lmm(
            df=df_filtered,
            marker_name=marker_base_name,
            formula=config['lmm']['formula'],
            include_channels=config['preprocessing'].get('include_channels'),
            exclude_channels=config['preprocessing'].get('exclude_channels'),
            onoff_max_value=config['preprocessing'].get('onoff_max_value'),
            min_predictor_variability=config['preprocessing'].get('min_predictor_variability'),
            predictor_of_interest=config['lmm']['predictor_of_interest'],
        )
        
        # Ensure subject is string type (critical for statsmodels mixedlm)
        df_behavioral['subject'] = df_behavioral['subject'].astype(str)
        
        logger.info(f"  Loaded {power_data.shape[0]} observations × {power_data.shape[1]} channels")
    except Exception as e:
        logger.error(f"Failed to load marker data: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. Fit LMM per electrode (using proven implementation from Statistics/)
    print("Step 2: Fitting LMM per electrode...", flush=True)
    logger.info("Step 2: Fitting LMM per electrode...")
    formula = config['lmm']['formula']
    predictor = config['lmm']['predictor_of_interest']
    
    print(f"  Formula: {formula}", flush=True)
    print(f"  Predictor: {predictor}", flush=True)
    print(f"  Channels: {len(channels)}", flush=True)
    logger.info(f"  Formula: {formula}")
    logger.info(f"  Predictor: {predictor}")
    logger.info(f"  Channels: {len(channels)}")
    
    # Use the proven run_lmm_per_channel from the previous pipeline
    t_stats, p_values, diagnostics = run_lmm_per_channel(
        power_data=power_data,
        df_behavioral=df_behavioral,
        formula=formula,
        predictor_of_interest=predictor,
        method=config['lmm']['method'],
        maxiter=config['lmm']['maxiter'],
        random_state=config['andrillon_clustering']['seed'],
        return_diagnostics=True
    )
    
    print(f"  LMM fitting complete: {diagnostics['n_converged']}/{len(channels)} channels converged", flush=True)
    logger.info(f"  LMM fitting complete: {diagnostics['n_converged']}/{len(channels)} channels converged")
    
    # 3. Generate permutations (Andrillon 2020 method)
    print("Step 3: Generating permutations...", flush=True)
    logger.info("Step 3: Generating permutations...")
    n_permutations = config['andrillon_clustering']['n_permutations']
    print(f"  Running {n_permutations} permutations", flush=True)
    logger.info(f"  Running {n_permutations} permutations")
    
    # Store permutation t-stats
    perm_t_stats = np.zeros((n_permutations, len(channels)))
    permutation_within = config['andrillon_clustering']['permutation_within']
    
    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 100 == 0 or perm_idx == 0:
            print(f"  Permutation {perm_idx+1}/{n_permutations}", flush=True)
        
        # Permute predictor within subject × task (Andrillon 2020 strategy)
        df_perm = df_behavioral.copy()
        groups = df_perm.groupby(permutation_within).groups
        for group_key, indices in groups.items():
            predictor_values = df_perm.loc[indices, predictor].values
            permuted_values = np.random.permutation(predictor_values)
            df_perm.loc[indices, predictor] = permuted_values
        
        # Fit LMM with permuted data
        perm_t, _, _ = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_perm,
            formula=formula,
            predictor_of_interest=predictor,
            method=config['lmm']['method'],
            maxiter=config['lmm']['maxiter'],
            random_state=config['andrillon_clustering']['seed'] + perm_idx,
            return_diagnostics=False
        )
        
        perm_t_stats[perm_idx, :] = perm_t
    
    print(f"  Permutations complete", flush=True)
    logger.info(f"  Permutations complete")
    
    # 4. Build adjacency matrix for these channels
    print("Step 4: Building adjacency matrix...", flush=True)
    logger.info("Step 4: Building adjacency matrix...")
    adjacency, _, _ = get_channel_adjacency(montage_path, channels)
    logger.info(f"  Adjacency matrix shape: {adjacency.shape}")
    
    # 5. Detect clusters (using proven find_clusters from Statistics/)
    print("Step 5: Detecting clusters...", flush=True)
    logger.info("Step 5: Detecting clusters...")
    clusters = find_clusters(
        t_stats=t_stats,
        perm_t_stats=perm_t_stats,
        adjacency=adjacency,
        threshold=config['andrillon_clustering']['cluster_alpha'],
        tail=1,  # One-tailed test (positive effects)
        n_permutations=n_permutations,
    )
    
    print(f"  Found {len(clusters)} significant clusters", flush=True)
    logger.info(f"  Found {len(clusters)} significant clusters")
    
    # 6. Package results
    results = {
        'marker_name': marker_name,
        'clusters': clusters,
        'real_t_stats': t_stats,
        'perm_t_stats': perm_t_stats,
        'real_p_values': p_values,
        'config': config,
        'n_observations': power_data.shape[0],
        'n_electrodes': len(channels),
        'channels': channels,
        'diagnostics': diagnostics,
    }
    
    return results


 def save_results(results: Dict, output_dir: Path, marker_name: str):
     """Save analysis results.

     Uses a filesystem-safe marker name so that path separators in
     marker identifiers (e.g., "state/wsmi_theta") do not create
     unintended subdirectories.
     """
     output_dir.mkdir(parents=True, exist_ok=True)

     # Create a safe filename component from the marker name
     # Examples:
     #   "state/wsmi_theta" -> "state_wsmi_theta"
     #   "evoked/wsmi_theta" -> "evoked_wsmi_theta"
     safe_marker_name = marker_name.replace('/', '_')

     # Save pickle
     pickle_path = output_dir / f"{safe_marker_name}_results.pkl"
     logger.info(f"Saving results to {pickle_path}")
     with open(pickle_path, 'wb') as f:
         pickle.dump(results, f)

     # Save CSV summary
     if results['clusters']:
         csv_path = output_dir / f"{safe_marker_name}_clusters.csv"
         logger.info(f"Saving cluster summary to {csv_path}")

         cluster_data = []
         for i, cluster in enumerate(results['clusters']):
             cluster_data.append({
                 'cluster_id': i,
                 'cluster_type': cluster.cluster_type,
                 'n_electrodes': len(cluster.electrodes),
                 'electrodes': ','.join(map(str, cluster.electrodes)),
                 'cluster_stat': cluster.cluster_stat,
                 'p_value': cluster.p_value,
             })

         df = pd.DataFrame(cluster_data)
         df.to_csv(csv_path, index=False)
     else:
         logger.info("No significant clusters found - skipping CSV")


def run_andrillon_pipeline(
    config_path: str,
    marker_name: Optional[str] = None,
):
    """
    Run complete Andrillon 2020 analysis pipeline.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    marker_name : str, optional
        If provided, analyze only this marker.
        If None, analyze all markers specified in config.
    """
    print("\n" + "="*80, flush=True)
    print("ANDRILLON 2020 CLUSTER-PERMUTATION PIPELINE", flush=True)
    print("="*80, flush=True)
    logger.info("="*80)
    logger.info("ANDRILLON 2020 CLUSTER-PERMUTATION PIPELINE")
    logger.info("="*80)
    
    # Load configuration
    print(f"Loading configuration from {config_path}", flush=True)
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Get adjacency matrix
    logger.info("Loading electrode adjacency matrix...")
    montage_path = Path(config['project']['montage_path'])
    
    # We'll get channel names from the first marker to build adjacency
    # This is a placeholder - adjacency will be built per marker with actual channels
    adjacency = None  # Will be built per marker
    logger.info("  Adjacency matrix will be built per marker based on available channels")
    
    # Get markers to analyze
    if marker_name:
        markers = [marker_name]
        logger.info(f"Analyzing single marker: {marker_name}")
    else:
        markers = get_marker_list(config)
        logger.info(f"Analyzing {len(markers)} markers")
    
    # Create output directory
    output_root = Path(config['project']['output_path'])
    
    # Determine model folder name from formula
    formula = config['lmm']['formula']
    # Extract fixed effects from formula (simple heuristic)
    fixed_effects = formula.split('~')[1].split('(')[0].strip()
    predictors = [p.strip() for p in fixed_effects.split('+')]
    model_folder = '_'.join(predictors)
    
    # Run analysis for each marker
    all_results = {}
    for i, marker in enumerate(markers):
        print(f"\n{'#'*80}", flush=True)
        print(f"MARKER {i+1}/{len(markers)}: {marker}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        logger.info(f"\n{'#'*80}")
        logger.info(f"MARKER {i+1}/{len(markers)}: {marker}")
        logger.info(f"{'#'*80}\n")
        
        try:
            # Run analysis
            results = run_marker_analysis(marker, config, str(montage_path))
            
            if results is not None:
                all_results[marker] = results
                
                # Save results
                output_dir = output_root / model_folder / marker
                print(f"Saving results to: {output_dir}", flush=True)
                logger.info(f"Saving results to: {output_dir}")
                save_results(results, output_dir, marker)
                print(f"✓ Results saved successfully", flush=True)
            else:
                print(f"⚠ Skipping marker {marker} due to errors", flush=True)
                logger.warning(f"Skipping marker {marker} due to errors")
                
        except Exception as e:
            logger.error(f"Error analyzing marker {marker}: {e}", exc_info=True)
            continue
    
    # Apply multiple comparisons correction if needed
    if len(all_results) > 1 and config['andrillon_clustering']['bonferroni_correction']:
        print("\n" + "="*80, flush=True)
        print("Applying Bonferroni correction across markers...", flush=True)
        print("="*80, flush=True)
        logger.info("\n" + "="*80)
        logger.info("Applying Bonferroni correction across markers...")
        logger.info("="*80)
        
        n_comparisons = len(all_results)
        logger.info(f"Number of comparisons: {n_comparisons}")
        
        for marker, results in all_results.items():
            if results['clusters']:
                logger.info(f"\nMarker: {marker}")
                logger.info(f"  Before correction: {len(results['clusters'])} clusters")
                
                corrected_clusters = apply_bonferroni_correction(
                    results['clusters'],
                    n_comparisons=n_comparisons,
                )
                
                logger.info(f"  After correction: {len(corrected_clusters)} clusters")
                
                # Update results
                results['clusters'] = corrected_clusters
                results['bonferroni_corrected'] = True
                results['n_comparisons'] = n_comparisons
                
                # Re-save with correction
                output_dir = output_root / model_folder / marker
                save_results(results, output_dir, marker)

    # Final summary
    print("\n" + "="*80, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Analyzed {len(all_results)} markers successfully", flush=True)

    total_clusters = sum(len(r['clusters']) for r in all_results.values())
    print(f"Total significant clusters found: {total_clusters}", flush=True)

    print(f"\nResults saved to: {output_root / model_folder}", flush=True)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Analyzed {len(all_results)} markers successfully")

    logger.info(f"Total significant clusters found: {total_clusters}")

    logger.info(f"\nResults saved to: {output_root / model_folder}")

    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Andrillon 2020 cluster-permutation pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--marker",
        type=str,
        default=None,
        help="Specific marker to analyze (optional, analyzes all if not provided)"
    )
    
    args = parser.parse_args()
    
    run_andrillon_pipeline(args.config, args.marker)
