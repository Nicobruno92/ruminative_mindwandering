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
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import mne
from scipy import sparse
from scipy import stats as sp_stats

# Add Statistics directory to path to import existing modules
stats_dir = Path(__file__).parent.parent / "Statistics"
sys.path.insert(0, str(stats_dir))

# Import ALL functions from the proven Statistics pipeline
from reader import load_all_probe_data, prepare_data_for_lmm, get_available_markers
from cluster_test import get_channel_adjacency, find_clusters_from_pvalues
from lmm_model import run_lmm_per_channel
from helpers import get_model_folder_name

# Import from local Stats_andrillon modules
from plot_results import create_results_report, create_raw_topography_report
from generate_summary_report import generate_summary_report

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
                # Exclude pair-wise matrix markers to avoid memory & channel count issues
                if marker_name.endswith('_pairs'):
                    continue
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
    marker_types_to_load = config['project']['marker_types']
    if marker_type is not None:
        marker_types_to_load = [marker_type]
    df_all = load_all_probe_data(
        features_root=config['project']['features_root'],
        subjects=config['project'].get('subjects'),
        tasks=config['project'].get('tasks'),
        marker_types=marker_types_to_load,
        specific_markers=[marker_base_name],
    )

    # Filter by marker type if specified
    if marker_type:
        logger.info(f"  Filtering data for marker_type: {marker_type}")
        df_filtered = df_all[df_all['marker_type'] == marker_type]
        logger.info(f"  Filtered from {len(df_all)} to {len(df_filtered)} rows")
    else:
        df_filtered = df_all

    # Prepare data for this specific marker (using base name without type prefix)
    preprocessing_cfg = config.get('preprocessing', {})
    project_cfg = config.get('project', {})
    onoff_max_value = preprocessing_cfg.get('onoff_max_value', project_cfg.get('onoff_max_value'))
    min_predictor_variability = preprocessing_cfg.get('min_predictor_variability', project_cfg.get('min_predictor_variability'))

    power_data, df_behavioral, channels = prepare_data_for_lmm(
        df=df_filtered,
        marker_name=marker_base_name,
        formula=config['lmm']['formula'],
        include_channels=preprocessing_cfg.get('include_channels'),
        exclude_channels=preprocessing_cfg.get('exclude_channels'),
        onoff_max_value=onoff_max_value,
        min_predictor_variability=min_predictor_variability,
        predictor_of_interest=config['lmm']['predictor_of_interest'],
    )

    # Ensure subject is string type (critical for statsmodels mixedlm)
    df_behavioral['subject'] = df_behavioral['subject'].astype(str)

    logger.info(f"  Loaded {power_data.shape[0]} observations × {power_data.shape[1]} channels")

    # Create output directory consistent with main Statistics pipeline
    output_root = Path(config['project']['output_path'])

    predictor = config['lmm'].get('predictor_of_interest', 'auto')
    model_folder = get_model_folder_name(config['lmm']['formula'], predictor)

    if marker_type:
        safe_marker_name = marker_base_name.replace('/', '_').replace(' ', '_')
        marker_folder = f"{marker_type}_{safe_marker_name}"
    else:
        marker_folder = marker_base_name.replace('/', '_').replace(' ', '_')
    output_dir = output_root / model_folder / marker_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Info object aligned to the original channel order for raw topographies
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)
    info_raw = mne.create_info(ch_names=list(channels), sfreq=250, ch_types='eeg')
    info_raw.set_montage(montage, on_missing='ignore')

    # Generate raw topography report before statistical analysis
    create_raw_topography_report(
        power_data=power_data,
        df_behavioral=df_behavioral,
        ch_names=channels,
        info=info_raw,
        marker_name=marker_name,
        output_dir=str(output_dir),
        subject_col='subject',
        predictor_of_interest=config['lmm']['predictor_of_interest'],
    )
    
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
    
    # Store permutation t- and p-stats.
    # NaN (not zero/one) so that failed-channel entries are excluded from
    # cluster formation in the null distribution, matching the observed data.
    perm_t_stats = np.full((n_permutations, len(channels)), np.nan)
    perm_p_values = np.full((n_permutations, len(channels)), np.nan)
    permutation_within = config['andrillon_clustering']['permutation_within']

    # Use the same LMM method and maxiter for permuted fits as for the observed
    # fit — different optimizers produce systematically different t-statistics and
    # convergence rates, which creates an asymmetric null distribution and
    # inflates false positives.
    lmm_method = config['lmm']['method']
    lmm_maxiter = config['lmm']['maxiter']
    base_seed = config['andrillon_clustering']['seed']

    for perm_idx in range(n_permutations):
        if (perm_idx + 1) % 100 == 0 or perm_idx == 0:
            print(f"  Permutation {perm_idx+1}/{n_permutations}", flush=True)

        # Deterministic per-permutation RNG so results are fully reproducible.
        perm_rng = np.random.default_rng(base_seed + perm_idx)

        # Permute predictor within subject × task (Andrillon 2020 strategy)
        df_perm = df_behavioral.copy()
        groups = df_perm.groupby(permutation_within).groups
        for group_key, indices in groups.items():
            predictor_values = df_perm.loc[indices, predictor].values
            permuted_values = perm_rng.permutation(predictor_values)
            df_perm.loc[indices, predictor] = permuted_values

        # Fit LMM with permuted data — identical settings to the observed fit
        perm_t, perm_p, _ = run_lmm_per_channel(
            power_data=power_data,
            df_behavioral=df_perm,
            formula=formula,
            predictor_of_interest=predictor,
            method=lmm_method,
            maxiter=lmm_maxiter,
            random_state=base_seed + perm_idx,
            return_diagnostics=False
        )
        
        perm_t_stats[perm_idx, :] = perm_t
        perm_p_values[perm_idx, :] = perm_p
    
    print(f"  Permutations complete", flush=True)
    logger.info(f"  Permutations complete")
    
    # 4. Build adjacency matrix for these channels
    print("Step 4: Building adjacency matrix...", flush=True)
    logger.info("Step 4: Building adjacency matrix...")
    adjacency, ch_names_ordered, channel_indices = get_channel_adjacency(montage_path, channels)
    logger.info(f"  Adjacency matrix shape: {adjacency.shape}")

    # Reorder statistics and channel list to match adjacency / montage order.
    # This mirrors the main Statistics pipeline to ensure spatial alignment.
    t_stats = t_stats[channel_indices]
    p_values = p_values[channel_indices]
    perm_t_stats = perm_t_stats[:, channel_indices]
    perm_p_values = perm_p_values[:, channel_indices]
    channels = [channels[idx] for idx in channel_indices]

    # Build exclusion mask for clustering based on configuration.
    # Excluded channels are still tested but cannot form or connect clusters.
    channel_excl_cfg = config.get('channel_exclusion', {})
    exclude_mask = None
    if channel_excl_cfg.get('enabled', False):
        excluded_names = set(channel_excl_cfg.get('channel_names', []))
        exclude_mask = np.array([ch in excluded_names for ch in channels], dtype=bool)

    # Create MNE Info object for visualization using the canonical channel order
    # from the adjacency/montage, but restricted to the channels present here.
    if montage_path.endswith('.bvef'):
        montage = mne.channels.read_custom_montage(montage_path)
    else:
        montage = mne.channels.make_standard_montage(montage_path)

    info = mne.create_info(ch_names=list(channels), sfreq=250, ch_types='eeg')
    # Ignore missing positions but keep available ones for plotting
    info.set_montage(montage, on_missing='ignore')
    
    # 5. Detect clusters (using proven find_clusters from Statistics/)
    print("Step 5: Detecting clusters...", flush=True)
    logger.info("Step 5: Detecting clusters...")

    cluster_alpha = float(config['andrillon_clustering']['cluster_alpha'])

    clusters_raw = find_clusters_from_pvalues(
        t_stats=t_stats,
        p_values=p_values,
        perm_t_stats=perm_t_stats,
        perm_p_values=perm_p_values,
        adjacency=adjacency,
        cluster_alpha=cluster_alpha,
        tail=0,
        n_permutations=n_permutations,
        stat_fun='sum',
        t_power=1.0,
        separate_signs=True,
        exclude=exclude_mask,
    )
    
    print(f"  Found {len(clusters_raw)} significant clusters", flush=True)
    logger.info(f"  Found {len(clusters_raw)} significant clusters")

    # Convert cluster dictionaries to arrays and summary statistics to match
    # the main Statistics pipeline expectations
    cluster_channels: List[np.ndarray] = []
    cluster_stats: List[float] = []
    cluster_p_values: List[float] = []
    for cluster in clusters_raw:
        if isinstance(cluster, dict):
            ch_idx = np.array(cluster.get('channels', []), dtype=int)
            stat_val = float(cluster.get('stat')) if cluster.get('stat') is not None else np.nan
            p_val = float(cluster.get('p_value')) if cluster.get('p_value') is not None else np.nan
        else:
            # Fallback for ClusterResult-like objects
            ch_idx = np.array(getattr(cluster, 'electrodes', []), dtype=int)
            stat_val = float(getattr(cluster, 'cluster_stat', np.nan))
            p_val = float(getattr(cluster, 'p_value', np.nan))

        cluster_channels.append(ch_idx)
        cluster_stats.append(stat_val)
        cluster_p_values.append(p_val)

    cluster_stats_arr = np.array(cluster_stats, dtype=float) if cluster_stats else np.array([], dtype=float)
    cluster_p_values_arr = np.array(cluster_p_values, dtype=float) if cluster_p_values else np.array([], dtype=float)

    # Compute number of significant clusters using Andrillon Monte Carlo alpha
    montecarlo_alpha = config['andrillon_clustering']['montecarlo_alpha']
    n_clusters = int(cluster_stats_arr.size)
    n_sig_clusters = int(np.sum(cluster_p_values_arr < montecarlo_alpha)) if n_clusters > 0 else 0

    # Derive an approximate t-threshold for visualization only.
    # We match the cluster_alpha proportion of the observed |t|-distribution.
    threshold_t = None
    if np.any(~np.isnan(t_stats)):
        t_abs = np.abs(t_stats[~np.isnan(t_stats)])
        q = max(0.0, min(1.0, 1.0 - float(config['andrillon_clustering']['cluster_alpha'])))
        threshold_t = float(np.quantile(t_abs, q)) if t_abs.size > 0 else None

    # 6. Package results in a structure compatible with the Statistics pipeline
    results = {
        # Core identifiers
        'marker_name': marker_name,
        'marker_type': marker_type,
        'marker_base_name': marker_base_name,

        # Channel-wise statistics
        't_stats': t_stats,
        'p_values': p_values,
        'real_t_stats': t_stats,        # Backwards compatibility
        'real_p_values': p_values,      # Backwards compatibility

        # Cluster information
        'clusters': cluster_channels,   # List[np.ndarray] for plotting/reporting
        'cluster_stats': cluster_stats_arr,
        'cluster_p_values': cluster_p_values_arr,
        'clusters_raw': clusters_raw,   # Original objects from find_clusters
        'n_clusters': n_clusters,
        'n_sig_clusters': n_sig_clusters,

        # Channel / montage info
        'ch_names': channels,
        'channels': channels,           # Backwards compatibility
        'info': info,

        # Data shape and counts
        'power_data_shape': power_data.shape,
        'n_subjects': df_behavioral['subject'].nunique(),
        'n_observations': power_data.shape[0],
        'n_electrodes': len(channels),

        # Model and clustering metadata
        'config': config,
        'formula': formula,
        'predictor_of_interest': predictor,
        'analysis_timestamp': datetime.now().isoformat(),
        'clustering_method': 'andrillon_permutation',
        'threshold': threshold_t,
        'alpha': montecarlo_alpha,
        'n_permutations': n_permutations,

        # Diagnostics from LMM fitting
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

    # Save pickle (Andrillon-specific filename)
    pickle_path = output_dir / f"{safe_marker_name}_results.pkl"
    logger.info(f"Saving results to {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Save generic results.pkl compatible with Statistics/generate_summary_report.py
    generic_pickle_path = output_dir / "results.pkl"
    logger.info(f"Saving generic results to {generic_pickle_path}")
    with open(generic_pickle_path, 'wb') as f:
        pickle.dump(results, f)

    # Save CSV summary using the numeric cluster representation
    clusters = results.get('clusters', [])
    cluster_stats = np.asarray(results.get('cluster_stats', []), dtype=float)
    cluster_p_values = np.asarray(results.get('cluster_p_values', []), dtype=float)

    if clusters and cluster_stats.size == len(clusters) and cluster_p_values.size == len(clusters):
        csv_path = output_dir / f"{safe_marker_name}_clusters.csv"
        logger.info(f"Saving cluster summary to {csv_path}")

        cluster_data = []
        for i, ch_idx in enumerate(clusters):
            # ch_idx is an array of channel indices into results['ch_names'] / 'channels'
            ch_idx = np.asarray(ch_idx, dtype=int)
            # Map indices to channel names when available
            ch_names = results.get('ch_names', results.get('channels', []))
            if ch_names:
                cluster_channels = [ch_names[j] for j in ch_idx]
            else:
                cluster_channels = ch_idx.tolist()

            stat_val = float(cluster_stats[i]) if i < cluster_stats.size else np.nan
            p_val = float(cluster_p_values[i]) if i < cluster_p_values.size else np.nan
            cluster_type = 'positive' if np.isfinite(stat_val) and stat_val >= 0 else 'negative'

            cluster_data.append({
                'cluster_id': i,
                'cluster_type': cluster_type,
                'n_electrodes': len(cluster_channels),
                'electrodes': ','.join(map(str, cluster_channels)),
                'cluster_stat': stat_val,
                'p_value': p_val,
            })

        df = pd.DataFrame(cluster_data)
        df.to_csv(csv_path, index=False)
    else:
        logger.info("No significant clusters found - skipping CSV")

    # Optionally generate figures using the shared plotting utilities
    config = results.get('config', {})
    output_cfg = config.get('output', {}) if isinstance(config, dict) else {}
    if output_cfg.get('save_figures', True):
        t_stats = results.get('t_stats')
        clusters = results.get('clusters', [])
        cluster_stats = results.get('cluster_stats')
        cluster_p_values = results.get('cluster_p_values')
        info = results.get('info')
        threshold = results.get('threshold')
        alpha = results.get('alpha', 0.05)

        if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
            threshold = 0.0

        if t_stats is not None and info is not None and cluster_stats is not None and cluster_p_values is not None:
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
                config=config if isinstance(config, dict) else None,
            )


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
    
    # Determine model folder name from formula using the same helper
    # as the main Statistics pipeline to ensure identical directory names
    formula = config['lmm']['formula']
    predictor = config['lmm'].get('predictor_of_interest', 'auto')
    model_folder = get_model_folder_name(formula, predictor)
    
    # Run analysis for each marker
    all_results = {}
    for i, marker in enumerate(markers):
        print(f"\n{'#'*80}", flush=True)
        print(f"MARKER {i+1}/{len(markers)}: {marker}", flush=True)
        print(f"{'#'*80}\n", flush=True)
        logger.info(f"\n{'#'*80}")
        logger.info(f"MARKER {i+1}/{len(markers)}: {marker}")
        logger.info(f"{'#'*80}\n")
        
        # Run analysis — errors surface immediately so root causes are visible
        results = run_marker_analysis(marker, config, str(montage_path))

        all_results[marker] = results

        # Derive marker-specific output directory consistent with
        # the main Statistics pipeline: <marker_type>_<marker_name>
        if '/' in marker:
            marker_type, marker_base_name = marker.split('/', 1)
        else:
            marker_type = None
            marker_base_name = marker

        safe_marker_name = marker_base_name.replace('/', '_').replace(' ', '_')
        if marker_type:
            marker_folder = f"{marker_type}_{safe_marker_name}"
        else:
            marker_folder = safe_marker_name

        output_dir = output_root / model_folder / marker_folder
        print(f"Saving results to: {output_dir}", flush=True)
        logger.info(f"Saving results to: {output_dir}")
        save_results(results, output_dir, marker)
        print(f"Results saved successfully", flush=True)
    
    # Final summary
    print("\n" + "="*80, flush=True)
    print("PIPELINE COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"Analyzed {len(all_results)} markers successfully", flush=True)

    total_clusters = 0
    for res in all_results.values():
        cpv = np.asarray(res.get('cluster_p_values', []), dtype=float)
        total_clusters += int(cpv.size)
    print(f"Total clusters detected (before MCC): {total_clusters}", flush=True)

    model_dir = output_root / model_folder
    print(f"\nResults saved to: {model_dir}", flush=True)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Analyzed {len(all_results)} markers successfully")
    logger.info(f"Total clusters detected (before MCC): {total_clusters}")
    logger.info(f"\nResults saved to: {model_dir}")

    # Generate summary report only when running the full set of markers
    if marker_name is None and len(all_results) > 0:
        print("\n" + "-"*80, flush=True)
        print("Generating Andrillon summary report (topoplots + tables)...", flush=True)
        print("-"*80, flush=True)
        generate_summary_report(
            model_dir=model_dir,
            alpha=config['andrillon_clustering']['montecarlo_alpha'],
            use_corrected=True,
            verbose=True,
        )

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
