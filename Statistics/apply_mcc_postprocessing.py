"""
Post-processing script to apply multiple comparisons correction (MCC) after SLURM array jobs.

This script:
1. Loads all individual marker results from pickle files
2. Groups markers by type (evoked/state)
3. Applies FDR/Bonferroni correction across markers
4. Saves corrected cluster summaries
5. Generates correction summary report

Usage:
    python Statistics/apply_mcc_postprocessing.py <model_dir> [--config CONFIG_PATH]
    
Example:
    python Statistics/apply_mcc_postprocessing.py results/lmm_cluster/onoff
"""

import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from multiple_comparisons import correct_cluster_p_values, create_correction_summary
from helpers import summarize_clusters
from plot_results import plot_cluster_topomap


def load_marker_results(model_dir: Path, verbose: bool = True) -> List[Dict]:
    """
    Load all marker results from a model directory.
    
    Parameters
    ----------
    model_dir : Path
        Path to model directory containing marker subdirectories
    verbose : bool
        Print progress information
        
    Returns
    -------
    List[Dict]
        List of result dictionaries
    """
    results = []
    
    # Find all marker subdirectories
    marker_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    if verbose:
        print(f"Scanning {len(marker_dirs)} marker directories...")
    
    for marker_dir in marker_dirs:
        results_pkl = marker_dir / "results.pkl"
        
        if not results_pkl.exists():
            if verbose:
                print(f"  ⚠ Skipping {marker_dir.name}: no results.pkl found")
            continue
        
        try:
            with open(results_pkl, 'rb') as f:
                result = pickle.load(f)
            
            # Ensure marker_type is present
            if 'marker_type' not in result:
                # Try to infer from directory name
                dir_name = marker_dir.name
                if dir_name.startswith('evoked_'):
                    result['marker_type'] = 'evoked'
                elif dir_name.startswith('state_'):
                    result['marker_type'] = 'state'
                else:
                    if verbose:
                        print(f"  ⚠ Skipping {marker_dir.name}: cannot determine marker_type")
                    continue
            
            results.append(result)
            
            if verbose:
                marker_name = result.get('marker_name', 'unknown')
                marker_type = result.get('marker_type', 'unknown')
                n_clusters = result.get('n_clusters', 0)
                print(f"  ✓ Loaded {marker_name} ({marker_type}): {n_clusters} clusters")
                
        except Exception as e:
            if verbose:
                print(f"  ✗ Error loading {marker_dir.name}: {e}")
            continue
    
    if verbose:
        print(f"\n✓ Loaded {len(results)} marker results")
    
    return results


def save_corrected_summaries(results: List[Dict], model_dir: Path, 
                            mcc_alpha: float, verbose: bool = True) -> None:
    """
    Save corrected cluster summaries for each marker.
    
    Parameters
    ----------
    results : List[Dict]
        List of corrected result dictionaries
    model_dir : Path
        Path to model directory
    mcc_alpha : float
        Alpha level for significance
    verbose : bool
        Print progress information
    """
    if verbose:
        print("\nSaving corrected cluster summaries...")
    
    for result in results:
        if result.get('n_clusters', 0) == 0:
            continue
        
        marker_name = result['marker_name']
        marker_type = result.get('marker_type', 'unknown')
        
        # Reconstruct marker directory path
        safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
        marker_folder_name = f"{marker_type}_{safe_marker_name}"
        marker_dir = model_dir / marker_folder_name
        
        if not marker_dir.exists():
            if verbose:
                print(f"  ⚠ Directory not found: {marker_dir}")
            continue
        
        # Create corrected cluster summary
        cluster_summary_corrected = summarize_clusters(
            result['clusters'],
            result['cluster_stats'],
            result['cluster_p_values_corrected'],
            result['ch_names'],
            mcc_alpha
        )
        
        # Add metadata
        cluster_summary_corrected['marker_name'] = marker_name
        cluster_summary_corrected['marker_type'] = marker_type
        cluster_summary_corrected['correction_method'] = result.get('correction_method', 'none')
        cluster_summary_corrected['correction_alpha'] = result.get('correction_alpha', mcc_alpha)
        cluster_summary_corrected['analysis_timestamp'] = result.get('analysis_timestamp', datetime.now().isoformat())
        
        # Add both uncorrected and corrected p-values for comparison
        cluster_summary_corrected['p_value_uncorrected'] = result['cluster_p_values']
        cluster_summary_corrected['p_value_corrected'] = result['cluster_p_values_corrected']
        cluster_summary_corrected['significant_uncorrected'] = result['cluster_p_values'] <= mcc_alpha
        cluster_summary_corrected['significant_corrected'] = result['cluster_rejected']
        
        # Save corrected summary
        csv_path_corrected = marker_dir / "cluster_summary_corrected.csv"
        cluster_summary_corrected.to_csv(csv_path_corrected, index=False)
        
        # Regenerate topomap plot with both uncorrected and corrected p-values
        try:
            safe_marker_name = marker_name.replace('/', '_').replace(' ', '_')
            plot_title = f"Spatial Cluster Test - {marker_name} (MCC Applied)"
            topomap_path = marker_dir / f"results_{safe_marker_name}_topomap_corrected.png"
            
            plot_cluster_topomap(
                t_stats=result['t_stats'],
                clusters=result['clusters'],
                cluster_p_values=result['cluster_p_values_corrected'],  # Corrected as primary
                info=result['info'],
                alpha=mcc_alpha,
                title=plot_title,
                save_path=str(topomap_path),
                cluster_p_values_uncorrected=result['cluster_p_values']  # Show uncorrected as empty circles
            )
        except Exception as e:
            if verbose:
                print(f"  ⚠ Could not regenerate plot for {marker_name}: {e}")
        
        if verbose:
            n_sig_before = np.sum(result['cluster_p_values'] <= mcc_alpha)
            n_sig_after = np.sum(result['cluster_rejected'])
            print(f"  ✓ {marker_name}: {n_sig_before} → {n_sig_after} significant clusters")


def main():
    parser = argparse.ArgumentParser(
        description="Apply multiple comparisons correction to SLURM array job results"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory (e.g., results/lmm_cluster/onoff)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Statistics/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("MULTIPLE COMPARISONS CORRECTION POST-PROCESSING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Convert to Path
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"✗ Error: Model directory not found: {model_dir}")
        return 1
    
    print(f"Model directory: {model_dir}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"✗ Error: Config file not found: {config_path}")
        return 1
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get correction settings
    mcc_config = config.get('multiple_comparisons', {})

    # Determine alpha level in a config-agnostic way so this script can be
    # reused for both the original Statistics pipeline and the Andrillon
    # pipeline.
    if 'alpha' in mcc_config and mcc_config['alpha'] is not None:
        mcc_alpha = mcc_config['alpha']
    else:
        # Fallbacks: legacy "clustering.alpha" (Statistics) or
        # "andrillon_clustering.montecarlo_alpha" (Andrillon pipeline).
        if 'clustering' in config and 'alpha' in config['clustering']:
            mcc_alpha = config['clustering']['alpha']
        elif 'andrillon_clustering' in config and 'montecarlo_alpha' in config['andrillon_clustering']:
            mcc_alpha = config['andrillon_clustering']['montecarlo_alpha']
        else:
            # Sensible default if nothing is specified
            mcc_alpha = 0.05

    print(f"Configuration loaded from: {config_path}")
    print(f"Alpha level: {mcc_alpha}")
    
    # Load all marker results
    print("\n" + "-"*80)
    print("STEP 1: Loading marker results")
    print("-"*80)
    
    results = load_marker_results(model_dir, verbose=args.verbose)
    
    if len(results) == 0:
        print("✗ No results found to process")
        return 1
    
    # Apply correction separately for evoked and state markers
    print("\n" + "-"*80)
    print("STEP 2: Applying multiple comparisons correction")
    print("-"*80)
    
    corrected_results = []
    
    for marker_type in ['evoked', 'state']:
        # Get correction method for this marker type
        correction_method = mcc_config.get(marker_type, False)
        
        # Filter results for this marker type
        type_results = [r for r in results if r.get('marker_type') == marker_type]
        
        if len(type_results) == 0:
            print(f"\nNo {marker_type} markers to correct")
            continue
        
        # Check if any markers have clusters
        markers_with_clusters = [r for r in type_results if r.get('n_clusters', 0) > 0]
        if len(markers_with_clusters) == 0:
            print(f"\n{marker_type.upper()} markers:")
            print(f"  Number of markers: {len(type_results)}")
            print(f"  No clusters found in any marker - skipping correction")
            corrected_results.extend(type_results)
            continue
        
        print(f"\n{marker_type.upper()} markers:")
        print(f"  Number of markers: {len(type_results)}")
        print(f"  Markers with clusters: {len(markers_with_clusters)}")
        print(f"  Correction method: {correction_method}")
        
        # Apply correction
        type_corrected = correct_cluster_p_values(
            results_list=type_results,
            correction_method=correction_method,
            alpha=mcc_alpha,
            verbose=True
        )
        
        corrected_results.extend(type_corrected)
    
    # Save corrected summaries
    print("\n" + "-"*80)
    print("STEP 3: Saving corrected cluster summaries")
    print("-"*80)
    
    save_corrected_summaries(corrected_results, model_dir, mcc_alpha, verbose=args.verbose)
    
    # Save correction summary
    print("\n" + "-"*80)
    print("STEP 4: Creating correction summary report")
    print("-"*80)
    
    correction_summary_path = model_dir / "multiple_comparisons_summary.csv"
    correction_summary = create_correction_summary(
        corrected_results,
        output_path=str(correction_summary_path)
    )
    
    print("\n✓ Correction summary:")
    print(correction_summary.to_string(index=False))
    
    # Update pipeline summary if it exists
    pipeline_summary_path = model_dir / "pipeline_summary.csv"
    if pipeline_summary_path.exists():
        print("\n" + "-"*80)
        print("STEP 5: Updating pipeline summary")
        print("-"*80)
        
        try:
            summary_df = pd.read_csv(pipeline_summary_path)
            
            # Update with corrected values
            for result in corrected_results:
                marker_name = result['marker_name']
                marker_type = result.get('marker_type', 'unknown')
                
                # Find matching row
                mask = (summary_df['marker_name'] == marker_name) & \
                       (summary_df['marker_type'] == marker_type)
                
                if mask.any():
                    # Update corrected counts
                    if 'cluster_rejected' in result:
                        n_sig_corrected = np.sum(result['cluster_rejected'])
                    else:
                        n_sig_corrected = result.get('n_sig_clusters', 0)
                    
                    summary_df.loc[mask, 'n_sig_clusters_corrected'] = n_sig_corrected
                    summary_df.loc[mask, 'correction_method'] = result.get('correction_method', 'none')
            
            # Save updated summary
            summary_df.to_csv(pipeline_summary_path, index=False)
            print(f"✓ Updated pipeline summary: {pipeline_summary_path}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not update pipeline summary: {e}")
    
    # Print completion message
    print("\n" + "="*80)
    print("CORRECTION POST-PROCESSING COMPLETED")
    print("="*80)
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processed: {len(corrected_results)} markers")
    print(f"Results saved to: {model_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())
