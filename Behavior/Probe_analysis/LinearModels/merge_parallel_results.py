#!/usr/bin/env python
"""
Merge results from parallel DV analyses and create final visualizations

Run after all parallel jobs complete to combine results and generate figures.

Usage:
    python merge_parallel_results.py
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Import visualization functions
from lmm_beta_analysis_enhanced import (
    create_univariate_visualization,
    create_enhanced_visualization,
    export_enhanced_results,
    export_experimental_effects,
    create_enhanced_report,
    DEPENDENT_VARIABLES,
    OUTPUT_DIR
)


def load_parallel_results():
    """
    Load all results from parallel DV analyses.
    
    Returns
    -------
    tuple
        (results_full, results_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie)
    """
    print("Loading parallel results...")
    
    output_dir = Path(OUTPUT_DIR)
    
    # Initialize result dictionaries
    univariate_full = {}
    univariate_ie = {}
    exp_effects_full = {}
    exp_effects_ie = {}
    complete_full = {}
    complete_ie = {}
    enhanced_full = {}
    enhanced_ie = {}
    
    available_dvs = []
    available_cont_preds_full = None
    available_cont_preds_ie = None
    
    # Load results for each DV and dataset
    for dv in DEPENDENT_VARIABLES:
        for dataset in ['full', 'ie']:
            result_file = output_dir / f"{dataset}_{dv}" / f"results_{dataset}_{dv}.pkl"
            
            if not result_file.exists():
                print(f"  Warning: Missing results for {dv} ({dataset})")
                continue
            
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            if dv not in available_dvs:
                available_dvs.append(dv)
            
            # Extract results by type
            if dataset == 'full':
                univariate_full[dv] = results.get('univariate', {})
                exp_effects_full.update(results.get('experimental', {}))
                complete_full[dv] = results.get('complete', {})
                enhanced_full[dv] = results.get('enhanced', {})
                
                if available_cont_preds_full is None and 'complete' in results:
                    available_cont_preds_full = results['complete'].get('all_predictors', [])
            else:
                univariate_ie[dv] = results.get('univariate', {})
                exp_effects_ie.update(results.get('experimental', {}))
                complete_ie[dv] = results.get('complete', {})
                enhanced_ie[dv] = results.get('enhanced', {})
                
                if available_cont_preds_ie is None and 'complete' in results:
                    available_cont_preds_ie = results['complete'].get('all_predictors', [])
    
    print(f"  Loaded results for {len(available_dvs)} DVs")
    print(f"  Available DVs: {available_dvs}")
    
    return (
        {'univariate': univariate_full, 'complete': complete_full, 'enhanced': enhanced_full},
        {'univariate': univariate_ie, 'complete': complete_ie, 'enhanced': enhanced_ie},
        exp_effects_full,
        exp_effects_ie,
        available_dvs,
        available_cont_preds_full,
        available_cont_preds_ie
    )


def main():
    print("="*80)
    print("MERGING PARALLEL LMM RESULTS")
    print("="*80)
    
    # Load all results
    (results_full, results_ie, exp_effects_full, exp_effects_ie, 
     available_dvs, available_cont_preds_full, available_cont_preds_ie) = load_parallel_results()
    
    # Export experimental effects
    print("\n" + "="*60)
    print("EXPORTING EXPERIMENTAL EFFECTS")
    print("="*60)
    export_experimental_effects(exp_effects_full, exp_effects_ie, available_dvs)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Univariate visualizations
    print("\nCreating univariate heatmaps...")
    create_univariate_visualization(
        results_full['univariate'], 
        results_ie['univariate'], 
        available_dvs,
        available_cont_preds_full, 
        available_cont_preds_ie
    )
    
    # Enhanced visualizations
    print("\nCreating enhanced heatmaps...")
    create_enhanced_visualization(
        results_full['enhanced'], 
        results_ie['enhanced'], 
        available_dvs,
        available_cont_preds_full, 
        available_cont_preds_ie
    )
    
    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    export_enhanced_results(
        results_full['complete'], 
        results_ie['complete'],
        results_full['enhanced'], 
        results_ie['enhanced'], 
        available_dvs
    )
    
    # Create report
    print("\n" + "="*60)
    print("CREATING REPORT")
    print("="*60)
    create_enhanced_report(
        results_full['complete'], 
        results_ie['complete'],
        results_full['enhanced'], 
        results_ie['enhanced'], 
        available_dvs
    )
    
    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - Univariate heatmaps (full & ie)")
    print("  - Enhanced heatmaps (full & ie)")
    print("  - Experimental effects tables")
    print("  - Model selection summaries")
    print("  - Comprehensive analysis report")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
