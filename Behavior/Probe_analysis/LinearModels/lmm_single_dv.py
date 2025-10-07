#!/usr/bin/env python
"""
Single DV LMM Analysis - for parallel execution across cluster nodes

This script analyzes a single dependent variable, allowing parallel execution
of different DVs on different nodes.

Usage:
    python lmm_single_dv.py --dv onoff --dataset full
    python lmm_single_dv.py --dv valence --dataset ie
"""

import argparse
import sys
import pickle
from pathlib import Path

# Import all necessary functions from main script
from lmm_beta_analysis_enhanced import (
    load_and_prepare_data,
    run_univariate_analysis,
    get_experimental_effects,
    run_complete_analysis,
    exhaustive_model_search,
    DEPENDENT_VARIABLES,
    OUTPUT_DIR
)


def main():
    parser = argparse.ArgumentParser(description='Run LMM analysis for a single DV')
    parser.add_argument('--dv', type=str, required=True, 
                       choices=DEPENDENT_VARIABLES,
                       help='Dependent variable to analyze')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['full', 'ie'],
                       help='Dataset type (full or ie)')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"SINGLE DV LMM ANALYSIS: {args.dv.upper()} ({args.dataset.upper()} dataset)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df_full, df_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie = load_and_prepare_data()
    
    # Select dataset
    if args.dataset == 'full':
        df = df_full
        available_cont_preds = available_cont_preds_full
    else:
        df = df_ie
        available_cont_preds = available_cont_preds_ie
    
    # Check if DV is available
    if args.dv not in available_dvs:
        print(f"ERROR: DV '{args.dv}' not available in dataset")
        sys.exit(1)
    
    dv = args.dv
    
    # Create output directory for this DV
    output_dir = Path(OUTPUT_DIR) / f"{args.dataset}_{dv}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Step 1: Univariate analysis
    print(f"\n{'='*60}")
    print(f"UNIVARIATE ANALYSIS: {dv}")
    print(f"{'='*60}")
    
    univariate_result = {}
    for pred in available_cont_preds:
        if f'{dv}_z' not in df.columns:
            print(f"Skipping - standardized version not available")
            break
            
        # Simplified univariate call for single predictor
        from lmm_beta_analysis_enhanced import fit_lmm_with_predictors
        model = fit_lmm_with_predictors(df, dv, [pred], args.dataset)
        
        if model['converged'] and pred in model['coefficients']:
            coef_info = model['coefficients'][pred]
            univariate_result[pred] = {
                'coefficients': {pred: coef_info},
                'bic': model['bic'],
                'converged': True
            }
            print(f"  {pred}: β = {coef_info['beta']:.3f}, p = {coef_info['p_value']:.3f}")
    
    results['univariate'] = univariate_result
    
    # Step 2: Experimental effects
    print(f"\n{'='*60}")
    print(f"EXPERIMENTAL EFFECTS: {dv}")
    print(f"{'='*60}")
    
    exp_effects = get_experimental_effects(df, [dv], args.dataset)
    results['experimental'] = exp_effects
    
    # Step 3: Complete analysis (all predictors)
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS: {dv}")
    print(f"{'='*60}")
    
    complete_model = fit_lmm_with_predictors(df, dv, available_cont_preds, args.dataset)
    results['complete'] = {
        'model': complete_model,
        'all_predictors': available_cont_preds
    }
    print(f"  Complete model: {len(available_cont_preds)} predictors, BIC = {complete_model['bic']:.2f}")
    
    # Step 4: Enhanced analysis (exhaustive search)
    print(f"\n{'='*60}")
    print(f"ENHANCED ANALYSIS (EXHAUSTIVE SEARCH): {dv}")
    print(f"{'='*60}")
    
    best_model_info, all_models_trajectory = exhaustive_model_search(
        df, dv, available_cont_preds, args.dataset, 
        use_parallel=True, max_predictors=None
    )
    
    results['enhanced'] = {
        'best_model': best_model_info['model'],
        'selected_predictors': best_model_info['predictors'],
        'all_predictors': available_cont_preds,
        'bic_trajectory': all_models_trajectory,
        'optimization_info': {
            'total_models_evaluated': len(all_models_trajectory),
            'n_predictors_selected': best_model_info['n_predictors'],
            'base_bic': all_models_trajectory[0]['bic'],
            'best_bic': best_model_info['bic'],
            'improvement': all_models_trajectory[0]['bic'] - best_model_info['bic']
        }
    }
    
    # Save results
    output_file = output_dir / f"results_{args.dataset}_{dv}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE FOR {dv}")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Selected predictors: {best_model_info['predictors']}")
    print(f"Final BIC: {best_model_info['bic']:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
