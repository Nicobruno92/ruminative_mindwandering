# %%
"""
Enhanced Linear Mixed Model Beta Analysis with Variable Selection

This enhanced script includes:
1. Time-on-task variable (cumulative probe number across SART tasks)
2. Two separate analyses: IE-specific (Sart2/4) and full dataset (all tasks)
3. Recursive variable selection using BIC for optimal model selection
4. Enhanced visualization with diagonal lines for dropped variables

Features:
- No plot.show() calls to avoid execution stops
- Time-on-task variable creation
- BIC-based model selection for each DV
- Advanced visualization with model selection indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from statsmodels.regression.mixed_linear_model import MixedLM
import itertools
from sklearn.preprocessing import StandardScaler
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Data paths (resolve relative to project root regardless of CWD)
THIS_FILE = Path(__file__).resolve()
# Project root is two levels up from this file: Behavior/Probe_analysis/..
PROJECT_ROOT = THIS_FILE.parents[2]

PROBE_DATA_PATH = str(PROJECT_ROOT / "results/Behavior/probe_data/probe_level_aggregated_data.csv")
PCA_DATA_PATH = str(PROJECT_ROOT / "results/Behavior/probe_data/pca_results.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results/Behavior/probe_data/lmm_beta_analysis_enhanced")

# Variables for analysis
DEPENDENT_VARIABLES = ['onoff', 'valence', 'time', 'selfother', 'confidence', 'PC1', 'PC2', 'PC3']

# Experimental predictors (categorical) - for full analysis
EXPERIMENTAL_PREDICTORS_FULL = ['condition_simple', 'group', 'condition_simple_x_group']

# Experimental predictors (categorical) - for IE analysis  
EXPERIMENTAL_PREDICTORS_IE = ['condition', 'group', 'condition_x_group']

# Continuous predictors (psychometric scales)
CONTINUOUS_PREDICTORS = ['age', 'bdi', 'rrs_tot', 'mwq', 'fne', 'self_esteem', 'ctq_tot', 'a_rsq', 'sris', 'time_on_task']

# Enhanced labels for display
VARIABLE_LABELS = {
    'onoff': 'OnOff',
    'valence': 'Valence', 
    'time': 'Time',
    'selfother': 'Self/Other',
    'confidence': 'Confidence',
    'PC1': 'PC1',
    'PC2': 'PC2', 
    'PC3': 'PC3'
}

PREDICTOR_LABELS = {
    'condition': 'Condition',
    'condition_simple': 'Condition',
    'group': 'Group',
    'condition_x_group': 'Condition×Group',
    'condition_simple_x_group': 'Condition×Group',
    'age': 'Age',
    'bdi': 'BDI',
    'rrs_tot': 'RRS',
    'mwq': 'MWQ',
    'fne': 'FNE',
    'self_esteem': 'Self-Est.',
    'ctq_tot': 'CTQ',
    'a_rsq': 'ARSQ',
    'sris': 'SRIS',
    'time_on_task': 'Time-on-Task'
}

# Analysis settings
APPLY_ONOFF_FILTER = True
ONOFF_MAX_EXCLUSIVE = 100.0
USE_FDR_CORRECTION = True
SIGNIFICANCE_THRESHOLD = 0.05

# Plot settings
FIGURE_SIZE = (20, 14)
DPI = 300

# =============================================================================


def create_time_on_task_variable(df):
    """
    Create time-on-task variable: cumulative probe number across SART tasks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with task and probe_number columns
        
    Returns
    -------
    pd.DataFrame
        Dataframe with time_on_task variable added
    """
    print("Creating time-on-task variable...")
    
    # Define task order
    task_order = {'Sart1': 1, 'Sart2': 2, 'Sart3': 3, 'Sart4': 4}
    
    # Sort by subject, task order, and probe number
    df = df.copy()
    df['task_order'] = df['task'].map(task_order)
    df = df.sort_values(['subject_id', 'task_order', 'probe_number']).reset_index(drop=True)
    
    # Calculate cumulative probe number within each subject
    df['time_on_task'] = df.groupby('subject_id').cumcount() + 1
    
    print(f"Time-on-task variable created. Range: {df['time_on_task'].min()} to {df['time_on_task'].max()}")
    
    return df


def load_and_prepare_data():
    """
    Load and prepare data for enhanced LMM analysis.
    
    Returns
    -------
    tuple
        (df_full, df_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie)
    """
    print("Loading and preparing data for enhanced LMM analysis...")
    
    # Load datasets
    probe_df = pd.read_csv(PROBE_DATA_PATH)
    pca_df = pd.read_csv(PCA_DATA_PATH)
    
    # Apply onoff filter if specified
    if APPLY_ONOFF_FILTER:
        before_n = len(probe_df)
        probe_df = probe_df[probe_df['onoff'] <= ONOFF_MAX_EXCLUSIVE].copy()
        after_n = len(probe_df)
        print(f"Applied onoff filter (≤{ONOFF_MAX_EXCLUSIVE}): {before_n} -> {after_n} observations")
    
    # Rename column to avoid Python parsing issues
    if 'self-esteem' in probe_df.columns:
        probe_df = probe_df.rename(columns={'self-esteem': 'self_esteem'})
    
    # Merge with PCA data
    merge_cols = ['subject_id', 'task', 'probe_number']
    df = pd.merge(probe_df, pca_df[merge_cols + ['PC1', 'PC2', 'PC3']], 
                  on=merge_cols, how='left', suffixes=('', '_pca'))
    
    print(f"Total observations after merging: {len(df)}")
    
    # Create time-on-task variable
    df = create_time_on_task_variable(df)
    
    # Create condition variable
    condition_map = {
        'inclusion': 'inclusion',
        'exclusion': 'exclusion', 
        'baseline': 'baseline'
    }
    df['condition'] = df['inclusion_exclusion'].map(condition_map)
    df = df[df['condition'].notna()].copy()
    
    # Prepare full dataset (all tasks)
    df_full = df.copy()
    
    # For full dataset, create a simplified condition variable
    # baseline vs experimental (inclusion + exclusion)
    df_full['condition_simple'] = df_full['condition'].map({
        'baseline': 'baseline',
        'inclusion': 'experimental', 
        'exclusion': 'experimental'
    })
    
    # Prepare IE-specific dataset (only Sart2 and Sart4)
    df_ie = df[df['task'].isin(['Sart2', 'Sart4'])].copy()
    df_ie = df_ie[df_ie['condition'].isin(['inclusion', 'exclusion'])].copy()
    
    print(f"Full dataset: {len(df_full)} observations from {df_full['subject_id'].nunique()} subjects")
    print(f"IE dataset: {len(df_ie)} observations from {df_ie['subject_id'].nunique()} subjects")
    
    # Set categorical variables for both datasets
    for data in [df_full, df_ie]:
        data['group'] = pd.Categorical(data['group'], 
                                     categories=['Controls', 'Risk of Depression'])
        data['subject_id'] = data['subject_id'].astype(str)
    
    # Set condition variables
    df_full['condition_simple'] = pd.Categorical(df_full['condition_simple'],
                                               categories=['baseline', 'experimental'])
    df_ie['condition'] = pd.Categorical(df_ie['condition'],
                                      categories=['inclusion', 'exclusion'])
    
    # Create interaction terms
    df_full['condition_simple_x_group'] = df_full['condition_simple'].astype(str) + '_' + df_full['group'].astype(str)
    df_ie['condition_x_group'] = df_ie['condition'].astype(str) + '_' + df_ie['group'].astype(str)
    
    # Check available variables
    available_dvs = [var for var in DEPENDENT_VARIABLES if var in df_full.columns]
    
    # For continuous predictors, check availability
    available_cont_preds_full = [var for var in CONTINUOUS_PREDICTORS if var in df_full.columns]
    available_cont_preds_ie = [var for var in CONTINUOUS_PREDICTORS if var in df_ie.columns]
    
    print(f"Available DVs: {available_dvs}")
    print(f"Available continuous predictors (full): {available_cont_preds_full}")
    print(f"Available continuous predictors (IE): {available_cont_preds_ie}")
    
    # Standardize variables in both datasets
    df_full = standardize_variables(df_full, available_dvs, available_cont_preds_full)
    df_ie = standardize_variables(df_ie, available_dvs, available_cont_preds_ie)
    
    return df_full, df_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie


def standardize_variables(df, dvs, cont_preds):
    """
    Standardize variables in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    dvs : list
        Dependent variables to standardize
    cont_preds : list
        Continuous predictors to standardize
        
    Returns
    -------
    pd.DataFrame
        Dataset with standardized variables
    """
    df = df.copy()
    
    variables_to_standardize = []
    for var in dvs + cont_preds:
        if var in df.columns and df[var].dtype in ['float64', 'int64']:
            variables_to_standardize.append(var)
    
    print(f"Standardizing {len(variables_to_standardize)} variables")
    
    for var in variables_to_standardize:
        if df[var].std() > 0:
            df[f'{var}_z'] = (df[var] - df[var].mean()) / df[var].std()
        else:
            print(f"Warning: Variable {var} has no variance")
            df[f'{var}_z'] = df[var]
    
    # Remove rows with missing data in key variables
    key_vars = ['subject_id', 'group'] + [f'{var}_z' for var in variables_to_standardize]
    before_clean = len(df)
    df = df.dropna(subset=key_vars)
    after_clean = len(df)
    
    if before_clean != after_clean:
        print(f"Removed {before_clean - after_clean} rows with missing data")
    
    return df


def fit_lmm_with_predictors(df, dv, predictors, dataset_type='full'):
    """
    Fit LMM with specific set of predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    dv : str
        Dependent variable
    predictors : list
        List of predictor variables
    dataset_type : str
        'full' or 'ie' to determine experimental predictors
        
    Returns
    -------
    dict
        Model results including BIC, coefficients, etc.
    """
    try:
        # Build formula with additive effects + interaction
        if dataset_type == 'full':
            # For full dataset: use condition_simple and group with interaction
            exp_formula = ("C(group, Treatment(reference='Controls'))")
        else:
            # For IE dataset: use condition (inclusion/exclusion) and group with interaction
            exp_formula = ("C(condition, Treatment(reference='inclusion')) + "
                          "C(group, Treatment(reference='Controls')) + "
                          "C(condition, Treatment(reference='inclusion')):"
                          "C(group, Treatment(reference='Controls'))")
        
        # Add continuous predictors
        cont_predictors = [f"{pred}_z" for pred in predictors if f"{pred}_z" in df.columns]
        if cont_predictors:
            formula = f"{dv}_z ~ {exp_formula} + {' + '.join(cont_predictors)}"
        else:
            formula = f"{dv}_z ~ {exp_formula}"
        
        # Fit model
        model = smf.mixedlm(formula, data=df, groups=df['subject_id'])
        result = model.fit(reml=False)
        
        # Extract coefficients for continuous predictors only
        params = result.params
        pvalues = result.pvalues
        std_errors = result.bse
        
        coefs = {}
        for pred in predictors:
            pred_z = f"{pred}_z"
            if pred_z in params.index:
                coefs[pred] = {
                    'beta': params[pred_z],
                    'se': std_errors[pred_z],
                    'p_value': pvalues[pred_z]
                }
        
        # Extract experimental effects (group, condition, interactions)
        for param_name in params.index:
            if param_name != 'Intercept' and param_name not in coefs:
                # Check if it's an experimental effect (group, condition, or interaction)
                if 'group' in param_name.lower() or 'condition' in param_name.lower() or ':' in param_name:
                    coefs[param_name] = {
                        'beta': params[param_name],
                        'se': std_errors[param_name],
                        'p_value': pvalues[param_name]
                    }
        
        return {
            'converged': result.converged,
            'bic': result.bic,
            'aic': result.aic,
            'loglik': result.llf,
            'formula': formula,
            'n_obs': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'coefficients': coefs,
            'n_predictors': len(predictors)
        }
        
    except Exception as e:
        return {
            'converged': False,
            'bic': np.inf,
            'aic': np.inf,
            'loglik': -np.inf,
            'formula': None,
            'n_obs': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'coefficients': {},
            'n_predictors': len(predictors),
            'error': str(e)
        }


def evaluate_single_predictor(args):
    """
    Helper function for parallel evaluation of a single predictor.
    
    Parameters
    ----------
    args : tuple
        (df, dv, test_predictors, dataset_type)
        
    Returns
    -------
    dict
        Model results with predictor info
    """
    df, dv, test_predictors, dataset_type, pred = args
    
    try:
        test_model = fit_lmm_with_predictors(df, dv, test_predictors, dataset_type)
        return {
            'pred': pred,
            'bic': test_model['bic'],
            'model': test_model,
            'converged': test_model['converged']
        }
    except Exception as e:
        return {
            'pred': pred,
            'bic': np.inf,
            'model': None,
            'converged': False,
            'error': str(e)
        }


def evaluate_predictor_combo(args):
    """
    Top-level helper for multiprocessing: evaluate a predictor combination.
    This avoids using lambdas/closures which aren't picklable on some systems.
    
    Parameters
    ----------
    args : tuple
        (df, dv, pred_list, dataset_type)
    
    Returns
    -------
    dict
        Result of fit_lmm_with_predictors
    """
    df, dv, pred_list, dataset_type = args
    return fit_lmm_with_predictors(df, dv, pred_list, dataset_type)


def exhaustive_model_search(df, dv, available_predictors, dataset_type='full', use_parallel=True, max_predictors=None):
    """
    Exhaustive search: tries ALL possible combinations of predictors.
    With n predictors, evaluates 2^n models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    dv : str
        Dependent variable
    available_predictors : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
    use_parallel : bool
        Use parallel processing
    max_predictors : int, optional
        Maximum number of predictors to include (for computational efficiency)
        If None, tries all combinations up to len(available_predictors)
        
    Returns
    -------
    tuple
        (best_model_info, all_models_evaluated)
    """
    from itertools import combinations
    
    print(f"  Exhaustive search for {dv} ({dataset_type})...")
    
    if max_predictors is None:
        max_predictors = len(available_predictors)
    
    print(f"    Evaluating all combinations up to {max_predictors} predictors")
    
    # Calculate total number of models
    total_models = sum(len(list(combinations(available_predictors, k))) 
                      for k in range(max_predictors + 1))
    print(f"    Total models to evaluate: {total_models}")
    # Rough estimate: ~0.5 seconds per model
    estimated_minutes = (total_models * 0.5) / 60
    print(f"    Estimated time: ~{estimated_minutes:.1f} minutes")
    
    all_models = []
    best_bic = np.inf
    best_model_info = None
    
    # Base model (0 predictors)
    base_model = fit_lmm_with_predictors(df, dv, [], dataset_type)
    model_info = {
        'predictors': [],
        'n_predictors': 0,
        'bic': base_model['bic'],
        'model': base_model
    }
    all_models.append(model_info)
    if base_model['bic'] < best_bic:
        best_bic = base_model['bic']
        best_model_info = model_info
    print(f"    Base model (0 predictors): BIC = {base_model['bic']:.2f}")
    
    # Evaluate all combinations by size
    for n_preds in range(1, max_predictors + 1):
        pred_combinations = list(combinations(available_predictors, n_preds))
        n_combos = len(pred_combinations)
        
        print(f"    Evaluating {n_combos} models with {n_preds} predictor(s)...")
        
        if use_parallel and n_combos > 1:
            # Parallel evaluation
            n_cores = min(mp.cpu_count() - 1, n_combos)
            args_list = [(df, dv, list(pred_combo), dataset_type) 
                        for pred_combo in pred_combinations]
            
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                results = list(executor.map(evaluate_predictor_combo, args_list))
            
            # Process results
            for pred_combo, result in zip(pred_combinations, results):
                if result['converged']:
                    model_info = {
                        'predictors': list(pred_combo),
                        'n_predictors': n_preds,
                        'bic': result['bic'],
                        'model': result
                    }
                    all_models.append(model_info)
                    
                    if result['bic'] < best_bic:
                        best_bic = result['bic']
                        best_model_info = model_info
        else:
            # Sequential evaluation
            for pred_combo in pred_combinations:
                pred_list = list(pred_combo)
                result = fit_lmm_with_predictors(df, dv, pred_list, dataset_type)
                
                if result['converged']:
                    model_info = {
                        'predictors': pred_list,
                        'n_predictors': n_preds,
                        'bic': result['bic'],
                        'model': result
                    }
                    all_models.append(model_info)
                    
                    if result['bic'] < best_bic:
                        best_bic = result['bic']
                        best_model_info = model_info
        
        print(f"      Best BIC so far: {best_bic:.2f}")
    
    print(f"    GLOBAL OPTIMUM: {best_model_info['n_predictors']} predictors, BIC = {best_bic:.2f}")
    print(f"    Selected predictors: {best_model_info['predictors']}")
    print(f"    Total models evaluated: {len(all_models)}")
    
    # Log detailed search information
    print(f"    Search summary:")
    print(f"      - Models converged: {len([m for m in all_models if m['model']['converged']])} / {len(all_models)}")
    print(f"      - Best model has {best_model_info['n_predictors']} predictor(s)")
    print(f"      - BIC improvement over base: {all_models[0]['bic'] - best_bic:.2f}")
    
    return best_model_info, all_models


def get_experimental_effects(df, available_dvs, dataset_type='full'):
    """
    Extract experimental effects separately (not part of univariate matrix).
    Fits one model per DV with ONLY experimental predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    available_dvs : list
        Available dependent variables
    dataset_type : str
        'full' or 'ie'
        
    Returns
    -------
    dict
        Dict with experimental coefficients for each DV
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENTAL EFFECTS: {dataset_type.upper()} DATASET")
    print(f"{'='*60}")
    
    results = {}
    
    for dv in available_dvs:
        if f'{dv}_z' not in df.columns:
            continue
            
        print(f"\nAnalyzing {dv}:")
        
        # Fit base model with only experimental predictors
        if dataset_type == 'full':
            formula = (f"{dv}_z ~ C(group, Treatment('Controls'))")
        else:
            formula = (f"{dv}_z ~ C(condition, Treatment('inclusion')) + "
                      f"C(group, Treatment('Controls')) + "
                      f"C(condition, Treatment('inclusion')):"
                      f"C(group, Treatment('Controls'))")
        
        try:
            model = smf.mixedlm(formula, data=df, groups=df['subject_id'])
            result = model.fit(reml=False)
            
            # Extract all experimental coefficients
            exp_coefs = {}
            for param_name in result.params.index:
                if param_name != 'Intercept':
                    exp_coefs[param_name] = {
                        'beta': result.params[param_name],
                        'se': result.bse[param_name],
                        'p_value': result.pvalues[param_name]
                    }
                    print(f"  {param_name}: β = {result.params[param_name]:.3f}, p = {result.pvalues[param_name]:.3f}")
            
            results[dv] = {
                'coefficients': exp_coefs,
                'bic': result.bic,
                'converged': result.converged
            }
        except Exception as e:
            print(f"  Model failed: {str(e)}")
            results[dv] = {
                'coefficients': {},
                'converged': False,
                'error': str(e)
            }
    
    return results


def run_univariate_analysis(df, available_dvs, available_cont_preds, dataset_type='full'):
    """
    Run univariate analysis: ONE continuous predictor at a time.
    Experimental predictors are ALWAYS in the base model (not tested individually).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
        
    Returns
    -------
    dict
        Results for each DV-continuous_predictor combination
    """
    print(f"\n{'='*60}")
    print(f"UNIVARIATE ANALYSIS: {dataset_type.upper()} DATASET")
    print(f"One continuous predictor at a time (experimental controls always included)")
    print(f"Available DVs: {available_dvs}")
    print(f"Available continuous predictors: {available_cont_preds}")
    print(f"{'='*60}")
    
    results = {}
    
    # Debug: Check which standardized variables exist
    standardized_dvs = [dv for dv in available_dvs if f'{dv}_z' in df.columns]
    standardized_preds = [pred for pred in available_cont_preds if f'{pred}_z' in df.columns]
    print(f"DEBUG: Standardized DVs available: {standardized_dvs}")
    print(f"DEBUG: Standardized predictors available: {standardized_preds}")
    
    for dv in available_dvs:
        if f'{dv}_z' not in df.columns:
            print(f"\nSkipping {dv} - standardized version '{dv}_z' not available")
            print(f"  Available _z columns: {[col for col in df.columns if col.endswith('_z')]}")
            continue
            
        print(f"\nAnalyzing {dv}:")
        dv_results = {}
        n_valid_models = 0
        
        # Test each CONTINUOUS predictor separately
        for pred in available_cont_preds:
            if f'{pred}_z' not in df.columns:
                print(f"  {pred}: skipping - standardized version not available")
                continue
                
            # Build formula: experimental base + one continuous predictor
            if dataset_type == 'full':
                exp_formula = ("C(group, Treatment('Controls'))")
            else:
                exp_formula = ("C(condition, Treatment('inclusion')) + "
                              "C(group, Treatment('Controls')) + "
                              "C(condition, Treatment('inclusion')):"
                              "C(group, Treatment('Controls'))")
            
            formula = f"{dv}_z ~ {exp_formula} + {pred}_z"
            
            try:
                model = smf.mixedlm(formula, data=df, groups=df['subject_id'])
                result = model.fit(reml=False)
                
                # Extract coefficient for THIS continuous predictor only
                pred_coef_name = f"{pred}_z"
                if pred_coef_name in result.params:
                    dv_results[pred] = {
                        'coefficients': {
                            pred: {
                                'beta': result.params[pred_coef_name],
                                'se': result.bse[pred_coef_name],
                                'p_value': result.pvalues[pred_coef_name]
                            }
                        },
                        'bic': result.bic,
                        'converged': result.converged
                    }
                    n_valid_models += 1
                    print(f"  {pred}: β = {result.params[pred_coef_name]:.3f}, "
                          f"p = {result.pvalues[pred_coef_name]:.3f}")
                else:
                    print(f"  {pred}: coefficient '{pred_coef_name}' not found in model params: {list(result.params.keys())}")
                    
            except Exception as e:
                print(f"  {pred}: model failed - {str(e)}")
                dv_results[pred] = {
                    'converged': False,
                    'coefficients': {},  # Include empty coefficients dict
                    'error': str(e)
                }
        
        print(f"  -> {n_valid_models} valid models for {dv}")
        results[dv] = dv_results
    
    print(f"\nUnivariate analysis complete: {len([dv for dv in results if len(results[dv]) > 0])} DVs with valid results")
    return results


def run_complete_analysis(df, available_dvs, available_cont_preds, dataset_type='full'):
    """
    Run complete analysis with ALL predictors (like original approach).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
        
    Returns
    -------
    dict
        Results for each DV with all predictors
    """
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS: {dataset_type.upper()} DATASET (ALL PREDICTORS)")
    print(f"{'='*60}")
    
    results = {}
    
    # Get experimental predictors based on dataset type
    if dataset_type == 'full':
        exp_predictors = EXPERIMENTAL_PREDICTORS_FULL
    else:
        exp_predictors = EXPERIMENTAL_PREDICTORS_IE
    
    for dv in available_dvs:
        if f'{dv}_z' not in df.columns:
            print(f"Skipping {dv} - standardized version not available")
            continue
            
        print(f"\nAnalyzing {dv}:")
        
        # Fit model with ALL available continuous predictors
        complete_model = fit_lmm_with_predictors(
            df, dv, available_cont_preds, dataset_type
        )
        
        # Extract coefficients for ALL predictors (experimental + continuous)
        all_results = {}
        
        # Add experimental predictors results
        model_coeffs = complete_model.get('coefficients', {})
        
        # Add continuous predictors results
        for pred in available_cont_preds:
            if pred in model_coeffs:
                all_results[pred] = model_coeffs[pred]
        
        # Store all coefficients
        complete_model['all_coefficients'] = all_results
        
        results[dv] = {
            'complete_model': complete_model,
            'all_predictors': available_cont_preds,
            'experimental_predictors': exp_predictors
        }
        
        print(f"  Complete model: {len(available_cont_preds)} continuous predictors, BIC = {complete_model.get('bic', 'N/A'):.2f}")
    
    return results


def run_enhanced_analysis(df, available_dvs, available_cont_preds, dataset_type='full'):
    """
    Run enhanced analysis with variable selection.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
        
    Returns
    -------
    dict
        Results for each DV with best models
    """
    print(f"\n{'='*60}")
    print(f"ENHANCED ANALYSIS: {dataset_type.upper()} DATASET")
    print(f"{'='*60}")
    
    results = {}
    
    for dv in available_dvs:
        if f'{dv}_z' not in df.columns:
            print(f"Skipping {dv} - standardized version not available")
            continue
            
        print(f"\nAnalyzing {dv}:")
        
        # Use exhaustive search - evaluate ALL possible combinations
        # With 10 predictors = 1024 models per DV
        # Total: ~16,000+ models across all DVs and datasets (may take 30-60 minutes)
        max_preds = None  # No limit - evaluate all combinations
        
        # Run exhaustive search
        best_model_info, all_models_trajectory = exhaustive_model_search(
            df, dv, available_cont_preds, dataset_type, 
            use_parallel=True, max_predictors=max_preds
        )
        
        # Extract best model and predictors
        best_model = best_model_info['model']
        selected_predictors = best_model_info['predictors']
        
        results[dv] = {
            'best_model': best_model,
            'selected_predictors': selected_predictors,
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
    
    return results


def create_univariate_visualization(results_full, results_ie, available_dvs, 
                                   available_cont_preds_full, available_cont_preds_ie):
    """
    Create correlation-style heatmaps for univariate analysis.
    
    Parameters
    ----------
    results_full : dict
        Univariate results from full dataset
    results_ie : dict
        Univariate results from IE dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds_full : list
        Available continuous predictors for full dataset
    available_cont_preds_ie : list
        Available continuous predictors for IE dataset
    """
    print(f"\n{'='*60}")
    print("CREATING UNIVARIATE CORRELATION-STYLE VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Create heatmaps for both datasets
    for dataset_type, results, available_cont_preds in [
        ('full', results_full, available_cont_preds_full),
        ('ie', results_ie, available_cont_preds_ie)
    ]:
        # Univariate matrix shows ONLY continuous predictors
        # Experimental effects reported separately
        all_predictors = available_cont_preds
        
        # Create beta matrix
        beta_matrix = np.full((len(available_dvs), len(all_predictors)), np.nan)
        p_matrix = np.full((len(available_dvs), len(all_predictors)), np.nan)
        
        for i, dv in enumerate(available_dvs):
            if dv in results:
                for j, pred in enumerate(all_predictors):
                    if pred in results[dv]:
                        model_result = results[dv][pred]
                        # Check if model converged and has coefficients
                        if model_result.get('converged', False):
                            # The structure is: model_result['coefficients'][pred]['beta']
                            if 'coefficients' in model_result and pred in model_result['coefficients']:
                                coef_info = model_result['coefficients'][pred]
                                beta_matrix[i, j] = coef_info['beta']
                                p_matrix[i, j] = coef_info['p_value']
        
        # Create DataFrame for easier handling
        beta_df = pd.DataFrame(beta_matrix, 
                              index=[VARIABLE_LABELS.get(dv, dv) for dv in available_dvs],
                              columns=[PREDICTOR_LABELS.get(pred, pred) for pred in all_predictors])
        
        p_df = pd.DataFrame(p_matrix,
                           index=[VARIABLE_LABELS.get(dv, dv) for dv in available_dvs],
                           columns=[PREDICTOR_LABELS.get(pred, pred) for pred in all_predictors])
        
        # Apply FDR correction
        flat_p = p_matrix[~np.isnan(p_matrix)]
        if len(flat_p) > 0:
            _, p_corrected, _, _ = multipletests(flat_p, method='fdr_bh', alpha=0.05)
            p_corrected_matrix = np.full_like(p_matrix, np.nan)
            p_corrected_matrix[~np.isnan(p_matrix)] = p_corrected
        else:
            p_corrected_matrix = p_matrix.copy()
        
        # Create significance mask
        sig_mask = np.where(np.isnan(p_corrected_matrix), True, p_corrected_matrix >= 0.05)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Create annotation matrix
        annot_matrix = np.empty_like(beta_matrix, dtype=object)
        for i in range(beta_matrix.shape[0]):
            for j in range(beta_matrix.shape[1]):
                if not np.isnan(beta_matrix[i, j]):
                    p_val = p_corrected_matrix[i, j]
                    if not np.isnan(p_val):
                        if p_val < 0.001:
                            sig_marker = '***'
                        elif p_val < 0.01:
                            sig_marker = '**'
                        elif p_val < 0.05:
                            sig_marker = '*'
                        else:
                            sig_marker = ''
                        
                        annot_matrix[i, j] = f"β = {beta_matrix[i, j]:.3f}\np = {p_val:.3f}{sig_marker}"
                    else:
                        annot_matrix[i, j] = f"β = {beta_matrix[i, j]:.3f}"
                else:
                    annot_matrix[i, j] = ""
        
        # Create heatmap - handle case with no significant results
        if np.all(sig_mask) or beta_df.isna().all().all():
            # No significant results - show empty plot with all values as 0
            plot_data = pd.DataFrame(0, index=beta_df.index, columns=beta_df.columns)
            ax = sns.heatmap(plot_data,
                       cmap='RdBu_r', 
                       center=0,
                       annot=annot_matrix,
                       fmt='',
                       cbar_kws={'label': 'Standardized Beta Coefficient'},
                       square=True,
                       linewidths=0.5,
                       vmin=-0.1,
                       vmax=0.1)
        else:
            ax = sns.heatmap(beta_df, 
                       mask=sig_mask,
                       cmap='RdBu_r', 
                       center=0,
                       annot=annot_matrix,
                       fmt='',
                       cbar_kws={'label': 'Standardized Beta Coefficient'},
                       square=True,
                       linewidths=0.5)
        
        control_text = "Group only" if dataset_type == 'full' else "Condition × Group"
        plt.title(f'Univariate LMM: Continuous Predictors Only\n'
                 f'(All models control for {control_text})\n'
                 f'{dataset_type.upper()} Dataset',
                 fontsize=14, fontweight='bold')
        
        plt.xlabel('Predictors', fontweight='bold')
        plt.ylabel('Dependent Variables', fontweight='bold')
        
        # Add significance legend
        plt.figtext(0.02, 0.02, 
                   'Significance levels (FDR p-values): * p < 0.05  ** p < 0.01  *** p < 0.001\n'
                   'Only significant cells shown | Color intensity = effect size',
                   fontsize=9, style='italic')
        
        # Add diagonal lines for uncorrected significant p-values using exact colormap colors
        # Import colormap to get exact colors
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        
        # Set up colormap normalization (same as heatmap)
        norm = Normalize(vmin=beta_df.min().min(), vmax=beta_df.max().max())
        cmap = cm.get_cmap('RdBu_r')
        
        for i in range(beta_matrix.shape[0]):
            for j in range(beta_matrix.shape[1]):
                p_raw = p_matrix[i, j]
                p_fdr = p_corrected_matrix[i, j] if not np.isnan(p_corrected_matrix[i, j]) else np.nan
                beta_val = beta_matrix[i, j] if not np.isnan(beta_matrix[i, j]) else 0
                
                # Show uncorrected significant results with colormap-based diagonal lines
                if not np.isnan(p_raw) and p_raw < 0.05 and (np.isnan(p_fdr) or p_fdr >= 0.05):
                    # Get exact color from colormap based on beta value
                    line_color = cmap(norm(beta_val))
                    
                    # Draw multiple diagonal lines across the cell
                    x_start, x_end = j, j + 1
                    y_start, y_end = i, i + 1
                    n_lines = 5  # Number of diagonal lines
                    
                    for line_idx in range(n_lines):
                        offset = (line_idx + 1) / (n_lines + 1)  # Space lines evenly
                        x1 = x_start + 0.1 + offset * 0.15
                        x2 = x_end - 0.1 - offset * 0.15
                        y1 = y_start + 0.1 + offset * 0.15
                        y2 = y_end - 0.1 - offset * 0.15
                        
                        ax.plot([x1, x2], [y1, y2], 
                               color=line_color, linewidth=2.5, alpha=0.9)

        plt.tight_layout()
        
        # Save matrices (beta, p_raw, p_fdr)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        beta_file = os.path.join(OUTPUT_DIR, f'univariate_beta_{dataset_type}.csv')
        p_raw_file = os.path.join(OUTPUT_DIR, f'univariate_p_raw_{dataset_type}.csv')
        p_fdr_file = os.path.join(OUTPUT_DIR, f'univariate_p_fdr_{dataset_type}.csv')
        beta_df.to_csv(beta_file)
        pd.DataFrame(p_matrix, index=beta_df.index, columns=beta_df.columns).to_csv(p_raw_file)
        pd.DataFrame(p_corrected_matrix, index=beta_df.index, columns=beta_df.columns).to_csv(p_fdr_file)

        # Save both formats
        png_file = os.path.join(OUTPUT_DIR, f'univariate_lmm_heatmap_{dataset_type}.png')
        svg_file = os.path.join(OUTPUT_DIR, f'univariate_lmm_heatmap_{dataset_type}.svg')
        
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.savefig(svg_file, bbox_inches='tight')
        
        print(f"Univariate heatmap saved for {dataset_type} dataset")
        plt.close()  # Close to free memory


def create_complete_heatmap(results, available_dvs, available_cont_preds, dataset_type):
    """
    Create heatmap for complete analysis (all predictors included).
    
    Parameters
    ----------
    results : dict
        Complete analysis results
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
    """
    # Get experimental predictors based on dataset type
    if dataset_type == 'full':
        exp_predictors = ['group']
    else:
        exp_predictors = EXPERIMENTAL_PREDICTORS_IE
    
    # Combine experimental and continuous predictors
    all_predictors = exp_predictors + available_cont_preds
    
    beta_matrix = pd.DataFrame(index=available_dvs, columns=all_predictors, dtype=float)
    pval_matrix = pd.DataFrame(index=available_dvs, columns=all_predictors, dtype=float)
    
    # Fill matrices - all predictors are "selected" in complete analysis
    for dv in available_dvs:
        if dv in results:
            coefficients = results[dv]['complete_model']['coefficients']
            
            # Handle experimental predictors
            for pred in exp_predictors:
                found_coeff = None
                if pred in coefficients:
                    found_coeff = coefficients[pred]
                else:
                    # Search by pattern in coefficient names
                    for coeff_name in coefficients.keys():
                        if pred == 'group' and 'group' in coeff_name.lower() and ':' not in coeff_name:
                            found_coeff = coefficients[coeff_name]
                            break
                        elif '_x_' in pred and ':' in coeff_name:
                            found_coeff = coefficients[coeff_name]
                            break
                        elif pred in ['condition', 'condition_simple'] and 'condition' in coeff_name and ':' not in coeff_name:
                            found_coeff = coefficients[coeff_name]
                            break
                
                if found_coeff:
                    beta_matrix.loc[dv, pred] = found_coeff['beta']
                    pval_matrix.loc[dv, pred] = found_coeff['p_value']
            
            # Handle continuous predictors
            for pred in available_cont_preds:
                if pred in coefficients:
                    beta_matrix.loc[dv, pred] = coefficients[pred]['beta']
                    pval_matrix.loc[dv, pred] = coefficients[pred]['p_value']
    
    # Apply FDR correction
    valid_pvals = pval_matrix.stack().dropna()
    if len(valid_pvals) > 0:
        _, pvals_corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
        pval_fdr_series = pd.Series(pvals_corrected, index=valid_pvals.index)
        pval_fdr_matrix = pval_fdr_series.unstack(fill_value=np.nan)
    else:
        pval_fdr_matrix = pval_matrix.copy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create beta matrix for colors
    beta_for_color = beta_matrix.copy()
    
    # Create significance mask
    significance_mask = (pval_fdr_matrix >= SIGNIFICANCE_THRESHOLD)
    # Apply mask to beta_for_color
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            if significance_mask.iloc[i, j]:
                beta_for_color.iloc[i, j] = 0
    
    # Create annotation matrix
    annot_matrix = np.full(beta_matrix.shape, '', dtype=object)
    
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            dv = available_dvs[i]
            pred = all_predictors[j]
            
            beta_val = beta_matrix.loc[dv, pred]
            p_val = pval_fdr_matrix.loc[dv, pred]
            
            if not pd.isna(beta_val):
                text = f"β = {beta_val:.3f}"
                if not pd.isna(p_val):
                    if p_val < 0.001:
                        text += "\\np < .001***"
                    elif p_val < 0.01:
                        text += f"\\np = {p_val:.3f}**"
                    elif p_val < 0.05:
                        text += f"\\np = {p_val:.3f}*"
                    else:
                        text += f"\\np = {p_val:.3f}"
                annot_matrix[i, j] = text
    
    # Create heatmap
    mask = pd.isna(beta_matrix)
    
    heatmap = sns.heatmap(
        beta_for_color,
        annot=annot_matrix,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        square=False,
        linewidths=2,
        linecolor='gray',
        cbar_kws={
            "shrink": 0.8,
            "aspect": 15,
            "pad": 0.02,
            "label": "Standardized Beta Coefficient"
        },
        annot_kws={'fontsize': 10, 'fontweight': 'bold', 'ha': 'center'},
        mask=mask,
        ax=ax
    )
    
    # Add diagonal patterns for uncorrected significant results using exact colormap colors
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Set up colormap normalization (same as heatmap: -0.6 to 0.6)
    norm = Normalize(vmin=-0.6, vmax=0.6)
    cmap = cm.get_cmap('RdBu_r')
    
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            dv = available_dvs[i]
            pred = all_predictors[j]
            
            p_raw = pval_matrix.loc[dv, pred]
            p_fdr = pval_fdr_matrix.loc[dv, pred] if dv in pval_fdr_matrix.index and pred in pval_fdr_matrix.columns else np.nan
            beta_val = beta_matrix.loc[dv, pred] if not pd.isna(beta_matrix.loc[dv, pred]) else 0
            
            # Show uncorrected significant results with colormap-based diagonal lines
            if not pd.isna(p_raw) and p_raw < 0.05 and (pd.isna(p_fdr) or p_fdr >= 0.05):
                # Get exact color from colormap based on beta value
                line_color = cmap(norm(beta_val))
                
                # Draw multiple diagonal lines across the cell
                x_start, x_end = j, j + 1
                y_start, y_end = i, i + 1
                n_lines = 4  # Number of diagonal lines
                
                for line_idx in range(n_lines):
                    offset = (line_idx + 1) / (n_lines + 1)
                    x1 = x_start + 0.1 + offset * 0.15
                    x2 = x_end - 0.1 - offset * 0.15
                    y1 = y_start + 0.1 + offset * 0.15
                    y2 = y_end - 0.1 - offset * 0.15
                    
                    ax.plot([x1, x2], [y1, y2], 
                           color=line_color, linewidth=2.5, alpha=0.9)
    
    # Enhance text visibility
    for text in ax.texts:
        text_content = text.get_text()
        if text_content and 'β =' in text_content:
            try:
                beta_line = [line for line in text_content.split('\\n') if 'β =' in line][0]
                beta_val = float(beta_line.split('=')[1].strip())
                
                p_lines = [line for line in text_content.split('\\n') if 'p ' in line]
                is_significant = any('*' in line for line in p_lines)
                
                if not is_significant:
                    text.set_color('black')
                    text.set_fontweight('normal')
                elif abs(beta_val) > 0.3:
                    text.set_color('white')
                    text.set_fontweight('bold')
                else:
                    text.set_color('black')
                    text.set_fontweight('bold')
            except:
                pass
    
    # Customize labels
    y_labels = [VARIABLE_LABELS.get(label, label) for label in available_dvs]
    x_labels = [PREDICTOR_LABELS.get(label, label) for label in all_predictors]
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    # Add title
    dataset_title = "Full Dataset (All Tasks)" if dataset_type == 'full' else "IE Dataset (Sart2/4 Only)"
    ax.set_title(f'Complete LMM Analysis (All Predictors Included)\\n{dataset_title}',
                fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle
    plt.figtext(0.5, 0.94,
               f'All {len(available_cont_preds)} continuous predictors included | FDR correction applied',
               ha='center', fontsize=12, style='italic')
    
    # Add legend
    legend_text = ("Significance levels (FDR p-values): * p < 0.05   ** p < 0.01   *** p < 0.001\\n"
                  "Colored diagonal lines = uncorrected p < 0.05 | Color intensity = effect size")
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.88, right=0.95)
    
    # Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_base = f'complete_lmm_heatmap_{dataset_type}'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename_base}.png'),
               dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename_base}.svg'),
               dpi=DPI, bbox_inches='tight')
    
    print(f"Complete analysis heatmap saved for {dataset_type} dataset")
    plt.close()  # Close to free memory
    
    # Also save matrices for record
    beta_out = beta_matrix.copy()
    p_raw_out = pval_matrix.copy()
    p_fdr_out = pval_fdr_matrix.copy()
    beta_out.index = [VARIABLE_LABELS.get(d, d) for d in beta_out.index]
    beta_out.columns = [PREDICTOR_LABELS.get(p, p) for p in beta_out.columns]
    p_raw_out.index = beta_out.index
    p_raw_out.columns = beta_out.columns
    p_fdr_out = p_fdr_out.reindex(index=beta_out.index, columns=beta_out.columns)
    beta_out.to_csv(os.path.join(OUTPUT_DIR, f'complete_beta_{dataset_type}.csv'))
    p_raw_out.to_csv(os.path.join(OUTPUT_DIR, f'complete_p_raw_{dataset_type}.csv'))
    p_fdr_out.to_csv(os.path.join(OUTPUT_DIR, f'complete_p_fdr_{dataset_type}.csv'))


def create_complete_visualization(results_full, results_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie):
    """
    Create complete analysis heatmaps for both datasets.
    
    Parameters
    ----------
    results_full : dict
        Complete analysis results from full dataset
    results_ie : dict
        Complete analysis results from IE dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds_full : list
        Available continuous predictors for full dataset
    available_cont_preds_ie : list
        Available continuous predictors for IE dataset
    """
    print("\n" + "="*60)
    print("CREATING COMPLETE ANALYSIS VISUALIZATIONS")
    print("="*60)
    
    # Create heatmaps for both datasets
    for dataset_type, results, available_cont_preds in [
        ('full', results_full, available_cont_preds_full),
        ('ie', results_ie, available_cont_preds_ie)
    ]:
        if results:  # Only create if we have results
            create_complete_heatmap(results, available_dvs, available_cont_preds, dataset_type)


def create_enhanced_visualization(results_full, results_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie):
    """
    Create enhanced visualization with model selection indicators.
    
    Parameters
    ----------
    results_full : dict
        Results from full dataset analysis
    results_ie : dict
        Results from IE dataset analysis
    available_dvs : list
        Available dependent variables
    available_cont_preds_full : list
        Available continuous predictors for full dataset
    available_cont_preds_ie : list
        Available continuous predictors for IE dataset
    """
    print("\n" + "="*60)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*60)
    
    # Create two separate plots: one for full dataset, one for IE dataset
    create_enhanced_heatmap(results_full, available_dvs, available_cont_preds_full, 'full')
    create_enhanced_heatmap(results_ie, available_dvs, available_cont_preds_ie, 'ie')


def create_enhanced_heatmap(results, available_dvs, available_cont_preds, dataset_type):
    """
    Create enhanced heatmap for specific dataset.
    
    Parameters
    ----------
    results : dict
        Analysis results
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    dataset_type : str
        'full' or 'ie'
    """
    # Create matrices
    # Get experimental predictors based on dataset type
    if dataset_type == 'full':
        exp_predictors = ['group']  # Only Group for full dataset (no Condition)
    else:
        exp_predictors = EXPERIMENTAL_PREDICTORS_IE
    
    # Combine experimental and continuous predictors
    all_predictors = exp_predictors + available_cont_preds
    
    n_dvs = len(available_dvs)
    n_preds = len(all_predictors)
    
    beta_matrix = pd.DataFrame(index=available_dvs, columns=all_predictors, dtype=float)
    pval_matrix = pd.DataFrame(index=available_dvs, columns=all_predictors, dtype=float)
    selected_matrix = pd.DataFrame(index=available_dvs, columns=all_predictors, dtype=bool)
    
    # Fill matrices
    for dv in available_dvs:
        if dv in results:
            result = results[dv]
            selected_preds = result['selected_predictors']
            coefficients = result['best_model']['coefficients']
            
            # Handle experimental predictors (always "selected" in enhanced analysis)
            for pred in exp_predictors:
                # Extract experimental coefficients from model
                model = result['best_model']
                all_coeffs = model.get('coefficients', {})
                
                # Try direct lookup first
                found_coeff = None
                if pred in all_coeffs:
                    found_coeff = all_coeffs[pred]
                else:
                    # Search by pattern in coefficient names
                    for coeff_name in all_coeffs.keys():
                        # For 'group': match any coeff with 'group' but not interaction
                        if pred == 'group' and 'group' in coeff_name.lower() and ':' not in coeff_name:
                            found_coeff = all_coeffs[coeff_name]
                            break
                        # For interactions: match coeffs with ':'
                        elif '_x_' in pred and ':' in coeff_name:
                            found_coeff = all_coeffs[coeff_name]
                            break
                        # For condition: match condition-related coeffs
                        elif pred in ['condition', 'condition_simple'] and 'condition' in coeff_name and ':' not in coeff_name:
                            found_coeff = all_coeffs[coeff_name]
                            break
                
                if found_coeff:
                    beta_matrix.loc[dv, pred] = found_coeff['beta']
                    pval_matrix.loc[dv, pred] = found_coeff['p_value']
                    selected_matrix.loc[dv, pred] = True
                else:
                    selected_matrix.loc[dv, pred] = False
            
            # Handle continuous predictors (only selected ones)
            for pred in available_cont_preds:
                if pred in selected_preds and pred in coefficients:
                    beta_matrix.loc[dv, pred] = coefficients[pred]['beta']
                    pval_matrix.loc[dv, pred] = coefficients[pred]['p_value']
                    selected_matrix.loc[dv, pred] = True
                else:
                    selected_matrix.loc[dv, pred] = False
    
    # Apply FDR correction to significant results only
    valid_pvals = pval_matrix.stack().dropna()
    if len(valid_pvals) > 0:
        _, pvals_corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
        pval_fdr_series = pd.Series(pvals_corrected, index=valid_pvals.index)
        pval_fdr_matrix = pval_fdr_series.unstack(fill_value=np.nan)
    else:
        pval_fdr_matrix = pval_matrix.copy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Create beta matrix for colors (set non-selected to 0)
    beta_for_color = beta_matrix.copy()
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            if not selected_matrix.iloc[i, j]:
                beta_for_color.iloc[i, j] = 0
    
    # Create significance mask
    significance_mask = (pval_fdr_matrix >= SIGNIFICANCE_THRESHOLD) | (~selected_matrix)
    # Apply mask to beta_for_color
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            if significance_mask.iloc[i, j]:
                beta_for_color.iloc[i, j] = 0
    
    # Create annotation matrix
    annot_matrix = np.full(beta_matrix.shape, '', dtype=object)
    
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            dv = available_dvs[i]
            pred = all_predictors[j]
            
            if selected_matrix.loc[dv, pred]:
                # Variable was selected
                beta_val = beta_matrix.loc[dv, pred]
                p_val = pval_fdr_matrix.loc[dv, pred]
                
                if not pd.isna(beta_val):
                    text = f"β = {beta_val:.3f}"
                    if not pd.isna(p_val):
                        if p_val < 0.001:
                            text += "\np < .001***"
                        elif p_val < 0.01:
                            text += f"\np = {p_val:.3f}**"
                        elif p_val < 0.05:
                            text += f"\np = {p_val:.3f}*"
                        else:
                            text += f"\np = {p_val:.3f}"
                    annot_matrix[i, j] = text
            else:
                # Variable was not selected - will add diagonal lines via masking
                annot_matrix[i, j] = "///"
    
    # Create heatmap
    mask = pd.isna(beta_matrix) & selected_matrix  # Only mask truly missing selected variables
    
    heatmap = sns.heatmap(
        beta_for_color,
        annot=annot_matrix,
        fmt='',
        cmap='RdBu_r',
        center=0,
        vmin=-0.6,
        vmax=0.6,
        square=False,
        linewidths=2,
        linecolor='gray',
        cbar_kws={
            "shrink": 0.8,
            "aspect": 15,
            "pad": 0.02,
            "label": "Standardized Beta Coefficient"
        },
        annot_kws={'fontsize': 10, 'fontweight': 'bold', 'ha': 'center'},
        mask=mask,
        ax=ax
    )
    
    # Add diagonal lines for non-selected variables
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            if not selected_matrix.iloc[i, j]:
                # Add diagonal lines to indicate variable not selected
                x_start, x_end = j, j + 1
                y_start, y_end = i, i + 1
                ax.plot([x_start + 0.1, x_end - 0.1], [y_start + 0.1, y_end - 0.1], 
                       'k-', linewidth=2, alpha=0.7)
                ax.plot([x_start + 0.1, x_end - 0.1], [y_end - 0.1, y_start + 0.1], 
                       'k-', linewidth=2, alpha=0.7)

    # Overlay diagonal patterns for uncorrected significant (selected) variables using exact colormap colors
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    # Set up colormap normalization (same as heatmap: -0.6 to 0.6)
    norm = Normalize(vmin=-0.6, vmax=0.6)
    cmap = cm.get_cmap('RdBu_r')
    
    for i in range(len(available_dvs)):
        for j in range(len(all_predictors)):
            dv = available_dvs[i]
            pred = all_predictors[j]
            if selected_matrix.loc[dv, pred]:
                p_raw = pval_matrix.loc[dv, pred]
                p_fdr = pval_fdr_matrix.loc[dv, pred] if dv in pval_fdr_matrix.index and pred in pval_fdr_matrix.columns else np.nan
                beta_val = beta_matrix.loc[dv, pred] if not pd.isna(beta_matrix.loc[dv, pred]) else 0
                
                # Show uncorrected significant results with colormap-based diagonal lines
                if not pd.isna(p_raw) and p_raw < 0.05 and (pd.isna(p_fdr) or p_fdr >= 0.05):
                    # Get exact color from colormap based on beta value
                    line_color = cmap(norm(beta_val))
                    
                    # Draw multiple diagonal lines across the cell
                    x_start, x_end = j, j + 1
                    y_start, y_end = i, i + 1
                    n_lines = 6  # More lines for enhanced models
                    
                    for line_idx in range(n_lines):
                        offset = (line_idx + 1) / (n_lines + 1)
                        x1 = x_start + 0.05 + offset * 0.12
                        x2 = x_end - 0.05 - offset * 0.12
                        y1 = y_start + 0.05 + offset * 0.12
                        y2 = y_end - 0.05 - offset * 0.12
                        
                        ax.plot([x1, x2], [y1, y2], 
                               color=line_color, linewidth=2.5, alpha=0.9)
    
    # Enhance text visibility
    for text in ax.texts:
        text_content = text.get_text()
        if text_content == "///":
            text.set_color('black')
            text.set_fontsize(8)
            text.set_alpha(0.3)
        elif text_content and 'β =' in text_content:
            try:
                beta_line = [line for line in text_content.split('\n') if 'β =' in line][0]
                beta_val = float(beta_line.split('=')[1].strip())
                
                p_lines = [line for line in text_content.split('\n') if 'p ' in line]
                is_significant = any('*' in line for line in p_lines)
                
                if not is_significant:
                    text.set_color('black')
                    text.set_fontweight('normal')
                elif abs(beta_val) > 0.3:
                    text.set_color('white')
                    text.set_fontweight('bold')
                else:
                    text.set_color('black')
                    text.set_fontweight('bold')
            except:
                pass
    
    # Customize labels
    y_labels = [VARIABLE_LABELS.get(label, label) for label in available_dvs]
    x_labels = [PREDICTOR_LABELS.get(label, label) for label in all_predictors]
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    # Add title
    dataset_title = "Full Dataset (All Tasks)" if dataset_type == 'full' else "IE Dataset (Sart2/4 Only)"
    ax.set_title(f'Enhanced LMM Analysis with Variable Selection\n{dataset_title}',
                fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle
    total_models = sum(len(results[dv]['selected_predictors']) for dv in results.keys())
    plt.figtext(0.5, 0.94,
               f'BIC-optimized models | Diagonal lines = variable not selected | FDR correction applied',
               ha='center', fontsize=12, style='italic')
    
    # Add legend
    legend_text = ("Significance levels (FDR p-values): * p < 0.05   ** p < 0.01   *** p < 0.001\n"
                  "/// = Variable not selected in optimal model   |   Color intensity = effect size")
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=11, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.88, right=0.95)
    
    # Save plots and matrices (beta, p_raw, p_fdr)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename_base = f'enhanced_lmm_heatmap_{dataset_type}'
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename_base}.png'),
               dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename_base}.svg'),
               dpi=DPI, bbox_inches='tight')
    
    print(f"Enhanced heatmap saved for {dataset_type} dataset")
    plt.close()  # Close to free memory

    # Also save matrices for record
    beta_out = beta_matrix.copy()
    p_raw_out = pval_matrix.copy()
    p_fdr_out = pval_fdr_matrix.copy()
    beta_out.index = [VARIABLE_LABELS.get(d, d) for d in beta_out.index]
    beta_out.columns = [PREDICTOR_LABELS.get(p, p) for p in beta_out.columns]
    p_raw_out.index = beta_out.index
    p_raw_out.columns = beta_out.columns
    p_fdr_out = p_fdr_out.reindex(index=beta_out.index, columns=beta_out.columns)
    beta_out.to_csv(os.path.join(OUTPUT_DIR, f'enhanced_beta_{dataset_type}.csv'))
    p_raw_out.to_csv(os.path.join(OUTPUT_DIR, f'enhanced_p_raw_{dataset_type}.csv'))
    p_fdr_out.to_csv(os.path.join(OUTPUT_DIR, f'enhanced_p_fdr_{dataset_type}.csv'))


def export_experimental_effects(exp_effects_full, exp_effects_ie, available_dvs):
    """
    Export experimental effects to CSV table (not matrix).
    
    Parameters
    ----------
    exp_effects_full : dict
        Experimental effects from full dataset
    exp_effects_ie : dict
        Experimental effects from IE dataset
    available_dvs : list
        Available dependent variables
    """
    print(f"\n{'='*60}")
    print("EXPORTING EXPERIMENTAL EFFECTS")
    print(f"{'='*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for dataset_type, results in [('full', exp_effects_full), ('ie', exp_effects_ie)]:
        rows = []
        
        for dv in available_dvs:
            if dv in results and results[dv]['converged']:
                for param_name, coef_info in results[dv]['coefficients'].items():
                    rows.append({
                        'dv': dv,
                        'dv_label': VARIABLE_LABELS.get(dv, dv),
                        'experimental_effect': param_name,
                        'beta': coef_info['beta'],
                        'se': coef_info['se'],
                        'p_value': coef_info['p_value']
                    })
        
        df = pd.DataFrame(rows)
        
        # Apply FDR correction
        if len(df) > 0:
            _, p_corrected, _, _ = multipletests(df['p_value'], method='fdr_bh')
            df['p_fdr'] = p_corrected
            df['significant'] = p_corrected < 0.05
            
            # Add significance markers
            df['sig_marker'] = ''
            df.loc[df['p_fdr'] < 0.001, 'sig_marker'] = '***'
            df.loc[(df['p_fdr'] >= 0.001) & (df['p_fdr'] < 0.01), 'sig_marker'] = '**'
            df.loc[(df['p_fdr'] >= 0.01) & (df['p_fdr'] < 0.05), 'sig_marker'] = '*'
        
        filename = os.path.join(OUTPUT_DIR, f'experimental_effects_{dataset_type}.csv')
        df.to_csv(filename, index=False)
        print(f"Experimental effects table saved for {dataset_type} dataset: {len(df)} effects")


def export_all_bic_values(enhanced_results_full, enhanced_results_ie, available_dvs):
    """
    Export comprehensive BIC values from exhaustive model search.
    
    Parameters
    ----------
    enhanced_results_full : dict
        Enhanced analysis results for full dataset
    enhanced_results_ie : dict
        Enhanced analysis results for IE dataset
    available_dvs : list
        Available dependent variables
    """
    print("\nExporting comprehensive BIC values from exhaustive search...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for dataset_type, results in [('full', enhanced_results_full), ('ie', enhanced_results_ie)]:
        if not results:
            continue
            
        # Collect all BIC trajectories
        all_bic_data = []
        
        for dv in available_dvs:
            if dv in results and 'bic_trajectory' in results[dv]:
                trajectory = results[dv]['bic_trajectory']
                
                for model_info in trajectory:
                    record = {
                        'dv': dv,
                        'dv_label': VARIABLE_LABELS.get(dv, dv),
                        'n_predictors': model_info['n_predictors'],
                        'predictors': ', '.join(model_info['predictors']) if model_info['predictors'] else 'Base Model',
                        'bic': model_info['bic'],
                        'is_best': False  # Will be marked later
                    }
                    all_bic_data.append(record)
                
                # Mark the best model for this DV
                if 'best_model' in results[dv]:
                    best_bic = results[dv]['best_model']['bic']
                    for record in all_bic_data:
                        if record['dv'] == dv and abs(record['bic'] - best_bic) < 1e-6:
                            record['is_best'] = True
                            break
        
        # Convert to DataFrame and sort
        bic_df = pd.DataFrame(all_bic_data)
        if len(bic_df) > 0:
            bic_df = bic_df.sort_values(['dv', 'bic']).reset_index(drop=True)
            
            # Add rank within each DV
            bic_df['bic_rank'] = bic_df.groupby('dv')['bic'].rank(method='dense')
            
            # Export full BIC table
            bic_file = os.path.join(OUTPUT_DIR, f'exhaustive_bic_search_{dataset_type}.csv')
            bic_df.to_csv(bic_file, index=False)
            
            print(f"  {dataset_type} dataset: {len(bic_df)} models saved to {os.path.basename(bic_file)}")
            
            # Create summary statistics
            summary_data = []
            for dv in available_dvs:
                if dv in results:
                    dv_models = bic_df[bic_df['dv'] == dv]
                    if len(dv_models) > 0:
                        best_model = dv_models[dv_models['is_best']]
                        base_model = dv_models[dv_models['n_predictors'] == 0]
                        
                        summary_record = {
                            'dv': dv,
                            'dv_label': VARIABLE_LABELS.get(dv, dv),
                            'total_models_evaluated': len(dv_models),
                            'base_bic': base_model['bic'].iloc[0] if len(base_model) > 0 else np.nan,
                            'best_bic': best_model['bic'].iloc[0] if len(best_model) > 0 else np.nan,
                            'best_n_predictors': best_model['n_predictors'].iloc[0] if len(best_model) > 0 else np.nan,
                            'best_predictors': best_model['predictors'].iloc[0] if len(best_model) > 0 else '',
                            'bic_improvement': (base_model['bic'].iloc[0] - best_model['bic'].iloc[0]) if len(base_model) > 0 and len(best_model) > 0 else np.nan
                        }
                        summary_data.append(summary_record)
            
            # Export summary
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_file = os.path.join(OUTPUT_DIR, f'bic_search_summary_{dataset_type}.csv')
                summary_df.to_csv(summary_file, index=False)
                print(f"  Summary saved to {os.path.basename(summary_file)}")


def export_enhanced_results(complete_results_full, complete_results_ie, 
                          enhanced_results_full, enhanced_results_ie, available_dvs):
    """
    Export enhanced results with model selection details.
    
    Parameters
    ----------
    complete_results_full : dict
        Complete analysis results from full dataset
    complete_results_ie : dict
        Complete analysis results from IE dataset
    enhanced_results_full : dict
        Enhanced analysis results from full dataset
    enhanced_results_ie : dict
        Enhanced analysis results from IE dataset
    available_dvs : list
        Available dependent variables
    """
    print("\n" + "="*60)
    print("EXPORTING ENHANCED RESULTS")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export detailed results for each dataset and analysis type
    all_results = [
        ('complete_full', complete_results_full, 'Complete Analysis - Full Dataset'),
        ('complete_ie', complete_results_ie, 'Complete Analysis - IE Dataset'),
        ('enhanced_full', enhanced_results_full, 'Enhanced Analysis - Full Dataset'),
        ('enhanced_ie', enhanced_results_ie, 'Enhanced Analysis - IE Dataset')
    ]
    
    for dataset_type, results, description in all_results:
        
        # Create summary table
        summary_data = []
        detailed_data = []
        
        for dv in available_dvs:
            if dv in results:
                result = results[dv]
                
                # Handle different result structures
                if 'complete_model' in result:
                    # Complete analysis
                    model = result['complete_model']
                    used_preds = result['all_predictors']
                    coefficients = model.get('all_coefficients', {})
                    analysis_type = 'complete'
                elif 'best_model' in result:
                    # Enhanced analysis
                    model = result['best_model']
                    used_preds = result['selected_predictors']
                    coefficients = model.get('coefficients', {})
                    analysis_type = 'enhanced'
                else:
                    continue
                
                # Summary row
                summary_data.append({
                    'dv': dv,
                    'analysis_type': analysis_type,
                    'n_predictors': len(used_preds),
                    'predictors': ', '.join(used_preds),
                    'bic': model.get('bic', np.nan),
                    'aic': model.get('aic', np.nan),
                    'loglik': model.get('loglik', np.nan),
                    'n_obs': model.get('n_obs', np.nan),
                    'n_subjects': model.get('n_subjects', np.nan),
                    'converged': model.get('converged', False)
                })
                
                # Detailed coefficients
                for pred, coef_info in coefficients.items():
                    detailed_data.append({
                        'dv': dv,
                        'predictor': pred,
                        'beta': coef_info['beta'],
                        'se': coef_info['se'],
                        'p_value': coef_info['p_value'],
                        'dataset': dataset_type,
                        'analysis_type': analysis_type
                    })
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(OUTPUT_DIR, f'model_selection_summary_{dataset_type}.csv')
        summary_df.to_csv(summary_file, index=False)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = os.path.join(OUTPUT_DIR, f'selected_coefficients_{dataset_type}.csv')
        detailed_df.to_csv(detailed_file, index=False)
        
        print(f"Results exported for {dataset_type} dataset")
    
    # Create comprehensive report
    create_enhanced_report(complete_results_full, complete_results_ie, 
                         enhanced_results_full, enhanced_results_ie, available_dvs)


def create_enhanced_report(complete_results_full, complete_results_ie, 
                         enhanced_results_full, enhanced_results_ie, available_dvs):
    """
    Create comprehensive enhanced analysis report.
    
    Parameters
    ----------
    complete_results_full : dict
        Complete analysis results from full dataset
    complete_results_ie : dict
        Complete analysis results from IE dataset
    enhanced_results_full : dict
        Enhanced analysis results from full dataset
    enhanced_results_ie : dict
        Enhanced analysis results from IE dataset
    available_dvs : list
        Available dependent variables
    """
    
    report_lines = [
        "# Enhanced Linear Mixed Model Analysis Report",
        "",
        f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Analysis Overview",
        "This enhanced analysis includes:",
        "1. **Time-on-task variable**: Cumulative probe number across SART tasks",
        "2. **Two dataset analyses**:",
        "   - Full dataset: All tasks (Sart1-4) with baseline vs experimental condition",
        "   - IE dataset: Only Sart2/4 with inclusion vs exclusion conditions",
        "3. **BIC-based variable selection**: Recursive forward selection for optimal models",
        "4. **Enhanced visualization**: Diagonal lines indicate variables not selected",
        "",
        "## Key Features",
        "- **Model Selection**: BIC-optimized models for each dependent variable",
        "- **Multiple Comparison Correction**: FDR correction applied to selected coefficients",
        "- **Repeated Measures**: Random intercepts by participant",
        "- **Standardized Coefficients**: All variables z-scored for comparability",
        "",
        "## Variable Selection Results"
    ]
    
    # Add results for each dataset - showing enhanced results as main focus
    for dataset_name, results in [("Enhanced Full Dataset", enhanced_results_full), ("Enhanced IE Dataset", enhanced_results_ie)]:
        report_lines.extend([
            f"",
            f"### {dataset_name}",
            ""
        ])
        
        if results:
            # Summary statistics
            total_selected = sum(len(results[dv]['selected_predictors']) for dv in results.keys())
            avg_selected = total_selected / len(results) if results else 0
            
            report_lines.append(f"**Summary**: {len(results)} models fitted, average {avg_selected:.1f} predictors selected per model")
            report_lines.append("")
            
            # Individual model results
            for dv in available_dvs:
                if dv in results:
                    result = results[dv]
                    selected_preds = result['selected_predictors']
                    best_model = result['best_model']
                    dv_label = VARIABLE_LABELS.get(dv, dv)
                    
                    report_lines.append(f"**{dv_label}**: {len(selected_preds)} predictors selected")
                    if selected_preds:
                        pred_labels = [PREDICTOR_LABELS.get(p, p) for p in selected_preds]
                        report_lines.append(f"  - Selected: {', '.join(pred_labels)}")
                        report_lines.append(f"  - BIC: {best_model['bic']:.2f}")
                        
                        # Show significant coefficients
                        sig_coefs = []
                        for pred, coef_info in best_model['coefficients'].items():
                            if coef_info['p_value'] < 0.05:
                                pred_label = PREDICTOR_LABELS.get(pred, pred)
                                sig_coefs.append(f"{pred_label} (β={coef_info['beta']:.3f}, p={coef_info['p_value']:.3f})")
                        
                        if sig_coefs:
                            report_lines.append(f"  - Significant: {'; '.join(sig_coefs)}")
                    else:
                        report_lines.append("  - No predictors selected (intercept-only model)")
                    report_lines.append("")
        else:
            report_lines.append("No results available for this dataset.")
    
    # Add methodology details
    report_lines.extend([
        "",
        "## Methodology",
        "### Variable Selection Process",
        "1. **Exhaustive Search**: Evaluates ALL possible combinations of predictors",
        "2. **BIC Optimization**: Selects model with globally minimum BIC",
        "3. **Complete Evaluation**: All 2^n combinations evaluated (no limits)",
        "4. **Guaranteed Optimum**: Unlike greedy algorithms, finds true global optimum",
        "",
        "### Statistical Approach",
        "- **Base Model**: Experimental predictors (condition/group) + random intercepts",
        "- **Candidate Predictors**: Age, psychometric scales, time-on-task",
        "- **Selection Criterion**: Bayesian Information Criterion (BIC)",
        "- **Multiple Comparisons**: FDR correction on final selected models",
        "",
        "## Interpretation Guide",
        "### Visualization Elements",
        "- **Colored cells**: Variables selected with standardized beta coefficients",
        "- **Diagonal lines (///)**: Variables not selected in optimal model",
        "- **White background**: Non-significant effects (FDR p ≥ 0.05)",
        "- **Color intensity**: Effect size magnitude",
        "",
        "### Model Selection Benefits",
        "- **Reduced overfitting**: BIC penalizes model complexity",
        "- **Improved interpretability**: Focus on most important predictors",
        "- **Better generalization**: Optimal bias-variance tradeoff",
        "",
        "## Files Generated",
        "- `enhanced_lmm_heatmap_full.png/svg` - Full dataset visualization",
        "- `enhanced_lmm_heatmap_ie.png/svg` - IE dataset visualization", 
        "- `model_selection_summary_full.csv` - Full dataset model summaries",
        "- `model_selection_summary_ie.csv` - IE dataset model summaries",
        "- `selected_coefficients_full.csv` - Full dataset selected coefficients",
        "- `selected_coefficients_ie.csv` - IE dataset selected coefficients",
        "- `enhanced_analysis_report.md` - This comprehensive report"
    ])
    
    # Save report
    report_file = os.path.join(OUTPUT_DIR, 'enhanced_analysis_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Enhanced analysis report saved to: {report_file}")


def main():
    """
    Run the complete enhanced LMM analysis pipeline.
    """
    print("="*80)
    print("ENHANCED LINEAR MIXED MODEL ANALYSIS")
    print("BIC-Based Variable Selection with Time-on-Task")
    print("="*80)
    
    # Step 1: Load and prepare data
    df_full, df_ie, available_dvs, available_cont_preds_full, available_cont_preds_ie = load_and_prepare_data()
    
    # Step 2a: Run univariate analysis (continuous predictors only)
    univariate_results_full = run_univariate_analysis(df_full, available_dvs, available_cont_preds_full, 'full')
    univariate_results_ie = run_univariate_analysis(df_ie, available_dvs, available_cont_preds_ie, 'ie')
    
    # Step 2a-bis: Get experimental effects separately
    exp_effects_full = get_experimental_effects(df_full, available_dvs, 'full')
    exp_effects_ie = get_experimental_effects(df_ie, available_dvs, 'ie')
    
    # Export experimental effects tables
    export_experimental_effects(exp_effects_full, exp_effects_ie, available_dvs)
    
    # Step 2b: Run complete analysis (all predictors) on both datasets
    complete_results_full = run_complete_analysis(df_full, available_dvs, available_cont_preds_full, 'full')
    complete_results_ie = run_complete_analysis(df_ie, available_dvs, available_cont_preds_ie, 'ie')
    
    # Step 2c: Run enhanced analysis (recursive selection) on both datasets
    enhanced_results_full = run_enhanced_analysis(df_full, available_dvs, available_cont_preds_full, 'full')
    enhanced_results_ie = run_enhanced_analysis(df_ie, available_dvs, available_cont_preds_ie, 'ie')
    
    # Step 3: Create visualizations
    # 3a: Univariate correlation-style matrices
    create_univariate_visualization(univariate_results_full, univariate_results_ie, available_dvs, 
                                   available_cont_preds_full, available_cont_preds_ie)
    
    # 3b: Complete analysis (all predictors) visualizations
    create_complete_visualization(complete_results_full, complete_results_ie, available_dvs, 
                                available_cont_preds_full, available_cont_preds_ie)
    
    # 3c: Enhanced (recursive selection) visualizations
    create_enhanced_visualization(enhanced_results_full, enhanced_results_ie, available_dvs, 
                                available_cont_preds_full, available_cont_preds_ie)
    
    # Step 4: Export results
    export_enhanced_results(complete_results_full, complete_results_ie, 
                          enhanced_results_full, enhanced_results_ie, available_dvs)
    
    # Step 4b: Export comprehensive BIC search results
    export_all_bic_values(enhanced_results_full, enhanced_results_ie, available_dvs)
    
    print(f"\n" + "="*80)
    print("ENHANCED ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Complete analysis: All {len(available_cont_preds_full)} predictors tested")
    print(f"- Enhanced analysis: BIC-optimized variable selection")
    print(f"- Full dataset: {len(enhanced_results_full)} models with variable selection")
    print(f"- IE dataset: {len(enhanced_results_ie)} models with variable selection")
    
    # Show comparison between complete and enhanced approaches
    print(f"\nModel Complexity Comparison:")
    for dataset_name, complete_results, enhanced_results in [
        ("Full", complete_results_full, enhanced_results_full), 
        ("IE", complete_results_ie, enhanced_results_ie)
    ]:
        if enhanced_results:
            avg_selected = np.mean([len(enhanced_results[dv]['selected_predictors']) for dv in enhanced_results.keys()])
            total_available = len(available_cont_preds_full) if dataset_name == "Full" else len(available_cont_preds_ie)
            print(f"- {dataset_name} dataset: {total_available} available → {avg_selected:.1f} selected on average")
            
            # Show some examples
            for dv in list(enhanced_results.keys())[:3]:  # Show first 3 examples
                n_selected = len(enhanced_results[dv]['selected_predictors'])
                if complete_results and dv in complete_results:
                    complete_bic = complete_results[dv]['complete_model'].get('bic', 'N/A')
                    enhanced_bic = enhanced_results[dv]['best_model'].get('bic', 'N/A')
                    print(f"  • {dv}: {total_available}→{n_selected} predictors, BIC: {complete_bic:.1f}→{enhanced_bic:.1f}")
                else:
                    print(f"  • {dv}: {n_selected} predictors selected")


if __name__ == "__main__":
    main()

# %%
