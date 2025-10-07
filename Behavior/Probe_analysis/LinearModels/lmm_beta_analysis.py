# %%
"""
Linear Mixed Model Beta Analysis - Recreating Correlation Matrix with LMM Standardized Betas

This script recreates the correlation matrix analysis using Linear Mixed Models (LMMs) 
with standardized betas instead of simple correlations. This approach accounts for the 
repeated measures structure and experimental design while quantifying relationship strength.

Key features:
- Repeated measures design with random intercepts by participant
- Standardized betas for direct comparison across variables
- Multiple comparison correction (FDR)
- Visualization matching the correlation matrix style
- Comprehensive results export

Data structure:
- Repeated measures: Multiple probes per participant
- Between-subjects factor: Group (Controls vs Risk of Depression)
- Within-subjects factor: Condition (inclusion vs exclusion vs baseline)
- Participant ID: subject_id (random effect)
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
from statsmodels.regression.mixed_linear_model import MixedLM

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Data paths
PROBE_DATA_PATH = "../../results/Behavior/probe_data/probe_level_aggregated_data.csv"
PCA_DATA_PATH = "../../results/Behavior/probe_data/pca_results.csv"
OUTPUT_DIR = "../../results/Behavior/probe_data/lmm_beta_analysis"

# Variables for analysis
DEPENDENT_VARIABLES = ['onoff', 'valence', 'time', 'selfother', 'confidence', 'PC1', 'PC2', 'PC3']

# Experimental predictors (categorical)
EXPERIMENTAL_PREDICTORS = ['condition', 'group', 'condition_x_group']

# Continuous predictors (psychometric scales)
CONTINUOUS_PREDICTORS = ['age', 'bdi', 'rrs_tot', 'mwq', 'fne', 'self_esteem', 'ctq_tot', 'a_rsq', 'sris']

# All predictors for matrix
ALL_PREDICTORS = EXPERIMENTAL_PREDICTORS + CONTINUOUS_PREDICTORS

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
    'group': 'Group',
    'condition_x_group': 'Condition×Group',
    'age': 'Age',
    'bdi': 'BDI',
    'rrs_tot': 'RRS',
    'mwq': 'MWQ',
    'fne': 'FNE',
    'self_esteem': 'Self-Est.',
    'ctq_tot': 'CTQ',
    'a_rsq': 'ARSQ',
    'sris': 'SRIS'
}

# Analysis settings
APPLY_ONOFF_FILTER = True
ONOFF_MAX_EXCLUSIVE = 100.0
USE_FDR_CORRECTION = True
SIGNIFICANCE_THRESHOLD = 0.05

# Plot settings
FIGURE_SIZE = (18, 12)
DPI = 300

# =============================================================================


def load_and_prepare_data():
    """
    Load and prepare data for LMM analysis with proper encoding and standardization.
    
    Returns
    -------
    tuple
        (df, available_dvs, available_cont_preds) - Prepared dataset and available variables
    """
    print("Loading and preparing data for LMM analysis...")
    
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
    
    # Merge with PCA data on relevant columns
    merge_cols = ['subject_id', 'task', 'probe_number']
    df = pd.merge(probe_df, pca_df[merge_cols + ['PC1', 'PC2', 'PC3']], 
                  on=merge_cols, how='left', suffixes=('', '_pca'))
    
    print(f"Total observations after merging: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    
    # Create condition variable for experimental design
    # Map inclusion_exclusion to cleaner condition names
    condition_map = {
        'inclusion': 'inclusion',
        'exclusion': 'exclusion', 
        'baseline': 'baseline'
    }
    df['condition'] = df['inclusion_exclusion'].map(condition_map)
    
    # Filter out unknown conditions
    df = df[df['condition'].notna()].copy()
    print(f"After filtering unknown conditions: {len(df)} observations")
    
    # Ensure we have the required variables
    required_vars = DEPENDENT_VARIABLES + CONTINUOUS_PREDICTORS + ['subject_id', 'condition', 'group']
    missing_vars = [var for var in required_vars if var not in df.columns]
    if missing_vars:
        print(f"Warning: Missing variables {missing_vars}")
        # Filter available variables from analysis
        available_dvs = [var for var in DEPENDENT_VARIABLES if var in df.columns]
        available_cont_preds = [var for var in CONTINUOUS_PREDICTORS if var in df.columns]
        print(f"Available DVs: {available_dvs}")
        print(f"Available continuous predictors: {available_cont_preds}")
    else:
        available_dvs = DEPENDENT_VARIABLES
        available_cont_preds = CONTINUOUS_PREDICTORS
    
    # Set categorical reference levels
    df['condition'] = pd.Categorical(df['condition'], 
                                    categories=['baseline', 'inclusion', 'exclusion'])
    df['group'] = pd.Categorical(df['group'], 
                                categories=['Controls', 'Risk of Depression'])
    
    # Create interaction term for condition × group
    df['condition_x_group'] = df['condition'].astype(str) + '_' + df['group'].astype(str)
    
    # Ensure subject_id is string for proper grouping
    df['subject_id'] = df['subject_id'].astype(str)
    
    # Standardize all continuous variables (DVs and continuous predictors)
    variables_to_standardize = []
    
    # Add available DVs
    for var in available_dvs:
        if var in df.columns and df[var].dtype in ['float64', 'int64']:
            variables_to_standardize.append(var)
    
    # Add available continuous predictors
    for var in available_cont_preds:
        if var in df.columns and df[var].dtype in ['float64', 'int64']:
            variables_to_standardize.append(var)
    
    print(f"Standardizing variables: {variables_to_standardize}")
    
    # Standardize variables and create _z versions
    for var in variables_to_standardize:
        if df[var].std() > 0:  # Only standardize if there's variance
            df[f'{var}_z'] = (df[var] - df[var].mean()) / df[var].std()
        else:
            print(f"Warning: Variable {var} has no variance, using original values")
            df[f'{var}_z'] = df[var]
    
    # Remove rows with missing data in key variables
    key_vars = ['subject_id', 'condition', 'group'] + [f'{var}_z' for var in variables_to_standardize]
    before_clean = len(df)
    df = df.dropna(subset=key_vars)
    after_clean = len(df)
    
    if before_clean != after_clean:
        print(f"Removed {before_clean - after_clean} rows with missing data in key variables")
    
    print(f"Final dataset: {len(df)} observations from {df['subject_id'].nunique()} subjects")
    print(f"Group distribution: {df['group'].value_counts().to_dict()}")
    print(f"Condition distribution: {df['condition'].value_counts().to_dict()}")
    
    return df, available_dvs, available_cont_preds


def fit_lmm_model(df, dv, predictor_type, predictor_name):
    """
    Fit a Linear Mixed Model for a specific DV and predictor.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    dv : str
        Dependent variable name
    predictor_type : str
        Type of predictor ('experimental' or 'continuous')
    predictor_name : str
        Name of the predictor variable
        
    Returns
    -------
    dict
        Results dictionary with beta, SE, t-value, p-value, and metadata
    """
    
    try:
        # Define formula based on predictor type
        if predictor_type == 'experimental':
            if predictor_name == 'condition':
                # Test condition effect (inclusion/exclusion vs baseline)
                formula = f"{dv}_z ~ C(condition, Treatment(reference='baseline')) + C(group)"
            elif predictor_name == 'group':
                # Test group effect
                formula = f"{dv}_z ~ C(group, Treatment(reference='Controls')) + C(condition)"
            elif predictor_name == 'condition_x_group':
                # Test interaction effect
                formula = f"{dv}_z ~ C(condition, Treatment(reference='baseline')) * C(group, Treatment(reference='Controls'))"
            else:
                raise ValueError(f"Unknown experimental predictor: {predictor_name}")
                
        elif predictor_type == 'continuous':
            # Test continuous predictor while controlling for experimental design
            formula = f"{dv}_z ~ {predictor_name}_z + C(condition, Treatment(reference='baseline')) + C(group, Treatment(reference='Controls'))"
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        # Fit the model
        model = smf.mixedlm(formula, data=df, groups=df['subject_id'])
        result = model.fit(reml=False)
        
        # Extract the coefficient of interest
        params = result.params
        pvalues = result.pvalues
        std_errors = result.bse
        tvalues = result.tvalues
        
        # Find the coefficient of interest based on predictor type
        if predictor_type == 'experimental':
            if predictor_name == 'condition':
                # Look for condition effects (inclusion or exclusion vs baseline)
                coef_names = [name for name in params.index if 'condition' in name and 'baseline' not in name]
                if coef_names:
                    # Use the first condition effect (typically inclusion)
                    coef_name = coef_names[0]
                else:
                    # Fallback to any condition coefficient
                    coef_name = [name for name in params.index if 'condition' in name][0]
            elif predictor_name == 'group':
                coef_name = [name for name in params.index if 'group' in name and ':' not in name][0]
            elif predictor_name == 'condition_x_group':
                coef_name = [name for name in params.index if ':' in name][0]
        else:  # continuous
            coef_name = f"{predictor_name}_z"
        
        if coef_name not in params.index:
            return {
                'dv': dv,
                'predictor': predictor_name,
                'predictor_type': predictor_type,
                'beta': np.nan,
                'se': np.nan,
                't_value': np.nan,
                'p_value': np.nan,
                'n_obs': len(df),
                'n_subjects': df['subject_id'].nunique(),
                'error': f'Coefficient {coef_name} not found',
                'converged': result.converged
            }
        
        return {
            'dv': dv,
            'predictor': predictor_name,
            'predictor_type': predictor_type,
            'beta': params[coef_name],
            'se': std_errors[coef_name],
            't_value': tvalues[coef_name], 
            'p_value': pvalues[coef_name],
            'n_obs': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'coef_name': coef_name,
            'converged': result.converged,
            'error': None
        }
        
    except Exception as e:
        return {
            'dv': dv,
            'predictor': predictor_name,
            'predictor_type': predictor_type,
            'beta': np.nan,
            'se': np.nan,
            't_value': np.nan,
            'p_value': np.nan,
            'n_obs': len(df) if df is not None else 0,
            'n_subjects': df['subject_id'].nunique() if df is not None else 0,
            'error': str(e),
            'converged': False
        }


def run_systematic_lmm_analysis(df, available_dvs, available_cont_preds):
    """
    Run LMM analysis systematically for all DV-predictor combinations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
        
    Returns
    -------
    pd.DataFrame
        Complete results dataframe
    """
    print("\n" + "="*60)
    print("RUNNING SYSTEMATIC LMM ANALYSIS")
    print("="*60)
    
    results_list = []
    all_predictors = EXPERIMENTAL_PREDICTORS + available_cont_preds
    total_models = len(available_dvs) * len(all_predictors)
    current_model = 0
    
    for dv in available_dvs:
        if f'{dv}_z' not in df.columns:
            print(f"Skipping {dv} - standardized version not available")
            continue
            
        print(f"\nAnalyzing dependent variable: {dv}")
        print("-" * 40)
        
        # Test experimental predictors
        for pred in EXPERIMENTAL_PREDICTORS:
            current_model += 1
            print(f"  Model {current_model}/{total_models}: {dv} ~ {pred}")
            
            result = fit_lmm_model(df, dv, 'experimental', pred)
            results_list.append(result)
            
            if result['error']:
                print(f"    Error: {result['error']}")
            else:
                print(f"    β = {result['beta']:.3f}, p = {result['p_value']:.3f}")
        
        # Test continuous predictors
        for pred in available_cont_preds:
            if f'{pred}_z' not in df.columns:
                print(f"  Skipping {pred} - standardized version not available")
                continue
                
            current_model += 1
            print(f"  Model {current_model}/{total_models}: {dv} ~ {pred}")
            
            result = fit_lmm_model(df, dv, 'continuous', pred)
            results_list.append(result)
            
            if result['error']:
                print(f"    Error: {result['error']}")
            else:
                print(f"    β = {result['beta']:.3f}, p = {result['p_value']:.3f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)
    
    print(f"\nCompleted {len(results_df)} models")
    print(f"Successful models: {sum(results_df['converged'])}")
    print(f"Failed models: {sum(~results_df['converged'])}")
    
    return results_df


def apply_multiple_comparison_correction(results_df):
    """
    Apply FDR correction for multiple comparisons.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from systematic analysis
        
    Returns
    -------
    pd.DataFrame
        Results with FDR-corrected p-values
    """
    print("\n" + "="*60)
    print("APPLYING MULTIPLE COMPARISON CORRECTION")
    print("="*60)
    
    results_corrected = results_df.copy()
    
    # Get valid p-values (not NaN and model converged)
    valid_mask = (~results_corrected['p_value'].isna()) & results_corrected['converged']
    valid_pvals = results_corrected.loc[valid_mask, 'p_value'].values
    
    print(f"Valid p-values for correction: {len(valid_pvals)}")
    
    if len(valid_pvals) > 0:
        # Apply FDR correction
        _, pvals_corrected, _, _ = multipletests(valid_pvals, method='fdr_bh')
        
        # Store corrected p-values
        results_corrected.loc[valid_mask, 'p_fdr'] = pvals_corrected
        results_corrected['p_fdr'] = results_corrected['p_fdr'].fillna(np.nan)
        
        # Mark significance
        results_corrected['significant_uncorrected'] = results_corrected['p_value'] < SIGNIFICANCE_THRESHOLD
        results_corrected['significant_fdr'] = results_corrected['p_fdr'] < SIGNIFICANCE_THRESHOLD
        
        # Summary statistics
        n_sig_uncorr = sum(results_corrected['significant_uncorrected'])
        n_sig_fdr = sum(results_corrected['significant_fdr'])
        
        print(f"Significant effects (uncorrected p < {SIGNIFICANCE_THRESHOLD}): {n_sig_uncorr}")
        print(f"Significant effects (FDR corrected p < {SIGNIFICANCE_THRESHOLD}): {n_sig_fdr}")
        
    else:
        print("No valid p-values found for correction")
        results_corrected['p_fdr'] = np.nan
        results_corrected['significant_uncorrected'] = False
        results_corrected['significant_fdr'] = False
    
    return results_corrected


def create_beta_matrices(results_df):
    """
    Create matrices for visualization (beta, p-values, significance).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results with corrections
        
    Returns
    -------
    tuple
        (beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix)
    """
    print("\n" + "="*60)
    print("CREATING BETA MATRICES")
    print("="*60)
    
    # Get available DVs and predictors from results
    available_dvs = results_df['dv'].unique()
    available_predictors = results_df['predictor'].unique()
    
    print(f"Creating matrices for {len(available_dvs)} DVs x {len(available_predictors)} predictors")
    
    # Initialize matrices
    beta_matrix = pd.DataFrame(index=available_dvs, columns=available_predictors, dtype=float)
    pval_matrix = pd.DataFrame(index=available_dvs, columns=available_predictors, dtype=float)
    pval_fdr_matrix = pd.DataFrame(index=available_dvs, columns=available_predictors, dtype=float)
    se_matrix = pd.DataFrame(index=available_dvs, columns=available_predictors, dtype=float)
    
    # Fill matrices
    for _, row in results_df.iterrows():
        dv = row['dv']
        pred = row['predictor']
        
        if pd.notna(row['beta']):
            beta_matrix.loc[dv, pred] = row['beta']
            pval_matrix.loc[dv, pred] = row['p_value']
            pval_fdr_matrix.loc[dv, pred] = row['p_fdr']
            se_matrix.loc[dv, pred] = row['se']
    
    print("Beta matrix created successfully")
    return beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix


def create_beta_heatmap(beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix):
    """
    Create heatmap visualization matching the correlation matrix style.
    
    Parameters
    ----------
    beta_matrix : pd.DataFrame
        Matrix of standardized beta coefficients
    pval_matrix : pd.DataFrame
        Matrix of uncorrected p-values
    pval_fdr_matrix : pd.DataFrame
        Matrix of FDR-corrected p-values
    se_matrix : pd.DataFrame
        Matrix of standard errors
    """
    print("\n" + "="*60)
    print("CREATING BETA COEFFICIENT HEATMAP")
    print("="*60)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Choose p-values for significance determination
    p_values_for_viz = pval_fdr_matrix if USE_FDR_CORRECTION else pval_matrix
    correction_label = "FDR-corrected" if USE_FDR_CORRECTION else "uncorrected"
    
    # Create significance-aware beta matrix for colors
    beta_for_color = beta_matrix.copy()
    significance_mask = p_values_for_viz >= SIGNIFICANCE_THRESHOLD
    beta_for_color[significance_mask] = 0  # Set non-significant to 0 for white color
    
    # Create annotation matrix
    annot_matrix = np.full(beta_matrix.shape, '', dtype=object)
    
    for i in range(beta_matrix.shape[0]):
        for j in range(beta_matrix.shape[1]):
            beta_val = beta_matrix.iloc[i, j]
            p_uncorr = pval_matrix.iloc[i, j]
            p_fdr = pval_fdr_matrix.iloc[i, j]
            se_val = se_matrix.iloc[i, j]
            
            # Choose which p-value to display
            p_display = p_fdr if USE_FDR_CORRECTION else p_uncorr
            
            if not pd.isna(beta_val):
                # Format beta value
                text = f"β = {beta_val:.3f}"
                
                # Add p-value with significance markers
                if not pd.isna(p_display):
                    if p_display < 0.001:
                        text += "\np < .001***"
                    elif p_display < 0.01:
                        text += f"\np = {p_display:.3f}**"
                    elif p_display < 0.05:
                        text += f"\np = {p_display:.3f}*"
                    else:
                        text += f"\np = {p_display:.3f}"
                
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
        linewidths=1.5,
        linecolor='gray',
        cbar_kws={
            "shrink": 0.8,
            "aspect": 15,
            "pad": 0.02,
            "label": "Standardized Beta Coefficient"
        },
        annot_kws={'fontsize': 9, 'fontweight': 'bold', 'ha': 'center'},
        mask=mask,
        ax=ax
    )
    
    # Enhance text visibility
    for text in ax.texts:
        text_content = text.get_text()
        if text_content and 'β =' in text_content:
            try:
                # Extract beta value
                beta_line = [line for line in text_content.split('\n') if 'β =' in line][0]
                beta_val = float(beta_line.split('=')[1].strip())
                
                # Check significance
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
    y_labels = [VARIABLE_LABELS.get(label, label) for label in beta_matrix.index]
    x_labels = [PREDICTOR_LABELS.get(label, label) for label in beta_matrix.columns]
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    # Add title and subtitle
    ax.set_title('Linear Mixed Model Standardized Beta Coefficients:\nProbe Dimensions & PCA Components vs Experimental & Psychometric Predictors',
                fontsize=16, fontweight='bold', pad=30)
    
    plt.figtext(0.5, 0.94,
               f'N = {pval_matrix.notna().sum().max()} observations | White cells = non-significant ({correction_label} p ≥ 0.05)',
               ha='center', fontsize=11, style='italic')
    
    # Add significance legend
    legend_text = (f"Significance levels ({correction_label} p-values): "
                  "* p < 0.05   ** p < 0.01   *** p < 0.001   |   "
                  "White background = non-significant   |   "
                  "Color intensity = effect size")
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.88, right=0.95)
    
    # Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'lmm_beta_heatmap.png'),
               dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'lmm_beta_heatmap.svg'),
               dpi=DPI, bbox_inches='tight')
    
    # plt.show()  # Commented out to avoid stopping execution


def export_results(results_df, beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix, available_dvs, available_cont_preds):
    """
    Export comprehensive results tables and matrices.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results dataframe
    beta_matrix : pd.DataFrame
        Beta coefficient matrix
    pval_matrix : pd.DataFrame
        P-value matrix
    pval_fdr_matrix : pd.DataFrame
        FDR-corrected p-value matrix
    se_matrix : pd.DataFrame
        Standard error matrix
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    """
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export main results table
    results_export = results_df.copy()
    results_export = results_export.sort_values(['significant_fdr', 'p_fdr'], 
                                               ascending=[False, True])
    
    results_file = os.path.join(OUTPUT_DIR, 'lmm_beta_results_complete.csv')
    results_export.to_csv(results_file, index=False)
    print(f"Complete results saved to: {results_file}")
    
    # Export matrices
    beta_matrix.to_csv(os.path.join(OUTPUT_DIR, 'beta_matrix.csv'))
    pval_matrix.to_csv(os.path.join(OUTPUT_DIR, 'pval_matrix.csv'))
    pval_fdr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'pval_fdr_matrix.csv'))
    se_matrix.to_csv(os.path.join(OUTPUT_DIR, 'se_matrix.csv'))
    print("Matrices saved to CSV files")
    
    # Create summary of significant effects
    sig_effects = results_df[results_df['significant_fdr']].copy()
    if len(sig_effects) > 0:
        sig_file = os.path.join(OUTPUT_DIR, 'significant_effects_fdr.csv')
        sig_effects.to_csv(sig_file, index=False)
        print(f"Significant effects (FDR) saved to: {sig_file}")
        
        print(f"\nSignificant Effects Summary (FDR corrected p < {SIGNIFICANCE_THRESHOLD}):")
        print("-" * 60)
        for _, row in sig_effects.iterrows():
            dv_label = VARIABLE_LABELS.get(row['dv'], row['dv'])
            pred_label = PREDICTOR_LABELS.get(row['predictor'], row['predictor'])
            print(f"{dv_label} ~ {pred_label}: β = {row['beta']:.3f}, p = {row['p_fdr']:.3f}")
    else:
        print("No significant effects after FDR correction")
    
    # Create summary report
    create_summary_report(results_df, beta_matrix, available_dvs, available_cont_preds)


def create_summary_report(results_df, beta_matrix, available_dvs, available_cont_preds):
    """
    Create comprehensive summary report.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results
    beta_matrix : pd.DataFrame
        Beta coefficient matrix
    available_dvs : list
        Available dependent variables
    available_cont_preds : list
        Available continuous predictors
    """
    
    report_lines = [
        "# Linear Mixed Model Beta Analysis - Summary Report",
        "",
        f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Models Fitted:** {len(results_df)}",
        f"**Successful Models:** {sum(results_df['converged'])}",
        "",
        "## Analysis Overview",
        "This analysis recreates correlation matrix patterns using Linear Mixed Models (LMMs)",
        "with standardized beta coefficients. This approach accounts for:",
        "- Repeated measures structure (random intercepts by participant)",
        "- Experimental design (condition and group effects)",
        "- Multiple comparison correction (FDR)",
        "",
        "## Variables Analyzed",
        f"**Dependent Variables:** {', '.join(available_dvs)}",
        f"**Experimental Predictors:** {', '.join(EXPERIMENTAL_PREDICTORS)}",
        f"**Continuous Predictors:** {', '.join(available_cont_preds)}",
        "",
        "## Statistical Method",
        "- **Model Type:** Linear Mixed Models with random intercepts by participant",
        "- **Coefficients:** Standardized betas (all variables z-scored)",
        "- **Multiple Comparisons:** FDR correction across all tests",
        f"- **Significance Threshold:** p < {SIGNIFICANCE_THRESHOLD}",
        "",
        "## Key Findings"
    ]
    
    # Add significant effects summary
    sig_effects = results_df[results_df['significant_fdr']]
    if len(sig_effects) > 0:
        report_lines.extend([
            f"",
            f"### Significant Effects (FDR corrected p < {SIGNIFICANCE_THRESHOLD})",
            ""
        ])
        
        for _, row in sig_effects.iterrows():
            dv_label = VARIABLE_LABELS.get(row['dv'], row['dv'])
            pred_label = PREDICTOR_LABELS.get(row['predictor'], row['predictor'])
            effect_size = "large" if abs(row['beta']) >= 0.5 else "medium" if abs(row['beta']) >= 0.3 else "small"
            report_lines.append(f"- **{dv_label} ~ {pred_label}:** β = {row['beta']:.3f}, p = {row['p_fdr']:.3f} ({effect_size} effect)")
    else:
        report_lines.extend([
            "",
            "### No significant effects survived FDR correction",
            ""
        ])
    
    # Add uncorrected significant effects summary
    uncorr_sig = results_df[results_df['significant_uncorrected']]
    if len(uncorr_sig) > 0:
        report_lines.extend([
            "",
            f"### Effects Significant at Uncorrected p < {SIGNIFICANCE_THRESHOLD} ({len(uncorr_sig)} total)",
            ""
        ])
        
        # Show top 10 strongest effects
        top_effects = uncorr_sig.reindex(uncorr_sig['beta'].abs().sort_values(ascending=False).index).head(10)
        for _, row in top_effects.iterrows():
            dv_label = VARIABLE_LABELS.get(row['dv'], row['dv'])
            pred_label = PREDICTOR_LABELS.get(row['predictor'], row['predictor'])
            fdr_note = " (FDR sig.)" if row['significant_fdr'] else ""
            report_lines.append(f"- {dv_label} ~ {pred_label}: β = {row['beta']:.3f}, p = {row['p_value']:.3f}{fdr_note}")
    
    report_lines.extend([
        "",
        "## Files Generated",
        "- `lmm_beta_heatmap.png/svg` - Visualization of standardized betas",
        "- `lmm_beta_results_complete.csv` - Complete model results",
        "- `beta_matrix.csv` - Beta coefficient matrix",
        "- `pval_matrix.csv` - Uncorrected p-value matrix",
        "- `pval_fdr_matrix.csv` - FDR-corrected p-value matrix",
        "- `se_matrix.csv` - Standard error matrix",
        "- `significant_effects_fdr.csv` - FDR-significant effects only",
        "- `lmm_summary_report.md` - This comprehensive report"
    ])
    
    # Save report
    report_file = os.path.join(OUTPUT_DIR, 'lmm_summary_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to: {report_file}")


def main():
    """
    Run the complete LMM beta analysis pipeline.
    """
    print("="*80)
    print("LINEAR MIXED MODEL BETA ANALYSIS")
    print("Recreating Correlation Matrix with Standardized Betas")
    print("="*80)
    
    # Step 1: Load and prepare data
    df, available_dvs, available_cont_preds = load_and_prepare_data()
    
    # Step 2: Run systematic LMM analysis
    results_df = run_systematic_lmm_analysis(df, available_dvs, available_cont_preds)
    
    # Step 3: Apply multiple comparison correction
    results_df = apply_multiple_comparison_correction(results_df)
    
    # Step 4: Create matrices for visualization
    beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix = create_beta_matrices(results_df)
    
    # Step 5: Create visualization
    create_beta_heatmap(beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix)
    
    # Step 6: Export results
    export_results(results_df, beta_matrix, pval_matrix, pval_fdr_matrix, se_matrix, available_dvs, available_cont_preds)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Final summary
    n_sig_fdr = sum(results_df['significant_fdr'])
    n_sig_uncorr = sum(results_df['significant_uncorrected'])
    
    print(f"\nFinal Summary:")
    print(f"- Total models fitted: {len(results_df)}")
    print(f"- Significant effects (FDR corrected): {n_sig_fdr}")
    print(f"- Significant effects (uncorrected): {n_sig_uncorr}")
    
    if n_sig_fdr > 0:
        sig_results = results_df[results_df['significant_fdr']].copy()
        strongest_idx = sig_results['beta'].abs().idxmax()
        strongest = sig_results.loc[strongest_idx]
        dv_label = VARIABLE_LABELS.get(strongest['dv'], strongest['dv'])
        pred_label = PREDICTOR_LABELS.get(strongest['predictor'], strongest['predictor'])
        print(f"- Strongest effect: {dv_label} ~ {pred_label} (β = {strongest['beta']:.3f})")


if __name__ == "__main__":
    main()

# %%
