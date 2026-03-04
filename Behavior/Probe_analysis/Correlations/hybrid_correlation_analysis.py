# %%
"""
Hybrid Correlation Analysis: Complete (Probes/PCA) & Partial (Psychometrics)

This script creates an intermediate correlation heatmap where:
- Probe-Probe and PCA-Probe correlations are COMPLETE (Spearman, no controls)
- Probe-Psychometric and PCA-Psychometric correlations are PARTIAL (controlling for probe dimensions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Data paths
PROBE_DATA_PATH = "results/Behavior/probe_data/probe_level_aggregated_data.csv"
PCA_DATA_PATH = "results/Behavior/probe_data/pca_results.csv"
OUTPUT_DIR = "results/Behavior/probe_data/hybrid_correlation_analysis"

# Variables for analysis
PROBE_DIMENSIONS = ['onoff', 'valence', 'time', 'selfother', 'confidence']
PCA_COMPONENTS = ['PC1', 'PC2', 'PC3']

# Key psychometric scales
PSYCHOMETRIC_SCALES = [
    'age', 'bdi', 'rrs_tot', 'mwq', 'fne', 'self-esteem',
    'ctq_tot', 'a_rsq', 'sris'
]

# Plot aesthetics
plt.style.use('default')
FIGURE_SIZE = (16, 7)  # More condensed dimensions
DPI = 300
SIGNIFICANCE_THRESHOLD = 0.05

# =============================================================================
# MULTIPLE COMPARISONS CORRECTION SETTING
# =============================================================================
# Controls how p-values are handled for visualization and significance marking:
# 
# USE_FDR_CORRECTION = True:  Uses FDR-corrected p-values (row-wise correction)
#                            - FDR correction applied SEPARATELY for each probe dimension/PCA component
#                            - Less stringent than matrix-wide correction
#                            - White cells: FDR-corrected p ≥ 0.05
#                            - Colored cells: FDR-corrected p < 0.05
#                            - Text annotations show FDR-corrected p-values
#
# USE_FDR_CORRECTION = False: Uses uncorrected p-values (most liberal)
#                            - White cells: uncorrected p ≥ 0.05  
#                            - Colored cells: uncorrected p < 0.05
#                            - Text annotations show uncorrected p-values
#
USE_FDR_CORRECTION = True  # ← Change this to False to use uncorrected p-values
# =============================================================================

# =============================================================================
# PSYCHOMETRIC MUTUAL CONTROL SETTING
# =============================================================================
# Controls whether psychometric scales are controlled for each other:
#
# CONTROL_PSYCHOMETRICS_MUTUALLY = True:
#   - When correlating two psychometric scales, controls for all OTHER psychometric scales
#   - Example: BDI vs RRS controls for MWQ, FNE, Self-Est, CTQ, ARSQ, SRIS
#   - More conservative approach accounting for intercorrelations among scales
#   - Isolates unique variance between two scales beyond shared variance with others
#
# CONTROL_PSYCHOMETRICS_MUTUALLY = False:
#   - Psychometric-psychometric correlations are simple (no controls)
#   - Only probe dimensions and PCA components have controls
#   - More liberal, shows total correlation between scales
#
CONTROL_PSYCHOMETRICS_MUTUALLY = False  # ← Change to False for simple psychometric correlations
# =============================================================================

# Enhanced labels for display
DIMENSION_LABELS = {
    'onoff': 'OnOff',
    'valence': 'Valence',
    'time': 'Time',
    'selfother': 'Self/Other',
    'confidence': 'Confidence',
    'PC1': 'PC1',
    'PC2': 'PC2',
    'PC3': 'PC3'
}

SCALE_LABELS = {
    'age': 'Age',
    'bdi': 'BDI',
    'rrs_tot': 'RRS',
    'mwq': 'MWQ',
    'fne': 'FNE',
    'self-esteem': 'Self-Est.',
    'ctq_tot': 'CTQ',
    'a_rsq': 'ARSQ',
    'sris': 'SRIS'
}


def load_and_prepare_data():
    """
    Load and prepare data for partial correlation analysis.
    
    Returns
    -------
    tuple
        (probe_level_df, subject_level_df) - probe-level data for probe-probe/probe-PCA correlations,
        subject-level data for psychometric correlations
    """
    print("Loading and preparing data...")
    
    # Load datasets
    probe_df = pd.read_csv(PROBE_DATA_PATH)
    pca_df = pd.read_csv(PCA_DATA_PATH)
    
    # Filter probe data (onoff <= 100 for probe-probe correlations)
    probe_df_filtered = probe_df[probe_df['onoff'] <= 100].copy()
    
    # Create probe-level dataset: merge probe data with PCA at probe level
    # IMPORTANT: Must merge on subject_id + task + probe_number because probe_number repeats across tasks
    pca_subset = pca_df[['subject_id', 'task', 'probe_number'] + PCA_COMPONENTS].copy()
    probe_level_df = pd.merge(probe_df_filtered, pca_subset, 
                              on=['subject_id', 'task', 'probe_number'], 
                              how='left')
    
    # CRITICAL: For PCA-probe correlations that match the PCA script,
    # we need to use the EXACT same data from the PCA results file
    # This ensures correlations match the PCA script exactly
    pca_probe_level_df = pca_df[['subject_id', 'valence', 'selfother', 'time', 'onoff', 'confidence'] + PCA_COMPONENTS].copy()
    
    # Aggregate probe dimensions to subject level for psychometric correlations
    probe_agg = probe_df_filtered.groupby('subject_id').agg({
        **{dim: 'mean' for dim in PROBE_DIMENSIONS},
        'age': 'first',
        'group': 'first',
        'gender': 'first',
        'bdi': 'first',
        'rrs_tot': 'first',
        'mwq': 'first',
        'fne': 'first',
        'self-esteem': 'first',
        'ctq_tot': 'first',
        'a_rsq': 'first',
        'sris': 'first'
    }).reset_index()
    
    # Aggregate PCA components to subject level
    pca_agg = pca_df.groupby('subject_id')[PCA_COMPONENTS].mean().reset_index()
    
    # Use LEFT JOIN to keep all probe subjects
    subject_level_df = pd.merge(probe_agg, pca_agg, on='subject_id', how='left')
    
    print(f"Probe-level dataset (all data): {len(probe_level_df)} probe responses from {probe_level_df['subject_id'].nunique()} subjects")
    print(f"PCA-matched dataset (onoff ≤ 50): {len(pca_probe_level_df)} probe responses from {pca_probe_level_df['subject_id'].nunique()} subjects")
    print(f"Subject-level dataset: {len(subject_level_df)} subjects")
    print(f"Groups: {subject_level_df['group'].value_counts().to_dict()}")
    print(f"Age: M={subject_level_df['age'].mean():.1f}, SD={subject_level_df['age'].std():.1f}")
    
    return probe_level_df, pca_probe_level_df, subject_level_df


def get_control_variables(row_var, col_var, available_scales):
    """
    Determine which variables to control for in partial correlation.
    
    Parameters
    ----------
    row_var : str
        Row variable (probe dimension or PCA component)
    col_var : str
        Column variable (probe dimension or psychometric scale)
    available_scales : list
        List of available psychometric scales
        
    Returns
    -------
    list
        Variables to control for
    """
    # Hybrid rule: If correlating with a probe dimension, DO NOT control for anything (Complete)
    if col_var in PROBE_DIMENSIONS:
        return []

    # If correlating with a psychometric scale, DO control for probe dimensions (Partial)
    if col_var in available_scales:
        control_vars = PROBE_DIMENSIONS.copy()
        
        # If the row variable is a probe dimension itself, we can't control for it
        if row_var in PROBE_DIMENSIONS:
            control_vars.remove(row_var)
            
        # If mutual control is enabled: also control for other psychometric scales
        if CONTROL_PSYCHOMETRICS_MUTUALLY:
            psychometric_controls = [scale for scale in available_scales 
                                    if scale != col_var]
            control_vars.extend(psychometric_controls)
            
        return control_vars
        
    return []


def compute_partial_correlations_with_significance(probe_level_df, pca_probe_level_df, subject_level_df):
    """
    Compute partial correlations with proper statistical testing using Pingouin.
    Uses probe-level data for probe-probe correlations.
    Uses PCA-matched data (onoff ≤ 50) for PCA-probe correlations to match PCA script.
    Uses subject-level data for psychometric correlations.
    
    Parameters
    ----------
    probe_level_df : pd.DataFrame
        Probe-level dataset (all data with onoff ≤ 100)
    pca_probe_level_df : pd.DataFrame
        PCA-matched dataset (only rows used for PCA: onoff ≤ 50, no missing valence/selfother/time)
    subject_level_df : pd.DataFrame
        Subject-level dataset
        
    Returns
    -------
    tuple
        (correlation_matrix, p_values, p_fdr_corrected, sample_sizes, results_df)
    """
    print("Computing partial correlations and significance tests...")
    print("  - Using PROBE-LEVEL data (all) for probe-probe correlations")
    print("  - Using PCA-MATCHED data (onoff ≤ 50) for PCA-probe correlations")
    print("  - Using SUBJECT-LEVEL data for psychometric correlations")
    
    available_scales = [var for var in PSYCHOMETRIC_SCALES if var in subject_level_df.columns]
    
    # Keep original matrix structure: probes/PCA in rows, probes/psychometrics in columns
    probe_pca_vars = PROBE_DIMENSIONS + PCA_COMPONENTS
    all_column_vars = PROBE_DIMENSIONS + available_scales
    
    # Initialize matrices
    n_rows = len(probe_pca_vars)
    n_cols = len(all_column_vars)
    
    corr_matrix = np.full((n_rows, n_cols), np.nan)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    n_matrix = np.full((n_rows, n_cols), 0)
    
    results_list = []
    
    # Compute partial correlations
    for i, row_var in enumerate(probe_pca_vars):
        for j, col_var in enumerate(all_column_vars):
            # Choose appropriate dataset based on variable types
            # Use probe-level (all) for probe-probe correlations
            # Use PCA-matched (onoff <= 50) for PCA-probe correlations
            # Use subject-level for psychometric correlations
            
            if row_var in PCA_COMPONENTS:
                # PCA-probe or PCA-psychometric correlation
                if col_var in PROBE_DIMENSIONS:
                    # PCA-probe: use PCA-matched dataset (onoff <= 50)
                    df_to_use = pca_probe_level_df
                    min_sample_base = 30
                else:
                    # PCA-psychometric: use subject-level
                    df_to_use = subject_level_df
                    min_sample_base = 10
            elif col_var in PROBE_DIMENSIONS:
                # Probe-probe: use full probe-level data
                df_to_use = probe_level_df
                min_sample_base = 30
            else:
                # Probe-psychometric: use subject-level data
                df_to_use = subject_level_df
                min_sample_base = 10
            
            # Check if both variables exist in the chosen dataframe
            if row_var not in df_to_use.columns or col_var not in df_to_use.columns:
                continue
            
            # Skip diagonal elements when correlating variable with itself
            if row_var == col_var:
                # Use appropriate dataset for sample size
                if col_var in PROBE_DIMENSIONS:
                    if row_var in probe_level_df.columns:
                        n_samples = probe_level_df[row_var].notnull().sum()
                    else:
                        n_samples = subject_level_df[row_var].notnull().sum()
                else:
                    n_samples = subject_level_df[row_var].notnull().sum()
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                n_matrix[i, j] = n_samples
                continue
            
            # Get control variables
            control_vars = get_control_variables(row_var, col_var, available_scales)
            
            # If no control variables, use regular correlation
            if not control_vars:
                valid_mask = df_to_use[[row_var, col_var]].notnull().all(axis=1)
                n_valid = valid_mask.sum()
                
                if n_valid >= min_sample_base:
                    x = df_to_use.loc[valid_mask, row_var]
                    y = df_to_use.loc[valid_mask, col_var]
                    
                    corr, p_val = spearmanr(x, y)
                    
                    corr_matrix[i, j] = corr
                    p_matrix[i, j] = p_val
                    n_matrix[i, j] = n_valid
                    
                    results_list.append({
                        'row_variable': row_var,
                        'column_variable': col_var,
                        'correlation_type': 'simple',
                        'control_variables': '',
                        'partial_correlation': corr,
                        'p_value': p_val,
                        'sample_size': n_valid,
                        'abs_correlation': abs(corr),
                        'data_level': 'probe' if col_var in PROBE_DIMENSIONS else 'subject'
                    })
                continue
            
            # Check which control variables are available and have data
            available_controls = [var for var in control_vars if var in df_to_use.columns]
            all_vars = [row_var, col_var] + available_controls
            
            # Get valid cases (all variables non-missing)
            valid_mask = df_to_use[all_vars].notnull().all(axis=1)
            n_valid = valid_mask.sum()
            
            # Minimum sample size considering controls
            min_sample_with_controls = len(available_controls) + min_sample_base
            if n_valid >= min_sample_with_controls:
                try:
                    # Use Pingouin for partial correlation
                    df_valid = df_to_use.loc[valid_mask, all_vars].copy()
                    
                    # Compute partial correlation using Pingouin
                    pcorr_result = pg.partial_corr(
                        data=df_valid,
                        x=row_var,
                        y=col_var,
                        covar=available_controls,
                        method='spearman'
                    )
                    
                    corr = pcorr_result['r'].iloc[0]
                    p_val = pcorr_result['p-val'].iloc[0]
                    
                    corr_matrix[i, j] = corr
                    p_matrix[i, j] = p_val
                    n_matrix[i, j] = n_valid

                    results_list.append({
                        'row_variable': row_var,
                        'column_variable': col_var,
                        'correlation_type': 'partial',
                        'control_variables': ', '.join(available_controls),
                        'partial_correlation': corr,
                        'p_value': p_val,
                        'sample_size': n_valid,
                        'abs_correlation': abs(corr),
                        'data_level': 'probe' if col_var in PROBE_DIMENSIONS else 'subject'
                    })
                    
                except Exception as e:
                    print(f"Error computing partial correlation for {row_var} vs {col_var}: {e}")
                    continue

    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=probe_pca_vars, columns=all_column_vars)
    p_df = pd.DataFrame(p_matrix, index=probe_pca_vars, columns=all_column_vars)
    n_df = pd.DataFrame(n_matrix, index=probe_pca_vars, columns=all_column_vars)
    
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction separately for two comparison groups
    if len(results_df) > 0:
        results_df['p_fdr_corrected'] = np.nan
        results_df['significant_fdr'] = False
        results_df['significant_uncorrected'] = results_df['p_value'] < SIGNIFICANCE_THRESHOLD
        
        # Apply FDR correction separately for each row (probe dimension/PCA component)
        # and within each row, separately for probe vs psychometric comparisons
        for probe_var in probe_pca_vars:
            row_mask = results_df['row_variable'] == probe_var
            if row_mask.sum() > 0:
                row_results = results_df[row_mask].copy()
                
                # Within this row, separate probe vs psychometric comparisons
                probe_comparisons = row_results[
                    row_results['column_variable'].isin(PROBE_DIMENSIONS)
                ]
                psychometric_comparisons = row_results[
                    row_results['column_variable'].isin(available_scales)
                ]
                
                # Apply FDR correction separately for each comparison type within this row
                for comparison_group in [probe_comparisons, psychometric_comparisons]:
                    if len(comparison_group) > 0:
                        p_values_group = comparison_group['p_value'].values
                        _, p_fdr_corrected_group, _, _ = multipletests(p_values_group, method='fdr_bh')
                        
                        # Update results_df with FDR-corrected p-values
                        group_indices = comparison_group.index
                        results_df.loc[group_indices, 'p_fdr_corrected'] = p_fdr_corrected_group
                        results_df.loc[group_indices, 'significant_fdr'] = p_fdr_corrected_group < SIGNIFICANCE_THRESHOLD
        
        # Create FDR-corrected p-value matrix
        p_fdr_matrix = np.full((n_rows, n_cols), np.nan)
        for _, row in results_df.iterrows():
            i = list(corr_df.index).index(row['row_variable'])
            j = list(corr_df.columns).index(row['column_variable'])
            p_fdr_matrix[i, j] = row['p_fdr_corrected']
        
        p_fdr_df = pd.DataFrame(p_fdr_matrix, index=probe_pca_vars, columns=all_column_vars)
    else:
        p_fdr_df = p_df.copy()
    
    return corr_df, p_df, p_fdr_df, n_df, results_df


def create_partial_correlation_heatmap(corr_df, p_df, p_fdr_df, n_df, results_df):
    """
    Create the partial correlation heatmap with visual significance coding.
    """
    print("Creating partial correlation visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Use FDR-corrected or uncorrected p-values based on configuration
    p_values_for_viz = p_fdr_df if USE_FDR_CORRECTION else p_df
    correction_label = "FDR-corrected" if USE_FDR_CORRECTION else "uncorrected"
    
    # Create significance mask based on configuration
    significance_mask = p_values_for_viz >= SIGNIFICANCE_THRESHOLD
    
    # Set non-significant correlations to 0 for color mapping
    corr_for_color = corr_df.copy()
    corr_for_color[significance_mask] = 0
    

    
    # Create annotation matrix with correlations and p-values
    annot_matrix = np.full(corr_df.shape, '', dtype=object)
    
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            corr_val = corr_df.iloc[i, j]
            p_uncorr = p_df.iloc[i, j]
            p_fdr = p_fdr_df.iloc[i, j]
            n_val = n_df.iloc[i, j]
            
            # Choose which p-value to display
            p_display = p_fdr if USE_FDR_CORRECTION else p_uncorr
            
            # Show text for ALL cells that have valid correlation data
            if not np.isnan(corr_val):
                # Format correlation value and p-value
                if not np.isnan(p_display):
                    if p_display < 0.001:
                        p_text = "(<.001)"
                    else:
                        p_text = f"({p_display:.3f})"
                    
                    text = f"{corr_val:.2f}\n{p_text}"
                else:
                    text = f"{corr_val:.2f}"
                
                annot_matrix[i, j] = text
    
    # Create the heatmap WITHOUT annot parameter to avoid matplotlib 3.8+ rendering bugs
    mask = np.isnan(corr_df.values)
    
    sns.heatmap(
        corr_for_color,
        annot=False,  # Changed to False - we'll add annotations manually
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
            "label": "Partial Correlation Coefficient (Spearman)"
        },
        mask=mask,
        ax=ax
    )
    
    # Manually add text annotations for matplotlib 3.8+ compatibility
    # This ensures all cells are properly annotated regardless of matplotlib version
    for i in range(corr_df.shape[0]):
        for j in range(corr_df.shape[1]):
            if not mask[i, j]:  # Only add text for non-masked cells
                text_content = annot_matrix[i, j]
                if text_content:  # Only add if there's actual text
                    # Parse the correlation value to determine text color
                    corr_val = corr_for_color.iloc[i, j]
                    
                    # Extract p-value to check significance
                    is_significant = ('*' in text_content)
                    
                    # Set text color based on correlation strength and significance
                    if not is_significant:
                        # Non-significant: black text on white background
                        text_color = 'black'
                    elif abs(corr_val) > 0.3:
                        # Significant and strong: white text on colored background
                        text_color = 'white'
                    else:
                        # Significant but weak: black text
                        text_color = 'black'
                    
                    ax.text(j + 0.5, i + 0.5, text_content,
                           ha='center', va='center',
                           fontsize=10, fontweight='bold',  # Slightly smaller to fit better
                           color=text_color)
    
    # Customize labels
    y_labels = [DIMENSION_LABELS.get(label, label) for label in corr_df.index]
    x_labels = []
    for label in corr_df.columns:
        if label in DIMENSION_LABELS:
            x_labels.append(DIMENSION_LABELS[label])
        elif label in SCALE_LABELS:
            x_labels.append(SCALE_LABELS[label])
        else:
            x_labels.append(label)
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12, fontweight='bold')
    
    # Add title
    title_text = ('Hybrid Correlation Matrix: Complete (Probes) & Partial (Psychometrics)')
    ax.set_title(title_text, fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle with sample info
    n_subjects = n_df.max().max()
    subtitle_text = (f'White cells = non-significant ({correction_label} p ≥ 0.05)')
    plt.figtext(0.5, 0.94, subtitle_text, ha='center', fontsize=14, style='italic')
    
    # Add control variable legend
    control_info = ("Controls: Probes vs Probes/PCA → none (Complete) | "
                   "Probes/PCA vs Psychometric → all probe dimensions (Partial)")
    
    plt.figtext(0.5, 0.91, control_info, ha='center', fontsize=10, 
               style='italic', color='darkblue')
    
    # Add significance legend at the bottom (updated to remove asterisk references)
    legend_text = (f"Significance ({correction_label}): "
                  "White background = non-significant (p ≥ 0.05)   |   "
                  "Color intensity = correlation strength")
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9))
    
    # Adjust layout to accommodate legends
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.12, top=0.88, right=0.95)
    
    # Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_correlation_heatmap.png'), 
               dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_correlation_heatmap.svg'), 
               dpi=DPI, bbox_inches='tight')
    
    plt.close()  # Close figure instead of showing to avoid blocking


def generate_partial_correlation_summary(results_df, df):
    """
    Generate comprehensive summary report for partial correlations.
    """
    print("\\n" + "="*80)
    print("HYBRID CORRELATION ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    # Dataset overview
    print(f"\\nDataset Overview:")
    print(f"- Total subjects: {len(df)}")
    print(f"- Groups: {df['group'].value_counts().to_dict()}")
    print(f"- Age: M = {df['age'].mean():.1f}, SD = {df['age'].std():.1f}, "
          f"Range = {df['age'].min():.0f}-{df['age'].max():.0f}")
    print(f"- Gender: {df['gender'].value_counts().to_dict()}")
    
    # Control variable summary
    print(f"\\nControl Variable Strategy:")
    print(f"- Probe vs Probe / PCA vs Probe → Complete Correlation (No controls)")
    if CONTROL_PSYCHOMETRICS_MUTUALLY:
        print(f"- Probe/PCA vs Psychometric scales → Partial Correlation (Controlling for probes AND other scales)")
    else:
        print(f"- Probe/PCA vs Psychometric scales → Partial Correlation (Controlling for probes only)")
    
    # FDR correction strategy
    print("\\nFDR Correction Strategy:")
    print("- Applied row-wise (separately for each probe dimension/PCA component)")
    print("- Within each row: probe comparisons FDR corrected separately from psychometric comparisons")
    print("- This prevents over-correction across conceptually different comparison types")
    
    # Correlation analysis summary
    print(f"\\nPartial Correlation Analysis Summary:")
    print(f"- Total partial correlations computed: {len(results_df)}")
    print(f"- Simple correlations: {sum(results_df['correlation_type'] == 'simple')}")
    print(f"- Partial correlations: {sum(results_df['correlation_type'] == 'partial')}")
    print(f"- Significant (uncorrected p < 0.05): {sum(results_df['significant_uncorrected'])}")
    print(f"- Significant (FDR corrected p < 0.05): {sum(results_df['significant_fdr'])}")
    
    # Break down by comparison groups
    probe_comps = results_df[results_df['column_variable'].isin(PROBE_DIMENSIONS)]
    psychometric_comps = results_df[~results_df['column_variable'].isin(PROBE_DIMENSIONS)]
    
    print(f"  - Probe dimension comparisons (FDR sig.): {sum(probe_comps['significant_fdr'])}/{len(probe_comps)}")
    print(f"  - Psychometric scale comparisons (FDR sig.): {sum(psychometric_comps['significant_fdr'])}/{len(psychometric_comps)}")
    
    # Show significant partial correlations (FDR corrected)
    sig_results = results_df[results_df['significant_fdr']].copy()
    if len(sig_results) > 0:
        print(f"\\nSignificant Partial Correlations (FDR corrected p < 0.05):")
        sig_results = sig_results.sort_values('abs_correlation', ascending=False)
        for i, row in sig_results.iterrows():
            row_label = DIMENSION_LABELS.get(row['row_variable'], row['row_variable'])
            col_label = SCALE_LABELS.get(row['column_variable'], row['column_variable'].upper())
            control_info = f" | Controls: {row['control_variables']}" if row['control_variables'] else ""
            print(f"  {row_label} ↔ {col_label}: r = {row['partial_correlation']:.3f}, "
                  f"p = {row['p_fdr_corrected']:.3f}, n = {row['sample_size']}{control_info}")
        
        # Effect size distribution
        effect_sizes = sig_results['abs_correlation']
        large_effects = sum(effect_sizes >= 0.5)
        medium_effects = sum((effect_sizes >= 0.3) & (effect_sizes < 0.5))
        small_effects = sum(effect_sizes < 0.3)
        
        print(f"\\nEffect Size Distribution (significant partial correlations):")
        print(f"  Large (|r| ≥ 0.5): {large_effects}")
        print(f"  Medium (0.3 ≤ |r| < 0.5): {medium_effects}")
        print(f"  Small (|r| < 0.3): {small_effects}")
    else:
        print(f"\\nNo partial correlations survived FDR correction.")
    
    # Strongest partial correlations (uncorrected)
    print(f"\\nStrongest Partial Correlations (uncorrected p < 0.05, |r| ≥ 0.3):")
    strong_results = results_df[
        (results_df['significant_uncorrected']) & 
        (results_df['abs_correlation'] >= 0.3)
    ].copy()
    
    if len(strong_results) > 0:
        strong_results = strong_results.sort_values('abs_correlation', ascending=False)
        for i, row in strong_results.head(15).iterrows():
            row_label = DIMENSION_LABELS.get(row['row_variable'], row['row_variable'])
            col_label = SCALE_LABELS.get(row['column_variable'], row['column_variable'].upper())
            fdr_mark = " (FDR sig.)" if row['significant_fdr'] else ""
            corr_type = "Simple" if row['correlation_type'] == 'simple' else "Partial"
            print(f"  {row_label} ↔ {col_label}: r = {row['partial_correlation']:.3f} ({corr_type}), "
                  f"p = {row['p_value']:.3f}, n = {row['sample_size']}{fdr_mark}")
    else:
        print("  No strong partial correlations found.")
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, 'hybrid_correlation_results.csv')
    results_df_sorted = results_df.sort_values(['significant_fdr', 'abs_correlation'], 
                                              ascending=[False, False])
    results_df_sorted.to_csv(results_file, index=False)
    print(f"\\nDetailed results saved to: {results_file}")
    
    return sig_results


def main():
    """
    Run the hybrid correlation analysis.
    """
    print("="*80)
    print("HYBRID CORRELATION ANALYSIS: COMPLETE (PROBES) & PARTIAL (PSYCHOMETRICS)")
    print("="*80)
    
    # Load and prepare data
    probe_level_df, pca_probe_level_df, subject_level_df = load_and_prepare_data()
    
    # Compute partial correlations with significance testing
    corr_df, p_df, p_fdr_df, n_df, results_df = compute_partial_correlations_with_significance(
        probe_level_df, pca_probe_level_df, subject_level_df
    )
    
    # Create visualization
    create_partial_correlation_heatmap(corr_df, p_df, p_fdr_df, n_df, results_df)
    
    # Generate summary report
    sig_results = generate_partial_correlation_summary(results_df, subject_level_df)
    
    print(f"\\n" + "="*80)
    print("HYBRID CORRELATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\\nKey findings:")
    if len(sig_results) > 0:
        print(f"- {len(sig_results)} significant partial correlations (FDR corrected)")
        strongest = sig_results.iloc[0]
        row_label = DIMENSION_LABELS.get(strongest['row_variable'], strongest['row_variable'])
        col_label = SCALE_LABELS.get(strongest['column_variable'], strongest['column_variable'].upper())
        print(f"- Strongest: {row_label} ↔ {col_label} (r = {strongest['partial_correlation']:.3f})")
        print(f"- Controls used: {strongest['control_variables']}")
    else:
        uncorr_sig = sum(results_df['significant_uncorrected'])
        print(f"- No partial correlations survived FDR correction")
        print(f"- {uncorr_sig} partial correlations significant at uncorrected p < 0.05")


if __name__ == "__main__":
    main()

# %%