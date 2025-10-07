# %%
"""
Final Probe-Psychometric Correlation Analysis

This script creates a comprehensive correlation heatmap between probe dimensions,
PCA components, and psychometric scales. Non-significant correlations are shown
in white, while significant correlations use the full color scale with p-values
displayed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================

# Data paths
PROBE_DATA_PATH = "results/Behavior/probe_data/probe_level_aggregated_data.csv"
PCA_DATA_PATH = "results/Behavior/probe_data/pca_results.csv"
OUTPUT_DIR = "results/Behavior/probe_data/final_correlation_analysis"

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
FIGURE_SIZE = (20, 12)  # Wider to accommodate more columns
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
USE_FDR_CORRECTION = True  # ← Change this to True for row-wise FDR correction
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

# =============================================================================


def load_and_prepare_data():
    """
    Load and prepare data for correlation analysis.
    
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


def compute_correlations_with_significance(probe_level_df, pca_probe_level_df, subject_level_df):
    """
    Compute correlations with proper statistical testing.
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
    print("Computing correlations and significance tests...")
    print("  - Using PROBE-LEVEL data (all) for probe-probe correlations")
    print("  - Using PCA-MATCHED data (onoff ≤ 50) for PCA-probe correlations")
    print("  - Using SUBJECT-LEVEL data for psychometric correlations")
    
    probe_pca_vars = PROBE_DIMENSIONS + PCA_COMPONENTS
    # Include ONLY probe dimensions (not PCA) AND psychometric scales as columns
    available_scales = [var for var in PSYCHOMETRIC_SCALES if var in subject_level_df.columns]
    all_column_vars = PROBE_DIMENSIONS + available_scales
    
    # Initialize matrices
    n_rows = len(probe_pca_vars)
    n_cols = len(all_column_vars)
    
    corr_matrix = np.full((n_rows, n_cols), np.nan)
    p_matrix = np.full((n_rows, n_cols), np.nan)
    n_matrix = np.full((n_rows, n_cols), 0)
    
    results_list = []
    
    # Compute correlations
    for i, row_var in enumerate(probe_pca_vars):
        for j, col_var in enumerate(all_column_vars):
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
            
            # Choose appropriate dataset based on variable types
            # Use probe-level (all) for probe-probe correlations
            # Use PCA-matched (onoff <= 50) for PCA-probe correlations
            # Use subject-level for psychometric correlations
            
            if row_var in PCA_COMPONENTS:
                # PCA-probe or PCA-psychometric correlation
                if col_var in PROBE_DIMENSIONS:
                    # PCA-probe: use PCA-matched dataset (onoff <= 50)
                    df_to_use = pca_probe_level_df
                    min_sample_size = 30
                else:
                    # PCA-psychometric: use subject-level
                    df_to_use = subject_level_df
                    min_sample_size = 10
            elif col_var in PROBE_DIMENSIONS:
                # Probe-probe: use full probe-level data
                df_to_use = probe_level_df
                min_sample_size = 30
            else:
                # Probe-psychometric: use subject-level data
                df_to_use = subject_level_df
                min_sample_size = 10
            
            # Check if both variables exist in the chosen dataframe
            if row_var not in df_to_use.columns or col_var not in df_to_use.columns:
                continue
                
            # Get valid pairs
            valid_mask = df_to_use[[row_var, col_var]].notnull().all(axis=1)
            n_valid = valid_mask.sum()
            
            if n_valid >= min_sample_size:
                x = df_to_use.loc[valid_mask, row_var]
                y = df_to_use.loc[valid_mask, col_var]
                
                corr, p_val = spearmanr(x, y)
                
                corr_matrix[i, j] = corr
                p_matrix[i, j] = p_val
                n_matrix[i, j] = n_valid

                results_list.append({
                    'probe_dimension': row_var,
                    'psychometric_scale': col_var,
                    'correlation': corr,
                    'p_value': p_val,
                    'sample_size': n_valid,
                    'abs_correlation': abs(corr),
                    'data_level': 'probe' if col_var in PROBE_DIMENSIONS else 'subject'
                })

    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_matrix, 
                          index=probe_pca_vars, 
                          columns=all_column_vars)
    
    p_df = pd.DataFrame(p_matrix,
                       index=probe_pca_vars,
                       columns=all_column_vars)
    
    n_df = pd.DataFrame(n_matrix,
                       index=probe_pca_vars,
                       columns=all_column_vars)
    
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction BY ROW (per probe dimension/PCA component)
    if len(results_df) > 0:
        results_df['p_fdr_corrected'] = np.nan
        results_df['significant_fdr'] = False
        results_df['significant_uncorrected'] = results_df['p_value'] < SIGNIFICANCE_THRESHOLD
        
        # Apply FDR correction separately for each probe dimension/PCA component
        for probe_var in probe_pca_vars:
            # Get results for this specific probe dimension/PCA component
            mask = results_df['probe_dimension'] == probe_var
            if mask.sum() > 0:  # If there are results for this dimension
                p_values_for_dim = results_df.loc[mask, 'p_value'].values
                _, p_fdr_corrected_dim, _, _ = multipletests(p_values_for_dim, 
                                                           method='fdr_bh')
                
                # Store FDR-corrected p-values for this dimension
                results_df.loc[mask, 'p_fdr_corrected'] = p_fdr_corrected_dim
                results_df.loc[mask, 'significant_fdr'] = p_fdr_corrected_dim < SIGNIFICANCE_THRESHOLD
        
        # Create FDR-corrected p-value matrix
        p_fdr_matrix = np.full((n_rows, n_cols), np.nan)
        for _, row in results_df.iterrows():
            i = list(corr_df.index).index(row['probe_dimension'])
            j = list(corr_df.columns).index(row['psychometric_scale'])
            p_fdr_matrix[i, j] = row['p_fdr_corrected']
        
        p_fdr_df = pd.DataFrame(p_fdr_matrix,
                               index=probe_pca_vars,
                               columns=all_column_vars)
    else:
        p_fdr_df = p_df.copy()
    
    return corr_df, p_df, p_fdr_df, n_df, results_df


def create_final_heatmap(corr_df, p_df, p_fdr_df, n_df, results_df):
    """
    Create the final correlation heatmap with visual significance coding.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix
    p_df : pd.DataFrame
        Uncorrected p-values
    p_fdr_df : pd.DataFrame
        FDR-corrected p-values
    n_df : pd.DataFrame
        Sample sizes
    results_df : pd.DataFrame
        Detailed results
    """
    print("Creating final visualization...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Choose which p-values to use based on configuration
    p_values_for_viz = p_fdr_df if USE_FDR_CORRECTION else p_df
    correction_label = "FDR-corrected" if USE_FDR_CORRECTION else "uncorrected"
    
    # Create masked correlation matrix (white for non-significant)
    corr_masked = corr_df.copy()
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
                # Format correlation value
                text = f"r = {corr_val:.3f}"
                
                # Add p-value information
                if not np.isnan(p_display):
                    if p_display < 0.001:
                        text += "\np < .001***"
                    elif p_display < 0.01:
                        text += f"\np = {p_display:.3f}**"
                    elif p_display < 0.05:
                        text += f"\np = {p_display:.3f}*"
                    else:
                        text += f"\np = {p_display:.3f}"
                
                # Add sample size (only if we have valid sample size)
                if n_val > 0:
                    text += f"\n(n={n_val:.0f})"
                
                annot_matrix[i, j] = text
    
    # Create custom colormap: white for non-significant, RdBu_r for significant
    from matplotlib.colors import ListedColormap
    import matplotlib.colors as mcolors
    
    # Get RdBu_r colormap
    rdbu_r = plt.cm.get_cmap('RdBu_r')
    
    # Create custom colormap
    colors_list = []
    n_colors = 256
    
    for i in range(n_colors):
        # Map from -1 to 1
        val = (i / (n_colors - 1)) * 2 - 1
        if abs(val) < 0.05:  # Near zero -> white for non-significant
            colors_list.append((1, 1, 1, 1))  # White
        else:
            colors_list.append(rdbu_r(i / (n_colors - 1)))
    
    custom_cmap = ListedColormap(colors_list)
    
    # Create the heatmap WITHOUT annot parameter to avoid matplotlib 3.8+ rendering bugs
    mask = np.isnan(corr_df.values)
    
    # Use significance-aware correlation matrix for colors
    heatmap = sns.heatmap(
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
            "label": "Spearman Correlation Coefficient"
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
                    
                    # Add text at cell center
                    ax.text(j + 0.5, i + 0.5, text_content,
                           ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color=text_color)
    
    # Customize labels
    y_labels = [DIMENSION_LABELS.get(label, label) for label in corr_df.index]
    # Combine dimension and scale labels for columns
    x_labels = []
    for label in corr_df.columns:
        if label in DIMENSION_LABELS:
            x_labels.append(DIMENSION_LABELS[label])
        elif label in SCALE_LABELS:
            x_labels.append(SCALE_LABELS[label])
        else:
            x_labels.append(label)
    
    ax.set_yticklabels(y_labels, rotation=0, fontsize=12, fontweight='bold')
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=12, fontweight='bold')
    
    # Add title
    ax.set_title('Correlation Matrix: Probe Dimensions & PCA Components vs Probe Dimensions & Psychometric Scales', 
                fontsize=18, fontweight='bold', pad=30)
    
    # Add subtitle with sample info
    plt.figtext(0.5, 0.94, 
               f'N = {n_df.max().max():.0f} subjects | White cells = non-significant ({correction_label} p ≥ 0.05)', 
               ha='center', fontsize=12, style='italic')
    
    # Add significance legend at the bottom
    legend_text = (f"Significance levels ({correction_label} p-values): "
                  "* p < 0.05   ** p < 0.01   *** p < 0.001   |   "
                  "White background = non-significant   |   "
                  "Color intensity = correlation strength")
    
    plt.figtext(0.5, 0.02, legend_text, fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.9))
    
    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.12, top=0.90, right=0.95)
    
    # Save plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_correlation_heatmap.png'), 
               dpi=DPI, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_correlation_heatmap.svg'), 
               dpi=DPI, bbox_inches='tight')
    
    plt.show()


def generate_summary_report(results_df, df):
    """
    Generate comprehensive summary report.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Correlation results
    df : pd.DataFrame
        Original dataset
    """
    print("\n" + "="*80)
    print("FINAL CORRELATION ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    # Dataset overview
    print(f"\nDataset Overview:")
    print(f"- Total subjects: {len(df)}")
    print(f"- Groups: {df['group'].value_counts().to_dict()}")
    print(f"- Age: M = {df['age'].mean():.1f}, SD = {df['age'].std():.1f}, Range = {df['age'].min():.0f}-{df['age'].max():.0f}")
    print(f"- Gender: {df['gender'].value_counts().to_dict()}")
    
    # Correlation analysis summary
    print(f"\nCorrelation Analysis Summary:")
    print(f"- Total correlations computed: {len(results_df)}")
    print(f"- Significant (uncorrected p < 0.05): {sum(results_df['significant_uncorrected'])}")
    print(f"- Significant (FDR corrected p < 0.05): {sum(results_df['significant_fdr'])}")
    
    # Show significant correlations based on configuration
    correction_type = "FDR corrected" if USE_FDR_CORRECTION else "uncorrected"
    significance_col = 'significant_fdr' if USE_FDR_CORRECTION else 'significant_uncorrected'
    p_col = 'p_fdr_corrected' if USE_FDR_CORRECTION else 'p_value'
    
    sig_results = results_df[results_df[significance_col]].copy()
    if len(sig_results) > 0:
        print(f"\nSignificant Correlations ({correction_type} p < 0.05):")
        sig_results = sig_results.sort_values('abs_correlation', ascending=False)
        for i, row in sig_results.iterrows():
            probe_label = DIMENSION_LABELS.get(row['probe_dimension'], row['probe_dimension'])
            scale_label = row['psychometric_scale'].upper()
            print(f"  {probe_label} ↔ {scale_label}: r = {row['correlation']:.3f}, "
                  f"p = {row[p_col]:.3f}, n = {row['sample_size']}")
        
        # Effect size distribution
        effect_sizes = sig_results['abs_correlation']
        large_effects = sum(effect_sizes >= 0.5)
        medium_effects = sum((effect_sizes >= 0.3) & (effect_sizes < 0.5))
        small_effects = sum(effect_sizes < 0.3)
        
        print(f"\nEffect Size Distribution (significant correlations):")
        print(f"  Large (|r| ≥ 0.5): {large_effects}")
        print(f"  Medium (0.3 ≤ |r| < 0.5): {medium_effects}")
        print(f"  Small (|r| < 0.3): {small_effects}")
    else:
        print(f"\nNo correlations survived FDR correction.")
    
    # Strongest correlations (uncorrected)
    print(f"\nStrongest Correlations (uncorrected p < 0.05, |r| ≥ 0.3):")
    strong_results = results_df[
        (results_df['significant_uncorrected']) & 
        (results_df['abs_correlation'] >= 0.3)
    ].copy()
    
    if len(strong_results) > 0:
        strong_results = strong_results.sort_values('abs_correlation', ascending=False)
        for i, row in strong_results.head(15).iterrows():
            probe_label = DIMENSION_LABELS.get(row['probe_dimension'], row['probe_dimension'])
            scale_label = row['psychometric_scale'].upper()
            fdr_mark = " (FDR sig.)" if row['significant_fdr'] else ""
            print(f"  {probe_label} ↔ {scale_label}: r = {row['correlation']:.3f}, "
                  f"p = {row['p_value']:.3f}, n = {row['sample_size']}{fdr_mark}")
    else:
        print("  No strong correlations found.")
    
    # Save detailed results
    results_file = os.path.join(OUTPUT_DIR, 'final_correlation_results.csv')
    results_df_sorted = results_df.sort_values(['significant_fdr', 'abs_correlation'], 
                                              ascending=[False, False])
    results_df_sorted.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    return sig_results


def main():
    """
    Run the final correlation analysis.
    """
    print("="*80)
    print("FINAL PROBE-PSYCHOMETRIC CORRELATION ANALYSIS")
    print("="*80)
    
    # Load and prepare data
    probe_level_df, pca_probe_level_df, subject_level_df = load_and_prepare_data()
    
    # Compute correlations with significance testing
    corr_df, p_df, p_fdr_df, n_df, results_df = compute_correlations_with_significance(
        probe_level_df, pca_probe_level_df, subject_level_df
    )
    
    # Create final visualization
    create_final_heatmap(corr_df, p_df, p_fdr_df, n_df, results_df)
    
    # Generate summary report
    sig_results = generate_summary_report(results_df, subject_level_df)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nKey findings:")
    if len(sig_results) > 0:
        print(f"- {len(sig_results)} significant correlations (FDR corrected)")
        strongest = sig_results.iloc[0]
        probe_label = DIMENSION_LABELS.get(strongest['probe_dimension'], strongest['probe_dimension'])
        scale_label = strongest['psychometric_scale'].upper()
        print(f"- Strongest: {probe_label} ↔ {scale_label} (r = {strongest['correlation']:.3f})")
    else:
        uncorr_sig = sum(results_df['significant_uncorrected'])
        print(f"- No correlations survived FDR correction")
        print(f"- {uncorr_sig} correlations significant at uncorrected p < 0.05")


if __name__ == "__main__":
    main()

# %%
