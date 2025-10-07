#!/usr/bin/env python3
"""
Simple LMM Analysis for On-task vs Off-task comparison

This script performs a simplified Linear Mixed-Effects Model (LMM) analysis
comparing on-task vs off-task conditions for specific markers (a, a_n, t, t_n)
aggregated by frontal and posterior electrode clusters.

Features:
- Loads probe-level data
- Aggregates electrodes by frontal/posterior ROIs
- Runs LMM for specified markers
- Creates raincloud plots for visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import warnings
import ptitprince as pt

# Silence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuration
CSV_FILE = 'results/aggregated_mne_markers/aggregated_mne_markers_onoff_5trials_go_correct_iqr_probe.csv'
OUTPUT_DIR = 'results/simple_lmm_analysis'
MARKERS = ['a', 't']

# Analysis parameters
OUTLIER_THRESHOLD = 3.0  # Z-score threshold for outlier removal
LOG_TRANSFORM = True     # Whether to log-transform data
Z_SCORE_NORMALIZE = True # Whether to z-score normalize data

# ROI definitions
FRONTAL_CHANNELS = ['AF2', 'AF1', 'AFz', 'F1', 'F2','Fz']

POSTERIOR_CHANNELS = ['P1', 'Pz', 'P2', 'CP1', 'CP2', 'CPz']

# Color scheme
COLOR_DICT = {'ontask': '#fa4617', 'offtask': '#281e78'}

def load_and_prepare_data(csv_file):
    """Load and prepare the data for analysis."""
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # Filter for markers of interest
    df = df[df['marker'].isin(MARKERS)].copy()
    
    # Map onoff_label to condition names
    df['condition'] = df['onoff_label'].map({'high': 'ontask', 'low': 'offtask'})
    
    # Add ROI information
    df['roi'] = df['channel'].apply(lambda x: 'frontal' if x in FRONTAL_CHANNELS 
                                   else 'posterior' if x in POSTERIOR_CHANNELS 
                                   else 'other')
    
    # Filter for ROI channels only
    df = df[df['roi'].isin(['frontal', 'posterior'])].copy()
    
    print(f"Data shape after filtering: {df.shape}")
    print(f"Markers: {df['marker'].unique()}")
    print(f"ROIs: {df['roi'].unique()}")
    print(f"Conditions: {df['condition'].unique()}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    
    return df

def remove_outliers_by_marker(df, marker, threshold=3.0):
    """Remove outlier subjects for a specific marker based on z-score threshold."""
    marker_data = df[df['marker'] == marker].copy()
    
    if marker_data.empty:
        return df
    
    # Calculate z-scores for each subject's mean value across all conditions/ROIs
    subject_means = marker_data.groupby('subject_id')['mean'].mean()
    z_scores = np.abs(stats.zscore(subject_means))
    
    # Identify outlier subjects
    outlier_subjects = subject_means[z_scores > threshold].index.tolist()
    
    if outlier_subjects:
        print(f"    Removing {len(outlier_subjects)} outlier subjects for {marker}: {outlier_subjects}")
        # Remove outlier subjects from the entire dataframe for this marker
        df_filtered = df[~((df['marker'] == marker) & (df['subject_id'].isin(outlier_subjects)))]
        return df_filtered
    else:
        print(f"    No outliers detected for {marker}")
        return df

def preprocess_data(df):
    """Apply log transformation and z-score normalization within participant."""
    df_processed = df.copy()
    
    print("Preprocessing data...")
    
    # Log transformation (add small constant to avoid log(0))
    if LOG_TRANSFORM:
        print("  Applying log transformation...")
        min_positive = df_processed[df_processed['mean'] > 0]['mean'].min()
        constant = min_positive / 1000 if min_positive > 0 else 1e-10
        df_processed['mean'] = np.log(df_processed['mean'] + constant)
    
    # Z-score normalization within each participant and marker
    if Z_SCORE_NORMALIZE:
        print("  Applying z-score normalization within participant...")
        df_processed['mean'] = df_processed.groupby(['subject_id', 'marker'])['mean'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
        )
    
    return df_processed

def aggregate_by_roi(df):
    """Aggregate data by ROI (averaging across channels within each ROI)."""
    print("Aggregating data by ROI...")
    
    # Group by subject, marker, roi, condition and average across channels
    agg_df = df.groupby(['subject_id', 'probe_number','marker', 'roi', 'condition']).agg({
        'mean': 'mean',
        'count': 'sum'
    }).reset_index()
    
    print(f"Aggregated data shape: {agg_df.shape}")
    
    # Remove outliers for each marker
    print("Removing outliers by marker...")
    df_no_outliers = agg_df.copy()
    
    for marker in MARKERS:
        df_no_outliers = remove_outliers_by_marker(df_no_outliers, marker, OUTLIER_THRESHOLD)
    
    print(f"Data shape after outlier removal: {df_no_outliers.shape}")
    
    # Apply preprocessing (log transform and z-score normalization)
    df_processed = preprocess_data(df_no_outliers)
    
    return df_processed

def run_lmm_analysis(df, marker, roi):
    """Run LMM analysis for a specific marker and ROI."""
    # Filter data
    data = df[(df['marker'] == marker) & (df['roi'] == roi)].copy()

    
    if data.empty:
        return None
    
    # Check sample sizes
    condition_counts = data['condition'].value_counts()
    if len(condition_counts) < 2 or condition_counts.min() < 5:
        print(f"  Insufficient data for {marker} in {roi} ROI")
        return None
    
    try:
        # Model specifications to evaluate (formula, re_formula description)
        model_specs = [
            ("mean ~ condition", None, "Random intercept"),
            ("mean ~ condition", "1", "Random intercept (explicit)"),
            ("mean ~ condition", "1 + condition", "Random intercept & slope")
        ]

        fitted_models = []  # collect successful fits for comparison

        for formula, re_formula, desc in model_specs:
            try:
                print(f"    Fitting model: {formula} with re_formula: {re_formula}")
                if re_formula is not None:
                    mod = mixedlm(formula, data, groups=data['subject_id'], re_formula=re_formula)
                else:
                    mod = mixedlm(formula, data, groups=data['subject_id'])

                fit_res = mod.fit(method='lbfgs', maxiter=200, disp=False)

                print(f"    Model fit: {fit_res.summary()}")

                fitted_models.append({
                    'result': fit_res,
                    'formula': formula,
                    're_formula': re_formula,
                    'description': desc,
                    'aic': fit_res.aic,
                    'bic': fit_res.bic,
                    'converged': getattr(fit_res, 'converged', True)
                })

            except Exception as e:
                print(f"    Model failed ({desc}): {e}")
                continue

        if not fitted_models:
            print(f"  All model specifications failed for {marker} in {roi} ROI")
            return None

        # Select best model based on lowest AIC among converged models; if none converged, lowest AIC overall
        converged_models = [m for m in fitted_models if m['converged']]
        comparison_pool = converged_models if converged_models else fitted_models
        best_model_info = min(comparison_pool, key=lambda m: m['aic'])
        print(f"    Best model: {best_model_info['description']}")
        best_res = best_model_info['result']

        # Extract statistics from best model
        if 'condition[T.ontask]' in best_res.params:
            coef = best_res.params['condition[T.ontask]']
            pval = best_res.pvalues['condition[T.ontask]']
            tval = best_res.tvalues['condition[T.ontask]']
        elif 'condition[T.offtask]' in best_res.params:
            coef = -best_res.params['condition[T.offtask]']
            pval = best_res.pvalues['condition[T.offtask]']
            tval = -best_res.tvalues['condition[T.offtask]']
        else:
            print(f"  Condition parameter not found for best model ({best_model_info['description']})")
            return None

        # Prepare result dictionary
        res_dict = {
            'marker': marker,
            'roi': roi,
            'coefficient': coef,
            'p_value': pval,
            't_value': tval,
            'n_obs': len(data),
            'n_subjects': data['subject_id'].nunique(),
            'model_used': best_model_info['description'],
            'aic': best_model_info['aic'],
            'bic': best_model_info['bic'],
            'converged': best_model_info['converged']
        }

        # Optional: Save comparison info (AIC/BIC of all models) as JSON-like string
        res_dict['all_model_aic'] = ";".join([
            f"{m['description']}={m['aic']:.1f}" for m in fitted_models
        ])

        return res_dict

    except Exception as e:
        print(f"  LMM fitting error for {marker} in {roi} ROI: {e}")
        return None

def run_all_lmm_analyses(df):
    """Run LMM analyses for all marker-ROI combinations."""
    print("\nRunning LMM analyses...")
    
    results = []
    for marker in MARKERS:
        for roi in ['frontal', 'posterior']:
            print(f"  Analyzing {marker} in {roi} ROI...")
            result = run_lmm_analysis(df, marker, roi)
            if result:
                results.append(result)
    
    if not results:
        print("No successful analyses!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 1:
        _, p_corrected = fdrcorrection(results_df['p_value'].values)
        results_df['p_corrected'] = p_corrected
        results_df['significant'] = results_df['p_corrected'] < 0.05
    else:
        results_df['p_corrected'] = results_df['p_value']
        results_df['significant'] = results_df['p_value'] < 0.05
    
    return results_df

def create_raincloud_plot(df, marker, roi, save_path=None):
    """Create raincloud plot for a specific marker and ROI."""
    # Filter data
    data = df[(df['marker'] == marker) & (df['roi'] == roi)].copy()

    agg_df = data.groupby(['subject_id', 'condition']).agg({
        'mean': 'mean',
        'count': 'sum'
    }).reset_index()
    
    if agg_df.empty:
        print(f"No data for {marker} in {roi} ROI")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create raincloud plot
    pt.RainCloud(
        x='condition', 
        y='mean', 
        data=agg_df,
        palette=[COLOR_DICT['offtask'], COLOR_DICT['ontask']],
        bw=0.2,
        width_viol=0.6,
        ax=ax,
        orient='h',
        alpha=0.65,
        dodge=True,
        pointplot=True,
        move=0.2
    )
    
    # Customize plot
    ax.set_title(f'{marker.upper()} - {roi.title()} ROI\nOn-task vs Off-task Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Mean Amplitude', fontsize=12)
    ax.set_ylabel('Condition', fontsize=12)
    
    # Add statistics text
    stats_text = f"N subjects: {data['subject_id'].nunique()}\n"
    stats_text += f"N observations: {len(data)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path+'.svg', dpi=300, bbox_inches='tight')
        print(f"  Raincloud plot saved: {save_path}")
    
    return fig

def create_all_raincloud_plots(df, output_dir):
    """Create raincloud plots for all marker-ROI combinations."""
    print("\nCreating raincloud plots...")
    
    for marker in MARKERS:
        for roi in ['frontal', 'posterior']:
            print(f"  Creating plot for {marker} in {roi} ROI...")
            
            save_path = os.path.join(output_dir, f'{marker}_{roi}_raincloud.png')
            fig = create_raincloud_plot(df, marker, roi, save_path)
            
            if fig:
                plt.close(fig)

def create_summary_plot(results_df, df, output_dir):
    """Create a summary plot with all results."""
    if results_df.empty:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, marker in enumerate(MARKERS):
        ax = axes[i]
        
        # Get data for this marker
        marker_results = results_df[results_df['marker'] == marker]
        marker_data = df[df['marker'] == marker]

        marker_data = marker_data.groupby(['subject_id', 'condition', 'roi']).agg({
            'mean': 'mean',
            'count': 'sum'
        }).reset_index()
        
        if marker_data.empty:
            ax.text(0.5, 0.5, f'No data for {marker}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{marker.upper()}')
            continue
        
        # Create raincloud plot for both ROIs
        pt.RainCloud(
            x='roi', 
            y='mean', 
            hue='condition',
            data=marker_data,
            palette=[COLOR_DICT['offtask'], COLOR_DICT['ontask']],
            bw=0.2,
            width_viol=0.6,
            ax=ax,
            orient='v',
            alpha=0.65,
            dodge=True,
            pointplot=True,
            move=0.2
        )
        
        # Add significance markers
        for _, row in marker_results.iterrows():
            roi_idx = 0 if row['roi'] == 'frontal' else 1
            if row['significant']:
                ax.text(roi_idx, ax.get_ylim()[1] * 0.9, '*', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_title(f'{marker.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('ROI', fontsize=12)
        ax.set_ylabel('Mean Amplitude', fontsize=12)
    
    plt.tight_layout()
    
    # Save summary plot
    save_path = os.path.join(output_dir, 'summary_raincloud_plots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path = os.path.join(output_dir, 'summary_raincloud_plots.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary plot saved: {save_path}")
    
    return fig

def save_results(results_df, output_dir):
    """Save results to CSV file."""
    if results_df.empty:
        print("No results to save")
        return
    
    # Save full results
    results_file = os.path.join(output_dir, 'lmm_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Results saved: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Preprocessing: Log transform={LOG_TRANSFORM}, Z-score normalize={Z_SCORE_NORMALIZE}")
    print(f"Outlier removal: Z-score threshold={OUTLIER_THRESHOLD}")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        # Significance markers for uncorrected p-values
        sig_uncorr = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        # Significance markers for FDR-corrected p-values  
        sig_corr = "***" if row['p_corrected'] < 0.001 else "**" if row['p_corrected'] < 0.01 else "*" if row['p_corrected'] < 0.05 else ""
        
        # Convergence indicator
        conv_indicator = "✓" if row.get('converged', True) else "✗"
        
        print(f"{row['marker']:>6} {row['roi']:>10}: coef={row['coefficient']:>8.3f}, "
              f"p_uncorr={row['p_value']:>8.3f} {sig_uncorr}, "
              f"p_FDR={row['p_corrected']:>8.3f} {sig_corr} [{conv_indicator}]")
    
    n_significant_uncorr = sum(results_df['p_value'] < 0.05)
    n_significant_fdr = sum(results_df['significant'])
    n_converged = sum(results_df.get('converged', [True] * len(results_df)))
    
    print(f"\nSignificant results (uncorrected p < 0.05): {n_significant_uncorr}/{len(results_df)}")
    print(f"Significant results (FDR-corrected p < 0.05): {n_significant_fdr}/{len(results_df)}")
    print(f"Converged models: {n_converged}/{len(results_df)}")

def create_preprocessing_summary(original_df, processed_df, output_dir):
    """Create a summary of preprocessing steps."""
    summary_file = os.path.join(output_dir, 'preprocessing_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("PREPROCESSING SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Log transformation: {LOG_TRANSFORM}\n")
        f.write(f"  Z-score normalization: {Z_SCORE_NORMALIZE}\n")
        f.write(f"  Outlier threshold: {OUTLIER_THRESHOLD} SD\n\n")
        
        f.write("Data shape changes:\n")
        f.write(f"  Original: {original_df.shape[0]} observations\n")
        f.write(f"  Processed: {processed_df.shape[0]} observations\n")
        f.write(f"  Removed: {original_df.shape[0] - processed_df.shape[0]} observations\n\n")
        
        f.write("Subjects by marker:\n")
        for marker in MARKERS:
            orig_subjects = original_df[original_df['marker'] == marker]['subject_id'].nunique()
            proc_subjects = processed_df[processed_df['marker'] == marker]['subject_id'].nunique()
            f.write(f"  {marker}: {orig_subjects} → {proc_subjects} subjects\n")
        
        f.write("\nData distribution after preprocessing:\n")
        for marker in MARKERS:
            marker_data = processed_df[processed_df['marker'] == marker]['mean']
            if len(marker_data) > 0:
                f.write(f"  {marker}: mean={marker_data.mean():.3f}, std={marker_data.std():.3f}, "
                       f"range=[{marker_data.min():.3f}, {marker_data.max():.3f}]\n")
    
    print(f"Preprocessing summary saved: {summary_file}")

def main():
    """Main function."""
    print("🧠 ENHANCED LMM ANALYSIS: ON-TASK vs OFF-TASK 🧠")
    print("="*60)
    print(f"Configuration:")
    print(f"  Log transformation: {LOG_TRANSFORM}")
    print(f"  Z-score normalization: {Z_SCORE_NORMALIZE}")
    print(f"  Outlier removal threshold: {OUTLIER_THRESHOLD} SD")
    print(f"  Frontal ROI: {len(FRONTAL_CHANNELS)} channels")
    print(f"  Posterior ROI: {len(POSTERIOR_CHANNELS)} channels")
    print("="*60)
    
    # Check input file
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found: {CSV_FILE}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    df = load_and_prepare_data(CSV_FILE)
    original_agg_df = df.groupby(['subject_id', 'probe_number','marker', 'roi', 'condition']).agg({
        'mean': 'mean', 'count': 'sum'
    }).reset_index()
    
    # Aggregate by ROI with preprocessing
    agg_df = aggregate_by_roi(df)
    
    # Create preprocessing summary
    create_preprocessing_summary(original_agg_df, agg_df, OUTPUT_DIR)
    
    # Run LMM analyses
    results_df = run_all_lmm_analyses(agg_df)
    
    # Save results
    save_results(results_df, OUTPUT_DIR)
    
    # Create raincloud plots
    create_all_raincloud_plots(agg_df, OUTPUT_DIR)
    
    # Create summary plot
    fig = create_summary_plot(results_df, agg_df, OUTPUT_DIR)
    if fig:
        plt.close(fig)
    
    print(f"\nAnalysis complete! Results saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main() 