#%%
# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# Set to True for interactive mode (Jupyter/script in subfolder), False for standalone
INTERACTIVE_MODE = True

# Path configuration based on mode
if INTERACTIVE_MODE:
    # Interactive mode: running from subfolder (e.g., Behavior/PCA/)
    DATA_PATH = '../../results/Behavior/probe_data/probe_level_aggregated_data.csv'
    BASE_OUTPUT_DIR = '../../results/Behavior/probe_data'
else:
    # Standalone mode: running from project root
    DATA_PATH = 'results/Behavior/probe_data/probe_level_aggregated_data.csv'
    BASE_OUTPUT_DIR = 'results/Behavior/probe_data'

# Derived paths
LMM_OUTPUT_DIR = f'{BASE_OUTPUT_DIR}/lmm_analysis'
PLOTS_OUTPUT_DIR = f'{BASE_OUTPUT_DIR}/lmm_plots'
TRAJECTORY_OUTPUT_DIR = f'{BASE_OUTPUT_DIR}/trajectory_analysis'
BASELINE_OUTPUT_DIR = f'{BASE_OUTPUT_DIR}/baseline_corrected_analysis'
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

# PCA Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
from scipy import stats

import statsmodels.formula.api as smf
import os
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


#%%
df = pd.read_csv(DATA_PATH)
df = df[df['onoff'] <=50]

# Select the variables for PCA
pca_variables = ['valence', 'selfother', 'time']
pca_data = df[pca_variables].dropna()

# Standardize the data
scaler = StandardScaler()

pca_data_scaled = scaler.fit_transform(pca_data)

# Scale by participant - group by participant and standardize within each participant
# pca_data_scaled = pca_data.copy()
# for subject_id in df['subject_id'].unique():
#     participant_mask = df['subject_id'] == subject_id
#     participant_data = pca_data[participant_mask]
#     if len(participant_data) > 1:  # Only scale if participant has multiple observations
#         scaler_participant = StandardScaler()
#         pca_data_scaled[participant_mask] = scaler_participant.fit_transform(participant_data)


# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(pca_data_scaled)

# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, 
                     columns=[f'PC{i+1}' for i in range(len(pca_variables))])

# Add original variables for reference
pca_df = pd.concat([pca_df, pca_data.reset_index(drop=True)], axis=1)

# Print PCA information
print("PCA Analysis Results:")
print("=" * 50)
print(f"Number of components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Print component loadings
print("\nComponent Loadings:")
print("=" * 50)
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca_variables))],
    index=pca_variables
)
print(loadings_df)

# Create scree plot using matplotlib/seaborn
plt.style.use('default')
fig_scree, ax = plt.subplots(figsize=(12, 8))

# Define colors consistent with the rest of the pipeline
colors = ['#2E86AB', '#F24236']  # Blue and red from our color scheme

# Plot explained variance ratio
x_vals = list(range(1, len(pca_variables) + 1))
ax.plot(x_vals, pca.explained_variance_ratio_, 'o-', 
        color=colors[0], linewidth=4, markersize=12, 
        label='Explained Variance Ratio', markeredgecolor='white', 
        markeredgewidth=2)

# Plot cumulative explained variance
ax.plot(x_vals, np.cumsum(pca.explained_variance_ratio_), 's-', 
        color=colors[1], linewidth=4, markersize=12, 
        label='Cumulative Explained Variance', markeredgecolor='white', 
        markeredgewidth=2)

# Customize for poster presentation
ax.set_title('PCA Scree Plot', fontsize=24, fontweight='bold', pad=30)
ax.set_xlabel('Principal Component', fontsize=20, fontweight='bold')
ax.set_ylabel('Explained Variance Ratio', fontsize=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
ax.set_xticks(x_vals)

# Enhanced legend
legend = ax.legend(title='', fontsize=16, frameon=True, fancybox=True, 
                  shadow=True, loc='center right')
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Enhanced styling
ax.grid(True, alpha=0.4, linewidth=1.5)
ax.set_facecolor('#FAFAFA')

# Enhance spines
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

# Add percentage labels on points
for i, (x, y) in enumerate(zip(x_vals, pca.explained_variance_ratio_)):
    ax.text(x, y + 0.02, f'{y*100:.1f}%', ha='center', va='bottom', 
           fontsize=14, fontweight='bold', color=colors[0])

plt.tight_layout()
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_scree_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_scree_plot.svg', dpi=300, bbox_inches='tight')
plt.show()

# Create biplot (PC1 vs PC2) using matplotlib/seaborn
plt.style.use('default')
fig_biplot, ax = plt.subplots(figsize=(12, 10))

# Define colors consistent with the rest of the pipeline
point_color = '#2E86AB'  # Blue for data points
loading_colors = ['#F24236', '#A23B72', '#F18F01']  # Red, purple, orange for loadings

# Add data points
ax.scatter(pca_result[:, 0], pca_result[:, 1], 
          c=point_color, alpha=0.6, s=50, edgecolors='white', 
          linewidth=1, label='Data Points')

# Add variable vectors (loadings)
scale_factor = 3  # Scale factor for visibility
for i, var in enumerate(pca_variables):
    # Draw arrow for loading vector
    ax.arrow(0, 0, 
             pca.components_[0, i] * scale_factor,
             pca.components_[1, i] * scale_factor,
             head_width=0.1, head_length=0.1, 
             fc=loading_colors[i % len(loading_colors)], 
             ec=loading_colors[i % len(loading_colors)],
             linewidth=3, alpha=0.8)
    
    # Add variable labels
    ax.text(pca.components_[0, i] * scale_factor * 1.15,
            pca.components_[1, i] * scale_factor * 1.15,
            var.capitalize(), fontsize=16, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                     alpha=0.8, edgecolor=loading_colors[i % len(loading_colors)]))

# Customize for poster presentation
ax.set_title('PCA Biplot (PC1 vs PC2)', fontsize=24, fontweight='bold', pad=30)
ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
              fontsize=18, fontweight='bold')
ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
              fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)

# Add grid and center lines
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.set_facecolor('#FAFAFA')

# Make axes equal for proper representation
ax.set_aspect('equal', adjustable='box')

# Enhanced legend (only for data points, since loadings are labeled)
legend = ax.legend(fontsize=16, frameon=True, fancybox=True, 
                  shadow=True, loc='upper right')
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Enhance spines
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_biplot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_biplot.svg', dpi=300, bbox_inches='tight')
plt.show()

# Create correlation heatmap of original variables with PC scores
corr_with_pc = pca_df[pca_variables + ['PC1', 'PC2', 'PC3']].corr()
pc_corr = corr_with_pc.loc[pca_variables, ['PC1', 'PC2', 'PC3']]

# Print correlation matrix for verification
print("\nCorrelation Matrix (Variables vs PCs):")
print("=" * 50)
print(pc_corr.round(3))

# Create heatmap with seaborn using RdBu_r colormap
plt.style.use('default')
fig_pc_corr, ax = plt.subplots(figsize=(12, 9))

# Create the heatmap WITHOUT annot parameter to avoid matplotlib 3.8+ rendering bugs
heatmap = sns.heatmap(pc_corr, 
                     annot=False,  # Changed to False - we'll add annotations manually
                     cmap='RdBu_r', 
                     center=0,
                     square=False,  # Allow rectangular cells for better text fit
                     linewidths=0.5,
                     cbar_kws={"shrink": 0.8, "aspect": 20, "pad": 0.1},
                     vmin=-1, vmax=1,
                     ax=ax)

# Manually add text annotations for matplotlib 3.8+ compatibility
# This ensures all cells are properly annotated regardless of matplotlib version
for i in range(pc_corr.shape[0]):
    for j in range(pc_corr.shape[1]):
        corr_val = pc_corr.iloc[i, j]
        text_str = f'{corr_val:.2f}'
        
        # Determine text color based on correlation strength
        if abs(corr_val) > 0.5:
            text_color = 'white'
        else:
            text_color = 'black'
        
        # Add text at cell center
        ax.text(j + 0.5, i + 0.5, text_str,
               ha='center', va='center',
               fontsize=12, fontweight='bold',
               color=text_color)

# Customize for poster presentation
ax.set_title('Correlation between Original Variables and Principal Components', 
             fontsize=20, fontweight='bold', pad=30)
ax.set_xlabel('Principal Component', fontsize=18, fontweight='bold')
ax.set_ylabel('Original Variable', fontsize=18, fontweight='bold')

# Enhanced tick styling
ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontweight='bold')
ax.set_yticklabels([var.capitalize() for var in pca_variables], 
                  rotation=0, fontweight='bold')

# Enhance spines
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

plt.tight_layout()
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{BASE_OUTPUT_DIR}/pca_correlation_heatmap.svg', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nPCA Summary:")
print("=" * 50)
print(f"Original variables: {pca_variables}")
print(f"Number of observations: {len(pca_data)}")
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.3f} ({pca.explained_variance_ratio_[0]*100:.1f}%) of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.3f} ({pca.explained_variance_ratio_[1]*100:.1f}%) of variance")
print(f"PC1 + PC2 explain {sum(pca.explained_variance_ratio_[:2]):.3f} ({sum(pca.explained_variance_ratio_[:2])*100:.1f}%) of variance")


# %%

# =============================================================================
# LINEAR MIXED MODEL ANALYSIS FOR PCA COMPONENTS (no external metadata merge)
# =============================================================================

print("\n" + "="*60)
print("LINEAR MIXED MODEL ANALYSIS FOR PCA COMPONENTS")
print("="*60)

def build_subject_task_condition_from_df(df_with_order):
    """Deprecated: CSV already contains inclusion_exclusion; mapping not needed."""
    return {}

def prepare_lmm_data(df_orig, pca_df):
    """
    Prepare data for LMM analysis using columns already present in the probe CSV.
    Requires: subject_id, task, group, inclusion_exclusion.
    """
    # First, identify valid PCA indices (rows without NaN in PCA variables)
    valid_pca_indices = df_orig[pca_variables].dropna().index
    print(f"Valid PCA indices: {len(valid_pca_indices)} out of {len(df_orig)} total rows")
    
    # Create a dataset with only valid PCA rows
    df_valid = df_orig.loc[valid_pca_indices].copy().reset_index(drop=True)
    
    # Ensure required columns exist
    required_cols = ['subject_id', 'task', 'group', 'inclusion_exclusion']
    missing = [c for c in required_cols if c not in df_valid.columns]
    if missing:
        raise KeyError(f"Missing required columns in input CSV: {missing}")
    
    # Normalize task labels
    def normalize_task_label(raw_task):
        t = str(raw_task).strip().lower().replace(' ', '').replace('-', '')
        if 'sart' in t:
            if '1' in t:
                return 'Sart1'
            if '2' in t:
                return 'Sart2'
            if '3' in t:
                return 'Sart3'
            if '4' in t:
                return 'Sart4'
        return raw_task
    df_valid['task'] = df_valid['task'].apply(normalize_task_label)

    # Filter to SART tasks
    TASKS_ALL = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
    df_tasks = df_valid[df_valid['task'].isin(TASKS_ALL)].copy()
    print(f"All tasks data: {df_tasks.shape[0]} rows")

    # Attach PCA components
    df_with_pca = pd.concat([df_tasks.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    
    # Create time_on_task variable (probe_number + 15 * (sart_number - 1))
    # Extract SART number from task name
    df_with_pca['sart_number'] = df_with_pca['task'].str.extract(r'(\d+)').astype(int)
    
    # Calculate time_on_task: probe_number + (15 * (sart_number - 1))
    # This ensures: Sart1 = 1-15, Sart2 = 16-30, Sart3 = 31-45, Sart4 = 46-60
    df_with_pca['time_on_task'] = df_with_pca['probe_number'] + (15 * (df_with_pca['sart_number'] - 1))
    
    print(f"Added time_on_task variable: range {df_with_pca['time_on_task'].min()} to {df_with_pca['time_on_task'].max()}")

    # Drop rows missing inclusion_exclusion or group
    df_with_pca = df_with_pca.dropna(subset=['inclusion_exclusion', 'group'])
    print(f"After removing missing inclusion_exclusion/group: {df_with_pca.shape[0]} rows")

    print(f"Final dataset for LMM: {df_with_pca.shape[0]} rows from {df_with_pca['subject_id'].nunique()} subjects")
    return df_with_pca

def run_lmm_analysis(data, dependent_var, formula, model_name, output_dir):
    """
    Run linear mixed model analysis
    """
    print(f"\n=== FITTING MODEL: {model_name} ===")
    print(f"Formula: {dependent_var} ~ {formula}")
    print(f"Sample size: {len(data)} observations from {data['subject_id'].nunique()} subjects")
    
    try:
        # Fit the model
        full_formula = f"{dependent_var} ~ {formula}"
        model = smf.mixedlm(full_formula, data, groups="subject_id").fit()

        print(model.summary())
        
        print("Model fitted successfully!")
        
        # Extract model fit metrics
        # Calculate n_groups from the data since model doesn't have this attribute
        n_groups = data['subject_id'].nunique()
        
        # Try to get AIC/BIC from model, calculate manually if not available
        aic_value = model.aic if hasattr(model, 'aic') and not pd.isna(model.aic) else None
        bic_value = model.bic if hasattr(model, 'bic') and not pd.isna(model.bic) else None
        
        # Manual calculation if not available from model
        if aic_value is None:
            # AIC = -2*log-likelihood + 2*k (where k is number of parameters)
            k = len(model.params)
            aic_value = -2 * model.llf + 2 * k
            print(f"Note: AIC calculated manually: {aic_value:.3f}")
        
        if bic_value is None:
            # BIC = -2*log-likelihood + k*ln(n)
            k = len(model.params)
            n = model.nobs
            bic_value = -2 * model.llf + k * np.log(n)
            print(f"Note: BIC calculated manually: {bic_value:.3f}")
        
        model_metrics = {
            'aic': aic_value,
            'bic': bic_value,
            'log_likelihood': model.llf,
            'log_likelihood_restricted': model.llf_fe if hasattr(model, 'llf_fe') else None,
            'n_observations': model.nobs,
            'n_groups': n_groups,
            'n_parameters': len(model.params),
            'n_fixed_effects': len(model.fe_params) if hasattr(model, 'fe_params') else None,
            'n_random_effects': len(model.cov_re) if hasattr(model, 'cov_re') else None,
            'converged': model.converged,
            'scale': model.scale if hasattr(model, 'scale') else None,
            'deviance': -2 * model.llf if model.llf is not None else None,
            'rsquared_within': None,  # Not directly available in mixedlm
            'rsquared_between': None  # Not directly available in mixedlm
        }
        
        # Print model fit metrics
        print("\n=== MODEL FIT METRICS ===")
        print(f"AIC: {model_metrics['aic']:.3f}")
        print(f"BIC: {model_metrics['bic']:.3f}")
        print(f"Log-Likelihood: {model_metrics['log_likelihood']:.3f}")
        print(f"N observations: {model_metrics['n_observations']}")
        print(f"N groups: {model_metrics['n_groups']}")
        print(f"Converged: {model_metrics['converged']}")
        if model_metrics['scale'] is not None:
            print(f"Scale: {model_metrics['scale']:.6f}")
        
        # Extract results
        results_df = pd.DataFrame({
            'predictor': model.params.index,
            'estimate': model.params.values,
            'std_error': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'conf_lower': model.conf_int().iloc[:, 0].values,
            'conf_upper': model.conf_int().iloc[:, 1].values
        })
        
        # Add significance flags
        results_df['significant_05'] = results_df['p_value'] < 0.05
        results_df['significant_01'] = results_df['p_value'] < 0.01
        
        # Save results
        results_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_results.csv')
        results_df.to_csv(results_file, index=False)
        
        # Save model metrics
        metrics_df = pd.DataFrame([model_metrics])
        metrics_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save enhanced model summary with metrics
        summary_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(str(model.summary()))
            f.write("\n\n" + "="*60 + "\n")
            f.write("MODEL FIT METRICS\n")
            f.write("="*60 + "\n")
            for key, value in model_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        f.write(f"{key.upper()}: {value:.6f}\n")
                    else:
                        f.write(f"{key.upper()}: {value}\n")
                else:
                    f.write(f"{key.upper()}: Not available\n")
        
        print(f"Results saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")
        print(f"Enhanced summary saved to: {summary_file}")
        
        return results_df, model
        
    except Exception as e:
        print(f"Error fitting model {model_name} for {dependent_var}: {str(e)}")
        
        # Save error info
        error_df = pd.DataFrame({
            'model_name': [f"{model_name}_{dependent_var}"],
            'error_message': [str(e)],
            'timestamp': [datetime.now()],
            'sample_size': [len(data)],
            'n_subjects': [data['subject_id'].nunique()]
        })
        error_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_error.csv')
        error_df.to_csv(error_file, index=False)
        
        return None, None

def analyze_lmm_results(results_df, model_name, dependent_var, output_dir):
    """
    Analyze and summarize LMM results
    """
    if results_df is None:
        return
    
    # Separate fixed effects from random effects
    fixed_effects = results_df[
        (~results_df['predictor'].str.contains('Var', na=False)) & 
        (results_df['predictor'] != 'Intercept')
    ].copy()
    
    random_effects = results_df[
        results_df['predictor'].str.contains('Var', na=False)
    ].copy()
    
    intercept = results_df[results_df['predictor'] == 'Intercept'].copy()
    
    print(f"\n=== RESULTS FOR {model_name} - {dependent_var} ===")
    
    # Show intercept
    if len(intercept) > 0:
        intercept_row = intercept.iloc[0]
        print(f"Intercept: β = {intercept_row['estimate']:.3f}, SE = {intercept_row['std_error']:.3f}, p = {intercept_row['p_value']:.4f}")
    
    # Show fixed effects
    if len(fixed_effects) > 0:
        print(f"\nFixed Effects:")
        print(f"Total fixed effects tested: {len(fixed_effects)}")
        
        # Apply Bonferroni correction within this component for fixed effects only
        fixed_effects['p_value_bonferroni'] = multipletests(fixed_effects['p_value'], method='bonferroni')[1]
        fixed_effects['significant_corrected'] = fixed_effects['p_value_bonferroni'] < 0.05
        
        # Show all fixed effects with their results
        for _, row in fixed_effects.iterrows():
            sig_mark = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            corrected_sig_mark = " (corrected)" if row['significant_corrected'] else ""
            print(f"- {row['predictor']}: β = {row['estimate']:.3f}, SE = {row['std_error']:.3f}, t = {row['t_value']:.2f}, p = {row['p_value']:.4f}{sig_mark}{corrected_sig_mark}")
        
        # Summarize significant results
        sig_uncorrected = fixed_effects[fixed_effects['p_value'] < 0.05]
        sig_corrected = fixed_effects[fixed_effects['significant_corrected']]
        
        print(f"\nSignificant fixed effects (p < 0.05): {len(sig_uncorrected)}")
        print(f"Significant fixed effects (Bonferroni corrected): {len(sig_corrected)}")
        
        # Save significant results
        if len(sig_uncorrected) > 0:
            sig_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_significant_results.csv')
            sig_uncorrected.to_csv(sig_file, index=False)
        
        if len(sig_corrected) > 0:
            corrected_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_significant_corrected.csv')
            sig_corrected.to_csv(corrected_file, index=False)
    else:
        print("No fixed effects found (intercept-only model)")
        fixed_effects = pd.DataFrame()  # Empty dataframe for return
    
    # Show random effects
    if len(random_effects) > 0:
        print(f"\nRandom Effects:")
        for _, row in random_effects.iterrows():
            var_name = row['predictor'].replace(' Var', '')
            print(f"- {var_name} variance: {row['estimate']:.3f} (SE = {row['std_error']:.3f})")
    
    return fixed_effects

# Create output directory for LMM results
lmm_output_dir = LMM_OUTPUT_DIR
os.makedirs(lmm_output_dir, exist_ok=True)

# Prepare data for LMM analysis
print("Preparing data for LMM analysis...")
df_lmm = prepare_lmm_data(df, pca_df)

df_lmm.to_csv(f'{BASE_OUTPUT_DIR}/pca_results.csv', index=False)
print("\nPCA PCA results saved to 'pca_results.csv'")

print(f"\nDataset overview:")
print(f"- Total observations: {len(df_lmm)}")
print(f"- Unique subjects: {df_lmm['subject_id'].nunique()}")
group_counts = df_lmm.groupby('group')['subject_id'].nunique()
group_obs = df_lmm['group'].value_counts()
for group in group_counts.index:
    print(f"- {group}: {group_counts[group]} participants, {group_obs[group]} total observations")
print(f"- Inclusion/Exclusion distribution: {df_lmm['inclusion_exclusion'].value_counts().to_dict()}")

# Create IE-only subset for IE-specific analyses (Sart2/Sart4 only)
df_lmm_ie = df_lmm[df_lmm['inclusion_exclusion'].isin(['inclusion', 'exclusion'])].copy()
print(f"\nIE-only subset: {len(df_lmm_ie)} observations from {df_lmm_ie['subject_id'].nunique()} subjects")

distribution_offtask = df_lmm.groupby(['subject_id', 'group']).count().reset_index()
# sns.violinplot(x='group', y='probe_number', data=distribution_offtask)
pt.RainCloud(x="group", y="probe_number", hue="group", data=distribution_offtask, 
            bw=0.2, width_viol=0.8, alpha=0.7, dodge=True,
            pointplot=True, move=0.15)
plt.show()

[# %%
# Model 1: Group effect (Controls vs Risk of Depression)
print("\n" + "="*60)
print("MODEL 1: PC ~ GROUP")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_group, model_group = run_lmm_analysis(
        df_lmm, pc, 'group', f'group_effect', lmm_output_dir
    )
    
    if results_group is not None:
        analyze_lmm_results(results_group, 'group_effect', pc, lmm_output_dir)

# %%
# Model 2: Inclusion/Exclusion effect
print("\n" + "="*60)
print("MODEL 2: PC ~ INCLUSION/EXCLUSION")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_ie, model_ie = run_lmm_analysis(
        df_lmm_ie, pc, 'inclusion_exclusion', f'inclusion_exclusion_effect', lmm_output_dir
    )
    
    if results_ie is not None:
        analyze_lmm_results(results_ie, 'inclusion_exclusion_effect', pc, lmm_output_dir)

# %%
# Model 3: Interaction effect (Group × Inclusion/Exclusion)
print("\n" + "="*60)
print("MODEL 3: PC ~ GROUP * INCLUSION/EXCLUSION")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_interaction, model_interaction = run_lmm_analysis(
        df_lmm_ie, pc, 'group * inclusion_exclusion', f'group_ie_interaction', lmm_output_dir
    )
    
    if results_interaction is not None:
        analyze_lmm_results(results_interaction, 'group_ie_interaction', pc, lmm_output_dir)

# %%
# Model 4: Time on task effect
print("\n" + "="*60)
print("MODEL 4: PC ~ TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_time, model_time = run_lmm_analysis(
        df_lmm, pc, 'time_on_task', f'time_on_task_effect', lmm_output_dir
    )
    
    if results_time is not None:
        analyze_lmm_results(results_time, 'time_on_task_effect', pc, lmm_output_dir)

# %%
# Model 5: Group + Time on task
print("\n" + "="*60)
print("MODEL 5: PC ~ GROUP + TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_group_time, model_group_time = run_lmm_analysis(
        df_lmm, pc, 'group + time_on_task', f'group_time_additive', lmm_output_dir
    )
    
    if results_group_time is not None:
        analyze_lmm_results(results_group_time, 'group_time_additive', pc, lmm_output_dir)

# %%
# Model 6: Group × Time on task interaction
print("\n" + "="*60)
print("MODEL 6: PC ~ GROUP * TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_group_time_int, model_group_time_int = run_lmm_analysis(
        df_lmm, pc, 'group * time_on_task', f'group_time_interaction', lmm_output_dir
    )
    
    if results_group_time_int is not None:
        analyze_lmm_results(results_group_time_int, 'group_time_interaction', pc, lmm_output_dir)

# %%
# Model 7: Inclusion/Exclusion + Time on task
print("\n" + "="*60)
print("MODEL 7: PC ~ INCLUSION/EXCLUSION + TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_ie_time, model_ie_time = run_lmm_analysis(
        df_lmm_ie, pc, 'inclusion_exclusion + time_on_task', f'ie_time_additive', lmm_output_dir
    )
    
    
    if results_ie_time is not None:
        analyze_lmm_results(results_ie_time, 'ie_time_additive', pc, lmm_output_dir)

# %%
# Model 8: Full model with all factors
print("\n" + "="*60)
print("MODEL 8: PC ~ GROUP * INCLUSION/EXCLUSION + TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_full, model_full = run_lmm_analysis(
        df_lmm_ie, pc, 'group * inclusion_exclusion + time_on_task', f'full_model_with_time', lmm_output_dir
    )
    
    if results_full is not None:
        analyze_lmm_results(results_full, 'full_model_with_time', pc, lmm_output_dir)

# %%
# Model 9: Inclusion/Exclusion × Time on task interaction
print("\n" + "="*60)
print("MODEL 9: PC ~ INCLUSION/EXCLUSION * TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_ie_time_int, model_ie_time_int = run_lmm_analysis(
        df_lmm_ie, pc, 'inclusion_exclusion * time_on_task', f'ie_time_interaction', lmm_output_dir
    )
    
    if results_ie_time_int is not None:
        analyze_lmm_results(results_ie_time_int, 'ie_time_interaction', pc, lmm_output_dir)

# %%
# Model 10: Group × Inclusion/Exclusion × Time on task (three-way interaction)
print("\n" + "="*60)
print("MODEL 10: PC ~ GROUP * INCLUSION/EXCLUSION * TIME_ON_TASK")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_three_way, model_three_way = run_lmm_analysis(
        df_lmm_ie, pc, 'group * inclusion_exclusion * time_on_task', f'three_way_interaction', lmm_output_dir
    )
    
    if results_three_way is not None:
        analyze_lmm_results(results_three_way, 'three_way_interaction', pc, lmm_output_dir)

# %%
# Model 11: Full model with Group × Time interaction
print("\n" + "="*60)
print("MODEL 11: PC ~ GROUP * TIME_ON_TASK + INCLUSION/EXCLUSION")
print("="*60)

# Run for each principal component
for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n--- Analyzing {pc} ---")
    results_group_time_ie, model_group_time_ie = run_lmm_analysis(
        df_lmm_ie, pc, 'group * time_on_task + inclusion_exclusion', f'group_time_int_plus_ie', lmm_output_dir
    )
    
    if results_group_time_ie is not None:
        analyze_lmm_results(results_group_time_ie, 'group_time_int_plus_ie', pc, lmm_output_dir)

# %%
# Create comprehensive summary report
print("\n" + "="*60)
print("CREATING LMM SUMMARY REPORT")
print("="*60)

def create_lmm_summary_report(output_dir, pca_info):
    """
    Create a comprehensive summary report of all LMM analyses
    """
    
    # Collect all results files
    results_files = [f for f in os.listdir(output_dir) if f.endswith('_results.csv')]
    
    report_lines = [
        "# Linear Mixed Model Analysis of PCA Components - Summary Report",
        "",
        f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Sample:** {len(df_lmm)} observations from {df_lmm['subject_id'].nunique()} subjects",
        "",
        "## PCA Information",
        f"**Variables used:** {', '.join(pca_variables)}",
        f"**PC1 explained variance:** {pca_info['explained_variance_ratio'][0]:.3f} ({pca_info['explained_variance_ratio'][0]*100:.1f}%)",
        f"**PC2 explained variance:** {pca_info['explained_variance_ratio'][1]:.3f} ({pca_info['explained_variance_ratio'][1]*100:.1f}%)",
        f"**PC3 explained variance:** {pca_info['explained_variance_ratio'][2]:.3f} ({pca_info['explained_variance_ratio'][2]*100:.1f}%)",
        f"**Total explained variance (PC1-3):** {sum(pca_info['explained_variance_ratio']):.3f} ({sum(pca_info['explained_variance_ratio'])*100:.1f}%)",
        "",
        "## Component Loadings",
        "### PC1 Loadings:",
        f"- Valence: {loadings_df.loc['valence', 'PC1']:.3f}",
        f"- Self-Other: {loadings_df.loc['selfother', 'PC1']:.3f}",
        f"- Time: {loadings_df.loc['time', 'PC1']:.3f}",
        "",
        "### PC2 Loadings:",
        f"- Valence: {loadings_df.loc['valence', 'PC2']:.3f}",
        f"- Self-Other: {loadings_df.loc['selfother', 'PC2']:.3f}",
        f"- Time: {loadings_df.loc['time', 'PC2']:.3f}",
        "",
        "### PC3 Loadings:",
        f"- Valence: {loadings_df.loc['valence', 'PC3']:.3f}",
        f"- Self-Other: {loadings_df.loc['selfother', 'PC3']:.3f}",
        f"- Time: {loadings_df.loc['time', 'PC3']:.3f}",
        "",
        "## Group Distribution", 
        f"- **Controls:** {sum(df_lmm['group'] == 'Controls')} ({sum(df_lmm['group'] == 'Controls')/len(df_lmm)*100:.1f}%)",
        f"- **Risk of Depression:** {sum(df_lmm['group'] == 'Risk of Depression')} ({sum(df_lmm['group'] == 'Risk of Depression')/len(df_lmm)*100:.1f}%)",
        "",
        "## Inclusion/Exclusion Distribution",
        f"- **Inclusion:** {sum(df_lmm['inclusion_exclusion'] == 'inclusion')} ({sum(df_lmm['inclusion_exclusion'] == 'inclusion')/len(df_lmm)*100:.1f}%)",
        f"- **Exclusion:** {sum(df_lmm['inclusion_exclusion'] == 'exclusion')} ({sum(df_lmm['inclusion_exclusion'] == 'exclusion')/len(df_lmm)*100:.1f}%)",
        "",
        "## Models Tested",
        "1. **Group Effect:** PC ~ Group (Controls vs Risk of Depression)",
        "2. **Inclusion/Exclusion Effect:** PC ~ Inclusion/Exclusion",  
        "3. **Interaction Effect:** PC ~ Group × Inclusion/Exclusion",
        "",
        "## Statistical Method",
        "- **Model:** Linear Mixed Models (LMM) with random intercepts by subject",
        "- **Clustering:** By subject ID to account for repeated measures",
        "- **Multiple Comparisons:** Bonferroni correction applied within each component",
        "",
        "## Key Findings"
    ]
    
    # Summarize findings from each model and component
    model_names = ['group_effect', 'inclusion_exclusion_effect', 'group_ie_interaction']
    model_titles = ['Group Effect', 'Inclusion/Exclusion Effect', 'Group × Inclusion/Exclusion Interaction']
    components = ['PC1', 'PC2', 'PC3']
    
    for model_name, model_title in zip(model_names, model_titles):
        report_lines.extend([
            f"",
            f"### {model_title}"
        ])
        
        any_significant = False
        for pc in components:
            results_file = os.path.join(output_dir, f'{model_name}_{pc}_results.csv')
            if os.path.exists(results_file):
                results = pd.read_csv(results_file)
                # Only look at fixed effects (exclude random effects and intercept)
                fixed_effects = results[
                    (~results['predictor'].str.contains('Var', na=False)) & 
                    (results['predictor'] != 'Intercept')
                ]
                
                if len(fixed_effects) > 0:
                    # Apply Bonferroni correction
                    fixed_effects['p_value_bonferroni'] = multipletests(fixed_effects['p_value'], method='bonferroni')[1]
                    fixed_effects['significant_corrected'] = fixed_effects['p_value_bonferroni'] < 0.05
                    
                    n_sig_uncorrected = sum(fixed_effects['p_value'] < 0.05)
                    n_sig_corrected = sum(fixed_effects['significant_corrected'])
                    
                    report_lines.append(f"**{pc}:** {n_sig_uncorrected} significant fixed effects (uncorrected), {n_sig_corrected} significant fixed effects (corrected)")
                    
                    # List significant effects
                    sig_effects = fixed_effects[fixed_effects['significant_corrected']]
                    if len(sig_effects) > 0:
                        any_significant = True
                        for _, row in sig_effects.iterrows():
                            report_lines.append(f"  - {row['predictor']}: β = {row['estimate']:.3f}, p = {row['p_value_bonferroni']:.4f}")
                else:
                    report_lines.append(f"**{pc}:** No fixed effects (intercept-only model)")
        
        if not any_significant:
            report_lines.append("- **No significant effects after correction**")
    
    report_lines.extend([
        "",
        "## Interpretation",
        "- **PC1:** Captures the primary dimension of mind-wandering variation",
        "- **PC2:** Captures the secondary dimension of mind-wandering variation", 
        "- **PC3:** Captures the tertiary dimension of mind-wandering variation",
        "",
        "Significant effects indicate that group membership and/or inclusion/exclusion conditions",
        "are associated with different patterns along the principal component dimensions.",
        "",
        "## Files Generated",
        "- `*_results.csv` - Complete model results for each component",
        "- `*_significant_results.csv` - Uncorrected significant effects",
        "- `*_significant_corrected.csv` - Bonferroni-corrected significant effects", 
        "- `*_summary.txt` - Detailed model output",
        "- `lmm_summary_report.md` - This comprehensive report"
    ])
    
    # Save report
    report_file = os.path.join(output_dir, 'lmm_summary_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"LMM summary report saved to: {report_file}")

# Create the summary report
pca_info = {
    'explained_variance_ratio': pca.explained_variance_ratio_
}

create_lmm_summary_report(lmm_output_dir, pca_info)

print("\n" + "="*60)
print("LMM ANALYSIS COMPLETE")
print("="*60)
print(f"All results saved to: {lmm_output_dir}")
print("\nGenerated files:")
for file in sorted(os.listdir(lmm_output_dir)):
    print(f"- {file}")

# %%

# =============================================================================
# VISUALIZATION OF LMM RESULTS
# =============================================================================


print("\n" + "="*60)
print("CREATING VISUALIZATIONS FOR LMM RESULTS")
print("="*60)

# Create output directory for plots
plots_output_dir = PLOTS_OUTPUT_DIR
os.makedirs(plots_output_dir, exist_ok=True)

def create_raincloud_plot(data, x_var, y_var, hue_var, title, filename, figsize=(14, 10)):
    """
    Create a raincloud plot for visualizing distributions and group comparisons
    """
    # Set poster-friendly style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=figsize)

    data = data.copy()
    # Only aggregate numeric columns we need (PC components and original variables)
    numeric_cols = ['PC1', 'PC2', 'PC3', 'valence', 'selfother', 'time']
    grouping_cols = ['subject_id', 'group', 'inclusion_exclusion']
    
    # Select only the columns we need and aggregate
    cols_to_keep = grouping_cols + [col for col in numeric_cols if col in data.columns]
    data = data[cols_to_keep].groupby(grouping_cols).mean().reset_index()
    
    # Define colors - more vibrant and distinct for poster
    if hue_var == 'group':
        palette = {'Controls': '#2E86AB', 'Risk of Depression': '#F24236'}
        hue_order = ['Controls', 'Risk of Depression']
    else:  # inclusion_exclusion
        palette = {'inclusion': '#A23B72', 'exclusion': '#F18F01'}
        hue_order = ['inclusion', 'exclusion']
    
    # Create raincloud plot
    pt.RainCloud(x=x_var, y=y_var, hue=hue_var, data=data, 
                palette=palette, hue_order=hue_order,
                bw=0.2, width_viol=0.8, alpha=0.7, dodge=True,
                pointplot=True, move=0.2, ax=ax)
    
    # Customize plot for poster presentation
    ax.set_title(title, fontsize=24, fontweight='bold', pad=30)
    ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=20, fontweight='bold')
    ax.set_ylabel(y_var, fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    ax.tick_params(axis='x', rotation=0)
    
    # Enhanced legend
    legend = ax.legend(title=hue_var.replace('_', '/').title(), 
                      title_fontsize=18, fontsize=16, 
                      frameon=True, fancybox=True, shadow=True,
                      loc='upper right')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_facecolor('#FAFAFA')
    
    # Add sample sizes with better styling
    for i, level in enumerate(data[x_var].unique()):
        for j, hue_level in enumerate(hue_order):
            subset = data[(data[x_var] == level) & (data[hue_var] == hue_level)]
            n = len(subset)
            # Position text based on subplot layout
            y_pos = ax.get_ylim()[1] * 0.92 - (j * 0.06 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
            ax.text(i + (j-0.5)*0.2, y_pos, f'n={n}', ha='center', va='top', 
                   fontsize=14, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Enhance spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_dir, filename), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_output_dir, filename + '.svg'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_interaction_plot(data, pc_var, filename, figsize=(18, 6)):
    """
    Create interaction plot showing group × inclusion/exclusion effects
    """
    # Set poster-friendly style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Define consistent ordering and colors for inclusion/exclusion
    ie_order = ['inclusion', 'exclusion']
    group_colors = {'Controls': '#2E86AB', 'Risk of Depression': '#F24236'}
    ie_colors = {'inclusion': '#A23B72', 'exclusion': '#F18F01'}
    
    # Plot 1: Main effects side by side
    ax1 = axes[0]
    # Group effect
    group_means = data.groupby('group')[pc_var].agg(['mean', 'sem']).reset_index()
    x_pos = [0, 1]
    colors = [group_colors[group] for group in group_means['group']]
    
    bars1 = ax1.bar(x_pos, group_means['mean'], yerr=group_means['sem'], 
                   color=colors, alpha=0.8, capsize=8, width=0.6, 
                   edgecolor='black', linewidth=2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(group_means['group'], rotation=0, fontsize=16, fontweight='bold')
    ax1.set_ylabel(pc_var, fontsize=18, fontweight='bold')
    ax1.set_title('Group Effect', fontsize=20, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_facecolor('#FAFAFA')
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, group_means['mean']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + group_means['sem'].max()*0.15,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Plot 2: Inclusion/Exclusion effect
    ax2 = axes[1]
    # Ensure consistent ordering of inclusion/exclusion
    ie_means = data.groupby('inclusion_exclusion')[pc_var].agg(['mean', 'sem']).reindex(ie_order).reset_index()
    x_pos = [0, 1]
    colors = [ie_colors[condition] for condition in ie_means['inclusion_exclusion']]
    
    bars2 = ax2.bar(x_pos, ie_means['mean'], yerr=ie_means['sem'], 
                   color=colors, alpha=0.8, capsize=8, width=0.6,
                   edgecolor='black', linewidth=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ie_means['inclusion_exclusion'], fontsize=16, fontweight='bold')
    ax2.set_ylabel(pc_var, fontsize=18, fontweight='bold')
    ax2.set_title('Inclusion/Exclusion Effect', fontsize=20, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.set_facecolor('#FAFAFA')
    ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars2, ie_means['mean']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + ie_means['sem'].max()*0.15,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Plot 3: Interaction plot
    ax3 = axes[2]
    
    # Create interaction plot with consistent ordering
    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        ie_means = group_data.groupby('inclusion_exclusion')[pc_var].agg(['mean', 'sem']).reindex(ie_order).reset_index()
        
        color = group_colors[group]
        x_positions = [0, 1]
        
        ax3.errorbar(x_positions, ie_means['mean'], yerr=ie_means['sem'], 
                    marker='o', linewidth=4, markersize=12, capsize=8, capthick=3,
                    label=group, color=color, alpha=0.9)
        
        # Add value labels
        for x, y, sem in zip(x_positions, ie_means['mean'], ie_means['sem']):
            ax3.text(x, y + sem + (ax3.get_ylim()[1] - ax3.get_ylim()[0])*0.03, 
                    f'{y:.3f}', ha='center', va='bottom', fontsize=14, 
                    fontweight='bold', color=color)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(ie_order, fontsize=16, fontweight='bold')
    ax3.set_ylabel(pc_var, fontsize=18, fontweight='bold')
    ax3.set_title('Group × Inclusion/Exclusion Interaction', fontsize=20, fontweight='bold', pad=20)
    ax3.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    
    # Enhanced legend
    legend = ax3.legend(title='Group', title_fontsize=16, fontsize=14, 
                       frameon=True, fancybox=True, shadow=True,
                       loc='upper right')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    ax3.grid(True, alpha=0.3, linewidth=1.5)
    ax3.set_facecolor('#FAFAFA')
    
    # Enhance spines for all subplots
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(plots_output_dir, filename), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_output_dir, filename.replace('.png', '.svg')), dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_time_on_task_plot(data, pc_var, filename, figsize=(18, 8)):
    """
    Create plots showing PC scores as a function of time on task
    """
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Define colors
    group_colors = ['#2E86AB', '#F24236']
    
    # Plot 1: Time on task by group
    ax1 = axes[0]
    
    for i, group in enumerate(['Controls', 'Risk of Depression']):
        group_data = data[data['group'] == group]
        
        # Create scatterplot with trend line
        ax1.scatter(group_data['time_on_task'], group_data[pc_var], 
                   alpha=0.6, color=group_colors[i], s=50, 
                   edgecolors='white', linewidth=1, label=group)
        
        # Add trend line using numpy polyfit
        if len(group_data) > 1:
            z = np.polyfit(group_data['time_on_task'], group_data[pc_var], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(group_data['time_on_task'].min(), 
                                group_data['time_on_task'].max(), 100)
            ax1.plot(x_trend, p(x_trend), color=group_colors[i], 
                    linewidth=3, alpha=0.8)
    
    ax1.set_xlabel('Time on Task (Cumulative Probes)', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel(f'{pc_var} Score', fontsize=16, fontweight='bold')
    ax1.set_title(f'{pc_var} vs Time on Task by Group', 
                 fontsize=18, fontweight='bold', pad=20)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.4)
    
    # Plot 2: Binned time on task analysis
    ax2 = axes[1]
    
    # Create time bins (early, middle, late)
    data_copy = data.copy()
    data_copy['time_bin'] = pd.cut(data_copy['time_on_task'], 
                                   bins=3, labels=['Early', 'Middle', 'Late'])
    
    # Create boxplot by time bin and group
    import seaborn as sns
    sns.boxplot(data=data_copy, x='time_bin', y=pc_var, hue='group', 
               ax=ax2, palette=group_colors, linewidth=2)
    
    ax2.set_xlabel('Time Period', fontsize=16, fontweight='bold')
    ax2.set_ylabel(f'{pc_var} Score', fontsize=16, fontweight='bold')
    ax2.set_title(f'{pc_var} by Time Period and Group', 
                 fontsize=18, fontweight='bold', pad=20)
    ax2.legend(title='Group', fontsize=12, title_fontsize=14)
    ax2.grid(True, alpha=0.4)
    
    # Style both plots
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_facecolor('#FAFAFA')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(plots_output_dir, filename), 
               dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_output_dir, 
               filename.replace('.png', '.svg')), 
               dpi=300, bbox_inches='tight')
    plt.show()


def create_descriptive_stats_table(data, pc_vars):
    """
    Create descriptive statistics table for each PC by group and condition
    """
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS FOR PRINCIPAL COMPONENTS")
    print("="*80)
    
    stats_list = []
    
    for pc in pc_vars:
        print(f"\n{pc} Statistics:")
        print("-" * 50)
        
        # Overall statistics
        overall_stats = data[pc].describe()
        print(f"Overall: Mean = {overall_stats['mean']:.3f}, SD = {overall_stats['std']:.3f}, "
              f"Min = {overall_stats['min']:.3f}, Max = {overall_stats['max']:.3f}")
        
        # By group
        print("\nBy Group:")
        group_stats = data.groupby('group')[pc].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
        print(group_stats)
        
        # By inclusion/exclusion
        print("\nBy Inclusion/Exclusion:")
        ie_stats = data.groupby('inclusion_exclusion')[pc].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
        print(ie_stats)
        
        # By group × inclusion/exclusion
        print("\nBy Group × Inclusion/Exclusion:")
        interaction_stats = data.groupby(['group', 'inclusion_exclusion'])[pc].agg(['count', 'mean', 'std']).round(3)
        print(interaction_stats)
        
        # Store for summary table
        for group in data['group'].unique():
            for ie in data['inclusion_exclusion'].unique():
                subset = data[(data['group'] == group) & (data['inclusion_exclusion'] == ie)]
                if len(subset) > 0:
                    stats_list.append({
                        'PC': pc,
                        'Group': group,
                        'Inclusion_Exclusion': ie,
                        'N': len(subset),
                        'Mean': subset[pc].mean(),
                        'SD': subset[pc].std(),
                        'SE': subset[pc].sem()
                    })
    
    # Create and save summary table
    stats_df = pd.DataFrame(stats_list)
    stats_file = os.path.join(plots_output_dir, 'descriptive_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"\nDescriptive statistics saved to: {stats_file}")
    
    return stats_df

def run_statistical_tests(data, pc_vars):
    """
    Run statistical tests for group and inclusion/exclusion differences
    """
    print("\n" + "="*80)
    print("STATISTICAL TESTS FOR GROUP DIFFERENCES")
    print("="*80)
    
    results_list = []
    
    for pc in pc_vars:
        print(f"\n{pc} Statistical Tests:")
        print("-" * 50)
        
        # Test for group differences
        controls = data[data['group'] == 'Controls'][pc].dropna()
        risk = data[data['group'] == 'Risk of Depression'][pc].dropna()
        
        # Independent t-test for groups
        t_stat, p_val = stats.ttest_ind(controls, risk)
        effect_size = (controls.mean() - risk.mean()) / np.sqrt(((len(controls)-1)*controls.var() + (len(risk)-1)*risk.var()) / (len(controls) + len(risk) - 2))
        
        print(f"Group Comparison (Controls vs Risk of Depression):")
        print(f"  t({len(controls) + len(risk) - 2}) = {t_stat:.3f}, p = {p_val:.4f}")
        print(f"  Cohen's d = {effect_size:.3f}")
        print(f"  Controls: M = {controls.mean():.3f}, SD = {controls.std():.3f}, N = {len(controls)}")
        print(f"  Risk: M = {risk.mean():.3f}, SD = {risk.std():.3f}, N = {len(risk)}")
        
        results_list.append({
            'PC': pc,
            'Comparison': 'Group',
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'group1_mean': controls.mean(),
            'group1_sd': controls.std(),
            'group1_n': len(controls),
            'group2_mean': risk.mean(),
            'group2_sd': risk.std(),
            'group2_n': len(risk)
        })
        
        # Test for inclusion/exclusion differences
        inclusion = data[data['inclusion_exclusion'] == 'inclusion'][pc].dropna()
        exclusion = data[data['inclusion_exclusion'] == 'exclusion'][pc].dropna()
        
        # Paired t-test (since it's within-subjects)
        # First, get matched pairs
        matched_data = []
        for subject in data['subject_id'].unique():
            subj_data = data[data['subject_id'] == subject]
            inc_vals = subj_data[subj_data['inclusion_exclusion'] == 'inclusion'][pc]
            exc_vals = subj_data[subj_data['inclusion_exclusion'] == 'exclusion'][pc]
            
            if len(inc_vals) > 0 and len(exc_vals) > 0:
                matched_data.append({
                    'subject_id': subject,
                    'inclusion': inc_vals.mean(),
                    'exclusion': exc_vals.mean()
                })
        
        if len(matched_data) > 0:
            matched_df = pd.DataFrame(matched_data)
            t_stat_paired, p_val_paired = stats.ttest_rel(matched_df['inclusion'], matched_df['exclusion'])
            effect_size_paired = (matched_df['inclusion'].mean() - matched_df['exclusion'].mean()) / matched_df[['inclusion', 'exclusion']].diff(axis=1)['exclusion'].std()
            
            print(f"\nInclusion/Exclusion Comparison (Paired t-test):")
            print(f"  t({len(matched_df) - 1}) = {t_stat_paired:.3f}, p = {p_val_paired:.4f}")
            print(f"  Cohen's d = {effect_size_paired:.3f}")
            print(f"  Inclusion: M = {matched_df['inclusion'].mean():.3f}, SD = {matched_df['inclusion'].std():.3f}")
            print(f"  Exclusion: M = {matched_df['exclusion'].mean():.3f}, SD = {matched_df['exclusion'].std():.3f}")
            print(f"  N pairs = {len(matched_df)}")
            
            results_list.append({
                'PC': pc,
                'Comparison': 'Inclusion_Exclusion',
                't_statistic': t_stat_paired,
                'p_value': p_val_paired,
                'effect_size': effect_size_paired,
                'group1_mean': matched_df['inclusion'].mean(),
                'group1_sd': matched_df['inclusion'].std(),
                'group1_n': len(matched_df),
                'group2_mean': matched_df['exclusion'].mean(),
                'group2_sd': matched_df['exclusion'].std(),
                'group2_n': len(matched_df)
            })
    
    # Save results
    results_df = pd.DataFrame(results_list)
    results_file = os.path.join(plots_output_dir, 'statistical_tests_results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nStatistical test results saved to: {results_file}")
    
    return results_df

# %%
# Generate all visualizations with comprehensive 2x3 grid
print("Creating visualizations for PC components...")

pc_components = ['PC1', 'PC2', 'PC3']

# Define consistent ordering and colors
GROUP_ORDER = ['Controls', 'Risk of Depression']
IE_ORDER = ['inclusion', 'exclusion']
GROUP_COLORS = ['#2E86AB', '#F24236']
IE_COLORS = ['#A23B72', '#F18F01']

def create_comprehensive_pc_plot(df_lmm, df_lmm_ie, pc_var, output_dir):
    """
    Create comprehensive 2x3 grid plot for a PC component.
    
    Grid layout:
    - Row 1: Raincloud (Group) | Raincloud (I/E) | Interaction
    - Row 2: Time by Group | Time by I/E | SART Trajectories
    """
    plt.style.use("default")
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # =========================================================================
    # ROW 1: DISTRIBUTION COMPARISONS
    # =========================================================================

    # Plot 1 (0,0): Raincloud for group comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_agg_group = df_lmm.groupby(["subject_id", "group"])[pc_var].mean().reset_index()
    n_participants_by_group = df_agg_group.groupby("group")["subject_id"].nunique().to_dict()
    
    pt.RainCloud(
        x="group",
        y=pc_var,
        data=df_agg_group,
        palette=GROUP_COLORS,
        order=GROUP_ORDER,
        bw=0.2,
        width_viol=0.6,
        alpha=0.7,
        dodge=True,
        pointplot=True,
        move=-0.1,
        ax=ax1,
    )
    ax1.set_title(
        f"{pc_var}: Group Effect",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Add sample sizes
    for i, group in enumerate(GROUP_ORDER):
        n = n_participants_by_group.get(group, 0)
        ax1.text(
            i, ax1.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 2 (0,1): Raincloud for inclusion/exclusion
    ax2 = fig.add_subplot(gs[0, 1])
    df_agg_ie = df_lmm_ie.groupby(["subject_id", "inclusion_exclusion"])[pc_var].mean().reset_index()
    n_participants_by_ie = df_agg_ie.groupby("inclusion_exclusion")["subject_id"].nunique().to_dict()
    
    pt.RainCloud(
        x="inclusion_exclusion",
        y=pc_var,
        data=df_agg_ie,
        palette=IE_COLORS,
        order=IE_ORDER,
        bw=0.2,
        width_viol=0.6,
        alpha=0.7,
        dodge=True,
        pointplot=True,
        move=-0.1,
        ax=ax2,
    )
    ax2.set_title(
        f"{pc_var}: Inclusion/Exclusion Effect",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax2.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Add sample sizes
    for i, condition in enumerate(IE_ORDER):
        n = n_participants_by_ie.get(condition, 0)
        ax2.text(
            i, ax2.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 3 (0,2): Interaction plot (Group × I/E)
    ax3 = fig.add_subplot(gs[0, 2])
    for group in df_lmm_ie["group"].dropna().unique():
        group_data = df_lmm_ie[df_lmm_ie["group"] == group]
        if len(group_data) == 0:
            continue
        ie_means = (
            group_data.groupby("inclusion_exclusion")[pc_var]
            .agg(["mean", "sem"])
            .reindex(IE_ORDER)
            .reset_index()
        )
        n_participants_group = group_data["subject_id"].nunique()
        color = GROUP_COLORS[0] if group == GROUP_ORDER[0] else GROUP_COLORS[1]
        x_positions = [0, 1]
        ax3.errorbar(
            x_positions,
            ie_means["mean"],
            yerr=ie_means["sem"],
            marker="o",
            linewidth=3,
            markersize=10,
            capsize=5,
            capthick=2,
            label=f"{group} (n={n_participants_group})",
            color=color,
            alpha=0.9,
        )
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(IE_ORDER, fontsize=14, fontweight="bold")
    ax3.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax3.set_title(
        f"{pc_var}: Group × I/E Interaction",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax3.legend(fontsize=12, title_fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # ROW 2: TIME-ON-TASK TRAJECTORIES
    # =========================================================================

    # Plot 4 (1,0): Time-on-task by group
    ax4 = fig.add_subplot(gs[1, 0])
    if 'time_on_task' in df_lmm.columns:
        for i, group in enumerate(GROUP_ORDER):
            if group in df_lmm["group"].values:
                group_data = df_lmm[df_lmm["group"] == group]
                time_group_agg = group_data.groupby("time_on_task")[pc_var].agg(["mean", "sem"]).reset_index()
                color = GROUP_COLORS[i]
                ax4.errorbar(
                    time_group_agg["time_on_task"],
                    time_group_agg["mean"],
                    yerr=time_group_agg["sem"],
                    marker="o",
                    linewidth=2.5,
                    markersize=6,
                    capsize=3,
                    alpha=0.8,
                    color=color,
                    label=group,
                )
        ax4.set_xlabel("Time on Task (Probe Number)", fontsize=14, fontweight="bold")
        ax4.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
        ax4.set_title(
            f"{pc_var}: Time-on-Task by Group",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax4.legend(fontsize=12, loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Time-on-task data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)

    # Plot 5 (1,1): Time-on-task by inclusion/exclusion (full trajectory)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'time_on_task' in df_lmm_ie.columns:
        for i, condition in enumerate(IE_ORDER):
            if condition in df_lmm_ie["inclusion_exclusion"].values:
                ie_data = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == condition]
                time_ie_agg = ie_data.groupby("time_on_task")[pc_var].agg(["mean", "sem"]).reset_index()
                color = IE_COLORS[i]
                ax5.errorbar(
                    time_ie_agg["time_on_task"],
                    time_ie_agg["mean"],
                    yerr=time_ie_agg["sem"],
                    marker="s",
                    linewidth=2.5,
                    markersize=6,
                    capsize=3,
                    alpha=0.8,
                    color=color,
                    label=condition.capitalize(),
                )
        ax5.set_xlabel("Time on Task (Probe Number)", fontsize=14, fontweight="bold")
        ax5.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
        ax5.set_title(
            f"{pc_var}: Time-on-Task by I/E",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax5.legend(fontsize=12, loc='best')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Time-on-task data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)

    # Plot 6 (1,2): SART Mean Trajectories by Group and Order
    ax6 = fig.add_subplot(gs[1, 2])
    if 'task' in df_lmm.columns and 'order (IE/EI)' in df_lmm.columns:
        sart_order_list = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        x_positions = [1, 2, 3, 4]
        order_styles = {'IE': '-', 'EI': '--'}
        
        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in df_lmm["group"].values:
                continue
            group_data = df_lmm[df_lmm["group"] == group]
            color = GROUP_COLORS[group_idx]
            
            for order in ['IE', 'EI']:
                if order not in group_data['order (IE/EI)'].values:
                    continue
                order_data = group_data[group_data['order (IE/EI)'] == order]
                n_subjects = order_data['subject_id'].nunique()
                
                means = []
                sems = []
                for sart in sart_order_list:
                    sart_data = order_data[order_data['task'] == sart]
                    if len(sart_data) > 0:
                        means.append(sart_data[pc_var].mean())
                        sems.append(sart_data[pc_var].sem())
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)
                
                linestyle = order_styles[order]
                label = f"{group} - {order} (n={n_subjects})"
                
                line = ax6.errorbar(
                    x_positions,
                    means,
                    yerr=sems,
                    marker='o',
                    linewidth=2.5,
                    markersize=8,
                    linestyle=linestyle,
                    capsize=5,
                    capthick=2,
                    alpha=0.8,
                    color=color,
                    label=label,
                )
        
        ax6.set_xticks(x_positions)
        ax6.set_xticklabels(sart_order_list, fontsize=12, fontweight='bold')
        ax6.set_xlabel("SART Task", fontsize=14, fontweight="bold")
        ax6.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
        ax6.set_title(
            f"{pc_var}: SART Trajectory by Group & Order\n(Solid=IE, Dashed=EI)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        # Create legend with only lines (no markers)
        handles, labels = ax6.get_legend_handles_labels()
        new_handles = []
        for handle in handles:
            if hasattr(handle, 'get_children'):
                lines = [child for child in handle.get_children() if hasattr(child, 'set_marker')]
                if lines:
                    line = lines[0]
                    line.set_marker('')
                    line.set_markersize(0)
                    new_handles.append(line)
            else:
                handle.set_marker('')
                handle.set_markersize(0)
                new_handles.append(handle)
        ax6.legend(new_handles, labels, fontsize=11, loc='best')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Order data not available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pc_var}_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{pc_var}_comprehensive_analysis.svg'), dpi=300, bbox_inches='tight')
    plt.show()

# Generate comprehensive plots for each PC
print("\nCreating comprehensive 2x3 grid plots for each PC component...")
for pc in pc_components:
    print(f"Creating comprehensive plot for {pc}...")
    create_comprehensive_pc_plot(df_lmm, df_lmm_ie, pc, plots_output_dir)

# 5. Generate descriptive statistics
print("\n5. Generating descriptive statistics...")
descriptive_stats = create_descriptive_stats_table(df_lmm, pc_components)

# 6. Run statistical tests
print("\n6. Running statistical tests...")
statistical_results = run_statistical_tests(df_lmm, pc_components)

print(f"\n" + "="*60)
print("VISUALIZATION COMPLETE")
print("="*60)
print(f"All plots and statistics saved to: {plots_output_dir}")
print("\nGenerated files:")
for file in sorted(os.listdir(plots_output_dir)):
    print(f"- {file}")

# %%

# =============================================================================
# TRAJECTORY ANALYSIS: PC COMPONENTS ACROSS TASKS AND TIME
# =============================================================================

print("\n" + "="*60)
print("TRAJECTORY ANALYSIS: PC COMPONENTS ACROSS TASKS")
print("="*60)

def prepare_trajectory_data(df_orig, pca_df):
    """
    Prepare data for trajectory analysis including all tasks (Sart1, Sart2, Sart3, Sart4)
    """
    # First, identify valid PCA indices (rows without NaN in PCA variables)
    valid_pca_indices = df_orig[pca_variables].dropna().index
    print(f"Valid PCA indices: {len(valid_pca_indices)} out of {len(df_orig)} total rows")
    
    # Create a dataset with only valid PCA rows
    df_valid_pca = df_orig.loc[valid_pca_indices].copy().reset_index(drop=True)
    
    # Add PCA components to the valid data and carry metadata directly
    df_with_pca = pd.concat([df_valid_pca.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)
    df_with_metadata = df_with_pca
    
    # Ensure inclusion_exclusion exists in input CSV
    TASKS_ALL = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
    if 'inclusion_exclusion' not in df_with_metadata.columns:
        raise KeyError("Input CSV must include 'inclusion_exclusion'")
    
    # Filter for all relevant tasks
    df_all_tasks = df_with_metadata[df_with_metadata['task'].isin(TASKS_ALL)].copy()
    print(f"All tasks data: {df_all_tasks.shape[0]} rows")
    
    # Create condition mapping for all tasks using existing inclusion_exclusion
    def map_condition(row):
        if row['task'] in ['Sart1', 'Sart3']:
            return 'baseline'
        return row['inclusion_exclusion']
    df_all_tasks['condition'] = df_all_tasks.apply(map_condition, axis=1)
    
    # Remove unknown conditions
    df_all_tasks = df_all_tasks[df_all_tasks['condition'] != 'unknown']
    print(f"After removing unknown conditions: {df_all_tasks.shape[0]} rows")
    
    # Create task order (1, 2, 3, 4)
    task_order_map = {'Sart1': 1, 'Sart2': 2, 'Sart3': 3, 'Sart4': 4}
    df_all_tasks['task_order'] = df_all_tasks['task'].map(task_order_map)
    
    # Remove rows where group information is missing
    df_all_tasks = df_all_tasks.dropna(subset=['group'])
    print(f"Final trajectory dataset: {df_all_tasks.shape[0]} rows from {df_all_tasks['subject_id'].nunique()} subjects")
    
    return df_all_tasks

def create_trajectory_plots(data, output_dir):
    """
    Create comprehensive trajectory plots for PC components
    """
    print("\nCreating trajectory plots...")
    
    pc_components = ['PC1', 'PC2', 'PC3']
    
    # 1. Mean trajectories across tasks
    print("1. Creating mean trajectory plots...")
    
    # Set poster-friendly style
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Define colors for groups
    group_colors = {'Controls': '#2E86AB', 'Risk of Depression': '#F24236'}
    
    for i, pc in enumerate(pc_components):
        ax = axes[i]
        
        # Calculate means by task order and group
        trajectory_means = data.groupby(['task_order', 'group'])[pc].agg(['mean', 'sem']).reset_index()
        
        # Plot trajectories for each group
        for group in data['group'].unique():
            group_data = trajectory_means[trajectory_means['group'] == group]
            color = group_colors[group]
            
            # Main trajectory line
            ax.errorbar(group_data['task_order'], group_data['mean'], 
                       yerr=group_data['sem'], marker='o', linewidth=5, 
                       markersize=15, capsize=10, capthick=3, label=group, 
                       color=color, alpha=0.9, markeredgecolor='white', 
                       markeredgewidth=2)
            
            # Add value labels with better styling
            for x, y, sem in zip(group_data['task_order'], group_data['mean'], group_data['sem']):
                ax.text(x, y + sem + 0.05, f'{y:.3f}', ha='center', va='bottom', 
                       fontsize=16, fontweight='bold', color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                               alpha=0.8, edgecolor=color, linewidth=2))
        
        # Customize plot for poster presentation
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Sart1\n(Baseline)', 'Sart2\n(Inc/Exc)', 
                           'Sart3\n(Baseline)', 'Sart4\n(Inc/Exc)'],
                          fontsize=16, fontweight='bold')
        ax.set_ylabel(f'{pc} Score', fontsize=20, fontweight='bold')
        ax.set_title(f'{pc} Trajectory Across Tasks', fontsize=24, fontweight='bold', pad=30)
        ax.tick_params(axis='both', which='major', labelsize=16, width=3, length=8)
        
        # Enhanced legend
        legend = ax.legend(title='Group', title_fontsize=18, fontsize=16, 
                          frameon=True, fancybox=True, shadow=True,
                          loc='upper right')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Enhanced grid and background
        ax.grid(True, alpha=0.4, linewidth=2, linestyle='-')
        ax.set_facecolor('#FAFAFA')
        
        # Add vertical lines to separate baseline from experimental tasks
        ax.axvline(1.5, linestyle='--', color='gray', alpha=0.6, linewidth=3)
        ax.axvline(2.5, linestyle='--', color='gray', alpha=0.6, linewidth=3)
        ax.axvline(3.5, linestyle='--', color='gray', alpha=0.6, linewidth=3)
        
        # Add phase labels
        ax.text(1.25, ax.get_ylim()[1]*0.95, 'Baseline', ha='center', va='top', 
               fontsize=14, fontweight='bold', style='italic', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
        ax.text(2.25, ax.get_ylim()[1]*0.95, 'Experimental', ha='center', va='top', 
               fontsize=14, fontweight='bold', style='italic',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))
        ax.text(3.25, ax.get_ylim()[1]*0.95, 'Baseline', ha='center', va='top', 
               fontsize=14, fontweight='bold', style='italic',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
        ax.text(3.75, ax.get_ylim()[1]*0.95, 'Experimental', ha='center', va='top', 
               fontsize=14, fontweight='bold', style='italic',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))
        
        # Enhance spines
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color('black')
    
    plt.tight_layout(pad=4.0)
    plt.savefig(os.path.join(output_dir, 'PC_trajectories_by_group.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'PC_trajectories_by_group.svg'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Individual subject trajectories (spaghetti plot)
    print("2. Creating individual subject trajectory plots...")
    
    # Set poster-friendly style
    plt.style.use('default')
    fig, axes = plt.subplots(3, 2, figsize=(12, 24))
    
    # Define colors
    group_colors = {'Controls': '#2E86AB', 'Risk of Depression': '#F24236'}
    ie_colors = {'inclusion': '#A23B72', 'exclusion': '#F18F01'}
    
    for i, pc in enumerate(pc_components):
        # By group
        ax1 = axes[i, 0]
        for group in data['group'].unique():
            group_data = data[data['group'] == group]
            color = group_colors[group]
            
            # Individual subject lines with better styling
            for subject_id in group_data['subject_id'].unique():
                subj_data = group_data[group_data['subject_id'] == subject_id]
                subj_means = subj_data.groupby('task_order')[pc].mean()
                
                ax1.plot(subj_means.index, subj_means.values, 
                        color=color, alpha=0.4, linewidth=1.5)
            
            # Calculate group means and confidence intervals
            group_stats = group_data.groupby('task_order')[pc].agg(['mean', 'sem', 'count']).reset_index()
            
            # Calculate 95% confidence intervals
            confidence_level = 0.95
            alpha = 1 - confidence_level
            group_stats['ci_lower'] = group_stats['mean'] - stats.t.ppf(1 - alpha/2, group_stats['count'] - 1) * group_stats['sem']
            group_stats['ci_upper'] = group_stats['mean'] + stats.t.ppf(1 - alpha/2, group_stats['count'] - 1) * group_stats['sem']
            
            # Add confidence interval shading
            ax1.fill_between(group_stats['task_order'], 
                           group_stats['ci_lower'], 
                           group_stats['ci_upper'],
                           color=color, alpha=0.2, label=f'{group} (95% CI)')
            
            # Add group mean with enhanced styling
            ax1.plot(group_stats['task_order'], group_stats['mean'], 
                    color=color, linewidth=6, label=f'{group} (mean)', alpha=0.95,
                    marker='o', markersize=12, markeredgecolor='white', 
                    markeredgewidth=2)
        
        # Customize plot for poster
        ax1.set_xticks([1, 2, 3, 4])
        ax1.set_xticklabels(['Sart1', 'Sart2', 'Sart3', 'Sart4'], 
                           fontsize=16, fontweight='bold')
        ax1.set_ylabel(f'{pc} Score', fontsize=18, fontweight='bold')
        ax1.set_title(f'{pc} Individual Trajectories by Group', 
                     fontsize=20, fontweight='bold', pad=20)
        ax1.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        
        # Enhanced legend
        legend = ax1.legend(title='Group', title_fontsize=16, fontsize=14, 
                           frameon=True, fancybox=True, shadow=True,
                           loc='upper right')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax1.grid(True, alpha=0.4, linewidth=1.5)
        ax1.set_facecolor('#FAFAFA')
        
        # Add vertical separators
        ax1.axvline(1.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        ax1.axvline(2.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        ax1.axvline(3.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        
        # By inclusion/exclusion order
        ax2 = axes[i, 1]
        
        # Get subjects with clear inclusion/exclusion order
        for order in ['inclusion', 'exclusion']:
            order_subjects = []
            for subject_id in data['subject_id'].unique():
                subj_data = data[data['subject_id'] == subject_id]
                sart2_condition = subj_data[subj_data['task'] == 'Sart2']['condition']
                if len(sart2_condition) > 0 and sart2_condition.iloc[0] == order:
                    order_subjects.append(subject_id)
            
            color = ie_colors[order]
            
            # Individual subject lines with better styling
            for subject_id in order_subjects:
                subj_data = data[data['subject_id'] == subject_id]
                subj_means = subj_data.groupby('task_order')[pc].mean()
                
                ax2.plot(subj_means.index, subj_means.values, 
                        color=color, alpha=0.4, linewidth=1.5)
            
            # Add order mean with enhanced styling and confidence intervals
            if order_subjects:
                order_data = data[data['subject_id'].isin(order_subjects)]
                
                # Calculate group means and confidence intervals
                order_stats = order_data.groupby('task_order')[pc].agg(['mean', 'sem', 'count']).reset_index()
                
                # Calculate 95% confidence intervals
                confidence_level = 0.95
                alpha = 1 - confidence_level
                order_stats['ci_lower'] = order_stats['mean'] - stats.t.ppf(1 - alpha/2, order_stats['count'] - 1) * order_stats['sem']
                order_stats['ci_upper'] = order_stats['mean'] + stats.t.ppf(1 - alpha/2, order_stats['count'] - 1) * order_stats['sem']
                
                # Add confidence interval shading
                ax2.fill_between(order_stats['task_order'], 
                               order_stats['ci_lower'], 
                               order_stats['ci_upper'],
                               color=color, alpha=0.2, label=f'{order} first (95% CI)')
                
                # Add mean line
                ax2.plot(order_stats['task_order'], order_stats['mean'], 
                        color=color, linewidth=6, label=f'{order} first (mean)', alpha=0.95,
                        marker='o', markersize=12, markeredgecolor='white', 
                        markeredgewidth=2)
        
        # Customize plot for poster
        ax2.set_xticks([1, 2, 3, 4])
        ax2.set_xticklabels(['Sart1', 'Sart2', 'Sart3', 'Sart4'], 
                           fontsize=16, fontweight='bold')
        ax2.set_ylabel(f'{pc} Score', fontsize=18, fontweight='bold')
        ax2.set_title(f'{pc} Individual Trajectories by Order', 
                     fontsize=20, fontweight='bold', pad=20)
        ax2.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
        
        # Enhanced legend
        legend = ax2.legend(title='Order', title_fontsize=16, fontsize=14, 
                           frameon=True, fancybox=True, shadow=True,
                           loc='upper right')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax2.grid(True, alpha=0.4, linewidth=1.5)
        ax2.set_facecolor('#FAFAFA')
        
        # Add vertical separators
        ax2.axvline(1.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        ax2.axvline(2.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        ax2.axvline(3.5, linestyle='--', color='gray', alpha=0.6, linewidth=2)
        
        # Enhance spines for both plots
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'PC_individual_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'PC_individual_trajectories.svg'), dpi=300, bbox_inches='tight')
    plt.show()

def analyze_trajectory_changes(data, output_dir):
    """
    Analyze changes from baseline to experimental conditions
    """
    print("\n" + "="*60)
    print("TRAJECTORY CHANGE ANALYSIS")
    print("="*60)
    
    pc_components = ['PC1', 'PC2', 'PC3']
    
    change_data = []
    
    for subject_id in data['subject_id'].unique():
        subj_data = data[data['subject_id'] == subject_id]
        if len(subj_data) == 0:
            continue
            
        group = subj_data['group'].iloc[0]
        
        # Get baseline average (Sart1 + Sart3)
        baseline_data = subj_data[subj_data['condition'] == 'baseline']
        
        # Get experimental data (Sart2 + Sart4)
        inclusion_data = subj_data[subj_data['condition'] == 'inclusion']
        exclusion_data = subj_data[subj_data['condition'] == 'exclusion']
        
        if len(baseline_data) > 0:
            for pc in pc_components:
                baseline_mean = baseline_data[pc].mean()
                
                # Calculate changes
                if len(inclusion_data) > 0:
                    inclusion_mean = inclusion_data[pc].mean()
                    inclusion_change = inclusion_mean - baseline_mean
                    
                    change_data.append({
                        'subject_id': subject_id,
                        'group': group,
                        'pc': pc,
                        'condition': 'inclusion',
                        'baseline_score': baseline_mean,
                        'experimental_score': inclusion_mean,
                        'change_score': inclusion_change
                    })
                
                if len(exclusion_data) > 0:
                    exclusion_mean = exclusion_data[pc].mean()
                    exclusion_change = exclusion_mean - baseline_mean
                    
                    change_data.append({
                        'subject_id': subject_id,
                        'group': group,
                        'pc': pc,
                        'condition': 'exclusion',
                        'baseline_score': baseline_mean,
                        'experimental_score': exclusion_mean,
                        'change_score': exclusion_change
                    })
    
    change_df = pd.DataFrame(change_data)
    
    if len(change_df) > 0:
        # Save change data
        change_file = os.path.join(output_dir, 'trajectory_change_scores.csv')
        change_df.to_csv(change_file, index=False)
        print(f"Change scores saved to: {change_file}")
        
        # Plot change scores
        plt.style.use('default')
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Define colors
        group_colors = ['#2E86AB', '#F24236']
        
        for i, pc in enumerate(pc_components):
            ax = axes[i]
            pc_data = change_df[change_df['pc'] == pc]
            
            sns.boxplot(data=pc_data, x='condition', y='change_score', hue='group', ax=ax, 
                       palette=group_colors, linewidth=2, boxprops=dict(alpha=0.8))
            sns.stripplot(data=pc_data, x='condition', y='change_score', hue='group', ax=ax, 
                         dodge=True, alpha=0.7, size=6)
            
            ax.axhline(y=0, linestyle='--', color='black', alpha=0.7, linewidth=3, label='No change')
            ax.set_title(f'{pc} Change from Baseline', fontsize=24, fontweight='bold', pad=20)
            ax.set_ylabel('Change Score (Experimental - Baseline)', fontsize=18, fontweight='bold')
            ax.set_xlabel('Experimental Condition', fontsize=18, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
            ax.tick_params(axis='x', rotation=0)
            
            # Enhanced legend
            legend = ax.legend(title='Group', title_fontsize=16, fontsize=14, 
                              frameon=True, fancybox=True, shadow=True,
                              loc='upper right')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)
            
            ax.grid(True, alpha=0.4, linewidth=1.5)
            ax.set_facecolor('#FAFAFA')
            
            # Enhance spines
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('black')
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(output_dir, 'PC_change_scores.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'PC_change_scores.svg'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical tests on change scores
        print("\nStatistical tests on change scores:")
        test_results = []
        
        for pc in pc_components:
            print(f"\n{pc} Change Score Analysis:")
            print("-" * 40)
            
            pc_data = change_df[change_df['pc'] == pc]
            
            # Test if change scores differ from zero (one-sample t-tests)
            for condition in ['inclusion', 'exclusion']:
                for group in ['Controls', 'Risk of Depression']:
                    subset = pc_data[(pc_data['condition'] == condition) & (pc_data['group'] == group)]
                    if len(subset) > 0:
                        t_stat, p_val = stats.ttest_1samp(subset['change_score'], 0)
                        mean_change = subset['change_score'].mean()
                        
                        print(f"{group} - {condition}: M = {mean_change:.3f}, t({len(subset)-1}) = {t_stat:.3f}, p = {p_val:.4f}")
                        
                        test_results.append({
                            'pc': pc,
                            'group': group,
                            'condition': condition,
                            'test_type': 'one_sample_t_test',
                            'mean_change': mean_change,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'n': len(subset)
                        })
        
        # Save test results
        test_results_df = pd.DataFrame(test_results)
        test_file = os.path.join(output_dir, 'trajectory_change_tests.csv')
        test_results_df.to_csv(test_file, index=False)
        print(f"\nChange score test results saved to: {test_file}")
        
        return change_df, test_results_df
    
    return None, None

# Run trajectory analysis
print("Preparing trajectory data...")
df_trajectory = prepare_trajectory_data(df, pca_df)

print(f"\nTrajectory dataset overview:")
print(f"- Total observations: {len(df_trajectory)}")
print(f"- Unique subjects: {df_trajectory['subject_id'].nunique()}")
print(f"- Tasks: {sorted(df_trajectory['task'].unique())}")
print(f"- Conditions: {sorted(df_trajectory['condition'].unique())}")
print(f"- Groups: {sorted(df_trajectory['group'].unique())}")

# Create trajectory plots
trajectory_output_dir = TRAJECTORY_OUTPUT_DIR
os.makedirs(trajectory_output_dir, exist_ok=True)

create_trajectory_plots(df_trajectory, trajectory_output_dir)

# Analyze trajectory changes
change_scores, change_tests = analyze_trajectory_changes(df_trajectory, trajectory_output_dir)

print(f"\n" + "="*60)
print("TRAJECTORY ANALYSIS COMPLETE")
print("="*60)
print(f"All trajectory plots and analyses saved to: {trajectory_output_dir}")
print("\nGenerated files:")
for file in sorted(os.listdir(trajectory_output_dir)):
    print(f"- {file}")

# %%


