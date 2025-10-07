#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
import numpy as np
import ptitprince as pt
import matplotlib.pyplot as plt
import seaborn as sns
import os


df_clusters = pd.read_csv('../../results/Behavior/Clustering/clustering_hierarchical.csv')


df_metadata = pd.read_csv('../../metadata_experiment.csv')
df_metadata['subject_id'] = df_metadata['subj']
df_merged = pd.merge(df_clusters, df_metadata, on=['subject_id'], how='left')


# df_merged.Cluster.replace(0, 'Neutral', inplace=True)
# df_merged.Cluster.replace(1, 'Positive', inplace=True)

df_merged.Cluster.replace(0, 'Low', inplace=True)
df_merged.Cluster.replace(1, 'Self-Centered', inplace=True)
df_merged.Cluster.replace(2, 'Positive-Future-Oriented', inplace=True)
# # Convert to categorical with ordered levels
# df_merged['Cluster'] = pd.Categorical(df_merged['Cluster'], 
#                                      categories=['Low', 'Self-Centered', 'Positive-Future-Oriented'],
#                                      ordered=False)

df_merged.group.replace(1, 'Controls', inplace=True)
df_merged.group.replace(2, 'Risk of Depression', inplace=True)

#%%
# proportions of clusters responses by participant
proportions = df_merged.groupby(['subject_id','group' ,'Cluster']).size().reset_index(name='count')
proportions['proportion'] = proportions['count']# / proportions.groupby('subject_id')['count'].transform('sum')

# Create violin plot of proportions of clusters responses by participant
fig, ax = plt.subplots(figsize=(12, 6))

# Create the raincloud plot
pt.RainCloud(x='Cluster', 
             y='proportion', 
             data=  proportions, 
             hue='group',
             ax=ax,
             palette='Set1',
             bw=0.2,
             width_viol=0.7,
             alpha=0.65,
             dodge=True,
             pointplot=True,
             move=0.15)

# Customize the plot
ax.set_title('Probe Response Proportions by Group (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Proportion of Responses', fontsize=12)
ax.legend(title='Group', title_fontsize=12, fontsize=10)

# Adjust layout and display
plt.tight_layout()
plt.savefig(os.path.join( '../../results/Behavior/Clustering/clusters_by_conditions.png'), dpi=500, bbox_inches='tight')
plt.show()
# %%

def load_metadata(metadata_file):
    """
    Load metadata and create inclusion/exclusion mapping based on order and task.
    
    Returns
    -------
    dict
        Dictionary mapping (subject_id, task) to inclusion/exclusion condition
    """
    print(f"Loading metadata from: {metadata_file}")
    metadata = pd.read_csv(metadata_file)
    
    # Create subject to order mapping
    subject_to_order = {}
    for _, row in metadata.iterrows():
        subject_id = row['subj']
        order = row['order (IE/EI)']
        subject_to_order[subject_id] = order
    
    # Create mapping for (subject, task) -> inclusion/exclusion
    subject_task_to_condition = {}
    
    for subject_id, order in subject_to_order.items():
        if order == 'IE':
            # IE: Inclusion first (Sart2), then Exclusion (Sart4)
            subject_task_to_condition[(subject_id, 'Sart2')] = 'inclusion'
            subject_task_to_condition[(subject_id, 'Sart4')] = 'exclusion'
        elif order == 'EI':
            # EI: Exclusion first (Sart2), then Inclusion (Sart4)
            subject_task_to_condition[(subject_id, 'Sart2')] = 'exclusion'
            subject_task_to_condition[(subject_id, 'Sart4')] = 'inclusion'
    
    print(f"Loaded metadata for {len(subject_to_order)} subjects")
    
    # Count conditions
    inclusion_count = sum(1 for cond in subject_task_to_condition.values() if cond == 'inclusion')
    exclusion_count = sum(1 for cond in subject_task_to_condition.values() if cond == 'exclusion')
    
    print(f"Inclusion conditions: {inclusion_count}")
    print(f"Exclusion conditions: {exclusion_count}")
    
    return subject_task_to_condition

TASKS_TO_INCLUDE = ['Sart2', 'Sart4']
subject_task_to_condition = load_metadata('../../metadata_experiment.csv')

# Filter for Sart2 and Sart4 tasks only
df_filtered = df_merged[df_merged['task'].isin(TASKS_TO_INCLUDE)].copy()
print(f"Filtered to Sart2/Sart4 tasks: {df_filtered.shape[0]} rows")

# Add inclusion/exclusion condition information
df_filtered['inclusion_exclusion'] = df_filtered.apply(
    lambda row: subject_task_to_condition.get((row['subject_id'], row['task']), None), 
    axis=1
)


# %%
import seaborn as sns


proportions = df_filtered.groupby(['subject_id','group','inclusion_exclusion','Cluster']).size().reset_index(name='count')
proportions['proportion'] = proportions['count']# / proportions.groupby('subject_id')['count'].transform('sum')

# 1) Define un mapping explícito de colores
palette_inc_exc = {
    'inclusion':  sns.color_palette('Set2')[0],   # verde suave
    'exclusion':  sns.color_palette('Set2')[1]    # anaranjado suave
}

# 2) Guarda también el orden, para que 'inclusion' sea siempre el primero
hue_order_inc_exc = ['inclusion', 'exclusion']

# --- PRIMER GRÁFICO (todos los sujetos) -----------------
fig, ax = plt.subplots(figsize=(12, 6))

# Create the raincloud plot
pt.RainCloud(x='Cluster', 
             y='proportion', 
             data=  proportions, 
             hue='inclusion_exclusion',
             ax=ax,
             palette=palette_inc_exc,
             bw=0.2,
             width_viol=0.7,
             alpha=0.65,
             dodge=True,
             pointplot=True,
             move=0.15)

ax.set_title('Probe Response Proportions by Inclusion/Exclusion (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Proportion of Responses', fontsize=12)
ax.legend(title='Condition', title_fontsize=12, fontsize=10)
plt.tight_layout()
plt.savefig('../../results/Behavior/Clustering/clusters_by_conditions_inc_exc.png', dpi=500, bbox_inches='tight')
plt.show()

# --- SEGUNDO BLOQUE (loop por group) --------------------
for group in proportions['group'].unique():
    group_data = proportions[proportions['group'] == group]

    fig, ax = plt.subplots(figsize=(12, 6))
    
        # Create the raincloud plot
    pt.RainCloud(x='Cluster', 
                y='proportion', 
                data=  group_data, 
                hue='inclusion_exclusion',
                ax=ax,
                palette=palette_inc_exc,
                bw=0.2,
                width_viol=0.7,
                alpha=0.65,
                dodge=True,
                pointplot=True,
                move=0.15)

    ax.set_title(f'Probe Response Proportions for Group {group} (Violin Plot)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Proportion of Responses', fontsize=12)
    ax.legend(title='Condition', title_fontsize=12, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'../../results/Behavior/Clustering/clusters_by_conditions_{group}.png', dpi=500, bbox_inches='tight')
    plt.show()
# %%

# =============================================================================
# MULTINOMIAL GEE ANALYSIS FOR CLUSTER LEVELS
# =============================================================================

import statsmodels.api as sm
from statsmodels.genmod import families
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=== MULTINOMIAL GEE ANALYSIS FOR CLUSTER LEVELS ===")

# Create output directory for GEE results
gee_output_dir = '../../results/Behavior/Clustering/gee_analysis'
os.makedirs(gee_output_dir, exist_ok=True)

def prepare_gee_data(df_input):
    """
    Prepare data for GEE analysis by converting clusters to numeric codes
    """
    df_gee = df_input.copy()
    
    # Map clusters to numeric codes (reference: Low = 0)
    cluster_mapping = {
        'Low': 0,
        'Self-Centered': 1, 
        'Positive-Future-Oriented': 2
    }
    
    df_gee['cluster_numeric'] = df_gee['Cluster'].map(cluster_mapping)
    
    # Convert group to binary (Controls = 0, Risk of Depression = 1)
    df_gee['group_binary'] = (df_gee['group'] == 'Risk of Depression').astype(int)
    
    # Convert inclusion/exclusion to binary (inclusion = 0, exclusion = 1)
    df_gee['ie_binary'] = (df_gee['inclusion_exclusion'] == 'exclusion').astype(int)
    
    # Create interaction term
    df_gee['group_ie_interaction'] = df_gee['group_binary'] * df_gee['ie_binary']
    
    return df_gee

def run_multinomial_logistic_regression(df_gee, formula_vars, model_name, output_dir):
    """
    Run multinomial logistic regression using statsmodels
    Since statsmodels doesn't have true multinomial GEE, we'll use multinomial logit
    with cluster-robust standard errors as an approximation
    """
    from statsmodels.discrete.discrete_model import MNLogit
    try:
        import patsy
    except ImportError:
        print("Warning: patsy not available, using manual design matrix creation")
        patsy = None
    
    print(f"\n=== FITTING MODEL: {model_name} ===")
    
    # Prepare design matrix
    formula = f"cluster_numeric ~ {formula_vars}"
    print(f"Formula: {formula}")
    
    try:
        if patsy is not None:
            # Create design matrix using patsy
            y, X = patsy.dmatrices(formula, df_gee, return_type='dataframe')
            y = y.iloc[:, 0].astype(int)  # Convert to integer
        else:
            # Manual design matrix creation
            y = df_gee['cluster_numeric'].astype(int)
            if 'group_binary * ie_binary' in formula_vars:
                X = pd.DataFrame({
                    'Intercept': 1,
                    'group_binary': df_gee['group_binary'],
                    'ie_binary': df_gee['ie_binary'],
                    'group_binary:ie_binary': df_gee['group_binary'] * df_gee['ie_binary']
                })
            elif 'group_binary' in formula_vars:
                X = pd.DataFrame({
                    'Intercept': 1,
                    'group_binary': df_gee['group_binary']
                })
            elif 'ie_binary' in formula_vars:
                X = pd.DataFrame({
                    'Intercept': 1,
                    'ie_binary': df_gee['ie_binary']
                })
            else:
                X = pd.DataFrame({'Intercept': 1}, index=df_gee.index)
        
        # Fit multinomial logit model
        model = MNLogit(y, X)
        
        # Add cluster information for robust standard errors
        groups = df_gee['subject_id']
        
        # Fit with cluster-robust standard errors
        result = model.fit(cov_type='cluster', cov_kwds={'groups': groups}, disp=False)
        
        print("Model fitted successfully!")
        print(f"Log-likelihood: {result.llf:.4f}")
        print(f"AIC: {result.aic:.4f}")
        print(f"BIC: {result.bic:.4f}")
        
        # Debug: print shapes and structure
        print(f"Params shape: {result.params.shape}")
        print(f"Params index: {list(result.params.index)}")
        print(f"BSE shape: {result.bse.shape}")
        print(f"TValues shape: {result.tvalues.shape}")
        print(f"PValues shape: {result.pvalues.shape}")
        
        # Extract results - handle multinomial structure properly
        try:
            conf_int = result.conf_int()
            print(f"Conf int shape: {conf_int.shape}")
            
            # Flatten the multinomial results properly
            # Each row represents a predictor, each column represents a response category
            rows = []
            for i, param_name in enumerate(result.params.index):
                for j, outcome_category in enumerate(result.params.columns):
                    rows.append({
                        'coefficient': f"{param_name}[{outcome_category}]",
                        'predictor': param_name,
                        'outcome_category': outcome_category,
                        'estimate': result.params.iloc[i, j],
                        'std_error': result.bse.iloc[i, j],
                        'z_value': result.tvalues.iloc[i, j],
                        'p_value': result.pvalues.iloc[i, j],
                        'conf_lower': conf_int.iloc[i*2+j, 0],  # Conf int is stacked
                        'conf_upper': conf_int.iloc[i*2+j, 1]
                    })
            
            summary_df = pd.DataFrame(rows)
            print(f"Summary DataFrame shape: {summary_df.shape}")
            
        except Exception as debug_e:
            print(f"Debug error in results extraction: {debug_e}")
            # Fallback: create minimal results from flattened params
            flat_params = result.params.values.flatten()
            flat_index = [f"{row}[{col}]" for row in result.params.index for col in result.params.columns]
            
            summary_df = pd.DataFrame({
                'coefficient': flat_index,
                'predictor': [idx.split('[')[0] for idx in flat_index],
                'outcome_category': [idx.split('[')[1].rstrip(']') for idx in flat_index],
                'estimate': flat_params,
                'std_error': [0] * len(flat_params),
                'z_value': [0] * len(flat_params),
                'p_value': [1] * len(flat_params),
                'conf_lower': [0] * len(flat_params),
                'conf_upper': [0] * len(flat_params)
            })
            print("Using fallback results structure")
        
        # Map outcome categories to cluster names
        cluster_names = {1: 'Self-Centered', 2: 'Positive-Future-Oriented'}
        summary_df['cluster_name'] = summary_df['outcome_category'].astype(int).map(cluster_names)
        
        # Add significance flags
        summary_df['significant_05'] = summary_df['p_value'] < 0.05
        summary_df['significant_01'] = summary_df['p_value'] < 0.01
        
        # Save results
        results_file = os.path.join(output_dir, f'{model_name}_results.csv')
        summary_df.to_csv(results_file, index=False)
        
        # Save model summary
        with open(os.path.join(output_dir, f'{model_name}_summary.txt'), 'w') as f:
            f.write(str(result.summary()))
        
        print(f"Results saved to: {results_file}")
        
        return summary_df, result
        
    except Exception as e:
        print(f"Error fitting model {model_name}: {str(e)}")
        
        # Save error info
        error_df = pd.DataFrame({
            'model_name': [model_name],
            'error_message': [str(e)],
            'timestamp': [datetime.now()],
            'sample_size': [len(df_gee)],
            'n_subjects': [df_gee['subject_id'].nunique()]
        })
        error_file = os.path.join(output_dir, f'{model_name}_error.csv')
        error_df.to_csv(error_file, index=False)
        
        return None, None

def analyze_significant_results(results_df, model_name, output_dir):
    """
    Analyze and summarize significant results
    """
    if results_df is None:
        return
    
    # Filter out intercepts and focus on predictors
    predictors = results_df[~results_df['predictor'].str.contains('Intercept', na=False)].copy()
    
    if len(predictors) == 0:
        print(f"No predictor variables found for {model_name}")
        return
    
    # Apply Bonferroni correction
    if len(predictors) > 0:
        predictors['p_value_bonferroni'] = multipletests(predictors['p_value'], method='bonferroni')[1]
        predictors['significant_corrected'] = predictors['p_value_bonferroni'] < 0.05
    
    # Summarize significant results
    sig_uncorrected = predictors[predictors['significant_05']]
    sig_corrected = predictors[predictors['significant_corrected']]
    
    print(f"\n=== SIGNIFICANT RESULTS FOR {model_name} ===")
    print(f"Total predictors tested: {len(predictors)}")
    print(f"Significant (p < 0.05): {len(sig_uncorrected)}")
    print(f"Significant (Bonferroni corrected): {len(sig_corrected)}")
    
    if len(sig_uncorrected) > 0:
        print("\nSignificant effects (uncorrected):")
        for _, row in sig_uncorrected.iterrows():
            print(f"- {row['cluster_name']} ~ {row['predictor']}: "
                  f"β = {row['estimate']:.3f}, p = {row['p_value']:.4f}")
    
    if len(sig_corrected) > 0:
        print("\nSignificant effects (Bonferroni corrected):")
        for _, row in sig_corrected.iterrows():
            print(f"- {row['cluster_name']} ~ {row['predictor']}: "
                  f"β = {row['estimate']:.3f}, p = {row['p_value_bonferroni']:.4f}")
    
    # Save significant results
    if len(sig_uncorrected) > 0:
        sig_file = os.path.join(output_dir, f'{model_name}_significant_results.csv')
        sig_uncorrected.to_csv(sig_file, index=False)
    
    if len(sig_corrected) > 0:
        corrected_file = os.path.join(output_dir, f'{model_name}_significant_corrected.csv')
        sig_corrected.to_csv(corrected_file, index=False)
    
    return predictors

# Prepare data for GEE analysis
print("Preparing data for GEE analysis...")
df_gee = prepare_gee_data(df_filtered)

print(f"Sample: {len(df_gee)} observations from {df_gee['subject_id'].nunique()} subjects")
print(f"Cluster distribution: {df_gee['Cluster'].value_counts().to_dict()}")
print(f"Group distribution: {df_gee['group'].value_counts().to_dict()}")
print(f"Inclusion/Exclusion distribution: {df_gee['inclusion_exclusion'].value_counts().to_dict()}")

# %%
# Model 1: Group effect (Controls vs Risk of Depression)
print("\n" + "="*60)
print("MODEL 1: CLUSTER ~ GROUP")
print("="*60)

results_group, model_group = run_multinomial_logistic_regression(
    df_gee, 'group_binary', 'group_effect', gee_output_dir
)

if results_group is not None:
    analyze_significant_results(results_group, 'group_effect', gee_output_dir)

# %%
# Model 2: Inclusion/Exclusion effect
print("\n" + "="*60)
print("MODEL 2: CLUSTER ~ INCLUSION/EXCLUSION")
print("="*60)

results_ie, model_ie = run_multinomial_logistic_regression(
    df_gee, 'ie_binary', 'inclusion_exclusion_effect', gee_output_dir
)

if results_ie is not None:
    analyze_significant_results(results_ie, 'inclusion_exclusion_effect', gee_output_dir)

# %%
# Model 3: Interaction effect (Group × Inclusion/Exclusion)
print("\n" + "="*60)
print("MODEL 3: CLUSTER ~ GROUP * INCLUSION/EXCLUSION")
print("="*60)

results_interaction, model_interaction = run_multinomial_logistic_regression(
    df_gee, 'group_binary * ie_binary', 'group_ie_interaction', gee_output_dir
)

if results_interaction is not None:
    analyze_significant_results(results_interaction, 'group_ie_interaction', gee_output_dir)

# %%
# Create comprehensive summary report
print("\n" + "="*60)
print("CREATING SUMMARY REPORT")
print("="*60)

def create_summary_report(output_dir):
    """
    Create a comprehensive summary report of all analyses
    """
    
    # Collect all results files
    results_files = [f for f in os.listdir(output_dir) if f.endswith('_results.csv')]
    
    report_lines = [
        "# Multinomial Analysis of Cluster Levels - Summary Report",
        "",
        f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Sample:** {len(df_gee)} observations from {df_gee['subject_id'].nunique()} subjects",
        "",
        "## Cluster Distribution",
        f"- **Low:** {sum(df_gee['Cluster'] == 'Low')} ({sum(df_gee['Cluster'] == 'Low')/len(df_gee)*100:.1f}%)",
        f"- **Self-Centered:** {sum(df_gee['Cluster'] == 'Self-Centered')} ({sum(df_gee['Cluster'] == 'Self-Centered')/len(df_gee)*100:.1f}%)",
        f"- **Positive-Future-Oriented:** {sum(df_gee['Cluster'] == 'Positive-Future-Oriented')} ({sum(df_gee['Cluster'] == 'Positive-Future-Oriented')/len(df_gee)*100:.1f}%)",
        "",
        "## Group Distribution", 
        f"- **Controls:** {sum(df_gee['group'] == 'Controls')} ({sum(df_gee['group'] == 'Controls')/len(df_gee)*100:.1f}%)",
        f"- **Risk of Depression:** {sum(df_gee['group'] == 'Risk of Depression')} ({sum(df_gee['group'] == 'Risk of Depression')/len(df_gee)*100:.1f}%)",
        "",
        "## Inclusion/Exclusion Distribution",
        f"- **Inclusion:** {sum(df_gee['inclusion_exclusion'] == 'inclusion')} ({sum(df_gee['inclusion_exclusion'] == 'inclusion')/len(df_gee)*100:.1f}%)",
        f"- **Exclusion:** {sum(df_gee['inclusion_exclusion'] == 'exclusion')} ({sum(df_gee['inclusion_exclusion'] == 'exclusion')/len(df_gee)*100:.1f}%)",
        "",
        "## Models Tested",
        "1. **Group Effect:** Cluster ~ Group (Controls vs Risk of Depression)",
        "2. **Inclusion/Exclusion Effect:** Cluster ~ Inclusion/Exclusion",  
        "3. **Interaction Effect:** Cluster ~ Group × Inclusion/Exclusion",
        "",
        "## Statistical Method",
        "- **Model:** Multinomial Logistic Regression with cluster-robust standard errors",
        "- **Reference Category:** Low cluster (most common)",
        "- **Clustering:** By subject ID to account for repeated measures",
        "- **Multiple Comparisons:** Bonferroni correction applied",
        "",
        "## Key Findings"
    ]
    
    # Summarize findings from each model
    model_summaries = {
        'group_effect': 'Group Effect',
        'inclusion_exclusion_effect': 'Inclusion/Exclusion Effect', 
        'group_ie_interaction': 'Group × Inclusion/Exclusion Interaction'
    }
    
    for model_name, model_title in model_summaries.items():
        results_file = os.path.join(output_dir, f'{model_name}_results.csv')
        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
            predictors = results[~results['predictor'].str.contains('Intercept', na=False)]
            
            if len(predictors) > 0:
                # Apply Bonferroni correction
                predictors['p_value_bonferroni'] = multipletests(predictors['p_value'], method='bonferroni')[1]
                predictors['significant_corrected'] = predictors['p_value_bonferroni'] < 0.05
                
                n_sig_uncorrected = sum(predictors['p_value'] < 0.05)
                n_sig_corrected = sum(predictors['significant_corrected'])
                
                report_lines.extend([
                    f"",
                    f"### {model_title}",
                    f"- **Total effects tested:** {len(predictors)}",
                    f"- **Significant (p < 0.05):** {n_sig_uncorrected}",
                    f"- **Significant (Bonferroni corrected):** {n_sig_corrected}"
                ])
                
                # List significant effects
                sig_effects = predictors[predictors['significant_corrected']]
                if len(sig_effects) > 0:
                    report_lines.append("- **Significant effects:**")
                    for _, row in sig_effects.iterrows():
                        report_lines.append(f"  - {row['cluster_name']} ~ {row['predictor']}: β = {row['estimate']:.3f}, p = {row['p_value_bonferroni']:.4f}")
                else:
                    report_lines.append("- **No significant effects after correction**")
    
    report_lines.extend([
        "",
        "## Interpretation",
        "- **Low cluster:** Baseline/reference category representing low mind-wandering",
        "- **Self-Centered cluster:** Mind-wandering focused on self-related thoughts", 
        "- **Positive-Future-Oriented cluster:** Mind-wandering with positive, future-focused content",
        "",
        "Significant effects indicate that group membership and/or inclusion/exclusion conditions",
        "are associated with different patterns of mind-wandering cluster membership.",
        "",
        "## Files Generated",
        "- `*_results.csv` - Complete model results",
        "- `*_significant_results.csv` - Uncorrected significant effects",
        "- `*_significant_corrected.csv` - Bonferroni-corrected significant effects", 
        "- `*_summary.txt` - Detailed model output",
        "- `summary_report.md` - This comprehensive report"
    ])
    
    # Save report
    report_file = os.path.join(output_dir, 'summary_report.md')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Summary report saved to: {report_file}")

create_summary_report(gee_output_dir)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"All results saved to: {gee_output_dir}")
print("\nGenerated files:")
for file in sorted(os.listdir(gee_output_dir)):
    print(f"- {file}")

# %%
