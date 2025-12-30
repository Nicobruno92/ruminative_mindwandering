#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
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
# Load probe data (already merged with metadata by extractor)
df = pd.read_csv('../../results/Behavior/probe_data/probe_level_aggregated_data.csv')

#%%
print(df.describe())

#distribution plots
fig = px.histogram(df, x='onoff', nbins=100, title='Distribution of On/Off')
fig.show()

fig = px.histogram(df, x='valence', nbins=100, title='Distribution of Valence')
fig.show()

fig = px.histogram(df, x='selfother', nbins=100, title='Distribution of Self/Other')
fig.show()

fig = px.histogram(df, x='time', nbins=100, title='Distribution of Time')
fig.show()

fig = px.histogram(df, x='confidence', nbins=100, title='Distribution of Confidence')
fig.show()

#%% 
# On/Off distribution by subject (one violin per subject on x-axis)
# Sort subjects for consistent ordering
subjects_sorted = sorted(df['subject_id'].unique())
fig = px.violin(
    df,
    x='onoff',
    y='subject_id',
    color='subject_id',
    category_orders={'subject_id': subjects_sorted},
    points='all',
    box=True,
    title='On/Off distribution by subject'
)
#figsize
fig.update_layout(width=600, height=1000)
fig.update_layout(xaxis_title='On/Off', yaxis_title='Subject')
fig.update_xaxes(tickangle=-45)
fig.show()

#%%
# Loop to compare all other conditions against onoff
conditions = ['valence', 'selfother', 'time', 'confidence']

for condition in conditions:
    # Create a 2D scatter plot comparing onoff vs current condition
    fig = px.scatter(df, x='onoff', y=condition, 
                     title=f'On/Off vs {condition.title()} Comparison',
                     labels={'onoff': 'On/Off', condition: condition.title()},
                     opacity=0.7)

    # Add trend line
    fig.add_trace(go.Scatter(x=df['onoff'], y=df[condition], 
                            mode='markers', 
                            marker=dict(size=8, opacity=0.7),
                            name='Data Points'))

    # Calculate and add trend line
    z = np.polyfit(df['onoff'].dropna(), df[condition].dropna(), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=df['onoff'], y=p(df['onoff']), 
                            mode='lines', 
                            line=dict(color='red', width=2),
                            name='Trend Line'))

    # Update layout
    fig.update_layout(
        xaxis_title='On/Off',
        yaxis_title=condition.title(),
        showlegend=True,
        width=800,
        height=600
    )

    # Show the plot
    fig.show()
    
    # Also create a correlation analysis
    correlation = df['onoff'].corr(df[condition])
    print(f"Correlation between On/Off and {condition.title()}: {correlation:.3f}")
    print("-" * 50)

#%%
# Loop to compare all conditions against each other (excluding onoff)
conditions = ['valence', 'selfother', 'time', 'confidence']

# Create all pairwise combinations
from itertools import combinations
condition_pairs = list(combinations(conditions, 2))

for condition1, condition2 in condition_pairs:
    # Create a 2D scatter plot comparing the two conditions
    fig = px.scatter(df, x=condition1, y=condition2, 
                     title=f'{condition1.title()} vs {condition2.title()} Comparison',
                     labels={condition1: condition1.title(), condition2: condition2.title()},
                     opacity=0.7)

    # Add trend line
    fig.add_trace(go.Scatter(x=df[condition1], y=df[condition2], 
                            mode='markers', 
                            marker=dict(size=8, opacity=0.7),
                            name='Data Points'))

    # Calculate and add trend line
    z = np.polyfit(df[condition1].dropna(), df[condition2].dropna(), 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(x=df[condition1], y=p(df[condition1]), 
                            mode='lines', 
                            line=dict(color='red', width=2),
                            name='Trend Line'))

    # Update layout
    fig.update_layout(
        xaxis_title=condition1.title(),
        yaxis_title=condition2.title(),
        showlegend=True,
        width=800,
        height=600
    )
    # fig.write_html(f'../../results/Behavior/probe_data/scatter_plot_{condition1}_{condition2}.html')
    # fig.write_image(f'../../results/Behavior/probe_data/scatter_plot_{condition1}_{condition2}.png')
    # Show the plot
    fig.show()
    
    # Also create a correlation analysis
    correlation = df[condition1].corr(df[condition2])
    print(f"Correlation between {condition1.title()} and {condition2.title()}: {correlation:.3f}")
    print("-" * 50)

# %%

# =============================================================================
# PREPARE DATA FOR ANALYSIS (no external metadata merge)
# =============================================================================

def prepare_lmm_data(df_orig):
    """
    Prepare data for LMM analysis using pre-merged probe CSV.
    Expects columns: subject_id, task, group, inclusion_exclusion (optional), order (IE/EI) (optional).
    """
    df_with_metadata = df_orig.copy()
    df_with_metadata['subject_id'] = df_with_metadata['subject_id'].astype(str)

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

    print(f"Unique task labels before normalization: {sorted(df_with_metadata['task'].astype(str).unique())}")
    df_with_metadata['task'] = df_with_metadata['task'].apply(normalize_task_label)
    print(f"Unique task labels after normalization: {sorted(df_with_metadata['task'].astype(str).unique())}")

    # Keep only SART tasks
    TASKS_ALL = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
    before_task_filter_n = df_with_metadata.shape[0]
    df_tasks = df_with_metadata[df_with_metadata['task'].isin(TASKS_ALL)].copy()
    print(f"Task filter: {before_task_filter_n} -> {df_tasks.shape[0]} rows kept (tasks in {TASKS_ALL})")

    # Ensure inclusion_exclusion present (derive from order if missing)
    if 'inclusion_exclusion' not in df_tasks.columns or df_tasks['inclusion_exclusion'].isna().any():
        order_map = {}
        if 'order (IE/EI)' in df_tasks.columns:
            for _, r in df_tasks[['subject_id', 'order (IE/EI)']].drop_duplicates('subject_id').iterrows():
                order_map[str(r['subject_id'])] = r['order (IE/EI)']

        def map_task_to_condition(row):
            task = row['task']
            subj = str(row['subject_id'])
            if task in ['Sart1', 'Sart3']:
                return 'baseline'
            order = order_map.get(subj)
            if task == 'Sart2':
                return 'inclusion' if order == 'IE' else ('exclusion' if order == 'EI' else None)
            if task == 'Sart4':
                return 'exclusion' if order == 'IE' else ('inclusion' if order == 'EI' else None)
            return None

        df_tasks['inclusion_exclusion'] = df_tasks.get('inclusion_exclusion')
        df_tasks['inclusion_exclusion'] = df_tasks.apply(map_task_to_condition, axis=1)

    # Drop rows with missing inclusion_exclusion or group
    df_no_missing_ie = df_tasks.dropna(subset=['inclusion_exclusion'])
    print(f"After removing missing inclusion/exclusion: {df_no_missing_ie.shape[0]} rows")
    df_final = df_no_missing_ie.dropna(subset=['group']) if 'group' in df_no_missing_ie.columns else df_no_missing_ie
    if 'group' in df_no_missing_ie.columns:
        print(f"After removing missing group info: {df_final.shape[0]} rows")

    print(f"Final dataset for LMM: {df_final.shape[0]} rows from {df_final['subject_id'].nunique()} subjects")
    return df_final

# Prepare data for analysis
print("Preparing data for analysis...")
df_lmm = prepare_lmm_data(df)

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

# =============================================================================
# BASELINE NORMALIZATION FOR INCLUSION/EXCLUSION ANALYSIS
# =============================================================================
# Normalize each manipulation block by its immediately preceding baseline:
# - Sart2 (first manipulation) normalized by Sart1 mean (per subject)
# - Sart4 (second manipulation) normalized by Sart3 mean (per subject)

print("\n" + "="*60)
print("APPLYING BASELINE NORMALIZATION")
print("="*60)

# Calculate baseline means per subject
baseline_means = {}
for subject in df_lmm['subject_id'].unique():
    subject_data = df_lmm[df_lmm['subject_id'] == subject]
    
    # Sart1 baseline (for normalizing Sart2)
    sart1_data = subject_data[subject_data['task'] == 'Sart1']
    if len(sart1_data) > 0:
        baseline_means[(subject, 'Sart1')] = sart1_data['onoff'].mean()
    
    # Sart3 baseline (for normalizing Sart4)
    sart3_data = subject_data[subject_data['task'] == 'Sart3']
    if len(sart3_data) > 0:
        baseline_means[(subject, 'Sart3')] = sart3_data['onoff'].mean()

print(f"Calculated baseline means for {len(baseline_means)} subject-SART combinations")

# Apply normalization to IE subset
def normalize_by_baseline(row):
    """
    Normalize onoff score by subtracting the appropriate baseline mean.
    - Sart2 → subtract Sart1 mean
    - Sart4 → subtract Sart3 mean
    """
    subject = row['subject_id']
    task = row['task']
    
    if task == 'Sart2':
        baseline_key = (subject, 'Sart1')
    elif task == 'Sart4':
        baseline_key = (subject, 'Sart3')
    else:
        return np.nan  # Should not happen in IE subset
    
    baseline_mean = baseline_means.get(baseline_key, np.nan)
    if pd.isna(baseline_mean):
        return np.nan
    
    return row['onoff'] - baseline_mean

df_lmm_ie['onoff_normalized'] = df_lmm_ie.apply(normalize_by_baseline, axis=1)

# Remove rows where normalization failed (missing baseline)
before_dropna = len(df_lmm_ie)
df_lmm_ie = df_lmm_ie.dropna(subset=['onoff_normalized'])
after_dropna = len(df_lmm_ie)
if before_dropna > after_dropna:
    print(f"Warning: Removed {before_dropna - after_dropna} rows with missing baseline data")

print(f"Normalized IE subset: {len(df_lmm_ie)} observations")
print(f"Normalized onoff range: [{df_lmm_ie['onoff_normalized'].min():.2f}, {df_lmm_ie['onoff_normalized'].max():.2f}]")
print(f"Mean normalized onoff: {df_lmm_ie['onoff_normalized'].mean():.2f} (SD: {df_lmm_ie['onoff_normalized'].std():.2f})")

# Create time_on_task variable (probe_number + 15 * (sart_number - 1))
# Extract SART number from task name
df_lmm['sart_number'] = df_lmm['task'].str.extract(r'(\d+)').astype(int)

# Calculate time_on_task: probe_number + (15 * (sart_number - 1))
# This ensures: Sart1 = 1-15, Sart2 = 16-30, Sart3 = 31-45, Sart4 = 46-60
df_lmm['time_on_task'] = df_lmm['probe_number'] + (15 * (df_lmm['sart_number'] - 1))

# IMPORTANT: Create relative_time_on_task for within-SART comparisons
# This is just the probe_number (1-15) within each SART, allowing inclusion/exclusion to overlap
df_lmm['relative_time_on_task'] = df_lmm['probe_number']

# Also add to IE subset
df_lmm_ie['sart_number'] = df_lmm_ie['task'].str.extract(r'(\d+)').astype(int)
df_lmm_ie['time_on_task'] = df_lmm_ie['probe_number'] + (15 * (df_lmm_ie['sart_number'] - 1))
df_lmm_ie['relative_time_on_task'] = df_lmm_ie['probe_number']  # Within-SART time

print(f"Added time_on_task variable: range {df_lmm['time_on_task'].min()} to {df_lmm['time_on_task'].max()}")
print(f"Added relative_time_on_task (within-SART): range {df_lmm['relative_time_on_task'].min()} to {df_lmm['relative_time_on_task'].max()}")

# %%

# =============================================================================
# LINEAR MIXED MODEL ANALYSIS FOR ON/OFF SCORES - SAME AS PCA ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("LINEAR MIXED MODEL ANALYSIS FOR ON/OFF SCORES")
print("="*60)

def run_lmm_analysis(data, dependent_var, formula, model_name, output_dir):
    """
    Run linear mixed model analysis - simplified version
    """
    print(f"\n=== FITTING MODEL: {model_name} ===")
    print(f"Formula: {dependent_var} ~ {formula}")
    print(f"Sample size: {len(data)} observations from {data['subject_id'].nunique()} subjects")
    
    try:
        # Fit the model
        full_formula = f"{dependent_var} ~ {formula}"
        model = smf.mixedlm(full_formula, data, groups="subject_id").fit()

        print("Model fitted successfully!")
        print(model.summary())
        
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
            'rsquared_within': None,
            'rsquared_between': None
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
        return None, None

# Create output directory for LMM results
lmm_output_dir = '../../results/Behavior/probe_data/lmm_analysis_onoff'
os.makedirs(lmm_output_dir, exist_ok=True)

# Model 1: Group effect (Controls vs Risk of Depression)
print("\n" + "="*60)
print("MODEL 1: ONOFF ~ GROUP")
print("="*60)

results_group, model_group = run_lmm_analysis(
    df_lmm, 'onoff', 'group', 'group_effect', lmm_output_dir
)

# Model 2: Inclusion/Exclusion effect (NORMALIZED)
print("\n" + "="*60)
print("MODEL 2: ONOFF_NORMALIZED ~ INCLUSION/EXCLUSION")
print("="*60)

results_ie, model_ie = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'inclusion_exclusion', 'inclusion_exclusion_effect', lmm_output_dir
)

# =============================================================================
# ONE-SAMPLE TESTS: Are inclusion/exclusion effects different from zero?
# =============================================================================
print("\n" + "="*60)
print("ONE-SAMPLE TESTS AGAINST BASELINE (H0: effect = 0)")
print("="*60)

from scipy import stats

# Test 1: Is INCLUSION effect different from zero?
inclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'inclusion']['onoff_normalized'].dropna()
t_incl, p_incl = stats.ttest_1samp(inclusion_data, 0)
mean_incl = inclusion_data.mean()
se_incl = inclusion_data.sem()
n_incl = len(inclusion_data)

print(f"\n1. INCLUSION vs Baseline (zero):")
print(f"   N = {n_incl}")
print(f"   Mean = {mean_incl:.4f} (SE = {se_incl:.4f})")
print(f"   t({n_incl-1}) = {t_incl:.4f}, p = {p_incl:.4f}")
if p_incl < 0.05:
    print(f"   *** SIGNIFICANT: Inclusion {'increases' if mean_incl > 0 else 'decreases'} relative to baseline")
else:
    print(f"   Not significant: Inclusion does not differ from baseline")

# Test 2: Is EXCLUSION effect different from zero?
exclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'exclusion']['onoff_normalized'].dropna()
t_excl, p_excl = stats.ttest_1samp(exclusion_data, 0)
mean_excl = exclusion_data.mean()
se_excl = exclusion_data.sem()
n_excl = len(exclusion_data)

print(f"\n2. EXCLUSION vs Baseline (zero):")
print(f"   N = {n_excl}")
print(f"   Mean = {mean_excl:.4f} (SE = {se_excl:.4f})")
print(f"   t({n_excl-1}) = {t_excl:.4f}, p = {p_excl:.4f}")
if p_excl < 0.05:
    print(f"   *** SIGNIFICANT: Exclusion {'increases' if mean_excl > 0 else 'decreases'} relative to baseline")
else:
    print(f"   Not significant: Exclusion does not differ from baseline")

# Bonferroni correction for multiple comparisons
p_bonf_incl = min(p_incl * 2, 1.0)
p_bonf_excl = min(p_excl * 2, 1.0)

print(f"\n3. Bonferroni-corrected p-values (2 tests):")
print(f"   Inclusion: p_corrected = {p_bonf_incl:.4f} {'***' if p_bonf_incl < 0.05 else ''}")
print(f"   Exclusion: p_corrected = {p_bonf_excl:.4f} {'***' if p_bonf_excl < 0.05 else ''}")

# Save results
one_sample_results = pd.DataFrame([
    {
        'Condition': 'Inclusion',
        'N': n_incl,
        'Mean': mean_incl,
        'SE': se_incl,
        't_statistic': t_incl,
        'p_value': p_incl,
        'p_bonferroni': p_bonf_incl,
        'Significant_uncorrected': p_incl < 0.05,
        'Significant_bonferroni': p_bonf_incl < 0.05
    },
    {
        'Condition': 'Exclusion',
        'N': n_excl,
        'Mean': mean_excl,
        'SE': se_excl,
        't_statistic': t_excl,
        'p_value': p_excl,
        'p_bonferroni': p_bonf_excl,
        'Significant_uncorrected': p_excl < 0.05,
        'Significant_bonferroni': p_bonf_excl < 0.05
    }
])

one_sample_file = os.path.join(lmm_output_dir, 'one_sample_tests_vs_baseline.csv')
one_sample_results.to_csv(one_sample_file, index=False)
print(f"\nOne-sample test results saved to: {one_sample_file}")

print("\nNote: Model 2 above tests if inclusion and exclusion DIFFER from each other.")
print("      These one-sample tests show if each condition differs from baseline (zero).")

# Model 3: Interaction effect (Group × Inclusion/Exclusion) (NORMALIZED)
print("\n" + "="*60)
print("MODEL 3: ONOFF_NORMALIZED ~ GROUP * INCLUSION/EXCLUSION")
print("="*60)

results_interaction, model_interaction = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'group * inclusion_exclusion', 'group_ie_interaction', lmm_output_dir
)

# Model 4: Group + Time on task
print("\n" + "="*60)
print("MODEL 4: ONOFF ~ GROUP + TIME_ON_TASK")
print("="*60)

results_group_time, model_group_time = run_lmm_analysis(
    df_lmm, 'onoff', 'group + time_on_task', 'group_time_additive', lmm_output_dir
)

# Model 5: Group × Time on task interaction
print("\n" + "="*60)
print("MODEL 5: ONOFF ~ GROUP * TIME_ON_TASK")
print("="*60)

results_group_time_int, model_group_time_int = run_lmm_analysis(
    df_lmm, 'onoff', 'group * time_on_task', 'group_time_interaction', lmm_output_dir
)

# =============================================================================
# MODEL 6: EXTRACT INDIVIDUAL TIME-ON-TASK SLOPES (OLS per subject)
# =============================================================================
# Simple linear regression per subject to capture individual time-on-task effect
# More variable than LMM random slopes (no shrinkage toward population mean)

print("\n" + "="*60)
print("MODEL 6: EXTRACTING INDIVIDUAL TIME-ON-TASK SLOPES (OLS)")
print("="*60)

# Calculate OLS slope per subject
tot_slopes_list = []
for subject_id in df_lmm['subject_id'].unique():
    subject_data = df_lmm[df_lmm['subject_id'] == subject_id]
    n_probes = len(subject_data)
    
    if n_probes >= 10:  # Need minimum points for reliable slope
        # Fit linear regression: onoff ~ time_on_task
        slope, intercept = np.polyfit(subject_data['time_on_task'], subject_data['onoff'], 1)
        
        # Calculate R² for quality check
        predicted = slope * subject_data['time_on_task'] + intercept
        ss_res = ((subject_data['onoff'] - predicted) ** 2).sum()
        ss_tot = ((subject_data['onoff'] - subject_data['onoff'].mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # New metric: difference between first 5 and last 5 probes
        # Sort by time_on_task to ensure correct order
        subject_sorted = subject_data.sort_values('time_on_task')
        first_5_mean = subject_sorted.head(5)['onoff'].mean()
        last_5_mean = subject_sorted.tail(5)['onoff'].mean()
        first_last_diff = last_5_mean - first_5_mean
        
        tot_slopes_list.append({
            'subject_id': subject_id,
            'time_on_task_slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'n_probes': n_probes,
            'tot_first_last_diff': first_last_diff
        })

tot_slopes_df = pd.DataFrame(tot_slopes_list)

# Add group information
subject_group_map = df_lmm.drop_duplicates('subject_id').set_index('subject_id')['group'].to_dict()
tot_slopes_df['group'] = tot_slopes_df['subject_id'].map(subject_group_map)

# Add BDI information
subject_bdi_map = df_lmm.drop_duplicates('subject_id').set_index('subject_id')['bdi'].to_dict()
tot_slopes_df['bdi'] = tot_slopes_df['subject_id'].map(subject_bdi_map)

# Summary statistics
print(f"\nTime-on-Task Slopes Summary (OLS per subject):")
print(f"  N subjects: {len(tot_slopes_df)}")
metrics = ['time_on_task_slope', 'tot_first_last_diff']
labels = ['Linear Slope', 'First-Last Diff']

for metric, label in zip(metrics, labels):
    print(f"\n{label}:")
    print(f"  Mean: {tot_slopes_df[metric].mean():.4f}")
    print(f"  SD: {tot_slopes_df[metric].std():.4f}")
    print(f"  Range: [{tot_slopes_df[metric].min():.4f}, {tot_slopes_df[metric].max():.4f}]")
    
    # Group comparison
    print(f"  By Group:")
    groups = tot_slopes_df['group'].dropna().unique()
    for group in groups:
        group_data = tot_slopes_df[tot_slopes_df['group'] == group][metric]
        print(f"    {group}: M = {group_data.mean():.4f}, SD = {group_data.std():.4f}")
    
    # T-test
    if len(groups) >= 2:
        g1 = tot_slopes_df[tot_slopes_df['group'] == groups[0]][metric]
        g2 = tot_slopes_df[tot_slopes_df['group'] == groups[1]][metric]
        t_stat, p_val = stats.ttest_ind(g1, g2)
        print(f"  Group difference t-test: t = {t_stat:.3f}, p = {p_val:.3f}")

# =============================================================================
# VISUALIZATION: Time-on-Task Slopes by Group
# =============================================================================
# Define colors for consistency
group_colors = ['#2E86AB', '#F24236']
ie_colors = ['#A23B72', '#F18F01']

# Define plots output directory
plots_output_dir = '../../results/Behavior/probe_data/lmm_plots_onoff'
os.makedirs(plots_output_dir, exist_ok=True)

# Save individual slopes
tot_slopes_file = os.path.join(lmm_output_dir, 'time_on_task_slopes_per_subject.csv')
tot_slopes_df.to_csv(tot_slopes_file, index=False)
print(f"\nIndividual time-on-task metrics saved to: {tot_slopes_file}")

# Also save to the main results folder for easy access by correlation scripts
main_results_slopes_file = '../../results/Behavior/probe_data/time_on_task_slopes_per_subject.csv'
tot_slopes_df.to_csv(main_results_slopes_file, index=False)
print(f"Also saved to: {main_results_slopes_file}")

print("\n" + "="*60)
print("CREATING TIME-ON-TASK SLOPES VISUALIZATION")
print("="*60)

metrics = ['time_on_task_slope', 'tot_first_last_diff']
metric_labels = ['Linear Slope', 'First-Last Diff']
metric_short = ['linear_slope', 'first_last_diff']

for metric, label, short in zip(metrics, metric_labels, metric_short):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Plot 1: Raincloud plot comparing groups
    ax1 = axes[0]
    pt.RainCloud(
        x="group",
        y=metric,
        data=tot_slopes_df,
        palette=group_colors,
        order=['Controls', 'Risk of Depression'],
        bw=0.2,
        width_viol=0.6,
        alpha=0.7,
        dodge=True,
        pointplot=True,
        move=-0.1,
        ax=ax1,
    )
    ax1.set_title(f"{label} by Group", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Group", fontsize=12, fontweight="bold")
    ax1.set_ylabel(label, fontsize=12, fontweight="bold")
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot of individual values with jitter
    ax2 = axes[1]
    for i, group in enumerate(['Controls', 'Risk of Depression']):
        group_data = tot_slopes_df[tot_slopes_df['group'] == group]
        x_jitter = np.random.normal(i, 0.1, len(group_data))
        ax2.scatter(x_jitter, group_data[metric], 
                    c=group_colors[i], alpha=0.7, s=80, edgecolor='white', linewidth=0.5,
                    label=f"{group} (n={len(group_data)})")
        # Add mean line
        mean_val = group_data[metric].mean()
        ax2.hlines(mean_val, i-0.3, i+0.3, colors=group_colors[i], linewidth=3)

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Controls', 'Risk of Depression'])
    ax2.set_title(f"Individual {label}", fontsize=16, fontweight="bold")
    ax2.set_xlabel("Group", fontsize=12, fontweight="bold")
    ax2.set_ylabel(label, fontsize=12, fontweight="bold")
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histogram of values by group
    ax3 = axes[2]
    for i, group in enumerate(['Controls', 'Risk of Depression']):
        group_data = tot_slopes_df[tot_slopes_df['group'] == group][metric]
        ax3.hist(group_data, bins=15, alpha=0.6, color=group_colors[i], 
                 label=f"{group}", edgecolor='white')
        ax3.axvline(group_data.mean(), color=group_colors[i], linestyle='--', linewidth=2)

    ax3.set_title(f"Distribution of {label}", fontsize=16, fontweight="bold")
    ax3.set_xlabel(label, fontsize=12, fontweight="bold")
    ax3.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Correlation with BDI
    ax4 = axes[3]
    for i, group in enumerate(['Controls', 'Risk of Depression']):
        group_data = tot_slopes_df[tot_slopes_df['group'] == group]
        if len(group_data.dropna(subset=['bdi', metric])) > 0:
            ax4.scatter(group_data['bdi'], group_data[metric], 
                       c=group_colors[i], alpha=0.7, s=80, edgecolor='white', linewidth=0.5,
                       label=group)

    # Add trend line and correlation for all data
    valid_data = tot_slopes_df.dropna(subset=['bdi', metric])
    if len(valid_data) > 5:
        line_slope, line_intercept, r_val, p_val_bdi, _ = stats.linregress(valid_data['bdi'], valid_data[metric])
        corr_spearman, p_spearman = stats.spearmanr(valid_data['bdi'], valid_data[metric])
        
        x_range = np.array([valid_data['bdi'].min(), valid_data['bdi'].max()])
        ax4.plot(x_range, line_slope * x_range + line_intercept, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        corr_text = f"Spearman r = {corr_spearman:.3f}\np = {p_spearman:.3f}"
        ax4.text(0.95, 0.95, corr_text, transform=ax4.transAxes, ha='right', va='top',
                 fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.set_title(f"{label} vs BDI", fontsize=16, fontweight="bold")
    ax4.set_xlabel("BDI Score", fontsize=12, fontweight="bold")
    ax4.set_ylabel(label, fontsize=12, fontweight="bold")
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_dir, f'time_on_task_{short}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_output_dir, f'time_on_task_{short}_analysis.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig) # Close to free memory

print(f"Slopes visualizations saved to: {plots_output_dir}")

# =============================================================================
# INTERVENTION DISTANCE MODELS (probe_number within block: 1-15)
# These test how the effect evolves within each manipulation block
# =============================================================================

# Model 7: Inclusion/Exclusion + Intervention distance (NORMALIZED)
print("\n" + "="*60)
print("MODEL 7: ONOFF_NORMALIZED ~ INCLUSION/EXCLUSION + PROBE_NUMBER")
print("="*60)

results_ie_distance, model_ie_distance = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'inclusion_exclusion + probe_number', 'ie_intervention_distance', lmm_output_dir
)

# Model 8: Group * Inclusion/Exclusion + Intervention distance (NORMALIZED)
print("\n" + "="*60)
print("MODEL 8: ONOFF_NORMALIZED ~ GROUP * INCLUSION/EXCLUSION + PROBE_NUMBER")
print("="*60)

results_group_ie_distance, model_group_ie_distance = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'group * inclusion_exclusion + probe_number', 'group_ie_plus_distance', lmm_output_dir
)

# Model 9: Inclusion/Exclusion × Intervention distance interaction (NORMALIZED)
print("\n" + "="*60)
print("MODEL 9: ONOFF_NORMALIZED ~ INCLUSION/EXCLUSION * PROBE_NUMBER")
print("="*60)

results_ie_distance_int, model_ie_distance_int = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'inclusion_exclusion * probe_number', 'ie_distance_interaction', lmm_output_dir
)

# Model 10: Group × Inclusion/Exclusion × Intervention distance (three-way) (NORMALIZED)
print("\n" + "="*60)
print("MODEL 10: ONOFF_NORMALIZED ~ GROUP * INCLUSION/EXCLUSION * PROBE_NUMBER")
print("="*60)

results_three_way_distance, model_three_way_distance = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'group * inclusion_exclusion * probe_number', 'three_way_with_distance', lmm_output_dir
)

# Model 11: Group × Intervention distance + Inclusion/Exclusion (NORMALIZED)
print("\n" + "="*60)
print("MODEL 11: ONOFF_NORMALIZED ~ GROUP * PROBE_NUMBER + INCLUSION/EXCLUSION")
print("="*60)

results_group_distance_ie, model_group_distance_ie = run_lmm_analysis(
    df_lmm_ie, 'onoff_normalized', 'group * probe_number + inclusion_exclusion', 'group_distance_plus_ie', lmm_output_dir
)

# %%

# =============================================================================
# VISUALIZATION OF LMM RESULTS - SIMPLIFIED
# =============================================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS FOR ON/OFF ANALYSIS")
print("="*60)

# Create output directory for plots
# Create time-on-task visualization
plt.style.use('default')

# Time-on-task visualization
print("\n" + "="*60)
print("CREATING TIME-ON-TASK VISUALIZATION")
print("="*60)

# Create time-on-task visualization
plt.style.use('default')

plt.style.use('default')
fig_time, axes_time = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Overall time-on-task trend
ax1 = axes_time[0, 0]
# Aggregate by time_on_task for cleaner visualization
time_agg = df_lmm.groupby('time_on_task')['onoff'].agg(['mean', 'sem']).reset_index()
ax1.errorbar(time_agg['time_on_task'], time_agg['mean'], yerr=time_agg['sem'], 
            marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8, color='#2E86AB')
ax1.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
ax1.set_ylabel('On/Off Score', fontsize=14, fontweight='bold')
ax1.set_title('On/Off Scores Across Time on Task', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Time-on-task by group
ax2 = axes_time[0, 1]
for i, group in enumerate(['Controls', 'Risk of Depression']):
    group_data = df_lmm[df_lmm['group'] == group]
    time_group_agg = group_data.groupby('time_on_task')['onoff'].agg(['mean', 'sem']).reset_index()
    color = group_colors[i]
    ax2.errorbar(time_group_agg['time_on_task'], time_group_agg['mean'], yerr=time_group_agg['sem'], 
                marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8, 
                color=color, label=group)
ax2.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
ax2.set_ylabel('On/Off Score', fontsize=14, fontweight='bold')
ax2.set_title('Time-on-Task Effect by Group', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: Intervention distance by inclusion/exclusion (probe 1-15 within block) - NORMALIZED
ax3 = axes_time[1, 0]
for i, condition in enumerate(['inclusion', 'exclusion']):
    ie_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == condition]
    # Use probe_number (1-15 within each manipulation block)
    probe_ie_agg = ie_data.groupby('probe_number')['onoff_normalized'].agg(['mean', 'sem']).reset_index()
    color = ie_colors[i]
    ax3.errorbar(probe_ie_agg['probe_number'], probe_ie_agg['mean'], yerr=probe_ie_agg['sem'], 
                marker='s', linewidth=2, markersize=6, capsize=3, alpha=0.8, 
                color=color, label=condition.capitalize())
ax3.set_xlabel('Intervention Distance (Probe Number)', fontsize=14, fontweight='bold')
ax3.set_ylabel('On/Off Score (Normalized)', fontsize=14, fontweight='bold')
ax3.set_title('Intervention Distance by I/E (Baseline-Corrected)', fontsize=16, fontweight='bold')
# Add baseline reference line
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.5, 15.5)  # Set x-axis limits for 1-15 probes

# Plot 4: Correlation scatter
ax4 = axes_time[1, 1]
scatter = ax4.scatter(df_lmm['time_on_task'], df_lmm['onoff'], 
                     c=df_lmm['group'].map({'Controls': 0, 'Risk of Depression': 1}), 
                     cmap='RdBu_r', alpha=0.6, s=30)
# Add trend line
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df_lmm['time_on_task'], df_lmm['onoff'])
x_trend = np.linspace(df_lmm['time_on_task'].min(), df_lmm['time_on_task'].max(), 100)
y_trend = slope * x_trend + intercept
ax4.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.8)
ax4.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
ax4.set_ylabel('On/Off Score', fontsize=14, fontweight='bold')
ax4.set_title(f'Time vs On/Off Correlation (r={r_value:.3f}, p={p_value:.3f})', fontsize=16, fontweight='bold')
ax4.grid(True, alpha=0.3)
# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Controls', 'Risk of Depression'])

plt.tight_layout(pad=3.0)
plt.savefig(os.path.join(plots_output_dir, 'onoff_time_on_task_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(plots_output_dir, 'onoff_time_on_task_analysis.svg'), dpi=300, bbox_inches='tight')
plt.show()

# Combined comprehensive 2x3 visualization
plt.style.use('default')
fig = plt.figure(figsize=(24, 14))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

# Define consistent ordering and colors
group_order = ['Controls', 'Risk of Depression']
ie_order = ['inclusion', 'exclusion']
group_colors = ['#2E86AB', '#F24236']
ie_colors = ['#A23B72', '#F18F01']

# =========================================================================
# ROW 1: DISTRIBUTION COMPARISONS
# =========================================================================

# Plot 1 (0,0): Raincloud for group comparison
ax1 = fig.add_subplot(gs[0, 0])
df_agg_group = df_lmm.groupby(["subject_id", "group"])['onoff'].mean().reset_index()
n_participants_by_group = df_agg_group.groupby("group")["subject_id"].nunique().to_dict()

pt.RainCloud(
    x="group",
    y='onoff',
    data=df_agg_group,
    palette=group_colors,
    order=group_order,
    bw=0.2,
    width_viol=0.6,
    alpha=0.7,
    dodge=True,
    pointplot=True,
    move=-0.1,
    ax=ax1,
)
ax1.set_title(
    "ON/OFF: Group Effect",
    fontsize=18,
    fontweight="bold",
    pad=15,
)
ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
ax1.set_ylabel("On/Off Score", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Add sample sizes
for i, group in enumerate(group_order):
    n = n_participants_by_group.get(group, 0)
    ax1.text(
        i, ax1.get_ylim()[1] * 0.95, f"n={n}",
        ha="center", va="top", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

# Plot 2 (0,1): Raincloud for inclusion/exclusion - NORMALIZED
ax2 = fig.add_subplot(gs[0, 1])
df_agg_ie = df_lmm_ie.groupby(["subject_id", "inclusion_exclusion"])['onoff_normalized'].mean().reset_index()
n_participants_by_ie = df_agg_ie.groupby("inclusion_exclusion")["subject_id"].nunique().to_dict()

pt.RainCloud(
    x="inclusion_exclusion",
    y='onoff_normalized',
    data=df_agg_ie,
    palette=ie_colors,
    order=ie_order,
    bw=0.2,
    width_viol=0.6,
    alpha=0.7,
    dodge=True,
    pointplot=True,
    move=-0.1,
    ax=ax2,
)
ax2.set_title(
    "ON/OFF: Inclusion/Exclusion Effect (Baseline-Corrected)",
    fontsize=18,
    fontweight="bold",
    pad=15,
)
ax2.set_xlabel("Condition", fontsize=14, fontweight="bold")
ax2.set_ylabel("On/Off Score (Normalized)", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)

# Add baseline reference line
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
ax2.legend(fontsize=10, loc='best')

# Add sample sizes
for i, condition in enumerate(ie_order):
    n = n_participants_by_ie.get(condition, 0)
    ax2.text(
        i, ax2.get_ylim()[1] * 0.95, f"n={n}",
        ha="center", va="top", fontsize=12, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
    )

# Plot 3 (0,2): Interaction plot (Group × I/E) - NORMALIZED
ax3 = fig.add_subplot(gs[0, 2])
for group in df_lmm_ie['group'].dropna().unique():
    group_data = df_lmm_ie[df_lmm_ie['group'] == group]
    if len(group_data) == 0:
        continue
    ie_means = (
        group_data.groupby("inclusion_exclusion")['onoff_normalized']
        .agg(["mean", "sem"])
        .reindex(ie_order)
        .reset_index()
    )
    n_participants_group = group_data["subject_id"].nunique()
    color = group_colors[0] if group == group_order[0] else group_colors[1]
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
ax3.set_xticklabels(ie_order, fontsize=14, fontweight="bold")
ax3.set_ylabel("On/Off Score (Normalized)", fontsize=14, fontweight="bold")
ax3.set_title(
    "ON/OFF: Group × I/E Interaction (Baseline-Corrected)",
    fontsize=18,
    fontweight="bold",
    pad=15,
)
# Add baseline reference line
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
ax3.legend(fontsize=12, title_fontsize=12, loc='best')
ax3.grid(True, alpha=0.3)

# =========================================================================
# ROW 2: TIME-ON-TASK TRAJECTORIES
# =========================================================================

# Plot 4 (1,0): Time-on-task by group
ax4 = fig.add_subplot(gs[1, 0])
for i, group in enumerate(group_order):
    if group in df_lmm["group"].values:
        group_data = df_lmm[df_lmm["group"] == group]
        time_group_agg = group_data.groupby("time_on_task")['onoff'].agg(["mean", "sem"]).reset_index()
        color = group_colors[i]
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
ax4.set_ylabel("On/Off Score", fontsize=14, fontweight="bold")
ax4.set_title(
    "ON/OFF: Time-on-Task by Group",
    fontsize=18,
    fontweight="bold",
    pad=15,
)
ax4.legend(fontsize=12, loc='best')
ax4.grid(True, alpha=0.3)

# Plot 5 (1,1): Intervention distance by inclusion/exclusion (probe 1-15 within block) - NORMALIZED
ax5 = fig.add_subplot(gs[1, 1])
for i, condition in enumerate(ie_order):
    if condition in df_lmm_ie["inclusion_exclusion"].values:
        ie_data = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == condition]
        # Use probe_number (1-15 within each manipulation block)
        probe_ie_agg = ie_data.groupby("probe_number")['onoff_normalized'].agg(["mean", "sem"]).reset_index()
        color = ie_colors[i]
        ax5.errorbar(
            probe_ie_agg["probe_number"],
            probe_ie_agg["mean"],
            yerr=probe_ie_agg["sem"],
            marker="s",
            linewidth=2.5,
            markersize=6,
            capsize=3,
            alpha=0.8,
            color=color,
            label=condition.capitalize(),
        )
ax5.set_xlabel("Intervention Distance (Probe Number)", fontsize=14, fontweight="bold")
ax5.set_ylabel("On/Off Score (Normalized)", fontsize=14, fontweight="bold")
ax5.set_title(
    "ON/OFF: Intervention Distance by I/E (Baseline-Corrected)",
    fontsize=18,
    fontweight="bold",
    pad=15,
)
# Add baseline reference line
ax5.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
ax5.legend(fontsize=12, loc='best')
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0.5, 15.5)  # Set x-axis limits for 1-15 probes

# Plot 6 (1,2): SART Mean Trajectories by Group and Order (4 points per line)
ax6 = fig.add_subplot(gs[1, 2])
if 'order (IE/EI)' in df_lmm.columns:
    # Calculate mean per SART for each group and order combination
    sart_order_list = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
    x_positions = [1, 2, 3, 4]  # X-axis positions for each SART
    
    # Define line styles for orders
    order_styles = {'IE': '-', 'EI': '--'}  # IE = Inclusion-Exclusion, EI = Exclusion-Inclusion
    
    for group_idx, group in enumerate(group_order):
        if group not in df_lmm["group"].values:
            continue
        group_data = df_lmm[df_lmm["group"] == group]
        color = group_colors[group_idx]
        
        for order in ['IE', 'EI']:
            if order not in group_data['order (IE/EI)'].values:
                continue
            order_data = group_data[group_data['order (IE/EI)'] == order]
            
            # Calculate n for this group-order combination
            n_subjects = order_data['subject_id'].nunique()
            
            means = []
            sems = []
            for sart in sart_order_list:
                sart_data = order_data[order_data['task'] == sart]
                if len(sart_data) > 0:
                    means.append(sart_data['onoff'].mean())
                    sems.append(sart_data['onoff'].sem())
                else:
                    means.append(np.nan)
                    sems.append(np.nan)
            
            # Plot line with error bars
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
    ax6.set_ylabel("On/Off Score", fontsize=14, fontweight="bold")
    ax6.set_title(
        "ON/OFF: SART Trajectory by Group & Order\n(Solid=IE, Dashed=EI)",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    # Create legend with only lines (no markers)
    handles, labels = ax6.get_legend_handles_labels()
    # Remove markers from legend handles - errorbar returns containers, need to access the line
    new_handles = []
    for handle in handles:
        if hasattr(handle, 'get_children'):
            # ErrorbarContainer - get the line component
            lines = [child for child in handle.get_children() if hasattr(child, 'set_marker')]
            if lines:
                line = lines[0]
                line.set_marker('')
                line.set_markersize(0)
                new_handles.append(line)
        else:
            # Regular line object
            handle.set_marker('')
            handle.set_markersize(0)
            new_handles.append(handle)
    ax6.legend(new_handles, labels, fontsize=11, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0.5, 4.5)
else:
    ax6.text(0.5, 0.5, 'Order data not available', 
            ha='center', va='center', transform=ax6.transAxes, fontsize=14)

# =========================================================================
# SAVE AND DISPLAY
# =========================================================================

plt.suptitle(
    "Comprehensive Analysis: ON/OFF",
    fontsize=22,
    fontweight="bold",
    y=0.995,
)

plt.savefig(os.path.join(plots_output_dir, 'onoff_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(plots_output_dir, 'onoff_comprehensive_analysis.svg'), dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

# Generate descriptive statistics
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS FOR ON/OFF SCORES")
print("="*60)

# Overall statistics
overall_stats = df_lmm['onoff'].describe()
print(f"Overall: Mean = {overall_stats['mean']:.3f}, SD = {overall_stats['std']:.3f}")

# By group
print("\nBy Group:")
group_stats = df_lmm.groupby('group')['onoff'].agg(['count', 'mean', 'std']).round(3)
print(group_stats)

# By inclusion/exclusion (RAW)
print("\nBy Inclusion/Exclusion (Raw):")
ie_stats = df_lmm_ie.groupby('inclusion_exclusion')['onoff'].agg(['count', 'mean', 'std']).round(3)
print(ie_stats)

# By inclusion/exclusion (NORMALIZED)
print("\nBy Inclusion/Exclusion (Baseline-Corrected):")
ie_stats_norm = df_lmm_ie.groupby('inclusion_exclusion')['onoff_normalized'].agg(['count', 'mean', 'std']).round(3)
print(ie_stats_norm)

# By group × inclusion/exclusion (RAW)
print("\nBy Group × Inclusion/Exclusion (Raw):")
interaction_stats = df_lmm_ie.groupby(['group', 'inclusion_exclusion'])['onoff'].agg(['count', 'mean', 'std']).round(3)
print(interaction_stats)

# By group × inclusion/exclusion (NORMALIZED)
print("\nBy Group × Inclusion/Exclusion (Baseline-Corrected):")
interaction_stats_norm = df_lmm_ie.groupby(['group', 'inclusion_exclusion'])['onoff_normalized'].agg(['count', 'mean', 'std']).round(3)
print(interaction_stats_norm)

# Save descriptive statistics (both raw and normalized)
stats_list = []
for group in df_lmm_ie['group'].unique():
    for ie in df_lmm_ie['inclusion_exclusion'].unique():
        subset = df_lmm_ie[(df_lmm_ie['group'] == group) & (df_lmm_ie['inclusion_exclusion'] == ie)]
        if len(subset) > 0:
            stats_list.append({
                'Group': group,
                'Inclusion_Exclusion': ie,
                'N': len(subset),
                'Mean_Raw': subset['onoff'].mean(),
                'SD_Raw': subset['onoff'].std(),
                'SE_Raw': subset['onoff'].sem(),
                'Mean_Normalized': subset['onoff_normalized'].mean(),
                'SD_Normalized': subset['onoff_normalized'].std(),
                'SE_Normalized': subset['onoff_normalized'].sem()
            })

stats_df = pd.DataFrame(stats_list)
stats_file = os.path.join(plots_output_dir, 'onoff_descriptive_statistics.csv')
stats_df.to_csv(stats_file, index=False)
print(f"\nDescriptive statistics (raw and normalized) saved to: {stats_file}")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Results saved to: {lmm_output_dir}")
print(f"Plots saved to: {plots_output_dir}")

# %%

