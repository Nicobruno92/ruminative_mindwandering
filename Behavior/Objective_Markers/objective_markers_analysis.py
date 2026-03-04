#%%
"""
Objective Markers Analysis at Probe Level

Analyzes objective markers of mind-wandering (omission errors, commission errors, RTCV)
aggregated at the probe level. Mirrors the analysis structure of probe_analysis_clean.py.

Objective markers:
- Omission rate: missed responses on go trials (failure to respond)
- Commission rate: false alarms on nogo trials (responded when shouldn't)
- RTCV: Reaction time coefficient of variation (RT_SD / RT_Mean)

Author: Analysis Assistant
"""

import os
import warnings
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import seaborn as sns
class _PT:
    def RainCloud(self, x, y, data, palette, order=None, ax=None, **kwargs):
        sns.violinplot(x=x, y=y, data=data, palette=palette, order=order, ax=ax, inner=None)
        sns.stripplot(x=x, y=y, data=data, palette=palette, order=order, ax=ax, dodge=True, alpha=0.7, size=4, jitter=0.2)
pt = _PT()
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to objective markers data (already aggregated per probe)
OBJECTIVE_MARKERS_PATH = '../../results/Behavior/objective_markers/objective_markers_per_probe.csv'
# Path to probe-level data with metadata
PROBE_DATA_PATH = '../../results/Behavior/probe_data/probe_level_aggregated_data.csv'

# Output directories
LMM_OUTPUT_DIR = '../../results/Behavior/objective_markers/lmm_analysis'
PLOTS_OUTPUT_DIR = '../../results/Behavior/objective_markers/lmm_plots'

# Objective markers to analyze
OBJECTIVE_MARKERS = ['omission_rate', 'commission_rate', 'rtcv', 'total_errors']
MARKER_LABELS = {
    'omission_rate': 'Omission Rate',
    'commission_rate': 'Commission Rate',
    'rtcv': 'RTCV (RT Variability)',
    'total_errors': 'Total Errors'
}

# N trials back from probe onset to include (set to None for all trials in window)
N_BACK_TO_PROBE = None  # e.g., 10 means only last 10 trials before probe

# Whether to apply within-subject z-scoring to objective markers before LMMs
# If True, each marker (and its baseline-corrected version) is z-scored
# within each subject across all available probes.
APPLY_WITHIN_SUBJECT_Z = True

# =============================================================================
# LOAD AND MERGE DATA
# =============================================================================
#%%
print("Loading objective markers data...")
df_markers = pd.read_csv(OBJECTIVE_MARKERS_PATH)
df_markers['total_errors'] = df_markers['omission_rate'] + df_markers['commission_rate']
print(f"Loaded {len(df_markers)} probe-level observations")

print("\nLoading probe metadata...")
df_probe = pd.read_csv(PROBE_DATA_PATH)
print(f"Loaded {len(df_probe)} probe observations with metadata")

# Merge objective markers with probe metadata
# Match on subject, task (sart), and probe_number
df_markers['subject'] = df_markers['subject'].astype(str)
df_probe['subject_id'] = df_probe['subject_id'].astype(str)

# Rename columns for merge
df_markers_renamed = df_markers.rename(columns={'sart': 'task'})

# Merge on subject, task, probe_number
df = pd.merge(
    df_markers_renamed,
    df_probe[['subject_id', 'task', 'probe_number', 'group', 'inclusion_exclusion', 
              'order (IE/EI)', 'onoff', 'valence', 'time', 'selfother', 'confidence']],
    left_on=['subject', 'task', 'probe_number'],
    right_on=['subject_id', 'task', 'probe_number'],
    how='inner'
)

print(f"\nMerged dataset: {len(df)} observations")
print(f"Unique subjects: {df['subject'].nunique()}")

#%%
# Display basic statistics
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS FOR OBJECTIVE MARKERS")
print("="*60)

for marker in OBJECTIVE_MARKERS:
    print(f"\n{MARKER_LABELS[marker]}:")
    print(df[marker].describe())

#%%
# Distribution plots
for marker in OBJECTIVE_MARKERS:
    fig = px.histogram(df, x=marker, nbins=50, 
                       title=f'Distribution of {MARKER_LABELS[marker]}')
    fig.show()

#%%
# Distribution by subject (violin plots)
subjects_sorted = sorted(df['subject'].unique())

for marker in OBJECTIVE_MARKERS:
    fig = px.violin(
        df,
        x=marker,
        y='subject',
        color='subject',
        category_orders={'subject': subjects_sorted},
        points='all',
        box=True,
        title=f'{MARKER_LABELS[marker]} Distribution by Subject'
    )
    fig.update_layout(width=600, height=1000)
    fig.update_layout(xaxis_title=MARKER_LABELS[marker], yaxis_title='Subject')
    fig.update_xaxes(tickangle=-45)
    fig.show()

#%%
# Correlations between objective markers and onoff
print("\n" + "="*60)
print("CORRELATIONS: OBJECTIVE MARKERS vs ON/OFF")
print("="*60)

for marker in OBJECTIVE_MARKERS:
    # Scatter plot with trend line
    fig = px.scatter(df, x='onoff', y=marker,
                     title=f'On/Off vs {MARKER_LABELS[marker]}',
                     labels={'onoff': 'On/Off', marker: MARKER_LABELS[marker]},
                     opacity=0.7)
    
    # Add trend line
    valid_data = df[['onoff', marker]].dropna()
    if len(valid_data) > 2:
        z = np.polyfit(valid_data['onoff'], valid_data[marker], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data['onoff'].min(), valid_data['onoff'].max(), 100)
        fig.add_trace(go.Scatter(x=x_range, y=p(x_range),
                                mode='lines',
                                line=dict(color='red', width=2),
                                name='Trend Line'))
    
    fig.update_layout(width=800, height=600)
    fig.show()
    
    # Correlation
    correlation = df['onoff'].corr(df[marker])
    print(f"Correlation between On/Off and {MARKER_LABELS[marker]}: {correlation:.3f}")
    print("-" * 50)

#%%
# Correlations between objective markers
print("\n" + "="*60)
print("CORRELATIONS: BETWEEN OBJECTIVE MARKERS")
print("="*60)

marker_pairs = list(combinations(OBJECTIVE_MARKERS, 2))

for marker1, marker2 in marker_pairs:
    fig = px.scatter(df, x=marker1, y=marker2,
                     title=f'{MARKER_LABELS[marker1]} vs {MARKER_LABELS[marker2]}',
                     labels={marker1: MARKER_LABELS[marker1], marker2: MARKER_LABELS[marker2]},
                     opacity=0.7)
    
    valid_data = df[[marker1, marker2]].dropna()
    if len(valid_data) > 2:
        z = np.polyfit(valid_data[marker1], valid_data[marker2], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid_data[marker1].min(), valid_data[marker1].max(), 100)
        fig.add_trace(go.Scatter(x=x_range, y=p(x_range),
                                mode='lines',
                                line=dict(color='red', width=2),
                                name='Trend Line'))
    
    fig.update_layout(width=800, height=600)
    fig.show()
    
    correlation = df[marker1].corr(df[marker2])
    print(f"Correlation between {MARKER_LABELS[marker1]} and {MARKER_LABELS[marker2]}: {correlation:.3f}")
    print("-" * 50)

#%%
# =============================================================================
# PREPARE DATA FOR LMM ANALYSIS
# =============================================================================

def prepare_lmm_data(df_orig: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for LMM analysis.
    
    Parameters
    ----------
    df_orig : pd.DataFrame
        Original merged dataframe
        
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for LMM analysis
    """
    df_with_metadata = df_orig.copy()
    df_with_metadata['subject'] = df_with_metadata['subject'].astype(str)
    
    # Normalize task labels
    def normalize_task_label(raw_task: str) -> str:
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
    print(f"Task filter: {before_task_filter_n} -> {df_tasks.shape[0]} rows kept")
    
    # Ensure inclusion_exclusion present (derive from order if missing)
    if 'inclusion_exclusion' not in df_tasks.columns or df_tasks['inclusion_exclusion'].isna().any():
        order_map = {}
        if 'order (IE/EI)' in df_tasks.columns:
            for _, r in df_tasks[['subject', 'order (IE/EI)']].drop_duplicates('subject').iterrows():
                order_map[str(r['subject'])] = r['order (IE/EI)']
        
        def map_task_to_condition(row):
            task = row['task']
            subj = str(row['subject'])
            if task in ['Sart1', 'Sart3']:
                return 'baseline'
            order = order_map.get(subj)
            if task == 'Sart2':
                return 'inclusion' if order == 'IE' else ('exclusion' if order == 'EI' else None)
            if task == 'Sart4':
                return 'exclusion' if order == 'IE' else ('inclusion' if order == 'EI' else None)
            return None
        
        df_tasks['inclusion_exclusion'] = df_tasks.apply(map_task_to_condition, axis=1)
    
    # Drop rows with missing inclusion_exclusion or group
    df_no_missing_ie = df_tasks.dropna(subset=['inclusion_exclusion'])
    print(f"After removing missing inclusion/exclusion: {df_no_missing_ie.shape[0]} rows")
    
    df_final = df_no_missing_ie.dropna(subset=['group']) if 'group' in df_no_missing_ie.columns else df_no_missing_ie
    if 'group' in df_no_missing_ie.columns:
        print(f"After removing missing group info: {df_final.shape[0]} rows")
    
    print(f"Final dataset for LMM: {df_final.shape[0]} rows from {df_final['subject'].nunique()} subjects")
    return df_final


print("\n" + "="*60)
print("PREPARING DATA FOR ANALYSIS")
print("="*60)

df_lmm = prepare_lmm_data(df)

print(f"\nDataset overview:")
print(f"- Total observations: {len(df_lmm)}")
print(f"- Unique subjects: {df_lmm['subject'].nunique()}")
group_counts = df_lmm.groupby('group')['subject'].nunique()
group_obs = df_lmm['group'].value_counts()
for group in group_counts.index:
    print(f"- {group}: {group_counts[group]} participants, {group_obs[group]} total observations")
print(f"- Inclusion/Exclusion distribution: {df_lmm['inclusion_exclusion'].value_counts().to_dict()}")

# Create IE-only subset for IE-specific analyses (Sart2/Sart4 only)
df_lmm_ie = df_lmm[df_lmm['inclusion_exclusion'].isin(['inclusion', 'exclusion'])].copy()
print(f"\nIE-only subset: {len(df_lmm_ie)} observations from {df_lmm_ie['subject'].nunique()} subjects")

#%%
# =============================================================================
# BASELINE NORMALIZATION FOR INCLUSION/EXCLUSION ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("APPLYING BASELINE NORMALIZATION")
print("="*60)

# Calculate baseline means per subject for each marker
baseline_means = {marker: {} for marker in OBJECTIVE_MARKERS}

for subject in df_lmm['subject'].unique():
    subject_data = df_lmm[df_lmm['subject'] == subject]
    
    for marker in OBJECTIVE_MARKERS:
        # Sart1 baseline (for normalizing Sart2)
        sart1_data = subject_data[subject_data['task'] == 'Sart1']
        if len(sart1_data) > 0:
            baseline_means[marker][(subject, 'Sart1')] = sart1_data[marker].mean()
        
        # Sart3 baseline (for normalizing Sart4)
        sart3_data = subject_data[subject_data['task'] == 'Sart3']
        if len(sart3_data) > 0:
            baseline_means[marker][(subject, 'Sart3')] = sart3_data[marker].mean()

print(f"Calculated baseline means for {len(baseline_means[OBJECTIVE_MARKERS[0]])} subject-SART combinations")


def normalize_by_baseline(row: pd.Series, marker: str) -> float:
    """
    Normalize marker score by subtracting the appropriate baseline mean.
    
    Parameters
    ----------
    row : pd.Series
        Row from dataframe
    marker : str
        Name of the marker column
        
    Returns
    -------
    float
        Baseline-normalized value
    """
    subject = row['subject']
    task = row['task']
    
    if task == 'Sart2':
        baseline_key = (subject, 'Sart1')
    elif task == 'Sart4':
        baseline_key = (subject, 'Sart3')
    else:
        return np.nan
    
    baseline_mean = baseline_means[marker].get(baseline_key, np.nan)
    if pd.isna(baseline_mean):
        return np.nan
    
    return row[marker] - baseline_mean


# Apply normalization for each marker
for marker in OBJECTIVE_MARKERS:
    normalized_col = f'{marker}_normalized'
    df_lmm_ie[normalized_col] = df_lmm_ie.apply(lambda row: normalize_by_baseline(row, marker), axis=1)

# Remove rows where normalization failed
before_dropna = len(df_lmm_ie)
df_lmm_ie = df_lmm_ie.dropna(subset=[f'{m}_normalized' for m in OBJECTIVE_MARKERS])
after_dropna = len(df_lmm_ie)
if before_dropna > after_dropna:
    print(f"Warning: Removed {before_dropna - after_dropna} rows with missing baseline data")

print(f"Normalized IE subset: {len(df_lmm_ie)} observations")

for marker in OBJECTIVE_MARKERS:
    normalized_col = f'{marker}_normalized'
    print(f"{MARKER_LABELS[marker]} normalized range: [{df_lmm_ie[normalized_col].min():.4f}, {df_lmm_ie[normalized_col].max():.4f}]")

#%%
# Create time_on_task variable
df_lmm['sart_number'] = df_lmm['task'].str.extract(r'(\d+)').astype(int)
df_lmm['time_on_task'] = df_lmm['probe_number'] + (15 * (df_lmm['sart_number'] - 1))
df_lmm['relative_time_on_task'] = df_lmm['probe_number']

df_lmm_ie['sart_number'] = df_lmm_ie['task'].str.extract(r'(\d+)').astype(int)
df_lmm_ie['time_on_task'] = df_lmm_ie['probe_number'] + (15 * (df_lmm_ie['sart_number'] - 1))
df_lmm_ie['relative_time_on_task'] = df_lmm_ie['probe_number']

print(f"Added time_on_task variable: range {df_lmm['time_on_task'].min()} to {df_lmm['time_on_task'].max()}")

#%%
# =============================================================================
# OPTIONAL WITHIN-SUBJECT Z-SCORING
# =============================================================================

def apply_within_subject_z_scoring(
    df_all: pd.DataFrame,
    df_ie_only: pd.DataFrame,
    markers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply within-subject z-scoring to markers (and normalized variants).

    Z-scoring is done separately for each subject using that subject's
    distribution across all available probes. Baseline-corrected variables
    (``<marker>_normalized``) are also z-scored within subject when present.
    """

    df_all_z = df_all.copy()
    df_ie_z = df_ie_only.copy()

    # Z-score raw markers within subject in df_all
    for marker in markers:
        if marker not in df_all_z.columns:
            continue
        df_all_z[marker] = df_all_z.groupby('subject')[marker].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) not in (0, np.nan) else x
        )

    # Z-score raw and baseline-normalized markers within subject in IE-only data
    for marker in markers:
        if marker in df_ie_z.columns:
            df_ie_z[marker] = df_ie_z.groupby('subject')[marker].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) not in (0, np.nan) else x
            )

        normalized_col = f"{marker}_normalized"
        if normalized_col in df_ie_z.columns:
            df_ie_z[normalized_col] = df_ie_z.groupby('subject')[normalized_col].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) not in (0, np.nan) else x
            )

    return df_all_z, df_ie_z


if APPLY_WITHIN_SUBJECT_Z:
    print("\n" + "="*60)
    print("APPLYING WITHIN-SUBJECT Z-SCORING TO OBJECTIVE MARKERS")
    print("="*60)
    df_lmm, df_lmm_ie = apply_within_subject_z_scoring(df_lmm, df_lmm_ie, OBJECTIVE_MARKERS)

#%%
# =============================================================================
# LINEAR MIXED MODEL ANALYSIS
# =============================================================================

os.makedirs(LMM_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


def _extract_formula_variables(formula: str) -> list[str]:
    """Extract variable names from a simple Patsy-style formula RHS.

    This is a lightweight parser that handles ``+``, ``*``, and ``:`` by
    expanding them into a set of unique variable names. It assumes that
    variable names themselves do not contain these operators.
    """

    # Replace interaction and product operators with additive form
    cleaned = formula.replace("*", "+").replace(":", "+")
    tokens = [t.strip() for t in cleaned.split("+")]
    vars_unique: set[str] = {t for t in tokens if t not in ("", "1")}
    return sorted(vars_unique)


def run_lmm_analysis(data: pd.DataFrame, dependent_var: str, formula: str, 
                     model_name: str, output_dir: str) -> tuple:
    """
    Run linear mixed model analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data for analysis
    dependent_var : str
        Name of dependent variable
    formula : str
        Model formula (without dependent variable)
    model_name : str
        Name for saving results
    output_dir : str
        Output directory path
        
    Returns
    -------
    tuple
        (results_df, model) or (None, None) if error
    """
    print(f"\n=== FITTING MODEL: {model_name} ===")
    print(f"Formula: {dependent_var} ~ {formula}")
    print(f"Sample size (raw): {len(data)} observations from {data['subject'].nunique()} subjects")

    # ------------------------------------------------------------------
    # Clean data before fitting to avoid indexing issues in statsmodels
    # ------------------------------------------------------------------
    predictor_vars = _extract_formula_variables(formula)
    cols_needed = ['subject', dependent_var] + predictor_vars
    cols_present = [c for c in cols_needed if c in data.columns]

    model_data = data[cols_present].copy()
    model_data = model_data.dropna(subset=[c for c in cols_present if c != 'subject'])
    model_data['subject'] = model_data['subject'].astype(str)
    model_data = model_data.reset_index(drop=True)

    print(f"Sample size (after dropna/reset): {len(model_data)} observations from {model_data['subject'].nunique()} subjects")

    full_formula = f"{dependent_var} ~ {formula}"
    model = smf.mixedlm(full_formula, model_data, groups="subject").fit()

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

    results_df['significant_05'] = results_df['p_value'] < 0.05
    results_df['significant_01'] = results_df['p_value'] < 0.01

    # Model fit metrics
    n_groups = model_data['subject'].nunique()

    aic_value = model.aic if hasattr(model, 'aic') and not pd.isna(model.aic) else None
    bic_value = model.bic if hasattr(model, 'bic') and not pd.isna(model.bic) else None

    if aic_value is None:
        k = len(model.params)
        aic_value = -2 * model.llf + 2 * k

    if bic_value is None:
        k = len(model.params)
        n = model.nobs
        bic_value = -2 * model.llf + k * np.log(n)

    model_metrics = {
        'aic': aic_value,
        'bic': bic_value,
        'log_likelihood': model.llf,
        'n_observations': model.nobs,
        'n_groups': n_groups,
        'n_parameters': len(model.params),
        'converged': model.converged,
        'scale': model.scale if hasattr(model, 'scale') else None
    }

    print("\n=== MODEL FIT METRICS ===")
    print(f"AIC: {model_metrics['aic']:.3f}")
    print(f"BIC: {model_metrics['bic']:.3f}")
    print(f"Log-Likelihood: {model_metrics['log_likelihood']:.3f}")
    print(f"N observations: {model_metrics['n_observations']}")
    print(f"N groups: {model_metrics['n_groups']}")
    print(f"Converged: {model_metrics['converged']}")

    # Save results
    results_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_results.csv')
    results_df.to_csv(results_file, index=False)

    metrics_df = pd.DataFrame([model_metrics])
    metrics_file = os.path.join(output_dir, f'{model_name}_{dependent_var}_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)

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

    print(f"Results saved to: {results_file}")

    return results_df, model


#%%
# =============================================================================
# RUN LMM MODELS FOR EACH OBJECTIVE MARKER
# =============================================================================

all_results = {}

for marker in OBJECTIVE_MARKERS:
    print("\n" + "="*80)
    print(f"ANALYZING: {MARKER_LABELS[marker].upper()}")
    print("="*80)
    
    normalized_marker = f'{marker}_normalized'
    marker_results = {}
    
    # Model 1: Group effect
    print("\n" + "="*60)
    print(f"MODEL 1: {marker.upper()} ~ GROUP")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm, marker, 'group', 'group_effect', LMM_OUTPUT_DIR
    )
    marker_results['group'] = (results, model)
    
    # Model 2: Inclusion/Exclusion effect (NORMALIZED)
    print("\n" + "="*60)
    print(f"MODEL 2: {normalized_marker.upper()} ~ INCLUSION/EXCLUSION")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm_ie, normalized_marker, 'inclusion_exclusion', 'inclusion_exclusion_effect', LMM_OUTPUT_DIR
    )
    marker_results['ie'] = (results, model)
    
    # Model 3: Interaction effect (Group × Inclusion/Exclusion) (NORMALIZED)
    print("\n" + "="*60)
    print(f"MODEL 3: {normalized_marker.upper()} ~ GROUP * INCLUSION/EXCLUSION")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm_ie, normalized_marker, 'group * inclusion_exclusion', 'group_ie_interaction', LMM_OUTPUT_DIR
    )
    marker_results['group_ie'] = (results, model)
    
    # Model 4: Group + Time on task
    print("\n" + "="*60)
    print(f"MODEL 4: {marker.upper()} ~ GROUP + TIME_ON_TASK")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm, marker, 'group + time_on_task', 'group_time_additive', LMM_OUTPUT_DIR
    )
    marker_results['group_time'] = (results, model)
    
    # Model 5: Group × Time on task interaction
    print("\n" + "="*60)
    print(f"MODEL 5: {marker.upper()} ~ GROUP * TIME_ON_TASK")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm, marker, 'group * time_on_task', 'group_time_interaction', LMM_OUTPUT_DIR
    )
    marker_results['group_time_int'] = (results, model)
    
    # Model 6: Inclusion/Exclusion + Intervention distance (NORMALIZED)
    print("\n" + "="*60)
    print(f"MODEL 6: {normalized_marker.upper()} ~ INCLUSION/EXCLUSION + PROBE_NUMBER")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm_ie, normalized_marker, 'inclusion_exclusion + probe_number', 'ie_intervention_distance', LMM_OUTPUT_DIR
    )
    marker_results['ie_distance'] = (results, model)
    
    # Model 7: Group × Inclusion/Exclusion × Intervention distance (NORMALIZED)
    print("\n" + "="*60)
    print(f"MODEL 7: {normalized_marker.upper()} ~ GROUP * INCLUSION/EXCLUSION * PROBE_NUMBER")
    print("="*60)
    results, model = run_lmm_analysis(
        df_lmm_ie, normalized_marker, 'group * inclusion_exclusion * probe_number', 'three_way_with_distance', LMM_OUTPUT_DIR
    )
    marker_results['three_way'] = (results, model)
    
    all_results[marker] = marker_results

#%%
# =============================================================================
# ONE-SAMPLE TESTS: Are inclusion/exclusion effects different from zero?
# =============================================================================

print("\n" + "="*60)
print("ONE-SAMPLE TESTS AGAINST BASELINE (H0: effect = 0)")
print("="*60)

one_sample_results_all = []

for marker in OBJECTIVE_MARKERS:
    normalized_marker = f'{marker}_normalized'
    print(f"\n--- {MARKER_LABELS[marker]} ---")
    
    # Test 1: Is INCLUSION effect different from zero?
    inclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'inclusion'][normalized_marker].dropna()
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
    
    # Test 2: Is EXCLUSION effect different from zero?
    exclusion_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == 'exclusion'][normalized_marker].dropna()
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
    
    # Bonferroni correction
    p_bonf_incl = min(p_incl * 2, 1.0)
    p_bonf_excl = min(p_excl * 2, 1.0)
    
    one_sample_results_all.extend([
        {
            'Marker': marker,
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
            'Marker': marker,
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

one_sample_df = pd.DataFrame(one_sample_results_all)
one_sample_file = os.path.join(LMM_OUTPUT_DIR, 'one_sample_tests_vs_baseline.csv')
one_sample_df.to_csv(one_sample_file, index=False)
print(f"\nOne-sample test results saved to: {one_sample_file}")

#%%
# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Define colors
group_colors = ['#2E86AB', '#F24236']
ie_colors = ['#A23B72', '#F18F01']
group_order = ['Controls', 'Risk of Depression']
ie_order = ['inclusion', 'exclusion']

plt.style.use('default')

for marker in OBJECTIVE_MARKERS:
    normalized_marker = f'{marker}_normalized'
    marker_label = MARKER_LABELS[marker]
    
    print(f"\nCreating plots for {marker_label}...")
    
    # =========================================================================
    # TIME-ON-TASK VISUALIZATION (4-panel)
    # =========================================================================
    fig_time, axes_time = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot 1: Overall time-on-task trend
    ax1 = axes_time[0, 0]
    time_agg = df_lmm.groupby('time_on_task')[marker].agg(['mean', 'sem']).reset_index()
    ax1.errorbar(time_agg['time_on_task'], time_agg['mean'], yerr=time_agg['sem'],
                marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8, color='#2E86AB')
    ax1.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
    ax1.set_ylabel(marker_label, fontsize=14, fontweight='bold')
    ax1.set_title(f'{marker_label} Across Time on Task', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time-on-task by group
    ax2 = axes_time[0, 1]
    for i, group in enumerate(['Controls', 'Risk of Depression']):
        group_data = df_lmm[df_lmm['group'] == group]
        time_group_agg = group_data.groupby('time_on_task')[marker].agg(['mean', 'sem']).reset_index()
        color = group_colors[i]
        ax2.errorbar(time_group_agg['time_on_task'], time_group_agg['mean'], yerr=time_group_agg['sem'],
                    marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8,
                    color=color, label=group)
    ax2.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(marker_label, fontsize=14, fontweight='bold')
    ax2.set_title(f'Time-on-Task Effect by Group', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Intervention distance by inclusion/exclusion (NORMALIZED)
    ax3 = axes_time[1, 0]
    for i, condition in enumerate(['inclusion', 'exclusion']):
        ie_data = df_lmm_ie[df_lmm_ie['inclusion_exclusion'] == condition]
        probe_ie_agg = ie_data.groupby('probe_number')[normalized_marker].agg(['mean', 'sem']).reset_index()
        color = ie_colors[i]
        ax3.errorbar(probe_ie_agg['probe_number'], probe_ie_agg['mean'], yerr=probe_ie_agg['sem'],
                    marker='s', linewidth=2, markersize=6, capsize=3, alpha=0.8,
                    color=color, label=condition.capitalize())
    ax3.set_xlabel('Intervention Distance (Probe Number)', fontsize=14, fontweight='bold')
    ax3.set_ylabel(f'{marker_label} (Normalized)', fontsize=14, fontweight='bold')
    ax3.set_title(f'Intervention Distance by I/E (Baseline-Corrected)', fontsize=16, fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.5, 15.5)
    
    # Plot 4: Correlation scatter
    ax4 = axes_time[1, 1]
    scatter = ax4.scatter(df_lmm['time_on_task'], df_lmm[marker],
                         c=df_lmm['group'].map({'Controls': 0, 'Risk of Depression': 1}),
                         cmap='RdBu_r', alpha=0.6, s=30)
    slope, intercept, r_value, p_value, std_err = linregress(df_lmm['time_on_task'], df_lmm[marker])
    x_trend = np.linspace(df_lmm['time_on_task'].min(), df_lmm['time_on_task'].max(), 100)
    y_trend = slope * x_trend + intercept
    ax4.plot(x_trend, y_trend, 'r-', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Time on Task (Probe Number)', fontsize=14, fontweight='bold')
    ax4.set_ylabel(marker_label, fontsize=14, fontweight='bold')
    ax4.set_title(f'Time vs {marker_label} (r={r_value:.3f}, p={p_value:.3f})', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Controls', 'Risk of Depression'])
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, f'{marker}_time_on_task_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, f'{marker}_time_on_task_analysis.svg'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # =========================================================================
    # COMPREHENSIVE 2x3 VISUALIZATION
    # =========================================================================
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)
    
    # Plot 1 (0,0): Raincloud for group comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_agg_group = df_lmm.groupby(["subject", "group"])[marker].mean().reset_index()
    n_participants_by_group = df_agg_group.groupby("group")["subject"].nunique().to_dict()
    
    pt.RainCloud(
        x="group",
        y=marker,
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
    ax1.set_title(f"{marker_label}: Group Effect", fontsize=18, fontweight="bold", pad=15)
    ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
    ax1.set_ylabel(marker_label, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    for i, group in enumerate(group_order):
        n = n_participants_by_group.get(group, 0)
        ax1.text(i, ax1.get_ylim()[1] * 0.95, f"n={n}",
                ha="center", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2 (0,1): Raincloud for inclusion/exclusion (NORMALIZED)
    ax2 = fig.add_subplot(gs[0, 1])
    df_agg_ie = df_lmm_ie.groupby(["subject", "inclusion_exclusion"])[normalized_marker].mean().reset_index()
    n_participants_by_ie = df_agg_ie.groupby("inclusion_exclusion")["subject"].nunique().to_dict()
    
    pt.RainCloud(
        x="inclusion_exclusion",
        y=normalized_marker,
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
    ax2.set_title(f"{marker_label}: I/E Effect (Baseline-Corrected)", fontsize=18, fontweight="bold", pad=15)
    ax2.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax2.set_ylabel(f"{marker_label} (Normalized)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax2.legend(fontsize=10, loc='best')
    
    for i, condition in enumerate(ie_order):
        n = n_participants_by_ie.get(condition, 0)
        ax2.text(i, ax2.get_ylim()[1] * 0.95, f"n={n}",
                ha="center", va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3 (0,2): Interaction plot (Group × I/E) - NORMALIZED
    ax3 = fig.add_subplot(gs[0, 2])
    for group in df_lmm_ie['group'].dropna().unique():
        group_data = df_lmm_ie[df_lmm_ie['group'] == group]
        if len(group_data) == 0:
            continue
        ie_means = (
            group_data.groupby("inclusion_exclusion")[normalized_marker]
            .agg(["mean", "sem"])
            .reindex(ie_order)
            .reset_index()
        )
        n_participants_group = group_data["subject"].nunique()
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
    ax3.set_ylabel(f"{marker_label} (Normalized)", fontsize=14, fontweight="bold")
    ax3.set_title(f"{marker_label}: Group × I/E Interaction", fontsize=18, fontweight="bold", pad=15)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax3.legend(fontsize=12, title_fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4 (1,0): Time-on-task by group
    ax4 = fig.add_subplot(gs[1, 0])
    for i, group in enumerate(group_order):
        if group in df_lmm["group"].values:
            group_data = df_lmm[df_lmm["group"] == group]
            time_group_agg = group_data.groupby("time_on_task")[marker].agg(["mean", "sem"]).reset_index()
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
    ax4.set_ylabel(marker_label, fontsize=14, fontweight="bold")
    ax4.set_title(f"{marker_label}: Time-on-Task by Group", fontsize=18, fontweight="bold", pad=15)
    ax4.legend(fontsize=12, loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5 (1,1): Intervention distance by inclusion/exclusion (NORMALIZED)
    ax5 = fig.add_subplot(gs[1, 1])
    for i, condition in enumerate(ie_order):
        if condition in df_lmm_ie["inclusion_exclusion"].values:
            ie_data = df_lmm_ie[df_lmm_ie["inclusion_exclusion"] == condition]
            probe_ie_agg = ie_data.groupby("probe_number")[normalized_marker].agg(["mean", "sem"]).reset_index()
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
    ax5.set_ylabel(f"{marker_label} (Normalized)", fontsize=14, fontweight="bold")
    ax5.set_title(f"{marker_label}: Intervention Distance by I/E", fontsize=18, fontweight="bold", pad=15)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
    ax5.legend(fontsize=12, loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0.5, 15.5)
    
    # Plot 6 (1,2): SART Mean Trajectories by Group and Order
    ax6 = fig.add_subplot(gs[1, 2])
    if 'order (IE/EI)' in df_lmm.columns:
        sart_order_list = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
        x_positions = [1, 2, 3, 4]
        order_styles = {'IE': '-', 'EI': '--'}
        
        for group_idx, group in enumerate(group_order):
            if group not in df_lmm["group"].values:
                continue
            group_data = df_lmm[df_lmm["group"] == group]
            color = group_colors[group_idx]
            
            for order in ['IE', 'EI']:
                if order not in group_data['order (IE/EI)'].values:
                    continue
                order_data = group_data[group_data['order (IE/EI)'] == order]
                n_subjects = order_data['subject'].nunique()
                
                means = []
                sems = []
                for sart in sart_order_list:
                    sart_data = order_data[order_data['task'] == sart]
                    if len(sart_data) > 0:
                        means.append(sart_data[marker].mean())
                        sems.append(sart_data[marker].sem())
                    else:
                        means.append(np.nan)
                        sems.append(np.nan)
                
                linestyle = order_styles[order]
                label = f"{group} - {order} (n={n_subjects})"
                
                ax6.errorbar(
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
        ax6.set_ylabel(marker_label, fontsize=14, fontweight="bold")
        ax6.set_title(f"{marker_label}: SART Trajectory by Group & Order\n(Solid=IE, Dashed=EI)",
                     fontsize=18, fontweight="bold", pad=15)
        ax6.legend(fontsize=11, loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0.5, 4.5)
    else:
        ax6.text(0.5, 0.5, 'Order data not available',
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
    
    plt.suptitle(f"Comprehensive Analysis: {marker_label}", fontsize=22, fontweight="bold", y=0.995)
    
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, f'{marker}_comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, f'{marker}_comprehensive_analysis.svg'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%%
# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS FOR OBJECTIVE MARKERS")
print("="*60)

for marker in OBJECTIVE_MARKERS:
    normalized_marker = f'{marker}_normalized'
    marker_label = MARKER_LABELS[marker]
    
    print(f"\n{'='*60}")
    print(f"{marker_label.upper()}")
    print("="*60)
    
    # Overall statistics
    overall_stats = df_lmm[marker].describe()
    print(f"Overall: Mean = {overall_stats['mean']:.4f}, SD = {overall_stats['std']:.4f}")
    
    # By group
    print("\nBy Group:")
    group_stats = df_lmm.groupby('group')[marker].agg(['count', 'mean', 'std']).round(4)
    print(group_stats)
    
    # By inclusion/exclusion (RAW)
    print("\nBy Inclusion/Exclusion (Raw):")
    ie_stats = df_lmm_ie.groupby('inclusion_exclusion')[marker].agg(['count', 'mean', 'std']).round(4)
    print(ie_stats)
    
    # By inclusion/exclusion (NORMALIZED)
    print("\nBy Inclusion/Exclusion (Baseline-Corrected):")
    ie_stats_norm = df_lmm_ie.groupby('inclusion_exclusion')[normalized_marker].agg(['count', 'mean', 'std']).round(4)
    print(ie_stats_norm)
    
    # By group × inclusion/exclusion (NORMALIZED)
    print("\nBy Group × Inclusion/Exclusion (Baseline-Corrected):")
    interaction_stats_norm = df_lmm_ie.groupby(['group', 'inclusion_exclusion'])[normalized_marker].agg(['count', 'mean', 'std']).round(4)
    print(interaction_stats_norm)

# Save descriptive statistics
stats_list = []
for marker in OBJECTIVE_MARKERS:
    normalized_marker = f'{marker}_normalized'
    for group in df_lmm_ie['group'].unique():
        for ie in df_lmm_ie['inclusion_exclusion'].unique():
            subset = df_lmm_ie[(df_lmm_ie['group'] == group) & (df_lmm_ie['inclusion_exclusion'] == ie)]
            if len(subset) > 0:
                stats_list.append({
                    'Marker': marker,
                    'Group': group,
                    'Inclusion_Exclusion': ie,
                    'N': len(subset),
                    'Mean_Raw': subset[marker].mean(),
                    'SD_Raw': subset[marker].std(),
                    'SE_Raw': subset[marker].sem(),
                    'Mean_Normalized': subset[normalized_marker].mean(),
                    'SD_Normalized': subset[normalized_marker].std(),
                    'SE_Normalized': subset[normalized_marker].sem()
                })

stats_df = pd.DataFrame(stats_list)
stats_file = os.path.join(PLOTS_OUTPUT_DIR, 'objective_markers_descriptive_statistics.csv')
stats_df.to_csv(stats_file, index=False)
print(f"\nDescriptive statistics saved to: {stats_file}")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"LMM results saved to: {LMM_OUTPUT_DIR}")
print(f"Plots saved to: {PLOTS_OUTPUT_DIR}")

# %%
