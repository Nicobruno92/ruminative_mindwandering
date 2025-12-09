#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTC Zone Analysis

Analyzes the relationship between VTC zones (In-the-Zone vs Out-of-the-Zone)
and subjective mind-wandering reports (on/off task).

Key analyses:
1. Proportion of time out-of-the-zone by group and time on task
2. Relationship between zone state and on/off ratings
3. Percentage of time out-of-the-zone within on-task vs off-task states

Based on methodology from:
- Esterman et al. (2013) - VTC methodology
- Adapted for thought probes with block-wise smoothing

Author: Analysis Assistant
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# ROOT = "/network/iss/"
ROOT = "/Volumes/"
BIDS_RAW_ROOT = ROOT + "cenir/analyse/meeg/CYBERSART/BIDS/raw"

# Input: VTC analysis results
VTC_TRIAL_PATH = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/vtc_analysis/vtc_trial_level.csv"
VTC_PROBE_PATH = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/vtc_analysis/vtc_probe_level.csv"

# Probe-level data with group info
PROBE_DATA_PATH = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/probe_data/probe_level_aggregated_data.csv"

# Output directories
OUTPUT_DIR = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/vtc_analysis"
PLOTS_DIR = ROOT + "levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/vtc_analysis/plots"

# On/Off threshold for categorizing subjective state
# Scale: 0 = off-task, 100 = on-task (HIGH values = ON-TASK)
ONOFF_THRESHOLD = 50  # Values ABOVE = on-task, BELOW = off-task

# Number of trials before probe to use for zone vs onoff analysis
# This restricts the analysis to the N trials immediately preceding each probe
N_TRIALS_BEFORE_PROBE = 20

# Group and condition labels
GROUP_ORDER = ['Controls', 'Risk of Depression']
GROUP_COLORS = ['#2E86AB', '#F24236']

TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load VTC data and merge with probe metadata (group info).
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (trial_level_df, probe_level_df) with group information
    """
    # Load VTC data
    df_trial = pd.read_csv(VTC_TRIAL_PATH)
    df_probe_vtc = pd.read_csv(VTC_PROBE_PATH)
    
    # Load probe metadata
    df_probe_meta = pd.read_csv(PROBE_DATA_PATH)
    
    # Standardize column names for merging
    df_trial['subject'] = df_trial['subject'].astype(str)
    df_probe_vtc['subject'] = df_probe_vtc['subject'].astype(str)
    df_probe_meta['subject_id'] = df_probe_meta['subject_id'].astype(str)
    
    # Rename task column for consistency
    df_trial = df_trial.rename(columns={'sart': 'task'})
    df_probe_vtc = df_probe_vtc.rename(columns={'sart': 'task'})
    
    # Merge probe-level VTC with metadata
    df_probe = pd.merge(
        df_probe_vtc,
        df_probe_meta[['subject_id', 'task', 'probe_number', 'group', 'inclusion_exclusion', 'order (IE/EI)']],
        left_on=['subject', 'task', 'probe_number'],
        right_on=['subject_id', 'task', 'probe_number'],
        how='left'
    )
    
    # Merge trial-level with group info
    subject_group_map = df_probe[['subject', 'group']].drop_duplicates().set_index('subject')['group'].to_dict()
    df_trial['group'] = df_trial['subject'].map(subject_group_map)
    
    # Add time on task variables
    df_trial['sart_number'] = df_trial['task'].str.extract(r'(\d+)').astype(int)
    df_trial['time_on_task'] = df_trial['probe_number'] + (15 * (df_trial['sart_number'] - 1))
    
    df_probe['sart_number'] = df_probe['task'].str.extract(r'(\d+)').astype(int)
    df_probe['time_on_task'] = df_probe['probe_number'] + (15 * (df_probe['sart_number'] - 1))
    
    return df_trial, df_probe


def calculate_out_of_zone_proportion(df_trial: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate proportion of time out-of-the-zone per probe.
    
    Parameters
    ----------
    df_trial : pd.DataFrame
        Trial-level data with zone_state column
        
    Returns
    -------
    pd.DataFrame
        Probe-level proportions
    """
    probe_props = []
    
    for (subj, task, probe), group_df in df_trial.groupby(['subject', 'task', 'probe_number']):
        n_total = len(group_df)
        n_out_zone = (group_df['zone_state'] == 'out_zone').sum()
        n_in_zone = (group_df['zone_state'] == 'in_zone').sum()
        
        prop_out = n_out_zone / n_total if n_total > 0 else np.nan
        prop_in = n_in_zone / n_total if n_total > 0 else np.nan
        
        # Get probe ratings
        onoff = group_df['onoff'].iloc[0] if 'onoff' in group_df.columns else np.nan
        
        probe_props.append({
            'subject': subj,
            'task': task,
            'probe_number': probe,
            'n_trials': n_total,
            'n_out_zone': n_out_zone,
            'n_in_zone': n_in_zone,
            'prop_out_zone': prop_out,
            'prop_in_zone': prop_in,
            'onoff': onoff
        })
    
    return pd.DataFrame(probe_props)


def calculate_out_zone_by_subjective_state(df_trial: pd.DataFrame, 
                                            onoff_threshold: float = ONOFF_THRESHOLD,
                                            n_trials_before_probe: int = None) -> pd.DataFrame:
    """
    Calculate percentage of time out-of-the-zone within each subjective state.
    
    Following the paper methodology:
    "We computed the duration in seconds spent in a state of high RT variability 
    within each subjective state—either on-task or off-task."
    
    Parameters
    ----------
    df_trial : pd.DataFrame
        Trial-level data with zone_state and onoff columns
    onoff_threshold : float
        Threshold for categorizing on-task vs off-task
    n_trials_before_probe : int, optional
        If specified, only use the N trials immediately before each probe.
        Uses 'distance_to_probe' column to filter trials.
        
    Returns
    -------
    pd.DataFrame
        Subject-level summary of out-of-zone time by subjective state
    """
    # Filter to only N trials before probe if specified
    if n_trials_before_probe is not None and 'distance_to_probe' in df_trial.columns:
        # distance_to_probe is negative (e.g., -30 to -1 before probe)
        # We want trials where distance_to_probe >= -n_trials_before_probe
        df_filtered = df_trial[df_trial['distance_to_probe'] >= -n_trials_before_probe].copy()
        print(f"  Filtering to {n_trials_before_probe} trials before each probe: "
              f"{len(df_filtered)}/{len(df_trial)} trials retained")
    else:
        df_filtered = df_trial.copy()
    
    results = []
    
    for subject in df_filtered['subject'].unique():
        subj_data = df_filtered[df_filtered['subject'] == subject]
        
        # Categorize probes as on-task or off-task based on onoff rating
        # Higher onoff = more on-task, lower = more off-task (0=off, 100=on)
        on_task_trials = subj_data[subj_data['onoff'] >= onoff_threshold]
        off_task_trials = subj_data[subj_data['onoff'] < onoff_threshold]
        
        # Calculate proportion out-of-zone within each subjective state
        n_on_task = len(on_task_trials)
        n_off_task = len(off_task_trials)
        
        n_out_zone_on_task = (on_task_trials['zone_state'] == 'out_zone').sum()
        n_out_zone_off_task = (off_task_trials['zone_state'] == 'out_zone').sum()
        
        prop_out_zone_on_task = n_out_zone_on_task / n_on_task if n_on_task > 0 else np.nan
        prop_out_zone_off_task = n_out_zone_off_task / n_off_task if n_off_task > 0 else np.nan
        
        # Get group
        group = subj_data['group'].iloc[0] if 'group' in subj_data.columns else np.nan
        
        results.append({
            'subject': subject,
            'group': group,
            'n_on_task_trials': n_on_task,
            'n_off_task_trials': n_off_task,
            'n_out_zone_on_task': n_out_zone_on_task,
            'n_out_zone_off_task': n_out_zone_off_task,
            'prop_out_zone_on_task': prop_out_zone_on_task,
            'prop_out_zone_off_task': prop_out_zone_off_task,
            'mean_onoff': subj_data['onoff'].mean()
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
#%%
print("=" * 60)
print("VTC ZONE ANALYSIS")
print("=" * 60)

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and merge data
print("\nLoading data...")
df_trial, df_probe = load_and_merge_data()

print(f"Trial-level data: {len(df_trial)} observations")
print(f"Probe-level data: {len(df_probe)} observations")
print(f"Unique subjects: {df_trial['subject'].nunique()}")

# Filter to subjects with group info
df_trial = df_trial.dropna(subset=['group'])
df_probe = df_probe.dropna(subset=['group'])

print(f"After filtering for group info: {df_trial['subject'].nunique()} subjects")

#%%
# =============================================================================
# ANALYSIS 1: OUT-OF-ZONE PROPORTION BY GROUP AND TIME ON TASK
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: OUT-OF-ZONE BY GROUP AND TIME ON TASK")
print("=" * 60)

# Calculate probe-level proportions
df_probe_props = calculate_out_of_zone_proportion(df_trial)

# Merge with group info
subject_group_map = df_trial[['subject', 'group']].drop_duplicates().set_index('subject')['group'].to_dict()
df_probe_props['group'] = df_probe_props['subject'].map(subject_group_map)
df_probe_props['sart_number'] = df_probe_props['task'].str.extract(r'(\d+)').astype(int)
df_probe_props['time_on_task'] = df_probe_props['probe_number'] + (15 * (df_probe_props['sart_number'] - 1))

# Descriptive statistics
print("\nDescriptive Statistics - Proportion Out-of-Zone:")
print(df_probe_props.groupby('group')['prop_out_zone'].agg(['mean', 'std', 'sem']))

#%%
# Plot 1: Time-on-task by group (4-panel figure)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel A: Overall time-on-task trend
ax1 = axes[0, 0]
time_agg = df_probe_props.groupby('time_on_task')['prop_out_zone'].agg(['mean', 'sem']).reset_index()
ax1.errorbar(time_agg['time_on_task'], time_agg['mean'], yerr=time_agg['sem'],
             marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8, color='#2E86AB')
ax1.set_xlabel('Time on Task (Probe Number)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax1.set_title('A. Out-of-Zone Across Time on Task', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Panel B: Time-on-task by group
ax2 = axes[0, 1]
for i, group in enumerate(GROUP_ORDER):
    group_data = df_probe_props[df_probe_props['group'] == group]
    if len(group_data) == 0:
        continue
    time_group_agg = group_data.groupby('time_on_task')['prop_out_zone'].agg(['mean', 'sem']).reset_index()
    ax2.errorbar(time_group_agg['time_on_task'], time_group_agg['mean'], yerr=time_group_agg['sem'],
                 marker='o', linewidth=2, markersize=6, capsize=3, alpha=0.8,
                 color=GROUP_COLORS[i], label=group)
ax2.set_xlabel('Time on Task (Probe Number)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax2.set_title('B. Out-of-Zone by Group Across Time', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

# Panel C: Group comparison (raincloud)
ax3 = axes[1, 0]
df_subj_mean = df_probe_props.groupby(['subject', 'group'])['prop_out_zone'].mean().reset_index()
pt.RainCloud(
    x='group', y='prop_out_zone', data=df_subj_mean,
    palette=GROUP_COLORS, order=GROUP_ORDER,
    bw=0.2, width_viol=0.6, alpha=0.7,
    dodge=True, pointplot=True, move=-0.1, ax=ax3
)
ax3.set_xlabel('Group', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax3.set_title('C. Group Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add sample sizes
for i, group in enumerate(GROUP_ORDER):
    n = df_subj_mean[df_subj_mean['group'] == group]['subject'].nunique()
    ax3.text(i, ax3.get_ylim()[1] * 0.95, f'n={n}',
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Panel D: SART trajectory by group
ax4 = axes[1, 1]
for i, group in enumerate(GROUP_ORDER):
    group_data = df_probe_props[df_probe_props['group'] == group]
    if len(group_data) == 0:
        continue
    sart_agg = group_data.groupby('task')['prop_out_zone'].agg(['mean', 'sem']).reindex(TASKS).reset_index()
    ax4.errorbar(range(1, 5), sart_agg['mean'], yerr=sart_agg['sem'],
                 marker='o', linewidth=2.5, markersize=8, capsize=5,
                 alpha=0.8, color=GROUP_COLORS[i], label=group)
ax4.set_xticks(range(1, 5))
ax4.set_xticklabels(TASKS, fontsize=11, fontweight='bold')
ax4.set_xlabel('SART Task', fontsize=12, fontweight='bold')
ax4.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax4.set_title('D. SART Trajectory by Group', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11, loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 1)

plt.suptitle('Out-of-Zone Analysis by Group and Time on Task', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'out_of_zone_time_on_task.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, 'out_of_zone_time_on_task.svg'), dpi=300, bbox_inches='tight')
# plt.show()

#%%
# LMM: Group × Time on Task interaction
print("\n--- LMM: Proportion Out-of-Zone ~ Group * Time on Task ---")
df_probe_props_clean = df_probe_props.dropna(subset=['prop_out_zone', 'group', 'time_on_task'])
df_probe_props_clean['subject'] = df_probe_props_clean['subject'].astype(str)

model = smf.mixedlm("prop_out_zone ~ group * time_on_task", 
                    df_probe_props_clean, 
                    groups="subject").fit()
print(model.summary())

#%%
# =============================================================================
# ANALYSIS 2: ZONE STATE VS ON/OFF RATINGS
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: ZONE STATE VS ON/OFF RATINGS")
print("=" * 60)

# Correlation at probe level
df_probe_props_valid = df_probe_props.dropna(subset=['prop_out_zone', 'onoff'])
r_pearson, p_pearson = pearsonr(df_probe_props_valid['prop_out_zone'], df_probe_props_valid['onoff'])
r_spearman, p_spearman = spearmanr(df_probe_props_valid['prop_out_zone'], df_probe_props_valid['onoff'])

print(f"\nProbe-level correlations (N={len(df_probe_props_valid)}):")
print(f"  Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
print(f"  Spearman rho = {r_spearman:.3f}, p = {p_spearman:.4f}")

#%%
# Plot 2: Zone vs On/Off relationship (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel A: Scatter plot with regression line
ax1 = axes[0, 0]
for i, group in enumerate(GROUP_ORDER):
    group_data = df_probe_props_valid[df_probe_props_valid['group'] == group]
    ax1.scatter(group_data['onoff'], group_data['prop_out_zone'],
                alpha=0.5, s=30, color=GROUP_COLORS[i], label=group)

# Overall regression line
z = np.polyfit(df_probe_props_valid['onoff'], df_probe_props_valid['prop_out_zone'], 1)
p = np.poly1d(z)
x_range = np.linspace(df_probe_props_valid['onoff'].min(), df_probe_props_valid['onoff'].max(), 100)
ax1.plot(x_range, p(x_range), 'k--', linewidth=2, alpha=0.8)

ax1.set_xlabel('On/Off Rating (higher = more on-task)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax1.set_title(f'A. Zone State vs On/Off Rating\n(r={r_pearson:.3f}, p={p_pearson:.4f})', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel B: Out-of-zone by on/off tertiles
ax2 = axes[0, 1]
df_probe_props_valid['onoff_tertile'] = pd.qcut(df_probe_props_valid['onoff'], 3, 
                                                  labels=['Low (Off-task)', 'Medium', 'High (On-task)'])
tertile_order = ['Low (Off-task)', 'Medium', 'High (On-task)']
tertile_colors = ['#2E86AB', '#F4A261', '#F24236']

pt.RainCloud(
    x='onoff_tertile', y='prop_out_zone', data=df_probe_props_valid,
    palette=tertile_colors, order=tertile_order,
    bw=0.2, width_viol=0.6, alpha=0.7,
    dodge=True, pointplot=True, move=-0.1, ax=ax2
)
ax2.set_xlabel('On/Off Tertile', fontsize=12, fontweight='bold')
ax2.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax2.set_title('B. Out-of-Zone by On/Off Tertiles', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel C: Percentage out-of-zone within on-task vs off-task states
# Only use N trials before each probe for this analysis
ax3 = axes[1, 0]
df_subj_state = calculate_out_zone_by_subjective_state(
    df_trial, 
    onoff_threshold=ONOFF_THRESHOLD,
    n_trials_before_probe=N_TRIALS_BEFORE_PROBE
)
df_subj_state = df_subj_state.dropna(subset=['group'])

# Reshape for plotting
df_melt = df_subj_state.melt(
    id_vars=['subject', 'group'],
    value_vars=['prop_out_zone_on_task', 'prop_out_zone_off_task'],
    var_name='subjective_state',
    value_name='prop_out_zone'
)
df_melt['subjective_state'] = df_melt['subjective_state'].map({
    'prop_out_zone_on_task': 'On-Task',
    'prop_out_zone_off_task': 'Off-Task'
})

state_colors = ['#2E86AB', '#F24236']
pt.RainCloud(
    x='subjective_state', y='prop_out_zone', data=df_melt,
    palette=state_colors, order=['On-Task', 'Off-Task'],
    bw=0.2, width_viol=0.6, alpha=0.7,
    dodge=True, pointplot=True, move=-0.1, ax=ax3
)
ax3.set_xlabel('Subjective State', fontsize=12, fontweight='bold')
ax3.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax3.set_title(f'C. Out-of-Zone Within Subjective States\n(last {N_TRIALS_BEFORE_PROBE} trials before probe)', 
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Paired t-test
on_task_vals = df_subj_state['prop_out_zone_on_task'].dropna()
off_task_vals = df_subj_state['prop_out_zone_off_task'].dropna()
# Match subjects
common_subjs = df_subj_state.dropna(subset=['prop_out_zone_on_task', 'prop_out_zone_off_task'])
t_stat, p_val = stats.ttest_rel(common_subjs['prop_out_zone_on_task'], 
                                 common_subjs['prop_out_zone_off_task'])
ax3.text(0.5, 0.95, f'Paired t-test: t={t_stat:.2f}, p={p_val:.4f}',
         transform=ax3.transAxes, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Panel D: Group × Subjective State interaction
ax4 = axes[1, 1]
for i, group in enumerate(GROUP_ORDER):
    group_data = df_melt[df_melt['group'] == group]
    if len(group_data) == 0:
        continue
    state_agg = group_data.groupby('subjective_state')['prop_out_zone'].agg(['mean', 'sem']).reindex(['On-Task', 'Off-Task']).reset_index()
    ax4.errorbar([0, 1], state_agg['mean'], yerr=state_agg['sem'],
                 marker='o', linewidth=2.5, markersize=10, capsize=5,
                 alpha=0.8, color=GROUP_COLORS[i], label=group)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['On-Task', 'Off-Task'], fontsize=12, fontweight='bold')
ax4.set_xlabel('Subjective State', fontsize=12, fontweight='bold')
ax4.set_ylabel('Proportion Out-of-Zone', fontsize=12, fontweight='bold')
ax4.set_title('D. Group × Subjective State Interaction', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11, loc='best')
ax4.grid(True, alpha=0.3)

plt.suptitle('Relationship Between VTC Zones and Subjective Mind-Wandering', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'zone_vs_onoff_analysis.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, 'zone_vs_onoff_analysis.svg'), dpi=300, bbox_inches='tight')
# plt.show()

#%%
# LMM: Proportion Out-of-Zone ~ Group * Subjective State
print("\n--- LMM: Proportion Out-of-Zone ~ Group * Subjective State ---")
df_melt_clean = df_melt.dropna(subset=['prop_out_zone', 'group', 'subjective_state'])
df_melt_clean['subject'] = df_melt_clean['subject'].astype(str)

model_state = smf.mixedlm("prop_out_zone ~ group * subjective_state", 
                          df_melt_clean, 
                          groups="subject").fit()
print(model_state.summary())

#%%
# =============================================================================
# ANALYSIS 3: COMPREHENSIVE SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Overall zone distribution
print("\nOverall Zone Distribution (trial-level):")
zone_dist = df_trial['zone_state'].value_counts(normalize=True)
for zone, prop in zone_dist.items():
    print(f"  {zone}: {prop*100:.1f}%")

# By group
print("\nProportion Out-of-Zone by Group (probe-level):")
group_stats = df_probe_props.groupby('group')['prop_out_zone'].agg(['count', 'mean', 'std', 'sem'])
print(group_stats.round(4))

# By subjective state
print("\nProportion Out-of-Zone by Subjective State (subject-level):")
print(f"  On-Task: mean={df_subj_state['prop_out_zone_on_task'].mean():.3f}, "
      f"std={df_subj_state['prop_out_zone_on_task'].std():.3f}")
print(f"  Off-Task: mean={df_subj_state['prop_out_zone_off_task'].mean():.3f}, "
      f"std={df_subj_state['prop_out_zone_off_task'].std():.3f}")

# By group × subjective state
print("\nProportion Out-of-Zone by Group × Subjective State:")
for group in GROUP_ORDER:
    group_data = df_subj_state[df_subj_state['group'] == group]
    print(f"  {group}:")
    print(f"    On-Task: mean={group_data['prop_out_zone_on_task'].mean():.3f}, "
          f"std={group_data['prop_out_zone_on_task'].std():.3f}")
    print(f"    Off-Task: mean={group_data['prop_out_zone_off_task'].mean():.3f}, "
          f"std={group_data['prop_out_zone_off_task'].std():.3f}")

#%%
# Save results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save probe-level proportions
probe_props_file = os.path.join(OUTPUT_DIR, 'vtc_probe_proportions.csv')
df_probe_props.to_csv(probe_props_file, index=False)
print(f"Saved: {probe_props_file}")

# Save subject-level by subjective state
subj_state_file = os.path.join(OUTPUT_DIR, 'vtc_by_subjective_state.csv')
df_subj_state.to_csv(subj_state_file, index=False)
print(f"Saved: {subj_state_file}")

# Save summary statistics
summary_stats = {
    'Analysis': ['Zone vs OnOff Correlation (Pearson)', 'Zone vs OnOff Correlation (Spearman)',
                 'On-Task vs Off-Task (Paired t-test)'],
    'Statistic': [r_pearson, r_spearman, t_stat],
    'P-value': [p_pearson, p_spearman, p_val],
    'N': [len(df_probe_props_valid), len(df_probe_props_valid), len(common_subjs)]
}
summary_df = pd.DataFrame(summary_stats)
summary_file = os.path.join(OUTPUT_DIR, 'vtc_zone_analysis_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"Saved: {summary_file}")

print("\n" + "=" * 60)
print("VTC ZONE ANALYSIS COMPLETE")
print("=" * 60)
print(f"Plots saved to: {PLOTS_DIR}")

# %%
