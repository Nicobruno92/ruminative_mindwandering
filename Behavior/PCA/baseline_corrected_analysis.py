#%%
# ========================================================================
# BASELINE-CORRECTED ANALYSIS FOR PCA COMPONENTS
# ========================================================================
# Run this after the main PCA analysis to perform baseline-corrected 
# analysis where Sart1 and Sart3 serve as baselines for Sart2 and Sart4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
import statsmodels.formula.api as smf
import os
from datetime import datetime
import sys

# Configuration (should match main PCA analysis)
INTERACTIVE_MODE = False  # Set to True when running from Jupyter/subfolder

if INTERACTIVE_MODE:
    DATA_PATH = '../../results/Behavior/probe_data/pca_results.csv'
    BASE_OUTPUT_DIR = '../../results/Behavior/probe_data'
else:
    DATA_PATH = 'results/Behavior/probe_data/pca_results.csv'
    BASE_OUTPUT_DIR = 'results/Behavior/probe_data'

BASELINE_OUTPUT_DIR = f'{BASE_OUTPUT_DIR}/baseline_corrected_analysis'


def run_lmm_analysis(data, dependent_var, formula, model_name, output_dir):
    """
    Run linear mixed model analysis
    """
    print(f"\n=== FITTING MODEL: {model_name} ===")
    print(f"Formula: {dependent_var} ~ {formula}")
    print(f"Sample size: {len(data)} observations from "
          f"{data['subject_id'].nunique()} subjects")
    
    try:
        # Fit the model
        full_formula = f"{dependent_var} ~ {formula}"
        model = smf.mixedlm(full_formula, data, groups="subject_id").fit()
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
        
        # Save results
        results_file = os.path.join(output_dir, 
                                   f'{model_name}_{dependent_var}_results.csv')
        results_df.to_csv(results_file, index=False)
        
        return results_df, model
        
    except Exception as e:
        print(f"Error fitting model {model_name} for {dependent_var}: {str(e)}")
        return None, None


def prepare_baseline_corrected_data(df_with_pca):
    """
    Prepare baseline-corrected data where:
    - Sart1 is baseline for Sart2 (inclusion condition)
    - Sart3 is baseline for Sart4 (exclusion condition)
    """
    baseline_corrected_data = []
    
    for subject_id in df_with_pca['subject_id'].unique():
        subj_data = df_with_pca[df_with_pca['subject_id'] == subject_id]
        
        # Get baseline values (Sart1 and Sart3)
        sart1_data = subj_data[subj_data['task'] == 'Sart1']
        sart3_data = subj_data[subj_data['task'] == 'Sart3']
        
        # Get experimental values (Sart2 and Sart4)
        sart2_data = subj_data[subj_data['task'] == 'Sart2']
        sart4_data = subj_data[subj_data['task'] == 'Sart4']
        
        # Calculate baseline means for each PC
        if len(sart1_data) > 0:
            sart1_baseline = {
                'PC1': sart1_data['PC1'].mean(),
                'PC2': sart1_data['PC2'].mean(),
                'PC3': sart1_data['PC3'].mean()
            }
        else:
            continue
            
        if len(sart3_data) > 0:
            sart3_baseline = {
                'PC1': sart3_data['PC1'].mean(),
                'PC2': sart3_data['PC2'].mean(),
                'PC3': sart3_data['PC3'].mean()
            }
        else:
            continue
        
        # Create corrected records for Sart2 (inclusion)
        if len(sart2_data) > 0:
            for _, row in sart2_data.iterrows():
                corrected_row = row.copy()
                corrected_row['PC1_corrected'] = (row['PC1'] - 
                                                 sart1_baseline['PC1'])
                corrected_row['PC2_corrected'] = (row['PC2'] - 
                                                 sart1_baseline['PC2'])
                corrected_row['PC3_corrected'] = (row['PC3'] - 
                                                 sart1_baseline['PC3'])
                corrected_row['condition_type'] = 'inclusion'
                corrected_row['baseline_task'] = 'Sart1'
                corrected_row['experimental_task'] = 'Sart2'
                baseline_corrected_data.append(corrected_row)
        
        # Create corrected records for Sart4 (exclusion)
        if len(sart4_data) > 0:
            for _, row in sart4_data.iterrows():
                corrected_row = row.copy()
                corrected_row['PC1_corrected'] = (row['PC1'] - 
                                                 sart3_baseline['PC1'])
                corrected_row['PC2_corrected'] = (row['PC2'] - 
                                                 sart3_baseline['PC2'])
                corrected_row['PC3_corrected'] = (row['PC3'] - 
                                                 sart3_baseline['PC3'])
                corrected_row['condition_type'] = 'exclusion'
                corrected_row['baseline_task'] = 'Sart3'
                corrected_row['experimental_task'] = 'Sart4'
                baseline_corrected_data.append(corrected_row)
    
    df_baseline_corrected = pd.DataFrame(baseline_corrected_data)
    
    print(f"Baseline-corrected dataset: {len(df_baseline_corrected)} "
          f"observations")
    print(f"From {df_baseline_corrected['subject_id'].nunique()} subjects")
    condition_dist = df_baseline_corrected['condition_type'].value_counts()
    print(f"Condition distribution: {condition_dist.to_dict()}")
    
    return df_baseline_corrected


def create_comprehensive_baseline_plot(data, pc_var, output_dir):
    """
    Create comprehensive 2x3 grid plot for baseline-corrected PC component.
    
    Grid layout:
    - Row 1: Raincloud (Group) | Raincloud (Condition) | Interaction
    - Row 2: Time by Group | Time by Condition | SART Trajectories
    """
    # Define consistent ordering and colors
    GROUP_ORDER = ['Controls', 'Risk of Depression']
    CONDITION_ORDER = ['inclusion', 'exclusion']
    GROUP_COLORS = ['#2E86AB', '#F24236']
    CONDITION_COLORS = ['#A23B72', '#F18F01']
    
    plt.style.use("default")
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    # =========================================================================
    # ROW 1: DISTRIBUTION COMPARISONS
    # =========================================================================

    # Plot 1 (0,0): Raincloud for group comparison
    ax1 = fig.add_subplot(gs[0, 0])
    df_agg_group = data.groupby(["subject_id", "group"])[pc_var].mean().reset_index()
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
        f"{pc_var}: Group Effect (Baseline Corrected)",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Group", fontsize=14, fontweight="bold")
    ax1.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax1.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
    ax1.grid(True, alpha=0.3)
    
    # Add sample sizes
    for i, group in enumerate(GROUP_ORDER):
        n = n_participants_by_group.get(group, 0)
        ax1.text(
            i, ax1.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 2 (0,1): Raincloud for condition comparison
    ax2 = fig.add_subplot(gs[0, 1])
    df_agg_cond = data.groupby(["subject_id", "condition_type"])[pc_var].mean().reset_index()
    n_participants_by_cond = df_agg_cond.groupby("condition_type")["subject_id"].nunique().to_dict()
    
    pt.RainCloud(
        x="condition_type",
        y=pc_var,
        data=df_agg_cond,
        palette=CONDITION_COLORS,
        order=CONDITION_ORDER,
        bw=0.2,
        width_viol=0.6,
        alpha=0.7,
        dodge=True,
        pointplot=True,
        move=-0.1,
        ax=ax2,
    )
    ax2.set_title(
        f"{pc_var}: Condition Effect (Baseline Corrected)",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Condition", fontsize=14, fontweight="bold")
    ax2.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax2.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Add sample sizes
    for i, condition in enumerate(CONDITION_ORDER):
        n = n_participants_by_cond.get(condition, 0)
        ax2.text(
            i, ax2.get_ylim()[1] * 0.95, f"n={n}",
            ha="center", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    # Plot 3 (0,2): Interaction plot (Group × Condition)
    ax3 = fig.add_subplot(gs[0, 2])
    for group in data["group"].dropna().unique():
        group_data = data[data["group"] == group]
        if len(group_data) == 0:
            continue
        cond_means = (
            group_data.groupby("condition_type")[pc_var]
            .agg(["mean", "sem"])
            .reindex(CONDITION_ORDER)
            .reset_index()
        )
        n_participants_group = group_data["subject_id"].nunique()
        color = GROUP_COLORS[0] if group == GROUP_ORDER[0] else GROUP_COLORS[1]
        x_positions = [0, 1]
        ax3.errorbar(
            x_positions,
            cond_means["mean"],
            yerr=cond_means["sem"],
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
    ax3.set_xticklabels(CONDITION_ORDER, fontsize=14, fontweight="bold")
    ax3.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
    ax3.set_title(
        f"{pc_var}: Group × Condition Interaction\n(Baseline Corrected)",
        fontsize=18,
        fontweight="bold",
        pad=15,
    )
    ax3.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
    ax3.legend(fontsize=12, title_fontsize=12, loc='best')
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # ROW 2: TIME-ON-TASK TRAJECTORIES
    # =========================================================================

    # Plot 4 (1,0): Time-on-task by group
    ax4 = fig.add_subplot(gs[1, 0])
    if 'time_on_task' in data.columns:
        for i, group in enumerate(GROUP_ORDER):
            if group in data["group"].values:
                group_data = data[data["group"] == group]
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
            f"{pc_var}: Time-on-Task by Group\n(Baseline Corrected)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax4.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
        ax4.legend(fontsize=12, loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Time-on-task data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)

    # Plot 5 (1,1): Time-on-task by condition
    ax5 = fig.add_subplot(gs[1, 1])
    if 'time_on_task' in data.columns:
        for i, condition in enumerate(CONDITION_ORDER):
            if condition in data["condition_type"].values:
                cond_data = data[data["condition_type"] == condition]
                time_cond_agg = cond_data.groupby("time_on_task")[pc_var].agg(["mean", "sem"]).reset_index()
                color = CONDITION_COLORS[i]
                ax5.errorbar(
                    time_cond_agg["time_on_task"],
                    time_cond_agg["mean"],
                    yerr=time_cond_agg["sem"],
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
            f"{pc_var}: Time-on-Task by Condition\n(Baseline Corrected)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax5.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
        ax5.legend(fontsize=12, loc='best')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Time-on-task data not available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)

    # Plot 6 (1,2): SART Mean Trajectories by Group and Condition
    ax6 = fig.add_subplot(gs[1, 2])
    if 'experimental_task' in data.columns:
        # For baseline corrected data, we show Sart2 (inclusion) and Sart4 (exclusion)
        sart_order_list = ['Sart2', 'Sart4']
        x_positions = [1, 2]
        
        for group_idx, group in enumerate(GROUP_ORDER):
            if group not in data["group"].values:
                continue
            group_data = data[data["group"] == group]
            color = GROUP_COLORS[group_idx]
            
            for cond_idx, condition in enumerate(CONDITION_ORDER):
                if condition not in group_data['condition_type'].values:
                    continue
                cond_data = group_data[group_data['condition_type'] == condition]
                n_subjects = cond_data['subject_id'].nunique()
                
                # Get mean for this condition (either Sart2 for inclusion or Sart4 for exclusion)
                sart_task = 'Sart2' if condition == 'inclusion' else 'Sart4'
                task_data = cond_data[cond_data['experimental_task'] == sart_task]
                
                if len(task_data) > 0:
                    mean_val = task_data[pc_var].mean()
                    sem_val = task_data[pc_var].sem()
                    
                    # Use different markers for conditions
                    marker = 'o' if condition == 'inclusion' else 's'
                    linestyle = '-' if condition == 'inclusion' else '--'
                    
                    ax6.errorbar(
                        x_positions[cond_idx],
                        mean_val,
                        yerr=sem_val,
                        marker=marker,
                        linewidth=2.5,
                        markersize=10,
                        linestyle='',  # No line connecting
                        capsize=5,
                        capthick=2,
                        alpha=0.8,
                        color=color,
                        label=f"{group} - {condition.capitalize()} (n={n_subjects})" if cond_idx == 0 and group == GROUP_ORDER[0] else "",
                    )
        
        ax6.set_xticks(x_positions)
        ax6.set_xticklabels(['Inclusion\n(Sart2)', 'Exclusion\n(Sart4)'], fontsize=12, fontweight='bold')
        ax6.set_xlabel("Experimental Condition", fontsize=14, fontweight="bold")
        ax6.set_ylabel(f"{pc_var} Score", fontsize=14, fontweight="bold")
        ax6.set_title(
            f"{pc_var}: Experimental SART by Group\n(Baseline Corrected)",
            fontsize=18,
            fontweight="bold",
            pad=15,
        )
        ax6.axhline(y=0, linestyle='--', color='black', alpha=0.5, linewidth=2)
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color=GROUP_COLORS[0], linewidth=2, markersize=10, label=GROUP_ORDER[0]),
            Line2D([0], [0], marker='o', color=GROUP_COLORS[1], linewidth=2, markersize=10, label=GROUP_ORDER[1]),
        ]
        ax6.legend(handles=legend_elements, fontsize=11, loc='best')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Task data not available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pc_var}_baseline_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{pc_var}_baseline_comprehensive.svg'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 60)
    print("BASELINE-CORRECTED ANALYSIS")
    print("Sart1 & Sart3 as baselines for Sart2 & Sart4 respectively")
    print("=" * 60)
    
    # Load the PCA results data
    if not os.path.exists(DATA_PATH):
        print(f"Error: PCA results file not found at {DATA_PATH}")
        print("Please run the main PCA analysis first.")
        return
    
    df_pca = pd.read_csv(DATA_PATH)
    print(f"Loaded PCA data: {len(df_pca)} observations from "
          f"{df_pca['subject_id'].nunique()} subjects")
    
    # Prepare baseline-corrected data
    df_baseline = prepare_baseline_corrected_data(df_pca)
    
    # Create output directory
    os.makedirs(BASELINE_OUTPUT_DIR, exist_ok=True)
    
    # Save baseline-corrected data
    output_csv = f'{BASELINE_OUTPUT_DIR}/baseline_corrected_pca_data.csv'
    df_baseline.to_csv(output_csv, index=False)
    print(f"Saved baseline-corrected data to: {output_csv}")
    
    # Run baseline-corrected models
    print("\n" + "=" * 60)
    print("BASELINE-CORRECTED MODELS")
    print("=" * 60)
    
    for pc in ['PC1_corrected', 'PC2_corrected', 'PC3_corrected']:
        print(f"\n--- BASELINE MODEL 1: {pc} ~ group ---")
        run_lmm_analysis(df_baseline, pc, 'group', 
                        'baseline_model_1_group', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 2: {pc} ~ condition_type ---")
        run_lmm_analysis(df_baseline, pc, 'condition_type', 
                        'baseline_model_2_condition', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 3: {pc} ~ group * condition_type ---")
        run_lmm_analysis(df_baseline, pc, 'group * condition_type', 
                        'baseline_model_3_interaction', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 4: {pc} ~ group + time_on_task ---")
        run_lmm_analysis(df_baseline, pc, 'group + time_on_task', 
                        'baseline_model_4_group_time', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 5: {pc} ~ group * time_on_task ---")
        run_lmm_analysis(df_baseline, pc, 'group * time_on_task', 
                        'baseline_model_5_group_time_interact', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 6: {pc} ~ group + condition_type + time_on_task ---")
        run_lmm_analysis(df_baseline, pc, 'group + condition_type + time_on_task', 
                        'baseline_model_6_additive', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 7: {pc} ~ group * condition_type + time_on_task ---")
        run_lmm_analysis(df_baseline, pc, 'group * condition_type + time_on_task', 
                        'baseline_model_7_interact_time', BASELINE_OUTPUT_DIR)
        
        print(f"\n--- BASELINE MODEL 8: {pc} ~ group * condition_type * time_on_task ---")
        run_lmm_analysis(df_baseline, pc, 'group * condition_type * time_on_task', 
                        'baseline_model_8_full_interact', BASELINE_OUTPUT_DIR)
    
    # Create baseline-corrected plots
    print("\nCreating baseline-corrected comprehensive plots...")
    for pc in ['PC1_corrected', 'PC2_corrected', 'PC3_corrected']:
        print(f"Creating comprehensive plot for {pc}...")
        create_comprehensive_baseline_plot(df_baseline, pc, BASELINE_OUTPUT_DIR)
    
    print(f"\nBaseline-corrected analysis complete!")
    print(f"Results saved to: {BASELINE_OUTPUT_DIR}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(BASELINE_OUTPUT_DIR)):
        print(f"- {file}")


if __name__ == "__main__":
    main()