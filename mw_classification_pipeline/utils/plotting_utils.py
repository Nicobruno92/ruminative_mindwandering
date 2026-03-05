import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from scipy import stats
import warnings
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings("ignore")

# Color palette for classification plots
COLORS = ["#DE237B", "#9AC529", "#F38A31", "#42B9B2"]

def get_comparison_color(comparison):
    """Get color for a comparison label."""
    return COLORS[0]

def set_plot_style(style='seaborn-v0_8'):
    """Set the plotting style."""
    plt.style.use(style)
    
    # Set figure parameters
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


# =============================================================================
# Basic plotting functions (migrated from Other_project)
# =============================================================================

def plot_confusion_matrix(avg_conf_matrix, negative_class_name, positive_class_name, cell_stats_df, comparison_results_path, filename_base):
    """
    Plot and save annotated and simple confusion matrices.
    """
    import shap
    
    # Save raw confusion matrix to CSV
    raw_cm_df = pd.DataFrame(
        avg_conf_matrix,
        index=[negative_class_name, positive_class_name],
        columns=[f"Predicted {negative_class_name}", f"Predicted {positive_class_name}"]
    )
    raw_cm_df.index.name = "True Class"
    raw_cm_path = f"{comparison_results_path}/{filename_base}_raw_confusion_matrix.csv"
    raw_cm_df.to_csv(raw_cm_path)
    
    # Normalized confusion matrix
    row_sums = avg_conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_conf_matrix = avg_conf_matrix / row_sums
    norm_cm_df = pd.DataFrame(
        norm_conf_matrix,
        index=[negative_class_name, positive_class_name],
        columns=[f"Predicted {negative_class_name}", f"Predicted {positive_class_name}"]
    )
    norm_cm_df.index.name = "True Class"
    norm_cm_path = f"{comparison_results_path}/{filename_base}_normalized_confusion_matrix.csv"
    norm_cm_df.to_csv(norm_cm_path)
    
    # Annotated heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        norm_conf_matrix, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=[negative_class_name, positive_class_name], 
        yticklabels=[negative_class_name, positive_class_name],
        linewidths=1,
        linecolor='black',
        cbar_kws={'label': 'Proportion of samples'}
    )
    for i in range(len(norm_conf_matrix)):
        for j in range(len(norm_conf_matrix[i])):
            count = avg_conf_matrix[i, j]
            ax.text(j + 0.5, i + 0.7, f"n={count:.1f}", ha="center", va="center", color="gray", fontsize=9)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix.png", dpi=300)
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix.svg")
    plt.close()
    
    # Simple version
    plt.figure(figsize=(8, 6))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=[negative_class_name, positive_class_name], 
                yticklabels=[negative_class_name, positive_class_name])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Row-Normalized)')
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix_simple.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix_simple.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_confusion_matrix_simple.svg")
    plt.close()
    
    # Save stats
    if cell_stats_df is not None:
        stats_path = f"{comparison_results_path}/{filename_base}_confusion_matrix_stats.csv"
        cell_stats_df.to_csv(stats_path, index=False)


def plot_auc_distribution(mean_auc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save AUC distribution histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mean_auc_list, bins=20, alpha=0.7)
    plt.axvline(np.mean(mean_auc_list), color='red', linestyle='--', label=f"Mean AUC: {np.mean(mean_auc_list):.3f}")
    plt.axvline(0.5, color='black', linestyle=':', label='Random chance (0.5)')
    plt.xlabel('AUC')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of AUC Scores ({n_runs} runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_auc_distribution.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_auc_distribution.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_auc_distribution.svg")
    plt.close()
    
    # Save data and summary
    auc_array = np.array(mean_auc_list)
    data_df = pd.DataFrame({'auc_values': auc_array})
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    percentile_values = np.percentile(auc_array, percentiles)
    
    summary_stats = {
        'statistic': ['mean', 'std', 'median', 'min', 'max', 'n_runs', 'sem', 'ci_95_lower', 'ci_95_upper'] + 
                     [f'percentile_{p}' for p in percentiles],
        'value': [np.mean(auc_array), np.std(auc_array), np.median(auc_array), np.min(auc_array), 
                  np.max(auc_array), len(auc_array), np.std(auc_array) / np.sqrt(len(auc_array)),
                  percentile_values[0], percentile_values[-1]] + list(percentile_values)
    }
    summary_df = pd.DataFrame(summary_stats)
    data_df.to_csv(f"{comparison_results_path}/{filename_base}_auc_distribution_data.csv", index=False)
    summary_df.to_csv(f"{comparison_results_path}/{filename_base}_auc_distribution_summary.csv", index=False)


def plot_balanced_accuracy_distribution(mean_bal_acc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save balanced accuracy distribution histogram.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mean_bal_acc_list, bins=20, alpha=0.7)
    plt.axvline(np.mean(mean_bal_acc_list), color='red', linestyle='--', label=f"Mean Bal Acc: {np.mean(mean_bal_acc_list):.3f}")
    plt.axvline(0.5, color='black', linestyle=':', label='Random chance (0.5)')
    plt.xlabel('Balanced Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Balanced Accuracy Scores ({n_runs} runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution.svg")
    plt.close()
    
    # Save data and summary
    bal_acc_array = np.array(mean_bal_acc_list)
    data_df = pd.DataFrame({'balanced_accuracy_values': bal_acc_array})
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    percentile_values = np.percentile(bal_acc_array, percentiles)
    
    summary_stats = {
        'statistic': ['mean', 'std', 'median', 'min', 'max', 'n_runs', 'sem', 'ci_95_lower', 'ci_95_upper'] + 
                     [f'percentile_{p}' for p in percentiles],
        'value': [np.mean(bal_acc_array), np.std(bal_acc_array), np.median(bal_acc_array), np.min(bal_acc_array),
                  np.max(bal_acc_array), len(bal_acc_array), np.std(bal_acc_array) / np.sqrt(len(bal_acc_array)),
                  percentile_values[0], percentile_values[-1]] + list(percentile_values)
    }
    summary_df = pd.DataFrame(summary_stats)
    data_df.to_csv(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution_data.csv", index=False)
    summary_df.to_csv(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution_summary.csv", index=False)


def plot_auprc_distribution(mean_auprc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save AUPRC (Average Precision) distribution histogram.
    AUPRC is better suited for imbalanced datasets than AUROC.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mean_auprc_list, bins=20, alpha=0.7, color='#9AC529')
    mean_val = np.mean(mean_auprc_list)
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean AUPRC: {mean_val:.3f}")
    plt.xlabel('AUPRC (Average Precision)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of AUPRC Scores ({n_runs} runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_auprc_distribution.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_auprc_distribution.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_auprc_distribution.svg")
    plt.close()
    
    # Save data and summary
    auprc_array = np.array(mean_auprc_list)
    data_df = pd.DataFrame({'auprc_values': auprc_array})
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    percentile_values = np.percentile(auprc_array, percentiles)
    
    summary_stats = {
        'statistic': ['mean', 'std', 'median', 'min', 'max', 'n_runs', 'sem', 'ci_95_lower', 'ci_95_upper'] + 
                     [f'percentile_{p}' for p in percentiles],
        'value': [np.mean(auprc_array), np.std(auprc_array), np.median(auprc_array), np.min(auprc_array),
                  np.max(auprc_array), len(auprc_array), np.std(auprc_array) / np.sqrt(len(auprc_array)),
                  percentile_values[0], percentile_values[-1]] + list(percentile_values)
    }
    summary_df = pd.DataFrame(summary_stats)
    data_df.to_csv(f"{comparison_results_path}/{filename_base}_auprc_distribution_data.csv", index=False)
    summary_df.to_csv(f"{comparison_results_path}/{filename_base}_auprc_distribution_summary.csv", index=False)


def plot_mcc_distribution(mean_mcc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save MCC (Matthews Correlation Coefficient) distribution histogram.
    MCC provides a balanced measure for imbalanced datasets, ranging from -1 to +1.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(mean_mcc_list, bins=20, alpha=0.7, color='#F38A31')
    mean_val = np.mean(mean_mcc_list)
    plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean MCC: {mean_val:.3f}")
    plt.axvline(0.0, color='black', linestyle=':', label='Random chance (0.0)')
    plt.xlabel('MCC (Matthews Correlation Coefficient)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of MCC Scores ({n_runs} runs)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_mcc_distribution.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_mcc_distribution.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_mcc_distribution.svg")
    plt.close()
    
    # Save data and summary
    mcc_array = np.array(mean_mcc_list)
    data_df = pd.DataFrame({'mcc_values': mcc_array})
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    percentile_values = np.percentile(mcc_array, percentiles)
    
    summary_stats = {
        'statistic': ['mean', 'std', 'median', 'min', 'max', 'n_runs', 'sem', 'ci_95_lower', 'ci_95_upper'] + 
                     [f'percentile_{p}' for p in percentiles],
        'value': [np.mean(mcc_array), np.std(mcc_array), np.median(mcc_array), np.min(mcc_array),
                  np.max(mcc_array), len(mcc_array), np.std(mcc_array) / np.sqrt(len(mcc_array)),
                  percentile_values[0], percentile_values[-1]] + list(percentile_values)
    }
    summary_df = pd.DataFrame(summary_stats)
    data_df.to_csv(f"{comparison_results_path}/{filename_base}_mcc_distribution_data.csv", index=False)
    summary_df.to_csv(f"{comparison_results_path}/{filename_base}_mcc_distribution_summary.csv", index=False)


def plot_roc_curve(mean_fpr, mean_tpr, std_tpr, comparison_results_path, filename_base, mean_auc, n_curves, 
                   comparison=None, all_fprs=None, all_tprs=None, all_aucs=None):
    """
    Plot and save mean ROC curve with confidence interval.
    """
    plt.figure(figsize=(10, 8))
    color = get_comparison_color(comparison) if comparison else 'b'
    label_name = COMPARISON_DISPLAY_NAMES.get(comparison, comparison) if comparison else None
    
    # Plot individual ROC curves if provided
    if all_fprs is not None and all_tprs is not None:
        for fpr, tpr in zip(all_fprs, all_tprs):
            if len(fpr) < 2 or len(tpr) < 2 or len(fpr) != len(tpr):
                continue
            plt.plot(fpr, tpr, lw=1, alpha=0.2, color=color)
    
    # Plot mean ROC curve
    std_auc = np.std(all_aucs) if all_aucs is not None else 0
    legend_label = f"Mean ROC (AUC = {mean_auc:.3f}"
    legend_label += f" ± {std_auc:.3f})" if std_auc > 0 else ")"
    
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=legend_label)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2, label='± 1 std. dev.')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f"ROC Curves"
    if label_name:
        title += f" - {label_name}"
    title += f" ({n_curves} valid curves)"
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{comparison_results_path}/{filename_base}_roc_curve.pdf", bbox_inches='tight')
    plt.savefig(f"{comparison_results_path}/{filename_base}_roc_curve.svg", bbox_inches='tight')
    plt.close()
    
    # Save ROC data
    roc_data_df = pd.DataFrame({
        'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'std_tpr': std_tpr,
        'tprs_upper': tprs_upper, 'tprs_lower': tprs_lower
    })
    roc_data_df.to_csv(f"{comparison_results_path}/{filename_base}_roc_curve_data.csv", index=False)


def plot_feature_importances(feature_names, mean_importances, std_importances, comparison_results_path, filename_base, top_n=20):
    """
    Plot and save bar plot of top N feature importances.
    """
    indices = np.argsort(mean_importances)[::-1]
    plt.figure(figsize=(12, 10))
    plt.barh(range(top_n), mean_importances[indices][:top_n], xerr=std_importances[indices][:top_n], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][:top_n])
    plt.xlabel('Mean Feature Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_feature_importances.png")
    plt.savefig(f"{comparison_results_path}/{filename_base}_feature_importances.pdf")
    plt.savefig(f"{comparison_results_path}/{filename_base}_feature_importances.svg")
    plt.close()
    
    # Save feature importances data
    importances_data_df = pd.DataFrame({
        'feature_name': feature_names,
        'mean_importance': mean_importances,
        'std_importance': std_importances,
        'rank': np.argsort(np.argsort(mean_importances)[::-1]) + 1
    })
    importances_data_df = importances_data_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    importances_data_df.to_csv(f"{comparison_results_path}/{filename_base}_feature_importances_data.csv", index=False)


def plot_shap_beeswarm(shap_values, x_test, feature_names, comparison, save_dir, save_prefix, max_display=20):
    """
    Create a SHAP beeswarm plot showing top features.
    
    Parameters:
    -----------
    shap_values : array-like
        SHAP values for each sample and feature
    x_test : array-like
        Feature values for coloring the plot
    feature_names : list
        Names of features
    comparison : str
        Title for the plot
    save_dir : str
        Directory to save the plot
    save_prefix : str
        Prefix for the saved file
    max_display : int
        Maximum number of features to display (default: 20)
    """
    import shap
    
    plt.figure(figsize=(12, max(8, min(max_display, len(feature_names))/2)))
    shap_values_to_plot = shap_values
    
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_to_plot = shap_values[0]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values_to_plot = shap_values[..., 0]
    
    valid_x = False
    if x_test is not None:
        try:
            arr = np.asarray(x_test)
            if arr.shape == shap_values_to_plot.shape and not np.all(arr == 0):
                valid_x = True
        except Exception:
            pass
    
    # Limit to max_display features
    actual_max_display = min(max_display, len(feature_names))
    
    if valid_x:
        shap.summary_plot(shap_values_to_plot, x_test, feature_names=feature_names, 
                         plot_type="dot", show=False, max_display=actual_max_display)
    else:
        shap.summary_plot(shap_values_to_plot, feature_names=feature_names, 
                         plot_type="dot", show=False, max_display=actual_max_display)
    
    plt.title(f"SHAP Values (Top {actual_max_display} Features) - {comparison}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    beeswarm_path = os.path.join(str(save_dir), f"{save_prefix}_shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.savefig(beeswarm_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(beeswarm_path.replace('.png', '.svg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return plt.gcf()


def plot_shap_feature_importance(shap_values, X, feature_names, comparison_results_path, filename_base, max_display=20):
    """
    Plot and save SHAP feature importance (bar) plot.
    """
    import shap
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, 
                     plot_type="bar", max_display=min(max_display, len(feature_names)))
    plt.tight_layout()
    out_path = f"{comparison_results_path}/{filename_base}_shap_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.replace('.png', '.pdf'), dpi=300)
    plt.savefig(out_path.replace('.png', '.svg'), dpi=300)
    plt.close()
    
    # Save SHAP importance data
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'mean_abs_shap_value': mean_abs_shap,
        'rank': np.argsort(np.argsort(mean_abs_shap)[::-1]) + 1
    })
    shap_importance_df = shap_importance_df.sort_values('mean_abs_shap_value', ascending=False).reset_index(drop=True)
    shap_importance_df.to_csv(f"{comparison_results_path}/{filename_base}_shap_importance_data.csv", index=False)


# =============================================================================
# End of migrated functions
# =============================================================================

def plot_auc_distribution_comparison(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Plot AUC distribution comparison between true and permuted labels with comprehensive data saving.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get AUC values with safety checks
    if 'mean_auc' not in true_results.columns:
        print(f"Warning: 'mean_auc' column not found in true_results for {dimension}")
        plt.close(fig)
        return None
    if 'mean_auc' not in perm_results.columns:
        print(f"Warning: 'mean_auc' column not found in perm_results for {dimension}")
        plt.close(fig)
        return None
        
    true_aucs = true_results['mean_auc'].dropna()
    perm_aucs = perm_results['mean_auc'].dropna()
    
    # Check if we have valid data
    if len(true_aucs) == 0 and len(perm_aucs) == 0:
        print(f"Warning: No valid AUC data to plot for {dimension}")
        plt.close(fig)
        return None
    
    # Plot 1: Distributions with professional colors
    ax1.hist(true_aucs, bins=20, alpha=0.7, label='True Labels', color=COLORS[0], density=True)
    ax1.hist(perm_aucs, bins=20, alpha=0.7, label='Permuted Labels', color='gray', density=True)
    ax1.axvline(true_aucs.mean(), color=COLORS[0], linestyle='--', linewidth=2, label=f'True Mean: {true_aucs.mean():.3f}')
    ax1.axvline(perm_aucs.mean(), color='black', linestyle='--', linewidth=2, label=f'Perm Mean: {perm_aucs.mean():.3f}')
    ax1.set_xlabel('AUC', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title(f'{dimension} - AUC Distribution ({model_type.upper()})', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots with professional colors
    data_to_plot = [true_aucs, perm_aucs]
    labels = ['True Labels', 'Permuted Labels']
    colors = [COLORS[0], 'lightgray']
    
    box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('AUC', fontweight='bold')
    ax2.set_title(f'{dimension} - AUC Comparison ({model_type.upper()})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Calculate comprehensive statistics
    if len(true_aucs) > 0 and len(perm_aucs) > 0:
        # Statistical tests
        ks_stat, ks_p = stats.ks_2samp(true_aucs, perm_aucs)
        mwu_stat, mwu_p = stats.mannwhitneyu(true_aucs, perm_aucs, alternative='greater')
        
        # Empirical p-value
        empirical_p = (perm_aucs >= true_aucs.mean()).mean()
        
        fig.suptitle(f'Mann-Whitney U: p = {mwu_p:.4g}, Empirical p = {empirical_p:.4g}', 
                    fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), dpi=300, bbox_inches='tight')
        print(f"Saved AUC distribution plot to: {save_path}")
        
        # Save comprehensive data and statistics
        true_array = np.array(true_aucs)
        perm_array = np.array(perm_aucs)
        
        # Create data DataFrame
        max_len = max(len(true_array), len(perm_array))
        data_df = pd.DataFrame({
            'true_auc': np.pad(true_array, (0, max_len - len(true_array)), constant_values=np.nan),
            'perm_auc': np.pad(perm_array, (0, max_len - len(perm_array)), constant_values=np.nan)
        })
        
        # Calculate percentiles
        percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
        true_percentiles = np.percentile(true_array, percentiles)
        perm_percentiles = np.percentile(perm_array, percentiles)
        
        # Summary statistics
        summary_df = pd.DataFrame({
            'group': ['true', 'permuted'],
            'mean': [np.mean(true_array), np.mean(perm_array)],
            'std': [np.std(true_array), np.std(perm_array)],
            'sem': [np.std(true_array) / np.sqrt(len(true_array)), 
                    np.std(perm_array) / np.sqrt(len(perm_array))],
            'median': [np.median(true_array), np.median(perm_array)],
            'ci_95_lower': [true_percentiles[0], perm_percentiles[0]],
            'ci_95_upper': [true_percentiles[7], perm_percentiles[7]]
        })
        
        # Statistical tests summary
        stats_df = pd.DataFrame({
            'test': ['mann_whitney_u', 'kolmogorov_smirnov', 'empirical_p'],
            'statistic': [mwu_stat, ks_stat, np.nan],
            'p_value': [mwu_p, ks_p, empirical_p]
        })
        
        # Save data
        base_path = save_path.replace('.png', '')
        data_df.to_csv(f"{base_path}_data.csv", index=False)
        summary_df.to_csv(f"{base_path}_summary.csv", index=False)
        stats_df.to_csv(f"{base_path}_stats.csv", index=False)
    
    return fig

def plot_balanced_accuracy_distribution_comparison(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Plot balanced accuracy distribution comparison between true and permuted labels.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get balanced accuracy values with safety checks
    if 'mean_balanced_accuracy' not in true_results.columns:
        print(f"Warning: 'mean_balanced_accuracy' column not found in true_results for {dimension}")
        plt.close(fig)
        return None
    if 'mean_balanced_accuracy' not in perm_results.columns:
        print(f"Warning: 'mean_balanced_accuracy' column not found in perm_results for {dimension}")
        plt.close(fig)
        return None
        
    true_bal_accs = true_results['mean_balanced_accuracy'].dropna()
    perm_bal_accs = perm_results['mean_balanced_accuracy'].dropna()
    
    # Check if we have valid data
    if len(true_bal_accs) == 0 and len(perm_bal_accs) == 0:
        print(f"Warning: No valid balanced accuracy data to plot for {dimension}")
        plt.close(fig)
        return None
    
    # Plot 1: Distributions
    ax1.hist(true_bal_accs, bins=20, alpha=0.7, label='True Labels', color='green', density=True)
    ax1.hist(perm_bal_accs, bins=20, alpha=0.7, label='Permuted Labels', color='orange', density=True)
    ax1.axvline(true_bal_accs.mean(), color='green', linestyle='--', linewidth=2, label=f'True Mean: {true_bal_accs.mean():.3f}')
    ax1.axvline(perm_bal_accs.mean(), color='orange', linestyle='--', linewidth=2, label=f'Perm Mean: {perm_bal_accs.mean():.3f}')
    ax1.set_xlabel('Balanced Accuracy')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{dimension} - Balanced Accuracy Distribution ({model_type.upper()})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots
    data_to_plot = [true_bal_accs, perm_bal_accs]
    labels = ['True Labels', 'Permuted Labels']
    colors = ['lightgreen', 'wheat']
    
    box_plot = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Balanced Accuracy')
    ax2.set_title(f'{dimension} - Balanced Accuracy Comparison ({model_type.upper()})')
    ax2.grid(True, alpha=0.3)
    
    # Calculate p-value
    if len(true_bal_accs) > 0 and len(perm_bal_accs) > 0:
        _, p_value = stats.mannwhitneyu(true_bal_accs, perm_bal_accs, alternative='greater')
        fig.suptitle(f'P-value (Mann-Whitney U): {p_value:.4f}', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved balanced accuracy distribution plot to: {save_path}")
    
    return fig

def plot_feature_importance_distribution(results_df, feature_cols, dimension, model_type, top_n=15, save_path=None):
    """
    Plot feature importance distribution across runs.
    
    Parameters:
    - results_df: DataFrame with results including feature importance columns
    - feature_cols: List of feature column names
    - dimension: Name of the dimension being analyzed
    - model_type: Type of model (rf, xgb, merf)
    - top_n: Number of top features to plot
    - save_path: Path to save the plot
    """
    # Extract feature importances
    importance_cols = [col for col in results_df.columns if col.startswith('importance_')]
    
    if not importance_cols:
        print("No feature importance columns found")
        return None
    
    # Calculate mean importance for each feature
    feature_importance_means = {}
    feature_importance_stds = {}
    
    for col in importance_cols:
        feature_name = col.replace('importance_', '')
        if feature_name in feature_cols:
            feature_importance_means[feature_name] = results_df[col].mean()
            feature_importance_stds[feature_name] = results_df[col].std()
    
    # Sort by mean importance
    sorted_features = sorted(feature_importance_means.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    if not top_features:
        print("No valid feature importances found")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    feature_names = [f[0] for f in top_features]
    mean_importances = [f[1] for f in top_features]
    std_importances = [feature_importance_stds.get(f[0], 0) for f in top_features]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(feature_names))
    bars = ax.barh(y_pos, mean_importances, xerr=std_importances, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Highest importance at the top
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{dimension} - Top {top_n} Feature Importances ({model_type.upper()})')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, mean_importances, std_importances)):
        ax.text(bar.get_width() + std_val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{mean_val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to: {save_path}")
    
    return fig

def plot_performance_metrics_summary(true_results, dimension, model_type, save_path=None):
    """
    Plot summary of all performance metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['mean_auc', 'mean_balanced_accuracy', 'fold_precision', 'fold_recall']
    metric_names = ['AUC', 'Balanced Accuracy', 'Precision', 'Recall']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        if metric in true_results.columns:
            values = true_results[metric].dropna()
            
            if len(values) > 0:
                # Histogram
                ax.hist(values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {values.mean():.3f}')
                ax.axvline(values.median(), color='orange', linestyle='--', linewidth=2, 
                          label=f'Median: {values.median():.3f}')
                
                ax.set_xlabel(name)
                ax.set_ylabel('Frequency')
                ax.set_title(f'{name} Distribution ({model_type.upper()})')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, f'No {metric} column found', ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle(f'{dimension} - Performance Metrics Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance metrics summary to: {save_path}")
    
    return fig

def create_dimension_comparison_plot(results_summary, model_type, metric='mean_auc', save_path=None):
    """
    Create a comparison plot across all dimensions.
    
    Parameters:
    - results_summary: Dictionary with dimension names as keys and result DataFrames as values
    - model_type: Type of model (rf, xgb, merf)
    - metric: Metric to compare ('mean_auc', 'mean_balanced_accuracy', etc.)
    - save_path: Path to save the plot
    """
    if not results_summary:
        print("No results to plot")
        return None
    
    # Prepare data for plotting
    dimension_names = []
    metric_means = []
    metric_stds = []
    
    for dimension, results_df in results_summary.items():
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) > 0:
                dimension_names.append(dimension)
                metric_means.append(values.mean())
                metric_stds.append(values.std())
    
    if not dimension_names:
        print(f"No valid data for metric {metric}")
        return None
    
    # Sort by performance
    sorted_data = sorted(zip(dimension_names, metric_means, metric_stds), 
                        key=lambda x: x[1], reverse=True)
    dimension_names, metric_means, metric_stds = zip(*sorted_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    x_pos = np.arange(len(dimension_names))
    bars = ax.bar(x_pos, metric_means, yerr=metric_stds, capsize=5, 
                  alpha=0.7, color='lightcoral', edgecolor='darkred')
    
    ax.set_xlabel('Phenomenology Dimensions')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Dimensions ({model_type.upper()})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dimension_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, metric_means, metric_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dimension comparison plot to: {save_path}")
    
    return fig

def plot_permutation_test_results(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Create comprehensive permutation test visualization.
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # AUC comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Check for column existence before plotting
    if 'mean_auc' not in true_results.columns or 'mean_auc' not in perm_results.columns:
        print(f"Warning: Required AUC columns not found for {dimension}")
        plt.close(fig)
        return None
        
    true_aucs = true_results['mean_auc'].dropna()
    perm_aucs = perm_results['mean_auc'].dropna()
    
    ax1.hist(perm_aucs, bins=20, alpha=0.6, color='red', label='Permuted', density=True)
    ax1.hist(true_aucs, bins=20, alpha=0.7, color='blue', label='True', density=True)
    ax1.axvline(true_aucs.mean(), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(perm_aucs.mean(), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('AUC')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{dimension} - AUC Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Balanced Accuracy comparison
    ax2 = fig.add_subplot(gs[1, :2])
    
    # Check for column existence
    if 'mean_balanced_accuracy' not in true_results.columns or 'mean_balanced_accuracy' not in perm_results.columns:
        print(f"Warning: Required balanced accuracy columns not found for {dimension}")
        plt.close(fig)
        return None
        
    true_bal_acc = true_results['mean_balanced_accuracy'].dropna()
    perm_bal_acc = perm_results['mean_balanced_accuracy'].dropna()
    
    ax2.hist(perm_bal_acc, bins=20, alpha=0.6, color='orange', label='Permuted', density=True)
    ax2.hist(true_bal_acc, bins=20, alpha=0.7, color='green', label='True', density=True)
    ax2.axvline(true_bal_acc.mean(), color='green', linestyle='--', linewidth=2)
    ax2.axvline(perm_bal_acc.mean(), color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Balanced Accuracy')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{dimension} - Balanced Accuracy Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # P-value calculation and display
    ax3 = fig.add_subplot(gs[:2, 2])
    
    stats_text = []
    
    if len(true_aucs) > 0 and len(perm_aucs) > 0:
        _, p_auc = stats.mannwhitneyu(true_aucs, perm_aucs, alternative='greater')
        empirical_p_auc = (perm_aucs >= true_aucs.mean()).mean()
        stats_text.extend([
            f"AUC Statistics:",
            f"True Mean: {true_aucs.mean():.4f} ± {true_aucs.std():.4f}",
            f"Perm Mean: {perm_aucs.mean():.4f} ± {perm_aucs.std():.4f}",
            f"Mann-Whitney p: {p_auc:.4f}",
            f"Empirical p: {empirical_p_auc:.4f}",
            ""
        ])
    
    if len(true_bal_acc) > 0 and len(perm_bal_acc) > 0:
        _, p_bal_acc = stats.mannwhitneyu(true_bal_acc, perm_bal_acc, alternative='greater')
        empirical_p_bal_acc = (perm_bal_acc >= true_bal_acc.mean()).mean()
        stats_text.extend([
            f"Balanced Accuracy Statistics:",
            f"True Mean: {true_bal_acc.mean():.4f} ± {true_bal_acc.std():.4f}",
            f"Perm Mean: {perm_bal_acc.mean():.4f} ± {perm_bal_acc.std():.4f}",
            f"Mann-Whitney p: {p_bal_acc:.4f}",
            f"Empirical p: {empirical_p_bal_acc:.4f}"
        ])
    
    ax3.text(0.1, 0.9, '\n'.join(stats_text), transform=ax3.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Statistical Summary')
    
    # Box plot comparison
    ax4 = fig.add_subplot(gs[2, :])
    data_to_plot = [true_aucs, perm_aucs, true_bal_acc, perm_bal_acc]
    labels = ['True AUC', 'Perm AUC', 'True Bal Acc', 'Perm Bal Acc']
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'wheat']
    
    box_plot = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Performance')
    ax4.set_title('Performance Metrics Comparison')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(f'{dimension} - Permutation Test Results ({model_type.upper()})', fontsize=16)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved permutation test plot to: {save_path}")
    
    return fig

def load_and_plot_results(results_path, dimensions, model_types=['rf'], n_folds=4):
    """
    Load results and create all plots for specified dimensions and models.
    
    Parameters:
    - results_path: Base path to results
    - dimensions: List of dimensions to plot
    - model_types: List of model types to include
    - n_folds: Number of folds used in cross-validation
    """
    set_plot_style()
    
    all_results = {}
    
    for model_type in model_types:
        # Determine model folder
        model_folder = f"{model_type.upper()}_Distribution"
        model_path = os.path.join(results_path, model_folder, f"folds_{n_folds}")
        
        if not os.path.exists(model_path):
            print(f"Path not found: {model_path}")
            continue
        
        all_results[model_type] = {}
        
        for dimension in dimensions:
            dimension_path = os.path.join(model_path, dimension)
            
            if not os.path.exists(dimension_path):
                print(f"Dimension path not found: {dimension_path}")
                continue
            
            # Load true results
            true_csv_path = None
            perm_csv_path = None
            
            # First, try to find result files in the root dimension directory
            for file in os.listdir(dimension_path):
                if file.endswith('_summary.csv'):
                    if 'permutation' not in file:
                        true_csv_path = os.path.join(dimension_path, file)
                    else:
                        perm_csv_path = os.path.join(dimension_path, file)
            
            # If no consolidated files found, look for the folds 5 directory structure
            if not true_csv_path:
                folds_5_path = os.path.join(results_path, model_folder, f"folds_5", dimension)
                if os.path.exists(folds_5_path):
                    for file in os.listdir(folds_5_path):
                        if file.endswith('_summary.csv'):
                            if 'permutation' not in file:
                                true_csv_path = os.path.join(folds_5_path, file)
                            else:
                                perm_csv_path = os.path.join(folds_5_path, file)
                    
                    # Update dimension_path to folds_5 if files found there
                    if true_csv_path:
                        dimension_path = folds_5_path
                        print(f"Using folds_5 results for {dimension}")
            
            # If still no files found, check for individual run files that need aggregation
            if not true_csv_path:
                # Check if there are individual run directories
                runs_dir = os.path.join(dimension_path, "runs")
                perm_runs_dir = os.path.join(dimension_path, "permutation", "runs")
                
                if os.path.exists(runs_dir) or os.path.exists(perm_runs_dir):
                    print(f"Found individual run files for {dimension}, but no consolidated summary. Consider running analysis first to generate consolidated results.")
                    continue
            
            if true_csv_path and os.path.exists(true_csv_path):
                true_results = pd.read_csv(true_csv_path)
                all_results[model_type][dimension] = {'true': true_results}
                
                # Load permutation results if available
                if perm_csv_path and os.path.exists(perm_csv_path):
                    perm_results = pd.read_csv(perm_csv_path)
                    all_results[model_type][dimension]['permutation'] = perm_results
                
                # Create plots directory
                plots_dir = os.path.join(dimension_path, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                # Get feature columns (approximate from importance columns)
                importance_cols = [col for col in true_results.columns if col.startswith('importance_')]
                feature_cols = [col.replace('importance_', '') for col in importance_cols]
                
                # Create individual plots
                print(f"Creating plots for {dimension} ({model_type})...")
                
                # Performance metrics summary
                plot_performance_metrics_summary(
                    true_results, dimension, model_type,
                    save_path=os.path.join(plots_dir, f'{model_type}_performance_summary.png')
                )
                
                # Feature importance plot
                if feature_cols:
                    plot_feature_importance_distribution(
                        true_results, feature_cols, dimension, model_type,
                        save_path=os.path.join(plots_dir, f'{model_type}_feature_importance.png')
                    )
                
                # Permutation test plots (if permutation results available)
                if perm_csv_path and os.path.exists(perm_csv_path):
                    perm_results = pd.read_csv(perm_csv_path)
                    
                    plot_auc_distribution_comparison(
                        true_results, perm_results, dimension, model_type,
                        save_path=os.path.join(plots_dir, f'{model_type}_auc_comparison.png')
                    )
                    
                    plot_balanced_accuracy_distribution_comparison(
                        true_results, perm_results, dimension, model_type,
                        save_path=os.path.join(plots_dir, f'{model_type}_balanced_acc_comparison.png')
                    )
                    
                    plot_permutation_test_results(
                        true_results, perm_results, dimension, model_type,
                        save_path=os.path.join(plots_dir, f'{model_type}_permutation_test.png')
                    )
            else:
                print(f"No valid results found for {dimension}")
        
        # Create cross-dimension comparison plots
        if all_results[model_type]:
            comparison_plots_dir = os.path.join(results_path, model_folder, 'comparison_plots')
            os.makedirs(comparison_plots_dir, exist_ok=True)
            
            # Prepare data for comparison
            dimension_results = {dim: data['true'] for dim, data in all_results[model_type].items() if 'true' in data}
            
            if dimension_results:
                create_dimension_comparison_plot(
                    dimension_results, model_type, metric='mean_auc',
                    save_path=os.path.join(comparison_plots_dir, f'{model_type}_auc_comparison.png')
                )
                
                create_dimension_comparison_plot(
                    dimension_results, model_type, metric='mean_balanced_accuracy',
                    save_path=os.path.join(comparison_plots_dir, f'{model_type}_balanced_acc_comparison.png')
                )
    
    print("All plots created successfully!")
    return all_results 

def create_model_comparison_plot(results_dict, metric='mean_auc', model_types=['rf'], save_path=None):
    """
    Create violin plots comparing different models across shared dimensions.
    
    Parameters:
    - results_dict: Dictionary with structure {model_name: {dimension: results_df}}
    - metric: Metric to compare ('mean_auc', 'mean_balanced_accuracy', etc.)
    - model_types: List of model types to include in comparison
    - save_path: Path to save the plot
    """
    if not results_dict:
        print("No results to plot")
        return None
    
    # Find common dimensions across all models
    all_models = list(results_dict.keys())
    if len(all_models) < 2:
        print("Need at least 2 models for comparison")
        return None
    
    # Get shared dimensions
    shared_dimensions = set(results_dict[all_models[0]].keys())
    for model in all_models[1:]:
        shared_dimensions = shared_dimensions.intersection(set(results_dict[model].keys()))
    
    if not shared_dimensions:
        print("No shared dimensions found across models")
        return None
    
    shared_dimensions = sorted(list(shared_dimensions))
    print(f"Comparing models across {len(shared_dimensions)} shared dimensions: {shared_dimensions}")
    
    # Prepare data for plotting
    plot_data = []
    
    for model_name in all_models:
        for dimension in shared_dimensions:
            if dimension in results_dict[model_name]:
                results_df = results_dict[model_name][dimension]
                if metric in results_df.columns:
                    values = results_df[metric].dropna()
                    for value in values:
                        plot_data.append({
                            'Model': model_name,
                            'Dimension': dimension,
                            'Value': value,
                            'Metric': metric
                        })
    
    if not plot_data:
        print(f"No valid data found for metric {metric}")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create violin plot
    sns.violinplot(data=plot_df, x='Dimension', y='Value', hue='Model', ax=ax, inner='box')
    
    # Add chance level line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Chance Level')
    
    # Customize plot
    ax.set_xlabel('Phenomenology Dimensions', fontsize=14)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison Across Models', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison plot to: {save_path}")
    
    return fig

def create_permutation_comparison_plot(results_dict, perm_results_dict, metric='mean_auc', save_path=None):
    """
    Create violin plots comparing true vs permutation results across models.
    
    Parameters:
    - results_dict: Dictionary with true results {model_name: {dimension: results_df}}
    - perm_results_dict: Dictionary with permutation results {model_name: {dimension: results_df}}
    - metric: Metric to compare
    - save_path: Path to save the plot
    """
    if not results_dict or not perm_results_dict:
        print("Need both true and permutation results")
        return None
    
    # Find common models and dimensions
    common_models = set(results_dict.keys()).intersection(set(perm_results_dict.keys()))
    if not common_models:
        print("No common models found")
        return None
    
    # Prepare data for plotting
    plot_data = []
    
    for model_name in common_models:
        true_results = results_dict[model_name]
        perm_results = perm_results_dict[model_name]
        
        common_dims = set(true_results.keys()).intersection(set(perm_results.keys()))
        
        for dimension in common_dims:
            # True results
            if metric in true_results[dimension].columns:
                values = true_results[dimension][metric].dropna()
                for value in values:
                    plot_data.append({
                        'Model': model_name,
                        'Dimension': dimension,
                        'Value': value,
                        'Type': 'True',
                        'Metric': metric
                    })
            
            # Permutation results
            if metric in perm_results[dimension].columns:
                values = perm_results[dimension][metric].dropna()
                for value in values:
                    plot_data.append({
                        'Model': model_name,
                        'Dimension': dimension,
                        'Value': value,
                        'Type': 'Permutation',
                        'Metric': metric
                    })
    
    if not plot_data:
        print(f"No valid data found for metric {metric}")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, axes = plt.subplots(len(common_models), 1, figsize=(16, 6*len(common_models)), squeeze=False)
    
    if len(common_models) == 1:
        axes = axes.reshape(1, -1)
    
    for i, model_name in enumerate(sorted(common_models)):
        ax = axes[i][0]
        model_data = plot_df[plot_df['Model'] == model_name]
        
        if not model_data.empty:
            sns.violinplot(data=model_data, x='Dimension', y='Value', hue='Type', 
                         ax=ax, inner='box', alpha=0.7)
            
            # Add chance level line
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
            
            # Customize subplot
            ax.set_xlabel('Phenomenology Dimensions', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{model_name.capitalize()} - True vs Permutation Results', fontsize=14)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust legend
            ax.legend(title='Type')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{metric.replace("_", " ").title()} - Permutation Test Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved permutation comparison plot to: {save_path}")
    
    return fig

def load_and_plot_model_comparisons(results_path, models=['human', 'chatgpt4o'], model_types=['rf'], n_folds=4):
    """
    Load results for multiple models and create comparison plots.
    
    Parameters:
    - results_path: Base path to results
    - models: List of model names to compare
    - model_types: List of model types to include
    - n_folds: Number of folds used in cross-validation
    """
    set_plot_style()
    
    all_results = {}
    all_perm_results = {}
    
    for model_type in model_types:
        all_results[model_type] = {}
        all_perm_results[model_type] = {}
        
        for model_name in models:
            # Determine model folder based on config structure
            if model_name == 'human':
                model_folder = f"Human_{model_type.upper()}_Distribution"
            elif model_name == 'chatgpt4o':
                model_folder = f"ChatGPT4o_{model_type.upper()}_Distribution"
            else:
                model_folder = f"{model_name.capitalize()}_{model_type.upper()}_Distribution"
            
            model_path = os.path.join(results_path, model_folder, f"folds_{n_folds}")
            
            if not os.path.exists(model_path):
                print(f"Path not found: {model_path}")
                continue
            
            all_results[model_type][model_name] = {}
            all_perm_results[model_type][model_name] = {}
            
            # Find all dimension directories
            if os.path.exists(model_path):
                for dimension_dir in os.listdir(model_path):
                    dimension_path = os.path.join(model_path, dimension_dir)
                    if os.path.isdir(dimension_path):
                        # Load true results
                        true_csv_path = None
                        perm_csv_path = None
                        
                        for file in os.listdir(dimension_path):
                            if file.endswith('_summary.csv'):
                                if 'permutation' not in file:
                                    true_csv_path = os.path.join(dimension_path, file)
                                else:
                                    perm_csv_path = os.path.join(dimension_path, file)
                        
                        # Load true results
                        if true_csv_path and os.path.exists(true_csv_path):
                            true_results = pd.read_csv(true_csv_path)
                            all_results[model_type][model_name][dimension_dir] = true_results
                        
                        # Load permutation results
                        if perm_csv_path and os.path.exists(perm_csv_path):
                            perm_results = pd.read_csv(perm_csv_path)
                            all_perm_results[model_type][model_name][dimension_dir] = perm_results
        
        # Create comparison plots for this model type
        if len(all_results[model_type]) >= 2:
            comparison_plots_dir = os.path.join(results_path, 'Model_Comparisons', model_type.upper())
            os.makedirs(comparison_plots_dir, exist_ok=True)
            
            print(f"Creating comparison plots for {model_type.upper()}")
            
            # AUC comparison
            create_model_comparison_plot(
                all_results[model_type], 
                metric='mean_auc',
                save_path=os.path.join(comparison_plots_dir, f'{model_type}_auc_comparison.png')
            )
            
            # Balanced accuracy comparison
            create_model_comparison_plot(
                all_results[model_type], 
                metric='mean_balanced_accuracy',
                save_path=os.path.join(comparison_plots_dir, f'{model_type}_balanced_acc_comparison.png')
            )
            
            # Permutation comparison (if permutation results available)
            if any(all_perm_results[model_type].values()):
                create_permutation_comparison_plot(
                    all_results[model_type],
                    all_perm_results[model_type],
                    metric='mean_auc',
                    save_path=os.path.join(comparison_plots_dir, f'{model_type}_permutation_comparison.png')
                )
    
    print("Model comparison plots created successfully!")
    return all_results, all_perm_results

def plot_feature_importances_boxplot(parsed_importances, feature_names, comparison_results_path, filename_base, contrast, color_idx, fig_path=None, markers=None):
    """
    Visualize feature importances as horizontal boxplots using Plotly with professional styling.
    
    Parameters:
    - parsed_importances: array-like (runs x features)
    - feature_names: list of feature names
    - comparison_results_path: results folder
    - filename_base: base filename
    - contrast: comparison name
    - color_idx: color index for the comparison
    - fig_path: folder to save (optional, defaults to comparison_results_path)
    - markers: list of features to show (optional)
    """
    import os
    import pandas as pd
    
    if fig_path is None:
        fig_path = comparison_results_path
    
    # DataFrame for boxplot
    feat_importances = pd.DataFrame(parsed_importances, columns=feature_names)
    boxplot_data = feat_importances.melt(var_name='Feature', value_name='Importance')
    
    # Sort by median
    medians = boxplot_data.groupby('Feature')['Importance'].median().sort_values()
    
    # Filter markers if defined
    if markers is not None:
        boxplot_data = boxplot_data[boxplot_data['Feature'].isin(markers)]
        medians = medians[medians.index.isin(markers)]
        categoryarray = [f for f in medians.index if f in markers]
    else:
        categoryarray = list(medians.index)
    
    # Create Plotly boxplot
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=boxplot_data['Feature'],
        x=boxplot_data['Importance'],
        orientation='h',
        marker_color=COLORS[color_idx % len(COLORS)],
        boxpoints='all',
        pointpos=0,
        jitter=0,
    ))
    
    fig.update_layout(
        yaxis=dict(
            title='Feature',
            categoryorder='array',
            categoryarray=categoryarray,
            tickfont={"size": 20},
            showgrid=True,
            automargin=True,
            dtick=1,
        ),
        xaxis=dict(
            title='Importance',
            visible=True,
            range=[0, 0.3],
            tickfont={"size": 20},
        ),
        width=650,
        height=1100,
        template='plotly_white',
        font=dict(
            family="Times New Roman",
            size=20,
            color="black"
        ),
        showlegend=False,
    )
    fig.update_traces(marker=dict(size=3))
    
    # Save plots
    filename = os.path.join(fig_path, f'{contrast}_feat_importances_boxplot')
    fig.write_image(filename + '.png')
    fig.write_image(filename + '.pdf')
    fig.write_image(filename + '.svg')
    
    # Save boxplot data to CSV
    boxplot_data.to_csv(os.path.join(fig_path, f'{contrast}_feat_importances_boxplot_data.csv'), index=False)
    
    # Calculate and save summary statistics for each feature
    summary_stats = boxplot_data.groupby('Feature')['Importance'].agg([
        'mean', 'std', 'median', 'min', 'max', 'count'
    ]).reset_index()
    summary_stats.columns = ['feature_name', 'mean', 'std', 'median', 'min', 'max', 'n_runs']
    summary_stats = summary_stats.sort_values('mean', ascending=False).reset_index(drop=True)
    summary_stats['rank'] = range(1, len(summary_stats) + 1)
    
    summary_stats.to_csv(os.path.join(fig_path, f'{contrast}_feat_importances_boxplot_summary.csv'), index=False)

def empirical_mean_permutation_pvalue(true_values, perm_values):
    """
    Compute the empirical p-value: fraction of permuted means >= mean of true values.
    """
    true_mean = np.median(true_values)
    perm_means = np.array(perm_values)
    # If perm_values is a 1D array of all permuted scores, treat as one set
    # If it's 2D (n_permutations, n_runs), flatten to 1D
    if perm_means.ndim > 1:
        perm_means = perm_means.flatten()
    p_empirical = np.mean(perm_means >= true_mean)
    return p_empirical, true_mean, perm_means

def plot_metric_distribution_with_stats(
    true_values, perm_values, metric_name, comparison_results_path, filename_base,
    ks_stat, ks_p, mwu_stat, mwu_p, n_true, n_perm, chance_value=0.5, title=None,
    empirical_p=None
):
    """
    Plot combined histogram for true and permuted values with statistical annotations.
    Optionally annotate empirical mean permutation p-value if provided.
    """
    # Filter out NaN values before plotting
    true_values_clean = np.array(true_values)
    true_values_clean = true_values_clean[~np.isnan(true_values_clean)]
    
    perm_values_clean = np.array(perm_values)
    perm_values_clean = perm_values_clean[~np.isnan(perm_values_clean)]
    
    # Check if we have valid data to plot
    if len(true_values_clean) == 0 and len(perm_values_clean) == 0:
        print(f"Warning: No valid data to plot for {metric_name}. Skipping plot generation.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create bins based on available data
    all_valid_data = []
    if len(true_values_clean) > 0:
        all_valid_data.extend(true_values_clean)
    if len(perm_values_clean) > 0:
        all_valid_data.extend(perm_values_clean)
    
    if len(all_valid_data) > 0:
        bins = np.histogram(all_valid_data, bins=20)[1]
    else:
        bins = 20
    # Plot histograms with cleaned data if available
    if len(perm_values_clean) > 0:
        plt.hist(perm_values_clean, bins=bins, alpha=0.6, color='gray', label=f'Permuted Labels (n={len(perm_values_clean)})')
        plt.axvline(np.mean(perm_values_clean), color='black', linestyle='--', label=f'Perm Mean: {np.mean(perm_values_clean):.3f}')
    else:
        print(f"Warning: No valid permuted values to plot for {metric_name}")
    
    if len(true_values_clean) > 0:
        plt.hist(true_values_clean, bins=bins, alpha=0.7, color=COLORS[0], label=f'True Labels (n={len(true_values_clean)})')
        plt.axvline(np.mean(true_values_clean), color=COLORS[0], linestyle='--', label=f'True Mean: {np.mean(true_values_clean):.3f}')
    else:
        print(f"Warning: No valid true values to plot for {metric_name}")
    if chance_value is not None:
        plt.axvline(chance_value, color='black', linestyle=':', label=f'Chance ({chance_value})')
    
    # Title
    if title is None:
        title = f"{metric_name.title()} Distribution"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(metric_name.title(), fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    
    # Statistics in upper left corner
    stats_text = (
        f"Mann-Whitney U: p = {mwu_p:.4g}, Kolmogorov-Smirnov: p = {ks_p:.4g}"
    )
    if empirical_p is not None:
        stats_text += f"\nEmpirical mean p = {empirical_p:.4g}"
    plt.gca().text(
        0.02, 0.98, stats_text, transform=plt.gca().transAxes,
        fontsize=10, va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    plt.legend(loc='upper right', fontsize=11, frameon=True, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{comparison_results_path}/{filename_base}_{metric_name}_distribution.png", dpi=300)
    plt.savefig(f"{comparison_results_path}/{filename_base}_{metric_name}_distribution.pdf", dpi=300)
    plt.savefig(f"{comparison_results_path}/{filename_base}_{metric_name}_distribution.svg", dpi=300)
    print(f"Saved {metric_name} distribution plot to {comparison_results_path}/{filename_base}_{metric_name}_distribution.png")
    plt.close()
    
    # Save comprehensive metric distribution data and statistics to CSV
    # Use the cleaned arrays for statistics
    true_array = true_values_clean
    perm_array = perm_values_clean
    
    # Create data DataFrame
    max_len = max(len(true_array), len(perm_array))
    data_df = pd.DataFrame({
        'true_values': np.pad(true_array, (0, max_len - len(true_array)), constant_values=np.nan),
        'perm_values': np.pad(perm_array, (0, max_len - len(perm_array)), constant_values=np.nan)
    })
    
    # Calculate percentiles for both distributions
    percentiles = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
    
    # Handle empty arrays for percentiles
    if len(true_array) > 0:
        true_percentiles = np.percentile(true_array, percentiles)
    else:
        true_percentiles = [np.nan] * len(percentiles)
    
    if len(perm_array) > 0:
        perm_percentiles = np.percentile(perm_array, percentiles)
    else:
        perm_percentiles = [np.nan] * len(percentiles)
    
    # Create comprehensive summary DataFrame with all statistics needed for manuscript
    # Handle empty arrays for statistics
    def safe_stat(arr, stat_func, default=np.nan):
        return stat_func(arr) if len(arr) > 0 else default
    
    summary_data = {
        'group': ['true', 'permuted'],
        'mean': [safe_stat(true_array, np.mean), safe_stat(perm_array, np.mean)],
        'std': [safe_stat(true_array, np.std), safe_stat(perm_array, np.std)],
        'sem': [safe_stat(true_array, lambda x: np.std(x) / np.sqrt(len(x))), 
                safe_stat(perm_array, lambda x: np.std(x) / np.sqrt(len(x)))],
        'median': [safe_stat(true_array, np.median), safe_stat(perm_array, np.median)],
        'min': [safe_stat(true_array, np.min), safe_stat(perm_array, np.min)],
        'max': [safe_stat(true_array, np.max), safe_stat(perm_array, np.max)],
        'n': [len(true_array), len(perm_array)],
        'ci_95_lower': [true_percentiles[0], perm_percentiles[0]],
        'ci_95_upper': [true_percentiles[7], perm_percentiles[7]]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate effect sizes
    if len(true_array) > 1 and len(perm_array) > 1:
        pooled_std = np.sqrt(((len(true_array) - 1) * np.var(true_array, ddof=1) + 
                             (len(perm_array) - 1) * np.var(perm_array, ddof=1)) / 
                            (len(true_array) + len(perm_array) - 2))
        if pooled_std > 0:
            cohens_d = (np.mean(true_array) - np.mean(perm_array)) / pooled_std
        else:
            cohens_d = np.nan
    else:
        cohens_d = np.nan
    
    # Statistical tests summary
    stats_df = pd.DataFrame({
        'test': ['mann_whitney_u', 'kolmogorov_smirnov', 'empirical_mean_pvalue', 'cohens_d'],
        'statistic': [mwu_stat, ks_stat, np.nan, cohens_d],
        'p_value': [mwu_p, ks_p, empirical_p if empirical_p is not None else np.nan, np.nan]
    })
    
    # Save statistics
    base_path = f"{comparison_results_path}/{filename_base}_{metric_name}_distribution"
    data_df.to_csv(f"{base_path}_data.csv", index=False)
    summary_df.to_csv(f"{base_path}_summary.csv", index=False)
    stats_df.to_csv(f"{base_path}_stats.csv", index=False)


def plot_consolidated_permutation_results(
    results_dict,
    dimension,
    model_type,
    save_path,
    filename_base,
    chance_values=None
):
    """
    Plot consolidated permutation test results for multiple metrics in a single figure.
    Also saves consolidated data and statistics CSVs.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing results for each metric.
        Structure:
        {
            'Metric Name': {
                'true_values': array-like,
                'perm_values': array-like,
                'p_value': float,
                'empirical_p': float (optional),
                'stats': dict (optional extra stats like mean, std)
            },
            ...
        }
    dimension : str
        Name of the dimension being analyzed.
    model_type : str
        Type of model (rf, xgb, etc.).
    save_path : str
        Directory to save the results.
    filename_base : str
        Base filename for saved files.
    chance_values : dict, optional
        Dictionary mapping metric names to chance levels (e.g., {'AUC': 0.5}).
    """
    if not results_dict:
        print("No results to plot.")
        return

    n_metrics = len(results_dict)
    # Determine grid size (approximate square)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    # Initialize containers for consolidated CSVs
    all_data_list = []
    stats_list = []
    
    for i, (metric_name, data) in enumerate(results_dict.items()):
        ax = axes[i]
        
        true_values = np.array(data['true_values'])
        perm_values = np.array(data['perm_values'])
        true_values = true_values[~np.isnan(true_values)]
        perm_values = perm_values[~np.isnan(perm_values)]
        
        if len(true_values) == 0 and len(perm_values) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            continue
            
        # Determine bins
        all_vals = np.concatenate([true_values, perm_values])
        if len(all_vals) > 0:
            bins = np.histogram(all_vals, bins=20)[1]
        else:
            bins = 20
            
        # Plot histograms
        if len(perm_values) > 0:
            ax.hist(perm_values, bins=bins, alpha=0.6, color='gray', 
                   label=f'Permuted (n={len(perm_values)})', density=True)
            ax.axvline(np.mean(perm_values), color='black', linestyle='--', 
                      label=f'Perm Mean: {np.mean(perm_values):.3f}')
        
        if len(true_values) > 0:
            ax.hist(true_values, bins=bins, alpha=0.7, color=COLORS[i % len(COLORS)], 
                   label=f'True (n={len(true_values)})', density=True)
            ax.axvline(np.mean(true_values), color=COLORS[i % len(COLORS)], linestyle='--', 
                      label=f'True Mean: {np.mean(true_values):.3f}')
            
        # Chance line
        chance = None
        if chance_values and metric_name in chance_values:
            chance = chance_values[metric_name]
        elif chance_values and metric_name.lower() in chance_values: # Try lower case
             chance = chance_values[metric_name.lower()]
             
        # Heuristics for chance if not provided
        if chance is None:
            if 'auc' in metric_name.lower() or 'balanced' in metric_name.lower():
                chance = 0.5
            elif 'mcc' in metric_name.lower():
                chance = 0.0
                
        if chance is not None:
             ax.axvline(chance, color='red', linestyle=':', label=f'Chance ({chance})')

        ax.set_title(f"{metric_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel(metric_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        
        # Add p-value text
        p_val = data.get('p_value', np.nan)
        emp_p = data.get('empirical_p', np.nan)
        
        stats_text = ""
        if not np.isnan(p_val) and not isinstance(p_val, str):
             stats_text += f"Mann-Whitney p: {p_val:.4f}\n"
        if not np.isnan(emp_p) and not isinstance(emp_p, str):
             stats_text += f"Empirical p: {emp_p:.4f}"
             
        if stats_text:
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)

        # Collect data for CSVs
        # Create a temp df for this metric
        max_len = max(len(true_values), len(perm_values))
        metric_df = pd.DataFrame({
            'run_idx': range(max_len),
            'metric': metric_name,
            'true_value': np.pad(true_values, (0, max_len - len(true_values)), constant_values=np.nan),
            'perm_value': np.pad(perm_values, (0, max_len - len(perm_values)), constant_values=np.nan)
        })
        all_data_list.append(metric_df)
        
        # Collect stats
        stats_entry = {
            'dimension': dimension,
            'model_type': model_type,
            'metric': metric_name,
            'true_mean': np.mean(true_values) if len(true_values) > 0 else np.nan,
            'true_std': np.std(true_values) if len(true_values) > 0 else np.nan,
            'perm_mean': np.mean(perm_values) if len(perm_values) > 0 else np.nan,
            'perm_std': np.std(perm_values) if len(perm_values) > 0 else np.nan,
            'p_value_mwu': p_val,
            'p_value_empirical': emp_p,
            'n_true': len(true_values),
            'n_perm': len(perm_values)
        }
        stats_list.append(stats_entry)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    fig.suptitle(f"{dimension} - Permutation Test Results ({model_type.upper()})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    os.makedirs(save_path, exist_ok=True)
    out_fig_path = os.path.join(save_path, f"{filename_base}_consolidated_permutation_results.pdf")
    plt.savefig(out_fig_path, dpi=300)
    # Also save png for preview
    plt.savefig(out_fig_path.replace('.pdf', '.png'), dpi=300)
    print(f"Saved consolidated permutation plot to {out_fig_path}")
    plt.close()
    
    # Save CSVs
    if all_data_list:
        combined_data_df = pd.concat(all_data_list, ignore_index=True)
        out_data_path = os.path.join(save_path, f"{filename_base}_consolidated_permutation_data.csv")
        combined_data_df.to_csv(out_data_path, index=False)
        print(f"Saved consolidated permutation data to {out_data_path}")
        
    if stats_list:
        stats_df = pd.DataFrame(stats_list)
        out_stats_path = os.path.join(save_path, f"{filename_base}_consolidated_stats.csv")
        stats_df.to_csv(out_stats_path, index=False)
        print(f"Saved consolidated stats to {out_stats_path}")

def generate_manuscript_summary_report(comparison_results_path, filename_base, metric_name='auc'):
    """
    Generate a comprehensive summary CSV that compiles all statistics needed for manuscript writing.
    """
    import os
    
    try:
        summary_data = {'metric': metric_name}
        
        # Load individual CSV files if they exist
        stats_files = [
            f"{comparison_results_path}/{filename_base}_auc_distribution_summary.csv",
            f"{comparison_results_path}/{filename_base}_balanced_acc_distribution_summary.csv",
            f"{comparison_results_path}/{filename_base}_{metric_name}_distribution_stats.csv",
            f"{comparison_results_path}/{filename_base}_feature_importances_summary.csv"
        ]
        
        for stats_file in stats_files:
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)
                for _, row in df.iterrows():
                    if 'statistic' in row and 'value' in row:
                        summary_data[row['statistic']] = row['value']
        
        # Create comprehensive manuscript summary DataFrame
        manuscript_df = pd.DataFrame([summary_data])
        
        # Save the comprehensive summary
        output_file = f"{comparison_results_path}/{filename_base}_manuscript_summary.csv"
        manuscript_df.to_csv(output_file, index=False)
        
        print(f"Generated comprehensive manuscript summary: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error generating manuscript summary: {e}")
        return None

def plot_loso_subject_metrics(loso_subject_df, save_path, filename_base):
    """
    Generate a grouped horizontal bar plot for LOSO subject metrics.
    
    Parameters
    ----------
    loso_subject_df : pd.DataFrame
        DataFrame containing subject metrics.
    save_path : str
        Directory to save the plot.
    filename_base : str
        Base filename pattern.
    """
    import matplotlib.pyplot as plt
    import os
    
    if loso_subject_df is None or loso_subject_df.empty:
        print("Warning: No LOSO subject data to plot.")
        return
    
    # Ensure subject is string
    df = loso_subject_df.copy()
    if 'subject' not in df.columns:
        print("Warning: 'subject' column not found in LOSO data.")
        return
    df['subject'] = df['subject'].astype(str)
    
    # Sort by AUC
    if 'auc' in df.columns:
        df = df.sort_values('auc', ascending=True)
    
    # Select available metrics
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'] # Green, Blue, Red, Purple
    
    metrics = {
        'auc': {'label': 'AUC', 'color': colors[0]},
        'balanced_accuracy': {'label': 'Bal Acc', 'color': colors[1]},
        'auprc': {'label': 'AUPRC', 'color': colors[2]},
        'mcc': {'label': 'MCC', 'color': colors[3]}
    }
    
    metrics_to_plot = [m for m in metrics.keys() if m in df.columns]
    
    if not metrics_to_plot:
        print("Warning: No recognized metrics found in LOSO data.")
        return
    
    n_subjects = len(df)
    n_metrics = len(metrics_to_plot)
    bar_height = 0.8 / n_metrics
    
    # Calculate figure height dynamically
    fig_height = max(6, n_subjects * n_metrics * 0.25)
    plt.figure(figsize=(12, fig_height))
    
    # Create grouped bar plot
    y_indices = np.arange(n_subjects)
    
    for i, metric in enumerate(metrics_to_plot):
        props = metrics[metric]
        # Offset bars
        offset = (i - n_metrics/2 + 0.5) * bar_height
        
        plt.barh(
            y_indices + offset, 
            df[metric], 
            height=bar_height,
            label=props['label'],
            color=props['color'],
            alpha=0.8,
            edgecolor='white'
        )
    
    # Add chance lines
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Chance (0.5)', alpha=0.5)
    plt.axvline(x=0.0, color='gray', linestyle=':', linewidth=1, label='Chance (0.0)', alpha=0.5)
    
    plt.yticks(y_indices, df['subject'])
    plt.xlabel('Metric Score')
    plt.ylabel('Subject ID')
    plt.title('LOSO Performance by Subject')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)
    plt.xlim(-0.2, 1.05)
    
    plt.tight_layout()
    
    # Save with "barplots" in name as requested/expected
    output_path = os.path.join(save_path, f"{filename_base}_loso_subject_barplots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Generated LOSO subject metrics barplot: {output_path}") 