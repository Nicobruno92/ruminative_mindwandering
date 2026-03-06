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
    Plot and save normalized confusion matrix using Plotly.
    """
    import plotly.figure_factory as ff
    
    # Save raw confusion matrix to CSV
    pd.DataFrame(avg_conf_matrix, index=[negative_class_name, positive_class_name], 
                 columns=[f"Predicted {negative_class_name}", f"Predicted {positive_class_name}"]).to_csv(
                     f"{comparison_results_path}/{filename_base}_raw_confusion_matrix.csv")
    
    # Normalize
    row_sums = avg_conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_conf_matrix = avg_conf_matrix / row_sums
    pd.DataFrame(norm_conf_matrix, index=[negative_class_name, positive_class_name], 
                 columns=[f"Predicted {negative_class_name}", f"Predicted {positive_class_name}"]).to_csv(
                     f"{comparison_results_path}/{filename_base}_normalized_confusion_matrix.csv")

    z = norm_conf_matrix
    x = [negative_class_name, positive_class_name]
    y = [negative_class_name, positive_class_name]
    
    z_text = [[f"{val:.2f}<br>(n={count:.1f})" for val, count in zip(row_norm, row_avg)] 
              for row_norm, row_avg in zip(norm_conf_matrix, avg_conf_matrix)]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis', showscale=True)
    fig.update_layout(template='plotly_white', title='<b>Confusion Matrix (Normalized)</b>',
                      xaxis_title='Predicted Label', yaxis_title='True Label', width=600, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_confusion_matrix"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save Confusion Matrix plot: {e}")
    
    if cell_stats_df is not None:
        cell_stats_df.to_csv(f"{comparison_results_path}/{filename_base}_confusion_matrix_stats.csv", index=False)
        
    return fig


def plot_auc_distribution(mean_auc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save AUC distribution using Plotly.
    """
    auc_array = np.array(mean_auc_list)
    pd.DataFrame({'auc_values': auc_array}).to_csv(f"{comparison_results_path}/{filename_base}_auc_distribution_data.csv", index=False)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=auc_array, marker_color='#DE237B', opacity=0.7, nbinsx=20))
    mean_val = np.mean(auc_array)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Chance (0.5)")
    fig.update_layout(template='plotly_white', title=f"<b>AUC Distribution</b> ({n_runs} runs)",
                      xaxis_title="AUC", yaxis_title="Frequency", width=800, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_auc_distribution"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save AUC plot: {e}")
    return fig


def plot_balanced_accuracy_distribution(mean_bal_acc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save balanced accuracy distribution using Plotly.
    """
    bal_acc_array = np.array(mean_bal_acc_list)
    pd.DataFrame({'balanced_accuracy_values': bal_acc_array}).to_csv(f"{comparison_results_path}/{filename_base}_balanced_acc_distribution_data.csv", index=False)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bal_acc_array, marker_color='#DE237B', opacity=0.7, nbinsx=20))
    mean_val = np.mean(bal_acc_array)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Chance (0.5)")
    fig.update_layout(template='plotly_white', title=f"<b>Balanced Accuracy Distribution</b> ({n_runs} runs)",
                      xaxis_title="Balanced Accuracy", yaxis_title="Frequency", width=800, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_balanced_acc_distribution"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save BalAcc plot: {e}")
    return fig


def plot_auprc_distribution(mean_auprc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save AUPRC distribution using Plotly.
    """
    auprc_array = np.array(mean_auprc_list)
    pd.DataFrame({'auprc_values': auprc_array}).to_csv(f"{comparison_results_path}/{filename_base}_auprc_distribution_data.csv", index=False)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=auprc_array, marker_color='#DE237B', opacity=0.7, nbinsx=20))
    mean_val = np.mean(auprc_array)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text=f"Mean: {mean_val:.3f}")
    fig.update_layout(template='plotly_white', title=f"<b>AUPRC Distribution</b> ({n_runs} runs)",
                      xaxis_title="AUPRC", yaxis_title="Frequency", width=800, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_auprc_distribution"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save AUPRC plot: {e}")
    return fig


def plot_mcc_distribution(mean_mcc_list, comparison_results_path, filename_base, n_runs):
    """
    Plot and save MCC distribution using Plotly.
    """
    mcc_array = np.array(mean_mcc_list)
    pd.DataFrame({'mcc_values': mcc_array}).to_csv(f"{comparison_results_path}/{filename_base}_mcc_distribution_data.csv", index=False)
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=mcc_array, marker_color='#DE237B', opacity=0.7, nbinsx=20))
    mean_val = np.mean(mcc_array)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="black", annotation_text=f"Mean: {mean_val:.3f}")
    fig.add_vline(x=0.0, line_dash="dash", line_color="red", annotation_text="Chance (0.0)")
    fig.update_layout(template='plotly_white', title=f"<b>MCC Distribution</b> ({n_runs} runs)",
                      xaxis_title="MCC", yaxis_title="Frequency", width=800, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_mcc_distribution"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save MCC plot: {e}")
    return fig


def plot_roc_curve(mean_fpr, mean_tpr, std_tpr, comparison_results_path, filename_base, mean_auc, n_curves, 
                   comparison=None, all_fprs=None, all_tprs=None, all_aucs=None):
    """
    Plot and save ROC curve using Plotly.
    """
    fig = go.Figure()
    color = '#DE237B'
    
    if all_fprs is not None and all_tprs is not None:
        for fpr, tpr in zip(all_fprs, all_tprs):
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=color, width=1), opacity=0.1, showlegend=False))
            
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1); tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    fig.add_trace(go.Scatter(x=np.concatenate([mean_fpr, mean_fpr[::-1]]), y=np.concatenate([tprs_upper, tprs_lower[::-1]]),
                             fill='toself', fillcolor='rgba(222,35,123,0.1)', line=dict(color='rgba(255,255,255,0)'), name='± 1 std. dev.'))
    
    fig.add_trace(go.Scatter(x=mean_fpr, y=mean_tpr, mode='lines', line=dict(color=color, width=3), name=f"Mean ROC (AUC={mean_auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='black'), name='Chance'))
    
    fig.update_layout(template='plotly_white', title=f"<b>ROC Curves ({n_curves} folds)</b>",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", width=800, height=800)
    
    out_path = f"{comparison_results_path}/{filename_base}_roc_curve"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save ROC plot: {e}")
    
    pd.DataFrame({'mean_fpr': mean_fpr, 'mean_tpr': mean_tpr, 'std_tpr': std_tpr}).to_csv(f"{out_path}_data.csv", index=False)
    return fig


def plot_feature_importances(feature_names, mean_importances, std_importances, comparison_results_path, filename_base, top_n=20):
    """
    Plot and save error-bar dot plot of top N feature importances using Plotly.
    """
    indices = np.argsort(mean_importances)[::-1][:top_n]
    # Reverse for Plotly (so the highest feature is at the top of the y-axis)
    indices_plot = indices[::-1]
    
    y_labels = [feature_names[i] for i in indices_plot]
    x_means = mean_importances[indices_plot]
    x_errs = std_importances[indices_plot]

    fig = go.Figure()

    # Add the points with error bars
    fig.add_trace(go.Scatter(
        x=x_means,
        y=y_labels,
        mode='markers',
        marker=dict(color='#DE237B', size=8, symbol='circle'),
        error_x=dict(
            type='data',
            array=x_errs,
            color='#DE237B',
            thickness=1.5,
            width=3
        ),
        name='Importance'
    ))

    # Update layout to match modern aesthetic
    fig.update_layout(
        template='plotly_white',
        title=dict(text=f'<b>Top {len(y_labels)} Feature Importances</b>', font=dict(size=16)),
        xaxis_title=dict(text='Importance', font=dict(size=14)),
        yaxis_title=dict(text='Feature', font=dict(size=14)),
        height=max(400, len(y_labels) * 30),
        width=800,
        margin=dict(l=150, r=30, t=60, b=60),
        yaxis=dict(
            gridcolor='#F0F0F0',
            showgrid=True,
            showline=False,
            zeroline=False
        ),
        xaxis=dict(
            gridcolor='#F0F0F0',
            showgrid=True,
            zeroline=False
        ),
        showlegend=False
    )

    out_path = f"{comparison_results_path}/{filename_base}_feature_importances"
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_image(f"{out_path}.svg")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save Plotly images: {e}")

    # Save feature importances data
    importances_data_df = pd.DataFrame({
        'feature_name': feature_names,
        'mean_importance': mean_importances,
        'std_importance': std_importances,
        'rank': np.argsort(np.argsort(mean_importances)[::-1]) + 1
    })
    importances_data_df = importances_data_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    importances_data_df.to_csv(f"{out_path}_data.csv", index=False)


def plot_shap_beeswarm(shap_values, x_test, feature_names, comparison, save_dir, save_prefix, max_display=20):
    """
    Create a custom Plotly-based SHAP beeswarm plot.
    Replicates the look of SHAP library but in Plotly.
    """
    import numpy as np
    import pandas as pd
    
    # Handle SHAP values structure
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[0] # Take first class for binary
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_vals = shap_values[..., 0]
    else:
        shap_vals = shap_values

    # Calculate mean absolute SHAP to find top features
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_indices = top_indices[::-1] # Reverse for plotting (top features at top)
    
    fig = go.Figure()
    
    # We'll create one trace per feature for better legend/control, or one trace total
    # For a beeswarm, we need to jitter the points on the Y-axis
    for i, idx in enumerate(top_indices):
        f_name = feature_names[idx]
        f_shap = shap_vals[:, idx]
        
        # Jitter logic
        # We can use Plotly's box/violin with points or just manual scatter
        # Here we do manual scatter with jitter
        y_center = i
        jitter = np.random.normal(0, 0.05, size=len(f_shap))
        y_positions = y_center + jitter
        
        f_values = None
        if x_test is not None:
            f_values = np.asarray(x_test)[:, idx]
            # Normalize for color scale
            if np.max(f_values) != np.min(f_values):
                norm_values = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
            else:
                norm_values = np.zeros_like(f_values)
        else:
            norm_values = np.zeros_like(f_shap)
            
        fig.add_trace(go.Scatter(
            x=f_shap,
            y=y_positions,
            mode='markers',
            name=f_name,
            marker=dict(
                size=5,
                color=norm_values if x_test is not None else '#DE237B',
                colorscale='Viridis' if x_test is not None else None,
                showscale=True if (i == len(top_indices)-1 and x_test is not None) else False,
                colorbar=dict(title="Feature Value", titleside="top", tickvals=[0, 1], ticktext=["Low", "High"]) if (i == len(top_indices)-1 and x_test is not None) else None,
                opacity=0.8
            ),
            hovertext=[f"Feature: {f_name}<br>SHAP: {s:.4f}<br>Value: {v:.4f}" for s, v in zip(f_shap, f_values)] if x_test is not None else None,
            showlegend=False
        ))

    fig.update_layout(
        template='plotly_white',
        title=f"<b>SHAP Beeswarm: {comparison}</b>",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_indices))),
            ticktext=[feature_names[idx] for idx in top_indices],
            showgrid=True,
            gridcolor='#F0F0F0'
        ),
        xaxis=dict(showgrid=True, gridcolor='#F0F0F0', zeroline=True, zerolinecolor='black'),
        height=max(600, len(top_indices) * 40),
        width=900,
        margin=dict(l=200, r=50, t=80, b=80)
    )
    
    out_path = os.path.join(str(save_dir), f"{save_prefix}_shap_beeswarm")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save SHAP beeswarm plot: {e}")
        
    return fig


def plot_shap_feature_importance(shap_values, X, feature_names, comparison_results_path, filename_base, max_display=20):
    """
    Plot and save SHAP feature importance (bar) plot using Plotly.
    """
    # Calculate mean absolute SHAP
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_vals = shap_values[0]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_vals = shap_values[..., 0]
    else:
        shap_vals = shap_values
        
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    indices_plot = indices[::-1]
    
    y_labels = [feature_names[i] for i in indices_plot]
    x_values = mean_abs_shap[indices_plot]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_values, y=y_labels, orientation='h',
        marker_color='#DE237B'
    ))
    
    fig.update_layout(
        template='plotly_white',
        title=dict(text='<b>SHAP Feature Importance</b>', font=dict(size=16)),
        xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)',
        yaxis_title='Feature',
        width=800, height=max(400, len(y_labels) * 30)
    )
    
    out_path = f"{comparison_results_path}/{filename_base}_shap_importance"
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save SHAP importance plot: {e}")
        
    # Save data
    pd.DataFrame({'feature_name': feature_names, 'mean_abs_shap_value': mean_abs_shap}).to_csv(
        f"{out_path}_data.csv", index=False)
        
    return fig


# =============================================================================
# End of migrated functions
# =============================================================================

def plot_auc_distribution_comparison(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Plot AUC distribution comparison between true and permuted labels with Plotly.
    """
    if 'mean_auc' not in true_results.columns or 'mean_auc' not in perm_results.columns:
        print(f"Warning: AUC columns not found for {dimension}")
        return None
        
    true_aucs = true_results['mean_auc'].dropna().values
    perm_aucs = perm_results['mean_auc'].dropna().values
    
    if len(true_aucs) == 0 and len(perm_aucs) == 0:
        return None

    fig = go.Figure()
    
    # Histograms
    fig.add_trace(go.Histogram(
        x=perm_aucs, name='Permuted', marker_color='#999999', opacity=0.6,
        histnorm='probability density', nbinsx=20
    ))
    fig.add_trace(go.Histogram(
        x=true_aucs, name='True Model', marker_color='#DE237B', opacity=0.7,
        histnorm='probability density', nbinsx=20
    ))
    
    # Statistical lines
    true_mean = np.mean(true_aucs)
    
    fig.add_vline(x=true_mean, line_dash="dash", line_color="#DE237B", 
                 annotation_text=f"Mean True: {true_mean:.3f}", annotation_position="top right")
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", 
                 annotation_text="Chance (0.5)", annotation_position="top left")
    
    # Calculate p-value
    empirical_p = (perm_aucs >= true_mean).mean()
    
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=f"<b>Distribution of AUC Scores</b> ({len(true_aucs)} runs)<br><sup>p-value: {empirical_p:.4f}</sup>",
            font=dict(size=18)
        ),
        xaxis_title="AUC",
        yaxis_title="Probability Density",
        barmode='overlay',
        width=800, height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_image(save_path.replace('.png', '.pdf'))
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save Plotly AUC distribution plot: {e}")
            
    return fig


def plot_balanced_accuracy_distribution_comparison(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Plot balanced accuracy distribution comparison between true and permuted labels using Plotly.
    """
    if 'mean_balanced_accuracy' not in true_results.columns or 'mean_balanced_accuracy' not in perm_results.columns:
        print(f"Warning: Balanced Accuracy columns not found for {dimension}")
        return None
        
    true_vals = true_results['mean_balanced_accuracy'].dropna().values
    perm_vals = perm_results['mean_balanced_accuracy'].dropna().values
    
    if len(true_vals) == 0 and len(perm_vals) == 0:
        return None

    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=perm_vals, name='Permuted', marker_color='#999999', opacity=0.6,
        histnorm='probability density', nbinsx=20
    ))
    fig.add_trace(go.Histogram(
        x=true_vals, name='True Model', marker_color='#DE237B', opacity=0.7,
        histnorm='probability density', nbinsx=20
    ))
    
    true_mean = np.mean(true_vals)
    fig.add_vline(x=true_mean, line_dash="dash", line_color="#DE237B", 
                 annotation_text=f"Mean True: {true_mean:.3f}", annotation_position="top right")
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", 
                 annotation_text="Chance (0.5)", annotation_position="top left")
    
    empirical_p = (perm_vals >= true_mean).mean()
    
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=f"<b>Distribution of Balanced Accuracy Scores</b> ({len(true_vals)} runs)<br><sup>p-value: {empirical_p:.4f}</sup>",
            font=dict(size=18)
        ),
        xaxis_title="Balanced Accuracy",
        yaxis_title="Probability Density",
        barmode='overlay',
        width=800, height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_image(save_path.replace('.png', '.pdf'))
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save Plotly BalAcc distribution plot: {e}")
            
    return fig

def plot_feature_importance_distribution(results_df, feature_cols, dimension, model_type, top_n=15, save_path=None):
    """
    Plot feature importance distribution across runs using Plotly.
    """
    importance_cols = [col for col in results_df.columns if col.startswith('importance_')]
    
    if not importance_cols:
        print("No feature importance columns found")
        return None
    
    feature_importance_means = {}
    feature_importance_stds = {}
    
    for col in importance_cols:
        feature_name = col.replace('importance_', '')
        if feature_name in feature_cols:
            feature_importance_means[feature_name] = results_df[col].mean()
            feature_importance_stds[feature_name] = results_df[col].std()
    
    sorted_features = sorted(feature_importance_means.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    if not top_features:
        print("No valid feature importances found")
        return None
    
    feature_names = [f[0] for f in top_features]
    mean_importances = [f[1] for f in top_features]
    std_importances = [feature_importance_stds.get(f[0], 0) for f in top_features]
    
    # Use the same dot plot style as plot_feature_importances
    indices_plot = list(range(len(feature_names)))[::-1]
    y_labels = [feature_names[i] for i in indices_plot]
    x_means = [mean_importances[i] for i in indices_plot]
    x_errs = [std_importances[i] for i in indices_plot]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_means, y=y_labels, mode='markers+text',
        marker=dict(color='#DE237B', size=8),
        error_x=dict(type='data', array=x_errs, color='#DE237B', thickness=1.5),
        text=[f"{m:.3f}" for m in x_means], textposition="middle right"
    ))

    fig.update_layout(
        template='plotly_white',
        title=f"<b>Top Feature Importances: {dimension}</b>",
        xaxis_title="Importance",
        yaxis=dict(gridcolor='#F0F0F0'),
        xaxis=dict(gridcolor='#F0F0F0'),
        height=max(400, len(y_labels) * 30),
        width=800,
        showlegend=False
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_image(save_path.replace('.png', '.pdf'))
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save feature importance distribution plot: {e}")
            
    return fig

def plot_performance_metrics_summary(true_results, dimension, model_type, save_path=None):
    """
    Plot summary of all performance metrics using Plotly.
    """
    from plotly.subplots import make_subplots
    
    metrics = ['mean_auc', 'mean_balanced_accuracy', 'fold_precision', 'fold_recall']
    metric_names = ['AUC', 'Balanced Accuracy', 'Precision', 'Recall']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=metric_names)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        if metric in true_results.columns:
            values = true_results[metric].dropna().values
            if len(values) > 0:
                fig.add_trace(go.Histogram(
                    x=values, name=name, marker_color='#DE237B', opacity=0.7, nbinsx=15
                ), row=row, col=col)
                
                # Add mean line
                mean_val = np.mean(values)
                fig.add_vline(x=mean_val, line_dash="dash", line_color="black", row=row, col=col)
            else:
                fig.add_annotation(text="No data", row=row, col=col, showarrow=False)
        else:
            fig.add_annotation(text="Not found", row=row, col=col, showarrow=False)

    fig.update_layout(
        template='plotly_white',
        title=f"<b>Performance Metrics Summary: {dimension}</b>",
        height=800, width=1000,
        showlegend=False
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save performance summary plot: {e}")
            
    return fig

def create_dimension_comparison_plot(results_summary, model_type, metric='mean_auc', save_path=None):
    """
    Create a comparison plot across all dimensions using Plotly.
    """
    if not results_summary:
        return None
    
    dimension_names = []
    metric_means = []
    metric_stds = []
    
    for dimension, results_df in results_summary.items():
        if metric in results_df.columns:
            values = results_df[metric].dropna().values
            if len(values) > 0:
                dimension_names.append(dimension)
                metric_means.append(np.mean(values))
                metric_stds.append(np.std(values))
    
    if not dimension_names:
        return None
    
    # Sort
    sorted_data = sorted(zip(dimension_names, metric_means, metric_stds), key=lambda x: x[1], reverse=True)
    dimension_names, metric_means, metric_stds = zip(*sorted_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dimension_names, y=metric_means,
        error_y=dict(type='data', array=metric_stds),
        marker_color='#DE237B'
    ))
    
    fig.update_layout(
        template='plotly_white',
        title=f"<b>Comparison of {metric.replace('_', ' ').title()} Across Dimensions</b>",
        xaxis_title="Phenomenology Dimensions",
        yaxis_title=metric.replace('_', ' ').title(),
        height=600, width=1200
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save dimension comparison plot: {e}")
            
    return fig

def plot_permutation_test_results(true_results, perm_results, dimension, model_type, save_path=None):
    """
    Create comprehensive permutation test visualization using Plotly subplots.
    """
    from plotly.subplots import make_subplots
    
    if 'mean_auc' not in true_results.columns or 'mean_auc' not in perm_results.columns:
        print(f"Warning: Required columns not found for {dimension}")
        return None
        
    true_aucs = true_results['mean_auc'].dropna().values
    perm_aucs = perm_results['mean_auc'].dropna().values
    true_bal = true_results['mean_balanced_accuracy'].dropna().values
    perm_bal = perm_results['mean_balanced_accuracy'].dropna().values
    
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=("AUC Distribution", "Balanced Accuracy Distribution", "Performance Summary"),
        specs=[[{"colspan": 1}, {"colspan": 1}], [{"colspan": 2}, None]]
    )
    
    # AUC Histogram
    fig.add_trace(go.Histogram(x=perm_aucs, name='Permuted AUC', marker_color='#999999', opacity=0.6, histnorm='probability density'), row=1, col=1)
    fig.add_trace(go.Histogram(x=true_aucs, name='True AUC', marker_color='#DE237B', opacity=0.7, histnorm='probability density'), row=1, col=1)
    
    # Balanced Accuracy Histogram
    fig.add_trace(go.Histogram(x=perm_bal, name='Permuted BalAcc', marker_color='#999999', opacity=0.6, histnorm='probability density'), row=1, col=2)
    fig.add_trace(go.Histogram(x=true_bal, name='True BalAcc', marker_color='#DE237B', opacity=0.7, histnorm='probability density'), row=1, col=2)
    
    # Box Plot Summary
    fig.add_trace(go.Box(y=true_aucs, name='True AUC', marker_color='#DE237B'), row=2, col=1)
    fig.add_trace(go.Box(y=perm_aucs, name='Perm AUC', marker_color='#999999'), row=2, col=1)
    fig.add_trace(go.Box(y=true_bal, name='True BalAcc', marker_color='#DE237B'), row=2, col=1)
    fig.add_trace(go.Box(y=perm_bal, name='Perm BalAcc', marker_color='#999999'), row=2, col=1)

    fig.update_layout(
        template='plotly_white',
        title=f"<b>Permutation Test Results: {dimension}</b>",
        height=900, width=1200,
        barmode='overlay'
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save permutation test plot: {e}")
            
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
    
def create_model_comparison_plot(results_summary, model_type, metric='mean_auc', save_path=None):
    """
    Create a comparison violin plot across dimensions and models using Plotly.
    """
    import plotly.express as px
    
    plot_data = []
    
    # results_summary is {model_name: {dimension: results_df}}
    for m_name, dims in results_summary.items():
        for d_name, df in dims.items():
            if metric in df.columns:
                vals = df[metric].dropna().values
                for v in vals:
                    plot_data.append({'Dimension': d_name, 'Model': m_name, 'Value': v})
    
    if not plot_data:
        return None
        
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.violin(df_plot, x='Dimension', y='Value', color='Model', box=True, points='all',
                    template='plotly_white', color_discrete_sequence=COLORS)
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"<b>{metric.replace('_', ' ').title()} Comparison Across Models</b>",
        xaxis_title="Phenomenology Dimensions",
        yaxis_title=metric.replace('_', ' ').title(),
        height=800, width=1200
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save model comparison plot: {e}")
            
    return fig


def create_permutation_comparison_plot(results_dict, perm_results_dict, metric='mean_auc', save_path=None):
    """
    Create violin plots comparing true vs permutation results across models using Plotly.
    """
    import plotly.express as px
    
    plot_data = []
    
    common_models = set(results_dict.keys()).intersection(set(perm_results_dict.keys()))
    
    for model_name in common_models:
        true_res = results_dict[model_name]
        perm_res = perm_results_dict[model_name]
        
        common_dims = set(true_res.keys()).intersection(set(perm_res.keys()))
        
        for dimension in common_dims:
            # True
            if metric in true_res[dimension].columns:
                t_vals = true_res[dimension][metric].dropna().values
                for v in t_vals:
                    plot_data.append({'Dimension': dimension, 'Type': 'True', 'Value': v, 'Model': model_name})
            
            # Permuted
            if metric in perm_res[dimension].columns:
                p_vals = perm_res[dimension][metric].dropna().values
                for v in p_vals:
                    plot_data.append({'Dimension': dimension, 'Type': 'Permuted', 'Value': v, 'Model': model_name})

    if not plot_data:
        return None
        
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.violin(df_plot, x='Dimension', y='Value', color='Type', facet_row='Model',
                    box=True, template='plotly_white', color_discrete_map={'True': '#DE237B', 'Permuted': '#999999'})
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"<b>True vs Permuted Labels Comparison ({metric})</b>",
        height=400 * len(common_models), width=1200
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.write_image(save_path, scale=2)
            fig.write_html(save_path.replace('.png', '.html'))
        except Exception as e:
            print(f"Warning: Could not save permutation comparison plot: {e}")
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
    Plot combined histogram for true and permuted values with statistical annotations using Plotly.
    """
    true_vals = np.array(true_values); true_vals = true_vals[~np.isnan(true_vals)]
    perm_vals = np.array(perm_values); perm_vals = perm_vals[~np.isnan(perm_vals)]
    
    if len(true_vals) == 0 and len(perm_vals) == 0:
        return

    fig = go.Figure()
    
    if len(perm_vals) > 0:
        fig.add_trace(go.Histogram(x=perm_vals, name='Permuted', marker_color='#999999', opacity=0.6, nbinsx=20))
        fig.add_vline(x=np.mean(perm_vals), line_dash="dash", line_color="black", annotation_text=f"Perm Mean: {np.mean(perm_vals):.3f}")
        
    if len(true_vals) > 0:
        fig.add_trace(go.Histogram(x=true_vals, name='True Labels', marker_color='#DE237B', opacity=0.7, nbinsx=20))
        fig.add_vline(x=np.mean(true_vals), line_dash="dash", line_color="#DE237B", annotation_text=f"True Mean: {np.mean(true_vals):.3f}")

    if chance_value is not None:
        fig.add_vline(x=chance_value, line_dash="dot", line_color="red", annotation_text=f"Chance ({chance_value})")

    stats_text = f"MW-p: {mwu_p:.4g}<br>KS-p: {ks_p:.4g}"
    if empirical_p is not None: stats_text += f"<br>Emp-p: {empirical_p:.4g}"
    
    fig.add_annotation(text=stats_text, xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False, bgcolor="white", opacity=0.8, align="left")
    
    fig.update_layout(template='plotly_white', barmode='overlay', title=f"<b>{title if title else metric_name.title() + ' Distribution'}</b>",
                      xaxis_title=metric_name.title(), yaxis_title="Frequency", width=800, height=600)
    
    out_path = f"{comparison_results_path}/{filename_base}_{metric_name}_distribution"
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save distribution plot: {e}")
    print(f"Saved {metric_name} distribution plot to {comparison_results_path}/{filename_base}_{metric_name}_distribution.png")
    
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
    Plot consolidated permutation test results for multiple metrics using Plotly.
    """
    import plotly.subplots as sp
    
    if not results_dict:
        print("No results to plot.")
        return

    n_metrics = len(results_dict)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    # Create subplots
    fig = sp.make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{m}</b>" for m in results_dict.keys()],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    all_data_list = []
    
    for i, (metric_name, data) in enumerate(results_dict.items()):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        true_values = np.array(data['true_values'])
        perm_values = np.array(data['perm_values'])
        true_values = true_values[~np.isnan(true_values)]
        perm_values = perm_values[~np.isnan(perm_values)]
        
        if len(true_values) == 0 and len(perm_values) == 0:
            continue
            
        # Histograms
        if len(perm_values) > 0:
            fig.add_trace(
                go.Histogram(
                    x=perm_values,
                    name=f'Permuted ({metric_name})',
                    marker_color='#999999',
                    opacity=0.6,
                    histnorm='probability density',
                    nbinsx=20,
                    showlegend=False
                ),
                row=row, col=col
            )
            # Permuted mean line
            fig.add_vline(
                x=np.mean(perm_values),
                line_dash="dash",
                line_color="black",
                annotation_text=f"Perm Mean: {np.mean(perm_values):.3f}",
                annotation_position="top",
                annotation_font_size=10,
                row=row, col=col
            )
        
        if len(true_values) > 0:
            fig.add_trace(
                go.Histogram(
                    x=true_values,
                    name=f'True ({metric_name})',
                    marker_color=COLORS[i % len(COLORS)],
                    opacity=0.7,
                    histnorm='probability density',
                    nbinsx=20,
                    showlegend=False
                ),
                row=row, col=col
            )
            # True mean line
            fig.add_vline(
                x=np.mean(true_values),
                line_dash="dash",
                line_color=COLORS[i % len(COLORS)],
                annotation_text=f"True Mean: {np.mean(true_values):.3f}",
                annotation_position="top left",
                annotation_font_size=10,
                row=row, col=col
            )
            
        # Chance line
        chance = None
        if chance_values and metric_name in chance_values:
            chance = chance_values[metric_name]
        elif chance_values and metric_name.lower() in chance_values:
             chance = chance_values[metric_name.lower()]
             
        if chance is None:
            if 'auc' in metric_name.lower() or 'balanced' in metric_name.lower():
                chance = 0.5
            elif 'mcc' in metric_name.lower():
                chance = 0.0
                
        if chance is not None:
             fig.add_vline(
                 x=chance,
                 line_dash="dot",
                 line_color="red",
                 annotation_text=f"Chance ({chance})",
                 annotation_position="bottom right",
                 annotation_font_size=10,
                 row=row, col=col
             )

        # Update axes
        fig.update_xaxes(title_text=metric_name, row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)
        
        # Add stats text as annotation if p-values exist
        p_val = data.get('p_value', np.nan)
        emp_p = data.get('empirical_p', np.nan)
        if not np.isnan(p_val) or not np.isnan(emp_p):
            stat_str = ""
            if not np.isnan(p_val) and not isinstance(p_val, str):
                stat_str += f"MW-p: {p_val:.4f}<br>"
            if not np.isnan(emp_p) and not isinstance(emp_p, str):
                stat_str += f"Emp-p: {emp_p:.4f}"
            
            fig.add_annotation(
                text=stat_str,
                xref=f"x{i+1 if i>0 else ''} domain", yref=f"y{i+1 if i>0 else ''} domain",
                x=0.05, y=0.95, showarrow=False,
                bgcolor="white", opacity=0.8,
                font=dict(size=10), align="left",
                row=row, col=col
            )

        # Data collection for CSV
        max_len = max(len(true_values), len(perm_values))
        metric_df = pd.DataFrame({
            'run_idx': range(max_len),
            'metric': metric_name,
            'true_value': np.pad(true_values, (0, max_len - len(true_values)), constant_values=np.nan),
            'perm_value': np.pad(perm_values, (0, max_len - len(perm_values)), constant_values=np.nan)
        })
        all_data_list.append(metric_df)

    fig.update_layout(
        template='plotly_white',
        title_text=f"<b>Permutation Test Results: {dimension}</b> ({model_type.upper()})",
        height=400 * n_rows,
        width=1000,
        barmode='overlay',
        showlegend=False
    )
    
    out_path = f"{save_path}/{filename_base}_consolidated"
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save Plotly permutation plots: {e}")

    # Save consolidated CSVs
    if all_data_list:
        pd.concat(all_data_list).to_csv(f"{out_path}_data.csv", index=False)      # Collect stats
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

    return fig

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
    Generate a grouped horizontal bar plot for LOSO subject metrics using Plotly.
    """
    if loso_subject_df is None or loso_subject_df.empty:
        return
    
    df = loso_subject_df.copy()
    df['subject'] = df['subject'].astype(str)
    if 'auc' in df.columns: df = df.sort_values('auc', ascending=True)
    
    metrics_info = {
        'auc': {'label': 'AUC', 'color': COLORS[0]},
        'balanced_accuracy': {'label': 'Bal Acc', 'color': COLORS[1]},
        'auprc': {'label': 'AUPRC', 'color': COLORS[2]},
        'mcc': {'label': 'MCC', 'color': COLORS[3]}
    }
    
    metrics_to_plot = [m for m in metrics_info.keys() if m in df.columns]
    if not metrics_to_plot: return
    
    fig = go.Figure()
    for m in metrics_to_plot:
        fig.add_trace(go.Bar(y=df['subject'], x=df[m], name=metrics_info[m]['label'], marker_color=metrics_info[m]['color'], orientation='h'))
        
    fig.add_vline(x=0.5, line_dash="dash", line_color="black", opacity=0.5)
    fig.add_vline(x=0.0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(template='plotly_white', barmode='group', title="<b>LOSO Performance by Subject</b>",
                      xaxis_title="Metric Score", yaxis_title="Subject ID", 
                      height=max(600, len(df) * 30), width=1000, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    out_path = os.path.join(save_path, f"{filename_base}_loso_subject_barplots")
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save LOSO subject metrics plot: {e}")
    
    return fig

def plot_subject_level_densities(true_subject_metrics_list, perm_subject_metrics_list, dimension_name, save_path, filename_base, metric='auc'):
    """
    Generate complex Plotly plot with True/Permuted densities per subject, and pie charts of significant subjects.
    true_subject_metrics_list: List of DataFrames/dicts per run, capturing subject metrics for True.
    perm_subject_metrics_list: List of DataFrames/dicts per run, capturing subject metrics for Permuted.
    """
    import plotly.subplots as sp
    from scipy.stats import mannwhitneyu
    
    # 1. Compile arrays of true vs permuted metrics per subject
    def _extract_data(metrics_list, type_label):
        extracted = []
        for run in metrics_list:
            if isinstance(run, pd.DataFrame):
                for _, r in run.iterrows():
                    val = r.get(metric, r.get(f'mean_{metric}'))
                    extracted.append({'subject': str(r['subject']), 'value': val, 'type': type_label})
            elif isinstance(run, dict):
                if 'loso_subject_metrics' in run and run['loso_subject_metrics'] is not None:
                    for r in run['loso_subject_metrics']:
                        val = r.get(metric, r.get(f'mean_{metric}'))
                        extracted.append({'subject': str(r['subject']), 'value': val, 'type': type_label})
                elif 'subject' in run: # WS direct dict format
                    val = run.get(metric, run.get(f'mean_{metric}'))
                    extracted.append({'subject': str(run['subject']), 'value': val, 'type': type_label})
            elif isinstance(run, (list, tuple)):
                extracted.extend(_extract_data(run, type_label))
        return extracted
        
    true_data = _extract_data(true_subject_metrics_list, 'True')
    perm_data = _extract_data(perm_subject_metrics_list, 'Permuted')
                 
    df = pd.DataFrame(true_data + perm_data)
    df = df.dropna()
    if df.empty: return
    
    subjects = sorted(df['subject'].unique(), key=lambda x: int(x) if x.isdigit() else x, reverse=True)
    
    # 2. Calculate significance per subject
    significant_subjects = 0
    p_values = {}
    for sub in subjects:
        t_vals = df[(df['subject'] == sub) & (df['type'] == 'True')]['value'].values
        p_vals = df[(df['subject'] == sub) & (df['type'] == 'Permuted')]['value'].values
        if len(t_vals) > 0 and len(p_vals) > 0:
            if np.mean(t_vals) > np.mean(p_vals):
                 _, p = mannwhitneyu(t_vals, p_vals, alternative='greater')
                 p_values[sub] = p
                 if p < 0.05: significant_subjects += 1
            else:
                 p_values[sub] = 1.0
        else:
            p_values[sub] = 1.0
            
    pct_sig = (significant_subjects / len(subjects)) * 100 if subjects else 0
    
    # 3. Create Subplots layout
    # Since we might call this per dimension or a consolidated grid, we do it per dimension.
    # Layout: Rows = 2 (Violins, Pie), Cols = 1
    # But to match user design: many dimensions col by col. We will do 1 dimension plot here.
    
    fig = sp.make_subplots(rows=2, cols=1, row_heights=[0.85, 0.15], shared_xaxes=False, vertical_spacing=0.05,
                           specs=[[{"type": "xy"}], [{"type": "domain"}]])
    
    # Add violins
    for i, sub in enumerate(subjects):
        t_vals = df[(df['subject'] == sub) & (df['type'] == 'True')]['value']
        p_vals = df[(df['subject'] == sub) & (df['type'] == 'Permuted')]['value']
        
        # We simulate Ridgeline by using Violins with strong horizontal orientation.
        if len(p_vals) > 0:
            fig.add_trace(go.Violin(x=p_vals, y=[sub]*len(p_vals), side='positive', line_color='rgba(0,0,0,0)',
                                    fillcolor='#d3d3d3', opacity=0.6, name='Permuted', hoverinfo='none', showlegend=(i==0)), row=1, col=1)
        if len(t_vals) > 0:
            fig.add_trace(go.Violin(x=t_vals, y=[sub]*len(t_vals), side='negative', line_color='rgba(0,0,0,0)',
                                    fillcolor='#DE237B', opacity=0.8, name='True', hoverinfo='none', showlegend=(i==0)), row=1, col=1)
            
        # Add significance stars
        p = p_values.get(sub, 1.0)
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if star:
            fig.add_annotation(x=max(t_vals.max(), p_vals.max() if len(p_vals)>0 else 0) + 0.05, y=sub, text=star, showarrow=False,
                               font=dict(size=14, color='black'), yanchor='middle', row=1, col=1)

    fig.update_traces(orientation='h', width=1.5, points=False, box_visible=False, meanline_visible=False, row=1, col=1)
    
    chance = 0.5 if metric in ('auc', 'balanced_accuracy') else 0.0
    fig.add_vline(x=chance, line_dash='dash', line_color='black', row=1, col=1)
    
    # Add Donut pie chart for % significant
    fig.add_trace(go.Pie(labels=['Sig', 'Not Sig'], values=[pct_sig, 100-pct_sig], hole=0.6,
                         marker=dict(colors=['#DE237B', '#d3d3d3']), textinfo='none', hoverinfo='none', showlegend=False), row=2, col=1)
    # Donut annotation
    fig.add_annotation(text=f"<b>{pct_sig:.0f}%</b>", xref='x domain', yref='y domain', x=0.5, y=0.5, 
                       showarrow=False, font=dict(size=16, color='#DE237B'), row=2, col=1)
    
    fig.update_layout(title=f"<b>Subject-Level Density: {dimension_name} ({metric.upper()})</b>",
                      template='plotly_white', height=max(600, len(subjects)*40), width=400,
                      yaxis=dict(title="Subject ID", tickmode='array', tickvals=subjects, ticktext=subjects),
                      xaxis=dict(title=metric.upper(), range=[max(0.0, chance-0.2), 1.0]),
                      violingap=0, violinmode='overlay')
                      
    out_path = os.path.join(save_path, f"{filename_base}_subject_densities_{metric}")
    os.makedirs(save_path, exist_ok=True)
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
    except Exception as e: print(f"Warning: Could not save subject densities plot: {e}")
    return fig


def plot_shap_comparative_boxplots(true_shap_runs, perm_shap_runs, feature_names, save_path, filename_base, num_features=10):
    """
    Generate horizontal boxplots comparing True Mean |SHAP| vs Permuted (same feature) vs Permuted (same rank) across runs.
    true_shap_runs: List of stacked SHAP value arrays per run (runs x samples x features)
    perm_shap_runs: List of stacked SHAP value arrays per perm_run (perm_runs x samples x features)
    """
    if not true_shap_runs or not perm_shap_runs: return
    
    # Calculate Mean |SHAP| per run for True
    true_means_per_run = np.array([np.mean(np.abs(run_shap), axis=0) for run_shap in true_shap_runs]) # shape: (n_runs, n_features)
    
    # Calculate Mean |SHAP| per run for Permuted
    perm_means_per_run = np.array([np.mean(np.abs(run_shap), axis=0) for run_shap in perm_shap_runs]) # shape: (n_perms, n_features)
    
    # Determine top N features from the TRUE overall mean
    true_overall_mean = np.mean(true_means_per_run, axis=0)
    top_indices = np.argsort(true_overall_mean)[::-1][:num_features]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    plot_data = []
    
    # Gather data for Top N features
    for rank, feature_idx in enumerate(top_indices):
        feature_name = feature_names[feature_idx]
        
        # 1. True distribution
        for val in true_means_per_run[:, feature_idx]:
             plot_data.append({'Feature': feature_name, 'Rank': rank, 'Value': val, 'Type': 'True Labels'})
             
        # 2. Permuted (Same Feature)
        for val in perm_means_per_run[:, feature_idx]:
             plot_data.append({'Feature': feature_name, 'Rank': rank, 'Value': val, 'Type': 'Shuffled (Same Feature)'})
             
        # 3. Permuted (Same Rank)
        # Sort each perm run's features by importance, get the one at `rank`
        for perm_run in perm_means_per_run:
             sorted_perm = np.sort(perm_run)[::-1]
             val = sorted_perm[rank]
             plot_data.append({'Feature': feature_name, 'Rank': rank, 'Value': val, 'Type': 'Shuffled (Same Rank)'})
             
    df = pd.DataFrame(plot_data)
    
    fig = go.Figure()
    # To match user's horizontal grouped structure
    types = ['True Labels', 'Shuffled (Same Feature)', 'Shuffled (Same Rank)']
    colors = ['#8CC63F', '#808080', '#D3E0A3'] # Green, Dark Gray, Light Green (as requested in style)
    
    # Plotly Box plots grouped
    for i, t in enumerate(types):
        df_t = df[df['Type'] == t]
        fig.add_trace(go.Box(
            x=df_t['Value'], y=df_t['Feature'], orientation='h', name=t, marker_color=colors[i],
            boxmean=True, line_width=1, marker_size=2
        ))
        
    fig.update_layout(
        template='plotly_white', boxmode='group', title="<b>Top Features SHAP Importance (True vs Shuffled)</b>",
        xaxis_title="Mean |SHAP value| across runs", yaxis_title="Feature", height=max(400, num_features*60), width=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(autorange="reversed") # Highest rank at top
    )
    
    out_path = os.path.join(save_path, f"{filename_base}_shap_comparative_boxplots")
    os.makedirs(save_path, exist_ok=True)
    try:
        fig.write_image(f"{out_path}.png", scale=2); fig.write_image(f"{out_path}.pdf"); fig.write_html(f"{out_path}.html")
        df.to_csv(f"{out_path}.csv", index=False)
    except Exception as e: print(f"Warning: Could not save SHAP comparative plot: {e}")
    return fig