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

def set_plot_style(style='seaborn-v0_8-white'):
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
        'savefig.bbox': 'tight',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
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
                colorbar=dict(title=dict(text="Feature Value", side="top"), tickvals=[0, 1], ticktext=["Low", "High"]) if (i == len(top_indices)-1 and x_test is not None) else None,
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
        pd.concat(all_data_list).to_csv(f"{out_path}_data.csv", index=False)

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


# =============================================================================
# SUBJECT-LEVEL RIDGELINE  —  True distributions vs Permuted per subject
# =============================================================================

def plot_subject_distribution_ridgelines(
    true_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    metric: str,
    save_path: str,
    filename_base: str,
    dimension: str,
) -> go.Figure:
    """
    Horizontal ridgeline plot comparing per-subject true vs permuted distributions.

    For each subject, the true distribution (colored, from N real runs) and the
    permuted distribution (gray, from N permutation runs) are drawn as horizontal
    half-violins stacked on the same y position. Significance stars are added on
    the right. A donut chart summarises the fraction of significant subjects.

    Parameters
    ----------
    true_df : pd.DataFrame
        Stacked per-subject metrics from all true runs. Must have columns
        ``subject`` and at least the column named by *metric*.
    perm_df : pd.DataFrame
        Same format but from permutation runs. May be empty.
    metric : str
        Bare metric column name: ``'auc'``, ``'balanced_accuracy'``,
        ``'auprc'``, or ``'mcc'``.
    save_path : str
        Directory where files are written.
    filename_base : str
        Prefix for output file names.
    dimension : str
        Contrast name used in the title.

    Returns
    -------
    go.Figure
    """
    import plotly.subplots as sp
    from scipy.stats import mannwhitneyu

    # --- normalise column names (strip mean_ prefix if present) -------------
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={c: c.replace("mean_", "", 1) for c in df.columns})

    true_df = _norm(true_df.copy())
    perm_df = _norm(perm_df.copy()) if not perm_df.empty else perm_df

    if metric not in true_df.columns:
        print(f"    ! Column '{metric}' not found in subject metrics — skipping")
        return

    # Keep only finite values
    true_df = true_df[true_df[metric].notna()]
    if perm_df.empty or metric not in perm_df.columns:
        perm_df = pd.DataFrame()

    subjects = sorted(
        true_df["subject"].astype(str).unique(),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )

    TRUE_COLOR  = "#DE237B"
    PERM_COLOR  = "rgba(180,180,180,0.65)"
    PERM_LINE   = "rgba(100,100,100,0.8)"
    CHANCE = 0.5 if metric in ("auc", "balanced_accuracy", "auprc") else 0.0

    # --- significance per subject -------------------------------------------
    p_values: dict = {}
    for sub in subjects:
        t = true_df[true_df["subject"].astype(str) == sub][metric].values
        p = (perm_df[perm_df["subject"].astype(str) == sub][metric].values
             if not perm_df.empty else np.array([]))
        if len(t) == 0:
            p_values[sub] = 1.0
        elif len(p) == 0:
            # No perm data: compare vs chance level (empirical)
            p_values[sub] = float(np.mean(t <= CHANCE)) if CHANCE is not None else 1.0
        elif len(t) == 1:
            # Single true value: empirical p from perm distribution
            p_values[sub] = float(np.mean(p >= t[0]))
        else:
            _, pv = mannwhitneyu(t, p, alternative="greater")
            p_values[sub] = float(pv)

    n_sig = sum(v < 0.05 for v in p_values.values())
    pct_sig = n_sig / len(subjects) * 100 if subjects else 0.0

    # --- layout: ridgeline rows + donut at bottom ---------------------------
    n_subj = len(subjects)
    main_height = max(500, n_subj * 38)
    donut_height = 220

    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[main_height, donut_height],
        vertical_spacing=0.02,
        specs=[[{"type": "xy"}], [{"type": "domain"}]],
    )

    legend_added = {"true": False, "perm": False}

    for i, sub in enumerate(reversed(subjects)):   # top = highest subject id
        t_vals = true_df[true_df["subject"].astype(str) == sub][metric].values
        p_vals = (perm_df[perm_df["subject"].astype(str) == sub][metric].values
                  if not perm_df.empty else np.array([]))

        # --- permuted half-violin (positive / right half) ---
        if len(p_vals) >= 2:
            fig.add_trace(
                go.Violin(
                    x=p_vals, y=[sub] * len(p_vals),
                    orientation="h",
                    side="positive",
                    fillcolor=PERM_COLOR, line_color=PERM_LINE,
                    width=1.6, points=False, box_visible=False, meanline_visible=True,
                    meanline=dict(color=PERM_LINE, width=1.5),
                    name="Permuted",
                    showlegend=not legend_added["perm"],
                    legendgroup="perm",
                    hoverinfo="x+name",
                ),
                row=1, col=1,
            )
            legend_added["perm"] = True
        elif len(p_vals) == 1:
            fig.add_trace(
                go.Scatter(
                    x=p_vals, y=[sub],
                    mode="markers",
                    marker=dict(color=PERM_LINE, size=7, symbol="line-ew-open"),
                    name="Permuted", showlegend=not legend_added["perm"],
                    legendgroup="perm",
                ),
                row=1, col=1,
            )
            legend_added["perm"] = True

        # --- true half-violin (negative / left half) ---
        if len(t_vals) >= 2:
            fig.add_trace(
                go.Violin(
                    x=t_vals, y=[sub] * len(t_vals),
                    orientation="h",
                    side="negative",
                    fillcolor=TRUE_COLOR,
                    line_color=TRUE_COLOR,
                    width=1.6, points=False, box_visible=True,
                    meanline_visible=True,
                    meanline=dict(color="white", width=1.5),
                    name="True",
                    showlegend=not legend_added["true"],
                    legendgroup="true",
                    hoverinfo="x+name",
                ),
                row=1, col=1,
            )
            legend_added["true"] = True
        elif len(t_vals) >= 1:
            fig.add_trace(
                go.Scatter(
                    x=t_vals, y=[sub],
                    mode="markers",
                    marker=dict(color=TRUE_COLOR, size=8, symbol="diamond"),
                    name="True", showlegend=not legend_added["true"],
                    legendgroup="true",
                ),
                row=1, col=1,
            )
            legend_added["true"] = True

        # --- significance stars ----------------------------------------------
        pv = p_values.get(sub, 1.0)
        star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        if star:
            all_x = np.concatenate([t_vals, p_vals]) if len(p_vals) else t_vals
            x_star = float(np.nanmax(all_x)) + 0.04 if len(all_x) else CHANCE + 0.06
            fig.add_annotation(
                x=x_star, y=sub,
                text=f"<b>{star}</b>",
                showarrow=False,
                font=dict(size=13, color="#111111"),
                xanchor="left", yanchor="middle",
                row=1, col=1,
            )

    # chance line
    fig.add_vline(
        x=CHANCE, line_dash="dash", line_color="#333333",
        line_width=1.5, opacity=0.7,
        row=1, col=1,
    )

    # --- donut chart ---------------------------------------------------------
    donut_labels = ["Significant", "Not significant"]
    donut_values = [round(pct_sig, 1), round(100 - pct_sig, 1)]
    fig.add_trace(
        go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.62,
            marker=dict(colors=[TRUE_COLOR, "#d3d3d3"]),
            textinfo="none",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_annotation(
        text=f"<b>{pct_sig:.0f}%</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.07,
        showarrow=False,
        font=dict(size=18, color=TRUE_COLOR),
        xanchor="center", yanchor="middle",
    )

    metric_label = metric.replace("_", " ").upper()
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>Subject Distributions — {dimension} | {metric_label}</b>",
            font=dict(size=14),
        ),
        height=main_height + donut_height + 80,
        width=520,
        violingap=0.05,
        violingroupgap=0.0,
        violinmode="overlay",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=11),
        ),
        xaxis=dict(
            title=metric_label,
            range=[max(-0.05, CHANCE - 0.25), 1.05],
        ),
        yaxis=dict(
            title="Subject ID",
            type="category",
            categoryorder="array",
            categoryarray=[str(s) for s in reversed(subjects)],
            tickfont=dict(size=10),
        ),
        margin=dict(l=60, r=60, t=60, b=30),
    )

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_subject_ridgelines_{metric}")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save subject ridgeline plot: {e}")

    return fig

def plot_true_vs_perm_violins(
    results_dict: dict,
    dimension: str,
    model_type: str,
    save_path: str,
    filename_base: str,
) -> go.Figure:
    """
    Generate side-by-side violin plots comparing true vs permuted distributions.

    One subplot per metric. Each subplot contains two violins (True in colour,
    Permuted in grey), individual run points jittered on top, a horizontal
    dashed chance line, and a p-value annotation.

    Parameters
    ----------
    results_dict : dict
        Keys are metric labels (str). Values are dicts with:
        - ``true_values``  : array-like of metric values from real runs
        - ``perm_values``  : array-like of metric values from permutation runs
        - ``p_value``      : Mann-Whitney U p-value (float)
        - ``empirical_p``  : empirical p-value (float)
    dimension : str
        Contrast name used in the figure title.
    model_type : str
        Model identifier used in the figure title.
    save_path : str
        Directory where output files are written.
    filename_base : str
        Prefix for output file names.

    Returns
    -------
    go.Figure
        The Plotly figure object.
    """
    import plotly.subplots as sp

    if not results_dict:
        print("No results to plot.")
        return

    # Chance levels per metric keyword
    CHANCE: dict = {"auc": 0.5, "balanced": 0.5, "mcc": 0.0, "auprc": 0.5}
    METRIC_COLORS: dict = {
        "AUC": "#4C72B0",
        "Balanced Accuracy": "#DD8452",
        "AUPRC": "#55A868",
        "MCC": "#C44E52",
    }
    PERM_COLOR = "#AAAAAA"
    JITTER_SEED = 42
    rng = np.random.default_rng(JITTER_SEED)

    n_metrics = len(results_dict)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>{m}</b>" for m in results_dict.keys()],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    for idx, (metric_name, data) in enumerate(results_dict.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        true_vals = np.asarray(data["true_values"], dtype=float)
        perm_vals = np.asarray(data["perm_values"], dtype=float)
        true_vals = true_vals[np.isfinite(true_vals)]
        perm_vals = perm_vals[np.isfinite(perm_vals)]

        true_color = METRIC_COLORS.get(metric_name, COLORS[idx % len(COLORS)])
        p_val = data.get("p_value", np.nan)
        emp_p = data.get("empirical_p", np.nan)

        # Determine chance level
        chance = None
        for kw, val in CHANCE.items():
            if kw in metric_name.lower():
                chance = val
                break

        # --- Permuted violin ---
        if len(perm_vals) > 0:
            fig.add_trace(
                go.Violin(
                    y=perm_vals,
                    x=["Permuted"] * len(perm_vals),
                    name=f"Permuted",
                    fillcolor=PERM_COLOR,
                    line_color="#666666",
                    opacity=0.7,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=(idx == 0),
                    legendgroup="perm",
                    points=False,
                ),
                row=row, col=col,
            )
            # Jittered strip
            jitter_x = rng.uniform(-0.08, 0.08, size=len(perm_vals))
            fig.add_trace(
                go.Scatter(
                    y=perm_vals,
                    x=["Permuted"] * len(perm_vals),
                    mode="markers",
                    marker=dict(color="#555555", size=4, opacity=0.6,
                                line=dict(width=0.5, color="white")),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        # --- True violin ---
        if len(true_vals) > 0:
            fig.add_trace(
                go.Violin(
                    y=true_vals,
                    x=["True"] * len(true_vals),
                    name=f"True",
                    fillcolor=true_color,
                    line_color=true_color,
                    opacity=0.75,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=(idx == 0),
                    legendgroup="true",
                    points=False,
                ),
                row=row, col=col,
            )
            # Jittered strip
            fig.add_trace(
                go.Scatter(
                    y=true_vals,
                    x=["True"] * len(true_vals),
                    mode="markers",
                    marker=dict(color=true_color, size=5, opacity=0.8,
                                line=dict(width=0.8, color="white")),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        # Chance line
        if chance is not None:
            fig.add_hline(
                y=chance,
                line_dash="dot",
                line_color="red",
                line_width=1.5,
                opacity=0.6,
                row=row, col=col,
            )

        # P-value annotation inside subplot
        p_text = ""
        if np.isfinite(p_val):
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
        if np.isfinite(emp_p):
            p_text += f"<br>p_emp = {emp_p:.3f}"

        if p_text:
            fig.add_annotation(
                text=p_text,
                xref=f"x{idx + 1}" if idx > 0 else "x",
                yref=f"y{idx + 1}" if idx > 0 else "y",
                x=0.98, y=0.02,
                xanchor="right", yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#333333"),
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                row=row, col=col,
            )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>True vs Permuted — {dimension} | {model_type.upper()}</b>",
            font=dict(size=16),
        ),
        violinmode="group",
        height=450 * n_rows,
        width=600 * n_cols,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Metric value")

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_true_vs_perm_violins")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save violin plot: {e}")

    return fig


# =============================================================================
# GLOBAL PERMUTATION HISTOGRAM  —  True distribution vs Permuted per metric
# =============================================================================

def plot_global_permutation_histogram(
    results_dict: dict,
    dimension: str,
    model_type: str,
    save_path: str,
    filename_base: str,
) -> go.Figure:
    """
    Overlaid histogram and KDE showing the null permutation distribution against
    the true run value(s) for each metric.

    Each subplot shows a filled gray histogram for the permuted values, a gray
    KDE curve, a colored vertical line at the mean of the true run values, and
    the empirical + MWU p-value.

    Parameters
    ----------
    results_dict : dict
        Same format as ``plot_true_vs_perm_violins``:
        keys = metric labels; values contain ``true_values``, ``perm_values``,
        ``p_value``, ``empirical_p``.
    dimension : str
        Contrast name used in the figure title.
    model_type : str
        Model identifier used in the figure title.
    save_path : str
        Directory where output files are written.
    filename_base : str
        Prefix for output file names.

    Returns
    -------
    go.Figure
    """
    import plotly.subplots as sp
    from scipy.stats import gaussian_kde

    if not results_dict:
        print("No results to plot (histogram).")
        return None

    CHANCE: dict = {"auc": 0.5, "balanced": 0.5, "mcc": 0.0, "auprc": 0.5}
    METRIC_COLORS: dict = {
        "AUC": "#4C72B0",
        "Balanced Accuracy": "#DD8452",
        "AUPRC": "#55A868",
        "MCC": "#C44E52",
    }
    PERM_FILL = "rgba(170,170,170,0.45)"
    PERM_LINE_COLOR = "#666666"

    n_metrics = len(results_dict)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig = sp.make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"<b>{m}</b>" for m in results_dict.keys()],
        vertical_spacing=0.18,
        horizontal_spacing=0.14,
    )

    for idx, (metric_name, data) in enumerate(results_dict.items()):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        true_vals = np.asarray(data["true_values"], dtype=float)
        perm_vals = np.asarray(data["perm_values"], dtype=float)
        true_vals = true_vals[np.isfinite(true_vals)]
        perm_vals = perm_vals[np.isfinite(perm_vals)]

        true_color = METRIC_COLORS.get(metric_name, "#DE237B")
        p_val = data.get("p_value", np.nan)
        emp_p = data.get("empirical_p", np.nan)

        chance = None
        for kw, val in CHANCE.items():
            if kw in metric_name.lower():
                chance = val
                break

        x_all = np.concatenate([true_vals, perm_vals]) if len(perm_vals) else true_vals
        x_min = float(np.nanmin(x_all)) - 0.05 if len(x_all) else -0.1
        x_max = float(np.nanmax(x_all)) + 0.05 if len(x_all) else 1.1
        x_range = np.linspace(x_min, x_max, 300)

        # --- Permuted histogram ---
        if len(perm_vals) >= 2:
            fig.add_trace(
                go.Histogram(
                    x=perm_vals,
                    histnorm="probability density",
                    marker=dict(color=PERM_FILL, line=dict(color=PERM_LINE_COLOR, width=0.5)),
                    name="Permuted" if idx == 0 else None,
                    showlegend=(idx == 0),
                    legendgroup="perm",
                    nbinsx=20,
                    opacity=0.7,
                ),
                row=row, col=col,
            )
            # KDE overlay for permuted
            kde_perm = gaussian_kde(perm_vals)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_perm(x_range),
                    mode="lines",
                    line=dict(color=PERM_LINE_COLOR, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row, col=col,
            )

        # --- True value(s): vertical line(s) ---
        if len(true_vals) >= 1:
            true_mean = float(np.mean(true_vals))
            # Vertical line at mean of true values
            fig.add_vline(
                x=true_mean,
                line=dict(color=true_color, width=2.5, dash="solid"),
                row=row, col=col,
            )
            # If multiple true values, add individual markers as rug
            if len(true_vals) > 1:
                kde_true = gaussian_kde(true_vals)
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde_true(x_range),
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=f"rgba{tuple(list(int(true_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + [0.2])}",
                        line=dict(color=true_color, width=2),
                        name="True" if idx == 0 else None,
                        showlegend=(idx == 0 and len(true_vals) > 1),
                        legendgroup="true",
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

        # Chance baseline
        if chance is not None:
            fig.add_vline(
                x=chance,
                line=dict(color="red", width=1.2, dash="dot"),
                row=row, col=col,
            )

        # P-value annotation
        p_lines = []
        if np.isfinite(p_val):
            p_lines.append(f"MWU p = {p_val:.3f}" if p_val >= 0.001 else "MWU p < 0.001")
        if np.isfinite(emp_p):
            p_lines.append(f"emp p = {emp_p:.3f}")
        if p_lines:
            fig.add_annotation(
                text="<br>".join(p_lines),
                xref=f"x{idx + 1}" if idx > 0 else "x",
                yref=f"y{idx + 1}" if idx > 0 else "y",
                x=0.97, y=0.97,
                xanchor="right", yanchor="top",
                showarrow=False,
                font=dict(size=10, color="#333333"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="#CCCCCC",
                borderwidth=1,
                row=row, col=col,
            )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>Permutation Null Distribution — {dimension} | {model_type.upper()}</b>",
            font=dict(size=15),
        ),
        barmode="overlay",
        height=380 * n_rows,
        width=580 * n_cols,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
        ),
    )
    fig.update_xaxes(title_text="Metric value")
    fig.update_yaxes(title_text="Density")

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_perm_histogram")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save permutation histogram: {e}")

    return fig


# =============================================================================
# FEATURE IMPORTANCE TRUE vs PERMUTED  — grouped bar comparison
# =============================================================================

def plot_feature_importances_true_vs_perm(
    true_feature_names: list,
    true_mean: np.ndarray,
    true_std: np.ndarray,
    perm_feature_names: list,
    perm_mean: np.ndarray,
    perm_std: np.ndarray,
    save_path: str,
    filename_base: str,
    top_n: int = 20,
) -> go.Figure:
    """
    Grouped horizontal bar chart comparing feature importances from true runs vs
    permutation runs.

    Features are sorted by true importance. The top *top_n* features are shown.
    True importance bars are drawn in the project pink; permuted bars in gray.
    Error bars represent the standard deviation across runs.

    Parameters
    ----------
    true_feature_names : list[str]
        Feature name list from true runs.
    true_mean : np.ndarray
        Per-feature mean importance across true runs.
    true_std : np.ndarray
        Per-feature std of importance across true runs.
    perm_feature_names : list[str]
        Feature name list from perm runs (must match order of true_feature_names).
    perm_mean : np.ndarray
        Per-feature mean importance across perm runs.
    perm_std : np.ndarray
        Per-feature std of importance across perm runs.
    save_path : str
        Directory where output files are written.
    filename_base : str
        Prefix for output file names.
    top_n : int
        Number of top features to display.

    Returns
    -------
    go.Figure
    """
    if not true_feature_names or len(true_mean) == 0:
        print("    ! No feature importances to compare — skipping")
        return None

    TRUE_COLOR = "#DE237B"
    PERM_COLOR = "#BBBBBB"
    PERM_LINE = "#888888"

    # Select top_n by true importance
    top_idx = np.argsort(true_mean)[::-1][:top_n]
    # Reverse so highest feature is at top in horizontal bar
    top_idx = top_idx[::-1]

    y_labels = [true_feature_names[i] for i in top_idx]
    t_mean = true_mean[top_idx]
    t_std = true_std[top_idx]

    # Align perm values to true feature ordering
    if perm_feature_names and len(perm_mean) > 0:
        perm_name_to_idx = {n: j for j, n in enumerate(perm_feature_names)}
        p_mean_aligned = np.array(
            [perm_mean[perm_name_to_idx[n]] if n in perm_name_to_idx else 0.0
             for n in [true_feature_names[i] for i in top_idx]]
        )
        p_std_aligned = np.array(
            [perm_std[perm_name_to_idx[n]] if n in perm_name_to_idx else 0.0
             for n in [true_feature_names[i] for i in top_idx]]
        )
        has_perm = True
    else:
        p_mean_aligned = np.zeros_like(t_mean)
        p_std_aligned = np.zeros_like(t_std)
        has_perm = False

    fig = go.Figure()

    # Permuted bars (behind)
    if has_perm:
        fig.add_trace(
            go.Bar(
                y=y_labels,
                x=p_mean_aligned,
                orientation="h",
                name="Permuted",
                marker=dict(color=PERM_COLOR, line=dict(color=PERM_LINE, width=0.8)),
                error_x=dict(
                    type="data",
                    array=p_std_aligned,
                    color=PERM_LINE,
                    thickness=1.5,
                    width=4,
                ),
                opacity=0.85,
            )
        )

    # True bars (foreground)
    fig.add_trace(
        go.Bar(
            y=y_labels,
            x=t_mean,
            orientation="h",
            name="True",
            marker=dict(color=TRUE_COLOR, line=dict(color=TRUE_COLOR, width=0.8)),
            error_x=dict(
                type="data",
                array=t_std,
                color=TRUE_COLOR,
                thickness=1.5,
                width=4,
            ),
            opacity=0.85,
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>Feature Importances — True vs Permuted (Top {len(y_labels)})</b>",
            font=dict(size=15),
        ),
        barmode="group",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(450, len(y_labels) * 35),
        width=820,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        ),
        yaxis=dict(tickfont=dict(size=11)),
        margin=dict(l=170, r=30, t=70, b=60),
    )

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_feature_importances_true_vs_perm")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save feature importance comparison plot: {e}")

    # Save CSV with comparison data
    pd.DataFrame(
        {
            "feature": y_labels,
            "true_mean": t_mean,
            "true_std": t_std,
            "perm_mean": p_mean_aligned if has_perm else [np.nan] * len(y_labels),
            "perm_std": p_std_aligned if has_perm else [np.nan] * len(y_labels),
        }
    ).to_csv(f"{out_path}.csv", index=False)

    return fig


# =============================================================================
# SHAP BEESWARM  —  Official shap library (matplotlib-based)
# =============================================================================

def plot_shap_beeswarm_official(
    shap_values: np.ndarray,
    x_test: np.ndarray,
    feature_names: list,
    save_path: str,
    filename_base: str,
    max_display: int = 20,
) -> None:
    """
    Render an official SHAP beeswarm plot using the ``shap`` library.

    Uses ``shap.plots.beeswarm`` with a ``shap.Explanation`` object so that
    feature values are properly used for the colour axis. Saves the matplotlib
    figure as PNG and PDF.

    Parameters
    ----------
    shap_values : np.ndarray
        2-D array of shape (n_samples, n_features).
    x_test : np.ndarray
        2-D array of shape (n_samples, n_features) holding the original feature
        values used for colour-coding. If None or all-zero, a zero-filled array
        is used (no colour variation).
    feature_names : list[str]
        Feature names corresponding to the columns of *shap_values*.
    save_path : str
        Directory where output files are written.
    filename_base : str
        Prefix for output file names.
    max_display : int
        Maximum number of features to display (top by mean |SHAP|).
    """
    import shap
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend — required for server/cluster

    # Handle multi-class SHAP output (take first class for binary)
    sv = shap_values
    if sv.ndim == 3:
        sv = sv[..., 0]

    # Ensure x_test has correct shape; fall back to zeros for colour if absent
    if x_test is None or x_test.shape != sv.shape:
        x_test = np.zeros_like(sv)

    explanation = shap.Explanation(
        values=sv,
        data=x_test,
        feature_names=[str(f) for f in feature_names],
    )

    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_shap_beeswarm")
    plt.savefig(f"{out_path}.png", dpi=200, bbox_inches="tight")
    plt.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close()


# =============================================================================
# TRUE vs PERMUTED COMPARISON PLOTS
# =============================================================================

def _sig_stars(p: float) -> str:
    """Convert empirical p-value to asterisk notation."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _empirical_p(true_vals: np.ndarray, perm_vals: np.ndarray) -> float:
    """
    Empirical p-value: fraction of permuted values >= mean of true values.

    Parameters
    ----------
    true_vals : np.ndarray
        Metric values from true (non-shuffled) runs.
    perm_vals : np.ndarray
        Metric values from permuted runs (null distribution).

    Returns
    -------
    float
        Empirical p-value in [0, 1].
    """
    if len(true_vals) == 0 or len(perm_vals) == 0:
        return 1.0
    return float(np.mean(np.array(perm_vals) >= np.mean(true_vals)))


def _extract_per_run_metric(all_results: list, metric: str) -> np.ndarray:
    """
    Extract one metric value per run from the all_results list.

    Each element of all_results is a run-summary dict. For LOSO, the run's
    average metric is stored as ``mean_{metric}``.

    Parameters
    ----------
    all_results : list of dict
        Output from run_distribution_analysis or run_permutation_distribution_analysis.
    metric : str
        Metric key (e.g. 'auc', 'balanced_accuracy').

    Returns
    -------
    np.ndarray
        One value per run, shape (n_runs,).
    """
    col = f"mean_{metric}"
    values = []
    for r in all_results:
        v = r.get(col)
        if v is not None and np.isfinite(float(v)):
            values.append(float(v))
    return np.array(values)


def _extract_subject_metric_distributions(
    all_results: list,
    metric: str,
) -> dict:
    """
    Build per-subject metric distributions across runs.

    For each run in all_results, the per-subject metrics are stored in
    ``loso_subject_metrics`` — a list of dicts with keys like 'subject',
    'auc', 'balanced_accuracy', 'mcc', etc.

    Parameters
    ----------
    all_results : list of dict
        Run-summary dicts from analysis_utils.
    metric : str
        Metric key (without 'mean_' prefix, matches loso_subject_metrics keys).

    Returns
    -------
    dict
        {subject_id: np.ndarray of values, one per run}
    """
    subject_data: dict = {}
    for run in all_results:
        sub_metrics = run.get("loso_subject_metrics")
        if not sub_metrics:
            continue
        for entry in sub_metrics:
            sub = str(entry.get("subject", ""))
            val = entry.get(metric, entry.get(f"mean_{metric}"))
            if val is None or not np.isfinite(float(val)):
                continue
            subject_data.setdefault(sub, []).append(float(val))
    return {k: np.array(v) for k, v in subject_data.items()}


def _find_degenerate_subjects(all_results: list, metric: str) -> set:
    """
    Find subjects whose metric is constant across all runs (std ≈ 0).

    A constant value across runs indicates a degenerate classifier (e.g. always
    predicts the same class). These subjects are excluded from aggregated
    statistics and visualisations.

    Parameters
    ----------
    all_results : list of dict
        True run-summary dicts.
    metric : str
        Metric key (without 'mean_' prefix).

    Returns
    -------
    set of str
        Subject IDs whose metric has zero variance across all runs.
    """
    subject_data = _extract_subject_metric_distributions(all_results, metric)
    return {
        sub for sub, vals in subject_data.items()
        if len(vals) >= 1 and np.std(vals) < 1e-9
    }


def _extract_per_run_metric_filtered(
    all_results: list,
    metric: str,
    exclude_subjects: set,
) -> np.ndarray:
    """
    Compute one per-run mean metric value, excluding specified subjects.

    Falls back to the stored global mean when a run has no subject-level data.

    Parameters
    ----------
    all_results : list of dict
        Run-summary dicts.
    metric : str
        Metric key (without 'mean_' prefix).
    exclude_subjects : set of str
        Subject IDs to skip when computing each run's mean.

    Returns
    -------
    np.ndarray
        One mean value per run, shape (n_runs,).
    """
    if not exclude_subjects:
        return _extract_per_run_metric(all_results, metric)
    values = []
    for run in all_results:
        sub_metrics = run.get("loso_subject_metrics") or []
        run_vals = []
        for entry in sub_metrics:
            sub = str(entry.get("subject", ""))
            if sub in exclude_subjects:
                continue
            val = entry.get(metric, entry.get(f"mean_{metric}"))
            if val is not None and np.isfinite(float(val)):
                run_vals.append(float(val))
        if run_vals:
            values.append(float(np.mean(run_vals)))
        else:
            col = f"mean_{metric}"
            v = run.get(col)
            if v is not None and np.isfinite(float(v)):
                values.append(float(v))
    return np.array(values)


def _extract_roc_data(all_results: list) -> tuple:
    """
    Collect all per-fold TPR/FPR arrays from across all runs.

    Returns
    -------
    tuple
        (all_fprs, all_tprs) — lists of arrays, one per fold × run.
    """
    all_fprs, all_tprs = [], []
    for r in all_results:
        for fpr, tpr in zip(r.get("fold_fprs", []), r.get("fold_tprs", [])):
            if len(fpr) > 1:
                all_fprs.append(np.asarray(fpr))
                all_tprs.append(np.asarray(tpr))
    return all_fprs, all_tprs


def _extract_confusion_matrices(all_results: list) -> list:
    """Collect all per-fold confusion matrices across all runs."""
    cms = []
    for r in all_results:
        for cm in r.get("fold_cms", []):
            if cm is not None:
                cms.append(np.asarray(cm))
    return cms


def _extract_feature_importances_matrix(all_results: list) -> np.ndarray:
    """
    Stack feature importances: shape (n_runs, n_features).

    Returns empty array if none are found.
    """
    rows = []
    for r in all_results:
        fi = r.get("feature_importances")
        if fi is not None and len(fi) > 0:
            rows.append(np.asarray(fi, dtype=float))
    if not rows:
        return np.empty((0,))
    return np.vstack(rows)


def _save_fig_formats(fig, out_path: str) -> None:
    """Save a matplotlib figure to PNG and PDF."""
    fig.savefig(f"{out_path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out_path}.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1) GLOBAL DISTRIBUTION COMPARISON — histogram per metric
# -----------------------------------------------------------------------------

def plot_global_distribution_comparison(
    true_all_results: list,
    perm_all_results: list,
    save_path: str,
    filename_base: str,
    metrics: list = None,
    chance_values: dict = None,
) -> None:
    """
    Stacked ridgeline KDE comparison: true vs permuted for each metric.

    One row per metric. True distribution is filled deep pink; permuted is
    filled grey. A vertical dashed line marks the chance level. The true
    median is marked with a solid line and the empirical p-value is annotated
    on the right of each row.

    Parameters
    ----------
    true_all_results, perm_all_results : list of dict
        Run-summary dicts returned by the analysis functions.
    save_path : str
        Output directory.
    filename_base : str
        File name prefix.
    metrics : list of str, optional
        Metric keys to plot. Defaults to auc, balanced_accuracy, mcc, auprc.
    chance_values : dict, optional
        Mapping metric → chance level. Defaults to auc=0.5, balanced_accuracy=0.5, mcc=0.0.
    """
    if metrics is None:
        metrics = ["auc", "balanced_accuracy", "mcc", "auprc"]
    if chance_values is None:
        chance_values = {"auc": 0.5, "balanced_accuracy": 0.5, "mcc": 0.0, "auprc": None}

    from scipy.stats import gaussian_kde

    COLOR_TRUE = "#DE237B"
    COLOR_PERM = "#AAAAAA"
    ALPHA_TRUE = 0.72
    ALPHA_PERM = 0.48

    label_map = {
        "auc": "AUC",
        "balanced_accuracy": "Balanced Accuracy",
        "mcc": "MCC",
        "auprc": "AUPRC",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
    }

    # Collect metrics that have at least some data.
    # Degenerate subjects (constant metric across all runs) are excluded from
    # per-run means so they do not pull the global distribution toward chance.
    valid = []
    for m in metrics:
        degenerate_subs = _find_degenerate_subjects(true_all_results, m)
        tv = _extract_per_run_metric_filtered(true_all_results, m, degenerate_subs)
        pv = _extract_per_run_metric_filtered(perm_all_results, m, degenerate_subs)
        if len(tv) > 0 or len(pv) > 0:
            valid.append((m, tv, pv))
    if not valid:
        return

    n = len(valid)
    row_h = 2.8
    fig, axes = plt.subplots(n, 1, figsize=(7, row_h * n), sharey=False,
                              facecolor='white')
    if n == 1:
        axes = [axes]

    legend_added = False

    for ax, (metric, true_vals, perm_vals) in zip(axes, valid):
        ml = label_map.get(metric, metric.upper())

        all_vals = np.concatenate([v for v in [true_vals, perm_vals] if len(v) > 0])
        if len(all_vals) == 0:
            ax.set_visible(False)
            continue

        spread = all_vals.max() - all_vals.min()
        pad = max(0.04, 0.06 * spread)
        x_min = float(all_vals.min()) - pad
        x_max = float(all_vals.max()) + pad
        x_grid = np.linspace(x_min, x_max, 300)

        # Permuted KDE (drawn first so true overlaps it)
        if len(perm_vals) >= 3:
            kde_p = gaussian_kde(perm_vals, bw_method="scott")
            y_p = kde_p(x_grid)
            ax.fill_between(
                x_grid, y_p, alpha=ALPHA_PERM, color=COLOR_PERM,
                linewidth=0,
                label="Shuffled Distribution" if not legend_added else "_nolegend_",
            )

        # True KDE
        if len(true_vals) >= 3:
            kde_t = gaussian_kde(true_vals, bw_method="scott")
            y_t = kde_t(x_grid)
            ax.fill_between(
                x_grid, y_t, alpha=ALPHA_TRUE, color=COLOR_TRUE,
                linewidth=0,
                label="True Distribution" if not legend_added else "_nolegend_",
            )
            legend_added = True

        # Chance dashed vertical line
        chance = chance_values.get(metric)
        if chance is not None:
            ax.axvline(chance, color="#333333", ls="--", lw=1.2, alpha=0.7)
            ax.text(
                chance, 0, "Chance", ha="center", va="bottom",
                fontsize=7, color="#555555",
                transform=ax.get_xaxis_transform(),
            )

        # True median + empirical p-value annotation
        if len(true_vals) > 0:
            med = float(np.median(true_vals))
            p_emp = _empirical_p(true_vals, perm_vals) if len(perm_vals) > 0 else None
            ax.axvline(med, color=COLOR_TRUE, ls="-", lw=2.5, alpha=0.9)
            # Numerical median label on the vertical line
            ax.text(
                med, 0.97, f"{med:.3f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=8, color=COLOR_TRUE, fontweight="bold",
            )
            if p_emp is not None:
                stars  = _sig_stars(p_emp)
                p_str  = f"p={p_emp:.3f}" if p_emp >= 0.001 else "p<0.001"
                label_str = f"  {p_str}" if not stars else f"  {stars}  {p_str}"
                ax.text(
                    0.98, 0.88, label_str,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=COLOR_TRUE, fontweight="bold",
                )

        # Metric label as rotated y-axis title
        ax.set_facecolor('white')
        ax.set_ylabel(ml, fontsize=11, fontweight="bold", rotation=0,
                      ha="right", va="center", labelpad=10)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    axes[-1].set_xlabel("Performance", fontsize=11)

    # Legend attached to first subplot
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_TRUE, alpha=ALPHA_TRUE, label="True Distribution"),
        Patch(facecolor=COLOR_PERM, alpha=ALPHA_PERM, label="Shuffled Distribution"),
    ]
    axes[0].legend(handles=handles, fontsize=9, loc="upper left",
                   frameon=False, bbox_to_anchor=(0.0, 1.15))

    fig.tight_layout(h_pad=1.5)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_global_distributions")
    _save_fig_formats(fig, out_path)

    # Save underlying data
    rows = []
    for m, tv, pv in valid:
        for val in tv:
            rows.append({"metric": m, "type": "true", "value": val})
        for val in pv:
            rows.append({"metric": m, "type": "permuted", "value": val})
    pd.DataFrame(rows).to_csv(f"{out_path}_data.csv", index=False)
    print(f"  ✓ Global distribution comparison → {out_path}.png")


# -----------------------------------------------------------------------------
# 2) SUBJECT-LEVEL VIOLIN COMPARISON
# -----------------------------------------------------------------------------

def plot_subject_violin_comparison(
    true_all_results: list,
    perm_all_results: list,
    save_path: str,
    filename_base: str,
    metrics: list = None,
    min_perm_values: int = 5,
) -> None:
    """
    Per-subject violin comparing true vs permuted distributions across runs.

    For each metric produces a figure with:
    - Horizontal violins per subject (true = pink, permuted = grey)
    - Significance stars based on empirical p-value (fraction of perm >= mean(true))
    - Donut chart showing % of subjects with p < 0.05
    - CSV with per-subject p-values

    Parameters
    ----------
    true_all_results, perm_all_results : list of dict
        Run-summary dicts.
    save_path : str
        Output directory.
    filename_base : str
        File name prefix.
    metrics : list of str, optional
        Metrics to plot. Defaults to ['auc', 'balanced_accuracy', 'mcc', 'auprc'].
    min_perm_values : int
        Minimum permuted values required to test a subject.
    """
    if metrics is None:
        metrics = ["auc", "balanced_accuracy", "mcc", "auprc"]

    COLOR_TRUE = "#DE237B"
    COLOR_PERM = "#AAAAAA"
    CHANCE = {"auc": 0.5, "balanced_accuracy": 0.5, "mcc": 0.0, "auprc": None}

    os.makedirs(save_path, exist_ok=True)

    for metric in metrics:
        true_sub = _extract_subject_metric_distributions(true_all_results, metric)
        perm_sub = _extract_subject_metric_distributions(perm_all_results, metric)

        # Keep subjects present in BOTH with non-degenerate true distributions.
        # Subjects with std=0 across all runs produced a constant (degenerate)
        # classifier and are excluded from plots and aggregated statistics.
        subjects = sorted(
            [s for s in (set(true_sub.keys()) & set(perm_sub.keys()))
             if np.std(true_sub[s]) > 1e-9],
            key=lambda x: int(x) if str(x).isdigit() else x,
        )
        if not subjects:
            continue

        # Per-subject empirical p-values
        p_vals = {}
        for sub in subjects:
            tv = true_sub[sub]
            pv = perm_sub.get(sub, np.array([]))
            if len(tv) > 0 and len(pv) >= min_perm_values:
                p_vals[sub] = _empirical_p(tv, pv)
            else:
                p_vals[sub] = 1.0

        n_sig = sum(1 for p in p_vals.values() if p < 0.05)
        pct_sig = 100 * n_sig / len(subjects) if subjects else 0

        # Figure layout: left panel = violins, right panel = donut
        fig = plt.figure(figsize=(10, max(6, 0.55 * len(subjects) + 2)),
                         facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.2], wspace=0.25)
        ax_v = fig.add_subplot(gs[0])
        ax_d = fig.add_subplot(gs[1])

        label_map = {
            "auc": "AUC", "balanced_accuracy": "Balanced Accuracy",
            "mcc": "MCC", "auprc": "AUPRC", "f1": "F1",
        }
        metric_label = label_map.get(metric, metric.upper())

        # Draw half-violin density plots per subject.
        # Both true (pink) and permuted (grey) fill above the centre line (same side).
        from scipy.stats import gaussian_kde

        positions = list(range(len(subjects)))
        HALF_HEIGHT = 0.38   # maximum half-violin height in axis units
        N_GRID = 200

        # Global x-range for consistent KDE grids
        all_metric_vals = np.concatenate(
            [v for sub in subjects
             for v in [true_sub.get(sub, np.array([])),
                       perm_sub.get(sub, np.array([]))]]
        )
        if len(all_metric_vals) == 0:
            continue
        x_pad = max(0.03, 0.05 * (all_metric_vals.max() - all_metric_vals.min()))
        x_min_g = float(all_metric_vals.min()) - x_pad
        x_max_g = float(all_metric_vals.max()) + x_pad
        x_grid = np.linspace(x_min_g, x_max_g, N_GRID)

        x_star_max = x_max_g  # track right edge for star placement

        for pos, sub in zip(positions, subjects):
            tv = true_sub[sub]
            pv = perm_sub.get(sub, np.array([]))

            for vals, color, sign in [(pv, COLOR_PERM, +1), (tv, COLOR_TRUE, +1)]:
                if len(vals) < 2 or np.std(vals) < 1e-9:
                    if len(vals) >= 1:
                        ax_v.scatter(float(np.mean(vals)), pos + sign * 0.12,
                                     color=color, s=25, zorder=5, alpha=0.85)
                    continue
                kde = gaussian_kde(vals, bw_method="scott")
                density = kde(x_grid)
                density_scaled = density / density.max() * HALF_HEIGHT
                # Half-violin: fill between center y=pos and y=pos ± density
                y_outer = pos + sign * density_scaled
                ax_v.fill_between(x_grid, pos, y_outer,
                                  color=color, alpha=0.80, linewidth=0)
                # Median tick (vertical line from center to density boundary)
                med_x = float(np.median(vals))
                med_y = pos + sign * float(kde([med_x])[0] / density.max() * HALF_HEIGHT)
                ax_v.plot([med_x, med_x], [pos, med_y],
                          color=color, lw=2.0, solid_capstyle="round", zorder=4)

            # Connect true and permuted medians with a thin line
            if len(tv) > 0 and len(pv) > 0:
                ax_v.plot(
                    [float(np.median(tv)), float(np.median(pv))],
                    [pos, pos],
                    color="#555555", lw=0.8, alpha=0.35, zorder=2,
                )

            # Significance star to the right
            p = p_vals.get(sub, 1.0)
            star = _sig_stars(p)
            if star:
                ax_v.text(
                    x_star_max + 0.01, pos, star,
                    va="center", ha="left", fontsize=13,
                    color=COLOR_TRUE, fontweight="bold",
                )

        # Chance line
        chance = CHANCE.get(metric)
        if chance is not None:
            ax_v.axvline(chance, color="black", ls="--", lw=1.2, alpha=0.5, label="Chance")

        ax_v.set_facecolor('white')
        ax_v.set_yticks(positions)
        ax_v.set_yticklabels([f"Sub-{s}" for s in subjects], fontsize=9)
        ax_v.set_xlabel(metric_label, fontsize=11)
        ax_v.set_title(f"Subject-Level True vs Permuted\n{metric_label}", fontsize=12, fontweight="bold")
        ax_v.spines[["top", "right"]].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLOR_TRUE, label="True"),
            Patch(facecolor=COLOR_PERM, label="Permuted"),
        ]
        ax_v.legend(handles=legend_handles, fontsize=9, loc="lower right")

        # Donut chart
        ax_d.axis("equal")
        wedge_sizes = [pct_sig, 100 - pct_sig]
        wedge_colors = [COLOR_TRUE, "#E0E0E0"]
        wedges, _ = ax_d.pie(
            wedge_sizes, colors=wedge_colors,
            startangle=90, wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        )
        ax_d.text(0, 0, f"{pct_sig:.0f}%", ha="center", va="center",
                  fontsize=16, fontweight="bold", color=COLOR_TRUE)
        ax_d.set_title(f"Sig. subjects\n(p<0.05)", fontsize=10)

        fig.suptitle(
            f"True vs Permuted — {metric_label}  |  n={len(subjects)} subjects",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()

        out_path = os.path.join(save_path, f"{filename_base}_subject_ridgelines_{metric}")
        _save_fig_formats(fig, out_path)

        # Save p-value CSV
        p_df = pd.DataFrame([
            {
                "subject": sub,
                "true_mean": float(np.mean(true_sub[sub])) if len(true_sub[sub]) else np.nan,
                "true_median": float(np.median(true_sub[sub])) if len(true_sub[sub]) else np.nan,
                "perm_mean": float(np.mean(perm_sub.get(sub, np.array([])))) if sub in perm_sub else np.nan,
                "perm_median": float(np.median(perm_sub.get(sub, np.array([])))) if sub in perm_sub else np.nan,
                "empirical_p": p_vals.get(sub, 1.0),
                "significant": int(p_vals.get(sub, 1.0) < 0.05),
                "stars": _sig_stars(p_vals.get(sub, 1.0)),
            }
            for sub in subjects
        ])
        p_df.to_csv(f"{out_path}_pvalues.csv", index=False)
        print(f"  ✓ Subject violin comparison ({metric}) → {out_path}.png")


# -----------------------------------------------------------------------------
# 3) ROC CURVE COMPARISON
# -----------------------------------------------------------------------------

def plot_roc_comparison(
    true_all_results: list,
    perm_all_results: list,
    save_path: str,
    filename_base: str,
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
) -> None:
    """
    Two-curve ROC plot: true mean ± CI in color, permuted mean ± CI in grey.

    Each fold from each run contributes one ROC curve. The mean is computed
    by interpolating all curves onto a common FPR grid.

    Parameters
    ----------
    true_all_results, perm_all_results : list of dict
        Run-summary dicts.
    save_path, filename_base : str
        Output path and file prefix.
    positive_class_name, negative_class_name : str
        Human-readable class labels for the plot title.
    """
    COLOR_TRUE = "#DE237B"
    COLOR_PERM = "#888888"
    FPR_GRID = np.linspace(0, 1, 200)

    def _interpolate_rocs(all_results):
        """Interpolate all fold ROC curves onto the common FPR grid."""
        fprs, tprs = _extract_roc_data(all_results)
        interp_tprs, aucs = [], []
        for fpr, tpr in zip(fprs, tprs):
            interp = np.interp(FPR_GRID, fpr, tpr)
            interp[0] = 0.0
            interp_tprs.append(interp)
            aucs.append(float(np.trapz(interp, FPR_GRID)))
        return interp_tprs, aucs

    true_tprs, true_aucs = _interpolate_rocs(true_all_results)
    perm_tprs, perm_aucs = _interpolate_rocs(perm_all_results)

    if not true_tprs:
        print("  ROC comparison: no true ROC data available — skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    ax.set_facecolor('white')

    # Permuted band (drawn first, behind)
    if perm_tprs:
        pm = np.mean(perm_tprs, axis=0)
        ps = np.std(perm_tprs, axis=0)
        ax.fill_between(FPR_GRID, pm - ps, pm + ps, alpha=0.15, color=COLOR_PERM)
        ax.plot(FPR_GRID, pm, color=COLOR_PERM, lw=1.5, ls="--",
                label=f"Permuted (AUC={np.mean(perm_aucs):.3f}±{np.std(perm_aucs):.3f})")

    # True band
    tm = np.mean(true_tprs, axis=0)
    ts = np.std(true_tprs, axis=0)
    tm[-1] = 1.0
    ax.fill_between(FPR_GRID, tm - ts, tm + ts, alpha=0.18, color=COLOR_TRUE)
    ax.plot(FPR_GRID, tm, color=COLOR_TRUE, lw=2.5,
            label=f"True (AUC={np.mean(true_aucs):.3f}±{np.std(true_aucs):.3f})")

    # Empirical p-value at run level
    true_run_aucs = _extract_per_run_metric(true_all_results, "auc")
    perm_run_aucs = _extract_per_run_metric(perm_all_results, "auc")
    p_emp = _empirical_p(true_run_aucs, perm_run_aucs) if len(perm_run_aucs) > 0 else None

    # Chance diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    title_p = f"  (p_emp={p_emp:.4f}{' ' + _sig_stars(p_emp) if _sig_stars(p_emp) else ''})" if p_emp is not None else ""
    ax.set_title(f"ROC Curve — True vs Permuted{title_p}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_roc_curve")
    _save_fig_formats(fig, out_path)

    # Save data
    roc_df = pd.DataFrame({
        "fpr": FPR_GRID,
        "true_mean_tpr": tm,
        "true_std_tpr": ts,
    })
    if perm_tprs:
        roc_df["perm_mean_tpr"] = pm
        roc_df["perm_std_tpr"] = ps
    roc_df.to_csv(f"{out_path}_data.csv", index=False)
    print(f"  ✓ ROC comparison → {out_path}.png")


# -----------------------------------------------------------------------------
# 4) CONFUSION MATRIX COMPARISON (2-panel)
# -----------------------------------------------------------------------------

def plot_confusion_matrix_comparison(
    true_all_results: list,
    perm_all_results: list,
    save_path: str,
    filename_base: str,
    negative_class_name: str = "OFF-task",
    positive_class_name: str = "ON-task",
) -> None:
    """
    Side-by-side normalized confusion matrices: true (left) vs permuted (right).

    Both matrices are averaged across all folds × runs and displayed on the
    same color scale for direct comparison.

    Parameters
    ----------
    true_all_results, perm_all_results : list of dict
        Run-summary dicts.
    save_path, filename_base : str
        Output path and file prefix.
    negative_class_name, positive_class_name : str
        Human-readable class labels.
    """
    true_cms = _extract_confusion_matrices(true_all_results)
    perm_cms = _extract_confusion_matrices(perm_all_results)

    if not true_cms:
        print("  CM comparison: no confusion matrices available — skipping")
        return

    def _normalize_cm(cms_list):
        avg = np.mean(cms_list, axis=0)
        row_sums = avg.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return avg / row_sums, avg

    true_norm, true_avg = _normalize_cm(true_cms)
    perm_has_data = bool(perm_cms)
    if perm_has_data:
        perm_norm, perm_avg = _normalize_cm(perm_cms)

    labels = [negative_class_name, positive_class_name]
    vmax = 1.0
    cmap = "RdPu"

    n_panels = 2 if perm_has_data else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                              facecolor='white')
    if n_panels == 1:
        axes = [axes]

    def _draw_cm(ax, norm_cm, avg_cm, title):
        im = ax.imshow(norm_cm, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
        for i in range(2):
            for j in range(2):
                color = "white" if norm_cm[i, j] > 0.6 else "black"
                ax.text(
                    j, i,
                    f"{norm_cm[i, j]:.2f}\n(n={avg_cm[i, j]:.1f})",
                    ha="center", va="center", fontsize=11,
                    color=color, fontweight="bold",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"Pred\n{l}" for l in labels], fontsize=9)
        ax.set_yticklabels([f"True\n{l}" for l in labels], fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw_cm(axes[0], true_norm, true_avg, "True Labels")
    if perm_has_data:
        _draw_cm(axes[1], perm_norm, perm_avg, "Permuted Labels")

    fig.suptitle("Confusion Matrix — True vs Permuted", fontsize=13, fontweight="bold")
    fig.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{filename_base}_confusion_matrix")
    _save_fig_formats(fig, out_path)

    # Save CSVs
    pd.DataFrame(true_norm, index=labels, columns=labels).to_csv(
        f"{out_path}_true_normalized.csv"
    )
    if perm_has_data:
        pd.DataFrame(perm_norm, index=labels, columns=labels).to_csv(
            f"{out_path}_perm_normalized.csv"
        )
    print(f"  ✓ Confusion matrix comparison → {out_path}.png")


# -----------------------------------------------------------------------------
# 5) FEATURE IMPORTANCE + SHAP COMPARISON
# -----------------------------------------------------------------------------

def plot_feature_importance_comparison(
    true_all_results: list,
    perm_all_results: list,
    feature_names,
    save_path: str,
    filename_base: str,
    top_n: int = 15,
    true_shap_runs: list = None,
    perm_shap_runs: list = None,
) -> None:
    """
    Paired horizontal bar chart: true vs permuted feature importances.

    For each of the top-N features (ranked by true mean importance) shows:
    - True importance (bright pink bar)
    - Permuted importance of the SAME feature (grey, muted bar)
    - Rank of that feature in the permuted distribution (annotated)

    If SHAP values are provided, an identical second panel is generated for SHAP.

    Also produces a rank-stability CSV comparing true ranks vs permuted ranks.

    Parameters
    ----------
    true_all_results, perm_all_results : list of dict
        Run-summary dicts.
    feature_names : array-like
        Feature name list.
    save_path, filename_base : str
        Output path and prefix.
    top_n : int
        Number of top features to display.
    true_shap_runs, perm_shap_runs : list of np.ndarray, optional
        Per-run stacked SHAP matrices (n_runs, n_samples, n_features).
        If provided, a second panel with SHAP values is generated.
    """
    feature_names = list(feature_names)
    COLOR_TRUE = "#DE237B"
    COLOR_PERM = "#AAAAAA"
    COLOR_PERM_RANK = "#D3E0A3"  # light greenish for same-rank permuted feature

    true_fi = _extract_feature_importances_matrix(true_all_results)
    perm_fi = _extract_feature_importances_matrix(perm_all_results)

    # Only use coefficient-based FI when at least some values are non-zero.
    has_fi = (
        true_fi.ndim == 2
        and true_fi.shape[0] > 0
        and np.any(true_fi != 0)
    )
    has_shap = (
        true_shap_runs is not None
        and perm_shap_runs is not None
        and len(true_shap_runs) > 0
        and len(perm_shap_runs) > 0
    )

    def _build_importance_arrays(fi_matrix):
        """Return (mean, std) per feature."""
        if fi_matrix.ndim < 2 or fi_matrix.shape[0] == 0:
            return np.zeros(len(feature_names)), np.zeros(len(feature_names))
        return fi_matrix.mean(axis=0), fi_matrix.std(axis=0)

    def _build_shap_arrays(shap_runs):
        """Mean |SHAP| per feature across all runs."""
        per_run = np.array([np.mean(np.abs(r), axis=0) for r in shap_runs
                            if r is not None and hasattr(r, "shape") and r.ndim == 2])
        if per_run.shape[0] == 0:
            return np.zeros(len(feature_names)), np.zeros(len(feature_names))
        return per_run.mean(axis=0), per_run.std(axis=0)

    os.makedirs(save_path, exist_ok=True)

    for mode, true_means, true_stds, perm_means, perm_stds in [
        *([("fi",
            *(_build_importance_arrays(true_fi)),
            *(_build_importance_arrays(perm_fi)))]
           if has_fi else []),
        *([("shap",
            *(_build_shap_arrays(true_shap_runs)),
            *(_build_shap_arrays(perm_shap_runs)))]
           if has_shap else []),
    ]:
        # Top N features by true mean importance
        top_idx = np.argsort(true_means)[::-1][:top_n]
        # Reverse so highest is at top of horizontal bar chart
        top_idx_plot = top_idx[::-1]

        feat_labels = [feature_names[i] for i in top_idx_plot]
        t_mean = true_means[top_idx_plot]
        t_std = true_stds[top_idx_plot]
        p_mean = perm_means[top_idx_plot]
        p_std = perm_stds[top_idx_plot]

        # Permuted same-rank values: for each true rank, what was the perm value at that rank?
        # In SHAP mode: use per-run mean |SHAP| per feature; in FI mode: use FI matrix.
        perm_rank_means = np.full(len(top_idx_plot), np.nan)
        perm_rank_stds = np.full(len(top_idx_plot), np.nan)
        if mode == "shap" and perm_shap_runs is not None and len(perm_shap_runs) > 0:
            perm_rank_per_run = []
            for r in perm_shap_runs:
                if r is not None and hasattr(r, "shape") and r.ndim == 2:
                    run_mean_abs = np.mean(np.abs(r), axis=0)
                    ranked_vals = np.sort(run_mean_abs)[::-1]
                    perm_rank_per_run.append(ranked_vals[:top_n])
            if perm_rank_per_run:
                perm_rank_matrix = np.vstack(perm_rank_per_run)
                for plot_pos, orig_rank in enumerate(range(len(top_idx_plot) - 1, -1, -1)):
                    perm_rank_means[plot_pos] = perm_rank_matrix[:, orig_rank].mean()
                    perm_rank_stds[plot_pos] = perm_rank_matrix[:, orig_rank].std()
        elif perm_fi.ndim == 2 and perm_fi.shape[0] > 0 and np.any(perm_fi != 0):
            perm_rank_per_run = []
            for run_fi in perm_fi:
                ranked_vals = np.sort(run_fi)[::-1]
                perm_rank_per_run.append(ranked_vals[:top_n])
            perm_rank_matrix = np.vstack(perm_rank_per_run)
            for plot_pos, orig_rank in enumerate(range(len(top_idx_plot) - 1, -1, -1)):
                perm_rank_means[plot_pos] = perm_rank_matrix[:, orig_rank].mean()
                perm_rank_stds[plot_pos] = perm_rank_matrix[:, orig_rank].std()

        y_pos = np.arange(len(feat_labels))
        bar_h = 0.26

        fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(feat_labels) + 2)))

        # True bars
        ax.barh(y_pos + bar_h, t_mean, xerr=t_std, height=bar_h,
                color=COLOR_TRUE, alpha=0.9, label="True", error_kw=dict(ecolor="#AA1060", lw=1))
        # Permuted same-feature bars
        ax.barh(y_pos, p_mean, xerr=p_std, height=bar_h,
                color=COLOR_PERM, alpha=0.75, label="Permuted (same feature)",
                error_kw=dict(ecolor="#666666", lw=1))
        # Permuted same-rank bars
        if not np.all(np.isnan(perm_rank_means)):
            ax.barh(y_pos - bar_h, perm_rank_means, xerr=perm_rank_stds, height=bar_h,
                    color=COLOR_PERM_RANK, alpha=0.75, label="Permuted (same rank)",
                    error_kw=dict(ecolor="#888888", lw=1))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_labels, fontsize=9)
        xlabel = "Mean |SHAP value|" if mode == "shap" else "Mean Importance"
        ax.set_xlabel(xlabel, fontsize=11)
        title_mode = "SHAP" if mode == "shap" else "Feature Importance"
        ax.set_title(f"Top {len(feat_labels)} Features — {title_mode}\nTrue vs Permuted (same feature & same rank)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        suffix = "shap_importance" if mode == "shap" else "feature_importance"
        out_path = os.path.join(save_path, f"{filename_base}_{suffix}")
        _save_fig_formats(fig, out_path)

        # Rank stability CSV
        rank_df = pd.DataFrame({
            "feature": feat_labels[::-1],  # re-reverse to descending rank order
            "true_rank": list(range(1, len(feat_labels) + 1)),
            "true_mean": t_mean[::-1],
            "true_std": t_std[::-1],
            "perm_mean_same_feature": p_mean[::-1],
            "perm_std_same_feature": p_std[::-1],
            "perm_mean_same_rank": perm_rank_means[::-1],
            "perm_std_same_rank": perm_rank_stds[::-1],
        })
        rank_df.to_csv(f"{out_path}_data.csv", index=False)
        print(f"  ✓ {title_mode} comparison → {out_path}.png")


# -----------------------------------------------------------------------------
# ORCHESTRATOR — call after permutation analysis completes
# -----------------------------------------------------------------------------

def generate_all_comparison_plots(
    true_all_results: list,
    perm_all_results: list,
    feature_names,
    save_path: str,
    filename_base: str,
    dimension: str = "",
    positive_class_name: str = "ON-task",
    negative_class_name: str = "OFF-task",
    metrics: list = None,
    top_n_features: int = 15,
    true_shap_runs: list = None,
    perm_shap_runs: list = None,
) -> None:
    """
    Generate the complete suite of true-vs-permuted comparison plots.

    This function orchestrates all comparison plots after both the true
    classification runs and permutation runs have completed. It reads
    exclusively from the in-memory all_results lists (no disk re-reading).

    Plots generated
    ---------------
    1. Global metric distribution histograms with empirical p-value
    2. Per-subject violin plots with significance stars and donut chart
    3. ROC curve comparison (true vs permuted)
    4. 2-panel confusion matrix comparison
    5. Feature importance comparison (true vs permuted, same feature & same rank)
    6. SHAP comparison (if SHAP values provided)

    Parameters
    ----------
    true_all_results : list of dict
        Run-summary dicts from ``run_distribution_analysis``.
    perm_all_results : list of dict
        Run-summary dicts from ``run_permutation_distribution_analysis``.
    feature_names : array-like
        Feature name list.
    save_path : str
        Root directory where plots are saved.
    filename_base : str
        File name prefix (e.g., 'lr_loso_20runs').
    dimension : str
        Contrast name, for plot titles.
    positive_class_name, negative_class_name : str
        Human-readable class labels.
    metrics : list of str, optional
        Metrics to plot. Defaults to auc, balanced_accuracy, mcc, auprc.
    top_n_features : int
        Number of top features for the importance comparison.
    true_shap_runs, perm_shap_runs : list of np.ndarray, optional
        Stacked SHAP values. If provided, SHAP comparison is generated.
    """
    if metrics is None:
        metrics = ["auc", "balanced_accuracy", "mcc", "auprc"]

    print(f"\n{'='*60}")
    print(f"Generating True vs Permuted comparison plots...")
    print(f"  Save path: {save_path}")
    print(f"{'='*60}")

    os.makedirs(save_path, exist_ok=True)

    plot_global_distribution_comparison(
        true_all_results, perm_all_results,
        save_path, filename_base, metrics=metrics,
    )

    plot_subject_violin_comparison(
        true_all_results, perm_all_results,
        save_path, filename_base, metrics=metrics,
    )

    plot_roc_comparison(
        true_all_results, perm_all_results,
        save_path, filename_base,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
    )

    plot_confusion_matrix_comparison(
        true_all_results, perm_all_results,
        save_path, filename_base,
        negative_class_name=negative_class_name,
        positive_class_name=positive_class_name,
    )

    plot_feature_importance_comparison(
        true_all_results, perm_all_results,
        feature_names, save_path, filename_base,
        top_n=top_n_features,
        true_shap_runs=true_shap_runs,
        perm_shap_runs=perm_shap_runs,
    )

    print(f"\n  All comparison plots saved to: {save_path}")
def _minmax_within_subject(df: pd.DataFrame, col: str, subject_col: str = "subject") -> pd.Series:
    """Min-max scale *col* within each subject. Subjects with zero range → NaN."""
    def _scale(g):
        mn, mx = g.min(), g.max()
        if mx == mn:
            return pd.Series(np.nan, index=g.index)
        return (g - mn) / (mx - mn)
    return df.groupby(subject_col)[col].transform(_scale)


def plot_probability_vs_raw(
    consolidated_df: pd.DataFrame,
    comparison_results_path: str,
    filename_base: str,
    proba_col: str = "proba_mean",
    normalize_within_subject: "list[str] | None" = None,
):
    '''
    Plot relationship between the assigned probability and each raw dimension score.

    Iterates over every ``*_first`` column (excluding ``y_true_first``) and
    produces a general scatter and a by-subject faceted plot for each dimension.
    Files are named ``{filename_base}_prob_vs_{dim}_{general,faceted}.png``.

    Parameters
    ----------
    normalize_within_subject : list of str, optional
        Dimension names (e.g. ``["confidence"]``) whose ``*_first`` column should
        be min-max scaled within each subject before plotting.  Produces an
        additional pair of plots suffixed ``_norm``:
        ``{filename_base}_prob_vs_{dim}_norm_{general,faceted}.png``.
    '''
    plots_dir = os.path.join(comparison_results_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    raw_cols = [c for c in consolidated_df.columns if c.endswith("_first") and c != "y_true_first"]
    if not raw_cols:
        print("No raw dimension column found for probability scatter plots.")
        return

    normalize_within_subject = normalize_within_subject or []

    for raw_col in raw_cols:
        dim = raw_col.replace("_first", "")
        _plot_prob_vs_dim(consolidated_df, raw_col, dim, proba_col, plots_dir, filename_base)

        if dim in normalize_within_subject and "subject" in consolidated_df.columns:
            norm_col = f"__{dim}_norm"
            df_norm = consolidated_df.copy()
            df_norm[norm_col] = _minmax_within_subject(df_norm, raw_col)
            df_norm = df_norm.rename(columns={norm_col: f"{norm_col}_first"})
            _plot_prob_vs_dim(
                df_norm, f"{norm_col}_first", f"{dim}_norm", proba_col, plots_dir, filename_base,
                xlabel=f"{dim} (min-max within subject)",
                title_suffix=" [normalized within subject]",
            )


def _plot_prob_vs_dim(
    consolidated_df: pd.DataFrame,
    raw_col: str,
    dim: str,
    proba_col: str,
    plots_dir: str,
    filename_base: str,
    xlabel: str = None,
    title_suffix: str = "",
):
    """Render and save the general + faceted scatter for one dimension column."""
    mask = consolidated_df[raw_col].notna() & consolidated_df[proba_col].notna()
    df_clean = consolidated_df[mask].copy()

    if len(df_clean) < 2:
        print(f"Not enough data for prob_vs_{dim} plots — skipping.")
        return

    xlabel = xlabel or f'Raw Score ({dim.replace("_norm", "")})'

    # ── 1. General plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_clean[raw_col], df_clean[proba_col]
    )
    stats_text = f"$\\beta$ = {slope:.3f}\n$r$ = {r_value:.2f}\n$p$ = {p_value:.1e}"

    sns.regplot(
        data=df_clean, x=raw_col, y=proba_col,
        scatter_kws={'alpha': 0.5, 's': 20, 'color': get_comparison_color("")},
        line_kws={'color': 'black', 'lw': 2},
        ax=ax,
    )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean Predicted Probability')
    ax.set_title(f'Overall: Probability vs {dim.replace("_norm", "")}{title_suffix}')
    ax.text(
        0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{filename_base}_prob_vs_{dim}_general.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ── 2. Faceted by subject ────────────────────────────────────────────────
    if "subject" not in df_clean.columns:
        print(f"No subject column — skipping faceted plot for {dim}.")
        return

    n_subj = df_clean['subject'].nunique()
    cols = min(6, max(1, n_subj))

    g = sns.lmplot(
        data=df_clean,
        x=raw_col,
        y=proba_col,
        col="subject",
        col_wrap=cols,
        height=3.5,
        aspect=1,
        facet_kws={"sharex": True, "sharey": True},
        scatter_kws={'alpha': 0.6, 's': 15, 'color': get_comparison_color("")},
        line_kws={'color': 'black', 'lw': 1.5},
    )

    g.set(ylim=(-0.05, 1.05))

    def _annotate(data, **kws):
        ax = plt.gca()
        x = data[raw_col]
        y = data[proba_col]
        if len(x) > 1:
            s, _, r, p, _ = stats.linregress(x, y)
            ax.text(
                0.05, 0.95,
                f"$\\beta$={s:.3f}\n$r$={r:.2f}\n$p$={p:.1e}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none'),
            )

    g.map_dataframe(_annotate)
    g.set_axis_labels(xlabel, 'Pred Probability')
    g.fig.subplots_adjust(top=0.92)
    dim_label = dim.replace("_norm", "")
    g.fig.suptitle(f'By Subject: Probability vs {dim_label}{title_suffix}', fontsize=16)

    faceted_path = os.path.join(plots_dir, f"{filename_base}_prob_vs_{dim}_faceted.png")
    g.savefig(faceted_path, dpi=300, bbox_inches='tight')
    plt.close(g.fig)


# =============================================================================
# AUC VS ON/OFF SCALE DISPERSION
# =============================================================================

def plot_auc_vs_onoff_dispersion(
    subject_auc_df: pd.DataFrame,
    df_prepared: pd.DataFrame,
    results_path: str,
    filename_base: str,
    pipeline_label: str = "LOSO",
    label_col: str = "onoff",
) -> None:
    """
    Scatter plot of per-subject AUC vs per-subject SD of the target ratings.

    Relates classification performance to how broadly each subject uses
    the categorization scale. A high SD means the subject used extreme
    values; a low SD means ratings were compressed around the center.

    Parameters
    ----------
    subject_auc_df : pd.DataFrame
        Columns: 'subject', 'auc'. One row per subject, AUC averaged over runs.
    df_prepared : pd.DataFrame
        Full prepared DataFrame with 'subject' and the ``label_col`` column.
    results_path : str
        Directory where plot files are saved.
    filename_base : str
        Filename prefix (matches the other result files for this run).
    pipeline_label : str
        Pipeline name for the plot title ('LOSO' or 'WithinSubject').
    label_col : str
        Name of the continuous rating column to compute dispersion on
        (e.g. 'onoff', 'valence', 'selfother', 'time', 'confidence').
    """
    if label_col not in df_prepared.columns:
        print(f"Warning: '{label_col}' column not found in df_prepared — skipping AUC vs dispersion plot.")
        return

    label_stats = (
        df_prepared.groupby("subject")[label_col]
        .agg(label_std="std", label_mean="mean")
        .reset_index()
    )

    plot_df = pd.merge(subject_auc_df, label_stats, on="subject", how="inner")
    if plot_df.empty:
        print(f"Warning: No subjects matched between AUC data and {label_col} data — skipping plot.")
        return

    x = plot_df["label_std"].values.astype(float)
    y = plot_df["auc"].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)

    r_val, p_val = np.nan, np.nan
    slope, intercept = np.nan, np.nan
    if valid.sum() >= 3:
        r_val, p_val = stats.pearsonr(x[valid], y[valid])
        slope, intercept, *_ = stats.linregress(x[valid], y[valid])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["label_std"],
        y=plot_df["auc"],
        mode='markers+text',
        text=plot_df["subject"].astype(str),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(color=COLORS[0], size=10, opacity=0.85,
                    line=dict(color='white', width=1)),
        name='Subjects',
        hovertemplate=(
            'Subject: %{text}<br>'
            f'{label_col} SD: %{{x:.2f}}<br>'
            'AUC: %{y:.3f}<extra></extra>'
        ),
    ))

    if np.isfinite(slope):
        x_line = np.array([np.nanmin(x[valid]), np.nanmax(x[valid])])
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name=f'r = {r_val:.2f}, p = {p_val:.3f}',
        ))

    fig.add_hline(
        y=0.5, line_dash='dash', line_color='gray', opacity=0.5,
        annotation_text='Chance (0.5)', annotation_position='bottom right',
    )

    corr_label = (
        f"r = {r_val:.3f}, p = {p_val:.3f}" if np.isfinite(r_val) else "r = N/A"
    )
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=(
                f'<b>{pipeline_label}: AUC vs {label_col} Scale Dispersion</b><br>'
                f'<sup>{corr_label} | n = {valid.sum()} subjects</sup>'
            ),
            font=dict(size=16),
        ),
        xaxis_title=f'Per-subject SD of {label_col} ratings (scale dispersion)',
        yaxis_title='AUC',
        yaxis=dict(range=[max(0.0, float(np.nanmin(y)) - 0.1),
                          min(1.0, float(np.nanmax(y)) + 0.1)]),
        width=800,
        height=650,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    os.makedirs(results_path, exist_ok=True)
    out_path = os.path.join(results_path, f"{filename_base}_auc_vs_{label_col}_dispersion")
    try:
        fig.write_image(f"{out_path}.png", scale=2)
        fig.write_image(f"{out_path}.pdf")
        fig.write_html(f"{out_path}.html")
    except Exception as e:
        print(f"Warning: Could not save AUC vs dispersion plot: {e}")

    plot_df.to_csv(f"{out_path}_data.csv", index=False)
    print(f"  Saved AUC vs {label_col} dispersion → {out_path}.png")


# =============================================================================
# AUC VS CLASS IMBALANCE
# =============================================================================

def plot_auc_vs_class_imbalance(
    subject_auc_df: pd.DataFrame,
    df_prepared: pd.DataFrame,
    results_path: str,
    filename_base: str,
    pipeline_label: str = "LOSO",
    label_col: str = "onoff",
) -> None:
    """
    Scatter plot of per-subject AUC vs per-subject class imbalance.

    Class imbalance is expressed as the minority-class ratio (n_minority /
    n_total), ranging from 0 (maximally imbalanced) to 0.5 (perfectly
    balanced).  This relates classification performance to whether each
    subject's binary labels are evenly split after the median binarization.

    Parameters
    ----------
    subject_auc_df : pd.DataFrame
        Columns: 'subject', 'auc'. One row per subject, AUC averaged over runs.
    df_prepared : pd.DataFrame
        Full prepared DataFrame with 'subject' and 'target' (binary 0/1).
    results_path : str
        Directory where plot files are saved.
    filename_base : str
        Filename prefix (matches the other result files for this run).
    pipeline_label : str
        Pipeline name for the plot title ('LOSO' or 'WithinSubject').
    label_col : str
        Dimension name used only for labelling (e.g. 'onoff', 'valence').
    """
    if "target" not in df_prepared.columns:
        print("Warning: 'target' column not found in df_prepared — skipping AUC vs imbalance plot.")
        return

    def _minority_ratio(s: pd.Series) -> float:
        counts = s.value_counts()
        if len(counts) < 2:
            return 0.0
        return float(counts.min()) / float(counts.sum())

    df_work = df_prepared.copy()
    df_work["subject"] = df_work["subject"].astype(str)
    imbalance_stats = (
        df_work.groupby("subject")["target"]
        .agg(minority_ratio=_minority_ratio, n_trials="count")
        .reset_index()
    )

    auc_work = subject_auc_df.copy()
    auc_work["subject"] = auc_work["subject"].astype(str)
    plot_df = pd.merge(auc_work, imbalance_stats, on="subject", how="inner")
    if plot_df.empty:
        print("Warning: No subjects matched between AUC data and target data — skipping imbalance plot.")
        return

    x = plot_df["minority_ratio"].values.astype(float)
    y = plot_df["auc"].values.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)

    r_val, p_val = np.nan, np.nan
    slope, intercept = np.nan, np.nan
    if valid.sum() >= 3:
        r_val, p_val = stats.pearsonr(x[valid], y[valid])
        slope, intercept, *_ = stats.linregress(x[valid], y[valid])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["minority_ratio"],
        y=plot_df["auc"],
        mode='markers+text',
        text=plot_df["subject"].astype(str),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(color=COLORS[0], size=10, opacity=0.85,
                    line=dict(color='white', width=1)),
        name='Subjects',
        hovertemplate=(
            'Subject: %{text}<br>'
            'Minority ratio: %{x:.3f}<br>'
            'AUC: %{y:.3f}<extra></extra>'
        ),
    ))

    if np.isfinite(slope):
        x_line = np.array([np.nanmin(x[valid]), np.nanmax(x[valid])])
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name=f'r = {r_val:.2f}, p = {p_val:.3f}',
        ))

    fig.add_hline(
        y=0.5, line_dash='dash', line_color='gray', opacity=0.5,
        annotation_text='Chance (0.5)', annotation_position='bottom right',
    )
    fig.add_vline(
        x=0.5, line_dash='dot', line_color='gray', opacity=0.4,
        annotation_text='Perfect balance', annotation_position='top left',
    )

    corr_label = (
        f"r = {r_val:.3f}, p = {p_val:.3f}" if np.isfinite(r_val) else "r = N/A"
    )
    fig.update_layout(
        template='plotly_white',
        title=dict(
            text=(
                f'<b>{pipeline_label}: AUC vs {label_col} Class Imbalance</b><br>'
                f'<sup>{corr_label} | n = {valid.sum()} subjects</sup>'
            ),
            font=dict(size=16),
        ),
        xaxis_title='Minority-class ratio (0 = fully imbalanced, 0.5 = balanced)',
        yaxis_title='AUC',
        xaxis=dict(range=[0.0, 0.55]),
        yaxis=dict(range=[max(0.0, float(np.nanmin(y)) - 0.1),
                          min(1.0, float(np.nanmax(y)) + 0.1)]),
        width=800,
        height=650,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    os.makedirs(results_path, exist_ok=True)
    out_path = os.path.join(results_path, f"{filename_base}_auc_vs_{label_col}_imbalance")
    fig.write_image(f"{out_path}.png", scale=2)
    fig.write_image(f"{out_path}.pdf")
    fig.write_html(f"{out_path}.html")
    plot_df.to_csv(f"{out_path}_data.csv", index=False)
    print(f"  Saved AUC vs {label_col} class imbalance → {out_path}.png")
