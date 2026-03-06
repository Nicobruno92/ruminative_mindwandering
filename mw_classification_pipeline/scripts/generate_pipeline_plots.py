#!/usr/bin/env python
"""
Standalone Script for Generating Pipeline Plots.

This script allows you to generate all Plotly visualizations for a completed
classification run (LOSO or WithinSubject) without needing to re-run the classification.
It automatically reads the saved `.csv` and/or `.pkl` files from the specified
results directory.

USAGE:
    python generate_pipeline_plots.py --results_dir path/to/results/folder

Project: depressed_mindwandering

example: 
python mw_classification_pipeline/scripts/generate_pipeline_plots.py --results_dir mw_classification_pipeline/results/MW_Classification/WithinSubject/on_vs_off_within_median/all

"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Add parent directory to sys.path so we can import utils.*
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.plotting_utils import (
    plot_metric_distribution_with_stats,
    plot_feature_importances,
    plot_consolidated_permutation_results,
    plot_loso_subject_metrics,
    plot_subject_level_densities,
    plot_shap_beeswarm,
    plot_shap_feature_importance,
    plot_shap_comparative_boxplots,
    plot_confusion_matrix,
    plot_roc_curve,
    set_plot_style
)

warnings.filterwarnings("ignore")

def find_file_by_suffix(directory, suffix):
    """Find a file in the directory ending with the given suffix."""
    for f in os.listdir(directory):
        if f.endswith(suffix):
            return os.path.join(directory, f)
    return None

def load_pkl(path):
    """Safely load a pickle file."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_importances(summary_df):
    """Extract feature names and importances from a summary DataFrame."""
    if summary_df is None or summary_df.empty:
        return [], [], []
    
    imp_cols = [c for c in summary_df.columns if c.startswith('importance_')]
    if not imp_cols:
        return [], [], []
    
    fnames = [c.replace('importance_', '') for c in imp_cols]
    mean_imp = summary_df[imp_cols].mean().values
    std_imp = summary_df[imp_cols].std().values
    return fnames, mean_imp, std_imp

def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from a pipeline results directory.")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the dimension results directory (e.g., results/.../LOSO/ON_vs_OFF/spectral)")
    parser.add_argument("--top_n_features", type=int, default=20, help="Number of top features to plot")
    parser.add_argument("--positive_class", type=str, default="ON-task", help="Name of positive class")
    parser.add_argument("--negative_class", type=str, default="OFF-task", help="Name of negative class")
    return parser.parse_args()

def main():
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory {results_dir} does not exist.")
        sys.exit(1)
        
    print(f"Generating plots for results in: {results_dir}")
    
    # 1. Deduce dimensions and model types from directory structure
    path_parts = Path(results_dir).parts
    try:
        if "WithinSubject" in path_parts:
            pipeline_type = "WithinSubject"
            dim_idx = path_parts.index("WithinSubject") + 1
        elif "LOSO" in path_parts:
            pipeline_type = "LOSO"
            dim_idx = path_parts.index("LOSO") + 1
        else:
            pipeline_type = "Unknown"
            dim_idx = -2
        dimension_name = path_parts[dim_idx] if dim_idx < len(path_parts) else "Unknown"
    except ValueError:
        pipeline_type = "Unknown"
        dimension_name = "Unknown"
        
    print(f"Detected Pipeline: {pipeline_type}, Dimension: {dimension_name}")
    
    # Configure plotting
    set_plot_style("seaborn-v0_8")
    
    # 2. Find True Result Files
    true_summary_csv = find_file_by_suffix(results_dir, "_summary.csv")
    true_detailed_pkl = find_file_by_suffix(results_dir, "_detailed.pkl")
    true_all_shap_pkl = find_file_by_suffix(results_dir, "_all_shap_values.pkl")
    true_subject_metrics_csv = find_file_by_suffix(results_dir, "_loso_subject_metrics.csv") or find_file_by_suffix(results_dir, "_ws_subject_metrics_averaged.csv")
    
    true_summary_df = pd.read_csv(true_summary_csv) if true_summary_csv else None
    true_detailed = load_pkl(true_detailed_pkl)
    true_shap_runs = load_pkl(true_all_shap_pkl)
    
    # Base filename extraction
    if true_summary_csv:
        filename_base = os.path.basename(true_summary_csv).replace("_summary.csv", "")
    else:
        filename_base = "pipeline_plot"
        
    # 3. Find Permutation Result Files
    perm_dir = os.path.join(results_dir, "permutation")
    perm_exists = os.path.isdir(perm_dir)
    
    perm_summary_csv = find_file_by_suffix(perm_dir, "_summary.csv") if perm_exists else None
    perm_detailed_pkl = find_file_by_suffix(perm_dir, "_detailed.pkl") if perm_exists else None
    perm_summary_df = pd.read_csv(perm_summary_csv) if perm_summary_csv else None
    perm_detailed = load_pkl(perm_detailed_pkl)
    
    # Extract feature names
    feature_names, mean_imp, std_imp = extract_importances(true_summary_df)
    
    # Attempt to load permuted SHAP values from permutation run folders
    perm_shap_runs = []
    if perm_exists:
        for run_f in os.listdir(perm_dir):
            run_p = os.path.join(perm_dir, run_f)
            if os.path.isdir(run_p):
                shap_f = find_file_by_suffix(run_p, "_shap_values.pkl") or find_file_by_suffix(run_p, "_shap_values_stacked.pkl")
                if shap_f:
                    data = load_pkl(shap_f)
                    if data and 'shap_values' in data:
                        perm_shap_runs.append(data['shap_values'])
    
    # ---------------------------------------------------------
    # Generate Plots
    # ---------------------------------------------------------
    
    # A. Feature Importances
    if len(feature_names) > 0:
        print("-> Plotting Feature Importances")
        plot_feature_importances(feature_names, mean_imp, std_imp, results_dir, filename_base, top_n=args.top_n_features)
        
    # B. SHAP Beeswarm & Feature Importance
    if true_shap_runs and len(true_shap_runs) > 0:
        print("-> Plotting SHAP Distributions")
        combined_shap = np.concatenate(true_shap_runs, axis=0) if isinstance(true_shap_runs, list) else true_shap_runs
        # We need X values for colors. Since we don't have X matrix here we pass dummy for now or skip colors
        # To avoid breaking we pass a dummy matrix of zeros (colors will be uniform)
        dummy_X = np.zeros(combined_shap.shape)
        try:
            plot_shap_beeswarm(combined_shap, dummy_X, feature_names, dimension_name, results_dir, filename_base, max_display=args.top_n_features)
            plot_shap_feature_importance(combined_shap, dummy_X, feature_names, results_dir, filename_base, max_display=args.top_n_features)
        except Exception as e:
            print(f"Warning: Could not plot SHAP beeswarm natively without original X data: {e}")
            
    # C. Subject-Level Metrics Barplot
    if true_subject_metrics_csv:
        print("-> Plotting Subject-Level Metrics")
        sub_df = pd.read_csv(true_subject_metrics_csv)
        plot_loso_subject_metrics(sub_df, results_dir, filename_base)
        
    # D. Permutation Distributions
    if true_summary_df is not None and perm_summary_df is not None:
        print("-> Plotting Consolidated Permutation Results")
        from scipy import stats as scipy_stats
        
        results_for_plotting = {}
        for metric in ["auc", "balanced_accuracy", "auprc", "mcc"]:
            col_name = f"mean_{metric}"
            if col_name in true_summary_df.columns and col_name in perm_summary_df.columns:
                true_vals = true_summary_df[col_name].dropna().values
                perm_vals = perm_summary_df[col_name].dropna().values
                
                if len(true_vals) > 0 and len(perm_vals) > 0:
                    _, mwu_p = scipy_stats.mannwhitneyu(true_vals, perm_vals, alternative="greater")
                    empirical_p = np.mean(perm_vals >= np.mean(true_vals))
                    label = "AUC" if metric == "auc" else "AUPRC" if metric == "auprc" else "MCC" if metric == "mcc" else "Balanced Accuracy"
                    results_for_plotting[label] = {
                        "true_values": true_vals,
                        "perm_values": perm_vals,
                        "p_value": mwu_p,
                        "empirical_p": empirical_p
                    }
        
        if results_for_plotting:
            plot_consolidated_permutation_results(results_for_plotting, dimension_name, "Model", results_dir, filename_base)
            
    # E. Advanced Visualizations (Densities & SHAP Boxplots)
    print("-> Plotting Advanced Custom Visualizations")
    
    # E.1 Subject-Level Densities
    # We need the comprehensive metrics lists. If we have detailed.pkl we use it. 
    # Otherwise, this is difficult to rebuild purely from summary CSVs because we need fold-level metrics per run.
    if true_detailed and perm_detailed:
        plot_subject_level_densities(true_detailed, perm_detailed, dimension_name, results_dir, filename_base, metric='auc')
    else:
        print("  ! Skipping Subject Level Densities (requires _detailed.pkl from run)")
        
    # E.2 SHAP Comparative Boxplots
    if true_shap_runs and perm_shap_runs:
        # Wrap True SHAP runs in a list if it's a single array (for single run scenarios)
        t_runs = true_shap_runs if isinstance(true_shap_runs, list) else [true_shap_runs]
        p_runs = perm_shap_runs if isinstance(perm_shap_runs, list) else [perm_shap_runs]
        plot_shap_comparative_boxplots(t_runs, p_runs, feature_names, results_dir, filename_base, num_features=args.top_n_features)
    else:
        print("  ! Skipping SHAP Comparative Boxplots (requires SHAP values from true and permutation runs)")
        
        
    # F. Confusion Matrix & ROC/PR Curves
    # These require 'fold_cms', 'fold_tprs', 'fold_fprs' which are saved in _detailed.pkl
    if true_detailed:
        print("-> Plotting Confusion Matrix and ROC Curve")
        cms_list = [cm for r in true_detailed for cm in r.get("fold_cms", [])]
        if cms_list:
            avg_cm = np.mean(cms_list, axis=0)
            cell_stats = pd.DataFrame({
                "cell": ["TN", "FP", "FN", "TP"],
                "mean": avg_cm.flatten(),
                "std": np.std(cms_list, axis=0).flatten(),
            })
            plot_confusion_matrix(avg_cm, args.negative_class, args.positive_class, cell_stats, results_dir, filename_base)
            
        all_tprs = [tpr for r in true_detailed for tpr in r.get("fold_tprs", [])]
        all_fprs = [fpr for r in true_detailed for fpr in r.get("fold_fprs", [])]
        if all_tprs and all_fprs:
            mean_fpr = np.linspace(0, 1, 100)
            tprs_interp = []
            aucs_interp = []
            for fpr, tpr in zip(all_fprs, all_tprs):
                if len(fpr) > 1:
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs_interp.append(interp_tpr)
                    aucs_interp.append(np.trapz(interp_tpr, mean_fpr))
            if tprs_interp:
                mean_tpr = np.mean(tprs_interp, axis=0)
                mean_tpr[-1] = 1.0
                plot_roc_curve(mean_fpr, mean_tpr, np.std(tprs_interp, axis=0), results_dir, filename_base, np.mean(aucs_interp), len(tprs_interp))
                
    print(f"\nFinished regenerating plots in {results_dir}")

if __name__ == "__main__":
    main()
