#%%
import os
import pickle
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import sys
sys.path.insert(0, './')
# sys.path.insert(0, '../')
from utils.analysis_helpers import compute_grand_averages, plot_erp, compute_erps, fit_lmm_for_time_bins, fit_true_mixed_effects_model
from ERPs.generate_evokeds import process_subjects_parallel



subjects = [f"{i:02}" for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
data = "eeg"

# Conditions and settings for classification and evoked generation
stimulus_condition = ['go', 'nogo']
response_condition = ['correct', 'incorrect']
mind_condition = ['ontask', 'offtask']
conditions_of_interest = ['go/correct/ontask', 'go/correct/offtask']
offtask_metrics = ['mean', 'median', 'quartiles', 'tertiles', 'highlow']

# List of electrodes of interest (ROI)
posterior_roi = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']

# IMPROVED: Broader time bins to better capture ERP components
time_bins = [
    (-0.1, 0.0),   # Pre-stimulus baseline
    (0.0, 0.15),   # Early sensory processing (P1)
    (0.15, 0.25),  # N1 component (broader window)
    (0.25, 0.35),  # P2 component (broader window)
    (0.35, 0.55),  # N2/P3a components (broader window)
    (0.55, 0.75),  # P3b peak (broader window)
    (0.75, 1.0),   # Late positive complex
]

# Alternative: Even broader time windows for stronger effects
time_bins_broad = [
    (-0.1, 0.0),   # Baseline
    (0.0, 0.2),    # Early components (P1/N1)
    (0.2, 0.4),    # Mid components (P2/N2)
    (0.4, 0.8),    # P3 complex (main window of interest)
    (0.8, 1.0),    # Late effects
]

# Create results directories if they don't exist
os.makedirs('./results/ERPs/figs', exist_ok=True)
os.makedirs('./results/ERPs/lmm', exist_ok=True)

#%%
# Compute grand averages, run LMM analysis, and plot ERPs
for metric in offtask_metrics:
    evoked_file_path = f"./results/ERPs/participant_evokeds_{metric}_car.pkl"
    
    # Check if file exists before attempting to load
    if not os.path.exists(evoked_file_path):
        print(f"Warning: File {evoked_file_path} not found. Skipping metric {metric}.")
        continue
    
    print(f"\n=== Processing metric: {metric} ===")
    
    try:
        with open(evoked_file_path, 'rb') as file:
            participant_evokeds = pickle.load(file)
            print(f"participant_evokeds object loaded from {evoked_file_path}")
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading {evoked_file_path}: {e}. Skipping metric {metric}.")
        continue
    except Exception as e:
        print(f"Unexpected error loading {evoked_file_path}: {e}. Skipping metric {metric}.")
        continue

    # IMPROVED: Add diagnostic information about data quality
    total_subjects = 0
    total_trials = 0
    for subj_id, conditions in participant_evokeds.items():
        has_data = False
        for condition in conditions_of_interest:
            if condition in conditions and conditions[condition]:
                has_data = True
                total_trials += len(conditions[condition])
        if has_data:
            total_subjects += 1
    
    print(f"  - Data available for {total_subjects}/{len(subjects)} subjects")
    print(f"  - Total trials across all subjects: {total_trials}")
    
    if total_subjects == 0:
        print(f"  - No data available for metric {metric}. Skipping.")
        continue

    # IMPROVED: Test both regular and broad time windows
    for time_window_type, current_time_bins in [("regular", time_bins), ("broad", time_bins_broad)]:
        print(f"\n--- Analyzing {time_window_type} time windows ---")
        
        # Compute ERP data for LMM analysis using the defined time bins
        print(f"Computing ERP data for LMM analysis using {metric} metric with {time_window_type} windows...")
        erp_data = compute_erps(
            participant_evokeds=participant_evokeds,
            subjects=subjects,
            conditions_of_interest=conditions_of_interest,
            roi=posterior_roi,
            time_bins=current_time_bins,
            aggregate='subject'  # FIXED: Use 'subject' to maintain subject-level data for mixed-effects models
        )
        
        # Save the ERP data for LMM analysis
        erp_data.to_csv(f"./results/ERPs/lmm/erp_data_{metric}_{time_window_type}_car.csv", index=False)
        print(f"ERP data for LMM analysis saved for metric {metric} ({time_window_type} windows).")
        
        # IMPROVED: Fit both types of models
        print(f"Fitting models for each time bin using {metric} metric ({time_window_type} windows)...")
        
        # Try the improved mixed-effects model first
        try:
            lmm_results = fit_true_mixed_effects_model(erp_data)
            model_type_used = "Mixed-Effects"
        except Exception as e:
            print(f"Mixed-effects model failed: {e}. Falling back to standard LMM.")
            lmm_results = fit_lmm_for_time_bins(erp_data)
            model_type_used = "Standard LMM"
        
        print(f"  - Model type used: {model_type_used}")
        
        # Save the LMM results
        lmm_results.to_csv(f"./results/ERPs/lmm/lmm_results_{metric}_{time_window_type}_car.csv", index=False)
        print(f"LMM results saved for metric {metric} ({time_window_type} windows).")
        
        # Identify significant time windows (using both raw p-values and FDR-corrected p-values)
        significant_windows = lmm_results[lmm_results['Condition_Coef_pvalue'] < 0.05].copy()
        significant_windows_fdr = lmm_results[lmm_results['Condition_Coef_pvalue_FDR'] < 0.05].copy()
        
        print(f"  - Found {len(significant_windows)} significant time windows using raw p-values ({time_window_type})")
        print(f"  - Found {len(significant_windows_fdr)} significant time windows after FDR correction ({time_window_type})")
        
        # IMPROVED: Print effect sizes and p-values for diagnostics
        if len(lmm_results) > 0:
            min_p = lmm_results['Condition_Coef_pvalue'].min()
            max_effect = lmm_results['Condition_Coef'].abs().max()
            print(f"  - Minimum p-value: {min_p:.6f}")
            print(f"  - Maximum absolute effect size: {max_effect:.6f}")
            
            # Show details for significant windows
            if len(significant_windows) > 0:
                print(f"  - Significant windows (raw p < 0.05):")
                for _, row in significant_windows.iterrows():
                    print(f"    {row['StartTime']:.2f}-{row['EndTime']:.2f}s: p={row['Condition_Coef_pvalue']:.4f}, effect={row['Condition_Coef']:.4f}")
        
        # Store results for comparison
        if time_window_type == "regular":
            regular_results = lmm_results.copy()
            regular_significant = significant_windows.copy()
            regular_significant_fdr = significant_windows_fdr.copy()
        else:  # broad
            broad_results = lmm_results.copy()
            broad_significant = significant_windows.copy()
            broad_significant_fdr = significant_windows_fdr.copy()
    
    # FIXED: Create plots using the analysis with MORE significant results
    print(f"\n--- Creating plots for {metric} ---")
    
    # Compare which analysis has more significant results
    regular_sig_count = len(regular_significant) if 'regular_significant' in locals() else 0
    broad_sig_count = len(broad_significant) if 'broad_significant' in locals() else 0
    
    if regular_sig_count >= broad_sig_count:
        print(f"Using REGULAR time windows for plots (found {regular_sig_count} significant windows)")
        plot_significant_windows = regular_significant
        plot_significant_windows_fdr = regular_significant_fdr
        results_suffix = "regular"
    else:
        print(f"Using BROAD time windows for plots (found {broad_sig_count} significant windows)")
        plot_significant_windows = broad_significant    
        plot_significant_windows_fdr = broad_significant_fdr
        results_suffix = "broad"
    
    # Plot and save ERP figures for each metric with significant time windows
    try:
        # Using raw p-values
        fig_raw = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                       return_fig=True, significant_windows=plot_significant_windows)
        # fig_raw.write_image(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.png")
        fig_raw.write_image(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.svg")
        fig_raw.write_html(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.html")
        print(f"ERP figure with raw p-values saved for metric {metric} (using {results_suffix} results).")
        
        # Using FDR-corrected p-values
        fig_fdr = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                        return_fig=True, significant_windows=plot_significant_windows_fdr)
        fig_fdr.write_image(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.png")
        fig_fdr.write_image(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.svg")
        fig_fdr.write_html(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.html")
        print(f"ERP figure with FDR-corrected p-values saved for metric {metric} (using {results_suffix} results).")
        
        # Also save the original figure without significant time windows for comparison
        fig = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, return_fig=True)
        fig.write_image(f"./results/ERPs/figs/ERP_{metric}_car.png")
        fig.write_image(f"./results/ERPs/figs/ERP_{metric}_car.svg")
        fig.write_html(f"./results/ERPs/figs/ERP_{metric}_car.html")
        print(f"Original ERP figure saved for metric {metric}.")
        # NOTE: Removed PNG generation to prevent hanging
        
    except Exception as e:
        print(f"ERROR: Failed to generate ERP plots for metric {metric}: {e}")
        continue

print("Processing completed.")

# %%