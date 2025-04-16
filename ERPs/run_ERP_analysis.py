#%%
import os
import pickle
import pandas as pd
import numpy as np

import plotly.graph_objects as go

import mne

import sys
# sys.path.insert(0, './')
sys.path.insert(0, '../')
from utils.analysis_helpers import compute_grand_averages, plot_erp, compute_erps, fit_lmm_for_time_bins
from ERPs.generate_evokeds import process_subjects_parallel

print('Packages loaded')

# Paths and settings
# root = "/network/iss/cenir/analyse/meeg/CYBERSART/"
root = "/Volumes/cenir/analyse/meeg/CYBERSART/"


derivatives_folder = os.path.join(root, "derivatives_nico")
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
posterior_roi = ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

# Define time bins for LMM analysis (in seconds)
time_bins = [
    (-0.1, 0.0),   # Pre-stimulus baseline
    (0.0, 0.1),    # Early processing
    (0.1, 0.2),    # P1 component
    (0.2, 0.3),    # N1 component
    (0.3, 0.4),    # P2 component
    (0.4, 0.5),    # N2 component
    (0.5, 0.6),    # P3 early
    (0.6, 0.7),    # P3 peak
    (0.7, 0.8),    # P3 late
    (0.8, 0.9),    # Late processing
    (0.9, 1.0)     # Very late processing
]

# Create results directories if they don't exist
os.makedirs('./results/ERPs/figs', exist_ok=True)
os.makedirs('./results/ERPs/lmm', exist_ok=True)

#%%
# Process subjects and save evokeds for each metric
print("Processing subjects and generating evokeds...")
process_subjects_parallel(root, subjects, tasks, metrics=offtask_metrics, data=data, 
                         ref_channels=['TP9', 'TP10'], distance=5, n_jobs=-1)

#%%
# Compute grand averages, run LMM analysis, and plot ERPs
for metric in offtask_metrics:
    print(f"Computing grand averages for metric: {metric}")

    # Compute grand averages for the specific metric
    participant_evokeds = compute_grand_averages(
        subjects=subjects,
        data=data,
        conditions_of_interest=conditions_of_interest,
        derivatives_folder=derivatives_folder,
        metric=metric,
    )
    
    output_file_path = f"./results/ERPs/participant_evokeds_{metric}_car.pkl"
    # Save the participant_evokeds dictionary to a file
    with open(output_file_path, 'wb') as file:
        pickle.dump(participant_evokeds, file)
        print(f"participant_evokeds object saved to {output_file_path}")
    
    # Compute ERP data for LMM analysis using the defined time bins
    print(f"Computing ERP data for LMM analysis using {metric} metric...")
    erp_data = compute_erps(
        participant_evokeds=participant_evokeds,
        subjects=subjects,
        conditions_of_interest=conditions_of_interest,
        roi=posterior_roi,
        time_bins=time_bins,
        aggregate='condition'  # Aggregate by condition
    )
    
    # Save the ERP data for LMM analysis
    erp_data.to_csv(f"./results/ERPs/lmm/erp_data_{metric}_car.csv", index=False)
    print(f"ERP data for LMM analysis saved for metric {metric}.")
    
    # Fit LMM models for each time bin
    print(f"Fitting LMM models for each time bin using {metric} metric...")
    lmm_results = fit_lmm_for_time_bins(erp_data)
    
    # Save the LMM results
    lmm_results.to_csv(f"./results/ERPs/lmm/lmm_results_{metric}_car.csv", index=False)
    print(f"LMM results saved for metric {metric}.")
    
    # Identify significant time windows (using both raw p-values and FDR-corrected p-values)
    significant_windows = lmm_results[lmm_results['Condition_Coef_pvalue'] < 0.05].copy()
    significant_windows_fdr = lmm_results[lmm_results['Condition_Coef_pvalue_FDR'] < 0.05].copy()
    
    print(f"Found {len(significant_windows)} significant time windows using raw p-values")
    print(f"Found {len(significant_windows_fdr)} significant time windows after FDR correction")
    
    # Plot and save ERP figures for each metric with significant time windows
    # Using raw p-values
    fig_raw = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                   return_fig=True, significant_windows=significant_windows)
    fig_raw.write_html(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.html")
    print(f"ERP figure with raw p-values saved for metric {metric}.")
    
    # Using FDR-corrected p-values
    fig_fdr = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                    return_fig=True, significant_windows=significant_windows_fdr)
    fig_fdr.write_html(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.html")
    print(f"ERP figure with FDR-corrected p-values saved for metric {metric}.")
    
    # Also save the original figure without significant time windows for comparison
    fig = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, return_fig=True)
    fig.write_html(f"./results/ERPs/figs/ERP_{metric}_car.html")
    print(f"Original ERP figure saved for metric {metric}.")

print("Processing completed.")

# %%