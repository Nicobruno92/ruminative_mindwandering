#%%
import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mne
import sys

sys.path.insert(0, './')
from utils.analysis_helpers import compute_grand_averages, plot_erp, compute_erps, fit_lmm_for_time_bins
from ERPs.generate_evokeds import process_subjects_parallel

print('Packages loaded')

# Paths and settings
root = "/network/iss/cenir/analyse/meeg/CYBERSART/"
# derivatives_folder = os.path.join(root, "derivatives_nico")
derivatives_folder = os.path.join(root, "BIDS", "derivatives",)
subjects = [f"{i:02}" for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
data = "eeg"

# Analysis settings
conditions_of_interest = ['go/correct/ontask', 'go/correct/offtask']
offtask_metrics = ['mean', 'median', 'quartiles', 'tertiles', 'highlow']
posterior_roi = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']
distance = 5

# Time bins for analysis
time_bins = [
    (-0.1, 0.0), (0.0, 0.15), (0.15, 0.25), (0.25, 0.35),
    (0.35, 0.55), (0.55, 0.75), (0.75, 1.0)
]

time_bins_broad = [
    (-0.1, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, 1.0)
]

# Create results directories
os.makedirs('./results/ERPs/figs', exist_ok=True)
os.makedirs('./results/ERPs/lmm', exist_ok=True)

#%%
# Process subjects and generate evokeds
print("Processing subjects and generating evokeds...")
print("Note: Evokeds will be baseline-corrected to (-0.3, 0) and cropped to (-0.3, 1.2) seconds")
process_subjects_parallel(root, subjects, tasks, metrics=offtask_metrics, data=data, 
                         ref_channels=['TP9', 'TP10'], distance=distance, n_jobs=8)

#%%
# Main analysis loop
for metric in offtask_metrics:
    print(f"\n=== Processing metric: {metric} ===")

    # Compute grand averages
    participant_evokeds = compute_grand_averages(
        subjects=subjects, data=data, conditions_of_interest=conditions_of_interest,
        derivatives_folder=derivatives_folder, metric=metric
    )
    
    # Data quality check
    total_subjects = sum(1 for subj_id, conditions in participant_evokeds.items() 
                        if any(conditions.get(cond, []) for cond in conditions_of_interest))
    print(f"Data available for {total_subjects}/{len(subjects)} subjects")
    
    if total_subjects == 0:
        print(f"No data available for {metric}. Skipping.")
        continue
    
    # Verify baseline correction was applied
    sample_evoked = None
    for subj_id, conditions in participant_evokeds.items():
        for cond in conditions_of_interest:
            if conditions.get(cond, []):
                sample_evoked = conditions[cond][0]
                break
        if sample_evoked:
            break
    
    if sample_evoked:
        baseline_window = sample_evoked.times[(sample_evoked.times >= -0.3) & (sample_evoked.times <= 0)]
        baseline_mean = np.mean(sample_evoked.data[:, (sample_evoked.times >= -0.3) & (sample_evoked.times <= 0)])
        print(f"Baseline correction check - mean amplitude in baseline window: {baseline_mean:.6f} µV")
    
    # Save evokeds
    with open(f"./results/ERPs/participant_evokeds_{metric}_car.pkl", 'wb') as file:
        pickle.dump(participant_evokeds, file)
    
    # Analyze both time window types and determine which has more significant results
    results = {}
    for time_type, bins in [("regular", time_bins), ("broad", time_bins_broad)]:
        # Compute ERP data and fit models
        erp_data = compute_erps(participant_evokeds, subjects, conditions_of_interest, 
                               posterior_roi, bins, aggregate='subject')
        lmm_results = fit_lmm_for_time_bins(erp_data)
        
        # Save results
        erp_data.to_csv(f"./results/ERPs/lmm/erp_data_{metric}_{time_type}_car.csv", index=False)
        lmm_results.to_csv(f"./results/ERPs/lmm/lmm_results_{metric}_{time_type}_car.csv", index=False)
        
        # Store significant windows
        sig_windows = lmm_results[lmm_results['Condition_Coef_pvalue'] < 0.05].copy()
        sig_windows_fdr = lmm_results[lmm_results['Condition_Coef_pvalue_FDR'] < 0.05].copy()
        
        results[time_type] = {
            'significant_windows': sig_windows,
            'significant_windows_fdr': sig_windows_fdr
        }
        
        print(f"  {time_type}: {len(sig_windows)} significant windows (raw), {len(sig_windows_fdr)} (FDR)")
    
    # Use the analysis with more significant results for plotting
    use_broad = len(results['broad']['significant_windows']) >= len(results['regular']['significant_windows'])
    plot_type = 'broad' if use_broad else 'regular'
    print(f"Using {plot_type} windows for plots")
    
    # Generate plots
    try:
        sig_windows = results[plot_type]['significant_windows']
        sig_windows_fdr = results[plot_type]['significant_windows_fdr']
        
        # Plot with raw p-values
        fig_raw = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                          return_fig=True, significant_windows=sig_windows)
        fig_raw.write_html(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.html")
        try:
            fig_raw.write_image(f"./results/ERPs/figs/ERP_{metric}_raw_pvals_car.png")
        except Exception as e:
            print(f"PNG export (raw p-values) skipped for {metric}: {e}")
        
        # Plot with FDR-corrected p-values
        fig_fdr = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, 
                          return_fig=True, significant_windows=sig_windows_fdr)
        fig_fdr.write_html(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.html")
        try:
            fig_fdr.write_image(f"./results/ERPs/figs/ERP_{metric}_fdr_pvals_car.png")
        except Exception as e:
            print(f"PNG export (FDR p-values) skipped for {metric}: {e}")
        
        # Plot without significant windows
        fig = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, return_fig=True)
        fig.write_html(f"./results/ERPs/figs/ERP_{metric}_car.html")
        try:
            fig.write_image(f"./results/ERPs/figs/ERP_{metric}_car.png")
        except Exception as e:
            print(f"PNG export (no windows) skipped for {metric}: {e}")
        
        print(f"All plots saved for {metric}")
        
    except Exception as e:
        print(f"ERROR generating plots for {metric}: {e}")

print("\nProcessing completed.")

# %%