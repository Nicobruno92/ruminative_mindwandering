import os
import pandas as pd

import plotly.graph_objects as go

import mne

import sys
sys.path.insert(0, './')
from utils.bids_compliance import load_evokeds
from utils.analysis_helpers import compute_grand_averages, plot_erp
from ERPs.generate_evokeds import process_subjects_parallel

print('Packages loaded')

# Paths and settings
# root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
root = "//l2export/iss02.cenir/analyse/meeg/CYBERSART/"
derivatives_folder = os.path.join(root, "derivatives_nico")
subjects = [f"{i:02}" for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
data = "eeg"

# Conditions and settings for classification and evoked generation
stimulus_condition = ['go', 'nogo']
response_condition = ['correct', 'incorrect']
mind_condition = ['ontask', 'offtask']
conditions_of_interest = ['go/correct/ontask', 'go/correct/offtask']
offtask_metrics = ['mean', 'median', 'quartiles', 'teriles', 'highlow']

# List of electrodes of interest (ROI)
posterior_roi = ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

# Process subjects and save evokeds for each metric
print("Processing subjects and generating evokeds...")
process_subjects_parallel(root, subjects, tasks, metrics=offtask_metrics, data=data, ref_channels=['TP9', 'TP10'], distance=5, n_jobs=-1)

# Compute grand averages and plot ERPs
for metric in offtask_metrics:
    print(f"Computing grand averages for metric: {metric}")

    # Compute grand averages for the specific metric
    participant_evokeds = compute_grand_averages(
        subjects=subjects,
        data=data,
        conditions_of_interest=conditions_of_interest,
        derivatives_folder=derivatives_folder,
    )
    
    # Plot and save ERP figures for each metric
    fig = plot_erp(participant_evokeds, conditions_of_interest, posterior_roi, return_fig=True)
    fig.write_image(f"./results/ERPs/figs/ERP_{metric}_mastoid.png")
    fig.write_html(f"./results/ERPs/figs/ERP_{metric}_mastoid.png")
    print(f"ERP figure saved for metric {metric}.")

print("Processing completed.")
