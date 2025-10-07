#%%
import os
import pickle
import pandas as pd

import plotly.graph_objects as go

import mne

import sys
sys.path.insert(0, './')
# sys.path.insert(0, '../')
from utils.analysis_helpers import compute_grand_averages, plot_erp
from ERPs.generate_evokeds import process_subjects_parallel

print('Packages loaded')

# Paths and settings
root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
# root = "//l2export/iss02.cenimodule r/analyse/meeg/CYBERSART/"

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
# Compute grand averages and plot ERPs

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