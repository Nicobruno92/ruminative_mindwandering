import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px

import mne
from mne.stats import spatio_temporal_cluster_test


import sys
sys.path.insert(0, './')
from utils.bids_compliance import load_evokeds
from utils.analysis_helpers import compute_grand_averages, plot_erp
from ERPs.create_evokeds2 import generate_save_evokeds_parallel

print('packages loaded')

# from utils.bids_compliance import read_epochs, save_evokeds

# from utils import bids_compliance

# %%
root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
# root = "//l2export/iss02.cenir/analyse/meeg/CYBERSART/"
# Defining the paths for saving results and raw data
derivatives_folder = os.path.join(root, "derivatives_nico")
# bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")


subjects = [str(i) if i > 9 else "0"+str(i) for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3','Sart4']
data = "eeg"

# Loop through conditions, create evoked objects, and combine
stimulus_condition = ['go', 'nogo']
response_condition = ['correct', 'incorrect']
mind_condition = ['ontask', 'offtask']
conditions_of_interest = ['go/correct/ontask' ,'go/correct/offtask']
offtask_metrics = ['mean', 'median', 'quartiles', 'teriles', 'highlow']

# List of electrodes of interest (ROI)
posterior_roi = ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']

#%%
for metric in offtask_metrics:
    print(f"Computing grand averages for metric {metric}")
    generate_save_evokeds_parallel(root, subjects, tasks, data="eeg", ref_channels=['TP9', 'TP10'], distance=5, split=metric, max_workers=4)
    
    # conditions_of_interest = ['nogo/correct','go/correct']

    # Compute grand averages
    participant_evokeds = compute_grand_averages(
        subjects=subjects,
        data=data,
        conditions_of_interest=conditions_of_interest,
        derivatives_folder=derivatives_folder,)
    
    fig = plot_erp(participant_evokeds,conditions_of_interest, posterior_roi)
    fig.write_image(f"../results/ERPs/figs/ERP_{metric}_CAR.png")
