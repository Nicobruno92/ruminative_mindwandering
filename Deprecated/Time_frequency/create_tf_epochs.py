#%%
import os
import pickle
import numpy as np
import pandas as pd
import gc


import plotly.graph_objects as go

import mne

from joblib import Parallel, delayed
import re

import sys
# sys.path.insert(0, './')
sys.path.insert(0, '../')
from utils.analysis_helpers import compute_erps, fit_lmm_for_time_bins, plot_erp
from utils.bids_compliance import read_epochs, save_tfr_epochs, read_tfr_epochs
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs


print('Packages loaded')

# Paths and settings
# root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
# root = "//l2export/iss02.cenimodule r/analyse/meeg/CYBERSART/"
root = "/Volumes/cenir/analyse/meeg/CYBERSART/"


derivatives_folder = os.path.join(root, "derivatives_nico")
subjects = [f"{i:02}" for i in range(31, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
data = "eeg"

# Conditions and settings for classification and evoked generation
stimulus_condition = ['go', 'nogo']
response_condition = ['correct', 'incorrect']
mind_condition = ['ontask', 'offtask']
conditions_of_interest = ['go/correct/ontask', 'go/correct/offtask']
# offtask_metrics = ['mean', 'median', 'quartiles', 'tertiles', 'highlow']
offtask_metrics = [ 'highlow']

#%%
class TimeFrequencyProcessor:
    def __init__(self, root, tasks, metrics, data="eeg", distance=5, n_jobs=4):
        self.root = root
        self.derivatives_folder = os.path.join(root, "derivatives_nico")
        self.tasks = tasks
        self.metrics = metrics
        self.data = data
        self.distance = distance
        self.n_jobs = n_jobs

    def classify_all_metrics(self, subject_epochs):
        """
        Classify epochs with multiple metrics.
        """
        classified_epochs_dict = {}
        for metric in self.metrics:
            classified_epochs_dict[metric] = classify_onoff_epochs(subject_epochs.copy(), split=metric)
        return classified_epochs_dict

    def process_subject_for_metrics(self, subject):
        """
        Process and classify epochs for a single subject.
        """
        epochs_tasks = []
        for task in self.tasks:
            try:
                epochs, events = read_epochs(self.derivatives_folder, subject, task, self.data, desc="autoPreproc")
                epochs_tasks.append(epochs.copy())
            except Exception as e:
                print(f"Skipping {subject} {task}: {e}")

        if not epochs_tasks:
            print(f"No data for subject {subject}")
            return None

        try:
            epochs_concat = mne.concatenate_epochs(epochs_tasks)
            filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, self.distance)
        except Exception as e:
            print(f"Failed concatenating or filtering epochs for subject {subject}: {e}")
            return None
        
        classified_epochs_dict = self.classify_all_metrics(filtered_epochs)

        return classified_epochs_dict

    def generate_save_epochs_tfr_allmetrics(self, subject, method = 'morlet', tmin=None, tmax=None,):
        """
        Generate and save PSDs for different metrics.
        """
        
        try:
            classified_epochs_dict = self.process_subject_for_metrics(subject)
            for metric, epochs_classified in classified_epochs_dict.items():
                freqs = np.linspace(1, 45)
                n_cycles = freqs / 2.0  # different number of cycle per frequency
                epochsTFR= epochs_classified.compute_tfr(method=method, freqs=freqs, n_cycles=n_cycles, decim = 2, output='power')

                save_tfr_epochs(epochsTFR, self.derivatives_folder, subject, self.data, desc=metric)
                print(f"PSDs saved for subject {subject} using metric {metric}")
                
                # Memory cleanup
                del epochs_classified, epochsTFR
                gc.collect()

            del classified_epochs_dict
            gc.collect()
            
        except Exception as e:
            print(f"Failed processing subject {subject}: {e}")
            
    def process_epochs_psd_subjects_parallel(self, subjects,  method = 'morlet', fmin = 0.5, fmax = 40, tmin=None, tmax=None):
        """
        Parallelized processing for multiple subjects.
        """
        
        Parallel(n_jobs=self.n_jobs)(
            delayed(self.generate_save_epochs_tfr_allmetrics)(subject,method = method, tmin=tmin, tmax=tmax) for subject in subjects
        )
#%%
tfr_processor = TimeFrequencyProcessor(root, tasks, offtask_metrics, n_jobs= 3)

print("Processing subjects and generating PSDs...")
epochs_metrics  = tfr_processor.process_epochs_psd_subjects_parallel(subjects)

# %%
epochs  = epochs_metrics['highlow']
# %%
freqs = np.linspace(1, 45)
n_cycles = freqs / 2.0  # different number of cycle per frequency
epochsTFR= epochs['go'].compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles, decim = 2, output='power', n_jobs=-1)
# %%
