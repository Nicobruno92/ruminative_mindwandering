# %%
import os
import numpy as np
import pandas as pd

import mne

import sys
sys.path.insert(0, './')
from utils.bids_compliance import read_epochs, save_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

# from utils import bids_compliance

# %%
# root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
root = "//l2export/iss02.cenir/analyse/meeg/CYBERSART/"
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

#%%
# Main loop to generate and save evokeds for each subject/session/task combination
for subject in subjects:
    epochs_tasks = []
    for task in tasks:
        print(f"Creating evokeds for subject {subject} for {task}")
        try:
            # Load the epochs and events
            epochs, events = read_epochs(
                derivatives_folder,
                subject,
                task,
                data,
                desc="autoPreproc",
            )
            
            rereference_epochs = epochs.set_eeg_reference(ref_channels=['TP9', 'TP10'],)
            
            epochs_tasks.append(rereference_epochs)

        except Exception as e:
            print(f"Could not load epochs for subject {subject} task {task}: {e}")
    
    try:
        # Concatenate all epochs
        epochs_concat = mne.concatenate_epochs(epochs_tasks)
        
        # Filter the epochs based on distance to probe
        filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, 5)
        
        #classify dimensional onoff epochs
        epochs_classified = classify_onoff_epochs(filtered_epochs, split="median")
        
        evokeds = []  # List to store evoked responses for all conditions

        # Loop through all combinations of conditions
        for stim in stimulus_condition:
            for resp in response_condition:
                for mind in mind_condition:
            #         # Define the event string corresponding to this condition
                    event_str = f'{stim}/{resp}/{mind}'
                    try:
                        # Select epochs for this specific condition
                        selected_epochs = filtered_epochs[event_str]

                        # Compute the evoked (average) response
                        evoked = selected_epochs.average()

                        # Resample the evoked to 500 Hz
                        # evoked.resample(500)

                        # # Shift time axis so that tmin = -0.3 seconds
                        # desired_tmin = -0.3
                        # current_tmin = evoked.times[0]
                        # shift_amount = desired_tmin - current_tmin
                        # evoked.shift_time(shift_amount, relative=True)

                        # Crop the evoked to the desired time range
                        evoked.crop(tmin=-0.3, tmax=1.2)

                        # Append evoked object to the list
                        # print(evoked.times[0], evoked.times[-1])
                        evokeds.append(evoked)
                    except Exception as e:
                        print(f"Could not create evoked response for {event_str}: {e}")
                        continue
        
        # Save all evokeds in a single file
        if evokeds:
            print(f"Saving evoked responses for subject {subject}")
            save_evokeds(evokeds, derivatives_folder, subject, data)
            
    except Exception as e:
        print(f"Could not concatenate epochs for subject {subject}: {e}")
        continue


# %%
#  # Load the epochs and events
# epochs, events = read_epochs(
#     derivatives_folder,
#     '06',
#     'Sart1',
#     'eeg',
#     desc="autoPreproc",
# )
