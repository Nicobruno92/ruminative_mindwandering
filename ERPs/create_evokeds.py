# %%
import os
import numpy as np
import pandas as pd

import mne

import sys
sys.path.insert(0, '../')
from utils.bids_compliance import read_epochs, save_evokeds

# from utils import bids_compliance

# %%
# root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
root = "/l2export/iss02.cenir/analyse/meeg/CYBERSART/"
# Defining the paths for saving results and raw data
derivatives_folder = os.path.join(root, "derivatives_nico")
# bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")


subjects = [str(i) if i > 9 else "0"+str(i) for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3','Sart4']
data = "eeg"

# Loop through conditions, create evoked objects, and combine
stimulus_condition = ['go', 'nogo']
response_condition = ['correct', 'incorrect']
# mind_condition = ['on-task', 'about-task', 'distracted', 'deliberate', 'spontaneous', 'blank', 'asleep']
# confidence_condition = ['a little confident', 'somewhat confident',  'very confident', 'completely confident']
# immersion_condition = ['a little immersed', 'somewhat immersed', 'very immersed' ,'completely immersed']

def filter_epochs_by_distance_to_probe(epochs, n):
    # Get the event information from the epochs object
    events = epochs.events  # This gives you an array with event information
    event_id = epochs.event_id  # Get the mapping of event labels to IDs
    
    # Create a list of the event codes you want to filter by
    filtered_event_ids = []

    # Loop through the event IDs and apply the filtering logic
    for event_name, event_code in event_id.items():
        # Split the event name by '/' and extract the "lasting,going" part
        parts = event_name.split('/')
        if len(parts) > 3:
            lasting_going = parts[3]  # Get the "lasting,going" sequence
            first_number = int(lasting_going.split(',')[0])  # Extract the first number
            # If the first number is less than n, add the event code to the filtered list
            if first_number <= n:
                filtered_event_ids.append(event_code)

    # Now we need to find the indices of epochs that have the filtered event codes
    event_indices = np.isin(events[:, 2], filtered_event_ids)  # Check if event code is in filtered list

    # Filter epochs based on the event indices
    filtered_epochs = epochs[event_indices]

    return filtered_epochs






#%%
for subject in subjects:
    for task in tasks:
        print(f"Creating evokeds for subject {subject} for {task}")
        
        # Load the epochs and events
        epochs, events = read_epochs(
            derivatives_folder,
            subject,
            task,
            data,
            desc="autoPreproc",
        )
        
        rereference_epochs = epochs.set_eeg_reference(ref_channels=['TP9', 'TP10'],)

        # Filter the epochs based on distance to probe
        filtered_epochs = filter_epochs_by_distance_to_probe(rereference_epochs, 6)

        evokeds = []  # List to store evoked responses for all conditions

        # Loop through all combinations of conditions
        for stim in stimulus_condition:
            for resp in response_condition:
                # Define the event string corresponding to this condition
                event_str = f'{stim}/{resp}'
                # Select epochs for this specific condition
                selected_epochs = filtered_epochs[event_str]

                # Compute the evoked (average) response
                evoked = selected_epochs.average()
                save_evokeds(evokeds, derivatives_folder, subject, task, data)
                        


#%%
# Main loop to generate and save evokeds for each subject/session/task combination
for subject in subjects:
    for task in tasks:
        print(f"Creating evokeds for subject {subject} for {task}")
        try:
            # Load the epochs and events
            epochs, events = read_epochs(
                derivatives_folder,
                subject,
                task,
                data,
                desc="autoPreprocParallel",
            )
            
            rereference_epochs = epochs.set_eeg_reference(ref_channels=['TP9', 'TP10'],)

            # Filter the epochs based on distance to probe
            filtered_epochs = filter_epochs_by_distance_to_probe(rereference_epochs, 6)

            evokeds = []  # List to store evoked responses for all conditions

            # Loop through all combinations of conditions
            for stim in stimulus_condition:
                for resp in response_condition:
                    for mind in mind_condition:
                        for conf in confidence_condition:
                            for immersion in immersion_condition:
                                # Define the event string corresponding to this condition
                                event_str = f'{stim}/{resp}/{mind}/{conf}/{immersion}'
                                try:
                                    # Select epochs for this specific condition
                                    selected_epochs = filtered_epochs[event_str]

                                    # Compute the evoked (average) response
                                    evoked = selected_epochs.average()

                                    # Resample the evoked to 500 Hz
                                    evoked.resample(500)

                                    # Shift time axis so that tmin = -0.3 seconds
                                    desired_tmin = -0.3
                                    current_tmin = evoked.times[0]
                                    shift_amount = desired_tmin - current_tmin
                                    evoked.shift_time(shift_amount, relative=True)

                                    # Crop the evoked to the desired time range
                                    evoked.crop(tmin=-0.3, tmax=1.2)

                                    # Append evoked object to the list
                                    # print(evoked.times[0], evoked.times[-1])
                                    evokeds.append(evoked)

                                except KeyError:
                                    continue

            # Save all evokeds in a single file
            if evokeds:
                save_evokeds(evokeds, derivatives_folder, subject, task, data)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue



# %%
#  # Load the epochs and events
epochs, events = read_epochs(
    derivatives_folder,
    '06',
    'Sart1',
    'eeg',
    desc="autoPreproc",
)
# %%
