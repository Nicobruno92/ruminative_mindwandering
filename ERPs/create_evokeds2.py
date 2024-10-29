import os
import mne
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, './')
from utils.bids_compliance import read_epochs, save_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

def process_subject(subject, root, tasks, data="eeg", ref_channels=['TP9', 'TP10'], distance=5, split="median"):
    """
    Process and save evoked responses for a single subject.
    
    Parameters are similar to `generate_save_evokeds`, but specific to a single subject.
    """
    derivatives_folder = os.path.join(root, "derivatives_nico")
    stimulus_condition = ['go', 'nogo']
    response_condition = ['correct', 'incorrect']
    mind_condition = ['ontask', 'offtask']
    epochs_tasks = []
    
    for task in tasks:
        try:
            epochs, events = read_epochs(derivatives_folder, subject, task, data, desc="autoPreproc")
            reref_epochs = epochs.set_eeg_reference(ref_channels=ref_channels)
            epochs_tasks.append(reref_epochs)
        except Exception as e:
            print(f"Skipping {subject} {task}: {e}")
    
    if not epochs_tasks:
        print(f"No data for subject {subject}")
        return  # Skip if no data
    
    try:
        epochs_concat = mne.concatenate_epochs(epochs_tasks)
        filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, distance)
        epochs_classified = classify_onoff_epochs(filtered_epochs, split=split)
        
        evokeds = []
        for stim in stimulus_condition:
            for resp in response_condition:
                for mind in mind_condition:
                    event_str = f"{stim}/{resp}/{mind}"
                    try:
                        selected_epochs = epochs_classified[event_str]
                        evoked = selected_epochs.average()
                        evoked.crop(tmin=-0.3, tmax=1.2)
                        evokeds.append(evoked)
                    except Exception as e:
                        print(f"Failed for condition {event_str}: {e}")
                        continue
        
        if evokeds:
            save_evokeds(evokeds, derivatives_folder, subject, data)
            print(f"Evoked responses saved for subject {subject}")
            
    except Exception as e:
        print(f"Failed processing for subject {subject}: {e}")

def generate_save_evokeds_parallel(root, subjects, tasks, data="eeg", ref_channels=['TP9', 'TP10'], distance=5, split="median", max_workers=4):
    """
    Parallelized version to generate and save evoked responses for multiple subjects.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_subject, subject, root, tasks, data, ref_channels, distance, split)
            for subject in subjects
        ]
        for future in futures:
            future.result()  # Wait for each task to complete

# Usage example:
# root = "//l2export/iss02.cenir/analyse/meeg/CYBERSART/"
root = "/network/lustre/iss02/cenir/analyse/meeg/CYBERSART/"
subjects = [f"{i:02}" for i in range(2, 43)]
tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']

generate_save_evokeds_parallel(root, subjects, tasks)
