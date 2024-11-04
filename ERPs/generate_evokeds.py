import os
import mne
from joblib import Parallel, delayed


import sys
sys.path.insert(0, './')
from utils.bids_compliance import read_epochs, save_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

def classify_all_metrics(subject_epochs, metrics=["mean", "median", "quartiles", "teriles", "highlow"]):
    """
    Classify epochs with multiple metrics to save computation time.
    
    Parameters:
    -----------
    subject_epochs : mne.Epochs
        The MNE Epochs object after loading and reference setting.
    metrics : list
        List of metrics to classify epochs by.
        
    Returns:
    --------
    classified_epochs_dict : dict
        Dictionary with metric names as keys and classified epochs as values.
    """
    classified_epochs_dict = {}
    for metric in metrics:
        classified_epochs_dict[metric] = classify_onoff_epochs(subject_epochs.copy(), split=metric)
    return classified_epochs_dict

def process_subject_for_metrics(subject, root, tasks, metrics, data="eeg", ref_channels=['TP9', 'TP10'], distance=5):
    """
    Process and classify epochs for multiple metrics for a single subject.
    
    Parameters:
    -----------
    subject : str
        Subject ID.
    root : str
        Root directory for data.
    tasks : list
        List of task names.
    metrics : list
        List of metrics to classify epochs by.
    """
    derivatives_folder = os.path.join(root, "derivatives_nico")
    epochs_tasks = []
    
    # Load epochs for all tasks
    for task in tasks:
        try:
            epochs, events = read_epochs(derivatives_folder, subject, task, data, desc="autoPreproc")
            reref_epochs = epochs.set_eeg_reference(ref_channels=ref_channels)
            epochs_tasks.append(reref_epochs.copy())
        except Exception as e:
            print(f"Skipping {subject} {task}: {e}")

    if not epochs_tasks:
        print(f"No data for subject {subject}")
        return None  # Skip if no data

    # Concatenate epochs across tasks
    try:
        epochs_concat = mne.concatenate_epochs(epochs_tasks)
        filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, distance)
    except Exception as e:
        print(f"Failed concatenating or filtering epochs for subject {subject}: {e}")
        return None

    # Classify for all metrics
    classified_epochs_dict = classify_all_metrics(filtered_epochs, metrics)
    
    return classified_epochs_dict

def generate_and_save_evokeds_for_metrics(classified_epochs_dict, subject, derivatives_folder, data, metrics, stimulus_condition=['go', 'nogo'], response_condition=['correct', 'incorrect'], mind_condition=['ontask', 'offtask']):
    """
    Generate and save evokeds for different metrics without reloading epochs.
    """
    for metric, epochs_classified in classified_epochs_dict.items():
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
                        print(f"Failed for condition {event_str} with metric {metric}: {e}")
                        continue

        if evokeds:
            save_evokeds(evokeds, derivatives_folder, subject, data, desc=metric)
            print(f"Evoked responses saved for subject {subject} using metric {metric}")

def process_and_save_subject(subject, root, tasks, metrics, data, ref_channels, distance):
    """
    Wrapper function for processing a single subject and saving the results.
    """
    derivatives_folder = os.path.join(root, "derivatives_nico")
    classified_epochs_dict = process_subject_for_metrics(subject, root, tasks, metrics, data, ref_channels, distance)

    if classified_epochs_dict:
        generate_and_save_evokeds_for_metrics(classified_epochs_dict, subject, derivatives_folder, data, metrics)
        
def process_subjects_parallel(root, subjects, tasks, metrics=["mean", "median", "quartiles", "teriles", "highlow"], data="eeg", ref_channels=['TP9', 'TP10'], distance=5, n_jobs=4):
    """
    Parallelized version to generate and save evoked responses for multiple subjects and metrics.
    """
    Parallel(n_jobs=n_jobs)(
        delayed(process_and_save_subject)(subject, root, tasks, metrics, data, ref_channels, distance)
        for subject in subjects
    )


if __name__ == '__main__':
    root = "//l2export/iss02.cenir/analyse/meeg/CYBERSART/"
    subjects = [f"{i:02}" for i in range(2, 43)]
    tasks = ['Sart1', 'Sart2', 'Sart3', 'Sart4']
    metrics = ['mean', 'median', 'quartiles', 'teriles', 'highlow']
    
    process_subjects_parallel(root, subjects, tasks, metrics, data="eeg", ref_channels=['TP9', 'TP10'], distance=5, n_jobs=4)
