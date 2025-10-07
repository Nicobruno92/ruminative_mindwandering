import os
import mne
from joblib import Parallel, delayed
import pickle

import sys
sys.path.insert(0, './')
from utils.bids_compliance import read_epochs, save_evokeds
from utils.analysis_helpers import filter_epochs_by_distance_to_probe, classify_onoff_epochs

def classify_epochs_sequentially(subject_epochs, metrics=["mean", "median", "quartiles", "tertiles", "highlow"]):
    """
    Classify epochs sequentially to avoid copying full Epochs for every metric.
    
    Parameters:
    -----------
    subject_epochs : mne.Epochs
        The MNE Epochs object after loading and reference setting.
    metrics : list
        List of metrics to classify epochs by.
        
    Returns:
    --------
    classified_epochs_dict : dict
        Dictionary with metrics as keys and classified epochs dictionaries as values.
    """
    classified_epochs_dict = {}
    
    for metric in metrics:
        print(f"Processing metric: {metric}")
        
        # Classify epochs for this metric
        classified_epochs = classify_onoff_epochs(subject_epochs, split=metric)
        classified_epochs_dict[metric] = classified_epochs
    
    return classified_epochs_dict

def process_subject_for_metrics(subject, root, tasks, metrics, data="eeg", ref_channels=['TP9', 'TP10'], distance=5):
    """
    Process and classify epochs for multiple metrics for a single subject.
    Fixed to handle event ID and channel mismatches across tasks.
    
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
    derivatives_folder = os.path.join(root, "BIDS", "derivatives")
    epochs_tasks = []
    
    # Load epochs for all tasks
    for task in tasks:
        try:
            epochs, events = read_epochs(derivatives_folder, subject, task, data, desc="evoked_epo")
            # Robust referencing: use specified ref channels if present; otherwise fall back to average reference
            if all(ch in epochs.ch_names for ch in ref_channels):
                epochs = epochs.set_eeg_reference(ref_channels=ref_channels)
            else:
                print(f"  Reference channels {ref_channels} missing for sub-{subject} {task}. Using average reference.")
                epochs = epochs.set_eeg_reference(ref_channels='average')
            epochs_tasks.append(epochs.copy())
        except Exception as e:
            print(f"Skipping {subject} {task}: {e}")

    if not epochs_tasks:
        print(f"No data for subject {subject}")
        return None  # Skip if no data

    # Fix concatenation issues
    try:
        # Ensure all epochs have the same channels
        common_channels = None
        for epochs in epochs_tasks:
            if common_channels is None:
                common_channels = set(epochs.ch_names)
            else:
                common_channels = common_channels.intersection(set(epochs.ch_names))
        
        if not common_channels:
            print(f"No common channels found for subject {subject}")
            return None
        
        common_channels = sorted(list(common_channels))
        
        # Pick only common channels for all epochs
        for i, epochs in enumerate(epochs_tasks):
            epochs_tasks[i] = epochs.pick_channels(common_channels, ordered=True)
        
        # Harmonize event IDs by creating a mapping
        all_event_names = set()
        for epochs in epochs_tasks:
            all_event_names.update(epochs.event_id.keys())
        
        # Create a unified event_id mapping
        unified_event_id = {name: i+1 for i, name in enumerate(sorted(all_event_names))}
        
        # Update each epochs object to use the unified mapping
        for epochs in epochs_tasks:
            # Create new events array with unified codes
            new_events = epochs.events.copy()
            old_to_new_mapping = {}
            for event_name, old_code in epochs.event_id.items():
                new_code = unified_event_id[event_name]
                old_to_new_mapping[old_code] = new_code
            
            # Update event codes in the events array
            for old_code, new_code in old_to_new_mapping.items():
                new_events[new_events[:, 2] == old_code, 2] = new_code
            
            # Update the epochs object
            epochs.events = new_events
            epochs.event_id = {name: unified_event_id[name] for name in epochs.event_id.keys()}
        
        # Now concatenate epochs across tasks
        epochs_concat = mne.concatenate_epochs(epochs_tasks)
        filtered_epochs = filter_epochs_by_distance_to_probe(epochs_concat, distance)
        
    except Exception as e:
        print(f"Failed concatenating or filtering epochs for subject {subject}: {e}")
        return None

    # Classify sequentially instead of copying for all metrics
    classified_epochs_dict = classify_epochs_sequentially(filtered_epochs, metrics)
    
    return classified_epochs_dict

def process_single_subject(subject, tasks, derivatives_folder, data, 
                         stimulus_condition=['go', 'nogo'], 
                         response_condition=['correct', 'incorrect'], 
                         mind_condition=['ontask', 'offtask']):
    """
    Process a single subject to generate evokeds for all metrics.
    """
    try:
        print(f"Processing subject {subject}...")
        
        # Load epochs for all tasks
        all_epochs = []
        for task in tasks:
            epochs_file = os.path.join(
                derivatives_folder, 
                f"sub-{subject}", 
                data, 
                f"sub-{subject}_task-{task}_{data}_concat_epochs.fif"
            )
            
            if os.path.exists(epochs_file):
                epochs = read_epochs(epochs_file, preload=True)
                # epochs.set_eeg_reference(ref_channels='average', projection=True)
                all_epochs.append(epochs)
            else:
                print(f"Epochs file not found: {epochs_file}")
                continue
        
        if not all_epochs:
            print(f"No valid epochs found for subject {subject}")
            return
        
        # Concatenate all epochs
        subject_epochs = mne.concatenate_epochs(all_epochs)
        
        # Classify epochs for all metrics sequentially
        classified_epochs_dict = classify_epochs_sequentially(
            subject_epochs, 
            metrics=["mean", "median", "quartiles", "tertiles", "highlow"]
        )
        
        # Generate and save evokeds for all metrics
        generate_and_save_evokeds_for_metrics(
            classified_epochs_dict, 
            subject, 
            derivatives_folder, 
            data, 
            ["mean", "median", "quartiles", "tertiles", "highlow"],
            stimulus_condition, 
            response_condition, 
            mind_condition,
            tasks  # Pass tasks to the function
        )
        
        print(f"Completed processing for subject {subject}")
        
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_and_save_evokeds_for_metrics(classified_epochs_dict, subject, derivatives_folder, data, metrics, stimulus_condition, response_condition, mind_condition, tasks):
    """
    Generate and save evokeds for different metrics without reloading epochs.
    Uses baseline correction on epochs before averaging.
    """
    for metric in metrics:
        if metric not in classified_epochs_dict:
            print(f"Metric {metric} not found in classified epochs for subject {subject}")
            continue
            
        classified_epochs = classified_epochs_dict[metric]
        print(f"Generating evokeds for metric: {metric}")
        evokeds = []
        total_attempts = 0
        successful_generations = 0
        
        # Iterate through each classification (e.g., 'low', 'high' for median split)
        for classification_name, epochs_for_classification in classified_epochs.items():
            print(f"Processing classification: {classification_name}")
            print(f"Total events in classification: {len(epochs_for_classification.event_id)}")
            
            for stim in stimulus_condition:
                for resp in response_condition:
                    total_attempts += 1
                    # Look for events that match the stimulus and response pattern
                    # The classification name is already incorporated in the epochs_for_classification
                    matching_events = []
                    for event_name in epochs_for_classification.event_id.keys():
                        # Check if the event contains the required stimulus and response components
                        if stim in event_name and resp in event_name:
                            matching_events.append(event_name)
                    
                    print(f"  Searching for {stim}/{resp}: found {len(matching_events)} matching events")
                    
                    if matching_events:
                        try:
                            # Use ALL matching events for this stim/resp within the classification
                            selected_epochs = epochs_for_classification[matching_events]
                            
                            if len(selected_epochs) > 0:
                                print(f"  Found {len(selected_epochs)} epochs for this condition (aggregated across {len(matching_events)} event types)")
                                
                                # Average epochs
                                evoked = selected_epochs.average(picks='eeg', method='median')
                                
                                # Crop to desired time window
                                evoked = evoked.crop(tmin=-0.1, tmax=0.8)
                                
                                # Map classification to mind wandering state for consistency
                                if classification_name in ['low', 'q1', 't1']:
                                    mind_state = 'ontask'
                                elif classification_name in ['high', 'q4', 't3']:
                                    mind_state = 'offtask'
                                else:
                                    mind_state = classification_name
                                
                                evoked.comment = f"{stim}/{resp}/{mind_state}/{classification_name}"
                                evokeds.append(evoked)
                                successful_generations += 1
                                
                                print(f"  ✓ Generated evoked for {evoked.comment}")
                                break  # Move to next stimulus/response combo
                            else:
                                print("  ✗ No epochs found after aggregation")
                        except Exception as e:
                            print(f"  ✗ Error aggregating events {matching_events[:3]}...: {e}")
                            continue
                    else:
                        print(f"  ✗ No matching events found for {stim}/{resp} in {classification_name}")
        
        print(f"Summary for {subject}-{metric}: {successful_generations}/{total_attempts} successful generations")
        
        # Save evokeds for this metric using BIDS-compliant helper so loading works downstream
        if evokeds:
            try:
                filepath = save_evokeds(evokeds, derivatives_folder, subject, data, desc=metric)
                print(f"✓ Saved {len(evokeds)} evokeds for {subject} - {metric}")
                print(f"Filepath: {filepath}")
            except Exception as e:
                print(f"✗ Failed to save evokeds for {subject} - {metric}: {e}")
        else:
            print(f"✗ No evokeds generated for {subject} - {metric}")

def process_and_save_subject(subject, root, tasks, metrics, data, ref_channels, distance):
    """
    Wrapper function for processing a single subject and saving the results.
    """
    derivatives_folder = os.path.join(root, "BIDS", "derivatives")
    classified_epochs_dict = process_subject_for_metrics(subject, root, tasks, metrics, data, ref_channels, distance)

    if classified_epochs_dict:
        generate_and_save_evokeds_for_metrics(
            classified_epochs_dict, 
            subject, 
            derivatives_folder, 
            data, 
            metrics,
            ['go', 'nogo'], 
            ['correct', 'incorrect'], 
            ['ontask', 'offtask'],
            tasks
        )
        
def process_subjects_parallel(root, subjects, tasks, metrics=["mean", "median", "quartiles", "tertiles", "highlow"], data="eeg", ref_channels=['TP9', 'TP10'], distance=5, n_jobs=4):
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
    metrics = ['mean', 'median', 'quartiles', 'tertiles', 'highlow']
    
    process_subjects_parallel(root, subjects, tasks, metrics, data="eeg", ref_channels=['TP9', 'TP10'], distance=5, n_jobs=4)
