import os
import json
import glob
import pandas as pd

import mne
from mne_bids import BIDSPath, write_raw_bids
from mne_bids.utils import _write_json, _write_tsv

#### BIDS ####

def read_raw_custom(subject, task, root):
    """
    Reads raw EEG data from a FieldTrip-preprocessed .mat file without requiring the date parameter.

    Parameters:
    - subject (str): Subject identifier (e.g., 'S002')
    - task (str): Task identifier (e.g., 'Sart1', 'Sart2', etc.)
    - root (str): Root directory of the data (default is 'DATA')
    - data_name (str): Name of the data variable in the .mat file (default is 'data')
    - original_data_path (str): Path to the original raw EEG file to extract info.

    Returns:
    - raw (mne.io.Raw): The raw EEG data with the correct info object.
    """
    # Construct the directory path for the subject's EEG data
    eeg_dir = os.path.join(root, f'sub-{subject}', 'eeg')
    
    # Create a search pattern for the .mat files matching the task
    pattern = f'CYBERSART_*_{task}.vhdr'
    search_path = os.path.join(eeg_dir, pattern)

    # Find all files matching the pattern
    matching_files = glob.glob(search_path)

    if not matching_files:
        raise FileNotFoundError(f"No files found for subject {subject} and task {task} in {eeg_dir}")

    if len(matching_files) > 1:
        # If multiple files are found, decide how to handle this
        # For this example, we'll select the first one and print a warning
        print(f"Warning: Multiple files found for subject {subject} and task {task}. Using the first one.")

    # Select the first matching file
    eeg_filepath = matching_files[0]

    # Optional: Extract the date from the filename if needed
    filename = os.path.basename(eeg_filepath)
    # Filename format: 'CYBERSART_YYYY-MM-DD_Task_eeg.mat'
    parts = filename.split('_')
    if len(parts) >= 3:
        date = parts[1]  # Extract the date part
    else:
        date = 'Unknown'

    print(f"Reading data for subject {subject}, task {task}, date {date}")

    # Read the FieldTrip .mat data using the extracted info
    raw = mne.io.read_raw_brainvision((eeg_filepath))
    
    raw.set_meas_date(date)

    return raw

#### BIDS ####

def make_bids_basename(subject, task, suffix, extension, desc = None):
    """
    Create a BIDS-compliant basename.
    
    Parameters:
    subject (str): Subject ID
    task (str): Task name
    suffix (str): Data suffix (e.g., 'eeg', 'gaze')
    extension (str): File extension
    
    Returns:
    str: BIDS-compliant basename
    """
    if desc == None:
        return f"sub-{subject}_task-{task}_{suffix}{extension}"
    else:
        return f"sub-{subject}__task-{task}_desc-{desc}_{suffix}{extension}"


def save_raw_bids_compliant(subject, task, data_type, raw, root_folder):
    """
    Save raw data in BIDS-compliant format using the BrainVision format.
    
    Parameters:
    subject (str): Subject ID
    task (str): Task name
    data_type (str): Data type (e.g., 'eeg', 'meg')
    raw (mne.io.Raw): Raw data to be saved
    root_folder (str): Root folder for BIDS data
    
    """
    if data_type == 'eeg':
        # Define the BIDS path using mne_bids
        bids_path = BIDSPath(subject=subject,
                            task=task,
                            datatype=data_type,
                            root=root_folder,
                            suffix=data_type,
                            extension='.vhdr',
                            check=True)
        
        # Use the bids_path to save your data
        write_raw_bids(raw, bids_path, format='BrainVision', allow_preload=True, overwrite=True)
        print(f"Data saved at BIDS path: {bids_path}")

def save_epoched_bids(epoched_data, root_path, subject, task, data, desc, events, event_id):
    """
    Save epoched data in BIDS format.
    
    Parameters
    ----------
    epoched_data : mne.Epochs
        The epoched data to be saved.
    root_path : str
        The root path of the BIDS dataset.
    subject : str
        Subject ID.
    task : str
        Task name.
    desc : str
        Descriptor for the dataset.
    """
    if not isinstance(epoched_data, mne.Epochs):
        raise ValueError("epoched_data must be an instance of mne.Epochs")
    
    # Create BIDS path
    bids_fname = make_bids_basename(subject=subject, task=task, suffix=data, extension='.fif', desc = desc,)
    bids_directory = os.path.join(root_path, f"sub-{subject}", data)
    # Ensure the directory exists
    if not os.path.exists(bids_directory):
        os.makedirs(bids_directory)
    
    bids_path = os.path.join(bids_directory, bids_fname)
    # Save the epoched data
    epoched_data.save(bids_path, overwrite=True)
    
    # Create events file
    event_id = event_id
    events_fname = os.path.join(bids_directory, make_bids_basename(subject=subject, task=task, suffix= 'events', extension='.tsv', desc = desc))
    events_df = pd.DataFrame(events, columns=['onset', 'duration', 'description'])
    events_df.to_csv(events_fname, sep='\t', index=False)
    print(f"Events saved at: {events_fname}")
    
    # Event metadata
    events_metadata = {
        "onset": {"Description": "Event onset", "Units": "seconds"},
        "duration": {"Description": "Event duration", "Units": "seconds"},
        "description": {"Description": "Event description", "event_id": event_id}
    }
    events_metadata_path = events_fname.replace('.tsv', '.json')
    with open(events_metadata_path, 'w') as f:
        json.dump(events_metadata, f, indent=4)
    print(f"Events metadata saved at: {events_metadata_path}")

    # Create JSON sidecar metadata for the epochs
    sidecar_json_fname = bids_path.replace('.fif', '.json')
    json_metadata = {
        "TaskName": task,
        "Manufacturer": "Brain Products",
        "RecordingType": "epoched",
        "SamplingFrequency": epoched_data.info['sfreq'],
        "PowerLineFrequency": epoched_data.info['line_freq'],
        "SoftwareFilters": str(epoched_data.info['highpass']) + '-' + str(epoched_data.info['lowpass']) + ' Hz',
        "EEGReference": epoched_data.info['ch_names'][0],  # Assuming the first channel as reference; adjust as necessary
        "EEGReference": "n/a",
        "EEGGround": "n/a",
        "EEGPlacementScheme": "based on the extended 10/20 system",
        "EEGChannelCount": 31,
        "EOGChannelCount": 0,
        "ECGChannelCount": 0,
        "EMGChannelCount": 0,
        "MiscChannelCount": 0,
        "TriggerChannelCount": 0,
        "ICA": True, 
        "Epoch_count": len(epoched_data),
        
    }

    with open(sidecar_json_fname, 'w') as f:
        json.dump(json_metadata, f, indent=4)
    print(f"Epoch object metadata saved at: {sidecar_json_fname}")

def read_epochs(root_path, subject, task, data, desc=None):
    """
    Read epoched data from a BIDS-compliant file.
    
    Parameters:
    ----------
    root_path : str
        Root path to the BIDS dataset.
    subject : str
        Subject ID.
    task : str
        Task name.
    data : str
        Data type, e.g., 'eeg' or 'gaze'.
    desc : str, optional
        Description for the data (if available).
    
    Returns:
    --------
    epochs : mne.Epochs
        The loaded epochs object.
    events : np.ndarray
        The events array associated with the epochs.
    """
    # Construct BIDS-compliant file name
    bids_fname = make_bids_basename(subject=subject, task=task, suffix=data, extension='.fif', desc=desc)
    bids_directory = os.path.join(root_path, f"sub-{subject}", data)
    
    # Full file path to the epochs file
    bids_path = os.path.join(bids_directory, bids_fname)
    
    if not os.path.exists(bids_path):
        raise FileNotFoundError(f"File not found: {bids_path}")
    
    # Load the epochs file
    epochs = mne.read_epochs(bids_path, preload=True)
    
    # Load events
    events_fname = make_bids_basename(subject=subject,task=task, suffix='events', extension='.tsv', desc=desc)
    events_path = os.path.join(bids_directory, events_fname)
    
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file not found: {events_path}")
    
    # Read events file
    events_df = pd.read_csv(events_path, sep='\t')
    events = events_df[['onset', 'duration', 'description']].values
    
    print(f"Loaded epochs from: {bids_path}")
    print(f"Loaded events from: {events_path}")
    
    return epochs, events


def save_evokeds(evokeds, derivatives_folder, subject, data, desc=None):
    """
    Saves a list of evoked objects to a specified folder structure in one file.

    Parameters:
    evokeds (list of mne.Evoked): List of evoked objects to save.
    derivatives_folder (str): Root folder where the data will be saved.
    subject (str): The subject ID (e.g., '12').
    task (str): The task name (e.g., 'sartauditiva').
    data (str): The type of data (e.g., 'eeg').

    Returns:
    str: The full path to the saved file.
    """
    # Define path for saving the evoked data
    output_folder = os.path.join(
        derivatives_folder, 
        f"sub-{subject}", 
        data,
    )
    
    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if desc == None:
        evoked_fname = os.path.join(
            output_folder, 
            f"sub-{subject}_evokeds-ave.fif"
        )
    else:
        evoked_fname = os.path.join(
            output_folder, 
            f"sub-{subject}_evokeds_desc-{desc}-ave.fif"
        )
    
    # Save all the evoked objects into one file
    mne.write_evokeds(evoked_fname, evokeds, overwrite=True)
    
    return evoked_fname  # Return the file path for confirmation or future use

def load_evokeds(derivatives_folder, subject, data, desc=None):
    """
    Loads evoked responses from a saved evoked file.

    Parameters:
    derivatives_folder (str): Root folder where the data is saved.
    subject (str): The subject ID (e.g., '12').
    task (str): The task name (e.g., 'sartauditiva').
    data (str): The type of data (e.g., 'eeg').

    Returns:
    list of mne.Evoked: A list of evoked objects for different conditions.
    """
    if desc == None:
        evoked_fname = os.path.join(
            derivatives_folder, 
            f"sub-{subject}", 
            data,
            f"sub-{subject}_evokeds-ave.fif"
        )
    else:
        evoked_fname = os.path.join(
            derivatives_folder, 
            f"sub-{subject}", 
            data,
            f"sub-{subject}_evokeds_desc-{desc}-ave.fif"
        )

    # Load the evoked data
    evokeds = mne.read_evokeds(evoked_fname)
    
    return evokeds