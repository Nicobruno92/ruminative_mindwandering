import os
import json
import glob
import numpy as np
import pandas as pd
import pickle


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


def combine_events_data_to_csv(tsv_path, json_path, output_csv_path):
    """
    Combines event information from TSV and JSON files into a single CSV file.
    
    Parameters
    ----------
    tsv_path : str
        Path to the TSV file containing event data
    json_path : str
        Path to the JSON file containing event ID mappings
    output_csv_path : str
        Path to save the combined CSV file
        
    Returns
    -------
    str
        Path to the saved CSV file
    """
    # Read TSV file
    events_df = pd.read_csv(tsv_path, sep='\t')
    
    # Read JSON file
    with open(json_path, 'r') as f:
        events_json = json.load(f)
    
    # Extract event_id mapping from JSON
    event_id = events_json.get('description', {}).get('event_id', {})
    if not event_id and isinstance(events_json, dict):
        # If the JSON structure is different, try to get event_id directly
        event_id = events_json
    
    # Create a new column for event_id
    events_df['event_id'] = None
    
    # Map event descriptions to their IDs
    for i, row in events_df.iterrows():
        desc = row['description']
        if desc in event_id:
            events_df.loc[i, 'event_id'] = event_id[desc]
    
    # Add a column for trial information relative to probe
    # This combines the probe number and the trial count in one readable format
    if 'probe_number' in events_df.columns and 'trial_count' in events_df.columns:
        events_df['trial_info'] = events_df.apply(
            lambda row: f"probe{row['probe_number'].replace('probe', '')}_{row['trial_count']}" 
            if pd.notnull(row['probe_number']) and pd.notnull(row['trial_count']) 
            else None, 
            axis=1
        )
    
    # Save the combined data to CSV
    events_df.to_csv(output_csv_path, index=False)
    print(f"Combined events data saved to: {output_csv_path}")
    
    return output_csv_path

def save_events_combined(events_df, events_metadata, events_fname, create_combined_csv=True):
    """
    Save processed events data to TSV, JSON files, and optionally a combined CSV.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Processed events DataFrame
    events_metadata : dict
        Updated events metadata
    events_fname : str
        Base filename for the events files (without extension)
    create_combined_csv : bool, optional
        Whether to create a combined CSV file (default is True)
        
    Returns
    -------
    None
    """
    # For TSV, only save basic event information for BIDS compliance
    basic_events_df = events_df[['onset', 'duration', 'description']].copy()
    basic_events_df.to_csv(events_fname, sep='\t', index=False)
    print(f"BIDS-compliant events TSV saved at: {events_fname}")
    
    # Save the events metadata to JSON
    events_metadata_path = events_fname.replace('.tsv', '.json')
    
    # Extract just the event_id for the JSON file to keep it simple
    simple_metadata = {
        "onset": {"Description": "Event onset", "Units": "seconds"},
        "duration": {"Description": "Event duration", "Units": "seconds"},
        "description": {"Description": "Event description", "event_id": events_metadata.get("description", {}).get("event_id", {})}
    }
    
    with open(events_metadata_path, 'w') as f:
        json.dump(simple_metadata, f, indent=4)
    print(f"Events metadata saved at: {events_metadata_path}")
    
    # Create combined CSV with all processed information if requested
    if create_combined_csv:
        output_csv_path = events_fname.replace('.tsv', '_combined.csv')
        # Directly save the full processed DataFrame to CSV
        events_df.to_csv(output_csv_path, index=False)
        print(f"Combined events data saved at: {output_csv_path}")

def save_epoched_bids(epoched_data, root_path, subject, task, data, desc, events, event_id, create_combined_csv=True):
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
    events : np.ndarray
        Events array with columns [onset, duration, id]
    event_id : dict
        Mapping of event descriptions to numerical IDs
    create_combined_csv : bool, optional
        Whether to create a combined CSV file (default is True)
    """
    if not isinstance(epoched_data, mne.Epochs):
        raise ValueError("epoched_data must be an instance of mne.Epochs")
    
    # Create BIDS path
    bids_fname = make_bids_basename(subject=subject, task=task, suffix=data, extension='.fif', desc=desc)
    bids_directory = os.path.join(root_path, f"sub-{subject}", data)
    # Ensure the directory exists
    if not os.path.exists(bids_directory):
        os.makedirs(bids_directory)
    
    bids_path = os.path.join(bids_directory, bids_fname)
    # Save the epoched data
    epoched_data.save(bids_path, overwrite=True)
    
    # Create basic events DataFrame
    events_fname = os.path.join(bids_directory, make_bids_basename(subject=subject, task=task, suffix='events', extension='.tsv', desc=desc))
    events_df = pd.DataFrame(events, columns=['onset', 'duration', 'description'])
    
    # Map numerical IDs back to descriptions
    id_to_desc = {id_num: desc for desc, id_num in event_id.items()}
    events_df['description'] = events_df['description'].map(id_to_desc)
    
    # Create basic event metadata
    events_metadata = {
        "onset": {"Description": "Event onset", "Units": "seconds"},
        "duration": {"Description": "Event duration", "Units": "seconds"},
        "description": {"Description": "Event description", "event_id": event_id}
    }
    
    # Process events to add detailed columns
    processed_df, updated_metadata = process_bids_events(events_df, events_metadata, event_id)
    
    # Save the processed events
    save_events_combined(processed_df, updated_metadata, events_fname, create_combined_csv)

    # Create JSON sidecar metadata for the epochs
    sidecar_json_fname = bids_path.replace('.fif', '.json')
    json_metadata = {
        "TaskName": task,
        "Manufacturer": "Brain Products",
        "RecordingType": "epoched",
        "SamplingFrequency": epoched_data.info['sfreq'],
        "PowerLineFrequency": epoched_data.info['line_freq'],
        "SoftwareFilters": str(epoched_data.info['highpass']) + '-' + str(epoched_data.info['lowpass']) + ' Hz',
        "EEGReference": "n/a",  # Removed duplicated field, kept the safer "n/a" value
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

def process_bids_events(events_df, events_metadata, event_id):
    """
    Process BIDS events data by adding additional columns based on event descriptions.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        DataFrame with columns ['onset', 'duration', 'description']
    events_metadata : dict
        Metadata dictionary for the events
    event_id : dict
        Mapping of event descriptions to numerical IDs
        
    Returns
    -------
    pd.DataFrame
        Processed events DataFrame with additional columns
    dict
        Updated events metadata with new column descriptions
    """
    # Make a copy to avoid modifying the original
    df = events_df.copy()
    
    # Create columns for each part of the description
    df['trial_type'] = None
    df['correctness'] = None
    df['onoff'] = None
    df['selfother'] = None
    df['valence'] = None
    df['time'] = None
    df['confidence'] = None
    df['average'] = None
    df['distance_to_probe'] = None
    df['probe_number'] = None
    df['trial_count'] = None
    
    # Parse the descriptions
    for i, desc in enumerate(df['description']):
        if isinstance(desc, str) and '/' in desc:
            parts = desc.split('/')
            
            # Basic trial type (first component)
            if parts[0] in ['go', 'nogo']:
                df.loc[i, 'trial_type'] = parts[0]
                
                # Process additional parts if they exist
                if len(parts) > 1:
                    # Correctness (second component)
                    if parts[1] in ['correct', 'incorrect']:
                        df.loc[i, 'correctness'] = parts[1]
                    
                    # Process probe-related information
                    for part in parts:
                        if 'onoff' in part:
                            df.loc[i, 'onoff'] = part
                        elif 'selfother' in part:
                            df.loc[i, 'selfother'] = part
                        elif 'valence' in part:
                            df.loc[i, 'valence'] = part
                        elif 'time' in part:
                            df.loc[i, 'time'] = part
                        elif 'confidence' in part:
                            df.loc[i, 'confidence'] = part
                        elif 'average' in part:
                            df.loc[i, 'average'] = part
                        elif part.startswith('probe') and part[5:].isdigit():
                            df.loc[i, 'probe_number'] = part
                        elif part.lstrip('-').isdigit() and len(parts) >= 3:
                            # Distance is typically third from end
                            df.loc[i, 'distance_to_probe'] = part
                        elif part.isdigit() and i > 0 and parts[0] in ['go', 'nogo']:
                            # Trial count is typically the last number
                            df.loc[i, 'trial_count'] = part
    
    # Extract numeric values from category columns
    for col in ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']:
        df[f'{col}_value'] = df[col].str.extract(r'(\d+)').astype('float')
    
    # Add metadata for new columns
    updated_metadata = events_metadata.copy()
    updated_metadata.update({
        "trial_type": {"Description": "Type of trial (go or nogo)"},
        "correctness": {"Description": "Whether the response was correct or incorrect"},
        "onoff": {"Description": "On-task vs off-task rating"},
        "onoff_value": {"Description": "Numeric value for on-task vs off-task rating"},
        "selfother": {"Description": "Self vs other-related thoughts rating"},
        "selfother_value": {"Description": "Numeric value for self vs other-related thoughts rating"},
        "valence": {"Description": "Positive vs negative valence rating"},
        "valence_value": {"Description": "Numeric value for positive vs negative valence rating"},
        "time": {"Description": "Past vs future time orientation rating"},
        "time_value": {"Description": "Numeric value for past vs future time orientation rating"},
        "confidence": {"Description": "Confidence in responses rating"},
        "confidence_value": {"Description": "Numeric value for confidence in responses rating"},
        "average": {"Description": "Average distraction rating"},
        "average_value": {"Description": "Numeric value for average distraction rating"},
        "distance_to_probe": {"Description": "Number of trials before/after a thought probe"},
        "probe_number": {"Description": "Identifier for the associated thought probe"},
        "trial_count": {"Description": "Sequential count of trials since last thought probe"}
    })
    
    return df, updated_metadata

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
    # bids_fname = make_bids_basename(subject=subject, task=task, suffix=data, extension='.fif', desc=desc)
    bids_fname = f"sub-{subject}_task-{task}_desc-{desc}_epo.fif"
    bids_directory = os.path.join(root_path, f"sub-{subject}", data)
    
    # Full file path to the epochs file
    bids_path = os.path.join(bids_directory, bids_fname)
    
    if not os.path.exists(bids_path):
        raise FileNotFoundError(f"File not found: {bids_path}")
    
    # Load the epochs file
    epochs = mne.read_epochs(bids_path, preload=True)
    
    # Load events
    # events_fname = make_bids_basename(subject=subject,task=task, suffix='events', extension='.tsv', desc=desc)
    events_fname = f"sub-{subject}_task-{task}_desc-{desc}_epo_events.tsv"
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
    # Preferred BIDS-compliant filename
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

    # Fallback to legacy filename if BIDS-compliant file does not exist
    if not os.path.exists(evoked_fname) and desc is not None:
        legacy_fname = os.path.join(
            derivatives_folder,
            f"sub-{subject}",
            data,
            f"sub-{subject}_task-concat_{data}_evokeds_{desc}.fif"
        )
        if os.path.exists(legacy_fname):
            evoked_fname = legacy_fname

    # Load the evoked data
    evokeds = mne.read_evokeds(evoked_fname)
    return evokeds

def save_psd_epochs(psds, derivatives_folder, subject, data, desc=None):
    """
    Saves PSDs and corresponding frequencies to a specified folder structure in one file using pickle.

    Parameters:
    -----------
    psds : np.ndarray
        PSD data (e.g., n_epochs x n_channels x n_freqs).
    freqs : np.ndarray
        Array of frequency values corresponding to PSDs.
    derivatives_folder : str
        Root folder where the data will be saved.
    subject : str
        Subject ID (e.g., '12').
    data : str
        Data type (e.g., 'eeg').
    desc : str, optional
        Descriptor for the file (e.g., 'mean', 'median').

    Returns:
    --------
    str
        Full path to the saved file.
    """
    # Create the output folder structure
    output_folder = os.path.join(
        derivatives_folder,
        f"sub-{subject}",
        data,
    )
    os.makedirs(output_folder, exist_ok=True)

    # Create the file name
    psd_fname = os.path.join(
        output_folder,
        f"sub-{subject}_psds{f'_desc-{desc}' if desc else ''}.pkl"
    )

    # Save the data using pickle
    with open(psd_fname, "wb") as f:
        pickle.dump(psds, f)

    return psd_fname

def read_psd_epochs(derivatives_folder, subject, data, desc=None):
    """
    Loads PSDs and corresponding frequencies from a pickle file.

    Parameters:
    -----------
    psd_fname : str
        Full path to the pickle file containing PSDs and frequencies.

    Returns:
    --------
    dict
        Dictionary with keys "psds" and "freqs" containing the PSD data and frequency array.
    """
        # Create the output folder structure
    output_folder = os.path.join(
        derivatives_folder,
        f"sub-{subject}",
        data,
    )
    os.makedirs(output_folder, exist_ok=True)

    # Create the file name
    psd_fname = os.path.join(
        output_folder,
        f"sub-{subject}_psds{f'_desc-{desc}' if desc else ''}.pkl"
    )

    with open(psd_fname, "rb") as f:
        data = pickle.load(f)
    return data


def save_tfr_epochs(epochs_tfr, derivatives_folder, subject, data, desc=None):
    """
    Saves PSDs and corresponding frequencies to a specified folder structure in one file using pickle.

    Parameters:
    -----------
    tfr : np.ndarray
    freqs : np.ndarray
        Array of frequency values corresponding to PSDs.
    derivatives_folder : str
        Root folder where the data will be saved.
    subject : str
        Subject ID (e.g., '12').
    data : str
        Data type (e.g., 'eeg').
    desc : str, optional
        Descriptor for the file (e.g., 'mean', 'median').

    Returns:
    --------
    str
        Full path to the saved file.
    """
    # Create the output folder structure
    output_folder = os.path.join(
        derivatives_folder,
        f"sub-{subject}",
        data,
    )
    os.makedirs(output_folder, exist_ok=True)

    # Create the file name
    tfr_fname = os.path.join(
        output_folder,
        f"sub-{subject}epochsTFR{f'_desc-{desc}' if desc else ''}"
    )
    
    epochs_tfr.save(tfr_fname, overwrite=True)

    return tfr_fname

def read_tfr_epochs(derivatives_folder, subject, data, desc=None):
    """
    Loads PSDs and corresponding frequencies from a pickle file.

    Parameters:
    -----------
    psd_fname : str
        Full path to the pickle file containing PSDs and frequencies.

    Returns:
    --------
    dict
        Dictionary with keys "psds" and "freqs" containing the PSD data and frequency array.
    """
    # Create the output folder structure
    output_folder = os.path.join(
        derivatives_folder,
        f"sub-{subject}",
        data,
    )

    # Create the file name
    tfr_fname = os.path.join(
        output_folder,
        f"sub-{subject}epochsTFR{f'_desc-{desc}' if desc else ''}"
    )

    data = mne.time_frequency.read_tfrs(tfr_fname)
    return data

