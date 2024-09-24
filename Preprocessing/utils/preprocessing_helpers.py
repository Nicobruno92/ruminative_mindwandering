import os
import json
import numpy as np
import pandas as pd

import mne
from mne.preprocessing.eyetracking import Calibration

#### EEG ####

def set_chs_montage(raw):
    rename_dict = {
    'FP1': 'Fp1',
    'FP2': 'Fp2',
    'FZ': 'Fz',
    'CZ': 'Cz',
    'PZ': 'Pz'
    }

    # Rename the channels
    raw.rename_channels(rename_dict)

    channel_types = {
        'Fp1': 'eeg', 'Fp2': 'eeg', 
        'F3': 'eeg', 'F4': 'eeg', 
        'C3': 'eeg', 'C4': 'eeg', 
        'P3': 'eeg', 'P4': 'eeg', 
        'O1': 'eeg', 'O2': 'eeg', 
        'F7': 'eeg', 'F8': 'eeg', 
        'T7': 'eeg', 'T8': 'eeg', 
        'P7': 'eeg', 'P8': 'eeg', 
        'Fz': 'eeg', 'Cz': 'eeg', 
        'Pz': 'eeg', 'IO': 'misc', 
        'FC1': 'eeg', 'FC2': 'eeg', 
        'CP1': 'eeg', 'CP2': 'eeg', 
        'FC5': 'eeg', 'FC6': 'eeg', 
        'CP5': 'eeg', 'CP6': 'eeg', 
        'FT9': 'eeg', 'FT10': 'eeg', 
        'TP9': 'eeg', 'TP10': 'eeg', 
        'ECG': 'ecg', 
        'R_EYE': 'eog', 'L_EYE': 'eog', 
        'AUDIO': 'stim', 
        'RESP': 'resp', 
        'GSR': 'gsr', 
        'triggerStream': 'stim'  # Assuming triggerStream as 'stim', adjust if it has another specific type
    }

    for i in range(1, 11):
        if f"muerto{i}" in raw.ch_names:
            channel_types[f'muerto{i}'] = 'misc'

    raw.set_channel_types(channel_types)

    bads = ['IO', 'L_EYE']

    for i in range(1, 11):
        if f"muerto{i}" in raw.ch_names:
            bads.append(f"muerto{i}")
    
    raw.info['bads'] = bads

    raw.drop_channels(raw.info['bads'])
    try: 
        # Path to your .bvef file
        bvef_file_path = '../../BC-32.bvef'
        # Load the montage
        montage = mne.channels.read_custom_montage(bvef_file_path)
    except:
        # Path to your .bvef file
        bvef_file_path = './BC-32.bvef'
        # Load the montage
        montage = mne.channels.read_custom_montage(bvef_file_path)

    # Apply the montage to your raw data
    raw.set_montage(montage)
    
    return raw

#### EYE-TRACKING ####

def make_eyetracking_mapping(raw):
    # Inicialmente configuramos el mapeo para los canales de eye-tracking y pupilometría
    mapping_eyetrack = {
        'FPOGX': ('eyegaze', 'px', 'left', 'x'),
        'FPOGY': ('eyegaze', 'px', 'left', 'y'),
        'LPOGX': ('eyegaze', 'deg', 'left', 'x'),
        'LPOGY': ('eyegaze', 'deg', 'left', 'y'),
        'RPOGX': ('eyegaze', 'deg', 'right', 'x'),
        'RPOGY': ('eyegaze', 'deg', 'right', 'y'),
        'BPOGX': ('eyegaze', 'px', 'right', 'x'),
        'BPOGY': ('eyegaze', 'px', 'left', 'y'),
        'LPD': ('pupil', 'px', 'left'),
        'RPD': ('pupil', 'px', 'right'),
    }

    # Configuramos los otros canales como 'misc'
    mapping_misc = {
        'TIME': 'misc',
        'TIMETICK': 'misc',
        'FPOGS': 'misc',
        'FPOGD': 'misc',
        'FPOGID': 'misc',
        'FPOGV': 'misc',
        'LPOGV': 'misc',
        'RPOGV': 'misc',
        'BPOGV': 'misc',
        'LPCX': 'misc',
        'LPCY': 'misc',
        'LPS': 'misc',
        'LPV': 'misc',
        'RPCX': 'misc',
        'RPCY': 'misc',
        'RPS': 'misc',
        'RPV': 'misc',
        'BKID': 'stim',
        'BKDUR': 'stim',
        'BKPMIN': 'misc',
    }

    # Asignamos los tipos de canales usando la función específica de MNE
    raw = mne.preprocessing.eyetracking.set_channel_types_eyetrack(raw, mapping_eyetrack)

    # Asignamos los canales 'misc'
    raw.set_channel_types(mapping_misc)
    
    raw.info['bads'] = ['TIME','TIMETICK']
    raw.drop_channels(raw.info['bads'])

    return raw

def parse_calibration_file(file_path):
    positions = []
    gaze_left = []
    gaze_right = []
    avg_error = None
    valid_points = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith('CALX'):
                _, x_value = line.strip().split(': ')
                try:
                    _, y_value = lines[i+1].strip().split(': ')
                    positions.append([float(x_value), float(y_value)])
                except IndexError:
                    print("Unexpected end of file while reading CALX value.")
            elif line.startswith('LX'):
                try:
                    _, lx = line.strip().split(': ')
                    _, ly = lines[i+1].strip().split(': ')
                    _, lv = lines[i+2].strip().split(': ')
                    if int(lv) == 1:  # Only consider valid gaze points
                        gaze_left.append([float(lx), float(ly)])
                except IndexError:
                    print("Unexpected end of file while reading LX value.")
        
        # Extract the summary for errors
        for line in reversed(lines):
            if line.startswith('  AVE_ERROR'):
                avg_error = float(line.split(': ')[1])
            elif line.startswith('  VALID_POINTS'):
                valid_points = int(line.split(': ')[1])
            if avg_error is not None and valid_points is not None:
                break

    return positions, gaze_left, gaze_right, avg_error, valid_points

def setup_calibration(file_path, eye='left'):
    """Setup the calibration object based on the parsed file data."""
    positions, gaze_left, gaze_right, avg_error, _ = parse_calibration_file(file_path)
    
    # Convert list to numpy arrays
    positions = np.array(positions)
    gaze_left = np.array(gaze_left)
    gaze_right = np.array(gaze_right)
    
    # Select gaze data based on eye
    gaze = gaze_left if eye == 'left' else gaze_right

    # Calculate offsets for each point
    offsets = np.linalg.norm(positions - gaze, axis=1)
    
    # Maximum error
    max_error = offsets.max()

    # Create the Calibration object
    calibration = Calibration(
        onset=-10,  # Example: adjust as needed
        model=f'HV{len(positions)}',  # Dynamically set based on number of points
        eye=eye,
        avg_error=avg_error,
        max_error=max_error,
        positions=positions,
        offsets=offsets,
        gaze=gaze,
        screen_size=(0.531, 0.298),  # Example values, replace with actual
        screen_distance=0.6,  # Example value in meters
        screen_resolution=(1920, 1080)  # Example resolution
    )
    return calibration
