import os
import json
import numpy as np
import pandas as pd

import mne

#### EEG ####

def set_chs_montage(raw):
    eog_chs = {'VEOG': 'eog', 'HEOG': 'eog'}
    raw.set_channel_types(eog_chs)
    
    # Get the absolute path to the BVEF file in the Preprocessing directory
    current_dir = os.path.dirname(os.path.abspath(__file__))  # utils directory
    project_root = os.path.dirname(current_dir)  # project root
    montage_file = os.path.join(project_root, 'Preprocessing', 'CACS-64_REF.bvef')
    
    # Check if file exists
    if not os.path.exists(montage_file):
        # Fallback to just using the filename in the current directory
        montage_file = os.path.join(project_root, 'CACS-64_REF.bvef')
        if not os.path.exists(montage_file):
            raise FileNotFoundError(f"Could not find montage file at {montage_file}")

    montage = mne.channels.read_custom_montage(montage_file)
    raw.set_montage(montage)
    
    raw.info['line_freq'] = 50
    
    return raw
