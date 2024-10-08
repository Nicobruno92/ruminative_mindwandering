import os
import json
import numpy as np
import pandas as pd

import mne

#### EEG ####

def set_chs_montage(raw):
    eog_chs = {'VEOG': 'eog', 'HEOG': 'eog'}
    raw.set_channel_types(eog_chs)
    
    try:
        montage_file = './Preprocessing/CACS-64_REF.bvef'
    except:
        montage_file = './CACS-64_REF.bvef'

    montage = mne.channels.read_custom_montage(montage_file)
    raw.set_montage(montage)
    
    raw.info['line_freq'] = 50
    
    return raw
