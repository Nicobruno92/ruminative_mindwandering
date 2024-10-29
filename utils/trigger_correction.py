import numpy as np
import mne
import random

class TriggerCorrector:
    def __init__(self, raw):
        # Extract annotations from the raw object and convert to a DataFrame
        self.df = raw.annotations.to_data_frame().copy()
        self.raw = raw

    def recode_annotations(self):
        # Step 1: Create new columns for the recoding process
        self.df['recoded'] = self.df['description']
        self.df['correctness'] = None

        # Step 2: Identify rows where stimuli occur and apply recoding
        for i in range(len(self.df)):
            # Recoding 'go' and 'nogo'
            if self.df.loc[i, 'description'] in ['Stimulus/S 41', 'Stimulus/S 43']:
                self.df.loc[i, 'recoded'] = 'go'
            elif self.df.loc[i, 'description'] == 'Stimulus/S 42':
                self.df.loc[i, 'recoded'] = 'nogo'

            # Step 3: Determine correctness based on following annotations
            if self.df.loc[i, 'recoded'] == 'go':
                if (i + 1 < len(self.df)) and self.df.loc[i + 1, 'description'] == 'Stimulus/S 44':
                    self.df.loc[i, 'correctness'] = 'correct'
                else:
                    self.df.loc[i, 'correctness'] = 'incorrect'
            elif self.df.loc[i, 'recoded'] == 'nogo':
                if (i + 1 < len(self.df)) and self.df.loc[i + 1, 'description'] == 'Stimulus/S 45':
                    self.df.loc[i, 'correctness'] = 'incorrect'
                else:
                    self.df.loc[i, 'correctness'] = 'correct'

    def retropropagate_tp_values(self):
        # Iterate through the dataframe to find thought probes and their corresponding answers
        for i in range(len(self.df)):
            if self.df.loc[i, 'description'] in ['Stimulus/S 31', 'Stimulus/S 32', 'Stimulus/S 33', 
                                                 'Stimulus/S 34', 'Stimulus/S 35', 'Stimulus/S 36']:
                # Identify the thought probe type
                probe_type = {
                    'Stimulus/S 31': 'onoff',
                    'Stimulus/S 32': 'selfother',
                    'Stimulus/S 33': 'time',
                    'Stimulus/S 34': 'valence',
                    'Stimulus/S 35': 'confidence',
                    'Stimulus/S 36': 'average'
                }.get(self.df.loc[i, 'description'], 'unknown')
                
                # Look for the next answer (in range 100-200)
                for j in range(i+1, len(self.df)):
                    if self.df.loc[j, 'description'].startswith('Stimulus/S'):
                        try:
                            code = int(''.join(filter(str.isdigit, self.df.loc[j, 'description'])))
                            if 100 <= code <= 200:
                                value = code - 100
                                self.df.loc[i, 'recoded'] = f'{probe_type}{value}'
                                break
                        except ValueError:
                            continue

    def retropropagate_tp_to_trials(self):
        # Initialize variables to store the most recent thought probe values
        tp_values = {'onoff': None, 'selfother': None, 'valence': None, 'time': None, 'confidence': None, 'average': None}

        # Iterate through the dataframe in reverse to retropropagate TP values to previous go/nogo trials
        for i in reversed(range(len(self.df))):
            recoded = self.df.loc[i, 'recoded']
            # Check if the current row is a thought probe and store the value
            if any(key in recoded for key in tp_values):
                for key in tp_values.keys():
                    if key in recoded:
                        tp_values[key] = recoded

            # When encountering a go/nogo trial, assign the thought probe values
            if recoded in ['go', 'nogo']:
                trial_info = f"{recoded}/{self.df.loc[i, 'correctness']}/" \
                             f"{tp_values['onoff']}/{tp_values['selfother']}/{tp_values['valence']}/" \
                             f"{tp_values['time']}/{tp_values['confidence']}/{tp_values['average']}"
                self.df.loc[i, 'recoded'] = trial_info

    def propagate_trial_info(self):
        trial_counter = 0
        probe_distance = 0
        passed_probe = False
        n_probe = 0

        # First pass: reverse through the dataframe
        for i in reversed(range(len(self.df))):
            recoded_value = self.df.loc[i, 'recoded']
            if 'Stimulus/S 36' in self.df.loc[i, 'description']:
                probe_distance = 0
                passed_probe = True
                n_probe += 1
            if passed_probe and recoded_value.startswith(('go', 'nogo')):
                probe_distance -= 1
                self.df.loc[i, 'recoded'] = f"{recoded_value}/{probe_distance}/probe{n_probe}"

        # Second pass: forward through the dataframe
        trial_counter = 0
        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            if 'Stimulus/S 36' in self.df.loc[i, 'description']:
                trial_counter = 0
                passed_probe = True
            if passed_probe and recoded_value.startswith(('go', 'nogo')):
                trial_counter += 1
                self.df.loc[i, 'recoded'] = f"{recoded_value}/{trial_counter}"

    def create_full_numerical_id_as_int(self):
        ids = []
        
        for i, row in self.df.iterrows():
            recoded_value = row['recoded']
            annotation_value = row['description']
            
            # Initialize a base value for the ID
            id_base = 0
            
            # Annotation type: go = 1, nogo = 2, probe = 3
            if recoded_value.startswith('go'):
                id_base = 1
            elif recoded_value.startswith('nogo'):
                id_base = 2
            elif recoded_value == 'Stimulus/S 44':
                id_base = 3
            elif recoded_value == 'Stimulus/S 45':
                id_base = 4
            elif annotation_value.startswith('Stimulus/'):
                id_base = 5
            else:
                id_base = 9  # for probes and others
            
            # Correctness: correct = 1, incorrect = 0
            if '/correct/' in recoded_value:
                correctness = 1
            else:
                correctness = 0
            
            # Extract the probe values (onoff, selfother, valence, etc.)
            probe_parts = recoded_value.split('/')
            onoff, selfother, valence, time, confidence, average = (0, 0, 0, 0, 0, 0)
            
            for part in probe_parts:
                if 'onoff' in part:
                    onoff = int(''.join(filter(str.isdigit, part)))
                elif 'selfother' in part:
                    selfother = int(''.join(filter(str.isdigit, part)))
                elif 'valence' in part:
                    valence = int(''.join(filter(str.isdigit, part)))
                elif 'time' in part:
                    time = int(''.join(filter(str.isdigit, part)))
                elif 'confidence' in part:
                    confidence = int(''.join(filter(str.isdigit, part)))
                elif 'average' in part:
                    average = int(''.join(filter(str.isdigit, part)))

            # Handle the case where there may not be enough parts to access trial_number and distance
            if len(probe_parts) >= 3 and probe_parts[-2].lstrip('-').isdigit():
                distance_to_probe = int(probe_parts[-2].lstrip('-'))
            else:
                distance_to_probe = 0
            if len(probe_parts) >= 3 and probe_parts[-1].isdigit():
                trial_number = int(probe_parts[-1])
            else:
                trial_number = 0
            
            # Combine all parts into a large integer
            numerical_id = int(f"{id_base:01}{correctness:01}{onoff:02}{selfother:02}{valence:02}{time:02}{confidence:02}{average:02}{distance_to_probe:02}{trial_number:02}")
            ids.append(numerical_id)
        
        self.df['numerical_id'] = ids
        return self.df
    
    def create_random_event_id(self):
        """Create a random ID for each unique recoded value."""
        # Get unique recoded values and assign a random ID to each
        unique_recoded = self.df['recoded'].unique()
        recoded_to_id = {recoded: random.randint(1, 2147483647) for recoded in unique_recoded}

        # Assign the random ID based on the recoded value
        self.df['numerical_id'] = self.df['recoded'].map(recoded_to_id)

    def process_annotations(self):
        # Combines all the steps into one function
        self.recode_annotations()
        self.retropropagate_tp_values()
        self.retropropagate_tp_to_trials()
        self.propagate_trial_info()
        self.create_random_event_id()

        # Create events and event_id based on random numerical ID
        onsets = mne.events_from_annotations(self.raw, event_id='auto')[0][:, 0]
        durations = np.zeros_like(onsets)
        ids = self.df['numerical_id'].values

        events = np.vstack((onsets, durations, ids)).T
        event_id = dict(zip(self.df['recoded'], self.df['numerical_id']))

        return events, event_id
