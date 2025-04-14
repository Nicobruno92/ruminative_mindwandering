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
        n_probe = 0
        passed_probe = False  # Initialize the variable here
        probe_distance = 0  # Initialize probe_distance variable

        # First pass: reverse through the dataframe
        for i in reversed(range(len(self.df))):
            recoded_value = self.df.loc[i, 'recoded']
            # Check if it's a 'go' or 'nogo' trial *before* checking for probe marker
            # This ensures we don't try to process probes themselves in this logic
            is_trial = recoded_value.startswith(('go', 'nogo'))
            
            if 'Stimulus/S 36' in self.df.loc[i, 'description']:
                probe_distance = 0
                passed_probe = True
                n_probe += 1
                # Skip adding probe distance/number to the probe marker itself
                continue  # Go to the next iteration

            if passed_probe and is_trial:
                probe_distance -= 1
                # Append probe distance and number
                self.df.loc[i, 'recoded'] = f"{recoded_value}/{probe_distance}/probe{n_probe}"

        # Second pass: forward through the dataframe
        trial_counter = 0
        passed_probe = False # Reset for forward pass
        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            
            # Check if it's a 'go' or 'nogo' trial that has probe info
            is_trial_with_probe_info = ('/' in recoded_value) and recoded_value.startswith(('go', 'nogo'))

            if 'Stimulus/S 36' in self.df.loc[i, 'description']:
                trial_counter = 0
                passed_probe = True
                 # Skip adding trial counter to the probe marker itself
                continue # Go to the next iteration

            if passed_probe and is_trial_with_probe_info:
                trial_counter += 1
                # Append trial counter to the existing string
                self.df.loc[i, 'recoded'] = f"{recoded_value}/{trial_counter}"
    
    def fix_probe_distances(self):
        """
        Ensures that probe distances are properly formatted.
        Keeps the negative values for trials before a probe and adds positive values for trials after a probe.
        This preserves directional information about trial position relative to probes.
        """
        # Dictionary to track trial counts after each probe
        post_probe_counters = {}
        
        # Iterate through the dataframe to identify trials after probes
        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            parts = recoded_value.split('/') if isinstance(recoded_value, str) else []
            
            # Detect probe markers
            if 'Stimulus/S 36' in self.df.loc[i, 'description']:
                current_probe = None
                # Find the probe number from surrounding trials
                for j in range(max(0, i-5), min(len(self.df), i+5)):
                    r_value = self.df.loc[j, 'recoded']
                    if isinstance(r_value, str) and 'probe' in r_value:
                        probe_parts = r_value.split('/')
                        for part in probe_parts:
                            if part.startswith('probe') and part[5:].isdigit():
                                current_probe = int(part[5:])
                                break
                        if current_probe:
                            break
                
                if current_probe:
                    # Initialize counter for this probe
                    post_probe_counters[current_probe] = 0
            
            # Process trials after probes with existing probe numbers
            elif isinstance(recoded_value, str) and recoded_value.startswith(('go', 'nogo')):
                probe_num = None
                for part in parts:
                    if part.startswith('probe') and part[5:].isdigit():
                        probe_num = int(part[5:])
                        break
                
                if probe_num in post_probe_counters:
                    # For trials after a probe, check if there's a negative distance value
                    has_negative_distance = False
                    for part in parts:
                        if part.startswith('-') and part[1:].isdigit():
                            has_negative_distance = True
                            break
                    
                    if not has_negative_distance:
                        # This is a trial after the probe, increment counter
                        post_probe_counters[probe_num] += 1
                        
                        # Extract and reconstruct the string with positive distance
                        new_parts = []
                        for part in parts:
                            if part == str(post_probe_counters[probe_num]):
                                # Replace the trial count with trial count and positive distance
                                new_parts.append(f"+{post_probe_counters[probe_num]}")
                                new_parts.append(part)
                            else:
                                new_parts.append(part)
                        
                        # Update the recoded value
                        self.df.loc[i, 'recoded'] = '/'.join(new_parts)

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
                # Check if it's one of the TP answers (S100-S200)
                if 'Stimulus/S' in annotation_value:
                     try:
                         code = int(''.join(filter(str.isdigit, annotation_value)))
                         if 100 <= code <= 200:
                             id_base = 6 # Assign specific code for TP answers
                         else:
                             id_base = 5 # Other stimuli
                     except ValueError:
                         id_base = 5 # Default for non-numeric stimuli
                else:
                     id_base = 5 # Other stimuli
            # Check for TP questions (S31-S36) based on recoded format
            elif any(key in recoded_value for key in ['onoff', 'selfother', 'time', 'valence', 'confidence', 'average']):
                 id_base = 7 # Specific code for TP questions
            else:
                id_base = 9  # for others
            
            # Correctness: correct = 1, incorrect = 0
            if '/correct/' in recoded_value:
                correctness = 1
            else:
                correctness = 0
            
            # Extract the probe values (onoff, selfother, valence, etc.)
            probe_parts = recoded_value.split('/')
            onoff, selfother, valence, time, confidence, average = (0, 0, 0, 0, 0, 0)
            probe_number = 0  # Initialize probe number
            distance_to_probe = 0
            trial_number = 0
            
            # Parse basic probe info first
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
                # Do not parse probe number, distance, trial here yet

            # Now parse distance, probe number, and trial number based on position
            # Expected format for trials: .../distance/probeN/trial_count
            if len(probe_parts) >= 4 and id_base in [1, 2]: # Only for go/nogo trials
                 # Distance is third from end
                 if probe_parts[-3].lstrip('-').isdigit():
                     distance_to_probe = abs(int(probe_parts[-3])) # Use absolute value

                 # Probe number is second from end
                 if probe_parts[-2].startswith('probe') and probe_parts[-2][5:].isdigit():
                     probe_number = int(probe_parts[-2][5:])

                 # Trial number is last
                 if probe_parts[-1].isdigit():
                     trial_number = int(probe_parts[-1])
            
            # Combine all parts into a large integer (now including probe_number correctly)
            # Ensure consistent field widths
            numerical_id = int(f"{id_base:01}{correctness:01}{onoff:02}{selfother:02}{valence:02}{time:02}{confidence:02}{average:02}{distance_to_probe:03}{probe_number:02}{trial_number:03}")
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
        self.fix_probe_distances()  # New step to ensure proper distance formatting
        self.create_random_event_id()

        # Create events and event_id based on random numerical ID
        onsets = mne.events_from_annotations(self.raw, event_id='auto')[0][:, 0]
        durations = np.zeros_like(onsets)
        ids = self.df['numerical_id'].values

        events = np.vstack((onsets, durations, ids)).T
        event_id = dict(zip(self.df['recoded'], self.df['numerical_id']))

        return events, event_id
