import numpy as np
import mne
import pandas as pd
import random
import os
try:
    import yaml
except Exception:
    yaml = None

class TriggerCorrector:
    def __init__(self, raw):
        # Extract annotations from the raw object and convert to a DataFrame
        self.df = raw.annotations.to_data_frame().copy()
        self.raw = raw

    def annotate_rest_periods(self):
        """
        Insert BAD_rest annotations spanning from each rest start to the first
        subsequent stimulus that starts a trial.

        rest_start is identified as 'Stimulus/S 31'.
        rest_end is the first of {'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43'}
        that appears after the rest_start.
        """
        potential_end_descriptions = {'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43'}

        bad_onsets = []
        bad_durations = []
        bad_descriptions = []

        descriptions = list(self.raw.annotations.description)
        onsets = list(self.raw.annotations.onset)

        for i in range(len(descriptions)):
            if descriptions[i] == 'Stimulus/S 31':
                start_onset = float(onsets[i])
                end_onset = None
                for j in range(i + 1, len(descriptions)):
                    if descriptions[j] in potential_end_descriptions:
                        end_onset = float(onsets[j])
                        break
                if end_onset is not None and end_onset > start_onset:
                    bad_onsets.append(start_onset)
                    bad_durations.append(end_onset - start_onset)
                    bad_descriptions.append('BAD_rest')

        if bad_onsets:
            new_annots = mne.Annotations(
                onset=bad_onsets,
                duration=bad_durations,
                description=bad_descriptions,
                orig_time=self.raw.annotations.orig_time,
            )
            # Append to raw annotations (use setter for compatibility)
            self.raw.set_annotations(self.raw.annotations + new_annots)

            # Refresh the working DataFrame so downstream logic stays aligned
            self.df = self.raw.annotations.to_data_frame().copy()

    def annotate_thought_probe_markers(self):
        """
        Insert THOUGHT_PROBE annotations at each 'Stimulus/S 31' onset.
        Does not remove or overwrite existing annotations and avoids duplicates
        if already present.
        """
        descriptions = list(self.raw.annotations.description)
        onsets = list(self.raw.annotations.onset)

        # Track existing THOUGHT_PROBE onsets to avoid duplicates (with small tolerance)
        existing_tp_onsets = {float(onsets[idx]) for idx, desc in enumerate(descriptions) if desc == 'THOUGHT_PROBE'}

        tp_onsets_to_add = []
        tp_durations_to_add = []
        tp_descriptions_to_add = []

        for i in range(len(descriptions)):
            if descriptions[i] == 'Stimulus/S 31':
                onset_val = float(onsets[i])
                if onset_val not in existing_tp_onsets:
                    tp_onsets_to_add.append(onset_val)
                    tp_durations_to_add.append(0.0)
                    tp_descriptions_to_add.append('THOUGHT_PROBE')

        if tp_onsets_to_add:
            tp_annots = mne.Annotations(
                onset=tp_onsets_to_add,
                duration=tp_durations_to_add,
                description=tp_descriptions_to_add,
                orig_time=self.raw.annotations.orig_time,
            )
            self.raw.set_annotations(self.raw.annotations + tp_annots)
            self.df = self.raw.annotations.to_data_frame().copy()

    def recode_annotations(self):
        # Step 1: Create new columns for the recoding process
        self.df['recoded'] = self.df['description']
        self.df['correctness'] = None
        self.df['rt'] = None  # Add RT column
        self.df['stim_timestamp'] = None  # Add stimulus timestamp column
        self.df['response_timestamp'] = None  # Add response timestamp column
        # Columns for second button presses
        self.df['rt_2nd'] = None
        self.df['stim_timestamp_2nd'] = None
        self.df['response_timestamp_2nd'] = None

        # Step 2: Identify rows where stimuli occur and apply recoding
        for i in range(len(self.df)):
            # Recoding 'go' and 'nogo' stimuli
            if self.df.loc[i, 'description'] in ['Stimulus/S 41', 'Stimulus/S 43']:
                self.df.loc[i, 'recoded'] = 'go'
                # Store stimulus timestamp
                self.df.loc[i, 'stim_timestamp'] = self.df.loc[i, 'onset']
            elif self.df.loc[i, 'description'] == 'Stimulus/S 42':
                self.df.loc[i, 'recoded'] = 'nogo'
                # Store stimulus timestamp
                self.df.loc[i, 'stim_timestamp'] = self.df.loc[i, 'onset']

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

        # Step 4: Recode response events (S44 and S45) with correctness and RT
        for i in range(len(self.df)):
            desc = self.df.loc[i, 'description']
            
            # Handle correct responses (S44)
            if desc == 'Stimulus/S 44':
                # Find preceding stimulus (should be go: S41 or S43)
                stimulus_idx = None
                for j in range(i - 1, max(-1, i - 10), -1):  # Look back up to 10 events
                    if self.df.loc[j, 'description'] in ['Stimulus/S 41', 'Stimulus/S 43']:
                        stimulus_idx = j
                        break
                
                if stimulus_idx is not None:
                    # Calculate RT in milliseconds
                    rt_sec = self.df.loc[i, 'onset'] - self.df.loc[stimulus_idx, 'onset']
                    # Convert to float if it's a Timedelta
                    if hasattr(rt_sec, 'total_seconds'):
                        rt_sec = rt_sec.total_seconds()
                    rt_ms = int(round(float(rt_sec) * 1000))
                    self.df.loc[i, 'recoded'] = f'response/correct/rt{rt_ms}'
                    self.df.loc[i, 'rt'] = rt_ms
                    # Store timestamps for both stimulus and response
                    self.df.loc[i, 'stim_timestamp'] = self.df.loc[stimulus_idx, 'onset']
                    self.df.loc[i, 'response_timestamp'] = self.df.loc[i, 'onset']
                else:
                    # Fallback if no stimulus found
                    self.df.loc[i, 'recoded'] = 'response/correct/rtNA'
                    self.df.loc[i, 'response_timestamp'] = self.df.loc[i, 'onset']
            
            # Handle incorrect responses (S45)
            elif desc == 'Stimulus/S 45':
                # Find preceding stimulus (should be nogo: S42)
                stimulus_idx = None
                for j in range(i - 1, max(-1, i - 10), -1):  # Look back up to 10 events
                    if self.df.loc[j, 'description'] == 'Stimulus/S 42':
                        stimulus_idx = j
                        break
                
                if stimulus_idx is not None:
                    # Calculate RT in milliseconds
                    rt_sec = self.df.loc[i, 'onset'] - self.df.loc[stimulus_idx, 'onset']
                    # Convert to float if it's a Timedelta
                    if hasattr(rt_sec, 'total_seconds'):
                        rt_sec = rt_sec.total_seconds()
                    rt_ms = int(round(float(rt_sec) * 1000))
                    self.df.loc[i, 'recoded'] = f'response/incorrect/rt{rt_ms}'
                    self.df.loc[i, 'rt'] = rt_ms
                    # Store timestamps for both stimulus and response
                    self.df.loc[i, 'stim_timestamp'] = self.df.loc[stimulus_idx, 'onset']
                    self.df.loc[i, 'response_timestamp'] = self.df.loc[i, 'onset']
                else:
                    # Fallback if no stimulus found
                    self.df.loc[i, 'recoded'] = 'response/incorrect/rtNA'
                    self.df.loc[i, 'response_timestamp'] = self.df.loc[i, 'onset']

    def detect_second_button_presses(self):
        """
        Detect and store second button presses that occur after the first response
        but before the next stimulus.
        
        Adds _2nd columns for: rt_2nd, stim_timestamp_2nd, response_timestamp_2nd
        """
        # Iterate through all events
        for i in range(len(self.df)):
            desc = self.df.loc[i, 'description']
            
            # Only process response events (S44 or S45)
            if desc not in ['Stimulus/S 44', 'Stimulus/S 45']:
                continue
            
            # Check if this is already marked as a first response
            # (has rt and response_timestamp but not rt_2nd)
            if self.df.loc[i, 'rt'] is None:
                continue
            
            # Look forward to find a potential second button press
            # before the next stimulus
            for j in range(i + 1, min(len(self.df), i + 10)):
                next_desc = self.df.loc[j, 'description']
                
                # Stop if we hit a stimulus (new trial started)
                if next_desc in ['Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43']:
                    break
                
                # Check if this is another response of the same type
                if next_desc == desc:
                    # This is a second button press!
                    # Calculate RT from the original stimulus
                    stim_ts = self.df.loc[i, 'stim_timestamp']
                    second_resp_ts = self.df.loc[j, 'onset']
                    
                    if stim_ts is not None:
                        rt_diff = second_resp_ts - stim_ts
                        if hasattr(rt_diff, 'total_seconds'):
                            rt_diff = rt_diff.total_seconds()
                        rt_2nd_ms = int(round(float(rt_diff) * 1000))
                        
                        # Store second press info in the FIRST response row
                        self.df.loc[i, 'rt_2nd'] = rt_2nd_ms
                        self.df.loc[i, 'stim_timestamp_2nd'] = stim_ts
                        self.df.loc[i, 'response_timestamp_2nd'] = second_resp_ts
                        
                        # Mark the second response event itself
                        correctness = 'correct' if desc == 'Stimulus/S 44' else 'incorrect'
                        self.df.loc[j, 'recoded'] = f'response_2nd/{correctness}/rt{rt_2nd_ms}'
                        self.df.loc[j, 'rt'] = rt_2nd_ms
                        self.df.loc[j, 'stim_timestamp'] = stim_ts
                        self.df.loc[j, 'response_timestamp'] = second_resp_ts
                    
                    # Only process the first second press found
                    break

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
        # Build a forward-ordered mapping of S36 (end-of-probe) rows to probe numbers (1..N)
        s36_indices = [i for i in range(len(self.df)) if self.df.loc[i, 'description'] == 'Stimulus/S 36']
        s36_index_to_num = {idx: num for num, idx in enumerate(s36_indices, start=1)}

        passed_probe = False
        probe_distance = 0
        current_probe_num = None

        # First pass: reverse through the dataframe
        for i in reversed(range(len(self.df))):
            recoded_value = self.df.loc[i, 'recoded']
            # Check if it's a 'go' or 'nogo' trial *before* checking for probe marker
            # This ensures we don't try to process probes themselves in this logic
            is_trial = recoded_value.startswith(('go', 'nogo'))
            
            if self.df.loc[i, 'description'] == 'Stimulus/S 36':
                probe_distance = 0
                passed_probe = True
                current_probe_num = s36_index_to_num.get(i)
                # Skip adding probe distance/number to the probe marker itself
                continue

            if passed_probe and is_trial:
                probe_distance -= 1
                # Append probe distance and number
                probe_token = f"/probe{current_probe_num}" if current_probe_num is not None else ""
                self.df.loc[i, 'recoded'] = f"{recoded_value}/{probe_distance}{probe_token}"

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

    def add_onoff_label(self):
        """
        Append '/ontask' or '/offtask' to recoded values based on onoff rating.
        Uses threshold: onoff >= 50 -> ontask, onoff < 50 -> offtask.
        Applies only to trial rows (go/nogo) that carry 'onoff' entries.
        """
        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            if not isinstance(recoded_value, str):
                continue
            # Only label trials, skip probe markers and question rows
            if not recoded_value.startswith(('go', 'nogo')):
                continue
            if ('onoff' not in recoded_value) or ('/ontask' in recoded_value or '/offtask' in recoded_value):
                continue

            onoff_value = None
            for part in recoded_value.split('/'):
                if part.startswith('onoff'):
                    digits = ''.join(filter(str.isdigit, part))
                    if digits:
                        onoff_value = int(digits)
                        break
            if onoff_value is None:
                continue

            label = 'ontask' if onoff_value >= 50 else 'offtask'

            # Insert the label just before the distance/probe/trial tokens if present;
            # otherwise, append at the end. We detect tokens by their pattern to avoid
            # disturbing the positional parsing.
            parts = recoded_value.split('/')

            # Identify indices for signed distance ('+N' or '-N'), 'probeN', and last pure digit (trial)
            signed_distance_idx = None
            probe_idx = None
            last_pure_digit_idx = None
            for idx, part in enumerate(parts):
                if (part.startswith('+') or part.startswith('-')) and part[1:].isdigit():
                    signed_distance_idx = idx
                elif part.startswith('probe') and part[5:].isdigit():
                    probe_idx = idx
                elif part.isdigit():
                    last_pure_digit_idx = idx

            insertion_index = len(parts)
            # If we have trial number detected, put the label before the distance/probe/trial block
            if last_pure_digit_idx is not None:
                # Aim to place before the first of the special tokens encountered towards the end
                candidate_indices = [idx for idx in [signed_distance_idx, probe_idx, last_pure_digit_idx] if idx is not None]
                if candidate_indices:
                    insertion_index = min(candidate_indices)

            parts.insert(insertion_index, label)
            self.df.loc[i, 'recoded'] = '/'.join(parts)

    # -------------------- Harmonization --------------------
    def _load_harmonization_config(self):
        """
        Load annotation harmonization config from YAML.
        Returns dict with 'enabled', 'ontask_threshold', 'ontask_dimension'.
        """
        defaults = {
            'enabled': True,
            'ontask_threshold': 50,
            'ontask_dimension': 'onoff',
        }
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(project_root, 'Preprocessing_pipeline_new', 'config.yaml')
            if yaml is None:
                return defaults
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict) and 'annotation_harmonization' in cfg:
                    harm_cfg = cfg.get('annotation_harmonization') or {}
                    return {
                        'enabled': harm_cfg.get('enabled', True),
                        'ontask_threshold': int(harm_cfg.get('ontask_threshold', 50)),
                        'ontask_dimension': harm_cfg.get('ontask_dimension', 'onoff'),
                    }
        except Exception:
            return defaults
        return defaults

    def harmonize_annotations(self):
        """
        Harmonize annotations to standardized format while preserving original content.
        
        Transforms:
        - Trial format: go/correct/onoff91/selfother51/.../ontask/-5/probe5/12
          To: go/correct/probe-5/trial-12/distance-5/ONTASK/onoff91/selfother51/.../ontask
        
        - THOUGHT_PROBE format: THOUGHT_PROBE/onoff91/selfother51/.../probe5
          To: THOUGHT_PROBE/probe-5/ONTASK/onoff91/selfother51/...
        
        The harmonization adds standardized prefixes (probe-, trial-, distance-)
        and ONTASK/OFFTASK classification based on configurable threshold.
        """
        harm_cfg = self._load_harmonization_config()
        if not harm_cfg.get('enabled', True):
            return
        
        ontask_threshold = harm_cfg.get('ontask_threshold', 50)
        ontask_dimension = harm_cfg.get('ontask_dimension', 'onoff')
        
        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            if not isinstance(recoded_value, str):
                continue
            
            parts = recoded_value.split('/')
            
            # Handle go/nogo trials
            if recoded_value.startswith(('go/', 'nogo/')):
                self._harmonize_trial(i, parts, ontask_threshold, ontask_dimension)
            
            # Handle THOUGHT_PROBE markers
            elif recoded_value.startswith('THOUGHT_PROBE/'):
                self._harmonize_thought_probe(i, parts, ontask_threshold, ontask_dimension)

    def _harmonize_trial(self, idx: int, parts: list, ontask_threshold: int, ontask_dimension: str):
        """
        Harmonize a single trial annotation.
        
        Input format: go/correct/onoff91/selfother51/.../ontask/-5/probe5/12
        Output format: go/correct/probe-5/trial-12/distance-5/ONTASK/onoff91/selfother51/.../ontask
        """
        # Extract components
        trial_type = parts[0]  # go or nogo
        correctness = parts[1] if len(parts) > 1 else 'unknown'  # correct or incorrect
        
        # Find probe number, distance, and trial number
        probe_num = None
        distance = None
        trial_num = None
        dimension_parts = []
        other_parts = []
        
        for part in parts[2:]:  # Skip trial_type and correctness
            # Check for probe number (probeN format)
            if part.startswith('probe') and part[5:].isdigit():
                probe_num = int(part[5:])
            # Check for signed distance (+N or -N)
            elif len(part) > 1 and part[0] in ['+', '-'] and part[1:].isdigit():
                distance = abs(int(part))
            # Check for pure digit (trial number) - take the last one
            elif part.isdigit():
                trial_num = int(part)
            # Check for dimension values (onoff91, selfother51, etc.)
            elif any(part.startswith(dim) for dim in ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']):
                dimension_parts.append(part)
            # Check for binary labels (onoffBin1, etc.) or ontask/offtask
            elif 'Bin' in part or part in ['ontask', 'offtask']:
                other_parts.append(part)
            else:
                other_parts.append(part)
        
        # Determine ONTASK/OFFTASK based on dimension value
        ontask_label = self._get_ontask_label(dimension_parts, ontask_threshold, ontask_dimension)
        
        # Build harmonized annotation
        harmonized = [trial_type, correctness]
        
        # Add standardized position info
        if probe_num is not None:
            harmonized.append(f"probe-{probe_num}")
        if trial_num is not None:
            harmonized.append(f"trial-{trial_num}")
        if distance is not None:
            harmonized.append(f"distance-{distance}")
        
        # Add ONTASK/OFFTASK label
        harmonized.append(ontask_label)
        
        # Add dimension values (original content preserved)
        harmonized.extend(dimension_parts)
        
        # Add other parts (binary labels, ontask/offtask lowercase)
        harmonized.extend(other_parts)
        
        self.df.loc[idx, 'recoded'] = '/'.join(harmonized)

    def _harmonize_thought_probe(self, idx: int, parts: list, ontask_threshold: int, ontask_dimension: str):
        """
        Harmonize a THOUGHT_PROBE annotation.
        
        Input format: THOUGHT_PROBE/onoff91/selfother51/.../probe5
        Output format: THOUGHT_PROBE/probe-5/ONTASK/onoff91/selfother51/...
        """
        # Find probe number and dimension values
        probe_num = None
        dimension_parts = []
        other_parts = []
        
        for part in parts[1:]:  # Skip THOUGHT_PROBE
            if part.startswith('probe') and part[5:].isdigit():
                probe_num = int(part[5:])
            elif any(part.startswith(dim) for dim in ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']):
                dimension_parts.append(part)
            elif 'Bin' in part:
                other_parts.append(part)
            else:
                other_parts.append(part)
        
        # Determine ONTASK/OFFTASK
        ontask_label = self._get_ontask_label(dimension_parts, ontask_threshold, ontask_dimension)
        
        # Build harmonized annotation
        harmonized = ['THOUGHT_PROBE']
        
        if probe_num is not None:
            harmonized.append(f"probe-{probe_num}")
        
        harmonized.append(ontask_label)
        harmonized.extend(dimension_parts)
        harmonized.extend(other_parts)
        
        self.df.loc[idx, 'recoded'] = '/'.join(harmonized)

    def _get_ontask_label(self, dimension_parts: list, threshold: int, dimension: str) -> str:
        """
        Determine ONTASK or OFFTASK label based on dimension value and threshold.
        """
        for part in dimension_parts:
            if part.startswith(dimension):
                digits = ''.join(filter(str.isdigit, part))
                if digits:
                    value = int(digits)
                    return 'ONTASK' if value >= threshold else 'OFFTASK'
        return 'UNKNOWN'

    # -------------------- Binarization helpers --------------------
    def _load_thresholds(self):
        """
        Load thresholds for thought dimensions from YAML config.
        Looks for Preprocessing_pipeline_new/config.yaml -> thought_thresholds.
        Defaults = 50 for all dimensions if not found or YAML unavailable.
        """
        defaults = {
            'onoff': 50,
            'selfother': 50,
            'valence': 50,
            'time': 50,
            'confidence': 50,
            'average': 50,
        }
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(project_root, 'Preprocessing_pipeline_new', 'config.yaml')
            if yaml is None:
                return defaults
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict) and 'thought_thresholds' in cfg:
                    merged = defaults.copy()
                    for k, v in (cfg.get('thought_thresholds') or {}).items():
                        if k in merged:
                            try:
                                merged[k] = int(v)
                            except Exception:
                                pass
                    return merged
        except Exception:
            return defaults
        return defaults

    def _extract_dimension_values(self, recoded_value: str):
        values = {
            'onoff': None,
            'selfother': None,
            'valence': None,
            'time': None,
            'confidence': None,
            'average': None,
        }
        parts = recoded_value.split('/')
        for part in parts:
            for key in values.keys():
                if part.startswith(key):
                    digits = ''.join(filter(str.isdigit, part))
                    if digits:
                        try:
                            values[key] = int(digits)
                        except Exception:
                            values[key] = None
        return values

    def append_binary_labels(self) -> None:
        """
        Append binarized tokens for all thought dimensions to all relevant rows
        (trials and THOUGHT_PROBE aggregates). For trials, also append 'gonogoBin1'
        for go and 'gonogoBin0' for nogo.

        Tokens are inserted before the distance/probe/trial index block when present
        to preserve positional parsing for distance/probe/trial.
        """
        thresholds = self._load_thresholds()

        for i in range(len(self.df)):
            recoded_value = self.df.loc[i, 'recoded']
            if not isinstance(recoded_value, str):
                continue

            is_trial = recoded_value.startswith(('go', 'nogo'))
            is_probe_marker = recoded_value.startswith('THOUGHT_PROBE/')
            if not (is_trial or is_probe_marker):
                # Also allow dimension-only rows (e.g., 'onoff91')
                contains_dimension = any(k in recoded_value for k in ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average'])
                if not contains_dimension:
                    continue

            values = self._extract_dimension_values(recoded_value)

            # Build bin tokens if not already present
            parts = recoded_value.split('/')
            existing_prefixes = {p.split('Bin')[0] + 'Bin' for p in parts if 'Bin' in p}
            bin_tokens = []
            for key, val in values.items():
                if val is None:
                    continue
                prefix = f"{key}Bin"
                if prefix in existing_prefixes:
                    continue
                thr = thresholds.get(key, 50)
                bin_val = 1 if val >= thr else 0
                bin_tokens.append(f"{prefix}{bin_val}")

            if is_trial:
                # Add gonogoBin
                if not any(p.startswith('gonogoBin') for p in parts):
                    bin_tokens.append('gonogoBin1' if recoded_value.startswith('go') else 'gonogoBin0')

            if not bin_tokens:
                continue

            # Find insertion index before distance/probe/trial tokens
            signed_distance_idx = None
            probe_idx = None
            last_pure_digit_idx = None
            for idx, part in enumerate(parts):
                if (part.startswith('+') or part.startswith('-')) and part[1:].isdigit():
                    signed_distance_idx = idx
                elif part.startswith('probe') and part[5:].isdigit():
                    probe_idx = idx
                elif part.isdigit():
                    last_pure_digit_idx = idx

            insertion_index = len(parts)
            candidate_indices = [idx for idx in [signed_distance_idx, probe_idx, last_pure_digit_idx] if idx is not None]
            if candidate_indices:
                insertion_index = min(candidate_indices)

            new_parts = parts[:insertion_index] + bin_tokens + parts[insertion_index:]
            self.df.loc[i, 'recoded'] = '/'.join(new_parts)

    def aggregate_thought_probe_info(self):
        """
        For each THOUGHT_PROBE marker, replace its recoded value with the
        aggregated probe answers in the canonical order and append the probe
        number (e.g., 'onoff91/selfother51/valence72/time68/confidence76/average71/probe15').

        The probe number is inferred from nearby trials that already carry
        '/probeN' (assigned in propagate_trial_info). If not found nearby,
        numbering falls back to counting 'Stimulus/S 36' occurrences up to that
        probe so it remains consistent.
        """
        if 'recoded' not in self.df.columns:
            return

        # Precompute cumulative count of S36 to provide a fallback probe index
        s36_cumsum = []
        count = 0
        for i in range(len(self.df)):
            if self.df.loc[i, 'description'] == 'Stimulus/S 36':
                count += 1
            s36_cumsum.append(count)

        for i in range(len(self.df)):
            if self.df.loc[i, 'description'] != 'THOUGHT_PROBE':
                continue

            # Collect probe answers between this marker and the next S36
            values = {k: None for k in ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']}

            # Search forward up to the next S36 (end of probe block)
            end_idx = None
            for j in range(i, len(self.df)):
                desc_j = self.df.loc[j, 'description']
                rec_j = str(self.df.loc[j, 'recoded']) if isinstance(self.df.loc[j, 'recoded'], str) else ''
                for key in list(values.keys()):
                    if values[key] is None and rec_j.startswith(key):
                        values[key] = rec_j
                if desc_j == 'Stimulus/S 36':
                    end_idx = j
                    break

            # Infer probe number from nearby trials carrying '/probeN'
            probe_num = None
            # Prefer scanning backward to find trials before the marker
            for j in range(i - 1, max(-1, i - 200), -1):
                rec_j = self.df.loc[j, 'recoded']
                if isinstance(rec_j, str) and rec_j.startswith(('go', 'nogo')):
                    for part in rec_j.split('/'):
                        if part.startswith('probe') and part[5:].isdigit():
                            probe_num = int(part[5:])
                            break
                if probe_num is not None:
                    break
            # Fallback: look forward
            if probe_num is None:
                for j in range(i + 1, min(len(self.df), i + 200)):
                    rec_j = self.df.loc[j, 'recoded']
                    if isinstance(rec_j, str) and rec_j.startswith(('go', 'nogo')):
                        for part in rec_j.split('/'):
                            if part.startswith('probe') and part[5:].isdigit():
                                probe_num = int(part[5:])
                                break
                    if probe_num is not None:
                        break

            # Second fallback: derive from s36 cumulative count up to end_idx
            if probe_num is None and end_idx is not None:
                probe_num = s36_cumsum[end_idx]

            ordered_keys = ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']
            parts = [values[k] for k in ordered_keys if values[k] is not None]
            if probe_num is not None:
                parts.append(f"probe{probe_num}")

            if parts:
                self.df.loc[i, 'recoded'] = 'THOUGHT_PROBE/' + '/'.join(parts)
    
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

            # Now parse distance, probe number, and trial number in a position-agnostic way
            # so that additional tokens (e.g., 'ontask'/'offtask') do not break parsing
            if id_base in [1, 2]:  # Only for go/nogo trials
                 # Distance: look for last signed numeric token like '+N' or '-N'
                 for token in reversed(probe_parts):
                     if (isinstance(token, str) and len(token) > 1 and (token[0] in ['+', '-']) and token[1:].isdigit()):
                         distance_to_probe = abs(int(token))
                         break

                 # Probe number: find 'probeN' token
                 for token in probe_parts:
                     if token.startswith('probe') and token[5:].isdigit():
                         probe_number = int(token[5:])
                         break

                 # Trial number: take the last pure-digit token
                 for token in reversed(probe_parts):
                     if token.isdigit():
                         trial_number = int(token)
                         break
            
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
        # 0) Add BAD_rest annotations before any recoding so DF includes them
        self.annotate_rest_periods()
        # 0b) Add explicit THOUGHT_PROBE markers at S31
        self.annotate_thought_probe_markers()
        self.recode_annotations()
        # Detect second button presses after initial response recoding
        self.detect_second_button_presses()
        self.retropropagate_tp_values()
        self.retropropagate_tp_to_trials()
        self.propagate_trial_info()
        self.fix_probe_distances()  # New step to ensure proper distance formatting
        # Add aggregated info to THOUGHT_PROBE markers and then label trials
        self.aggregate_thought_probe_info()
        # 5) Append /ontask or /offtask labels based on onoff value (trials only)
        self.add_onoff_label()
        # 6) Append binarized labels for all thought dimensions and gonogo
        self.append_binary_labels()
        # 7) Harmonize annotations to standardized format (probe-N, trial-N, distance-N, ONTASK/OFFTASK)
        self.harmonize_annotations()
        
        # CRITICAL FIX: Write the modified DataFrame back to self.raw.annotations
        # before generating events.
        
        # 1. Prepare onsets (convert from datetime if needed)
        onsets = self.df['onset'].values
        
        # Check if onsets are datetime-like and need conversion to float seconds
        needs_conversion = False
        if len(onsets) > 0:
            first_val = onsets[0]
            if hasattr(first_val, 'total_seconds'):
                needs_conversion = True
            elif np.issubdtype(type(first_val), np.datetime64):
                needs_conversion = True
            elif hasattr(first_val, 'timestamp'):  # pandas Timestamp
                needs_conversion = True
        
        if needs_conversion:
            orig_time = self.raw.annotations.orig_time
            
            # Convert to float seconds relative to start of recording
            onsets_list = []
            for t in self.df['onset']:
                try:
                    # If t is a pandas Timestamp or datetime
                    if hasattr(t, 'timestamp'):
                        # Get Unix timestamp in seconds
                        t_seconds = t.timestamp()
                        
                        if orig_time is not None and hasattr(orig_time, 'timestamp'):
                            # Subtract orig_time to get relative offset
                            orig_seconds = orig_time.timestamp()
                            onsets_list.append(t_seconds - orig_seconds)
                        else:
                            # No orig_time, assume t is already a reasonable timestamp
                            # MNE may use Unix epoch, so subtract epoch
                            epoch = pd.Timestamp("1970-01-01", tz='UTC').timestamp()
                            onsets_list.append(t_seconds - epoch)
                    elif hasattr(t, 'total_seconds'):
                        # It's a timedelta
                        onsets_list.append(t.total_seconds())
                    elif np.issubdtype(type(t), np.datetime64):
                        # numpy datetime64 - convert to float seconds from epoch
                        t_ns = np.datetime64(t, 'ns').astype(float)
                        t_seconds = t_ns / 1e9
                        onsets_list.append(t_seconds)
                    else:
                        # Already a float or int
                        onsets_list.append(float(t))
                except Exception:
                    # If all else fails, try to directly cast
                    onsets_list.append(float(t) if isinstance(t, (int, float)) else 0.0)
            
            onsets = np.array(onsets_list)
        
        # 2. Create new Annotations object
        new_annots = mne.Annotations(
            onset=onsets,
            duration=self.df['duration'].values,
            description=self.df['recoded'].values, # Use the RECODED description
            orig_time=self.raw.annotations.orig_time
        )
        
        # 3. Update raw
        self.raw.set_annotations(new_annots)
        
        # Build events directly from annotations to avoid length mismatch issues
        events, event_id = mne.events_from_annotations(self.raw, event_id='auto')
        return events, event_id
