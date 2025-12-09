#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract SART behavioral trial data from BIDS raw events.tsv files.

This script reads the events.tsv files from BIDS raw data and extracts
trial-level behavioral information including stimulus type (go/nogo),
responses, reaction times, correctness, and associated probe information.

Author: Analysis Assistant
Date: 2025-10-15
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
# ROOT = "/network/iss/"
ROOT = "/Volumes/"
BIDS_RAW_ROOT =  ROOT + "cenir/analyse/meeg/CYBERSART/BIDS/raw"
BIDS_OUTPUT_ROOT = ROOT + "cenir/analyse/meeg/CYBERSART/BIDS/raw"
SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
            "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23",
            "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
            "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]
# =============================================================================


def parse_trial_type(trial_type: str) -> Dict[str, any]:
    """
    Parse a trial_type string to extract all relevant information.
    
    Parameters
    ----------
    trial_type : str
        The trial_type string from events.tsv
        
    Returns
    -------
    Dict[str, any]
        Dictionary with extracted information
        
    Notes
    -----
    Expected format: "go/correct/onoff99/selfother81/valence97/time46/confidence99/..."
    """
    if not isinstance(trial_type, str):
        return {}
    
    info = {}
    
    # Split by '/' to get components
    components = trial_type.split('/')
    
    # Extract trial class (go/nogo)
    if 'go' in components:
        info['trial_class'] = 'go'
    elif 'nogo' in components:
        info['trial_class'] = 'nogo'
    
    # Extract correctness marker (if present in trial_type)
    if 'correct' in components:
        info['correct_marker'] = 'correct'
    elif 'incorrect' in components:
        info['correct_marker'] = 'incorrect'
    
    # Extract numeric values using regex
    for component in components:
        # Probe ratings
        if component.startswith('onoff'):
            match = re.search(r'onoff(\d+)', component)
            if match:
                info['onoff'] = float(match.group(1))
        elif component.startswith('selfother'):
            match = re.search(r'selfother(\d+)', component)
            if match:
                info['selfother'] = float(match.group(1))
        elif component.startswith('valence'):
            match = re.search(r'valence(\d+)', component)
            if match:
                info['valence'] = float(match.group(1))
        elif component.startswith('time'):
            match = re.search(r'time(\d+)', component)
            if match:
                info['time'] = float(match.group(1))
        elif component.startswith('confidence'):
            match = re.search(r'confidence(\d+)', component)
            if match:
                info['confidence'] = float(match.group(1))
        
        # Distance to probe (negative number)
        elif re.match(r'^-?\d+$', component):
            info['distance_to_probe'] = int(component)
        
        # Probe number
        elif component.startswith('probe'):
            match = re.search(r'probe(\d+)', component)
            if match:
                info['probe_number'] = int(match.group(1))
    
    return info


def parse_response_info(trial_type: str) -> Dict[str, any]:
    """
    Parse response information from a response trial_type string.
    
    Parameters
    ----------
    trial_type : str
        The trial_type string for a response event
        
    Returns
    -------
    Dict[str, any]
        Dictionary with response information (RT, correctness)
        
    Notes
    -----
    Expected format: "response/correct/rt551" or "response_2nd/correct/rt600"
    """
    if not isinstance(trial_type, str) or 'response' not in trial_type:
        return {}
    
    info = {'is_response': True}
    
    # Check if this is a second button press
    if 'response_2nd' in trial_type:
        info['is_second_press'] = True
    else:
        info['is_second_press'] = False
    
    # Extract RT
    rt_match = re.search(r'rt(\d+)', trial_type)
    if rt_match:
        info['rt'] = int(rt_match.group(1))
    
    # Extract correctness
    if 'correct' in trial_type:
        info['response_correct'] = True
    elif 'incorrect' in trial_type:
        info['response_correct'] = False
    
    return info


def extract_trials_from_events_file(events_file: Path, subject_id: str, task: str) -> pd.DataFrame:
    """
    Extract trial-level behavioral data from a single events.tsv file.
    
    Parameters
    ----------
    events_file : Path
        Path to the events.tsv file
    subject_id : str
        Subject identifier
    task : str
        Task identifier (e.g., 'Sart1')
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing trial-level behavioral data
        
    Notes
    -----
    Matches stimulus events with response events to compute RTs and correctness.
    """
    if not events_file.exists():
        print(f"Warning: Events file not found: {events_file}")
        return pd.DataFrame()
    
    try:
        # Read the events file
        df_events = pd.read_csv(events_file, sep='\t')
        
        # Separate stimulus and response events
        stimulus_mask = df_events['trial_type'].str.contains(r'\b(go|nogo)\b', na=False, regex=True)
        response_mask = df_events['trial_type'].str.contains('response', na=False)
        
        df_stimuli = df_events[stimulus_mask].copy()
        df_responses = df_events[response_mask].copy()
        
        if len(df_stimuli) == 0:
            print(f"Warning: No stimulus events found in {events_file}")
            return pd.DataFrame()
        
        # Parse stimulus information
        df_stimuli['parsed_info'] = df_stimuli['trial_type'].apply(parse_trial_type)
        
        # Parse response information
        df_responses['parsed_response'] = df_responses['trial_type'].apply(parse_response_info)
        
        # Build trial data
        trials = []
        
        for idx, stim_row in df_stimuli.iterrows():
            stim_onset = stim_row['onset']
            stim_info = stim_row['parsed_info']
            
            # Get stimulus timestamp (use onset time)
            stim_timestamp = stim_onset
            
            # Find matching responses (within reasonable time window, e.g., 2 seconds)
            # Get ALL responses after this stimulus
            matching_responses = df_responses[
                (df_responses['onset'] > stim_onset) & 
                (df_responses['onset'] <= stim_onset + 2.0)
            ].copy()
            
            # Separate first and second button presses
            first_presses = matching_responses[
                ~matching_responses['parsed_response'].apply(lambda x: x.get('is_second_press', False))
            ]
            second_presses = matching_responses[
                matching_responses['parsed_response'].apply(lambda x: x.get('is_second_press', False))
            ]
            
            # Get the first response after this stimulus
            if len(first_presses) > 0:
                resp_row = first_presses.iloc[0]
                resp_info = resp_row['parsed_response']
                has_response = True
                rt = resp_info.get('rt', None)
                response_timestamp = resp_row['onset']
                
                # Check for second button press
                if len(second_presses) > 0:
                    resp_2nd_row = second_presses.iloc[0]
                    resp_2nd_info = resp_2nd_row['parsed_response']
                    rt_2nd = resp_2nd_info.get('rt', None)
                    response_timestamp_2nd = resp_2nd_row['onset']
                else:
                    rt_2nd = None
                    response_timestamp_2nd = None
            else:
                has_response = False
                rt = None
                response_timestamp = None
                rt_2nd = None
                response_timestamp_2nd = None
            
            # Determine correctness
            trial_class = stim_info.get('trial_class', None)
            if trial_class == 'go':
                correct = has_response  # Correct if responded
            elif trial_class == 'nogo':
                correct = not has_response  # Correct if did NOT respond
            else:
                correct = None
            
            # Build trial dictionary
            trial = {
                'subject': int(subject_id),
                'sart': task,
                'trial_number': len(trials) + 1,  # Sequential trial number
                'distance_to_probe': stim_info.get('distance_to_probe', None),
                'trial_class': trial_class,
                'response': has_response,
                'rt': rt,
                'correct': correct,
                'probe_number': stim_info.get('probe_number', None),
                'onoff': stim_info.get('onoff', None),
                'valence': stim_info.get('valence', None),
                'time': stim_info.get('time', None),
                'selfother': stim_info.get('selfother', None),
                'confidence': stim_info.get('confidence', None),
                # New columns for timestamps and second button presses
                'stim_timestamp': stim_timestamp,
                'response_timestamp': response_timestamp,
                'rt_2nd': rt_2nd,
                'response_timestamp_2nd': response_timestamp_2nd
            }
            
            trials.append(trial)
        
        df_trials = pd.DataFrame(trials)
        
        # Fix distance_to_probe: recalculate as distance to NEXT probe
        # The raw distance values count from the previous probe, we need to flip this
        if len(df_trials) > 0 and 'probe_number' in df_trials.columns:
            # Find all probe boundaries (where probe_number changes)
            probe_boundaries = []
            for i in range(len(df_trials) - 1):
                if df_trials.iloc[i]['probe_number'] != df_trials.iloc[i + 1]['probe_number']:
                    probe_boundaries.append(i + 1)  # Index where new probe starts
            
            # For each trial, calculate distance to the next probe boundary
            corrected_distances = []
            for i in range(len(df_trials)):
                # Find the next probe boundary after this trial
                next_boundary = None
                for boundary in probe_boundaries:
                    if boundary > i:
                        next_boundary = boundary
                        break
                
                if next_boundary is not None:
                    # Distance is negative, counting down to the next probe
                    corrected_distances.append(-(next_boundary - i))
                else:
                    # This is in the last probe block - count from current position to end
                    # These trials lead up to the final probe
                    trials_to_end = len(df_trials) - i
                    corrected_distances.append(-trials_to_end)
            
            df_trials['distance_to_probe'] = corrected_distances
        
        print(f"Extracted {len(df_trials)} trials from {subject_id}/{task}")
        return df_trials
        
    except Exception as e:
        print(f"Error processing events file {events_file}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def extract_all_trials() -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Extract trial data from all subjects and tasks (using default config).
    
    Returns
    -------
    Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (subject_id, task) to trial DataFrames
        
    Notes
    -----
    Returns a dictionary to allow per-subject/task saving in BIDS format.
    """
    return extract_all_trials_custom(BIDS_RAW_ROOT, SUBJECTS, TASKS)


def extract_all_trials_custom(bids_raw_root: str, subjects: List[str], tasks: List[str]) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Extract trial data from specified subjects and tasks.
    
    Parameters
    ----------
    bids_raw_root : str
        Path to BIDS raw data root
    subjects : List[str]
        List of subject IDs to process
    tasks : List[str]
        List of tasks to process
    
    Returns
    -------
    Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (subject_id, task) to trial DataFrames
        
    Notes
    -----
    Returns a dictionary to allow per-subject/task saving in BIDS format.
    """
    all_trials = {}
    
    for subject_id in subjects:
        subject_dir = Path(bids_raw_root) / f"sub-{subject_id}" / "eeg"
        
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        
        for task in tasks:
            events_file = subject_dir / f"sub-{subject_id}_task-{task}_events.tsv"
            
            df_trials = extract_trials_from_events_file(events_file, subject_id, task)
            
            if not df_trials.empty:
                all_trials[(subject_id, task)] = df_trials
    
    return all_trials


def save_trials_bids_format(all_trials: Dict[Tuple[str, str], pd.DataFrame], bids_output_root: str = None) -> None:
    """
    Save trial data in BIDS format: sub-XX/beh/sub-XX_task-SartX.csv
    
    Parameters
    ----------
    all_trials : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (subject_id, task) to trial DataFrames
    bids_output_root : str, optional
        Path to BIDS output root (defaults to BIDS_OUTPUT_ROOT)
        
    Notes
    -----
    Creates the beh directory structure and saves CSV files following BIDS conventions.
    """
    if bids_output_root is None:
        bids_output_root = BIDS_OUTPUT_ROOT
    bids_root = Path(bids_output_root)
    
    for (subject_id, task), df_trials in all_trials.items():
        # Create subject beh directory
        beh_dir = bids_root / f"sub-{subject_id}" / "beh"
        beh_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output file
        output_file = beh_dir / f"sub-{subject_id}_task-{task}.csv"
        
        # Save to CSV
        df_trials.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")


def validate_extracted_data(all_trials: Dict[Tuple[str, str], pd.DataFrame]) -> None:
    """
    Validate the extracted trial data and print summary statistics.
    
    Parameters
    ----------
    all_trials : Dict[Tuple[str, str], pd.DataFrame]
        Dictionary mapping (subject_id, task) to trial DataFrames
        
    Notes
    -----
    Checks data integrity and prints useful summary information.
    """
    if not all_trials:
        print("Error: No trial data extracted!")
        return
    
    # Concatenate all trials for summary
    df_all = pd.concat(all_trials.values(), ignore_index=True)
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total trials extracted: {len(df_all)}")
    print(f"Unique subjects: {df_all['subject'].nunique()}")
    print(f"Unique tasks: {df_all['sart'].nunique()}")
    print(f"Total subject-task combinations: {len(all_trials)}")
    
    # Trials per subject/task
    trials_per_file = [len(df) for df in all_trials.values()]
    print(f"\nTrials per file: mean={np.mean(trials_per_file):.1f}, "
          f"std={np.std(trials_per_file):.1f}, "
          f"range=[{np.min(trials_per_file)}, {np.max(trials_per_file)}]")
    
    # Trial class distribution
    print(f"\n=== TRIAL CLASS DISTRIBUTION ===")
    trial_class_counts = df_all['trial_class'].value_counts()
    for trial_class, count in trial_class_counts.items():
        pct = 100 * count / len(df_all)
        print(f"  {trial_class}: {count} ({pct:.1f}%)")
    
    # Response rate
    response_rate = df_all['response'].mean() * 100
    print(f"\n=== RESPONSE STATISTICS ===")
    print(f"Overall response rate: {response_rate:.1f}%")
    
    # Accuracy
    accuracy = df_all['correct'].mean() * 100
    print(f"Overall accuracy: {accuracy:.1f}%")
    
    # RT statistics (for trials with responses)
    df_with_rt = df_all[df_all['rt'].notna()]
    if len(df_with_rt) > 0:
        print(f"\nReaction time (ms): mean={df_with_rt['rt'].mean():.1f}, "
              f"std={df_with_rt['rt'].std():.1f}, "
              f"median={df_with_rt['rt'].median():.1f}")
    
    # Check for missing values in key columns
    print(f"\n=== MISSING VALUES ===")
    key_cols = ['trial_class', 'response', 'correct', 'probe_number']
    for col in key_cols:
        missing = df_all[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} ({100*missing/len(df_all):.1f}%)")
        else:
            print(f"  {col}: 0")
    
    # Probe information coverage
    probe_cols = ['onoff', 'valence', 'time', 'selfother', 'confidence']
    print(f"\n=== PROBE INFORMATION COVERAGE ===")
    for col in probe_cols:
        non_missing = df_all[col].notna().sum()
        pct = 100 * non_missing / len(df_all)
        print(f"  {col}: {non_missing} trials ({pct:.1f}%)")


def main():
    """
    Main execution function.
    
    Notes
    -----
    Coordinates the entire trial extraction process from BIDS raw data.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Extract SART behavioral trial data from BIDS events.tsv files'
    )
    parser.add_argument('--bids-raw-root', default=BIDS_RAW_ROOT,
                        help='Path to BIDS raw data root')
    parser.add_argument('--bids-output-root', default=BIDS_OUTPUT_ROOT,
                        help='Path to BIDS output root')
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS,
                        help='List of subject IDs to process')
    parser.add_argument('--tasks', nargs='+', default=TASKS,
                        help='List of tasks to process')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: process only first subject/task')
    
    args = parser.parse_args()
    
    # Use arguments
    bids_raw_root = args.bids_raw_root
    bids_output_root = args.bids_output_root
    subjects = args.subjects[:1] if args.test_mode else args.subjects
    tasks = args.tasks[:1] if args.test_mode else args.tasks
    
    print("=== SART TRIAL EXTRACTION FROM BIDS RAW DATA ===")
    print(f"BIDS Raw Root: {bids_raw_root}")
    print(f"BIDS Output Root: {bids_output_root}")
    print(f"Processing {len(subjects)} subjects and {len(tasks)} tasks")
    if args.test_mode:
        print("[TEST MODE: Processing only first subject/task]")
    
    # Extract all trial data
    print(f"\nExtracting trial data...")
    all_trials = extract_all_trials_custom(bids_raw_root, subjects, tasks)
    
    if not all_trials:
        print("Error: No trial data was extracted. Please check the input paths and data.")
        return
    
    # Validate the data
    validate_extracted_data(all_trials)
    
    # Save in BIDS format
    print(f"\n=== SAVING DATA IN BIDS FORMAT ===")
    save_trials_bids_format(all_trials, bids_output_root)
    
    # Show example from first file
    first_key = list(all_trials.keys())[0]
    df_example = all_trials[first_key]
    print(f"\n=== EXAMPLE: First 10 trials from sub-{first_key[0]}/{first_key[1]} ===")
    print(df_example.head(10).to_string(index=False))
    
    print(f"\n=== EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
