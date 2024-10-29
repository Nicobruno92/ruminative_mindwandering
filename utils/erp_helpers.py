import os
import mne
import numpy as np
import pandas as pd

def compute_grand_averages(
    subjects,
    tasks,
    data,
    conditions_of_interest,
    derivatives_folder,
    sfreq=500,
    desired_tmin=-0.3,
    desired_tmax=1.2,
):
    """
    Computes grand average evoked responses for specified conditions.

    Parameters:
    - subjects: list of subject IDs (e.g., ['01', '02', ...])
    - tasks: list of task names (e.g., ['sartauditiva', 'sartvisual'])
    - data: type of data (e.g., 'eeg')
    - conditions_of_interest: list of condition strings to compare (e.g., ['nogo/correct/on-task'])
    - derivatives_folder: path to the derivatives folder
    - sfreq: desired sampling frequency (default: 500 Hz)
    - desired_tmin: desired start time of evoked data (default: -0.3 sec)
    - desired_tmax: desired end time of evoked data (default: 1.2 sec)

    Returns:
    - grand_averages: dictionary with conditions as keys and grand average evoked objects as values
    """

    # Initialize dictionaries to hold participant-level evokeds
    participant_evokeds = {condition: [] for condition in conditions_of_interest}

    for subject in subjects:
        # Initialize a dictionary to hold the evokeds for this subject per condition
        subject_evokeds = {condition: [] for condition in conditions_of_interest}
        for task in tasks:
            # Construct the path to the evoked file
            evoked_fname = os.path.join(
                derivatives_folder,
                f"sub-{subject}",
                data,
                f"sub-{subject}_task-{task}_evokeds-ave.fif",
            )

            # Check if the file exists
            if not os.path.exists(evoked_fname):
                continue  # Skip if the file does not exist

            # Load the evoked objects
            try:
                evokeds = mne.read_evokeds(evoked_fname)
            except Exception as e:
                print(
                    f"Could not read evoked file for subject {subject} task {task}: {e}"
                )
                continue

            # Extract evokeds for conditions of interest using component-wise matching
            for evoked in evokeds:
                # Resample to desired sampling frequency
                evoked.resample(sfreq, npad="auto")

                # Shift time if needed to align with desired_tmin start
                current_tmin = evoked.times[0]
                if current_tmin != desired_tmin:
                    evoked.shift_time(desired_tmin - current_tmin, relative=True)

                # Crop to the same time window for all
                evoked.crop(tmin=desired_tmin, tmax=desired_tmax)

                # Split the evoked comment into components
                evoked_components = evoked.comment.replace(' × ', '/').split('/')

                # Check the conditions using component-wise matching
                for condition in conditions_of_interest:
                    condition_components = condition.split('/')
                    # Check if all condition components are in the evoked comment components
                    if all(comp in evoked_components for comp in condition_components):
                        # Append the evoked to the subject's list for that condition
                        subject_evokeds[condition].append(evoked)

        # Combine evokeds per condition for this subject
        for condition in conditions_of_interest:
            evokeds_list = subject_evokeds[condition]
            if evokeds_list:
                # Combine the evokeds (average)
                combined_evoked = mne.combine_evoked(evokeds_list, weights="nave")
                combined_evoked.comment = condition  # Set the comment to the condition

                # Append the participant-level evoked to the grand list
                participant_evokeds[condition].append(combined_evoked)
            else:
                print(f"No evokeds found for subject {subject}, condition {condition}")

    # Compute grand averages across participants
    grand_averages = {}

    for condition in conditions_of_interest:
        # Get the list of participant-level evokeds for this condition
        evokeds_list = participant_evokeds[condition]

        if evokeds_list:
            # Combine the evokeds (average)
            grand_average_evoked = mne.grand_average(evokeds_list)
            grand_average_evoked.comment = f"{condition}"

            # Store the grand average evoked
            grand_averages[condition] = grand_average_evoked
        else:
            print(f"No participant evokeds found for condition {condition}")

    return grand_averages, participant_evokeds

def compute_participant_evokeds(
    subjects,
    data,
    conditions_of_interest,
    derivatives_folder,
    sfreq=500,
    desired_tmin=-0.3,
    desired_tmax=1.2,
):
    # Initialize a dictionary to hold evokeds per participant per condition and task
    participant_evokeds = {}

    for subject in subjects:
        evoked_fname = os.path.join(
            derivatives_folder,
            f"sub-{subject}",
            data,
            f"sub-{subject}_evokeds-ave.fif",
        )

        if not os.path.exists(evoked_fname):
            continue

        try:
            evokeds = mne.read_evokeds(evoked_fname)
        except Exception as e:
            print(
                f"Could not read evoked file for subject {subject}: {e}"
            )
            continue

        for evoked in evokeds:
            evoked.resample(sfreq, npad="auto")
            current_tmin = evoked.times[0]
            if current_tmin != desired_tmin:
                evoked.shift_time(desired_tmin - current_tmin, relative=True)
            evoked.crop(tmin=desired_tmin, tmax=desired_tmax)

            evoked_components = evoked.comment.replace(' × ', '/').split('/')

            for condition in conditions_of_interest:
                condition_components = condition.split('/')
                if all(comp in evoked_components for comp in condition_components):
                    participant_evokeds[subject][condition].append(evoked)

    return participant_evokeds


def compute_erps(participant_evokeds, subjects, conditions_of_interest, roi, tmin, tmax):
    data_list = []

    for subject in subjects:
            for condition in conditions_of_interest:
                evokeds = participant_evokeds.get(subject, {}).get(condition, [])
                if not evokeds:
                    continue  # Skip if no evokeds for this condition

                # Combine evokeds for this subject, task, and condition
                combined_evoked = mne.combine_evoked(evokeds, weights='equal')

                # Get picks for ROI
                picks = mne.pick_channels(combined_evoked.info['ch_names'], roi)

                # Get time indices for the specified time window
                times = combined_evoked.times
                time_mask = (times >= tmin) & (times <= tmax)

                # Extract data and compute mean amplitude
                data = combined_evoked.data[picks][:, time_mask].mean()

                # Extract immersion level (ordinal variable)
                # Assuming the last component of the condition string is the immersion level
                immersion_level = condition.split('/')[-1]
                # Map immersion level to numerical values
                immersion_mapping = {
                    'a little immersed': 1,
                    'somewhat immersed': 2,
                    'very immersed': 3,
                    'completely immersed': 4
                }
                immersion_value = immersion_mapping.get(immersion_level, np.nan)

                # Extract task status ('on-task' or 'spontaneous')
                task_status = 'on-task' if 'on-task' in condition else 'spontaneous'

                # Append data to the list
                data_list.append({
                    'participant': subject,
                    'condition': task_status,
                    'immersion': immersion_value,
                    'mean_amplitude': data,
                })

    return pd.DataFrame(data_list)