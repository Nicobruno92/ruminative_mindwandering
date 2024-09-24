#%%
# Import necessary libraries for the preprocessing
import os
from git import Repo
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne_bids import BIDSPath, read_raw_bids #,print_dir_tree
# Importing libraries for automatic rejection of bad epochs
from autoreject import AutoReject, get_rejection_threshold
from pyprep import NoisyChannels

# tag automatically ICA components
# requires pytorch
from mne_icalabel import label_components

# Import helper functions for preprocessing
from utils.log_preprocessing import LogPreprocessingDetails
from utils import bids_compliance

"""
The following script performs EEG data preprocessing through several steps:
1. Read raw file
2. Band pass filter 1 to 45hz
3. Crop signal from tmin to tmax
4. Visual inspection of channels. Drop bads
5. Epochs of -0.3s to 1.2s
6. Autoreject Epochs
7. Manual inspection of Epochs
8. ICA
9. Interpolate bad channels
10. Rereferenced to grand average
"""

df = pd.DataFrame(columns = ['subject', 'session', 'task', 'data', 'status'])

##################################
#####          LOAD          #####
##################################
# Get the current working directory
cwd = os.getcwd()

# Assuming the script is run from within the repository
repo = Repo(os.getcwd(), search_parent_directories=True)
repo_root = repo.git.rev_parse("--show-toplevel")

# Define the file path components
results_folder = os.path.join(repo_root, 'results')
# print_dir_tree(os.path.join(repo_root, results_folder,return_str = False))

subjects = [str(i) if i > 9 else "0"+str(i) for i in range(1, 25)]
sessions = ["a", "b"]
tasks = ['sartauditiva', 'sartvisual']
data = "eeg"


#%%
for subject in subjects:
    for session in sessions:
        for task in tasks:
            print(f"Processing subject {subject} for session {session} and {task}")
            try:
                #convert multi XDF to .vhdr
                read_lsl(results_folder, subject, session)
                # Create a BIDSPath object
                bids_path = BIDSPath(
                    subject=subject,
                    session=session,
                    task=task,
                    datatype=data,
                    suffix=data,
                    extension=".vhdr",
                    root=results_folder,
                )
                print(bids_path)

                ##################################
                #####      FOR SAVING        #####
                ##################################

                # Defining the paths for saving results and raw data
                derivatives_folder = os.path.join(repo_root, "derivatives")
                bids_dir = os.path.join(derivatives_folder, f"sub-{subject}", f"ses-{session}", "eeg")
                os.makedirs(bids_dir, exist_ok=True)

                # Initialize a report to document the preprocessing steps
                report = mne.Report(
                    title=f"Preprocessing sub-{subject} for session {session} and {task}"
                )

                # Path to the JSON file where preprocessing details will be stored
                json_path = os.path.join(
                    repo_root, derivatives_folder, "logs_autopreprocessing_details_all_subjects_eeg.json"
                )

                # Initialize the logging class
                log_preprocessing = LogPreprocessingDetails(json_path, subject, session, task)

                ##################################
                ########   1.READ RAW   ##########
                ##################################

                # Read Raw bids
                raw = read_raw_bids(bids_path)

                # import bad chs from another task of same session
                # Important to check if also bad!!!
                raw.info["bads"] = log_preprocessing.import_bad_channels_another_task()

                print(raw.info)

                # Add the raw data info to the report
                report.add_raw(raw=raw, title="Raw", psd=True)

                # Log the raw data info
                log_preprocessing.log_detail("info", str(raw.info))

                # %% 2.FILTERING
                ##################################
                #         2.FILTERING            #
                ##################################

                # Apply a band-pass filter to keep frequencies between 1 and 45 Hz
                hpass = 0.5
                lpass = 45
                raw_filtered = raw.load_data().copy().notch_filter(np.arange(50, 250, 50)).filter(l_freq=hpass, h_freq=lpass)

                # Save the filtered data
                # bids_path.update(root = derivatives_folder, description = 'filtered')
                # write_raw_bids(raw_filtered, bids_path, format='BrainVision', allow_preload=True, overwrite=True)

                # Log the filter settings
                log_preprocessing.log_detail("hpass_filter", hpass)
                log_preprocessing.log_detail("lpass_filter", lpass)
                log_preprocessing.log_detail("filter_type", "bandpass")

                # Add the filtered data to the report
                report.add_raw(raw=raw_filtered, title="Filtered Raw", psd=True)

                #automatically mark bad channels
                nd = NoisyChannels(raw_filtered,do_detrend = False, random_state=42)
                nd.find_all_bads(ransac=True, channel_wise=True) #if it slows down, set channel_wise to False
                bads = nd.get_bads()
                print(f"Bad channels detected: {bads}")
                if bads != None:
                    raw_filtered.info["bads"] = bads


                # Log the identified bad channels
                log_preprocessing.log_detail("bad_channels", raw_filtered.info["bads"])

                # %%
                ##################################
                ######    LOAD TRIGGGERS   #######
                ##################################
                # Filter annotations by description
                filtered_annotations = mne.Annotations(onset=[], duration=[], description=[])

                for ann in raw_filtered.annotations:
                    if "go/" in ann["description"] or "nogo/" in ann["description"]:
                        filtered_annotations.append(ann["onset"], ann["duration"], ann["description"])

                raw_filtered.set_annotations(filtered_annotations)

                events, event_id = mne.events_from_annotations(raw_filtered)

                # %%
                # 4-EPOCHING
                ##################################
                #########    4.EPOCHS   ##########
                ##################################
                # Segment the continuous data into epochs of 2 seconds
                tmin = -0.3
                tmax = 1.2
                # baseline correction should be done after ICA
                epochs = mne.Epochs(
                    raw_filtered,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    preload=False,
                    verbose=False,
                    baseline = None
                )

                # Save the epoched data
                # bids_compliance.save_epoched_bids(epochs, derivatives_folder, subject, session,
                #                                   task, data, desc = 'epoched', events = events, event_id =event_id)

                # Add the epochs to the report
                report.add_epochs(epochs=epochs, title="Epochs")

                # Log the number of epochs and their duration
                log_preprocessing.log_detail("n_epochs", len(epochs))
                log_preprocessing.log_detail("tmin", tmin)
                log_preprocessing.log_detail("tmax", tmax)

                # %%
                # REJECT EPOCHS
                ##################################
                ######    REJECT EPOCHS   ########
                ##################################

                # TODO: add rejection for acceloremeter (available from sub 14 onwards)
                folds=10 # increase for more accuracy or decrease for speed
                # Automatically reject bad epochs using AutoReject
                ar = AutoReject(thresh_method="bayesian_optimization", cv = folds, random_state=42, n_jobs = -1, )
                epochs_clean = ar.fit_transform(epochs)
                reject = get_rejection_threshold(epochs)

                # Log the epochs rejected by AutoReject
                ar_reject_epochs = [
                    n_epoch
                    for n_epoch, log in enumerate(epochs_clean.drop_log)
                    if log == ("AUTOREJECT",)
                ]

                log_preprocessing.log_detail("autoreject_epochs", ar_reject_epochs)
                log_preprocessing.log_detail("autoreject_threshold", reject)
                log_preprocessing.log_detail("len_autoreject_epochs", len(ar_reject_epochs))

                # Add the cleaned epochs to the report
                report.add_epochs(epochs=epochs_clean, title="Epochs clean", psd=False)

                # Save the cleaned epochs
                epochs_clean.drop_bad()
                # bids_compliance.save_epoched_bids(epochs_clean, derivatives_folder, subject, session,
                #                                   task, data, desc = 'epochedClean', events = events, event_id =event_id)

                # %%
                ##################################
                ######         ICA        ########
                ##################################
                # Parameters for ICA (Independent Component Analysis) to remove artifacts
                n_components = 0.99
                method = "picard"  # The algorithm to use for ICA
                max_iter = (
                    "auto"  # Maximum number of iterations; typically should be higher, like 500 or 1000
                )
                random_state = 42  # Seed for random number generator for reproducibility

                # Initialize the ICA object with the specified parameters
                ica = mne.preprocessing.ICA(
                    n_components=n_components,
                    method=method,
                    max_iter=max_iter,
                    random_state=random_state,
                )

                # Fit the ICA model to the cleaned epochs
                ica.fit(epochs_clean)

                # Find EOG artifacts in the data via pattern matching
                eog_components, eog_scores = ica.find_bads_eog(
                    inst=epochs_clean,
                    ch_name="R_EYE",
                )

                # Find ECG artifacts in the data via pattern matching
                ecg_components, ecg_scores = ica.find_bads_ecg(
                    inst=epochs_clean,
                    ch_name="ECG",
                )

                # Find muscle artifacts in the data via pattern matching
                muscle_components, muscle_scores = ica.find_bads_muscle(epochs_clean, threshold=0.7)

                # Get the labels from ICLabel
                ic_labels = label_components(epochs_clean, ica, method="iclabel")

                # Extract ICA component labels
                label_names = ic_labels['labels']

                # Combine all artifact components from the pattern matching methods
                pattern_matching_artifacts = np.unique(ecg_components + eog_components + muscle_components)

                # Identify the ICA components that correspond to a 'channel noise' in ICLabel
                channel_artifact_indices = [i for i, label in enumerate(label_names) if label == 'channel noise']

                # Find components that coincide between pattern matching and ICLabel output for exclusion
                # We'll only exclude components that match the artifacts found via pattern matching 
                # and are classified as 'muscle artifact', 'eye blink', 'heart beat', or 'channel noise'
                to_exclude = []
                for idx in pattern_matching_artifacts:
                    if label_names[idx] in ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']:
                        to_exclude.append(idx)

                # Also ensure to include 'channel noise' components that were found only by ICLabel
                to_exclude = np.unique(to_exclude + channel_artifact_indices)

                # Exclude the selected components
                ica.exclude = to_exclude.tolist()

                # Print the components being excluded
                print(f"Components being excluded: {ica.exclude}")
                # Add the ICA results to the report
                report.add_ica(ica, title="ICA", inst=epochs_clean)

                # Apply the ICA solution to the cleaned epochs
                epochs_ica = ica.apply(inst=epochs_clean)

                # Log the ICA parameters and excluded components
                log_preprocessing.log_detail("ica_components", ica.exclude)
                log_preprocessing.log_detail("ica_method", method)
                log_preprocessing.log_detail("ica_max_iter", max_iter)
                log_preprocessing.log_detail("ica_random_state", random_state)


                ##### FINAL EPOCH CLEANING #######
                baseline = (-0.3, 0)  # to be done after ICA
                epochs_ica.apply_baseline(baseline)
                log_preprocessing.log_detail("baseline", baseline)


                
                ##################################
                ######    REJECT EPOCHS   ########
                ##################################
                # TODO: add rejection for acceloremeter (available from sub 14 onwards)
                folds=10 # increase for more accuracy or decrease for speed
                # Automatically reject bad epochs using AutoReject
                ar = AutoReject(thresh_method="bayesian_optimization", cv = folds, random_state=42, n_jobs = -1, )
                epochs_ica_clean = ar.fit_transform(epochs_ica)
                reject = get_rejection_threshold(epochs_ica)

                # Log the epochs rejected by AutoReject
                ar_reject_epochs = [
                    n_epoch
                    for n_epoch, log in enumerate(epochs_ica_clean.drop_log)
                    if log == ("AUTOREJECT",)
                ]

                log_preprocessing.log_detail("autoreject_epochs", ar_reject_epochs)
                log_preprocessing.log_detail("autoreject_threshold", reject)
                log_preprocessing.log_detail("len_autoreject_epochs", len(ar_reject_epochs))

                # Add the cleaned epochs to the report
                report.add_epochs(epochs=epochs_ica_clean, title="Epochs clean after ICA", psd=False)

                # Save the cleaned epochs
                epochs_ica_clean.drop_bad()
                
                                # Log the epochs dropped by ICA
                log_preprocessing.log_detail("epochs_drop_log", epochs_ica_clean.drop_log)
                log_preprocessing.log_detail("epochs_drop_log_description", epochs_ica_clean.drop_log)

                # Save the epochs after ICA application and drop epochs
                # bids_compliance.save_epoched_bids(epochs_ica, derivatives_folder, subject, session,
                #                                   task, data, desc = 'epochedICA', events = events, event_id =event_id)

                # %%
                ##################################
                ######   Interpolate chs  ########
                ##################################
                # Interpolate bad channels in the epochs after ICA application
                epochs_interpolate = epochs_ica_clean.copy().interpolate_bads()

                # Log the interpolated channels
                log_preprocessing.log_detail("interpolated_channels", epochs_ica_clean.info["bads"])

                ##################################
                #######    Rereference   #########
                ##################################
                # Rereference the data to the grand average reference
                epochs_rereferenced, ref_data = mne.set_eeg_reference(
                    inst=epochs_interpolate, ref_channels="average", copy=True
                )

                # Add the final epochs to the report
                report.add_epochs(
                    epochs=epochs_rereferenced, title="Epochs interpolated and rereferenced", psd=True
                )

                # Log the rereferencing details
                log_preprocessing.log_detail("rereferenced_channels", "grand_average")

                #######################################
                #######    SAVE FINAL FILES   #########
                #######################################

                # Save the rereferenced epochs
                bids_compliance.save_epoched_bids(
                    epochs_rereferenced,
                    derivatives_folder,
                    subject,
                    session,
                    task,
                    data,
                    desc="autoPreproc",
                    events=events,
                    event_id=event_id,
                )

                # Save the report as an HTML file
                html_report_fname = bids_compliance.make_bids_basename(
                    subject=subject,
                    session=session,
                    task=task,
                    suffix=data,
                    extension=".html",
                    desc="autoPreprocReport",
                )
                report.save(os.path.join(bids_dir, html_report_fname), open_browser=False,overwrite=True)

                # Save the preprocessing details to the JSON file
                log_preprocessing.save_preprocessing_details()

                df.loc[len(df)] = {"subject": subject, "session": session, "task": task, "data": data, "status": 'preprocessed'}

            except:
                print(f"Error in subject {subject}")
                df.loc[len(df)] = {"subject": subject, "session": session, "task": task, "data": data, "status": 'failed preprocessing'}
                continue
                    
            df.to_csv(os.path.join(repo_root, "derivatives", "preprocessing_status.csv"), index=False)
