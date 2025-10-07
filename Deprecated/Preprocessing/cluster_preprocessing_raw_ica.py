# preprocess_subject_task_raw_ica.py

# Import necessary libraries for the preprocessing
import os
import sys

# Add the project root directory to Python path to find the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
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
from utils.bids_compliance import read_raw_custom, save_raw_bids_compliant, save_epoched_bids,make_bids_basename
from utils.preprocessing_helpers import set_chs_montage
from utils.trigger_correction import TriggerCorrector

# For memory optimization
import gc

# Read the subject and task from the command-line arguments
subject_id = sys.argv[1]
subject = f"S0{subject_id}"
task = sys.argv[2]
data = 'eeg'

data_root = "/network/iss/cenir/analyse/meeg/CYBERSART/"

raw_path = os.path.join(data_root,"_RAW_DATA")

# Initialize dataframe
df = pd.DataFrame(columns=['subject', 'task', 'data', 'status'])

print(f"Processing subject {subject} for {task}")

##################################
#####      FOR SAVING        #####
##################################
# Defining the paths for saving results and raw data
derivatives_folder = os.path.join(data_root, "derivatives_nico")
derivative_bids_dir = os.path.join(derivatives_folder, f"sub-{subject_id}", "eeg")
os.makedirs(derivative_bids_dir, exist_ok=True)

# Initialize a report to document the preprocessing steps
report = mne.Report(
    title=f"Preprocessing sub-{subject_id} for {task}", verbose = False
)

# Path to the JSON file where preprocessing details will be stored
json_path = os.path.join(
    derivatives_folder, "logs_preprocessing_details_all_subjects_eeg.json"
)

# Initialize the logging class
log_preprocessing = LogPreprocessingDetails(json_path, subject_id, task)

##################################
########   1.READ RAW   ##########
##################################
raw = read_raw_custom(subject,task, root=raw_path)

#set channel montage
raw = set_chs_montage(raw)

# import bad chs from another task of same session
# Important to check if also bad!!!
raw.info["bads"] = log_preprocessing.import_bad_channels_another_task()

print(raw.info)

# Add the raw data info to the report
report.add_raw(raw=raw, title="Raw", psd=True)

# Log the raw data info
log_preprocessing.log_detail("info", str(raw.info))

##################################
########   2.FILTERING   #########
##################################
# Apply a band-pass filter to keep frequencies between 1 and 45 Hz
hpass = 0.5
lpass = 45
raw_filtered = raw.load_data().copy().notch_filter(np.arange(50, 250, 50)).filter(l_freq=hpass, h_freq=lpass)

# Free memory by removing raw data that's no longer needed
del raw
gc.collect()

# Log the filter settings
log_preprocessing.log_detail("hpass_filter", hpass)
log_preprocessing.log_detail("lpass_filter", lpass)
log_preprocessing.log_detail("filter_type", "bandpass")

##################################
### 3.Visual inspection of CHs ###
##################################
# Automatically mark bad channels
nd = NoisyChannels(raw_filtered, do_detrend=False, random_state=42)
nd.find_all_bads(ransac=True, channel_wise=True) #if it slows down, set channel_wise to False
bads = nd.get_bads()
print(f"Bad channels detected: {bads}")
if bads != None:
    raw_filtered.info["bads"] = bads

# Add the filtered data to the report
report.add_raw(raw=raw_filtered, title="Filtered Raw", psd=True)

# Log the identified bad channels
log_preprocessing.log_detail("bad_channels", raw_filtered.info["bads"])

##################################
########   4. ICA ON RAW   #######
##################################
# Parameters for ICA (Independent Component Analysis) to remove artifacts
n_components = 0.99
method = "infomax"  # The algorithm to use for ICA
max_iter = "auto"  # Maximum number of iterations
random_state = 42  # Seed for random number generator for reproducibility

# Initialize the ICA object with the specified parameters
ica = mne.preprocessing.ICA(
    n_components=n_components,
    method=method,
    max_iter=max_iter,
    random_state=random_state,
    fit_params=dict(extended=True),
)

# Fit the ICA model to the filtered raw data
ica.fit(raw_filtered)

# find EOG artifacts in the data via pattern matching, and exclude the EOG-related ICA components
eog_components, eog_scores = ica.find_bads_eog(
    inst=raw_filtered,
    ch_name=["VEOG","HEOG"]  # a channel close to the eye
)
print(f"EOG components detected: {eog_components}")

# find muscle artifacts in the data via pattern matching, and exclude the muscle-related ICA components
muscle_components, muscle_scores = ica.find_bads_muscle(raw_filtered, threshold=0.7)
print(f"Muscle components detected: {muscle_components}")

# Combine all artifact components from the pattern matching methods
pattern_matching_artifacts = np.unique(eog_components + muscle_components)

##### Classify the components using ICLabel model #######
# run the model on the ICA components
ic_labels = label_components(raw_filtered, ica, method="iclabel")
# print labels of each component
print("Classification of all ICA components. Results:")
print(ic_labels["labels"])

# Extract ICA component labels
label_names = ic_labels['labels']

# Identify the ICA components that correspond to a 'channel noise' in ICLabel
channel_artifact_indices = [i for i, label in enumerate(label_names) if label == 'channel noise']
cardiac_artifact_indices = [i for i, label in enumerate(label_names) if label == 'heart']

# Find components that coincide between pattern matching and ICLabel output for exclusion
to_exclude = []
for idx in pattern_matching_artifacts:
    if label_names[idx] in ['muscle artifact', 'eye blink', 'heart beat', 'channel noise']:
        to_exclude.append(idx)
        
if len(eog_components) > 0 and eog_components[0] < 3:
    to_exclude.append(eog_components[0])

# Also ensure to include 'channel noise' components that were found only by ICLabel
to_exclude = np.unique(to_exclude + channel_artifact_indices + cardiac_artifact_indices)

# Exclude the selected components
ica.exclude = to_exclude.tolist()

# Add the ICA results to the report
report.add_ica(ica, title="ICA", inst=raw_filtered)

# Apply the ICA solution to the filtered raw data
raw_ica = ica.apply(inst=raw_filtered.copy())

# Log the ICA parameters and excluded components
log_preprocessing.log_detail("ica_components", ica.exclude)
log_preprocessing.log_detail("ica_method", method)
log_preprocessing.log_detail("ica_max_iter", max_iter)
log_preprocessing.log_detail("ica_random_state", random_state)

# Free memory by removing data that's no longer needed
del raw_filtered
gc.collect()

##################################
#######   5. REREFERENCE   #######
##################################
raw_ica = mne.add_reference_channels(raw_ica.load_data(), ref_channels=["FCz"])

# Path to your .bvef file
current_dir = os.path.dirname(os.path.abspath(__file__))  # Preprocessing directory
bvef_file_path = os.path.join(current_dir, 'CACS-64_withREF.bvef')

# Check if file exists
if not os.path.exists(bvef_file_path):
    raise FileNotFoundError(f"Could not find montage file at {bvef_file_path}")

# Load the extended montage
montage = mne.channels.read_custom_montage(bvef_file_path)

# Apply the montage to your raw data
raw_ica.set_montage(montage)

# Rereference the data to the grand average reference
raw_rereferenced, ref_data = mne.set_eeg_reference(
    inst=raw_ica, ref_channels="average", copy=True
)

# Log the rereferencing details
log_preprocessing.log_detail("rereferenced_channels", "grand_average")

del raw_ica
gc.collect()

##################################
######   6. INTERPOLATE CHS  #####
##################################
# Interpolate bad channels in the raw data
raw_interpolated = raw_rereferenced.copy().interpolate_bads()

# Log the interpolated channels
log_preprocessing.log_detail("interpolated_channels", raw_rereferenced.info["bads"])

# Add the rereference and interpolated raw data to the report
report.add_raw(raw=raw_interpolated, title="Raw rereferenced and interpolated", psd=True)

del raw_rereferenced
gc.collect()

##################################
########   7. EPOCHING   #########
##################################
##################################
######    LOAD TRIGGGERS   #######
##################################
# recode annotations
processor = TriggerCorrector(raw_interpolated)
# get the recoded events and event_id
events, event_id = processor.process_annotations()
#filter out all non go nogo events
# Filter the event_id for 'go' and 'nogo' events
go_nogo_event_id = {key: value for key, value in event_id.items() if 'go' in key or 'nogo' in key}

# Segment the continuous data into epochs of 2 seconds
tmin = -0.3
tmax = 1.2
# Apply baseline correction using the pre-stimulus period
epochs = mne.Epochs(
    raw_interpolated,
    events=events,
    event_id=go_nogo_event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(-0.3, 0),  # Baseline correction from start of epoch to stimulus onset
    preload=True,
    verbose=False,
)

# Add the epochs to the report
report.add_epochs(epochs=epochs, title="Epochs")

# Log the number of epochs and their duration
log_preprocessing.log_detail("n_epochs", len(epochs))
log_preprocessing.log_detail("tmin", tmin)
log_preprocessing.log_detail("tmax", tmax)

# Free memory by removing data that's no longer needed
del raw_interpolated
gc.collect()

##################################
######   8. REJECT EPOCHS   ######
##################################
# Enhanced AutoReject parameters for better performance
n_interpolate = np.array([1, 4, 8, 16, 32])  # Range of channels to interpolate
consensus = np.linspace(0.1, 1.0, 11)  # Range of consensus values to try
cv_folds = 10  # Cross-validation folds

# Automatically reject bad epochs using AutoReject with optimized parameters
ar = AutoReject(
    n_interpolate=n_interpolate,
    consensus=consensus,
    thresh_method="bayesian_optimization", 
    cv=cv_folds, 
    random_state=42, 
    n_jobs=-1,
    picks='eeg'  # Only apply AutoReject to EEG channels
)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
reject = get_rejection_threshold(epochs)

# Log additional AutoReject statistics
n_interpolated_per_epoch = np.sum(reject_log.labels == 1, axis=1)
print(f"Mean interpolated channels per epoch: {np.mean(n_interpolated_per_epoch):.1f}")
print(f"Optimal consensus value: {ar.consensus}")
print(f"Optimal n_interpolate: {ar.n_interpolate}")

# Log the reject log details (handle potential arrays)
consensus_val = ar.consensus if np.isscalar(ar.consensus) else ar.consensus[0]
n_interp_val = ar.n_interpolate if np.isscalar(ar.n_interpolate) else ar.n_interpolate[0]

log_preprocessing.log_detail("autoreject_consensus", float(consensus_val))
log_preprocessing.log_detail("autoreject_n_interpolate", int(n_interp_val))
log_preprocessing.log_detail("mean_interpolated_channels", float(np.mean(n_interpolated_per_epoch)))

# Log the epochs rejected by AutoReject
ar_reject_epochs = [
    n_epoch
    for n_epoch, log in enumerate(epochs_clean.drop_log)
    if log == ("AUTOREJECT",)
]

log_preprocessing.log_detail("autoreject_epochs", ar_reject_epochs)
log_preprocessing.log_detail("autoreject_threshold", reject)
log_preprocessing.log_detail("len_autoreject_epochs", len(ar_reject_epochs))

# Manually inspect and reject bad epochs (optional)
manual_reject_epochs = [
    n_epoch for n_epoch, log in enumerate(epochs_clean.drop_log) if log == ("USER",)
]
print(f"Manually rejected epochs: {manual_reject_epochs}")
total_epochs_rejected = (
    (len(ar_reject_epochs) + len(manual_reject_epochs)) / len(epochs) * 100
)
print(f"Total epochs rejected: {total_epochs_rejected}%")
log_preprocessing.log_detail("manual_reject_epochs", manual_reject_epochs)
log_preprocessing.log_detail("len_manual_reject_epochs", len(manual_reject_epochs))

# Add the cleaned epochs to the report
report.add_epochs(epochs=epochs_clean, title="Epochs clean", psd=True)

# Save the cleaned epochs
epochs_clean.drop_bad()

# Log the epochs dropped
log_preprocessing.log_detail("epochs_drop_log", epochs_clean.drop_log)
log_preprocessing.log_detail("epochs_drop_log_description", "Final drop log after AutoReject")

del epochs
gc.collect()

##################################
#####   SAVE PREPROCESSED   ######
##################################
# Save the final processed epochs
save_epoched_bids(
    epochs_clean,
    derivatives_folder,
    subject_id,
    task,
    data,
    desc="autoPreproc",
    events=events,
    event_id=event_id,
)

# Create P300 evoked response for the report
p300_evoked = mne.combine_evoked([epochs_clean['go/correct'].average(), epochs_clean['nogo/correct'].average()], weights=[1,-1])
report.add_evokeds(
    evokeds=[p300_evoked],  # List of evoked
    titles=["Evoked P300 Go/Nogo"],  # List of titles
)

# Save the report as an HTML file
html_report_fname = make_bids_basename(
    subject=subject_id,
    task=task,
    suffix=data,
    extension=".html",
    desc="autoPreprocReport",
)
report.save(os.path.join(derivative_bids_dir, html_report_fname), open_browser=False, overwrite=True)

# Save the preprocessing details to the JSON file
log_preprocessing.save_preprocessing_details()

# Save the final status of the preprocessing to the dataframe
df.loc[len(df)] = {"subject": subject_id, "task": task, "data": 'eeg', "status": 'preprocessed'}

df.to_csv(os.path.join(derivatives_folder,"preprocessing_status.csv"), index=False)

print(f"Preprocessing completed for subject {subject_id}, task {task}") 