import mne
import numpy as np
import re

import plotly.graph_objects as go
import plotly.express as px

import sys
sys.path.insert(0, './')
from utils.bids_compliance import load_evokeds


def filter_epochs_by_distance_to_probe(epochs, n):
    list_trials = [f"-{i}" for i in range(1, n+1)]
    # print(list_trials)
    return epochs[list_trials]

def classify_onoff_epochs(epochs, event_id_prefix="onoff", split="median"):
    """
    Classify epochs as 'on-task' or 'off-task' based on onoffXX values in the event names,
    or into quartiles/tertiles based on the specified split type.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The MNE Epochs object containing event annotations.
    event_id_prefix : str, optional
        The prefix used for on-task/off-task classification, default is "onoff".
    split : str, optional
        Specify the method to split the data. Options are:
        - "median" (default): Split by the median.
        - "mean": Split by the mean.
        - "quartiles": Split into four groups (quartiles).
        - "tertiles": Split into three groups (tertiles).
    
    Returns:
    --------
    epochs : mne.Epochs
        The modified Epochs object with event names updated based on the chosen split method.
    """
    event_dict = epochs.event_id

    # Step 1: Extract unique onoffXX values using regex
    onoff_values = []
    onoff_event_names = []
    for event in event_dict.keys():
        if event_id_prefix in event:
            # Use regex to find digits immediately following the prefix (e.g., "onoff")
            match = re.search(rf"{event_id_prefix}(\d+)", event)
            if match:
                onoff_value = int(match.group(1))  # Extract the numeric part
                onoff_values.append(onoff_value)
                onoff_event_names.append(event)

    # Step 2: Split based on the specified method
    if onoff_values:  # Check if there are any valid onoff values
        if split == "mean":
            threshold = np.mean(onoff_values)
            thresholds = [threshold]
        elif split == "median":
            threshold = np.median(onoff_values)
            thresholds = [threshold]
        elif split == "quartiles":
            thresholds = np.percentile(onoff_values, [25, 50, 75])  # 3 thresholds for 4 groups
        elif split == "tertiles":
            thresholds = np.percentile(onoff_values, [33.33, 66.66])  # 2 thresholds for 3 groups
        elif split == "highlow":
            thresholds = [50]
        else:
            raise ValueError(f"Invalid split option '{split}'. Choose from 'median', 'mean', 'quartiles', or 'tertiles'.")

        # Step 3: Create new event names based on the split
        new_event_dict = {}
        for event, onoff_value in zip(onoff_event_names, onoff_values):
            if split in ["median", "mean", "highlow"]:
                if onoff_value >= thresholds[0]:
                    new_event_name = event + "/ontask"
                else:
                    new_event_name = event + "/offtask"
            elif split == "quartiles":
                if onoff_value <= thresholds[0]:
                    new_event_name = event + "/offtask"
                elif onoff_value <= thresholds[1]:
                    new_event_name = event + "/Q2"
                elif onoff_value <= thresholds[2]:
                    new_event_name = event + "/Q3"
                else:
                    new_event_name = event + "/ontask"
            elif split == "tertiles":
                if onoff_value <= thresholds[0]:
                    new_event_name = event + "/offtask"
                elif onoff_value <= thresholds[1]:
                    new_event_name = event + "/T2"
                else:
                    new_event_name = event + "/ontask"

            # Add the new event name to the dictionary, replacing the old one
            new_event_dict[new_event_name] = event_dict[event]

        # Step 4: Replace the old event names with the new ones
        epochs.event_id = new_event_dict

        # Step 5: Return the modified epochs object
        return epochs

    else:
        raise ValueError("No valid onoff values found in the event descriptions.")


def compute_grand_averages(
    subjects,
    data,
    conditions_of_interest,
    derivatives_folder,
    metric=None,
    # sfreq=500,
    # desired_tmin=-0.3,
    # desired_tmax=1.2,
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

    import os
    import mne
    import numpy as np

    participant_evokeds = {}

    for subject in subjects:
        participant_evokeds[subject] = {condition: [] for condition in conditions_of_interest} 
        # Load the evoked objects
        try:
            evokeds = load_evokeds(derivatives_folder, subject, data, metric)
        except Exception as e:
            print(
                f"Could not read evoked file for subject {subject}: {e}"
            )
            continue

        for evoked in evokeds:
            evoked_components = evoked.comment.replace(' × ', '/').split('/')

            for condition in conditions_of_interest:
                condition_components = condition.split('/')
                if all(comp in evoked_components for comp in condition_components):
                        participant_evokeds[subject][condition].append(evoked)

    return participant_evokeds


def plot_erp(evokeds,conditions_of_interest, roi, return_fig = False):       
    # Prepare the figure
    fig = go.Figure()

    # Line and style dictionaries
    color_dict = {'ontask': '#de237b', 'offtask': '#42b9b2'}
    linestyle_dict = {'go': 'solid', 'nogo': 'dash'}

    # Loop over each condition in conditions_of_interest
    for condition in conditions_of_interest:
        condition_data = []
        
        # Loop over each participant in participant_evokeds
        for participant_id, conditions in evokeds.items():
            # Check if the condition is available and if evoked_list is non-empty for the participant
            if condition in conditions and conditions[condition]:
                evoked_list = conditions[condition]
                
                # Get indices of electrodes that match roi
                roi_idx = [evoked_list[0].ch_names.index(ch) for ch in roi if ch in evoked_list[0].ch_names]
                
                # Stack data to compute mean and error across evoked_list (only for ROI channels)
                data = np.stack([evoked.data[roi_idx].mean(axis=0) for evoked in evoked_list])  # Average across ROI channels
                condition_data.append(data)
        
        # Aggregate across participants for this condition if data exists
        if condition_data:
            data = np.vstack(condition_data)
            time = evoked_list[0].times  # Time axis (assuming it's the same across all evokeds)
            mean_values = data.mean(axis=0)  # Mean across trials
            std_error = data.std(axis=0) / np.sqrt(data.shape[0])  # Standard error
            ci95_lower = mean_values - (1.96 * std_error)
            ci95_upper = mean_values + (1.96 * std_error)
            
            # Plot the mean line
            condition_key = condition.split('/')[-1]  # Get 'ontask' or 'offtask'
            fig.add_trace(go.Scatter(
                x=time,
                y=mean_values,
                mode='lines',
                name=condition_key,
                line=dict(
                    color=color_dict[condition_key],
                    dash=linestyle_dict.get(condition.split('/')[0], 'solid')  # Line style based on go/nogo
                )
            ))

            # Plot the confidence interval
            fig.add_trace(go.Scatter(
                x=np.concatenate([time, time[::-1]]),
                y=np.concatenate([ci95_upper, ci95_lower[::-1]]),
                fill='toself',
                fillcolor=color_dict[condition_key],
                opacity=0.3,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ))

    # Adding a vertical line at time=0
    fig.add_shape(type="line",
                x0=0, y0=-5/1000000, x1=0, y1=5/1000000,
                line=dict(color="black", width=2, dash="dash"))

    fig.add_shape(type="line",
                x0=-.3, y0=0, x1=1.2, y1=0,
                line=dict(color="black", width=1,))

    # Ajustar el diseño, incluyendo la leyenda en la parte inferior
    fig.update_layout(
        width = 1000,
        height = 600,
        template = 'plotly_white',
        font=dict(size=20),
        # title='Evoked Responses by Condition with CI95 Error Bands (Posterior ROI)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (µV)',
        legend_title='Condition',
        plot_bgcolor='white',
        legend=dict(
            orientation="h",  # Hacer que la leyenda esté horizontal
            yanchor="bottom",
            y=-0.3,  # Colocar la leyenda debajo del gráfico
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            showline=True,
            linecolor='gray',
            linewidth=2,
            ticks='outside'
        ),
        yaxis=dict(
            showline=True,
            linecolor='gray',
            linewidth=1,
            ticks='outside'
        )
    )
    
    if return_fig:
        return fig
    else:
        # Mostrar la gráfica
        fig.show()