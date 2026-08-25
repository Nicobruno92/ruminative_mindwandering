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
        - "highlow": Split into high and low groups.
    
    Returns:
    --------
    classified_epochs : dict
        A dictionary containing the classified epochs.
    """
    import re
    import numpy as np
    import copy
    
    # Extract onoff values from event names
    onoff_values = []
    for event_name in epochs.event_id.keys():
        match = re.search(f'{event_id_prefix}(\d+)', event_name)
        if match:
            onoff_values.append(int(match.group(1)))
    
    onoff_values = np.array(onoff_values)
    
    if len(onoff_values) == 0:
        raise ValueError(f"No events found with prefix '{event_id_prefix}' in event names.")
    
    # Define classification thresholds based on split type
    if split == "median":
        threshold = np.median(onoff_values)
        conditions = {
            'low': onoff_values <= threshold,
            'high': onoff_values > threshold
        }
    elif split == "mean":
        threshold = np.mean(onoff_values)
        conditions = {
            'low': onoff_values <= threshold,
            'high': onoff_values > threshold
        }
    elif split == "quartiles":
        q25, q75 = np.percentile(onoff_values, [25, 75])
        conditions = {
            'q1': onoff_values <= q25,
            'q2': (onoff_values > q25) & (onoff_values <= np.median(onoff_values)),
            'q3': (onoff_values > np.median(onoff_values)) & (onoff_values <= q75),
            'q4': onoff_values > q75
        }
    elif split == "tertiles":
        q33, q67 = np.percentile(onoff_values, [33, 67])
        conditions = {
            't1': onoff_values <= q33,
            't2': (onoff_values > q33) & (onoff_values <= q67),
            't3': onoff_values > q67
        }
    elif split == "highlow":
        # Use 33rd and 67th percentiles to define high/low, excluding middle
        q33, q67 = np.percentile(onoff_values, [33, 67])
        conditions = {
            'low': onoff_values <= q33,
            'high': onoff_values >= q67
        }
    else:
        raise ValueError(f"Unknown split type: {split}. Use 'median', 'mean', 'quartiles', 'tertiles', or 'highlow'.")
    
    # Create new events and event_id mappings for each condition
    classified_epochs = {}
    
    for condition_name, condition_mask in conditions.items():
        # Create new epochs for this condition
        new_epochs = epochs.copy()
        new_events = []
        new_event_id = {}
        
        for i, (event_name, event_code) in enumerate(epochs.event_id.items()):
            match = re.search(f'{event_id_prefix}(\d+)', event_name)
            if match:
                onoff_value = int(match.group(1))
                onoff_index = np.where(onoff_values == onoff_value)[0]
                
                if len(onoff_index) > 0 and condition_mask[onoff_index[0]]:
                    # This event belongs to the current condition
                    new_event_name = event_name.replace(f'{event_id_prefix}{onoff_value}', condition_name)
                    
                    # Find all events with this event code and update them
                    event_indices = np.where(epochs.events[:, 2] == event_code)[0]
                    for idx in event_indices:
                        new_event = epochs.events[idx].copy()
                        new_event[2] = len(new_event_id) + 1  # Assign new event code
                        new_events.append(new_event)
                    
                    new_event_id[new_event_name] = len(new_event_id) + 1
        
        if new_events:
            new_events = np.array(new_events)
            new_epochs.events = new_events
            new_epochs.event_id = new_event_id
            
            # Keep only the events that belong to this condition
            new_epochs = new_epochs[list(new_event_id.keys())]
            
            classified_epochs[condition_name] = new_epochs
    
    return classified_epochs


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
    Uses mne.combine_evoked to properly average multiple evokeds for the same condition.

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

    # Combine multiple evokeds for the same condition using mne.combine_evoked
    for subject in participant_evokeds:
        for condition in conditions_of_interest:
            evoked_list = participant_evokeds[subject][condition]
            if len(evoked_list) > 1:
                # Use mne.combine_evoked to properly average multiple evokeds
                try:
                    combined_evoked = mne.combine_evoked(evoked_list, weights='equal')
                    participant_evokeds[subject][condition] = [combined_evoked]
                except Exception as e:
                    print(f"Failed to combine evokeds for {subject}, {condition}: {e}")
                    # Keep the original list if combining fails
            elif len(evoked_list) == 1:
                # Keep as list for consistency
                participant_evokeds[subject][condition] = evoked_list
            else:
                # Empty list - no data for this condition
                participant_evokeds[subject][condition] = []

    return participant_evokeds


def compute_erps(participant_evokeds, subjects, conditions_of_interest, roi, time_bins, aggregate='condition'):
    """
    Compute ERP data for LMM analysis using specified time bins.
    
    Parameters:
    -----------
    participant_evokeds : dict
        Dictionary containing evoked responses for each participant.
    subjects : list
        List of subject IDs.
    conditions_of_interest : list
        List of condition names to include in the analysis.
    roi : list
        List of electrode names (region of interest).
    time_bins : list
        List of time bin tuples (start_time, end_time) in seconds.
    aggregate : str, optional
        How to aggregate the data, either 'condition' or 'subject'.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing ERP data for LMM analysis.
    """
    import pandas as pd
    import numpy as np
    
    all_data = []
    
    for subject in subjects:
        if subject not in participant_evokeds:
            continue
            
        for condition in conditions_of_interest:
            if condition not in participant_evokeds[subject]:
                continue
                
            evoked = participant_evokeds[subject][condition]
            
            # Handle case where evoked might be a list of evoked objects
            if isinstance(evoked, list):
                if len(evoked) > 0:
                    evoked = evoked[0]  # Use the first evoked object in the list
                else:
                    continue  # Skip if the list is empty
            
            # Get channel indices for ROI - FIXED: use evoked.ch_names to get proper indices
            ch_indices = []
            roi_channels_found = []
            for ch in roi:
                if ch in evoked.ch_names:
                    ch_indices.append(evoked.ch_names.index(ch))
                    roi_channels_found.append(ch)
            
            if not ch_indices:
                continue
                
            # Get times
            times = evoked.times
            
            # Process each time bin
            for time_bin in time_bins:
                start_time, end_time = time_bin
                
                # Find indices corresponding to the time bin
                time_mask = (times >= start_time) & (times <= end_time)
                
                if not any(time_mask):
                    continue
                    
                # Extract data for this time bin and ROI
                data = evoked.data[ch_indices][:, time_mask]
                
                # Calculate mean amplitude across time for each channel
                mean_amplitudes = np.mean(data, axis=1)
                
                # For each channel in ROI - FIXED: use roi_channels_found instead of roi indices
                for ch_idx, ch_name in enumerate(roi_channels_found):
                    row_data = {
                        'Subject': subject,
                        'Condition': condition,
                        'Channel': ch_name,
                        'TimeWindow': f"{start_time:.1f}-{end_time:.1f}",
                        'StartTime': start_time,
                        'EndTime': end_time,
                        'MeanAmplitude': mean_amplitudes[ch_idx]
                    }
                    all_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Aggregate if needed
    if aggregate == 'condition':
        # Average across channels and subjects for each condition and time window
        df = df.groupby(['Condition', 'TimeWindow', 'StartTime', 'EndTime']).agg({
            'MeanAmplitude': 'mean'
        }).reset_index()
    elif aggregate == 'subject':
        # Average across channels for each subject, condition, and time window
        df = df.groupby(['Subject', 'Condition', 'TimeWindow', 'StartTime', 'EndTime']).agg({
            'MeanAmplitude': 'mean'
        }).reset_index()
    
    return df


def fit_lmm_for_time_bins(erp_data):
    """
    Fit linear mixed models for each time bin in the ERP data.
    
    Parameters:
    -----------
    erp_data : pandas.DataFrame
        DataFrame containing ERP data with time bins.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing LMM results for each time bin.
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import fdrcorrection
    
    # Initialize results list
    results_list = []
    
    # Get unique time windows
    time_windows = erp_data['TimeWindow'].unique()
    
    # For each time window, fit a model
    for time_window in time_windows:
        # Filter data for this time window
        window_data = erp_data[erp_data['TimeWindow'] == time_window].copy()
        
        # Check if we have both conditions
        conditions = window_data['Condition'].unique()
        if len(conditions) < 2:
            continue
            
        # Create a dummy variable for condition (0 for first condition, 1 for second)
        window_data['ConditionDummy'] = (window_data['Condition'] == conditions[1]).astype(int)
        
        # Fit the model
        try:
            # IMPROVED: Try mixed-effects model first, fall back to simple OLS
            try:
                # Mixed-effects model with random intercept for subject
                from statsmodels.regression.mixed_linear_model import MixedLM
                model = MixedLM.from_formula('MeanAmplitude ~ ConditionDummy', 
                                           data=window_data, groups=window_data['Subject'])
                result = model.fit(method='lbfgs')
                model_type = "Mixed-Effects"
            except:
                # Fall back to simple OLS if mixed-effects fails
                model = smf.ols('MeanAmplitude ~ ConditionDummy', data=window_data)
                result = model.fit()
                model_type = "OLS"
            
            # Extract relevant information
            start_time = window_data['StartTime'].iloc[0]
            end_time = window_data['EndTime'].iloc[0]
            
            # Get coefficient for condition effect
            condition_coef = result.params.get('ConditionDummy', np.nan)
            condition_pvalue = result.pvalues.get('ConditionDummy', np.nan)
            
            # Store results
            results_list.append({
                'TimeWindow': time_window,
                'StartTime': start_time,
                'EndTime': end_time,
                'Condition_Coef': condition_coef,
                'Condition_Coef_pvalue': condition_pvalue,
                'ModelType': model_type
            })
        except Exception as e:
            print(f"Error fitting model for time window {time_window}: {e}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction if we have multiple time windows
    if len(results_df) > 1:
        _, corrected_pvals = fdrcorrection(results_df['Condition_Coef_pvalue'].values)
        results_df['Condition_Coef_pvalue_FDR'] = corrected_pvals
    else:
        results_df['Condition_Coef_pvalue_FDR'] = results_df['Condition_Coef_pvalue']
    
    return results_df


def fit_true_mixed_effects_model(erp_data):
    """
    Fit a proper mixed-effects model with random effects for subjects.
    This is an alternative to the above function with better statistical modeling.
    Includes improved convergence handling and fallback options.
    
    Parameters:
    -----------
    erp_data : pandas.DataFrame
        DataFrame containing ERP data with time bins.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing mixed-effects model results for each time bin.
    """
    import pandas as pd
    import numpy as np
    import warnings
    
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("statsmodels MixedLM not available, falling back to standard LMM")
        return fit_lmm_for_time_bins(erp_data)
    
    from statsmodels.stats.multitest import fdrcorrection
    import statsmodels.formula.api as smf
    
    # Initialize results list
    results_list = []
    
    # Get unique time windows
    time_windows = erp_data['TimeWindow'].unique()
    
    # Suppress convergence warnings temporarily for cleaner output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        
        # For each time window, fit a mixed-effects model
        for time_window in time_windows:
            # Filter data for this time window
            window_data = erp_data[erp_data['TimeWindow'] == time_window].copy()
            
            # Check if we have both conditions and multiple subjects
            conditions = window_data['Condition'].unique()
            subjects = window_data['Subject'].unique()
            
            if len(conditions) < 2 or len(subjects) < 3:
                continue
                
            # Create a dummy variable for condition
            window_data['ConditionDummy'] = (window_data['Condition'] == conditions[1]).astype(int)
            
            # Initialize variables
            condition_coef = np.nan
            condition_pvalue = np.nan
            condition_tvalue = np.nan
            model_type = "Failed"
            
            try:
                # First try: Standard mixed-effects model
                model = MixedLM.from_formula('MeanAmplitude ~ ConditionDummy', 
                                           data=window_data, 
                                           groups=window_data['Subject'])
                result = model.fit(method='lbfgs', maxiter=100, reml=False)
                
                # Extract results if successful
                condition_coef = result.params.get('ConditionDummy', np.nan)
                condition_pvalue = result.pvalues.get('ConditionDummy', np.nan)
                condition_tvalue = result.tvalues.get('ConditionDummy', np.nan)
                model_type = "Mixed-Effects"
                
            except Exception:
                try:
                    # Second try: REML method
                    model = MixedLM.from_formula('MeanAmplitude ~ ConditionDummy', 
                                               data=window_data, 
                                               groups=window_data['Subject'])
                    result = model.fit(method='powell', maxiter=50, reml=True)
                    
                    condition_coef = result.params.get('ConditionDummy', np.nan)
                    condition_pvalue = result.pvalues.get('ConditionDummy', np.nan)
                    condition_tvalue = result.tvalues.get('ConditionDummy', np.nan)
                    model_type = "Mixed-Effects-REML"
                    
                except Exception:
                    try:
                        # Third try: Simple OLS as fallback
                        model = smf.ols('MeanAmplitude ~ ConditionDummy', data=window_data)
                        result = model.fit()
                        
                        condition_coef = result.params.get('ConditionDummy', np.nan)
                        condition_pvalue = result.pvalues.get('ConditionDummy', np.nan)
                        condition_tvalue = result.tvalues.get('ConditionDummy', np.nan)
                        model_type = "OLS-Fallback"
                        
                    except Exception as e:
                        print(f"All model fitting failed for time window {time_window}: {e}")
                        continue
            
            # Extract relevant information
            start_time = window_data['StartTime'].iloc[0]
            end_time = window_data['EndTime'].iloc[0]
            
            # Store results
            results_list.append({
                'TimeWindow': time_window,
                'StartTime': start_time,
                'EndTime': end_time,
                'Condition_Coef': condition_coef,
                'Condition_Coef_pvalue': condition_pvalue,
                'Condition_tvalue': condition_tvalue,
                'N_subjects': len(subjects),
                'N_observations': len(window_data),
                'ModelType': model_type
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Apply FDR correction if we have multiple time windows
    if len(results_df) > 1 and not results_df['Condition_Coef_pvalue'].isna().all():
        valid_pvals = results_df['Condition_Coef_pvalue'].dropna().values
        if len(valid_pvals) > 0:
            _, corrected_pvals = fdrcorrection(valid_pvals)
            
            # Map corrected p-values back to the DataFrame
            corrected_dict = dict(zip(results_df.dropna(subset=['Condition_Coef_pvalue']).index, corrected_pvals))
            results_df['Condition_Coef_pvalue_FDR'] = results_df.index.map(corrected_dict).fillna(results_df['Condition_Coef_pvalue'])
        else:
            results_df['Condition_Coef_pvalue_FDR'] = results_df['Condition_Coef_pvalue']
    else:
        results_df['Condition_Coef_pvalue_FDR'] = results_df['Condition_Coef_pvalue']
    
    return results_df


def plot_erp(evokeds, conditions_of_interest, roi, return_fig=False, significant_windows=None):       
    # Prepare the figure
    fig = go.Figure()

    # Line and style dictionaries
    color_dict = {'ontask': '#281e78', 'offtask': '#fa4617'}
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
            ci95_lower = mean_values - (1.96 * std_error)/2
            ci95_upper = mean_values + (1.96 * std_error)/2
            
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


    # Add significant time windows as grey shaded areas FIRST (behind everything)
    if significant_windows is not None and len(significant_windows) > 0:
        for _, window in significant_windows.iterrows():
            fig.add_shape(
                type="rect",
                x0=window['StartTime'], y0=0,  # Cover full plot height
                x1=window['EndTime'], y1=1,    # using paper coordinates
                fillcolor="lightgrey",
                opacity=0.4,
                layer="below",
                line_width=0,
                xref="x",
                yref="paper"  # Use paper coordinates for full height coverage
            )

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

def split_event_names_to_columns(csv_file_path, split_method="highlow"):
    import pandas as pd
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Define the columns to split
    columns_to_split = ["onoff", "confidence"]
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        event_name = row['event_name']
        
        # Split the event_name into components
        components = event_name.split('/')
        
        # Extract values for each specified column
        for component in components:
            for col in columns_to_split:
                if col in component:
                    # Extract the numeric value
                    value = int(re.search(rf"{col}(\d+)", component).group(1))
                    df.at[index, col] = value
                    
                    # Determine binary split
                    if split_method == "highlow":
                        df.at[index, f"{col}_binary"] = 1 if value >= 50 else 0
                    elif split_method == "midpoint":
                        midpoint = 50  # Assuming midpoint is 50 for simplicity
                        df.at[index, f"{col}_binary"] = 1 if value >= midpoint else 0
                    # Add more methods if needed
    
    # Save the modified DataFrame back to CSV
    df.to_csv(csv_file_path, index=False)
    
    return df