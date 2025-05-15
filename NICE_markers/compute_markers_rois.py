import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NICE_markers.helpers_eeg import all_markers
from utils.bids_compliance import read_epochs


def process_subject_task(derivatives_folder, subject, task, output_dir, roi_mode='whole', per_epoch=True):
    """
    Process a single subject and task with different ROI configurations.
    """
    try:
        # Load epochs using the BIDS-compliant read_epochs function
        print(f"Loading data for subject {subject}, task {task}")
        epochs, events = read_epochs(
            root_path=derivatives_folder,
            subject=subject,
            task=task,
            data="eeg",
            desc="autoPreproc"
        )
        
        # Extract event IDs for each epoch
        event_ids = epochs.events[:, 2]
        event_id_names = {v: k for k, v in epochs.event_id.items()}
        event_names = [event_id_names.get(e, f"event-{e}") for e in event_ids]
        
        print(f"Available channels: {epochs.ch_names}")
        print(f"Computing markers for subject {subject} (task {task}) "
              f"with ROI mode: {roi_mode}, per_epoch: {per_epoch}")
        
        # Compute markers with specified ROI mode
        # For any mode ('whole', 'all', or ROI dict), ensure we properly handle 
        # event IDs and list values
        markers = all_markers(epochs, tmin=0, tmax=2, roi=roi_mode, per_epoch=per_epoch)
        
        # Initialize rows list for any mode
        rows = []
        
        # Handle the data appropriately based on ROI mode and per_epoch flag
        if roi_mode == 'whole':
            # For whole brain (one value per marker across all channels)
            if per_epoch:
                # Per-epoch mode: Create rows with epoch-specific data
                for epoch_idx in range(len(epochs)):
                    for marker_name, value in markers.items():
                        # Skip non-numeric/invalid values
                        if value is None or value is np.nan:
                            continue
                            
                        try:
                            # Handle different data types
                            if isinstance(value, np.ndarray):
                                if value.ndim == 1 and len(value) == len(epochs):
                                    # One value per epoch
                                    marker_value = float(value[epoch_idx])
                                elif value.ndim > 1 and value.shape[0] == len(epochs):
                                    # For multi-dimensional data per epoch
                                    marker_value = float(np.mean(value[epoch_idx]))
                                else:
                                    # For arrays not aligned with epochs, use the same value
                                    marker_value = float(np.mean(value))
                            elif isinstance(value, list) and len(value) == len(epochs):
                                # List with one value per epoch
                                marker_value = float(value[epoch_idx])
                            else:
                                # Scalar or other value
                                marker_value = float(value)
                                
                            # Create a row for this marker and epoch
                            rows.append({
                                'marker': marker_name,
                                'value': marker_value,
                                'subject_id': subject,
                                'task': task,
                                'epoch': epoch_idx,
                                'event_id': event_ids[epoch_idx],
                                'event_name': event_names[epoch_idx]
                            })
                        except (ValueError, TypeError):
                            # Skip values that can't be converted to float
                            pass
            else:
                # Averaged mode: One row per marker
                for marker_name, value in markers.items():
                    if value is None or value is np.nan:
                        continue
                    
                    try:
                        # Convert any value to a single float
                        if isinstance(value, (np.ndarray, list)):
                            marker_value = float(np.mean(value))
                        else:
                            marker_value = float(value)
                            
                        rows.append({
                            'marker': marker_name,
                            'value': marker_value,
                            'subject_id': subject,
                            'task': task
                        })
                    except (ValueError, TypeError):
                        pass
        elif roi_mode == 'all':
            # Per-electrode mode: Each marker has one value per channel
            if per_epoch:
                # One row per epoch, channel, and marker
                for marker_name, values in markers.items():
                    if not isinstance(values, (np.ndarray, list)) or len(values) == 0:
                        continue
                    
                    values_array = np.asarray(values)
                    
                    if values_array.ndim >= 2 and values_array.shape[0] == len(epochs):
                        # Values per epoch and channel: shape (n_epochs, n_channels)
                        for epoch_idx in range(len(epochs)):
                            for ch_idx, channel in enumerate(epochs.ch_names):
                                if ch_idx < values_array.shape[1]:
                                    try:
                                        value = float(values_array[epoch_idx, ch_idx])
                                        rows.append({
                                            'marker': marker_name,
                                            'channel': channel,
                                            'value': value,
                                            'subject_id': subject,
                                            'task': task,
                                            'epoch': epoch_idx,
                                            'event_id': event_ids[epoch_idx],
                                            'event_name': event_names[epoch_idx]
                                        })
                                    except (ValueError, TypeError, IndexError):
                                        pass
                    elif values_array.ndim == 1 and len(values_array) == len(epochs.ch_names):
                        # One value per channel, repeat for each epoch
                        for epoch_idx in range(len(epochs)):
                            for ch_idx, channel in enumerate(epochs.ch_names):
                                try:
                                    value = float(values_array[ch_idx])
                                    rows.append({
                                        'marker': marker_name,
                                        'channel': channel,
                                        'value': value,
                                        'subject_id': subject,
                                        'task': task,
                                        'epoch': epoch_idx,
                                        'event_id': event_ids[epoch_idx],
                                        'event_name': event_names[epoch_idx]
                                    })
                                except (ValueError, TypeError, IndexError):
                                    pass
            else:
                # Average mode: One row per channel and marker
                for marker_name, values in markers.items():
                    if not isinstance(values, (np.ndarray, list)) or len(values) == 0:
                        continue
                    
                    values_array = np.asarray(values)
                    
                    # For averaged data, we expect shape (n_channels,)
                    if values_array.ndim > 1:
                        # Reduce to 1D array with one value per channel
                        if values_array.shape[1] == len(epochs.ch_names):
                            values_array = values_array.mean(axis=0)
                        else:
                            values_array = values_array.mean(axis=1)
                    
                    # Create a row for each channel
                    for ch_idx, channel in enumerate(epochs.ch_names):
                        try:
                            if ch_idx < len(values_array):
                                value = float(values_array[ch_idx])
                            else:
                                value = np.nan
                                
                            rows.append({
                                'marker': marker_name,
                                'channel': channel,
                                'value': value,
                                'subject_id': subject,
                                'task': task
                            })
                        except (ValueError, TypeError, IndexError):
                            pass
        elif isinstance(roi_mode, dict):
            # ROI-based mode: For each marker, we have values per ROI
            for marker_name, roi_values in markers.items():
                if not isinstance(roi_values, dict):
                    continue
                
                # Process each ROI
                for roi_name, value in roi_values.items():
                    if value is None or value is np.nan:
                        continue
                        
                    try:
                        if per_epoch:
                            # Check if value contains per-epoch data
                            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(epochs):
                                # One value per epoch
                                for epoch_idx, epoch_val in enumerate(value):
                                    rows.append({
                                        'marker': marker_name,
                                        'roi': roi_name,
                                        'value': float(epoch_val),
                                        'subject_id': subject,
                                        'task': task,
                                        'epoch': epoch_idx,
                                        'event_id': event_ids[epoch_idx],
                                        'event_name': event_names[epoch_idx]
                                    })
                            else:
                                # Same value for all epochs
                                avg_val = float(np.mean(value)) if isinstance(value, (np.ndarray, list)) else float(value)
                                for epoch_idx in range(len(epochs)):
                                    rows.append({
                                        'marker': marker_name,
                                        'roi': roi_name,
                                        'value': avg_val,
                                        'subject_id': subject,
                                        'task': task,
                                        'epoch': epoch_idx,
                                        'event_id': event_ids[epoch_idx],
                                        'event_name': event_names[epoch_idx]
                                    })
                        else:
                            # Average mode: One row per ROI and marker
                            avg_val = float(np.mean(value)) if isinstance(value, (np.ndarray, list)) else float(value)
                            rows.append({
                                'marker': marker_name,
                                'roi': roi_name,
                                'value': avg_val,
                                'subject_id': subject,
                                'task': task
                            })
                    except (ValueError, TypeError):
                        # Skip values that can't be processed
                        pass
        
        # Create DataFrame and save results
        if not rows:
            print(f"Warning: No valid marker data for subject {subject} (task {task})")
            return None
            
        df = pd.DataFrame(rows)
        
        # Determine output file name based on ROI mode
        if roi_mode == 'whole':
            output_file = os.path.join(output_dir, f'sub-{subject}_task-{task}_whole_brain.csv')
        elif roi_mode == 'all':
            output_file = os.path.join(output_dir, f'sub-{subject}_task-{task}_per_electrode.csv')
        else:  # ROI dict
            output_file = os.path.join(output_dir, f'sub-{subject}_task-{task}_per_roi.csv')
            
        df.to_csv(output_file, index=False)
        print(f"Saved results to {output_file}")
        
        return df
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing subject {subject} (task {task}): {error_msg}")
        # Handle specific error messages...
        return None