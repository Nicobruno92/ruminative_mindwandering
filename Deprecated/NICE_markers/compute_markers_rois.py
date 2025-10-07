import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NICE_markers.helpers_eeg import all_markers
from utils.bids_compliance import read_epochs

# Define standard ROIs for EEG analysis
ROIS = {
    'prefrontal': ['Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8'],
    'frontal': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz'],
    'central': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz', 'CPz'],
    'temporal': ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    'parietal': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz'],
    'occipital': ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz']
}

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
        
        # Remove EOG channels at the start
        eog_channels = {'VEOG', 'HEOG', 'EOG', 'EOG1', 'EOG2'}
        all_channels = [ch for ch in epochs.ch_names if ch not in eog_channels]
        if len(all_channels) not in (64, 65):
            raise RuntimeError(f"After removing EOGs, expected 64 or 65 EEG channels, got {len(all_channels)}: {all_channels}")
        epochs = epochs.pick_channels(all_channels)
        n_channels = len(all_channels)
        n_epochs = len(epochs)
        
        # Ensure data is downsampled to 500 Hz if needed
        current_sfreq = epochs.info['sfreq']
        if current_sfreq > 500:
            print(f"Downsampling from {current_sfreq} Hz to 500 Hz")
            epochs = epochs.resample(500)
        
        # Extract event IDs for each epoch
        event_ids = epochs.events[:, 2]
        event_id_names = {v: k for k, v in epochs.event_id.items()}
        
        # Process event names to make sure annotations are consistent
        event_names = []
        for e in event_ids:
            raw_name = event_id_names.get(e, f"event-{e}")
            
            # Clean up and standardize event name format
            if 'None' in raw_name and '/' in raw_name:
                # Format: go/correct/None/None/None/None/None/None/1
                # Extract the basic parts and the trial number if present
                parts = raw_name.split('/')
                basic_parts = [p for p in parts[:2] if p != 'None'] 
                
                # Try to extract trial number from the end
                trial_num = None
                for i in range(len(parts)-1, -1, -1):
                    if parts[i].isdigit():
                        trial_num = parts[i]
                        break
                
                # Reconstruct a more consistent name
                if trial_num:
                    event_names.append(f"{'/'.join(basic_parts)}/trial{trial_num}")
                else:
                    event_names.append('/'.join(basic_parts))
            
            # Handle probe annotations - make sure probe and number are together
            elif 'probe' in raw_name:
                parts = raw_name.split('/')
                new_parts = []
                probe_part = None
                
                for part in parts:
                    if 'probe' in part:
                        probe_part = part
                    elif probe_part and part.isdigit() and len(part) <= 2:
                        # This is likely a trial number that belongs with the probe
                        probe_part = f"{probe_part}/{part}"
                    else:
                        new_parts.append(part)
                
                if probe_part:
                    new_parts.append(probe_part)
                
                event_names.append('/'.join(new_parts))
            else:
                event_names.append(raw_name)
        
        print(f"Available channels: {epochs.ch_names}")
        print(f"Computing markers for subject {subject} (task {task}) "
              f"with ROI mode: {roi_mode}, per_epoch: {per_epoch}")
        
        # Compute markers with specified ROI mode
        markers = all_markers(epochs, tmin=0, tmax=2, roi=roi_mode, per_epoch=per_epoch)
        
        # Initialize rows list for any mode
        rows = []
        
        # Get all available channels (full channel list from all_markers)
        all_channels = epochs.ch_names
        n_channels = len(all_channels)
        n_epochs = len(epochs)

        # Handle the data appropriately based on ROI mode and per_epoch flag
        if roi_mode == 'whole':
            # For whole brain (one value per marker across all channels)
            if per_epoch:
                # Per-epoch mode: Create rows with epoch-specific data
                for epoch_idx in range(n_epochs):
                    for marker_name, value in markers.items():
                        if value is None or value is np.nan:
                            continue
                        value_array = np.array(value)
                        if value_array.ndim == 1 and value_array.shape[0] == n_epochs:
                            marker_value = float(value_array[epoch_idx])
                        elif value_array.ndim > 1 and value_array.shape[0] == n_epochs:
                            marker_value = float(np.mean(value_array[epoch_idx]))
                        else:
                            marker_value = float(np.nanmean(value_array))
                        rows.append({
                            'marker': marker_name,
                            'value': marker_value,
                            'subject_id': subject,
                            'task': task,
                            'epoch': epoch_idx,
                            'event_id': event_ids[epoch_idx],
                            'event_name': event_names[epoch_idx]
                        })
            else:
                # Averaged mode: One row per marker
                for marker_name, value in markers.items():
                    if value is None or value is np.nan:
                        continue
                    value_array = np.array(value)
                    marker_value = float(np.nanmean(value_array))
                    rows.append({
                        'marker': marker_name,
                        'value': marker_value,
                        'subject_id': subject,
                        'task': task
                    })
        elif roi_mode == 'all':
            # Per-electrode mode: Each marker has one value per channel and epoch
            print(f"Processing per-electrode data for {len(markers)} markers")
            for marker_name, values in markers.items():
                if values is None:
                    continue
                values_array = np.array(values)
                if values_array.ndim == 2 and values_array.shape[0] == n_channels:
                    # (n_channels, n_epochs)
                    for ch_idx, channel in enumerate(all_channels):
                        for epoch_idx in range(min(n_epochs, values_array.shape[1])):
                            value = float(values_array[ch_idx, epoch_idx])
                            rows.append({
                                'marker': marker_name,
                                'channel': channel,
                                'value': value if not np.isnan(value) else None,
                                'subject_id': subject,
                                'task': task,
                                'epoch': epoch_idx,
                                'event_id': event_ids[epoch_idx],
                                'event_name': event_names[epoch_idx]
                            })
                elif values_array.ndim == 1 and values_array.shape[0] == n_channels:
                    # (n_channels,) - not per-epoch
                    for ch_idx, channel in enumerate(all_channels):
                        rows.append({
                            'marker': marker_name,
                            'channel': channel,
                            'value': float(values_array[ch_idx]),
                            'subject_id': subject,
                            'task': task
                        })
                else:
                    print(f"[ERROR] Marker {marker_name} did not return a per-channel array for per-electrode mode. Shape: {values_array.shape}")
                    for ch_idx, channel in enumerate(all_channels):
                        for epoch_idx in range(n_epochs):
                            rows.append({
                                'marker': marker_name,
                                'channel': channel,
                                'value': np.nan,
                                'subject_id': subject,
                                'task': task,
                                'epoch': epoch_idx,
                                'event_id': event_ids[epoch_idx],
                                'event_name': event_names[epoch_idx]
                            })
        elif isinstance(roi_mode, dict):
            # ROI-based mode: For each marker, we have values per ROI and epoch
            for marker_name, roi_values in markers.items():
                if not isinstance(roi_values, dict):
                    print(f"Warning: Marker {marker_name} in ROI mode did not return a dictionary. Skipping.")
                    continue
                for roi_name, value in roi_values.items():
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        continue
                    value_array = np.array(value)
                    if value_array.ndim == 1 and value_array.shape[0] == n_epochs:
                        # One value per epoch
                        for epoch_idx, epoch_val in enumerate(value_array):
                            rows.append({
                                'marker': marker_name,
                                'roi': roi_name,
                                'value': float(epoch_val) if not np.isnan(epoch_val) else None,
                                'subject_id': subject,
                                'task': task,
                                'epoch': epoch_idx,
                                'event_id': event_ids[epoch_idx],
                                'event_name': event_names[epoch_idx]
                            })
                    elif value_array.ndim == 0 or (value_array.ndim == 1 and value_array.size == 1):
                        # Single scalar value - use for all epochs
                        avg_val = float(value_array.item() if value_array.size == 1 else value_array)
                        for epoch_idx in range(n_epochs):
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
                        print(f"[WARNING] ROI Marker {marker_name} ROI {roi_name} returned array of shape {value_array.shape}, which is not supported. Filling with NaN.")
                        for epoch_idx in range(n_epochs):
                            rows.append({
                                'marker': marker_name,
                                'roi': roi_name,
                                'value': np.nan,
                                'subject_id': subject,
                                'task': task,
                                'epoch': epoch_idx,
                                'event_id': event_ids[epoch_idx],
                                'event_name': event_names[epoch_idx]
                            })
        
        # Create DataFrame and save results
        if not rows:
            print(f"Warning: No valid marker data for subject {subject} (task {task})")
            return None
            
        df = pd.DataFrame(rows)
        print(f"Created DataFrame with {len(df)} rows from {len(markers)} markers")
        
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