"""
Reader module for loading aggregated probe marker data for LMM-based spatial cluster permutation testing.

This module handles data loading from the output of aggregate_markers_by_probe.py, which creates
CSV files with long-format data where each row represents one channel-marker combination for one probe.
The reader merges all probe files across subjects and arranges data appropriately for cluster permutation
LMM testing across channels for each marker.
"""

import numpy as np
import pandas as pd
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import warnings


def load_all_probe_data(features_root: str, 
                       subjects: Optional[List[str]] = None,
                       tasks: Optional[List[str]] = None,
                       marker_types: Optional[List[str]] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Load all aggregated probe marker data from CSV files.
    
    Parameters
    ----------
    features_root : str
        Root directory containing features/sub-XX/eeg/junifer/ structure
    subjects : Optional[List[str]]
        List of subject IDs to include (e.g., ["02", "03", "04"]). If None, loads all subjects.
    tasks : Optional[List[str]]
        List of tasks to include (e.g., ["Sart1", "Sart2"]). If None, loads all tasks.
    marker_types : Optional[List[str]]
        List of marker types to include (e.g., ["evoked", "state"]). If None, loads all types.
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all probe data in long format
        Columns: subject, task, probe_number, onoff, marker_type, marker, channel, value, 
                 plus additional metadata columns
    """
    features_path = Path(features_root)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features root directory not found: {features_root}")
    
    # Find all CSV files matching the pattern
    pattern = "**/sub-*_task-*_desc-probe-*_*_aggMarkers.csv"
    csv_files = list(features_path.glob(pattern))
    
    if verbose:
        print(f"Found {len(csv_files)} aggregated marker CSV files")
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No aggregated marker CSV files found in {features_root}")
    
    # Filter files based on parameters
    filtered_files = []
    for file_path in csv_files:
        # Extract subject, task, and marker_type from filename
        # Format: sub-XX_task-YY_desc-probe-NNN_[evoked|state]_aggMarkers.csv
        filename = file_path.name
        
        # Parse subject
        if filename.startswith("sub-"):
            subject = filename.split("_")[0].replace("sub-", "")
            if subjects is not None and subject not in subjects:
                continue
        else:
            continue
            
        # Parse task
        if "_task-" in filename:
            task = filename.split("_task-")[1].split("_")[0]
            if tasks is not None and task not in tasks:
                continue
        else:
            continue
            
        # Parse marker type
        if "_aggMarkers.csv" in filename:
            marker_type = filename.split("_desc-probe-")[1].split("_aggMarkers.csv")[0]
            if "_" in marker_type:
                marker_type = marker_type.split("_")[-1]  # Get last part (evoked or state)
            else:
                marker_type = "mixed"  # No type specified
                
            if marker_types is not None and marker_type not in marker_types:
                continue
        else:
            continue
            
        filtered_files.append(file_path)
    
    if verbose:
        print(f"After filtering: {len(filtered_files)} files to load")
        if subjects:
            print(f"  Subjects: {subjects}")
        if tasks:
            print(f"  Tasks: {tasks}")
        if marker_types:
            print(f"  Marker types: {marker_types}")
    
    # Load all CSV files
    all_data = []
    for file_path in filtered_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            if verbose and len(all_data) % 50 == 0:
                print(f"  Loaded {len(all_data)} files...")
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            continue
    
    if len(all_data) == 0:
        raise ValueError("No valid CSV files could be loaded")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    if verbose:
        print(f"Combined dataset shape: {combined_df.shape}")
        print(f"Subjects: {sorted(combined_df['subject'].unique())}")
        print(f"Tasks: {sorted(combined_df['task'].unique())}")
        print(f"Marker types: {sorted(combined_df['marker_type'].unique())}")
        print(f"Markers: {sorted(combined_df['marker'].unique())}")
        print(f"Channels: {len(combined_df['channel'].unique())} channels")
        print(f"Probes per subject-task: {combined_df.groupby(['subject', 'task'])['probe_number'].nunique().describe()}")
    
    return combined_df


def prepare_data_for_lmm(df: pd.DataFrame, 
                        marker_name: str,
                        formula: str,
                        include_channels: Optional[List[str]] = None,
                        exclude_channels: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Prepare data for LMM analysis for a specific marker.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined dataframe from load_all_probe_data()
    marker_name : str
        Name of the marker to analyze (e.g., "EEG_psd_bands_spectralpower_alpha")
    formula : str
        R-style formula string for the LMM (e.g., "power ~ onoff + (1|subject)")
    include_channels : Optional[List[str]]
        List of channels to include. If None, includes all channels.
    exclude_channels : Optional[List[str]]
        List of channels to exclude. Applied after include_channels.
        
    Returns
    -------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with columns for LMM formula
        - 'subject': subject identifier for random effects
        - 'power': marker values (dependent variable)
        - Additional columns as specified in formula
    channels : List[str]
        Sorted list of channel names corresponding to power_data columns
        
    Notes
    -----
    This function:
    1. Filters data for the specific marker
    2. Pivots from long format to wide format (channels as columns)
    3. Creates behavioral dataframe with predictor variables
    4. Validates formula variables exist in the data
    """
    # Filter for specific marker (ensure state and evoked are separate)
    marker_df = df[df['marker'] == marker_name].copy()
    
    if len(marker_df) == 0:
        available_markers = sorted(df['marker'].unique())
        raise ValueError(f"Marker '{marker_name}' not found in data. Available markers: {available_markers}")
    
    # Ensure we have only one marker type for this marker
    marker_types_in_data = marker_df['marker_type'].unique()
    if len(marker_types_in_data) > 1:
        raise ValueError(f"Marker '{marker_name}' appears in multiple types: {marker_types_in_data}. State and evoked markers must be processed separately.")
    
    # Get the marker type for validation
    marker_type = marker_types_in_data[0]
    
    # Filter channels if specified
    if include_channels is not None:
        marker_df = marker_df[marker_df['channel'].isin(include_channels)]
        if len(marker_df) == 0:
            raise ValueError(f"No data found for marker '{marker_name}' with specified channels: {include_channels}")
    
    if exclude_channels is not None:
        marker_df = marker_df[~marker_df['channel'].isin(exclude_channels)]
        if len(marker_df) == 0:
            raise ValueError(f"No data found for marker '{marker_name}' after excluding channels: {exclude_channels}")
    
    # Get unique channels and sort them consistently (deterministic)
    channels = sorted(marker_df['channel'].unique())
    n_channels = len(channels)
    
    if n_channels == 0:
        raise ValueError(f"No channels found for marker '{marker_name}' after filtering")
    
    # Get unique observations (subject-task-probe combinations)
    # Create a composite key for each observation (deterministic)
    marker_df['observation_id'] = (marker_df['subject'].astype(str) + '_' + 
                                  marker_df['task'].astype(str) + '_' + 
                                  marker_df['probe_number'].astype(str))
    
    # Sort observations deterministically
    observations = sorted(marker_df['observation_id'].unique())
    n_observations = len(observations)
    
    if n_observations == 0:
        raise ValueError(f"No observations found for marker '{marker_name}'")
    
    # Initialize power data array
    power_data = np.full((n_observations, n_channels), np.nan)
    
    # Create behavioral dataframe - get first row for each observation
    # This ensures we have the behavioral variables for each observation
    df_behavioral = marker_df.groupby('observation_id').first().reset_index()
    
    # Ensure we have the right number of behavioral rows
    if len(df_behavioral) != n_observations:
        raise ValueError(f"Mismatch between observations ({n_observations}) and behavioral data ({len(df_behavioral)})")
    
    # Sort behavioral dataframe to match observation order
    df_behavioral['obs_order'] = df_behavioral['observation_id'].map({obs: i for i, obs in enumerate(observations)})
    df_behavioral = df_behavioral.sort_values('obs_order').reset_index(drop=True)
    df_behavioral = df_behavioral.drop('obs_order', axis=1)
    
    # Fill power data array - pivot from long to wide format
    for i, obs_id in enumerate(observations):
        obs_data = marker_df[marker_df['observation_id'] == obs_id]
        
        # Fill power values for each channel
        for j, channel in enumerate(channels):
            channel_data = obs_data[obs_data['channel'] == channel]
            if len(channel_data) > 0:
                power_data[i, j] = float(channel_data['value'].iloc[0])
    
    # Remove columns that aren't needed for behavioral analysis
    cols_to_drop = ['observation_id', 'marker', 'channel', 'value']
    df_behavioral = df_behavioral.drop(columns=[col for col in cols_to_drop if col in df_behavioral.columns])
    
    # Ensure we have required columns
    required_cols = ['subject', 'task', 'probe_number', 'onoff']
    missing_cols = [col for col in required_cols if col not in df_behavioral.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in behavioral data: {missing_cols}")
    
    # Validate formula variables exist
    validate_formula_variables(df_behavioral, formula)
    
    # Check data quality
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 2:
        raise ValueError(f"Need at least 2 subjects for mixed effects model, found {n_subjects}")
    
    # Check for sufficient data
    valid_data = np.sum(~np.isnan(power_data))
    total_data = power_data.size
    if valid_data / total_data < 0.1:  # At least 10% valid data
        raise ValueError(f"Insufficient valid data: {valid_data}/{total_data} ({valid_data/total_data:.1%})")
    
    return power_data, df_behavioral, channels


def load_data(data_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Legacy function for backward compatibility.
    
    This function now expects data_path to be a features root directory,
    and loads all available probe data.
    
    Parameters
    ----------
    data_path : str
        Path to the features root directory
        
    Returns
    -------
    power_data : np.ndarray
        Placeholder - use prepare_data_for_lmm() instead
    df_behavioral : pd.DataFrame
        Combined dataframe with all probe data
    """
    warnings.warn("load_data() is deprecated. Use load_all_probe_data() and prepare_data_for_lmm() instead.")
    
    # Load all probe data
    df_behavioral = load_all_probe_data(data_path, verbose=True)
    
    # Return placeholder power_data and the behavioral dataframe
    power_data = np.array([])  # Placeholder
    
    return power_data, df_behavioral


def validate_formula_variables(df_behavioral: pd.DataFrame, formula: str) -> None:
    """
    Validate that all variables in the formula exist in the behavioral DataFrame.
    
    Parameters
    ----------
    df_behavioral : pd.DataFrame
        Behavioral data
    formula : str
        R-style formula string
        
    Raises
    ------
    ValueError
        If any variable in the formula is not found in the DataFrame
    """
    import re
    
    # Extract variable names from formula (simple regex)
    # Match words that are not 'power' or operators
    variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
    variables = [v for v in variables if v not in ['power', 'subject']]
    
    missing = [v for v in variables if v not in df_behavioral.columns]
    if missing:
        raise ValueError(f"Variables not found in behavioral data: {missing}")


def prepare_channel_data(power_data: np.ndarray, 
                         df_behavioral: pd.DataFrame,
                         channel_idx: int,
                         channels: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare data for a single channel for LMM analysis.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data
    channel_idx : int
        Index of the channel to extract
    channels : Optional[List[str]]
        List of channel names corresponding to channel indices
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 'power' column and behavioral variables
    """
    df = df_behavioral.copy()
    df['power'] = power_data[:, channel_idx]
    
    # Add channel name if available
    if channels is not None and channel_idx < len(channels):
        df['channel'] = channels[channel_idx]
    
    return df


def get_available_markers(features_root: str, 
                         marker_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Get list of available markers from the aggregated data.
    
    This function ensures that state and evoked markers are treated as completely separate,
    even if they have the same marker name. Each marker type maintains its own list.
    
    Parameters
    ----------
    features_root : str
        Root directory containing features
    marker_types : Optional[List[str]]
        List of marker types to filter by (e.g., ["evoked", "state"])
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping marker types to lists of marker names.
        State and evoked markers are kept separate even if names are similar.
    """
    # Load a sample of data to get marker information
    sample_df = load_all_probe_data(features_root, verbose=False)
    
    # Group markers by type (state and evoked are completely separate)
    marker_info = {}
    for marker_type in sample_df['marker_type'].unique():
        if marker_types is None or marker_type in marker_types:
            # Get markers for this specific type only
            type_data = sample_df[sample_df['marker_type'] == marker_type]
            type_markers = sorted(type_data['marker'].unique())
            marker_info[marker_type] = type_markers
    
    return marker_info


def get_channel_names(features_root: str) -> List[str]:
    """
    Get list of available channel names from the aggregated data.
    
    Parameters
    ----------
    features_root : str
        Root directory containing features
        
    Returns
    -------
    List[str]
        Sorted list of channel names
    """
    # Load a sample of data to get channel information
    sample_df = load_all_probe_data(features_root, verbose=False)
    return sorted(sample_df['channel'].unique())
