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
                       qa_exclusions: Optional[Dict[str, set]] = None,
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
    qa_exclusions : Optional[Dict[str, set]]
        Dictionary mapping marker types to sets of (subject, task) tuples to exclude.
        Example: {'evoked': {('02', 'Sart1'), ('03', 'Sart2')}, 'state': {('04', 'Sart3')}}
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
    qa_excluded_files = []
    
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
        
        # Check QA exclusions
        if qa_exclusions is not None and marker_type in qa_exclusions:
            if (subject, task) in qa_exclusions[marker_type]:
                qa_excluded_files.append((subject, task, marker_type, file_path))
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
        if qa_exclusions is not None and len(qa_excluded_files) > 0:
            print(f"  QA excluded: {len(qa_excluded_files)} files")
            # Group by marker type for summary
            from collections import defaultdict
            exclusions_by_type = defaultdict(int)
            for _, _, mtype, _ in qa_excluded_files:
                exclusions_by_type[mtype] += 1
            for mtype, count in sorted(exclusions_by_type.items()):
                print(f"    {mtype}: {count} files excluded")
    
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


def filter_subjects_by_variability(df: pd.DataFrame,
                                   predictor_column: str,
                                   min_variability: Optional[str] = None,
                                   subject_column: str = 'subject',
                                   verbose: bool = True) -> pd.DataFrame:
    """
    Filter out subjects with insufficient within-subject variability in a predictor.
    
    This is useful to remove subjects who have constant or near-constant predictor values,
    which can cause issues in mixed-effects models and don't contribute meaningful
    within-subject variance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with subject-level observations
    predictor_column : str
        Name of the predictor column to check variability (e.g., 'onoff', 'valence')
    min_variability : Optional[str]
        Variability threshold specification:
        - None or False: No filtering
        - "auto": Remove subjects with zero variance (std == 0)
        - float/int: Minimum standard deviation required
        - "quantile_X": Remove bottom X% of subjects by variance (e.g., "quantile_10")
    subject_column : str
        Name of the subject identifier column (default: 'subject')
    verbose : bool
        Whether to print filtering information
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe with subjects removed
        
    Examples
    --------
    # Remove subjects with zero variance
    df_filtered = filter_subjects_by_variability(df, 'onoff', min_variability='auto')
    
    # Remove subjects with std < 5
    df_filtered = filter_subjects_by_variability(df, 'onoff', min_variability=5)
    
    # Remove bottom 10% of subjects by variance
    df_filtered = filter_subjects_by_variability(df, 'onoff', min_variability='quantile_10')
    """
    # No filtering if min_variability is None or False
    if min_variability is None or min_variability is False:
        if verbose:
            print("  No predictor variability filtering applied")
        return df
    
    # Check that predictor column exists
    if predictor_column not in df.columns:
        raise ValueError(f"Predictor column '{predictor_column}' not found in dataframe. "
                        f"Available columns: {list(df.columns)}")
    
    # Compute within-subject standard deviation for the predictor
    subject_stats = df.groupby(subject_column)[predictor_column].agg(['std', 'count', 'mean'])
    subject_stats = subject_stats.rename(columns={'std': 'predictor_std', 
                                                   'count': 'n_obs',
                                                   'mean': 'predictor_mean'})
    
    # Handle NaN std (occurs when subject has only 1 observation)
    subject_stats['predictor_std'] = subject_stats['predictor_std'].fillna(0)
    
    n_subjects_before = len(subject_stats)
    
    # Determine threshold based on min_variability specification
    if isinstance(min_variability, str):
        if min_variability.lower() == 'auto':
            # Remove subjects with zero variance
            threshold = 0
            subjects_to_keep = subject_stats[subject_stats['predictor_std'] > threshold].index
            threshold_description = "zero variance (auto)"
        elif min_variability.lower().startswith('quantile_'):
            # Extract percentile (e.g., "quantile_10" -> 10)
            try:
                percentile = float(min_variability.split('_')[1])
                if not 0 <= percentile <= 100:
                    raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
                
                # Compute threshold as the Xth percentile of std
                threshold = np.percentile(subject_stats['predictor_std'], percentile)
                subjects_to_keep = subject_stats[subject_stats['predictor_std'] > threshold].index
                threshold_description = f"{percentile}th percentile (threshold={threshold:.3f})"
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid quantile specification: '{min_variability}'. "
                               f"Use format 'quantile_X' where X is 0-100. Error: {e}")
        else:
            raise ValueError(f"Invalid min_variability string: '{min_variability}'. "
                           f"Use 'auto', 'quantile_X', or a numeric value.")
    else:
        # Numeric threshold
        try:
            threshold = float(min_variability)
            subjects_to_keep = subject_stats[subject_stats['predictor_std'] > threshold].index
            threshold_description = f"std > {threshold}"
        except (TypeError, ValueError):
            raise ValueError(f"Invalid min_variability value: {min_variability}. "
                           f"Must be 'auto', 'quantile_X', or a number.")
    
    # Filter dataframe
    df_filtered = df[df[subject_column].isin(subjects_to_keep)].copy()
    
    n_subjects_after = len(subjects_to_keep)
    n_removed = n_subjects_before - n_subjects_after
    
    if verbose:
        print(f"\n  Predictor variability filter ({predictor_column}):")
        print(f"    Threshold: {threshold_description}")
        print(f"    Subjects before: {n_subjects_before}")
        print(f"    Subjects after: {n_subjects_after}")
        print(f"    Subjects removed: {n_removed} ({100*n_removed/n_subjects_before:.1f}%)")
        
        if n_removed > 0:
            removed_subjects = sorted(list(set(subject_stats.index) - set(subjects_to_keep)))
            removed_stats = subject_stats.loc[removed_subjects]
            print(f"    Removed subjects: {removed_subjects}")
            print(f"    Removed subjects' std range: [{removed_stats['predictor_std'].min():.3f}, "
                  f"{removed_stats['predictor_std'].max():.3f}]")
            print(f"    Kept subjects' std range: [{subject_stats.loc[subjects_to_keep, 'predictor_std'].min():.3f}, "
                  f"{subject_stats.loc[subjects_to_keep, 'predictor_std'].max():.3f}]")
    
    # Warn if too many subjects removed
    if n_removed / n_subjects_before > 0.3:  # More than 30%
        warnings.warn(f"Variability filter removed {n_removed}/{n_subjects_before} subjects ({100*n_removed/n_subjects_before:.1f}%). "
                     f"Consider using a less strict threshold.")
    
    return df_filtered


def prepare_data_for_lmm(df: pd.DataFrame, 
                        marker_name: str,
                        formula: str,
                        include_channels: Optional[List[str]] = None,
                        exclude_channels: Optional[List[str]] = None,
                        pca_data: Optional[pd.DataFrame] = None,
                        onoff_max_value: Optional[float] = None,
                        min_predictor_variability: Optional[str] = None,
                        predictor_of_interest: Optional[str] = None) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
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
    pca_data : Optional[pd.DataFrame]
        DataFrame with PCA results containing columns: subject, task, probe_number,
        PC1, PC2, PC3, time_on_task. If provided, these variables will be merged
        into the behavioral dataframe and available for use in the formula.
    onoff_max_value : Optional[float]
        Maximum value for onoff variable. If provided, only observations where
        onoff <= onoff_max_value will be included in the analysis.
    min_predictor_variability : Optional[str]
        Minimum within-subject variability required for the predictor.
        See filter_subjects_by_variability() for options.
    predictor_of_interest : Optional[str]
        Name of the predictor to use for variability filtering.
        Required if min_predictor_variability is specified.
        
    Returns
    -------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with columns for LMM formula
        - 'subject': subject identifier for random effects
        - 'power': marker values (dependent variable)
        - Additional columns as specified in formula
        - If pca_data provided: PC1, PC2, PC3, time_on_task
    channels : List[str]
        Sorted list of channel names corresponding to power_data columns
        
    Notes
    -----
    This function:
    1. Filters data for the specific marker
    2. Pivots from long format to wide format (channels as columns)
    3. Creates behavioral dataframe with predictor variables
    4. Merges PCA data if provided
    5. Validates formula variables exist in the data
    """
    # Filter for specific marker (ensure state and evoked are separate)
    marker_df = df[df['marker'] == marker_name].copy()
    
    if len(marker_df) == 0:
        available_markers = sorted(df['marker'].unique())
        raise ValueError(f"Marker '{marker_name}' not found in data. Available markers: {available_markers}")
    
    # Apply onoff filter if specified
    if onoff_max_value is not None:
        if 'onoff' not in marker_df.columns:
            raise ValueError("onoff_max_value specified but 'onoff' column not found in data")
        
        n_before = len(marker_df)
        marker_df = marker_df[marker_df['onoff'] <= onoff_max_value].copy()
        n_after = len(marker_df)
        
        if n_after == 0:
            raise ValueError(
                f"No observations remain after filtering onoff <= {onoff_max_value}. "
                f"Original data had {n_before} observations."
            )
        
        print(f"  Filtered by onoff <= {onoff_max_value}: {n_before} -> {n_after} observations "
              f"({100 * n_after / n_before:.1f}% retained)")
    
    # Apply predictor variability filter if specified
    if min_predictor_variability is not None and min_predictor_variability is not False:
        if predictor_of_interest is None:
            raise ValueError("predictor_of_interest must be specified when using min_predictor_variability")
        
        # Filter subjects by variability in the predictor
        marker_df = filter_subjects_by_variability(
            df=marker_df,
            predictor_column=predictor_of_interest,
            min_variability=min_predictor_variability,
            subject_column='subject',
            verbose=True
        )
        
        if len(marker_df) == 0:
            raise ValueError(
                f"No observations remain after filtering by predictor variability. "
                f"All subjects had insufficient variability in '{predictor_of_interest}'."
            )
    
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
    
    # Fill power data array - pivot from long to wide format (OPTIMIZED)
    # Use pandas pivot for much faster performance
    pivot_df = marker_df.pivot(index='observation_id', columns='channel', values='value')
    
    # Reindex to ensure correct order and fill missing values
    pivot_df = pivot_df.reindex(index=observations, columns=channels)
    
    # Convert to numpy array
    power_data = pivot_df.values
    
    # Remove columns that aren't needed for behavioral analysis
    cols_to_drop = ['observation_id', 'marker', 'channel', 'value']
    df_behavioral = df_behavioral.drop(columns=[col for col in cols_to_drop if col in df_behavioral.columns])
    
    # Ensure subject column is always string type (required for mixedlm)
    # This prevents TypeError when statsmodels tries to perform bitwise operations
    df_behavioral['subject'] = df_behavioral['subject'].astype(str)
    
    # Compute time_on_task directly from probe_number and task
    # This is independent of PCA and should always be available
    # Formula: probe_number + (15 * (sart_number - 1))
    # Sart1: probes 1-15 -> time_on_task 1-15
    # Sart2: probes 1-15 -> time_on_task 16-30
    # Sart3: probes 1-15 -> time_on_task 31-45
    # Sart4: probes 1-15 -> time_on_task 46-60
    task_to_sart_number = {
        'Sart1': 1,
        'Sart2': 2,
        'Sart3': 3,
        'Sart4': 4
    }
    
    df_behavioral['sart_number'] = df_behavioral['task'].map(task_to_sart_number)
    if df_behavioral['sart_number'].isna().any():
        unknown_tasks = df_behavioral[df_behavioral['sart_number'].isna()]['task'].unique()
        raise ValueError(f"Unknown task names found: {unknown_tasks}. Expected: Sart1, Sart2, Sart3, Sart4")
    
    df_behavioral['time_on_task'] = (
        df_behavioral['probe_number'] + 
        (15 * (df_behavioral['sart_number'] - 1))
    ).astype(int)
    
    # Drop temporary column
    df_behavioral = df_behavioral.drop('sart_number', axis=1)
    
    # Merge PCA data if provided (for PC1, PC2, PC3 only - NOT time_on_task)
    if pca_data is not None:
        
        # Standardize subject format in both dataframes to ensure matching
        # Convert to zero-padded 2-digit string (e.g., "2" -> "02")
        pca_data = pca_data.copy()  # Don't modify original
        df_behavioral['subject'] = df_behavioral['subject'].astype(str).str.zfill(2)
        pca_data['subject'] = pca_data['subject'].astype(str).str.zfill(2)
        
        # Also ensure probe_number is same type (int) in both
        df_behavioral['probe_number'] = df_behavioral['probe_number'].astype(int)
        pca_data['probe_number'] = pca_data['probe_number'].astype(int)
        
        # Select only PCA columns (PC1, PC2, PC3) - NOT time_on_task
        # time_on_task is computed directly above and should not be overwritten
        pca_cols_to_merge = ['subject', 'task', 'probe_number', 'PC1', 'PC2', 'PC3']
        available_pca_cols = [col for col in pca_cols_to_merge if col in pca_data.columns]
        
        if len(available_pca_cols) >= 4:  # Need at least merge keys + one PC
            pca_data_subset = pca_data[available_pca_cols].copy()
            
            # Merge on subject, task, and probe_number using left join
            n_before = len(df_behavioral)
            df_behavioral = df_behavioral.merge(
                pca_data_subset,
                on=['subject', 'task', 'probe_number'],
                how='left',
                suffixes=('', '_pca')
            )
            
            # Check if merge was successful
            if len(df_behavioral) != n_before:
                warnings.warn(
                    f"PCA merge changed number of rows: {n_before} -> {len(df_behavioral)}. "
                    "This may indicate duplicate keys in PCA data."
                )
            
            # Explicitly ensure PCA columns exist
            pca_cols = ['PC1', 'PC2', 'PC3']
            for col in pca_cols:
                if col not in df_behavioral.columns:
                    df_behavioral[col] = np.nan
            
            # Check how many observations have complete PCA data
            n_with_pca = df_behavioral[pca_cols].notna().all(axis=1).sum()
            n_missing = n_before - n_with_pca
            
            if n_missing > 0:
                print(f"  PCA data (PC1-PC3): {n_with_pca}/{n_before} observations have complete data. "
                      f"{n_missing} observations have NaN values.")
        else:
            warnings.warn(f"PCA data missing expected columns. Available: {available_pca_cols}")
            # Ensure PC columns exist as NaN
            for col in ['PC1', 'PC2', 'PC3']:
                if col not in df_behavioral.columns:
                    df_behavioral[col] = np.nan
    
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
    Get list of available markers from the aggregated data by scanning filenames.
    
    This function ensures that state and evoked markers are treated as completely separate,
    even if they have the same marker name. Each marker type maintains its own list.
    
    Uses fast filesystem scan instead of loading data.
    
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
    import re
    features_path = Path(features_root)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features root directory not found: {features_root}")
    
    # Fast scan: find CSV files and extract marker info from filenames
    # Pattern: sub-XX_task-SartX_desc-probe-XXX_TYPE_aggMarkers.csv
    # where TYPE is like "evoked" or "state"
    pattern = "**/sub-*_task-*_desc-probe-*_*_aggMarkers.csv"
    csv_files = list(features_path.glob(pattern))
    
    if not csv_files:
        raise ValueError(f"No aggregated marker CSV files found in {features_root}")
    
    # Extract marker types from filenames first
    # Filename format: sub-02_task-Sart1_desc-probe-001_evoked_aggMarkers.csv
    # The marker type is the part before _aggMarkers.csv
    marker_info = {}
    files_by_type = {}
    
    # Group files by marker type (extracted from filename)
    for f in csv_files:
        # Extract type from filename: ...desc-probe-XXX_TYPE_aggMarkers.csv
        fname = f.name
        match = re.search(r'_desc-probe-\d+_(\w+)_aggMarkers\.csv$', fname)
        if match:
            mtype = match.group(1)
            if mtype not in files_by_type:
                files_by_type[mtype] = f
    
    # Read one sample file per marker type to get marker names
    for mtype, sample_file in files_by_type.items():
        if marker_types is not None and mtype not in marker_types:
            continue
        try:
            sample_df = pd.read_csv(sample_file, usecols=['marker'])
            type_markers = sorted(sample_df['marker'].unique())
            marker_info[mtype] = type_markers
        except Exception as e:
            raise ValueError(f"Failed to read marker info from {sample_file}: {e}")
    
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
