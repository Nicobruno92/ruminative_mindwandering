import os
import glob
import sys
import argparse
import numpy as np
import pandas as pd
import re
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"Processing {desc}...")
        return iterable
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy import stats
from scipy.stats import trim_mean

# Get the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the project root directory (parent of script_dir)
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add the project root directory to Python path
sys.path.insert(0, project_root)


def safe_trimmed_mean(values, proportiontocut=0.1):
    """
    Calculate trimmed mean with safety for small sample sizes.
    
    Parameters:
    -----------
    values : array-like
        Values to calculate trimmed mean for
    proportiontocut : float
        Proportion to cut from each end (default: 0.1 = 10%)
    
    Returns:
    --------
    float
        Trimmed mean, falls back to regular mean if too few values
    """
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN values
    
    if len(values) < 4:  # If ≤ 4 values, use normal mean
        return np.mean(values) if len(values) > 0 else np.nan
    
    # For small samples, reduce trimming
    if len(values) < 10:
        proportiontocut = 0.05  # Only 5% from each end
    
    try:
        return trim_mean(values, proportiontocut)
    except:
        return np.mean(values) if len(values) > 0 else np.nan


def remove_outliers_zscore(df, column='value', threshold=3.0):
    """
    Remove outliers using Z-score method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to check for outliers
    threshold : float
        Z-score threshold (default: 3.0)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df.empty or column not in df.columns:
        return df
    
    z_scores = np.abs(stats.zscore(df[column], nan_policy='omit'))
    return df[z_scores < threshold]


def remove_outliers_iqr(df, column='value', multiplier=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to check for outliers
    multiplier : float
        IQR multiplier (default: 1.5)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df.empty or column not in df.columns:
        return df
    
    values = df[column].dropna()
    if len(values) < 4:  # Need at least 4 values for meaningful IQR
        return df
    
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def remove_outliers_modified_zscore(df, column='value', threshold=3.5):
    """
    Remove outliers using Modified Z-score method (more robust).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to check for outliers
    threshold : float
        Modified Z-score threshold (default: 3.5)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df.empty or column not in df.columns:
        return df
    
    values = df[column].dropna()
    if len(values) < 3:
        return df
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    if mad == 0:
        return df  # No variation in data
    
    modified_z_scores = 0.6745 * (values - median) / mad
    mask = np.abs(modified_z_scores) < threshold
    
    return df.loc[values.index[mask]]


def remove_outliers_percentile(df, column='value', lower_percentile=1, upper_percentile=99):
    """
    Remove outliers using percentile method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    column : str
        Column name to check for outliers
    lower_percentile : float
        Lower percentile threshold (default: 1)
    upper_percentile : float
        Upper percentile threshold (default: 99)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df.empty or column not in df.columns:
        return df
    
    values = df[column].dropna()
    if len(values) < 10:  # Need sufficient data for percentiles
        return df
    
    lower_bound = values.quantile(lower_percentile / 100)
    upper_bound = values.quantile(upper_percentile / 100)
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def apply_outlier_removal(df, method='iqr', threshold=None, show_progress=True):
    """
    Apply outlier removal to the DataFrame, grouped by subject, marker, and channel.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    method : str
        Outlier removal method ('zscore', 'iqr', 'modified_zscore', 'percentile', 'none')
    threshold : float or dict
        Threshold parameter(s) for the outlier method
    show_progress : bool
        Whether to show progress bar
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    if df is None or df.empty or method == 'none':
        return df
    
    if 'value' not in df.columns:
        print("Warning: 'value' column not found, skipping outlier removal")
        return df
    
    # Define grouping columns for outlier removal (per subject, marker, channel)
    group_cols = ['subject_id', 'marker']
    
    # Add session_id if available (for multi-session studies)
    if 'session_id' in df.columns:
        group_cols.append('session_id')
    
    # Check if all required columns exist
    missing_cols = [col for col in group_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for outlier removal: {missing_cols}")
        return df
    
    original_size = len(df)
    print(f"Applying outlier removal method: {method}")
    print(f"Original data size: {original_size} rows")
    
    # Set default thresholds based on method
    if threshold is None:
        if method == 'zscore':
            threshold = 3.0
        elif method == 'iqr':
            threshold = 1.5
        elif method == 'modified_zscore':
            threshold = 3.5
        elif method == 'percentile':
            threshold = {'lower': 1, 'upper': 99}
    
    # Apply outlier removal per group
    cleaned_groups = []
    groups = df.groupby(group_cols, observed=True)
    
    if show_progress:
        groups = tqdm(groups, desc="Removing outliers")
    
    outliers_removed = 0
    
    for name, group in groups:
        if len(group) < 5:  # Skip groups with too few observations
            cleaned_groups.append(group)
            continue
        
        original_group_size = len(group)
        
        if method == 'zscore':
            cleaned_group = remove_outliers_zscore(group, threshold=threshold)
        elif method == 'iqr':
            cleaned_group = remove_outliers_iqr(group, multiplier=threshold)
        elif method == 'modified_zscore':
            cleaned_group = remove_outliers_modified_zscore(group, threshold=threshold)
        elif method == 'percentile':
            if isinstance(threshold, dict):
                cleaned_group = remove_outliers_percentile(
                    group, 
                    lower_percentile=threshold.get('lower', 1),
                    upper_percentile=threshold.get('upper', 99)
                )
            else:
                cleaned_group = remove_outliers_percentile(group)
        else:
            print(f"Warning: Unknown outlier removal method '{method}', skipping")
            cleaned_group = group
        
        outliers_removed += (original_group_size - len(cleaned_group))
        cleaned_groups.append(cleaned_group)
    
    if not cleaned_groups:
        print("Warning: No groups remained after outlier removal")
        return df
    
    # Combine cleaned groups
    result_df = pd.concat(cleaned_groups, ignore_index=True)
    
    final_size = len(result_df)
    removed_percentage = (outliers_removed / original_size) * 100
    
    print(f"Outlier removal complete:")
    print(f"  - Outliers removed: {outliers_removed} ({removed_percentage:.2f}%)")
    print(f"  - Final data size: {final_size} rows")
    
    # Force garbage collection
    del cleaned_groups
    gc.collect()
    
    return result_df


def process_event_names_vectorized(df, split_method="highlow", show_progress=True):
    """
    Vectorized processing of event_name column for much better performance.
    
    IMPORTANT NOTES:
    - distance_to_probe: 0 = last trial before probe, positive values = previous trials
    - Values already come in final units (no reconversion in aggregation)
    - distance_to_probe initialized as NaN to avoid unparsed events
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing event_name column
    split_method : str
        Method for splitting values (backward compatibility)
    show_progress : bool
        Whether to display progress bar
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional columns for conditions
    """
    if df is None or df.empty or 'event_name' not in df.columns:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Pre-allocate all columns with proper dtypes for better performance
    result_df["trial_type"] = pd.Categorical(["unknown"] * len(result_df), 
                                           categories=['go', 'nogo', 'unknown'])
    result_df["correctness"] = pd.Categorical(["unknown"] * len(result_df), 
                                            categories=['correct', 'incorrect', 'unknown'])
    
    # KEY CHANGE 1: Initialize distance_to_probe as NaN (not 0)
    result_df["distance_to_probe"] = np.full(len(result_df), np.nan, dtype='float32')
    result_df["probe_number"] = np.full(len(result_df), np.nan, dtype='float32')
    
    # Handle both old format (binary splits) and new format (categorical responses)
    # Check if this is the old format with binary splits
    sample_events = result_df['event_name'].dropna().head(10).astype(str)
    has_binary_format = any('onoff' in event or 'valence' in event for event in sample_events)
    
    if has_binary_format:
        # Old format with binary splits
        binary_split_cols = ["onoff", "selfother", "valence", "time", 
                            "confidence", "average"]
        
        # Create binary and label columns with optimal dtypes
        for col in binary_split_cols:
            result_df[col] = np.zeros(len(result_df), dtype='float64')
            result_df[f"{col}_binary"] = np.zeros(len(result_df), dtype='int8')
            result_df[f"{col}_label"] = pd.Categorical(["low"] * len(result_df), 
                                                      categories=["low", "high"])
    else:
        # New format with categorical responses (mind wandering states)
        probe_response_order = ['on-task', 'about-task', 'distracted', 'deliberate', 'spontaneous', 'blank', 'asleep']
        
        # Create probe response, confidence, and immersion columns
        result_df["probe_response"] = pd.Categorical(["unknown"] * len(result_df), 
                                                   categories=probe_response_order + ['unknown'])
        
        # Confidence levels in order
        confidence_levels = ['not confident', 'somewhat confident', 'very confident', 'completely confident']
        result_df["confidence_level"] = pd.Categorical(["unknown"] * len(result_df),
                                                      categories=confidence_levels + ['unknown'])
        
        # Immersion levels in order  
        immersion_levels = ['not immersed', 'a little immersed', 'somewhat immersed', 'very immersed', 'completely immersed']
        result_df["immersion_level"] = pd.Categorical(["unknown"] * len(result_df),
                                                     categories=immersion_levels + ['unknown'])
    
    # Vectorized processing using string methods and regex
    event_names = result_df['event_name'].fillna('').astype(str)
    
    # Extract trial_type vectorized
    go_mask = event_names.str.startswith('go/')
    nogo_mask = event_names.str.startswith('nogo/')
    result_df.loc[go_mask, 'trial_type'] = 'go'
    result_df.loc[nogo_mask, 'trial_type'] = 'nogo'
    
    # Extract correctness vectorized
    correct_mask = event_names.str.contains('/correct/')
    incorrect_mask = event_names.str.contains('/incorrect/')
    result_df.loc[correct_mask, 'correctness'] = 'correct'
    result_df.loc[incorrect_mask, 'correctness'] = 'incorrect'
    
    if has_binary_format:
        # Extract numeric values for binary columns using vectorized regex
        for col in binary_split_cols:
            pattern = rf"{col}(\d+)"
            matches = event_names.str.extract(pattern, expand=False)
            valid_matches = pd.to_numeric(matches, errors='coerce')
            mask = ~valid_matches.isna()
            
            if mask.any():
                # Store the actual numeric value
                result_df.loc[mask, col] = valid_matches[mask].astype('float64')
                
                # Determine binary value and label vectorized
                binary_vals = (valid_matches >= 50).astype('int8')
                result_df.loc[mask, f"{col}_binary"] = binary_vals[mask]
                result_df.loc[mask, f"{col}_label"] = np.where(
                    binary_vals[mask] == 1, "high", "low")
        
        # Extract distance_to_probe vectorized (old format: /-X/)
        dist_matches = event_names.str.extract(r'/-(\d+)/', expand=False)
        valid_dist = pd.to_numeric(dist_matches, errors='coerce')
        dist_mask = valid_dist.notna()
        if dist_mask.any():
            result_df.loc[dist_mask, 'distance_to_probe'] = -valid_dist[dist_mask].astype('int32')
        
        # Extract probe number vectorized (old format: /probeX)
        probe_matches = event_names.str.extract(r'/probe(\d+)', expand=False)
        valid_probe = pd.to_numeric(probe_matches, errors='coerce')
        probe_mask = valid_probe.notna()
        if probe_mask.any():
            result_df.loc[probe_mask, 'probe_number'] = valid_probe[probe_mask].astype('int32')
    else:
        # New format processing
        # Extract distance_to_probe vectorized (new format: /segment-X/distance,trial/)
        dist_matches = event_names.str.extract(r'/segment-\d+/\d+,(-?\d+)/', expand=False)
        valid_dist = pd.to_numeric(dist_matches, errors='coerce')
        
        # KEY CHANGE 1: Only fill rows that successfully matched the regex
        dist_mask = valid_dist.notna()
        if dist_mask.any():
            result_df.loc[dist_mask, 'distance_to_probe'] = valid_dist[dist_mask].astype('int32')
        
        # Extract probe number vectorized (new format: /segment-X/)
        probe_matches = event_names.str.extract(r'/segment-(\d+)/', expand=False)
        valid_probe = pd.to_numeric(probe_matches, errors='coerce')
        probe_mask = valid_probe.notna()
        if probe_mask.any():
            result_df.loc[probe_mask, 'probe_number'] = valid_probe[probe_mask].astype('int32')
        
        # Extract probe response (mind wandering states)
        for probe_resp in probe_response_order:
            mask = event_names.str.contains(f'/{probe_resp}/', na=False)
            result_df.loc[mask, 'probe_response'] = probe_resp
        
        # Extract confidence levels
        for conf_level in confidence_levels:
            mask = event_names.str.contains(f'/{conf_level}/', na=False)
            result_df.loc[mask, 'confidence_level'] = conf_level
        
        # Extract immersion levels 
        for imm_level in immersion_levels:
            # Use word boundaries to avoid partial matches
            pattern = f'/{re.escape(imm_level)}(?:/|$)'
            mask = event_names.str.contains(pattern, na=False, regex=True)
            result_df.loc[mask, 'immersion_level'] = imm_level
        
        # KEY CHANGE 3: Optional - propagate probe labels to all trials of same probe_number
        # This allows each trial to inherit the mental state declared in the probe
        for col in ['probe_response', 'confidence_level', 'immersion_level']:
            if col in result_df.columns:
                # Group by probe_number and forward-fill within each group
                result_df[col] = result_df.groupby('probe_number', observed=True)[col].bfill()
    
    # Force garbage collection
    gc.collect()
    
    return result_df


def process_single_file(file_info):
    """
    Process a single file - designed for parallel execution.
    
    Parameters:
    -----------
    file_info : tuple
        (file_path, trials_before_probe, only_go_correct, outlier_method, outlier_threshold)
    
    Returns:
    --------
    pandas.DataFrame or None
        Processed DataFrame or None if error
    """
    file, trials_before_probe, only_go_correct, outlier_method, outlier_threshold = file_info
    
    try:
        filename = os.path.basename(file)
        
        # Optimized reading with better dtypes
        file_size = os.path.getsize(file)
        
        # Define optimal dtypes upfront
        dtype_dict = {
            'subject_id': 'category',
            'marker': 'category', 
            'task': 'category',
            'channel': 'category',
            'unit': 'category',
            'epoch': 'int32',  # More efficient than default int64
            'event_id': 'int32',
            'value': 'float64'  # Need full precision for very small spectral values (~1e-13)
        }
        
        # Read file with chunking for large files
        if file_size > 50 * 1024 * 1024:  # 50MB threshold
            chunks = []
            try:
                for chunk in pd.read_csv(file, dtype=dtype_dict, chunksize=250000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                del chunks
            except pd.errors.EmptyDataError:
                print(f"Warning: Empty file {filename}")
                return None
            except pd.errors.ParserError as e:
                print(f"Warning: Parse error in file {filename}: {e}")
                return None
        else:
            try:
                df = pd.read_csv(file, dtype=dtype_dict)
            except pd.errors.EmptyDataError:
                print(f"Warning: Empty file {filename}")
                return None
            except pd.errors.ParserError as e:
                print(f"Warning: Parse error in file {filename}: {e}")
                return None
        
        # Early validation and filtering
        if 'value' not in df.columns or df.empty:
            return None
        
        # Handle missing unit column
        if 'unit' not in df.columns:
            df['unit'] = df['marker'].apply(get_marker_unit_from_name)
        
        # Apply unit conversion early
        apply_unit_conversions(df)
        
        # Process event names with vectorized method
        df = process_event_names_vectorized(df, show_progress=False)
        
        if df is None or df.empty:
            return None
        
        # Early filtering to reduce data size significantly
        df = apply_early_filters(df, trials_before_probe, only_go_correct)
        
        if df is None or df.empty:
            return None
        
        # APPLY OUTLIER REMOVAL BEFORE PROCESSING EVENT NAMES
        if outlier_method and outlier_method != 'none':
            df = apply_outlier_removal(df, method=outlier_method, threshold=outlier_threshold, show_progress=False)
            
            if df is None or df.empty:
                print(f"Warning: No data remaining after outlier removal for {filename}")
                return None
        
        return df
        
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None


def apply_unit_conversions(df):
    """Apply unit conversions in-place for efficiency."""
    if 'marker' not in df.columns:
        return
    
    # NOTE: Unit conversions have been moved to the compute_mne_markers.py script
    # The raw MNE marker files now already contain values in the correct units
    # No additional conversions are needed during aggregation
    
    # REMOVED: Spectral markers conversion (already applied in compute script)
    # spectral_markers = ['d', 't', 'a', 'b', 'g', 'se', 'msf', 'sef90', 'sef95']
    # spectral_mask = df['marker'].isin(spectral_markers)
    # if spectral_mask.any():
    #     df.loc[spectral_mask, 'value'] *= 1e12  # REMOVED - causes double conversion
    
    # REMOVED: Time-domain markers conversion (already applied in compute script)
    # time_markers = ['cnv', 'p1', 'n1', 'p2', 'p3a', 'p3b', 
    #                'mean_amp', 'p2p_amp', 'rms', 'std', 'var', 'kurtosis', 'skew']
    # time_mask = df['marker'].isin(time_markers)
    # if time_mask.any():
    #     df.loc[time_mask, 'value'] *= 1e6  # REMOVED - causes double conversion
    
    # No conversions applied - values are already in correct units
    pass


def apply_early_filters(df, trials_before_probe, only_go_correct):
    """Apply early filtering to reduce data size before aggregation."""
    # Filter by distance to probe early
    if 'distance_to_probe' in df.columns and 'probe_number' in df.columns:
        if len(df) < 1000:  # Test data
            dist_mask = df['distance_to_probe'] <= 0
        else:
            dist_mask = ((df['distance_to_probe'] >= -trials_before_probe) & 
                        (df['distance_to_probe'] <= 0) & 
                        (df['probe_number'] >= 0))
        
        df = df[dist_mask].copy()
        
        if df.empty:
            return None
    
    # Apply go/correct filter early if needed
    if only_go_correct and 'trial_type' in df.columns and 'correctness' in df.columns:
        go_correct_mask = ((df['trial_type'] == 'go') & 
                          (df['correctness'] == 'correct'))
        df = df[go_correct_mask].copy()
        
        if df.empty:
            return None
    
    return df


def remove_subject_outliers_single_file(df, method='iqr', iqr_factor=1.5, z_threshold=3.0):
    """
    Remove outliers for the single subject in this file based on marker-channel means.
    This is much more efficient than doing it across all subjects later.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame for a single subject
    method : str
        Method for outlier detection ('iqr', 'zscore', or 'both')
    iqr_factor : float
        Factor for IQR-based outlier detection
    z_threshold : float
        Z-score threshold for outlier detection
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outlier marker-channel combinations removed
    """
    if df is None or df.empty:
        return df
    
    # Calculate subject means for each marker-channel combination
    subject_means = df.groupby(['marker', 'channel'], observed=True)['value'].mean()
    
    if subject_means.empty:
        return df
    
    # For single subject, we can't use population statistics
    # Instead, we'll flag extreme values within this subject's data
    # This catches technical artifacts or clearly problematic channels
    
    outlier_marker_channels = set()
    
    for (marker, channel), mean_val in subject_means.items():
        # Get all values for this marker-channel combination
        mask = (df['marker'] == marker) & (df['channel'] == channel)
        values = df.loc[mask, 'value'].values
        
        if len(values) == 0:
            continue
        
        # Flag if the mean is extremely different from the median (suggests artifacts)
        median_val = np.median(values)
        
        # Simple artifact detection: if mean is very different from median
        if abs(mean_val) > 0:  # Avoid division by zero
            relative_diff = abs(mean_val - median_val) / abs(mean_val)
            if relative_diff > 2.0:  # Mean is more than 200% different from median
                outlier_marker_channels.add((marker, channel))
                continue
        
        # Also flag channels with extreme variance (likely artifacts)
        if len(values) > 1:
            std_val = np.std(values)
            if std_val > 0 and abs(mean_val) > 0:
                cv = std_val / abs(mean_val)  # Coefficient of variation
                if cv > 5.0:  # Very high variability
                    outlier_marker_channels.add((marker, channel))
    
    # Remove outlier marker-channel combinations
    if outlier_marker_channels:
        for marker, channel in outlier_marker_channels:
            mask = (df['marker'] == marker) & (df['channel'] == channel)
            df = df[~mask].copy()
    
    return df


def find_files_bids(input_dir, desc='mne_markers'):
    """
    Find all MNE marker files in the BIDS features folder structure.
    
    Parameters:
    -----------
    input_dir : str
        Directory to search in (should be the features folder root)
    desc : str
        Description for the markers (default: 'mne_markers')
    
    Returns:
    --------
    list
        List of file paths matching the specified type in BIDS structure
    """
    # Try to import BIDS compliance function
    utils_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils')
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    
    try:
        from bids_compliance import find_eeg_files_bids
        files = find_eeg_files_bids(input_dir, desc=desc)
        return files
    except ImportError as e:
        print(f"Error importing BIDS compliance functions: {e}")
        # Fallback to manual search
        from pathlib import Path
        
        features_path = Path(input_dir)
        if not features_path.exists():
            print(f"Features folder not found: {input_dir}")
            return []
        
        # Pattern to match EEG marker files in BIDS structure
        pattern = f"**/sub-*/ses-*/eeg/*desc-{desc}_features.csv"
        files = list(features_path.glob(pattern))
        
        return [str(f) for f in files]


def find_files(input_dir, file_type='mne_markers'):
    """
    Find all MNE marker files in the input directory.
    Updated to work with BIDS structure by default, with fallback to old structure.
    
    Parameters:
    -----------
    input_dir : str
        Directory to search in
    file_type : str
        Type of file to search for (default: 'mne_markers')
    
    Returns:
    --------
    list
        List of file paths matching the specified type
    """
    # First try BIDS structure
    bids_files = find_files_bids(input_dir, desc=file_type)
    
    if bids_files:
        print(f"Found {len(bids_files)} files using BIDS structure")
        return bids_files
    
    # Fallback to old flat structure
    print("No BIDS files found, trying flat structure...")
    pattern = os.path.join(input_dir, f"*_{file_type}.csv")
    files = glob.glob(pattern)
    
    if files:
        print(f"Found {len(files)} files using flat structure")
    
    return files


def load_and_process_files_parallel(files, trials_before_probe=5, only_go_correct=False, max_workers=None, remove_outliers=True, outlier_method='iqr', outlier_threshold=None):
    """
    Load and process files in parallel for much better performance.
    
    Parameters:
    -----------
    files : list
        List of file paths to load and process
    trials_before_probe : int
        Number of trials before probe
    only_go_correct : bool
        Whether to filter to go/correct trials only
    max_workers : int
        Maximum number of worker processes
    remove_outliers : bool
        Whether to remove outliers at subject level
    outlier_method : str
        Method for outlier detection
    outlier_threshold : float or dict
        Threshold parameter for outlier detection
    
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with additional condition columns
    """
    if not files:
        print("No files found matching the specified pattern.")
        return None
    
    if max_workers is None:
        max_workers = min(len(files), mp.cpu_count() - 1, 8)  # Cap at 8 to avoid memory issues
    
    print(f"Processing {len(files)} files with {max_workers} parallel workers...")
    if remove_outliers:
        print(f"Subject-level outlier removal enabled using {outlier_method} method")
    
    # Prepare file information for parallel processing
    # Set default threshold if not provided
    if outlier_threshold is None:
        if outlier_method == 'iqr':
            outlier_threshold = 1.5
        elif outlier_method == 'zscore':
            outlier_threshold = 3.0
        elif outlier_method == 'modified_zscore':
            outlier_threshold = 3.5
        elif outlier_method == 'percentile':
            outlier_threshold = {'lower': 1, 'upper': 99}
        else:
            outlier_threshold = None
    
    # Only pass outlier_method and outlier_threshold if outlier removal is enabled
    if remove_outliers and outlier_method != 'none':
        file_infos = [(file, trials_before_probe, only_go_correct, outlier_method, outlier_threshold) for file in files]
    else:
        file_infos = [(file, trials_before_probe, only_go_correct, 'none', None) for file in files]
    
    processed_dfs = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(process_single_file, info): info[0] 
                         for info in file_infos}
        
        # Collect results with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None:
                        processed_dfs.append(result)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
                finally:
                    pbar.update(1)
    
    if not processed_dfs:
        print("No valid data found in the specified files.")
        return None
    
    # Combine all dataframes efficiently
    print("Combining all processed data...")
    try:
        final_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Final validation
        if 'value' not in final_df.columns:
            print("Error: Missing 'value' column in combined data")
            return None
        
        # Verify 'value' column is numeric
        if not pd.api.types.is_numeric_dtype(final_df['value']):
            print("Warning: 'value' column is not numeric. Attempting to convert...")
            final_df['value'] = pd.to_numeric(final_df['value'], errors='coerce')
            if final_df['value'].isna().all():
                print("Error: Failed to convert 'value' column to numeric")
                return None
        
        print(f"Final combined dataset has {len(final_df)} rows and {len(final_df.columns)} columns")
        return final_df
        
    except Exception as e:
        print(f"Error combining data: {e}")
        return None
    finally:
        # Clear memory
        del processed_dfs
        gc.collect()


def get_marker_unit_from_name(marker_name):
    """
    Return the unit for each marker type (backward compatibility function).
    
    Parameters
    ----------
    marker_name : str
        Name of the marker
        
    Returns
    -------
    str
        Unit string for the marker
    """
    # Spectral power markers (absolute power)
    spectral_power_markers = ['d', 't', 'a', 'b', 'g']
    
    # Normalized spectral markers (ratios)
    normalized_markers = ['d_n', 't_n', 'a_n', 'b_n', 'g_n']
    
    # Frequency markers
    frequency_markers = ['msf', 'sef90', 'sef95']
    
    # Amplitude markers (time domain)
    amplitude_markers = ['cnv', 'p1', 'n1', 'p2', 'p3a', 'p3b', 
                        'mean_amp', 'p2p_amp', 'rms', 'std']
    
    # Variance markers
    variance_markers = ['var']
    
    # Dimensionless markers
    dimensionless_markers = ['se', 'kurtosis', 'skew']
    
    if marker_name in spectral_power_markers:
        return 'µV²/Hz'
    elif marker_name in normalized_markers:
        return 'ratio'
    elif marker_name in frequency_markers:
        return 'Hz'
    elif marker_name in amplitude_markers:
        return 'µV'
    elif marker_name in variance_markers:
        return 'µV²'
    elif marker_name in dimensionless_markers:
        return 'dimensionless'
    else:
        return 'unknown'


def aggregate_by_probe_optimized(df, trials_before_probe=5, 
                                only_go_correct=False, input_files=None, 
                                max_workers=None, remove_outliers=True, 
                                outlier_method='iqr', outlier_threshold=None):
    """
    Optimized version of aggregate_by_probe with parallel processing and subject-level outlier removal.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the processed data, or None to load from input_files
    trials_before_probe : int
        Number of trials before each probe to include in the aggregation
    only_go_correct : bool
        Whether to include only 'go' and 'correct' trials
    input_files : list
        List of files to process if df is None
    max_workers : int
        Maximum number of worker processes
    remove_outliers : bool
        Whether to remove outliers at subject level during file loading
    outlier_method : str
        Method for outlier detection (currently uses artifact detection)
    outlier_threshold : float or dict
        Threshold parameter for outlier detection
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame aggregated by probe, including only relevant trials
    """
    # If df is None, load data from files with parallel processing
    if df is None:
        if input_files is None or not input_files:
            print("Error: No data or input files provided to aggregate_by_probe")
            return None
        
        print(f"Loading data from {len(input_files)} MNE marker files...")
        df = load_and_process_files_parallel(
            input_files, 
            trials_before_probe, 
            only_go_correct, 
            max_workers,
            remove_outliers,
            outlier_method,
            outlier_threshold
        )
        
        if df is None or df.empty:
            print("No data loaded from input files.")
            return None
        
        print(f"Loaded data shape: {df.shape}")
    
    if df is None or df.empty:
        print("Empty DataFrame provided to aggregate_by_probe")
        return None
    
    # At this point, early filtering and subject-level outlier removal have been applied during file processing
    print(f"Data ready for probe aggregation - rows: {len(df)}")
    
    # Identify condition columns we want to preserve
    condition_columns = []
    for col in ["onoff", "selfother", "valence", "time", "confidence", "average"]:
        label_col = f"{col}_label"
        if label_col in df.columns:
            condition_columns.append(label_col)
        elif f"{col}_binary" in df.columns:
            condition_columns.append(f"{col}_binary")
        elif col in df.columns:
            condition_columns.append(col)
    
    # Add trial_type and correctness if present
    for special_col in ['trial_type', 'correctness']:
        if special_col in df.columns:
            condition_columns.append(special_col)
    
    # Define groupby columns for MNE markers (all are per-electrode)
    groupby_cols = ['subject_id', 'task', 'probe_number', 'marker', 'channel']
    
    # Add unit column if it exists
    if 'unit' in df.columns:
        groupby_cols.append('unit')
    
    # Add all condition columns to the groupby 
    groupby_cols.extend(condition_columns)
    
    # Check if all groupby columns exist
    missing_groupby = [col for col in groupby_cols if col not in df.columns]
    if missing_groupby:
        print(f"Warning: Missing groupby columns: {missing_groupby}")
        groupby_cols = [col for col in groupby_cols if col in df.columns]
        
        if not groupby_cols:
            print("Error: No groupby columns available")
            return None
    
    print(f"Aggregating by probe ({len(df)} trials)...")
    print(f"Using groupby columns: {groupby_cols}")
    
    # Optimized aggregation with better memory management
    try:
        # Use efficient aggregation with multiple robust statistics at once
        probe_agg = df.groupby(groupby_cols, observed=True, sort=False)['value'].agg([
            ('trimmed_mean', lambda x: safe_trimmed_mean(x, 0.1)),  # 10% trimmed mean
            ('median', 'median'),
            ('mean', 'mean'),  # Keep traditional mean for comparison
            ('std', 'std'), 
            ('count', 'count')
        ]).reset_index()
        
        # Handle any NaN std values
        probe_agg['std'] = probe_agg['std'].fillna(0.0)
        
        # Verify that statistical columns maintain float64 and don't become object
        for stat_col in ['trimmed_mean', 'median', 'mean']:
            if stat_col in probe_agg.columns:
                probe_agg[stat_col] = pd.to_numeric(probe_agg[stat_col], errors='coerce')
        
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None
    
    # Free memory
    del df
    gc.collect()
    
    if probe_agg is not None:
        print(f"Aggregated into {len(probe_agg)} probe-level records")
    else:
        print("No data after aggregation.")
    
    return probe_agg


def aggregate_by_conditions_optimized(df, conditions):
    """
    Optimized version of aggregate_by_conditions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    conditions : list
        List of condition columns to group by
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated DataFrame
    """
    if df is None or df.empty:
        return None
    
    print("Preparing for condition-based aggregation...")
    
    # Define base groupby columns for MNE markers
    groupby_cols = ['subject_id', 'task', 'marker', 'channel']
    
    # Add unit column if it exists
    if 'unit' in df.columns:
        groupby_cols.append('unit')
    
    # Identify condition columns efficiently
    all_condition_cols = []
    for condition in conditions:
        if condition in ['trial_type', 'correctness']:
            if condition in df.columns:
                all_condition_cols.append(condition)
            continue
        
        # For regular conditions, try to find label version
        label_col = f"{condition}_label"
        if label_col in df.columns:
            all_condition_cols.append(label_col)
    
    if not all_condition_cols:
        print(f"Warning: None of the specified conditions {conditions} found in data.")
        return None
    
    # Add all identified condition label columns to groupby
    groupby_cols.extend(all_condition_cols)
    
    print(f"Aggregating by conditions: {conditions}")
    print(f"Using groupby columns: {groupby_cols}")
    
    # Ensure all required columns exist
    missing_cols = [col for col in groupby_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns for groupby: {missing_cols}")
        groupby_cols = [col for col in groupby_cols if col in df.columns]
    
    # Prepare aggregation with data validation
    try:
        valid_data = (~df['mean'].isna()) & (df['count'] > 0)
        if valid_data.any():
            df_filtered = df[valid_data]
            agg_df = df_filtered.groupby(groupby_cols, observed=True, sort=False)['mean'].agg([
                'mean', 'std', 'count'
            ]).reset_index()
            
            # Rename columns for consistency
            agg_df.columns = [*groupby_cols, 'mean', 'std', 'count']
        else:
            agg_df = None
    except Exception as e:
        print(f"Error during aggregation: {e}")
        return None
    
    # Free memory
    del df
    gc.collect()
    
    if agg_df is None or agg_df.empty:
        print("No data after aggregation.")
        return None
    
    # Remove rows with count=0
    agg_df = agg_df[agg_df['count'] > 0]
    
    print(f"Final aggregated data has {len(agg_df)} rows and {len(agg_df.columns)} columns")
    return agg_df


def aggregate_by_subject_optimized(df, conditions):
    """
    Optimized version of aggregate_by_subject.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    conditions : list
        List of condition columns to group by
    
    Returns:
    --------
    pandas.DataFrame
        Aggregated DataFrame by subject
    """
    if df is None or df.empty:
        return None
    
    # First, aggregate by task-level
    print("First aggregating by task...")
    task_df = aggregate_by_conditions_optimized(df, conditions)
    
    if task_df is None or task_df.empty:
        return None
    
    # Define groupby columns for subject-level aggregation
    groupby_cols = ['subject_id', 'marker', 'channel']
    
    # Add unit column if it exists
    if 'unit' in task_df.columns:
        groupby_cols.append('unit')
    
    # Add categorical condition columns to groupby
    for condition in conditions:
        label_col = f"{condition}_label"
        if label_col in task_df.columns:
            groupby_cols.append(label_col)
    
    # Add trial_type and correctness if present
    for special_col in ['trial_type', 'correctness']:
        if special_col in task_df.columns and special_col in conditions:
            groupby_cols.append(special_col)
    
    # Perform the subject-level aggregation
    print("Now aggregating across subjects...")
    print(f"Using groupby columns: {groupby_cols}")
    
    try:
        subject_df = task_df.groupby(groupby_cols, observed=True, sort=False)['mean'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        # Rename columns for consistency
        subject_df.columns = [*groupby_cols, 'mean', 'std', 'count']
        
    except Exception as e:
        print(f"Error during subject-level aggregation: {e}")
        return task_df  # Return task-level as fallback
    
    # Free memory
    del task_df
    gc.collect()
    
    print(f"Aggregated into {len(subject_df)} subject-level records")
    return subject_df


# Removed the complex outlier removal function - now done at file level for efficiency


def run_aggregation(input_dir='./results/mne_markers', 
                    output_dir='./results/aggregated_mne_markers',
                    conditions=['onoff'],
                    aggregate_level='task',
                    trials_before_probe=5,
                    only_go_correct=False,
                    max_workers=None,
                    output_file=None,
                    remove_outliers=True,
                    outlier_method='modified_zscore',
                    outlier_threshold=3.5):
    """
    Run the optimized MNE marker aggregation process with outlier removal.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the MNE marker CSV files
    output_dir : str
        Directory to save the aggregated results
    conditions : list
        Conditions to aggregate by (e.g., ['onoff', 'valence'])
    aggregate_level : str
        Level to aggregate: 'probe', 'task' or 'subject'
    trials_before_probe : int
        Number of trials before each probe to include
    only_go_correct : bool
        Whether to include only 'go' and 'correct' trials
    max_workers : int
        Maximum number of worker processes (auto-detected if None)
    output_file : str, optional
        Output filename (auto-generated if not provided)
    remove_outliers : bool
        Whether to remove outliers based on subject means
    outlier_method : str
        Method for outlier detection ('iqr', 'zscore', 'modified_zscore', 'percentile', or 'none')
    outlier_threshold : float
        Threshold parameter for outlier detection
    
    Returns:
    --------
    str
        Path to the saved output file
    """
    # Handle input/output paths to be relative to project root
    if input_dir.startswith('./') or not os.path.isabs(input_dir):
        input_dir = os.path.abspath(os.path.join(project_root, input_dir.lstrip('./')))
    
    if output_dir.startswith('./') or not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, output_dir.lstrip('./')))
    
    # Validate input parameters
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return None
    
    if aggregate_level not in ['probe', 'task', 'subject']:
        print(f"Error: Invalid aggregate level {aggregate_level}. "
              f"Must be one of: 'probe', 'task', 'subject'")
        return None
    
    # Validate conditions
    if not conditions or not isinstance(conditions, list):
        print("Warning: Invalid conditions provided. Using default 'onoff'")
        conditions = ['onoff']
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return None
    
    # Set optimal number of workers if not provided
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 8)  # Leave one core free, cap at 8
    
    # Print configuration for logging
    print("\n=== Optimized MNE Marker Aggregation Configuration ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Conditions: {conditions}")
    print(f"Aggregation level: {aggregate_level}")
    print(f"Trials before probe: {trials_before_probe}")
    print(f"Only include Go/Correct trials: {only_go_correct}")
    print(f"Max parallel workers: {max_workers}")
    
    # Find MNE marker files
    try:
        files = find_files(input_dir, 'mne_markers')
        print(f"Found {len(files)} MNE marker files to process.")
        
        if not files:
            print(f"No MNE marker files found in {input_dir}")
            return None
    except Exception as e:
        print(f"Error finding files: {e}")
        return None
    
    try:
        # First aggregation: by probe (with parallel processing)
        print("Aggregating by probe with parallel processing...")
        probe_df = aggregate_by_probe_optimized(
            df=None,
            trials_before_probe=trials_before_probe,
            only_go_correct=only_go_correct,
            input_files=files,
            max_workers=max_workers,
            remove_outliers=remove_outliers,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold
        )
        
        if probe_df is None or probe_df.empty:
            print("No data after probe-level aggregation, exiting.")
            return None
        
        print(f"Probe-level data shape: {probe_df.shape}")
        
        # Choose aggregation level
        if aggregate_level == 'probe':
            agg_df = probe_df
        else:
            try:
                if aggregate_level == 'task':
                    agg_df = aggregate_by_conditions_optimized(probe_df, conditions)
                else:  # subject level
                    agg_df = aggregate_by_subject_optimized(probe_df, conditions)
                    
                if agg_df is None or agg_df.empty:
                    print("No data after condition aggregation, using probe data instead.")
                    agg_df = probe_df
            except Exception as e:
                print(f"Error during {aggregate_level}-level aggregation: {e}")
                print("Using probe-level data as fallback...")
                agg_df = probe_df
        
        # Free memory
        if aggregate_level != 'probe':
            del probe_df
            gc.collect()
        
        # Generate output filename if not provided
        if output_file is None:
            conditions_str = '_'.join(conditions).replace('/', '_').replace('\\', '_')
            go_correct_str = "_go_correct" if only_go_correct else ""
            outlier_str = f"_{outlier_method}" if outlier_method != 'none' else ""
            output_file_name = (
                f"aggregated_mne_markers_{conditions_str}_"
                f"{trials_before_probe}trials{go_correct_str}{outlier_str}_{aggregate_level}.csv"
            )
            output_path = os.path.join(output_dir, output_file_name)
        else:
            output_path = os.path.join(output_dir, output_file)
        
        print(f"Will write to: {output_path}")
        
        # Save the aggregated data
        try:
            # Use scientific notation for very small numbers
            agg_df.to_csv(output_path, index=False, float_format='%.6e')
            print(f"Data saved to {output_path}")
        except Exception as e:
            print(f"Error saving output file: {e}")
            return None
        
        # Final cleanup
        del agg_df
        gc.collect()
        
        return output_path
    except Exception as e:
        print(f"Unexpected error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main function that handles command-line arguments and direct execution.
    """
    # Set up basic logging
    print(f"Starting optimized MNE marker aggregation at {pd.Timestamp.now()}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Check if arguments were provided
    if len(sys.argv) > 1:
        # Use argparse for command-line arguments
        parser = argparse.ArgumentParser(
            description='Aggregate MNE markers data based on conditions with optimized parallel processing.'
        )
        parser.add_argument(
            '--input_dir', type=str, default='./results/mne_markers',
            help='Directory containing the MNE marker CSV files'
        )
        parser.add_argument(
            '--output_dir', type=str, default='./results/aggregated_mne_markers',
            help='Directory to save the aggregated results'
        )
        parser.add_argument(
            '--conditions', type=str, nargs='+', 
            default=['onoff'],
            help='Conditions to aggregate by (e.g., onoff valence trial_type)'
        )
        parser.add_argument(
            '--aggregate_level', type=str, choices=['probe', 'task', 'subject'], 
            default='task', help='Level to aggregate: probe, task or subject'
        )
        parser.add_argument(
            '--trials_before_probe', type=int, default=5,
            help='Number of trials before each probe to include'
        )
        parser.add_argument(
            '--only_go_correct', action='store_true',
            help='Include only go and correct trials'
        )
        parser.add_argument(
            '--max_workers', type=int, 
            help='Maximum number of parallel worker processes (auto-detected if not provided)'
        )
        parser.add_argument(
            '--output_file', type=str, 
            help='Output filename (auto-generated if not provided)'
        )
        parser.add_argument(
            '--remove_outliers', action='store_true', default=True,
            help='Remove outliers based on subject means (default: True)'
        )
        parser.add_argument(
            '--no_remove_outliers', dest='remove_outliers', action='store_false',
            help='Disable outlier removal'
        )
        parser.add_argument(
            '--outlier_method', type=str, choices=['iqr', 'zscore', 'modified_zscore', 'percentile', 'none'], 
            default='iqr', help='Method for outlier detection (default: iqr)'
        )
        parser.add_argument(
            '--outlier_threshold', type=float, default=None,
            help='Threshold for outlier detection (auto-set based on method if not provided)'
        )
        parser.add_argument(
            '--quiet', action='store_true',
            help='Disable all progress bars for cleaner logs'
        )
        
        args = parser.parse_args()
        
        # Set tqdm to respect the quiet flag
        if args.quiet:
            from functools import partialmethod
            tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        
        # Run with the provided arguments
        try:
            output_path = run_aggregation(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                conditions=args.conditions,
                aggregate_level=args.aggregate_level,
                trials_before_probe=args.trials_before_probe,
                only_go_correct=args.only_go_correct,
                max_workers=args.max_workers,
                output_file=args.output_file,
                remove_outliers=args.remove_outliers,
                outlier_method=args.outlier_method,
            )
            
            if output_path:
                print(f"Aggregation completed successfully. Results saved to: {output_path}")
                sys.exit(0)
            else:
                print("Aggregation failed. Check the logs for errors.")
                sys.exit(1)
        except Exception as e:
            print(f"Error in main execution: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main() 