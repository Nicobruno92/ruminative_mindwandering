"""
Helper functions for the LMM-based spatial cluster permutation testing pipeline.

This module provides utility functions for data preprocessing, including:
- Subject-level normalization (z-score, min-max, robust scaling)
- Data quality checks
- Preprocessing utilities
- PCA data loading and merging
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Literal, List, Union
from pathlib import Path
import warnings
import re


def extract_all_formula_variables(formula: str) -> list:
    """
    Extract all variable names from a formula (excluding 'power' and 'subject').
    
    Parameters
    ----------
    formula : str
        R-style formula (e.g., "power ~ onoff + time_on_task + (1|subject)")
        
    Returns
    -------
    list
        List of variable names used in the formula
        
    Examples
    --------
    >>> extract_all_formula_variables("power ~ onoff + (1|subject)")
    ['onoff']
    >>> extract_all_formula_variables("power ~ onoff + time_on_task + (1|subject)")
    ['onoff', 'time_on_task']
    """
    # Extract all alphanumeric variable names
    variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)
    
    # Filter out reserved words
    reserved = {'power', 'subject'}
    variables = [v for v in variables if v not in reserved]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vars = []
    for v in variables:
        if v not in seen:
            unique_vars.append(v)
            seen.add(v)
    
    return unique_vars


def extract_fixed_effects_from_formula(formula: str) -> str:
    """
    Extract fixed effects from LMM formula to create folder name.
    
    Extracts the predictor variables (fixed effects) from the formula,
    excluding the response variable and random effects.
    
    Parameters
    ----------
    formula : str
        LMM formula in the format "response ~ predictors + (random_effects)"
        Examples:
        - "power ~ onoff + (1|subject)" -> "onoff"
        - "power ~ onoff + valence + (1|subject)" -> "onoff_valence"
        - "power ~ onoff * valence + (1|subject)" -> "onoff_x_valence"
        - "power ~ onoff + (1 + onoff|subject)" -> "onoff"
    
    Returns
    -------
    str
        Folder name based on fixed effects, with spaces and operators replaced by underscores
        
    Examples
    --------
    >>> extract_fixed_effects_from_formula("power ~ onoff + (1|subject)")
    'onoff'
    >>> extract_fixed_effects_from_formula("power ~ onoff + valence + (1|subject)")
    'onoff_valence'
    >>> extract_fixed_effects_from_formula("power ~ onoff * valence + arousal + (1|subject)")
    'onoff_x_valence_arousal'
    """
    # Remove whitespace for easier parsing
    formula_clean = formula.strip()
    
    # Split by ~ to separate response from predictors
    if '~' not in formula_clean:
        raise ValueError(f"Invalid formula format: {formula}. Expected format: 'response ~ predictors'")
    
    parts = formula_clean.split('~')
    if len(parts) != 2:
        raise ValueError(f"Invalid formula format: {formula}. Expected single '~' separator")
    
    predictors_part = parts[1].strip()
    
    # Remove random effects (anything in parentheses)
    # Match patterns like (1|subject) or (1 + onoff|subject)
    fixed_effects_only = re.sub(r'\([^)]*\)', '', predictors_part)
    
    # Clean up the remaining string
    fixed_effects_only = fixed_effects_only.strip()
    
    # Handle interaction terms: replace * with 'x' for folder name
    # This makes "onoff * valence" become "onoff_x_valence"
    fixed_effects_only = fixed_effects_only.replace('*', ' x ')
    
    # Replace +, :, and multiple spaces with single space
    fixed_effects_only = re.sub(r'[+:]', ' ', fixed_effects_only)
    fixed_effects_only = re.sub(r'\s+', ' ', fixed_effects_only)
    fixed_effects_only = fixed_effects_only.strip()
    
    # Split into individual terms and remove duplicates while preserving order
    terms = []
    seen = set()
    for term in fixed_effects_only.split():
        term = term.strip()
        if term and term not in seen and term != '1':  # Exclude intercept '1'
            terms.append(term)
            seen.add(term)
    
    if not terms:
        raise ValueError(f"No fixed effects found in formula: {formula}")
    
    # Join with underscore to create folder name
    folder_name = '_'.join(terms)
    
    return folder_name


def get_model_folder_name(formula: str, predictor_of_interest: str) -> str:
    """
    Build the output folder name encoding both fixed effects and the predictor of interest.

    The resulting name has the form::

        {fixed_effects}__target_{predictor_of_interest}

    This allows running the same formula with different predictors of interest
    (e.g. "onoff", "valence") while saving results to separate directories.

    Parameters
    ----------
    formula : str
        LMM formula, e.g. ``"power ~ onoff + valence + (1|subject)"``.
    predictor_of_interest : str
        The single predictor whose t-statistic drives the cluster permutation
        test, e.g. ``"onoff"`` or ``"valence"``.
        Pass ``"auto"`` or an empty string to fall back to the legacy behaviour
        (folder name = fixed effects only, no ``__target_`` suffix).

    Returns
    -------
    str
        Folder name to use as the model output directory.

    Examples
    --------
    >>> get_model_folder_name(
    ...     "power ~ onoff + valence + (1|subject)", "onoff")
    'onoff_valence__target_onoff'
    >>> get_model_folder_name(
    ...     "power ~ onoff + (1|subject)", "auto")
    'onoff'
    """
    fixed_effects_part = extract_fixed_effects_from_formula(formula)

    # Legacy / auto-detect path: no suffix
    if not predictor_of_interest or predictor_of_interest.strip().lower() == 'auto':
        return fixed_effects_part

    # Sanitize predictor name for filesystem use (handles interaction terms like onoff:valence)
    pred_clean = predictor_of_interest.replace(':', '_').replace(' ', '_').strip('_')
    return f"{fixed_effects_part}__target_{pred_clean}"


def _apply_normalization_method(
    channel_data: np.ndarray,
    valid_data: np.ndarray,
    method: str
) -> np.ndarray:
    """Apply normalization method to channel data."""
    if method == 'zscore':
        mean = np.mean(valid_data)
        std = np.std(valid_data, ddof=1)
        if std > 0:
            return (channel_data - mean) / std
        else:
            return channel_data - mean
    elif method == 'minmax':
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val > min_val:
            return (channel_data - min_val) / (max_val - min_val)
        else:
            return np.where(np.isnan(channel_data), np.nan, 0.5)
    elif method == 'robust':
        median = np.median(valid_data)
        q75, q25 = np.percentile(valid_data, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            return (channel_data - median) / iqr
        else:
            return channel_data - median


def normalize_by_subject(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    method: Literal['zscore', 'minmax', 'robust'] = 'zscore',
    subject_col: str = 'subject',
    channel_wise: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Normalize marker values by subject to account for individual baseline differences.
    
    This function applies subject-specific normalization either per channel
    (channel-wise) or globally across all channels.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with subject identifiers
    method : {'zscore', 'minmax', 'robust'}
        Normalization method:
        - 'zscore': Z-score normalization (mean=0, std=1)
        - 'minmax': Min-max scaling (0-1 range)
        - 'robust': Robust scaling using median and IQR
    subject_col : str
        Column name containing subject identifiers
    channel_wise : bool
        If True (default), normalize each channel independently per subject.
        If False, normalize globally using all channels together per subject.
        - channel_wise=True: Each channel has mean=0, std=1 within subject
          Pros: Controls baseline differences between channels
          Cons: Loses information about relative differences between channels
        - channel_wise=False: All channels use same mean/std per subject
          Pros: Preserves relationships between channels
          Cons: Channels with higher power dominate
    verbose : bool
        Whether to print normalization statistics
        
    Returns
    -------
    np.ndarray
        Normalized power data with same shape as input
        
    Notes
    -----
    - Subjects with insufficient data (< 2 observations) are left unchanged
    - NaN values are preserved and excluded from normalization calculations
    - For 'zscore': (X - mean) / std
    - For 'minmax': (X - min) / (max - min)
    - For 'robust': (X - median) / IQR
    
    Examples
    --------
    >>> # Channel-wise z-score normalization (default)
    >>> normalized_data = normalize_by_subject(power_data, df_behavioral)
    
    >>> # Global z-score normalization (preserves channel relationships)
    >>> normalized_data = normalize_by_subject(
    ...     power_data, df_behavioral, channel_wise=False
    ... )
    
    >>> # Robust normalization (less sensitive to outliers)
    >>> normalized_data = normalize_by_subject(
    ...     power_data, df_behavioral, method='robust'
    ... )
    """
    if power_data.shape[0] != len(df_behavioral):
        raise ValueError(
            f"Shape mismatch: power_data has {power_data.shape[0]} observations "
            f"but df_behavioral has {len(df_behavioral)} rows"
        )
    
    if subject_col not in df_behavioral.columns:
        raise ValueError(f"Column '{subject_col}' not found in df_behavioral")
    
    # Validate method
    valid_methods = ['zscore', 'minmax', 'robust']
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    # Initialize normalized data (copy to avoid modifying original)
    normalized_data = power_data.copy()
    n_observations, n_channels = power_data.shape
    
    # Get unique subjects
    subjects = df_behavioral[subject_col].unique()
    n_subjects = len(subjects)
    
    if verbose:
        mode_str = "channel-wise" if channel_wise else "global"
        print(f"Normalizing data by subject using '{method}' method ({mode_str})...")
        print(f"  Subjects: {n_subjects}")
        print(f"  Channels: {n_channels}")
        print(f"  Observations: {n_observations}")
    
    # Track normalization statistics
    skipped_subjects = []
    normalized_subjects = []
    
    # Normalize each subject independently
    for subject in subjects:
        # Get indices for this subject
        subject_mask = df_behavioral[subject_col] == subject
        subject_indices = np.where(subject_mask)[0]
        n_subject_obs = len(subject_indices)
        
        # Skip subjects with insufficient data
        if n_subject_obs < 2:
            skipped_subjects.append(subject)
            continue
        
        # Extract subject's data
        subject_data = power_data[subject_indices, :]
        
        if channel_wise:
            # OPTION A: Channel-wise normalization
            # Each channel normalized independently
            for ch_idx in range(n_channels):
                channel_data = subject_data[:, ch_idx]
                
                # Skip if all NaN
                if np.all(np.isnan(channel_data)):
                    continue
                
                # Get valid (non-NaN) values
                valid_mask = ~np.isnan(channel_data)
                valid_data = channel_data[valid_mask]
                
                # Skip if insufficient valid data
                if len(valid_data) < 2:
                    continue
                
                # Apply normalization method
                normalized_values = _apply_normalization_method(
                    channel_data, valid_data, method
                )
                
                # Update normalized data for this subject and channel
                normalized_data[subject_indices, ch_idx] = normalized_values
        else:
            # OPTION B: Global normalization
            # All channels use same normalization parameters
            
            # Get all valid data across all channels for this subject
            valid_mask = ~np.isnan(subject_data)
            valid_data = subject_data[valid_mask]
            
            # Skip if insufficient valid data
            if len(valid_data) < 2:
                skipped_subjects.append(subject)
                continue
            
            # Compute global statistics
            if method == 'zscore':
                mean = np.mean(valid_data)
                std = np.std(valid_data, ddof=1)
                if std > 0:
                    normalized_data[subject_indices, :] = (subject_data - mean) / std
                else:
                    normalized_data[subject_indices, :] = subject_data - mean
                    
            elif method == 'minmax':
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                if max_val > min_val:
                    normalized_data[subject_indices, :] = (
                        (subject_data - min_val) / (max_val - min_val)
                    )
                else:
                    normalized_data[subject_indices, :] = np.where(
                        np.isnan(subject_data), np.nan, 0.5
                    )
                    
            elif method == 'robust':
                median = np.median(valid_data)
                q75, q25 = np.percentile(valid_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    normalized_data[subject_indices, :] = (
                        (subject_data - median) / iqr
                    )
                else:
                    normalized_data[subject_indices, :] = subject_data - median
        
        normalized_subjects.append(subject)
    
    if verbose:
        print(f"✓ Normalization completed")
        print(f"  Normalized subjects: {len(normalized_subjects)}")
        if skipped_subjects:
            print(f"  Skipped subjects (< 2 observations): {skipped_subjects}")
        
        # Print summary statistics
        valid_original = power_data[~np.isnan(power_data)]
        valid_normalized = normalized_data[~np.isnan(normalized_data)]
        
        print(f"\n  Original data statistics:")
        print(f"    Mean: {np.mean(valid_original):.3f}")
        print(f"    Std: {np.std(valid_original):.3f}")
        print(f"    Range: [{np.min(valid_original):.3f}, {np.max(valid_original):.3f}]")
        
        print(f"\n  Normalized data statistics:")
        print(f"    Mean: {np.mean(valid_normalized):.3f}")
        print(f"    Std: {np.std(valid_normalized):.3f}")
        print(f"    Range: [{np.min(valid_normalized):.3f}, {np.max(valid_normalized):.3f}]")
    
    return normalized_data


def normalize_predictors(
    df: pd.DataFrame,
    method: Literal['zscore', 'minmax'] = 'zscore',
    subject_col: str = 'subject',
    predictors: Union[List[str], str] = 'all',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Normalize predictor columns within each subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data
    method : {'zscore', 'minmax'}
        Normalization method
    subject_col : str
        Column name containing subject identifiers
    predictors : list of str or 'all'
        List of column names to normalize. If 'all', will attempt to normalize
        all numeric columns except subject_col and known metadata.
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    pd.DataFrame
        DataFrame with normalized predictors (new columns created if needed, 
        or inplace modification - currently returns modified copy)
    """
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in DataFrame")
    
    df_norm = df.copy()
    
    # Determine which columns to normalize
    if predictors == 'all':
        # Auto-detect numeric columns
        numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude common non-predictor columns
        exclude_cols = [subject_col, 'trial', 'epoch', 'probe_number', 'marker_type', 'passed']
        cols_to_norm = [c for c in numeric_cols if c not in exclude_cols]
    else:
        # Use specified columns
        cols_to_norm = [c for c in predictors if c in df.columns]
        missing = [c for c in predictors if c not in df.columns]
        if missing and verbose:
            print(f"Warning: Predictors not found in data: {missing}")
            
    if not cols_to_norm:
        if verbose:
            print("No predictors to normalize found.")
        return df_norm
        
    n_subjects = df_norm[subject_col].nunique()
    if verbose:
        print(f"Normalizing predictors ({method}) by subject...")
        print(f"  Predictors: {cols_to_norm}")
        print(f"  Subjects: {n_subjects}")
        
    # Ensure columns are float type to avoid FutureWarning when assigning normalized values
    for col in cols_to_norm:
        if not np.issubdtype(df_norm[col].dtype, np.floating):
             df_norm[col] = df_norm[col].astype(float)
        
    # Process each subject
    for subject in df_norm[subject_col].unique():
        subj_mask = df_norm[subject_col] == subject
        
        # Skip if insufficient data
        if subj_mask.sum() < 2:
            continue
            
        for col in cols_to_norm:
            vals = df_norm.loc[subj_mask, col].values.astype(float) # Ensure float
            
            # Handle NaN
            valid_mask = ~np.isnan(vals)
            if valid_mask.sum() < 2:
                continue
                
            valid_vals = vals[valid_mask]
            
            if method == 'zscore':
                mean = np.mean(valid_vals)
                std = np.std(valid_vals, ddof=1)
                if std > 1e-10:
                    vals[valid_mask] = (valid_vals - mean) / std
                else:
                    # Zero variance - center only
                    vals[valid_mask] = valid_vals - mean
            
            elif method == 'minmax':
                min_val = np.min(valid_vals)
                max_val = np.max(valid_vals)
                if max_val > min_val:
                    vals[valid_mask] = (valid_vals - min_val) / (max_val - min_val)
                else:
                    # Constant value - map to 0.5 or 0
                    vals[valid_mask] = 0.5
            
            df_norm.loc[subj_mask, col] = vals

    return df_norm


def check_normalization_assumptions(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    subject_col: str = 'subject',
    min_obs_per_subject: int = 5
) -> dict:
    """
    Check if data is suitable for subject-level normalization.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with subject identifiers
    subject_col : str
        Column name containing subject identifiers
    min_obs_per_subject : int
        Minimum number of observations required per subject
        
    Returns
    -------
    dict
        Dictionary with diagnostic information:
        - 'suitable': bool, whether normalization is recommended
        - 'n_subjects': int, number of subjects
        - 'obs_per_subject': dict, observations per subject
        - 'subjects_below_threshold': list, subjects with insufficient data
        - 'warnings': list, warning messages
    """
    if subject_col not in df_behavioral.columns:
        raise ValueError(f"Column '{subject_col}' not found in df_behavioral")
    
    # Count observations per subject
    obs_per_subject = df_behavioral[subject_col].value_counts().to_dict()
    n_subjects = len(obs_per_subject)
    
    # Find subjects below threshold
    subjects_below_threshold = [
        subj for subj, count in obs_per_subject.items() 
        if count < min_obs_per_subject
    ]
    
    # Generate warnings
    warnings_list = []
    
    if n_subjects < 2:
        warnings_list.append(
            f"Only {n_subjects} subject(s) found. Normalization requires multiple subjects."
        )
    
    if subjects_below_threshold:
        warnings_list.append(
            f"{len(subjects_below_threshold)} subject(s) have < {min_obs_per_subject} observations: "
            f"{subjects_below_threshold[:5]}{'...' if len(subjects_below_threshold) > 5 else ''}"
        )
    
    # Check for extreme imbalance
    obs_counts = list(obs_per_subject.values())
    if max(obs_counts) / min(obs_counts) > 10:
        warnings_list.append(
            f"Large imbalance in observations per subject (range: {min(obs_counts)}-{max(obs_counts)})"
        )
    
    # Determine if suitable
    suitable = (
        n_subjects >= 2 and 
        len(subjects_below_threshold) / n_subjects < 0.5  # Less than 50% below threshold
    )
    
    return {
        'suitable': suitable,
        'n_subjects': n_subjects,
        'obs_per_subject': obs_per_subject,
        'subjects_below_threshold': subjects_below_threshold,
        'warnings': warnings_list
    }


def apply_preprocessing(
    power_data: np.ndarray,
    df_behavioral: pd.DataFrame,
    config: dict,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Apply preprocessing steps based on configuration.
    
    This is a convenience function that applies normalization and other
    preprocessing steps according to the pipeline configuration.
    
    Parameters
    ----------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data
    config : dict
        Configuration dictionary with 'preprocessing' section
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    processed_data : np.ndarray
        Preprocessed power data
    preprocessing_info : dict
        Dictionary with information about applied preprocessing steps
        
    Examples
    --------
    >>> config = {
    ...     'preprocessing': {
    ...         'normalize_by_subject': True,
    ...         'normalization_method': 'zscore'
    ...     }
    ... }
    >>> processed_data, info = apply_preprocessing(power_data, df_behavioral, config)
    """
    preprocessing_info = {
        'steps_applied': [],
        'original_shape': power_data.shape,
        'original_stats': {
            'mean': float(np.nanmean(power_data)),
            'std': float(np.nanstd(power_data)),
            'min': float(np.nanmin(power_data)),
            'max': float(np.nanmax(power_data))
        }
    }
    
    # Start with copy of original data
    processed_data = power_data.copy()
    
    # Get preprocessing config (with defaults)
    preproc_config = config.get('preprocessing', {})
    
    # Apply subject-level normalization if enabled
    normalize = preproc_config.get('normalize_by_subject', False)
    if normalize:
        method = preproc_config.get('normalization_method', 'zscore')
        subject_col = preproc_config.get('subject_column', 'subject')
        channel_wise = preproc_config.get('channel_wise', True)
        
        if verbose:
            print("\nApplying subject-level normalization...")
        
        # Check assumptions first
        check_result = check_normalization_assumptions(
            processed_data, df_behavioral, subject_col
        )
        
        if not check_result['suitable']:
            warnings.warn(
                "Data may not be suitable for subject-level normalization. "
                f"Warnings: {check_result['warnings']}"
            )
        
        # Apply normalization
        processed_data = normalize_by_subject(
            processed_data,
            df_behavioral,
            method=method,
            subject_col=subject_col,
            channel_wise=channel_wise,
            verbose=verbose
        )
        
        mode_str = "channel_wise" if channel_wise else "global"
        preprocessing_info['steps_applied'].append(
            f'normalize_by_subject_{method}_{mode_str}'
        )
        preprocessing_info['normalization'] = {
            'method': method,
            'subject_col': subject_col,
            'channel_wise': channel_wise,
            'check_result': check_result
        }
    
    # Add final statistics
    preprocessing_info['final_stats'] = {
        'mean': float(np.nanmean(processed_data)),
        'std': float(np.nanstd(processed_data)),
        'min': float(np.nanmin(processed_data)),
        'max': float(np.nanmax(processed_data))
    }
    
    if verbose and preprocessing_info['steps_applied']:
        print(f"\n✓ Preprocessing completed: {', '.join(preprocessing_info['steps_applied'])}")
    elif verbose:
        print("\nNo preprocessing steps applied")
    
    return processed_data, preprocessing_info


def load_pca_data(pca_results_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load PCA results and prepare for merging with behavioral data.
    
    This function loads the PCA results CSV file containing PC1, PC2, PC3 values
    and time_on_task for each probe response.
    
    Parameters
    ----------
    pca_results_path : str
        Path to the pca_results.csv file
    verbose : bool
        Whether to print loading information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, task, probe_number, PC1, PC2, PC3, time_on_task
        
    Notes
    -----
    The PCA results file is expected to have:
    - subject_id or subject: subject identifier (converted to match 'subject' column)
    - task: task name (e.g., 'Sart1', 'Sart2')
    - probe_number: probe number within task
    - PC1, PC2, PC3: principal component scores
    - time_on_task: calculated as probe_number + (15 * (sart_number - 1))
    
    Examples
    --------
    >>> pca_df = load_pca_data("results/Behavior/probe_data/pca_results.csv")
    >>> print(pca_df.columns)
    Index(['subject', 'task', 'probe_number', 'PC1', 'PC2', 'PC3', 'time_on_task'], dtype='object')
    """
    pca_path = Path(pca_results_path)
    
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA results file not found: {pca_results_path}")
    
    if verbose:
        print(f"Loading PCA data from: {pca_results_path}")
    
    # Load PCA results
    pca_df = pd.read_csv(pca_path)
    
    # Standardize column names
    # The PCA file may have 'subject_id' instead of 'subject'
    if 'subject_id' in pca_df.columns and 'subject' not in pca_df.columns:
        pca_df = pca_df.rename(columns={'subject_id': 'subject'})
    
    # Ensure subject is string type for consistent merging
    if 'subject' in pca_df.columns:
        pca_df['subject'] = pca_df['subject'].astype(str)
    
    # Select only the columns we need for merging
    required_cols = ['subject', 'task', 'probe_number', 'PC1', 'PC2', 'PC3', 'time_on_task']
    
    # Check which columns are available
    missing_cols = [col for col in required_cols if col not in pca_df.columns]
    
    if missing_cols:
        raise ValueError(
            f"PCA results file is missing required columns: {missing_cols}\n"
            f"Available columns: {list(pca_df.columns)}"
        )
    
    # Select only required columns
    pca_df = pca_df[required_cols].copy()
    
    # Remove duplicates (keep first occurrence)
    n_before = len(pca_df)
    pca_df = pca_df.drop_duplicates(subset=['subject', 'task', 'probe_number'], keep='first')
    n_after = len(pca_df)
    
    if n_before != n_after and verbose:
        print(f"  Warning: Removed {n_before - n_after} duplicate entries")
    
    if verbose:
        print(f"✓ Loaded PCA data: {len(pca_df)} observations")
        print(f"  Subjects: {pca_df['subject'].nunique()}")
        print(f"  Tasks: {sorted(pca_df['task'].unique())}")
        print(f"  PC1 range: [{pca_df['PC1'].min():.3f}, {pca_df['PC1'].max():.3f}]")
        print(f"  PC2 range: [{pca_df['PC2'].min():.3f}, {pca_df['PC2'].max():.3f}]")
        print(f"  PC3 range: [{pca_df['PC3'].min():.3f}, {pca_df['PC3'].max():.3f}]")
        print(f"  time_on_task range: [{pca_df['time_on_task'].min():.1f}, {pca_df['time_on_task'].max():.1f}]")
    
    return pca_df


def load_qa_summary(qa_summary_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load QA summary file from preprocessing pipeline.
    
    This function loads the qa_summary.csv file generated by the preprocessing
    pipeline, which contains quality metrics and pass/fail status for each
    subject-task-epoch_type combination.
    
    Parameters
    ----------
    qa_summary_path : str
        Path to the qa_summary.csv file
    verbose : bool
        Whether to print loading information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with QA metrics including:
        - subject: subject identifier
        - task: task name (e.g., 'Sart1', 'Sart2')
        - epoch_type: 'evoked' or 'state'
        - passed: boolean indicating if file passed QA
        - fail_reasons: comma-separated list of failure reasons
        - Additional QA metrics (bad channels, ICA components, etc.)
        
    Notes
    -----
    The QA summary file is generated by steps_qa_metrics.py in the preprocessing
    pipeline. Each subject-task combination has two rows: one for evoked epochs
    and one for state epochs.
    
    Examples
    --------
    >>> qa_df = load_qa_summary("/path/to/derivatives/qa_summary.csv")
    >>> passed_files = qa_df[qa_df['passed'] == True]
    >>> print(f"Passed: {len(passed_files)} / {len(qa_df)} files")
    """
    qa_path = Path(qa_summary_path)
    
    if not qa_path.exists():
        raise FileNotFoundError(f"QA summary file not found: {qa_summary_path}")
    
    if verbose:
        print(f"Loading QA summary from: {qa_summary_path}")
    
    # Load QA summary
    qa_df = pd.read_csv(qa_path)
    
    # Standardize column names
    required_cols = ['subject', 'task', 'epoch_type', 'passed']
    missing_cols = [col for col in required_cols if col not in qa_df.columns]
    
    if missing_cols:
        raise ValueError(
            f"QA summary file is missing required columns: {missing_cols}\n"
            f"Available columns: {list(qa_df.columns)}"
        )
    
    # Ensure subject is string type for consistent merging
    qa_df['subject'] = qa_df['subject'].astype(str)
    
    # Ensure passed is boolean
    qa_df['passed'] = qa_df['passed'].astype(bool)
    
    # Remove duplicates (keep first occurrence)
    n_before = len(qa_df)
    qa_df = qa_df.drop_duplicates(subset=['subject', 'task', 'epoch_type'], keep='first')
    n_after = len(qa_df)
    
    if n_before != n_after and verbose:
        print(f"  Warning: Removed {n_before - n_after} duplicate entries")
    
    if verbose:
        print(f"✓ Loaded QA summary: {len(qa_df)} entries")
        print(f"  Subjects: {qa_df['subject'].nunique()}")
        print(f"  Tasks: {sorted(qa_df['task'].unique())}")
        print(f"  Epoch types: {sorted(qa_df['epoch_type'].unique())}")
        
        # Summary statistics
        n_total = len(qa_df)
        n_passed = qa_df['passed'].sum()
        n_failed = n_total - n_passed
        
        print(f"\n  QA Status:")
        print(f"    Passed: {n_passed} ({100 * n_passed / n_total:.1f}%)")
        print(f"    Failed: {n_failed} ({100 * n_failed / n_total:.1f}%)")
        
        # Breakdown by epoch type
        for epoch_type in sorted(qa_df['epoch_type'].unique()):
            type_df = qa_df[qa_df['epoch_type'] == epoch_type]
            n_type_total = len(type_df)
            n_type_passed = type_df['passed'].sum()
            print(f"    {epoch_type}: {n_type_passed}/{n_type_total} passed ({100 * n_type_passed / n_type_total:.1f}%)")
        
        # Show most common failure reasons if available
        if 'fail_reasons' in qa_df.columns and n_failed > 0:
            failed_df = qa_df[qa_df['passed'] == False]
            fail_reasons = failed_df['fail_reasons'].dropna()
            if len(fail_reasons) > 0:
                # Split comma-separated reasons and count
                all_reasons = []
                for reasons_str in fail_reasons:
                    if isinstance(reasons_str, str) and reasons_str:
                        all_reasons.extend([r.strip() for r in reasons_str.split(',')])
                
                if all_reasons:
                    from collections import Counter
                    reason_counts = Counter(all_reasons)
                    print(f"\n  Most common failure reasons:")
                    for reason, count in reason_counts.most_common(5):
                        print(f"    {reason}: {count}")
    
    return qa_df


def get_qa_exclusion_list(
    qa_df: pd.DataFrame,
    marker_type: str,
    verbose: bool = True
) -> Tuple[set, dict]:
    """
    Get list of subject-task combinations to exclude based on QA results.
    
    Parameters
    ----------
    qa_df : pd.DataFrame
        QA summary dataframe from load_qa_summary()
    marker_type : str
        Marker type to filter by ('evoked' or 'state')
    verbose : bool
        Whether to print exclusion information
        
    Returns
    -------
    exclusion_set : set
        Set of (subject, task) tuples to exclude
    exclusion_info : dict
        Dictionary with exclusion statistics and details
        
    Notes
    -----
    This function filters the QA summary for the specified marker type
    and returns a set of (subject, task) combinations that failed QA.
    """
    # Filter for marker type
    type_df = qa_df[qa_df['epoch_type'] == marker_type].copy()
    
    if len(type_df) == 0:
        warnings.warn(f"No QA entries found for marker type '{marker_type}'")
        return set(), {'n_total': 0, 'n_excluded': 0, 'excluded_list': []}
    
    # Get failed entries
    failed_df = type_df[type_df['passed'] == False].copy()
    
    # Create exclusion set
    exclusion_set = set()
    for _, row in failed_df.iterrows():
        exclusion_set.add((str(row['subject']), str(row['task'])))
    
    # Gather exclusion info
    exclusion_info = {
        'n_total': len(type_df),
        'n_excluded': len(exclusion_set),
        'n_passed': len(type_df) - len(exclusion_set),
        'excluded_list': sorted(list(exclusion_set)),
        'marker_type': marker_type
    }
    
    # Add failure reasons if available
    if 'fail_reasons' in failed_df.columns:
        exclusion_info['fail_reasons'] = failed_df[['subject', 'task', 'fail_reasons']].to_dict('records')
    
    if verbose and len(exclusion_set) > 0:
        print(f"\n  QA Exclusions for {marker_type} epochs:")
        print(f"    Total files: {exclusion_info['n_total']}")
        print(f"    Excluded: {exclusion_info['n_excluded']} ({100 * exclusion_info['n_excluded'] / exclusion_info['n_total']:.1f}%)")
        print(f"    Included: {exclusion_info['n_passed']} ({100 * exclusion_info['n_passed'] / exclusion_info['n_total']:.1f}%)")
        
        if len(exclusion_set) <= 10:
            print(f"    Excluded files: {exclusion_info['excluded_list']}")
        else:
            print(f"    First 10 excluded: {exclusion_info['excluded_list'][:10]}")
    
    return exclusion_set, exclusion_info


def parse_formula_components(formula: str) -> dict:
    """
    Parse formula into its components for easier manipulation.
    
    Parameters
    ----------
    formula : str
        LMM formula (e.g., "power ~ onoff + valence + (1|subject)")
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'response': response variable (e.g., 'power')
        - 'fixed_effects': list of fixed effect terms
        - 'random_effects': list of random effect specifications
        - 'has_interaction': bool, whether formula contains interactions (*)
        - 'has_random_slopes': bool, whether random effects include slopes
        
    Examples
    --------
    >>> parse_formula_components("power ~ onoff + (1|subject)")
    {'response': 'power', 'fixed_effects': ['onoff'], 'random_effects': ['(1|subject)'], 
     'has_interaction': False, 'has_random_slopes': False}
    """
    formula_clean = formula.strip()
    
    if '~' not in formula_clean:
        raise ValueError(f"Invalid formula: {formula}. Expected format: 'response ~ predictors'")
    
    parts = formula_clean.split('~')
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}. Expected single '~' separator")
    
    response = parts[0].strip()
    predictors_part = parts[1].strip()
    
    # Extract random effects (anything in parentheses)
    random_effects = re.findall(r'\([^)]*\|[^)]*\)', predictors_part)
    
    # Check for random slopes (anything other than just '1' before '|')
    has_random_slopes = any(
        not re.match(r'^\(\s*1\s*\|', re_term)
        for re_term in random_effects
    )
    
    # Remove random effects to get fixed effects
    fixed_part = re.sub(r'\([^)]*\)', '', predictors_part).strip()
    
    # Check for interactions
    has_interaction = '*' in fixed_part or ':' in fixed_part
    
    # Split fixed effects by '+'
    fixed_effects = [term.strip() for term in fixed_part.split('+') if term.strip() and term.strip() != '1']
    
    return {
        'response': response,
        'fixed_effects': fixed_effects,
        'random_effects': random_effects,
        'has_interaction': has_interaction,
        'has_random_slopes': has_random_slopes
    }


def validate_cluster_data(
    observed_t_stats: np.ndarray,
    df_behavioral: pd.DataFrame,
    predictor_of_interest: str
) -> bool:
    """
    Validate data for cluster permutation testing.
    
    Parameters
    ----------
    observed_t_stats : np.ndarray
        Observed t-statistics from LMM
    df_behavioral : pd.DataFrame
        Behavioral data from aggregated probe markers
    predictor_of_interest : str
        Name of predictor variable
        
    Returns
    -------
    bool
        True if data is valid for cluster testing
    """
    # Check if predictor exists
    # For interaction terms (e.g. "onoff:valence"), check constituent variables
    if ':' in predictor_of_interest:
        components = [v.strip() for v in predictor_of_interest.split(':')]
        for comp in components:
            if comp not in df_behavioral.columns:
                print(f"Interaction component '{comp}' (from '{predictor_of_interest}') not found in data")
                return False
            if df_behavioral[comp].nunique() < 2:
                print(f"Component '{comp}' has insufficient variation")
                return False
    else:
        if predictor_of_interest not in df_behavioral.columns:
            print(f"Predictor '{predictor_of_interest}' not found in data")
            return False
        
        # Check for sufficient variation in predictor
        unique_vals = df_behavioral[predictor_of_interest].unique()
        if len(unique_vals) < 2:
            print(f"Predictor '{predictor_of_interest}' has insufficient variation")
            return False
    
    # Check for sufficient subjects
    n_subjects = df_behavioral['subject'].nunique()
    if n_subjects < 3:
        print(f"Need at least 3 subjects for cluster testing, found {n_subjects}")
        return False
    
    # Check t-statistics
    if np.all(np.isnan(observed_t_stats)):
        print("All t-statistics are NaN")
        return False
    
    # Check for reasonable t-statistics range
    valid_t_stats = observed_t_stats[~np.isnan(observed_t_stats)]
    if len(valid_t_stats) == 0:
        print("No valid t-statistics found")
        return False
    
    t_range = np.max(np.abs(valid_t_stats))
    if t_range < 0.1:
        print(f"T-statistics are very small (max |t| = {t_range:.3f})")
        return False
    
    return True


def summarize_clusters(
    clusters: list,
    cluster_stats: np.ndarray,
    cluster_p_values: np.ndarray,
    ch_names: list,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Create summary DataFrame for significant clusters.
    
    Parameters
    ----------
    clusters : list
        List of channel index arrays
    cluster_stats : np.ndarray
        Cluster statistics
    cluster_p_values : np.ndarray
        Cluster p-values
    ch_names : list
        Channel names
    alpha : float
        Significance threshold
        
    Returns
    -------
    pd.DataFrame
        Summary with columns: cluster_id, n_channels, channels, statistic, p_value, significant
    """
    summary = []
    
    for i, (cluster, stat, pval) in enumerate(zip(clusters, cluster_stats, cluster_p_values)):
        cluster_channels = [ch_names[idx] for idx in cluster]
        summary.append({
            'cluster_id': i + 1,
            'n_channels': len(cluster),
            'channels': ', '.join(cluster_channels),
            'statistic': stat,
            'p_value': pval,
            'significant': pval < alpha
        })
    
    return pd.DataFrame(summary)
