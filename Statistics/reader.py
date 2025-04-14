"""
Reader module for loading spectral power data and behavioral information.

This module handles data loading for LMM-based spatial cluster permutation testing.
Expected input format will be defined after data structure is provided by user.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Tuple


def load_data(data_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load spectral power data and behavioral information.
    
    Parameters
    ----------
    data_path : str
        Path to the data file (pickle or other format)
        
    Returns
    -------
    power_data : np.ndarray
        Power values with shape (n_observations, n_channels)
    df_behavioral : pd.DataFrame
        Behavioral data with columns including 'subject' and predictor variables
        Must contain at minimum:
        - 'subject': subject identifier for random effects
        - Additional columns as specified in LMM formula
        
    Notes
    -----
    This is a placeholder implementation. The actual data loading logic
    will be implemented after the data structure is provided.
    
    Expected data structure:
    - Per-epoch power values for each channel
    - Associated behavioral/cognitive markers per epoch
    - Subject identifiers for grouping
    """
    # Placeholder implementation
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract power_data and behavioral DataFrame
    # This will depend on the actual data structure
    if isinstance(data, dict):
        power_data = data.get('power_data')
        df_behavioral = data.get('behavioral_data')
    elif isinstance(data, tuple):
        power_data, df_behavioral = data
    else:
        raise ValueError("Unsupported data format")
    
    # Validate data
    if not isinstance(power_data, np.ndarray):
        raise ValueError("power_data must be a numpy array")
    if not isinstance(df_behavioral, pd.DataFrame):
        raise ValueError("df_behavioral must be a pandas DataFrame")
    if 'subject' not in df_behavioral.columns:
        raise ValueError("df_behavioral must contain 'subject' column")
    if len(power_data) != len(df_behavioral):
        raise ValueError("power_data and df_behavioral must have same length")
    
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
                         channel_idx: int) -> pd.DataFrame:
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
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with 'power' column and behavioral variables
    """
    df = df_behavioral.copy()
    df['power'] = power_data[:, channel_idx]
    return df
