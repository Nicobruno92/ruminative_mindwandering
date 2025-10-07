#!/usr/bin/env python3
"""
Extract probe information from BIDS raw events.tsv files.

This script reads the events.tsv files from BIDS raw data and extracts unique probe instances,
avoiding the repeated entries that occur due to probe information propagation during trials.
Each probe instance is extracted only once with all its associated phenomenological ratings.

Author: Analysis Assistant
Date: 2025-09-24
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
BIDS_RAW_ROOT = "/network/iss/cenir/analyse/meeg/CYBERSART/BIDS/raw"
# Write to the canonical path used by downstream analyses
OUTPUT_FILE = "/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/probe_data/probe_level_aggregated_data.csv"
SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def extract_probe_info_from_trial_type(trial_type: str) -> Optional[Dict[str, any]]:
    """
    Extract probe information from a trial_type string.
    
    Parameters
    ----------
    trial_type : str
        The trial_type string containing probe information
        
    Returns
    -------
    Optional[Dict[str, any]]
        Dictionary with extracted probe information, or None if no probe info found
        
    Notes
    -----
    Expected format: "go/correct/onoff96/selfother8/valence94/time49/confidence97/average62/..."
    Contains probe ratings and probe number information.
    """
    if not isinstance(trial_type, str) or 'probe' not in trial_type:
        return None
        
    try:
        # Extract numeric values for different conditions using regex
        probe_info = {}
        
        # Extract onoff value - look for patterns like "onoff96"
        onoff_match = re.search(r'onoff(\d+)', trial_type)
        probe_info['onoff'] = float(onoff_match.group(1)) if onoff_match else None
        
        # Extract selfother value - look for patterns like "selfother8"
        selfother_match = re.search(r'selfother(\d+)', trial_type)
        probe_info['selfother'] = float(selfother_match.group(1)) if selfother_match else None
        
        # Extract valence value - look for patterns like "valence94"
        valence_match = re.search(r'valence(\d+)', trial_type)
        probe_info['valence'] = float(valence_match.group(1)) if valence_match else None
        
        # Extract time value - look for patterns like "time49"
        time_match = re.search(r'time(\d+)', trial_type)
        probe_info['time'] = float(time_match.group(1)) if time_match else None
        
        # Extract confidence value - look for patterns like "confidence97"
        confidence_match = re.search(r'confidence(\d+)', trial_type)
        probe_info['confidence'] = float(confidence_match.group(1)) if confidence_match else None
        
        # Extract average value - look for patterns like "average62"
        average_match = re.search(r'average(\d+)', trial_type)
        probe_info['average'] = float(average_match.group(1)) if average_match else None
        
        # Extract probe number - look for patterns like "probe15"
        probe_match = re.search(r'probe(\d+)', trial_type)
        probe_info['probe_number'] = float(probe_match.group(1)) if probe_match else None
        
        # Only return if we found probe information
        if probe_info['probe_number'] is not None:
            return probe_info
        else:
            return None
            
    except Exception as e:
        print(f"Warning: Error parsing trial_type '{trial_type}': {e}")
        return None


def extract_probes_from_events_file(events_file: Path, subject_id: str, task: str) -> List[Dict[str, any]]:
    """
    Extract unique probe instances from a single events.tsv file.
    
    Parameters
    ----------
    events_file : Path
        Path to the events.tsv file
    subject_id : str
        Subject identifier
    task : str
        Task identifier
        
    Returns
    -------
    List[Dict[str, any]]
        List of dictionaries containing probe information
        
    Notes
    -----
    Each probe appears multiple times in the events file with different distance values.
    We extract only one instance per probe with all the phenomenological ratings.
    """
    if not events_file.exists():
        print(f"Warning: Events file not found: {events_file}")
        return []
        
    try:
        # Read the events file
        df_events = pd.read_csv(events_file, sep='\t')
        
        # Filter for entries that contain probe information
        probe_rows = df_events[df_events['trial_type'].str.contains('probe', na=False)]
        
        if len(probe_rows) == 0:
            print(f"Warning: No probe events found in {events_file}")
            return []
            
        # Extract probe information from each row
        probes_data = []
        seen_probes = set()  # To track unique probes
        
        for _, row in probe_rows.iterrows():
            probe_info = extract_probe_info_from_trial_type(row['trial_type'])
            
            if probe_info is not None:
                probe_number = probe_info['probe_number']
                
                # Only add if we haven't seen this probe number yet for this subject/task
                if probe_number not in seen_probes:
                    probe_data = {
                        'subject_id': int(subject_id),
                        'task': task,
                        'probe_number': probe_number,
                        'onoff': probe_info['onoff'],
                        'selfother': probe_info['selfother'],
                        'valence': probe_info['valence'],
                        'time': probe_info['time'],
                        'confidence': probe_info['confidence'],
                        'average': probe_info['average']
                    }
                    probes_data.append(probe_data)
                    seen_probes.add(probe_number)
        
        print(f"Extracted {len(probes_data)} unique probes from {subject_id}/{task}")
        return probes_data
        
    except Exception as e:
        print(f"Error reading events file {events_file}: {e}")
        return []


def extract_all_probes() -> pd.DataFrame:
    """
    Extract probe information from all subjects and tasks.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all extracted probe information
        
    Notes
    -----
    Iterates through all subjects and tasks, reading their events.tsv files
    and extracting unique probe instances.
    """
    all_probes = []
    
    for subject_id in SUBJECTS:
        subject_dir = Path(BIDS_RAW_ROOT) / f"sub-{subject_id}" / "eeg"
        
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
            
        for task in TASKS:
            events_file = subject_dir / f"sub-{subject_id}_task-{task}_events.tsv"
            
            probes_data = extract_probes_from_events_file(events_file, subject_id, task)
            all_probes.extend(probes_data)
    
    if not all_probes:
        print("Error: No probe data extracted from any files!")
        return pd.DataFrame()
        
    # Convert to DataFrame
    df_probes = pd.DataFrame(all_probes)
    
    # Sort by subject, task, and probe number for consistency
    df_probes = df_probes.sort_values(['subject_id', 'task', 'probe_number']).reset_index(drop=True)
    
    return df_probes


def validate_extracted_data(df_probes: pd.DataFrame) -> None:
    """
    Validate the extracted probe data and print summary statistics.
    
    Parameters
    ----------
    df_probes : pd.DataFrame
        The extracted probe data
        
    Notes
    -----
    Checks data integrity and prints useful summary information.
    """
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total probes extracted: {len(df_probes)}")
    print(f"Unique subjects: {df_probes['subject_id'].nunique()}")
    print(f"Unique tasks: {df_probes['task'].nunique()}")
    print(f"Probe number range: {df_probes['probe_number'].min():.0f} - {df_probes['probe_number'].max():.0f}")
    
    # Check for missing values
    missing_counts = df_probes.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nMissing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count}")
    else:
        print(f"\nNo missing values found.")
    
    # Summary statistics for probe ratings
    print(f"\n=== PROBE RATINGS SUMMARY ===")
    rating_cols = ['onoff', 'selfother', 'valence', 'time', 'confidence', 'average']
    for col in rating_cols:
        if col in df_probes.columns:
            print(f"{col}: mean={df_probes[col].mean():.1f}, std={df_probes[col].std():.1f}, range=[{df_probes[col].min():.0f}, {df_probes[col].max():.0f}]")
    
    # Probes per subject/task
    probes_per_subject_task = df_probes.groupby(['subject_id', 'task']).size()
    print(f"\nProbes per subject/task: mean={probes_per_subject_task.mean():.1f}, std={probes_per_subject_task.std():.1f}")
    print(f"Range: {probes_per_subject_task.min()} - {probes_per_subject_task.max()}")


def main():
    """
    Main execution function.
    
    Notes
    -----
    Coordinates the entire probe extraction process from BIDS raw data.
    """
    print("=== PROBE EXTRACTION FROM BIDS RAW DATA ===")
    print(f"BIDS Raw Root: {BIDS_RAW_ROOT}")
    print(f"Output File: {OUTPUT_FILE}")
    print(f"Processing {len(SUBJECTS)} subjects and {len(TASKS)} tasks")
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all probe data
    print(f"\nExtracting probe data...")
    df_probes = extract_all_probes()
    
    if df_probes.empty:
        print("Error: No probe data was extracted. Please check the input paths and data.")
        return
    
    # Merge metadata (sex/gender, age, group, inclusion_exclusion, psychometrics) and include all columns
    meta_path = Path("/network/iss/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/metadata_CyberSART.xlsx")
    if meta_path.exists():
        df_meta = pd.read_excel(meta_path, sheet_name="Metadata")
        # Harmonize identifiers and key fields
        df_meta = df_meta.copy()
        # Create subject_id from 'subject' if not present
        if "subject_id" not in df_meta.columns:
            if "subject" in df_meta.columns:
                df_meta["subject_id"] = df_meta["subject"]
            else:
                df_meta["subject_id"] = np.nan
        # Ensure numeric subject_id
        df_meta["subject_id"] = pd.to_numeric(df_meta["subject_id"], errors="coerce").astype("Int64")

        # Create a unified 'sex' column if missing, preferring 'gender'; fallback to one-hot sexe columns
        if "sex" not in df_meta.columns:
            if "gender" in df_meta.columns:
                df_meta["sex"] = df_meta["gender"]
            elif {"sexe___M", "sexe___F"}.issubset(set(df_meta.columns)):
                def _derive_sex(row: pd.Series) -> Optional[str]:
                    m = row.get("sexe___M")
                    f = row.get("sexe___F")
                    if pd.notna(m) and pd.notna(f):
                        if m == 1 and f == 0:
                            return "M"
                        if m == 0 and f == 1:
                            return "F"
                    return None
                df_meta["sex"] = df_meta.apply(_derive_sex, axis=1)

        # If metadata provides 'inclusion_exclusion' as order codes (IE/EI), normalize into 'order (IE/EI)'
        if "inclusion_exclusion" in df_meta.columns and "order (IE/EI)" not in df_meta.columns:
            unique_ie_vals = (
                df_meta["inclusion_exclusion"].dropna().astype(str).str.upper().unique().tolist()
            )
            if set(unique_ie_vals).issubset({"IE", "EI"}):
                df_meta["order (IE/EI)"] = df_meta["inclusion_exclusion"].astype(str).str.upper()

        # Drop duplicate rows per subject_id, keep first occurrence
        df_meta = df_meta.drop_duplicates("subject_id")

        # Merge
        df_probes = pd.merge(df_probes, df_meta, on="subject_id", how="left")

        # Map group labels if numeric 1/2
        if "group" in df_probes.columns:
            try:
                df_probes.loc[df_probes["group"] == 1, "group"] = "Controls"
                df_probes.loc[df_probes["group"] == 2, "group"] = "Risk of Depression"
            except Exception:
                pass

        # Add inclusion_exclusion based on order and task (as in the previous pipeline)
        def map_inclusion_exclusion(row: pd.Series) -> Optional[str]:
            task = str(row.get("task", ""))
            order = str(row.get("order (IE/EI)", ""))
            if task in ["Sart1", "Sart3"]:
                return "baseline"
            if task == "Sart2":
                if order == "IE":
                    return "inclusion"
                if order == "EI":
                    return "exclusion"
            if task == "Sart4":
                if order == "IE":
                    return "exclusion"
                if order == "EI":
                    return "inclusion"
            return None

        df_probes["inclusion_exclusion"] = df_probes.apply(map_inclusion_exclusion, axis=1)

    # Validate the data
    validate_extracted_data(df_probes)
    
    # Save to CSV
    df_probes.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProbe data saved to: {OUTPUT_FILE}")
    
    # Show first few rows as example
    print(f"\n=== FIRST 10 ROWS ===")
    print(df_probes.head(10).to_string(index=False))
    
    print(f"\n=== EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
