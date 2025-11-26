#!/usr/bin/env python3
"""
Extract Mood Scale (EVA) and Need Threat Scale (NTS) data from BIDS raw events.tsv files.

This script reads the events.tsv files from BIDS raw data and extracts scale responses
that are administered at specific blocks during the experimental session:
- Mood Scale (EVA): 4 questions, administered at beginning, between tasks, and end
- Need Threat Scale (NTS): 4 questions, administered before/after Cyberball blocks

The extraction is based on trigger information:
- 'onQuestionDisplay' events with data=triggerID (11-15 for NTS, 21-25 for EVA)
- 'onAnswer' events with data=answer value (0.0-10.0)

Author: Analysis Assistant
Date: 2025-11-13
"""

# =============================================================================
# CONFIGURATION - Modify these variables as needed
# =============================================================================
BIDS_RAW_ROOT = "/Volumes/cenir/analyse/meeg/CYBERSART/BIDS/raw"
OUTPUT_DIR = "/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/results/Behavior/scales_data"
OUTPUT_FILE_NTS = OUTPUT_DIR + "/nts_aggregated_data.csv"
OUTPUT_FILE_EVA = OUTPUT_DIR + "/eva_aggregated_data.csv"
SUBJECTS = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43"]
TASKS = ["Sart1", "Sart2", "Sart3", "Sart4"]
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Trigger ID mappings based on the provided information
NTS_TRIGGERS = {
    11: 'NTSrejected',
    12: 'NTSappreciated',
    13: 'NTSmaster',
    14: 'NTSlifeNoSens',
    15: 'NTSaverage'
}

EVA_TRIGGERS = {
    21: 'EVAtense',
    22: 'EVAfeel',
    23: 'EVAmood',
    24: 'EVAhurt',
    25: 'EVAaverage'
}


def extract_scale_responses(df_events: pd.DataFrame, scale_triggers: Dict[int, str], 
                           scale_name: str) -> List[Dict[str, any]]:
    """
    Extract scale responses from events dataframe.
    
    Parameters
    ----------
    df_events : pd.DataFrame
        Events dataframe from BIDS file
    scale_triggers : Dict[int, str]
        Mapping of trigger IDs to question names
    scale_name : str
        Name of the scale ('NTS' or 'EVA')
        
    Returns
    -------
    List[Dict[str, any]]
        List of dictionaries containing scale responses organized by block
        
    Notes
    -----
    The extraction logic:
    1. Find 'Stimulus/S XX' events where XX matches scale trigger IDs (11-15 for NTS, 21-25 for EVA)
    2. Find the next Stimulus event after each question (answer is encoded as value 100-200)
    3. Decode answer: (value - 100) / 10.0 to get 0-10 scale
    4. Group responses by temporal blocks (based on timing)
    """
    scale_responses = []
    
    # Filter for question display events - they appear as "Stimulus/S XX" where XX is the trigger ID
    # The value column contains the trigger ID
    question_events = df_events[
        (df_events['trial_type'].str.contains('Stimulus/S', na=False)) & 
        (df_events['value'].isin(scale_triggers.keys()))
    ].copy()
    
    if len(question_events) == 0:
        return []
    
    # Sort by onset time
    question_events = question_events.sort_values('onset').reset_index(drop=True)
    
    # Answer events are Stimulus events with values in range 100-200 (encoded answers)
    # According to documentation: "question Answer / average : 100-200"
    answer_events = df_events[
        (df_events['trial_type'].str.contains('Stimulus/S', na=False)) & 
        (df_events['value'] >= 100) & 
        (df_events['value'] <= 200)
    ].copy()
    answer_events = answer_events.sort_values('onset').reset_index(drop=True)
    
    # Group questions into blocks based on temporal proximity
    # Questions within 60 seconds are considered part of the same block
    blocks = []
    current_block = []
    
    for idx, q_row in question_events.iterrows():
        if len(current_block) == 0:
            current_block.append(q_row)
        else:
            # Check time difference with last question in current block
            time_diff = q_row['onset'] - current_block[-1]['onset']
            if time_diff < 60:  # Within 60 seconds = same block
                current_block.append(q_row)
            else:
                # Start new block
                blocks.append(current_block)
                current_block = [q_row]
    
    # Add last block
    if len(current_block) > 0:
        blocks.append(current_block)
    
    # Process each block
    for block_num, block_questions in enumerate(blocks, start=1):
        block_data = {
            'block_number': block_num,
            'block_onset': block_questions[0]['onset']
        }
        
        # Extract answers for each question in the block
        for q_row in block_questions:
            trigger_id = int(q_row['value'])
            question_name = scale_triggers.get(trigger_id, f'unknown_{trigger_id}')
            question_onset = q_row['onset']
            
            # Find the next answer event after this question
            next_answers = answer_events[answer_events['onset'] > question_onset]
            
            if len(next_answers) > 0:
                answer_row = next_answers.iloc[0]
                # Decode the answer: values are encoded as 100-200
                # According to documentation: "question Answer / average : 100-200"
                # The actual scale is 0-10, so we use (value - 100) / 10.0
                encoded_value = answer_row['value']
                answer_value = (encoded_value - 100) / 10.0
                block_data[question_name] = answer_value
            else:
                block_data[question_name] = np.nan
        
        # Calculate total score (sum of individual questions, excluding average)
        question_cols = [q for q in scale_triggers.values() if 'average' not in q.lower()]
        valid_scores = [block_data.get(q) for q in question_cols if pd.notna(block_data.get(q))]
        
        if len(valid_scores) > 0:
            block_data['total_score'] = sum(valid_scores)
        else:
            block_data['total_score'] = np.nan
        
        scale_responses.append(block_data)
    
    return scale_responses


def extract_scales_from_events_file(events_file: Path, subject_id: str, task: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract both NTS and EVA scale data from a single events.tsv file.
    
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
    Tuple[List[Dict], List[Dict]]
        Tuple of (NTS responses, EVA responses)
    """
    if not events_file.exists():
        print(f"Warning: Events file not found: {events_file}")
        return [], []
    
    try:
        # Read the events file
        df_events = pd.read_csv(events_file, sep='\t')
        
        # Extract NTS responses
        nts_responses = extract_scale_responses(df_events, NTS_TRIGGERS, 'NTS')
        for response in nts_responses:
            response['subject_id'] = int(subject_id)
            response['task'] = task
        
        # Extract EVA responses
        eva_responses = extract_scale_responses(df_events, EVA_TRIGGERS, 'EVA')
        for response in eva_responses:
            response['subject_id'] = int(subject_id)
            response['task'] = task
        
        print(f"Extracted {len(nts_responses)} NTS blocks and {len(eva_responses)} EVA blocks from {subject_id}/{task}")
        return nts_responses, eva_responses
        
    except Exception as e:
        print(f"Error reading events file {events_file}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def extract_all_scales() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract scale information from all subjects and tasks.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (NTS DataFrame, EVA DataFrame)
    """
    all_nts = []
    all_eva = []
    
    for subject_id in SUBJECTS:
        subject_dir = Path(BIDS_RAW_ROOT) / f"sub-{subject_id}" / "eeg"
        
        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue
        
        for task in TASKS:
            events_file = subject_dir / f"sub-{subject_id}_task-{task}_events.tsv"
            
            nts_data, eva_data = extract_scales_from_events_file(events_file, subject_id, task)
            all_nts.extend(nts_data)
            all_eva.extend(eva_data)
    
    # Convert to DataFrames
    df_nts = pd.DataFrame(all_nts) if all_nts else pd.DataFrame()
    df_eva = pd.DataFrame(all_eva) if all_eva else pd.DataFrame()
    
    # Sort by subject, task, and block number for consistency
    if not df_nts.empty:
        df_nts = df_nts.sort_values(['subject_id', 'task', 'block_number']).reset_index(drop=True)
    
    if not df_eva.empty:
        df_eva = df_eva.sort_values(['subject_id', 'task', 'block_number']).reset_index(drop=True)
    
    return df_nts, df_eva


def add_metadata_to_scales(df_scales: pd.DataFrame, scale_name: str) -> pd.DataFrame:
    """
    Add metadata (demographics, group, psychometrics) to scale data.
    
    Parameters
    ----------
    df_scales : pd.DataFrame
        Scale data
    scale_name : str
        Name of the scale ('NTS' or 'EVA')
        
    Returns
    -------
    pd.DataFrame
        Scale data with metadata merged
    """
    meta_path = Path("/Volumes/levy/analyze/valerocabre/analyse/nbruno/depressed_mindwandering/metadata_CyberSART.xlsx")
    
    if not meta_path.exists():
        print(f"Warning: Metadata file not found: {meta_path}")
        return df_scales
    
    try:
        df_meta = pd.read_excel(meta_path, sheet_name="Metadata")
        df_meta = df_meta.copy()
        
        # Create subject_id from 'subject' if not present
        if "subject_id" not in df_meta.columns:
            if "subject" in df_meta.columns:
                df_meta["subject_id"] = df_meta["subject"]
            else:
                df_meta["subject_id"] = np.nan
        
        # Ensure numeric subject_id
        df_meta["subject_id"] = pd.to_numeric(df_meta["subject_id"], errors="coerce").astype("Int64")
        
        # Create a unified 'sex' column if missing
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
        
        # Normalize inclusion_exclusion order codes
        if "inclusion_exclusion" in df_meta.columns and "order (IE/EI)" not in df_meta.columns:
            unique_ie_vals = (
                df_meta["inclusion_exclusion"].dropna().astype(str).str.upper().unique().tolist()
            )
            if set(unique_ie_vals).issubset({"IE", "EI"}):
                df_meta["order (IE/EI)"] = df_meta["inclusion_exclusion"].astype(str).str.upper()
        
        # Drop duplicate rows per subject_id
        df_meta = df_meta.drop_duplicates("subject_id")
        
        # Merge
        df_scales = pd.merge(df_scales, df_meta, on="subject_id", how="left")
        
        # Map group labels if numeric 1/2
        if "group" in df_scales.columns:
            try:
                df_scales.loc[df_scales["group"] == 1, "group"] = "Controls"
                df_scales.loc[df_scales["group"] == 2, "group"] = "Risk of Depression"
            except Exception:
                pass
        
        # Add inclusion_exclusion based on order and task
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
        
        df_scales["inclusion_exclusion"] = df_scales.apply(map_inclusion_exclusion, axis=1)
        
    except Exception as e:
        print(f"Error adding metadata: {e}")
        import traceback
        traceback.print_exc()
    
    return df_scales


def validate_scale_data(df_scales: pd.DataFrame, scale_name: str) -> None:
    """
    Validate the extracted scale data and print summary statistics.
    
    Parameters
    ----------
    df_scales : pd.DataFrame
        The extracted scale data
    scale_name : str
        Name of the scale ('NTS' or 'EVA')
    """
    print(f"\n=== {scale_name} VALIDATION SUMMARY ===")
    print(f"Total {scale_name} blocks extracted: {len(df_scales)}")
    print(f"Unique subjects: {df_scales['subject_id'].nunique()}")
    print(f"Unique tasks: {df_scales['task'].nunique()}")
    
    # Check for missing values in scale questions
    scale_cols = [col for col in df_scales.columns if scale_name in col]
    missing_counts = df_scales[scale_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\nMissing values in {scale_name} questions:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  {col}: {count}")
    else:
        print(f"\nNo missing values in {scale_name} questions.")
    
    # Summary statistics for scale questions
    print(f"\n=== {scale_name} QUESTIONS SUMMARY ===")
    for col in scale_cols:
        if col in df_scales.columns and 'average' not in col.lower():
            values = df_scales[col].dropna()
            if len(values) > 0:
                print(f"{col}: mean={values.mean():.2f}, std={values.std():.2f}, range=[{values.min():.1f}, {values.max():.1f}]")
    
    # Total score statistics
    if 'total_score' in df_scales.columns:
        total_scores = df_scales['total_score'].dropna()
        if len(total_scores) > 0:
            print(f"\nTotal Score: mean={total_scores.mean():.2f}, std={total_scores.std():.2f}, range=[{total_scores.min():.1f}, {total_scores.max():.1f}]")
    
    # Blocks per subject/task
    blocks_per_subject_task = df_scales.groupby(['subject_id', 'task']).size()
    print(f"\nBlocks per subject/task: mean={blocks_per_subject_task.mean():.2f}, std={blocks_per_subject_task.std():.2f}")
    print(f"Range: {blocks_per_subject_task.min()} - {blocks_per_subject_task.max()}")


def main():
    """
    Main execution function.
    
    Coordinates the entire scale extraction process from BIDS raw data.
    """
    print("=== MOOD SCALE (EVA) AND NEED THREAT SCALE (NTS) EXTRACTION ===")
    print(f"BIDS Raw Root: {BIDS_RAW_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Processing {len(SUBJECTS)} subjects and {len(TASKS)} tasks")
    
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all scale data
    print(f"\nExtracting scale data...")
    df_nts, df_eva = extract_all_scales()
    
    if df_nts.empty and df_eva.empty:
        print("Error: No scale data was extracted. Please check the input paths and data.")
        return
    
    # Add metadata to both scales
    if not df_nts.empty:
        print(f"\nAdding metadata to NTS data...")
        df_nts = add_metadata_to_scales(df_nts, 'NTS')
        validate_scale_data(df_nts, 'NTS')
        df_nts.to_csv(OUTPUT_FILE_NTS, index=False)
        print(f"\nNTS data saved to: {OUTPUT_FILE_NTS}")
        print(f"\n=== FIRST 10 NTS ROWS ===")
        print(df_nts.head(10).to_string(index=False))
    else:
        print("\nWarning: No NTS data extracted.")
    
    if not df_eva.empty:
        print(f"\nAdding metadata to EVA data...")
        df_eva = add_metadata_to_scales(df_eva, 'EVA')
        validate_scale_data(df_eva, 'EVA')
        df_eva.to_csv(OUTPUT_FILE_EVA, index=False)
        print(f"\nEVA data saved to: {OUTPUT_FILE_EVA}")
        print(f"\n=== FIRST 10 EVA ROWS ===")
        print(df_eva.head(10).to_string(index=False))
    else:
        print("\nWarning: No EVA data extracted.")
    
    print(f"\n=== EXTRACTION COMPLETE ===")


if __name__ == "__main__":
    main()
