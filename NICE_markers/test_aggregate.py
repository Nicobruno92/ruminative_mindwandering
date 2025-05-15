#!/usr/bin/env python
"""
Validation test for the NICE markers aggregation code.
This script tests the basic functionality of the aggregation code
with a small sample dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the module we want to test
from NICE_markers.aggregate_markers import run_aggregation

def create_test_data(output_dir, file_type, num_subjects=2, num_tasks=2):
    """
    Create test data for testing the aggregation functionality.
    
    Parameters:
    -----------
    output_dir : str
        Directory to write test files to
    file_type : str
        Type of file to create ('per_electrode', 'per_roi', 'whole_brain')
    num_subjects : int
        Number of subjects to create data for
    num_tasks : int
        Number of tasks per subject
    
    Returns:
    --------
    list
        List of created file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    file_paths = []
    
    for subject in range(1, num_subjects + 1):
        for task_num in range(1, num_tasks + 1):
            task = f"Sart{task_num}"
            
            # Create a basic DataFrame
            basic_data = {
                'subject_id': subject,
                'task': task,
                'marker': 'a',  # Single marker for simplicity
                'value': np.random.normal(0, 1, 36),  # Random values
                'event_name': [],
                'distance_to_probe': [],
                'probe_number': []
            }
            
            # Add electrode/roi if needed
            if file_type in ['per_electrode', 'per_roi']:
                column_name = 'channel' if file_type == 'per_electrode' else 'roi'
                basic_data[column_name] = []
            
            # Create event names for different conditions
            rows = []
            for onoff in ['on', 'off']:
                for valence in ['pos', 'neg']:
                    for selfother in ['self', 'other']:
                        for distance in [-1, -2, -3]:
                            for probe in [1, 2]:
                                event_name = f"go/correct/{onoff}50/{valence}50/{selfother}50/time50/confidence50/{distance}/{probe}"
                                
                                row = {
                                    'subject_id': subject,
                                    'task': task,
                                    'marker': 'a',
                                    'value': np.random.normal(0, 1),
                                    'event_name': event_name,
                                    'distance_to_probe': distance,
                                    'probe_number': probe
                                }
                                
                                # Add electrode/roi if needed
                                if file_type == 'per_electrode':
                                    for channel in ['Fz', 'Cz', 'Pz']:
                                        row_copy = row.copy()
                                        row_copy['channel'] = channel
                                        rows.append(row_copy)
                                elif file_type == 'per_roi':
                                    for roi in ['frontal', 'parietal', 'temporal']:
                                        row_copy = row.copy()
                                        row_copy['roi'] = roi
                                        rows.append(row_copy)
                                else:
                                    rows.append(row)
            
            # Create DataFrame from rows
            df = pd.DataFrame(rows)
            
            # Create file path
            filename = f"sub-{subject}_task-{task}_{file_type}.csv"
            file_path = os.path.join(output_dir, filename)
            
            # Save the DataFrame
            df.to_csv(file_path, index=False)
            file_paths.append(file_path)
            
            print(f"Created test file: {file_path} with {len(df)} rows")
    
    print(f"Created {len(file_paths)} test files in {output_dir}")
    return file_paths

def test_aggregation():
    """Run tests for all file types with aggregation"""
    
    file_types = ["whole_brain", "per_roi", "per_electrode"]
    conditions = ["onoff", "valence", "selfother", "time", "confidence"]
    
    for file_type in file_types:
        print(f"Testing aggregation for {file_type} files...")
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, "nice_markers")
        output_dir = os.path.join(temp_dir, "aggregated_markers")
        
        try:
            # Create test data
            create_test_data(input_dir, file_type)
            
            # Run aggregation
            output_path = run_aggregation(
                input_dir=input_dir,
                output_dir=output_dir,
                file_type=file_type,
                conditions=conditions,
                aggregate_level="task",
                trials_before_probe=5,
                only_go_correct=False,
                max_files_per_batch=2
            )
            
            # Check result
            if output_path and os.path.exists(output_path):
                print(f"Success! Output file created: {output_path}")
                
                # Validate output file
                df = pd.read_csv(output_path)
                print(f"Output file has {len(df)} rows and the following columns:")
                print(", ".join(df.columns))
                
                # Check if all condition columns are present
                missing_conditions = [c for c in conditions if c not in df.columns]
                if missing_conditions:
                    print(f"Warning: Missing condition columns: {missing_conditions}")
                    return 1
                
                # Check if any data is present
                if len(df) == 0:
                    print("Warning: Output file has no rows")
                    return 1
            else:
                print(f"Error: Task-level aggregation failed to produce output file")
                return 1
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                print(f"Warning: Could not clean up temporary directory {temp_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_aggregation()) 