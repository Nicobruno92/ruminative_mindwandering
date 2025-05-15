#!/usr/bin/env python
"""
Simple script to test if we can read the CSV files and process them.
This is a stripped-down version for debugging.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
import re
import traceback

def main():
    # Define data and output directories
    data_dir = "results/nice_markers"
    output_dir = "results/simple_topomaps"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    try:
        csv_files = glob.glob(os.path.join(data_dir, "sub-*_task-Sart*_per_electrode.csv"))
        print(f"Found {len(csv_files)} CSV files")
        
        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            return
        
        # Only process the first 2 files for testing
        csv_files = csv_files[:2]
        print(f"Processing only the first 2 files:")
        for f in csv_files:
            print(f"  - {os.path.basename(f)}")
        
        # Process each file
        for file_path in csv_files:
            try:
                print(f"\nProcessing file: {os.path.basename(file_path)}")
                
                # Read only a small chunk of the file
                chunk = pd.read_csv(file_path, nrows=10)
                print(f"Loaded chunk with {len(chunk)} rows")
                
                # Display column names
                print(f"Columns: {chunk.columns.tolist()}")
                
                # Show first rows
                print("\nFirst rows:")
                print(chunk.head(2))
                
                # Extract markers and channels
                markers = chunk['marker'].unique()
                channels = chunk['channel'].unique()
                
                print(f"\nFound {len(markers)} markers: {markers[:5]}...")
                print(f"Found {len(channels)} channels: {channels[:5]}...")
                
                # Extract condition values from event_name
                condition = 'onoff'
                print(f"\nExtracting {condition} values from event names...")
                
                for i, row in chunk.iterrows():
                    event_name = row['event_name']
                    match = re.search(rf"{condition}(\d+)", event_name)
                    if match:
                        value = int(match.group(1))
                        binary = 1 if value >= 50 else 0
                        print(f"  Row {i}: {condition}={value}, binary={binary}")
                
            except Exception as e:
                print(f"Error processing file {file_path}:")
                traceback.print_exc()
        
        print("\nSimple test completed!")
        
    except Exception as e:
        print(f"Unhandled exception:")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error:")
        traceback.print_exc() 