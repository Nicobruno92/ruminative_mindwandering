import os
import sys
import pandas as pd
import time

# Add the project root directory to Python path to find the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NICE_markers.compute_markers_rois import process_subject_task

# Define ROIs
ROIS = {
    'prefrontal': ['Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8'],
    'frontal': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz'],
    'central': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'Cz', 'CPz'],
    'temporal': ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'],
    'parietal': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'Pz'],
    'occipital': ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz']
}

def main():
    # Define paths
    data_root = "/network/iss/cenir/analyse/meeg//CYBERSART/"
    derivatives_folder = os.path.join(data_root, "derivatives_nico")
    output_dir = os.path.join("results", "test_markers")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose a subject and task to test
    subject = "43"  # Use an existing subject
    task = "Sart3"  # Use an existing task
    
    print(f"Testing with subject {subject}, task {task}")
    
    # Process with all three ROI modes
    success = False
    
    # 1. Whole brain analysis
    try:
        print(f"Running whole brain analysis...")
        start_time = time.time()
        result_whole = process_subject_task(
            derivatives_folder, subject, task, output_dir, roi_mode='whole'
        )
        end_time = time.time()
        print(f"Whole brain analysis completed in {end_time - start_time:.2f} seconds")
        if result_whole is not None:
            success = True
            print(f"Whole brain result shape: {result_whole.shape}")
    except Exception as e:
        print(f"Error in whole brain analysis: {str(e)}")
    
    # 2. Per-electrode analysis
    try:
        print(f"Running per-electrode analysis...")
        start_time = time.time()
        result_electrode = process_subject_task(
            derivatives_folder, subject, task, output_dir, roi_mode='all'
        )
        end_time = time.time()
        print(f"Per-electrode analysis completed in {end_time - start_time:.2f} seconds")
        if result_electrode is not None:
            success = True
            print(f"Per-electrode result shape: {result_electrode.shape}")
    except Exception as e:
        print(f"Error in per-electrode analysis: {str(e)}")
    
    # 3. ROI-based analysis
    try:
        print(f"Running ROI-based analysis...")
        start_time = time.time()
        result_roi = process_subject_task(
            derivatives_folder, subject, task, output_dir, roi_mode=ROIS
        )
        end_time = time.time()
        print(f"ROI-based analysis completed in {end_time - start_time:.2f} seconds")
        if result_roi is not None:
            success = True
            print(f"ROI-based result shape: {result_roi.shape}")
    except Exception as e:
        print(f"Error in ROI-based analysis: {str(e)}")
    
    # Check the output directory
    print("\nChecking output files:")
    output_files = os.listdir(output_dir)
    for file in output_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  - {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main() 