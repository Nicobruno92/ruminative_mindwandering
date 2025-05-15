import os
import sys
import glob
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

def get_all_subjects(derivatives_folder):
    """Find all subject directories in the derivatives folder."""
    # First check if the directory exists
    if not os.path.exists(derivatives_folder):
        print(f"ERROR: Derivatives folder does not exist: {derivatives_folder}")
        return []
    
    # List subjects by listing the directory directly
    try:
        # Get all items in the directory
        all_items = os.listdir(derivatives_folder)
        
        # Filter for directories that look like subjects (start with 'sub-')
        subject_dirs = [item for item in all_items 
                        if os.path.isdir(os.path.join(derivatives_folder, item)) 
                        and item.startswith('sub-')]
        
        if not subject_dirs:
            # If no directories with 'sub-' prefix, list all directories
            subject_dirs = [item for item in all_items 
                            if os.path.isdir(os.path.join(derivatives_folder, item))]
            print(f"No 'sub-' directories found, using all directories: {subject_dirs}")
        
        return subject_dirs
    except Exception as e:
        print(f"Error listing directory contents: {str(e)}")
        return []

def get_all_tasks_for_subject(derivatives_folder, subject):
    """Find all tasks for a given subject using direct file listing."""
    subject_folder = os.path.join(derivatives_folder, subject)
    
    # Try both 'eeg' subfolder and direct subject folder
    eeg_folder = os.path.join(subject_folder, "eeg")
    if os.path.exists(eeg_folder):
        folder_to_check = eeg_folder
    else:
        folder_to_check = subject_folder
    
    # Print for debugging
    print(f"Looking for files in {folder_to_check}")
    
    # List all files in the directory
    try:
        all_files = []
        # Recursive search for all files
        for root, _, files in os.walk(folder_to_check):
            for file in files:
                if file.endswith('.fif') and 'autoPreproc' in file:
                    all_files.append(os.path.join(root, file))
        
        # Extract task names from file names
        tasks = set()
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            for part in parts:
                if part.startswith('task-'):
                    task = part.replace('task-', '')
                    tasks.add(task)
        
        return list(tasks)
    except Exception as e:
        print(f"Error finding tasks: {str(e)}")
        return []

def process_file(derivatives_folder, subject, task, output_dir):
    """Process a single subject/task with all ROI modes."""
    success = False
    
    try:
        # Extract numeric subject ID
        subject_id = subject
        if subject.startswith('sub-'):
            subject_id = subject.replace('sub-', '')
        if subject_id.startswith('S0'):
            subject_id = subject_id.replace('S0', '')
        
        # 1. Whole brain analysis
        try:
            print("Running whole brain analysis...")
            start_time = time.time()
            result_whole = process_subject_task(
                derivatives_folder, subject_id, task, output_dir, roi_mode='whole'
            )
            end_time = time.time()
            print(f"Whole brain analysis completed in {end_time - start_time:.2f} seconds")
            if result_whole is not None:
                success = True
        except Exception as e:
            print(f"Error in whole brain analysis: {str(e)}")
        
        # 2. Per-electrode analysis
        try:
            print("Running per-electrode analysis...")
            start_time = time.time()
            result_electrode = process_subject_task(
                derivatives_folder, subject_id, task, output_dir, roi_mode='all'
            )
            end_time = time.time()
            print(f"Per-electrode analysis completed in {end_time - start_time:.2f} seconds")
            if result_electrode is not None:
                success = True
        except Exception as e:
            print(f"Error in per-electrode analysis: {str(e)}")
        
        # 3. ROI-based analysis
        try:
            print("Running ROI-based analysis...")
            start_time = time.time()
            result_roi = process_subject_task(
                derivatives_folder, subject_id, task, output_dir, roi_mode=ROIS
            )
            end_time = time.time()
            print(f"ROI-based analysis completed in {end_time - start_time:.2f} seconds")
            if result_roi is not None:
                success = True
        except Exception as e:
            print(f"Error in ROI-based analysis: {str(e)}")
    
    except Exception as e:
        print(f"Error processing {subject}, task {task}: {str(e)}")
        success = False
    
    return success

def main():
    # Define paths
    # data_root = "/Volumes/cenir/analyse/meeg/CYBERSART/"
    data_root = "/network/iss/home/nicolas.bruno/depressed_mindwandering/CYBERSART/"
    derivatives_folder = os.path.join(data_root, "derivatives_nico")
    output_dir = os.path.join( "results", "nice_markers")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subjects
    subjects = get_all_subjects(derivatives_folder)
    
    if not subjects:
        print(f"No subject directories found in {derivatives_folder}")
        return
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Track successful and failed files
    successful_files = []
    failed_files = []
    
    # Process each subject
    for subject in subjects:
        # Get all tasks for this subject
        tasks = get_all_tasks_for_subject(derivatives_folder, subject)
        
        if not tasks:
            print(f"No tasks found for subject {subject}")
            continue
        
        print(f"Found {len(tasks)} tasks for {subject}: {tasks}")
        
        # Process each task
        for task in tasks:
            print(f"\nProcessing {subject}, task {task}...")
            
            success = process_file(derivatives_folder, subject, task, output_dir)
            
            # Record success/failure
            file_id = f"{subject}_task-{task}"
            if success:
                successful_files.append(file_id)
            else:
                failed_files.append(file_id)
    
    # Print summary
    print("\nProcessing summary:")
    print(f"Successfully processed {len(successful_files)} files")
    if successful_files:
        print("Successful files:")
        for file in successful_files:
            print(f"  - {file}")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files:")
        for file in failed_files:
            print(f"  - {file}")
    
    # Combine result files into summary files
    print("\nCombining results...")
    
    try:
        # Whole brain
        whole_brain_files = glob.glob(os.path.join(output_dir, "*_whole_brain.csv"))
        if whole_brain_files:
            dfs = [pd.read_csv(f) for f in whole_brain_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_path = os.path.join(output_dir, "all_subjects_whole_brain.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"Combined whole brain results saved to {combined_path}")
        
        # Per electrode
        electrode_files = glob.glob(os.path.join(output_dir, "*_per_electrode.csv"))
        if electrode_files:
            dfs = [pd.read_csv(f) for f in electrode_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_path = os.path.join(output_dir, "all_subjects_per_electrode.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"Combined per-electrode results saved to {combined_path}")
        
        # Per ROI
        roi_files = glob.glob(os.path.join(output_dir, "*_per_roi.csv"))
        if roi_files:
            dfs = [pd.read_csv(f) for f in roi_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_path = os.path.join(output_dir, "all_subjects_per_roi.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"Combined per-ROI results saved to {combined_path}")
    
    except Exception as e:
        print(f"Error combining results: {str(e)}")

if __name__ == "__main__":
    main() 