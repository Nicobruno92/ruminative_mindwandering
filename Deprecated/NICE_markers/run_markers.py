import os
import sys
import glob
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Add the project root directory to Python path to find the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from NICE_markers.compute_markers_rois import process_subject_task, ROIS

def process_file(derivatives_folder, subject, task, output_dir, modes):
    """Process a single subject/task with selected ROI modes."""
    success = False
    try:
        # Extract numeric subject ID if needed
        subject_id = subject
        if subject.startswith('sub-'):
            subject_id = subject.replace('sub-', '')
        if subject_id.startswith('S0'):
            subject_id = subject_id.replace('S0', '')
        
        print(f"Processing {subject}, task {task}")
        
        # Run selected modes
        if 'whole' in modes:
            try:
                print(f"Running whole brain analysis for {subject}, task {task}...")
                start_time = time.time()
                result_whole = process_subject_task(
                    derivatives_folder, subject_id, task, output_dir, roi_mode='whole'
                )
                end_time = time.time()
                print(f"Whole brain analysis for {subject}, task {task} completed in {end_time - start_time:.2f} seconds")
                if result_whole is not None:
                    success = True
            except Exception as e:
                print(f"Error in whole brain analysis for {subject}, task {task}: {str(e)}")

        if 'all' in modes or 'electrodes' in modes:
            try:
                print(f"Running per-electrode analysis for {subject}, task {task}...")
                start_time = time.time()
                result_electrode = process_subject_task(
                    derivatives_folder, subject_id, task, output_dir, roi_mode='all'
                )
                end_time = time.time()
                print(f"Per-electrode analysis for {subject}, task {task} completed in {end_time - start_time:.2f} seconds")
                if result_electrode is not None:
                    # Check number of unique channels
                    if hasattr(result_electrode, 'channel') or ('channel' in getattr(result_electrode, 'columns', [])):
                        n_unique_channels = len(set(result_electrode['channel']))
                        if n_unique_channels not in (64, 65):
                            print(f"[WARNING] Per-electrode output has {n_unique_channels} unique channels (expected 64 or 65, EEG only).")
                    success = True
            except Exception as e:
                print(f"Error in per-electrode analysis for {subject}, task {task}: {str(e)}")

        if 'roi' in modes:
            try:
                print(f"Running ROI-based analysis for {subject}, task {task}...")
                start_time = time.time()
                result_roi = process_subject_task(
                    derivatives_folder, subject_id, task, output_dir, roi_mode=ROIS
                )
                end_time = time.time()
                print(f"ROI-based analysis for {subject}, task {task} completed in {end_time - start_time:.2f} seconds")
                if result_roi is not None:
                    success = True
            except Exception as e:
                print(f"Error in ROI-based analysis for {subject}, task {task}: {str(e)}")
    
    except Exception as e:
        print(f"Error processing {subject}, task {task}: {str(e)}")
        success = False
    return subject, task, success

def main():
    parser = argparse.ArgumentParser(description="Run NICE marker analysis for subjects/tasks.")
    parser.add_argument('--modes', type=str, default='whole,all,roi',
                        help="Comma-separated list of analysis modes to run: whole,all,roi (default: all modes)")
    parser.add_argument('--max-workers', type=int, default=8,
                        help="Maximum number of parallel workers (default: 8)")
    parser.add_argument('--subject', type=str, required=True,
                        help="Subject ID to process (e.g., sub-01 or just 01)")
    parser.add_argument('--task', type=str, default=None,
                        help="Specific task to process (e.g., Sart1). If not set, process all tasks for the subject.")
    
    args = parser.parse_args()
    
    # Set up paths
    derivatives_folder = "/network/iss/cenir/analyse/meeg/CYBERSART/derivatives_nico"
    output_dir = os.path.join("results", "nice_markers")
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse modes
    modes = [m.strip().lower() for m in args.modes.split(',') if m.strip()]
    print(f"Selected analysis modes: {modes}")
    
    # Process the subject (with optional task filtering)
    subject = args.subject
    print(f"Processing subject: {subject}")
    
    # If task is specified, process only that task
    if args.task:
        print(f"Processing only task: {args.task}")
        subject, task, success = process_file(derivatives_folder, subject, args.task, output_dir, modes)
        if success:
            print(f"Successfully processed {subject}, task {args.task}")
        else:
            print(f"Failed to process {subject}, task {args.task}")
    else:
        # Process all tasks for this subject
        print("Finding tasks for subject...")
        subject_folder = os.path.join(derivatives_folder, subject)
        
        # Try both 'eeg' subfolder and direct subject folder
        eeg_folder = os.path.join(subject_folder, "eeg")
        if os.path.exists(eeg_folder):
            folder_to_check = eeg_folder
        else:
            folder_to_check = subject_folder
        
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
            
            tasks = list(tasks)
            if not tasks:
                print(f"No tasks found for subject {subject}")
                return
            
            print(f"Found {len(tasks)} tasks for {subject}: {tasks}")
            
            # Process each task
            successful_tasks = []
            failed_tasks = []
            
            for task in tasks:
                subject, task, success = process_file(derivatives_folder, subject, task, output_dir, modes)
                if success:
                    successful_tasks.append(task)
                else:
                    failed_tasks.append(task)
            
            # Print summary
            print(f"\nProcessing summary for {subject}:")
            print(f"Successfully processed {len(successful_tasks)} tasks")
            if successful_tasks:
                print("Successful tasks:")
                for task in successful_tasks:
                    print(f"  - {task}")
            if failed_tasks:
                print(f"Failed to process {len(failed_tasks)} tasks:")
                for task in failed_tasks:
                    print(f"  - {task}")
                    
        except Exception as e:
            print(f"Error finding tasks: {str(e)}")
            return

if __name__ == "__main__":
    main() 